"""Learning, replay provenance, resume and planner compatibility checks."""
from dataclasses import asdict, replace
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wmal.datasets.motion_sequences import load_motion_episodes, contiguous_sequences
from wmal.locomotion.contracts import G1State, G1VelocityAction, G1Goal

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


def fixture(path):
    rng, rows = np.random.default_rng(4), []
    for episode in range(8):
        split = 'train' if episode < 4 else ('validation' if episode < 6 else 'test')
        state = G1State(f'fixture-{episode}',0,0.,0.,0.,.8,0.,0.,0.,0.,0.,0.,0.,.8)
        for step in range(12):
            action = G1VelocityAction(float(rng.uniform(-.2,.3)),float(rng.uniform(-.1,.1)),
                                     float(rng.uniform(-.2,.2)),.5)
            c,s = math.cos(state.yaw),math.sin(state.yaw)
            bvx,bvy = .8*action.vx,.8*action.vy
            vx,vy = c*bvx-s*bvy,s*bvx+c*bvy
            after = replace(state,step_id=step+1,sim_time_s=state.sim_time_s+.5,
                x=state.x+vx*.5,y=state.y+vy*.5,yaw=state.yaw+action.yaw_rate*.5,
                vx=vx,vy=vy,yaw_rate=action.yaw_rate)
            rows.append({'split':split,'before':asdict(state),'action':asdict(action),'after':asdict(after)})
            state = after
    payload = {'schema':'wmal.g1.transitions.v1','rows':rows,'data_source':'synthetic_test_fixture'}
    path.write_text(json.dumps(payload))
    return payload


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'data.json'
        self.payload = fixture(self.path)

    def test_leakage_duplicates_and_discontinuous_states_are_rejected(self):
        for change in ('split','duplicate','state'):
            payload = json.loads(json.dumps(self.payload))
            if change == 'split': payload['rows'][1]['split'] = 'test'
            if change == 'duplicate': payload['rows'].append(payload['rows'][0])
            if change == 'state': payload['rows'][1]['before']['x'] += .1
            self.path.write_text(json.dumps(payload))
            with self.subTest(change=change), self.assertRaises(ValueError): load_motion_episodes(self.path)

    def test_sequences_do_not_cross_missing_transitions(self):
        self.payload['rows'].pop(5)
        self.path.write_text(json.dumps(self.payload))
        splits, duration = load_motion_episodes(self.path)
        windows = contiguous_sequences(splits['train'],3)
        self.assertEqual(duration,.5)
        self.assertEqual(len(windows['fixture-0']),7)
        self.assertTrue(all(all(a[2]==b[0] for a,b in zip(w,w[1:])) for w in windows['fixture-0']))


@unittest.skipUnless(TORCH_AVAILABLE,'Install learning extra for neural training tests')
class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset = self.root/'data.json'
        fixture(self.dataset)

    def config(self, epochs=5):
        from wmal.training.motion_trainer import TrainingConfig
        return TrainingConfig(hidden=(24,24),members=2,epochs=epochs,batch_size=32,horizon=3,
                              learning_rate=.003,rollout_weight=.2,seed=12)

    def test_network_learns_roundtrips_and_plans(self):
        from wmal.training.motion_trainer import train_motion,evaluate_motion
        from wmal.models.motion_network import load
        from wmal.locomotion.planner import G1RolloutPlanner
        path = self.root/'best.pt'
        report = train_motion(self.dataset,path,self.config(35))
        self.assertLess(report['best_validation_loss'], report['history'][0]['validation_loss']*.75)
        self.assertFalse(report['test_used_for_selection'])
        splits,_ = load_motion_episodes(self.dataset)
        state,action,_ = next(iter(splits['test'].values()))[0]
        model = load({'checkpoint':path})
        original = model.predict(state,action,.5)
        copy = self.root/'copy.pt'
        model.save(copy)
        restored = load({'checkpoint':copy})
        self.assertEqual(restored.version,model.version)
        self.assertEqual(restored.predict(state,action,.5),original)
        plan = G1RolloutPlanner(restored,samples=4,horizon=2).plan(state,G1Goal(.3,0.))
        self.assertEqual(plan.model_version,model.version)
        self.assertEqual(plan.uncertainty_kind,'ensemble_spread')
        self.assertEqual(plan.predicted_state.step_id,state.step_id+1)
        metrics = evaluate_motion(self.dataset,path,horizon=3)
        self.assertEqual(len(metrics['position_rmse_m_by_step']),3)
        self.assertEqual(len(metrics['episode_position_rmse_m_by_step']),2)
        with self.assertRaises(ValueError): model.predict(state,action,.2)

    def test_resume_matches_continuous_cpu_training(self):
        from wmal.training.motion_trainer import train_motion
        uninterrupted = train_motion(self.dataset,self.root/'full.pt',self.config(6))
        first = train_motion(self.dataset,self.root/'resumed.pt',self.config(3))
        resumed = train_motion(self.dataset,self.root/'resumed.pt',self.config(6),resume=first['latest_checkpoint'])
        self.assertEqual(uninterrupted['model_version'],resumed['model_version'])
        self.assertEqual([r['validation_loss'] for r in uninterrupted['history']],
                         [r['validation_loss'] for r in resumed['history']])
        with self.assertRaises(ValueError):
            train_motion(self.dataset,self.root/'resumed.pt',replace(self.config(7),horizon=2),
                         resume=resumed['latest_checkpoint'])

    def test_test_targets_do_not_affect_training_or_normalization(self):
        from wmal.training.motion_trainer import train_motion
        first = train_motion(self.dataset,self.root/'first.pt',self.config(2))
        payload = json.loads(self.dataset.read_text())
        for row in payload['rows']:
            if row['split'] == 'test':
                row['before']['x'] += 100.
                row['after']['x'] += 100.
        self.dataset.write_text(json.dumps(payload))
        second = train_motion(self.dataset,self.root/'second.pt',self.config(2))
        self.assertEqual(first['model_version'],second['model_version'])

    def test_rollout_loss_backpropagates_through_sequence(self):
        import torch
        from wmal.training.motion_trainer import tensors,normalization,sequence_loss
        from wmal.models.motion_network import build_network
        splits,_ = load_motion_episodes(self.dataset)
        groups = contiguous_sequences(splits['train'],3)
        arrays = tensors(next(iter(groups.values()))[:2],'cpu')
        before = arrays[0].detach().requires_grad_()
        stats = {key:torch.tensor(value) for key,value in normalization(splits['train']).items()}
        model = build_network((16,16))
        _,_,rollout = sequence_loss(model,before,arrays[1],arrays[2],stats,1.)
        rollout.backward()
        self.assertGreater(float(before.grad[:,0].abs().sum()),0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()),0.)


if __name__ == '__main__': unittest.main()
