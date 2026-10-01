"""Research scheduler: only receipts advance stages; rejected plans are bounded."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import runpy
import tempfile
import unittest

import numpy as np

from wmal.agents.predictive_skill_agent import (AgentTask, TaskStage, PredictiveSkillAgent,
                                               SkillCandidate, VisualObservation)


class IdentityDynamics:
    version = 'test-dynamics-v1'
    semantics = {'action_order': ['x'], 'state_order': ['x']}
    normalization = {'state_scale': [1.]}

    def predict(self, rgb, state, actions):
        states = state + np.cumsum(actions, axis=0)
        return {'model_version': self.version, 'states': np.stack([states, states]),
                'frames': np.broadcast_to(rgb, (2,len(actions),*rgb.shape)).copy(),
                'diagnostics': {'architecture': 'categorical_rssm',
                                'inference_mode': 'categorical_probability_proxy',
                                'members': [{'prior_entropy': [1.]*len(actions), 'posterior_entropy': .5}]}}


def observation(step, value):
    return VisualObservation('episode', step, np.zeros((3,32,32)), np.array([value]))


class AgentRSSMIntegrationTests(unittest.TestCase):
    def agent(self, *, baseline='A2'):
        task = AgentTask('mission', np.array([.2]), ('move',), tolerance=.02, max_cycles=5,
                         stages=(TaskStage('first', np.array([.1])), TaskStage('second', np.array([.2]))))
        return PredictiveSkillAgent(task, IdentityDynamics(), baseline=baseline,
                                    action_lower=[-.2], action_upper=[.2])

    def test_reobserve_budget_is_internal_and_cannot_loop_forever(self):
        agent = self.agent()
        for i in range(3):
            receipt = agent.plan(observation(0,0.), [])
        self.assertEqual(agent.state.status, 'needs_review')
        self.assertEqual(agent.state.reobservations, 3)
        self.assertEqual(receipt.evidence['reobservations_remaining'], 0)
        with self.assertRaises(ValueError):
            agent.plan(observation(0,0.), [])

    def test_feedback_only_multistage_and_diagnostics_audit(self):
        agent = self.agent()
        candidate = SkillCandidate('move', np.array([[.1]]))
        decision = agent.plan(observation(0,0.), [candidate])
        self.assertEqual(agent.active_stage.name, 'first')
        self.assertEqual(agent.state.completed_subgoals, [])
        self.assertEqual(decision.evidence['world_model_diagnostics']['architecture'], 'categorical_rssm')
        feedback = agent.record_feedback(decision, observation(1,.1))
        self.assertEqual(agent.active_stage.name, 'second')
        self.assertEqual(agent.state.completed_subgoals, ['first'])
        self.assertEqual(agent.state.belief['step_id'], 1)
        self.assertEqual(agent.state.belief['model_version'], IdentityDynamics.version)
        self.assertEqual(feedback['target_stage'], 'first')
        decision = agent.plan(observation(1,.1), [candidate])
        agent.record_feedback(decision, observation(2,.2))
        self.assertEqual(agent.state.status, 'succeeded')
        self.assertEqual(agent.state.completed_subgoals, ['first','second'])

    def test_successful_receipt_resets_consecutive_reobserve_budget(self):
        agent = self.agent()
        agent.plan(observation(0,0.), [])
        receipt = agent.plan(observation(0,0.), [SkillCandidate('move', np.array([[.1]]))])
        agent.record_feedback(receipt, observation(1,.1))
        self.assertEqual(agent.state.reobservations, 0)

    def test_array_snapshots_and_forged_execution_receipt(self):
        agent = self.agent()
        source = np.array([[.1]])
        candidate = SkillCandidate('move', source)
        source[:] = .9
        np.testing.assert_array_equal(candidate.actions, [[.1]])
        receipt = agent.plan(observation(0,0.), [candidate])
        with self.assertRaises(ValueError):
            receipt.actions[:] = .5
        forged = replace(receipt, prefix_length=2, actions=np.array([[.1],[.1]]))
        with self.assertRaises(ValueError):
            agent.record_feedback(forged, observation(2,.2))
        agent.record_feedback(receipt, observation(1,.1))
        self.assertEqual(agent.state.completed_subgoals, ['first'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch unavailable')
    def test_run_task_generates_actions_toward_active_waypoint(self):
        # Controlled environment below is a known real task-state transition,
        # not a mock of the Agent. It exposes final-goal-only proposal bugs.
        script = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/visual_skill_agent.py'))
        class Model:
            version = 'nominal-only'
            semantics = {'action_order':['shoulder','elbow'], 'state_order':['q1','q2','target1','target2']}
        class Session:
            semantics = Model.semantics
            target = np.array([.35,.85])
            step = 0
            data = type('Data', (), {'time':0.})()
            def observe(self):
                return VisualObservation('episode', self.step, np.zeros((3,32,32)), np.tile(self.target,2))
            def execute_actions(self, actions, **kwargs):
                self.target += np.asarray(actions).sum(0)
                self.step += len(actions)
                return self.observe()
        with tempfile.TemporaryDirectory() as tmp:
            with (Path(tmp)/'events.jsonl').open('w') as log:
                report = script['run_task'](Session(), Model(), None, [.35,.85], baseline='A1',
                         horizon=1, max_cycles=12, error_budget=1., seed=0, log=log,
                         artifact_dir=Path(tmp), waypoints=([.43,.93],))
            self.assertEqual(report['state']['status'],'succeeded')
            self.assertEqual(report['state']['completed_subgoals'],['waypoint_1','final_goal'])
