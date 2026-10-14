"""Real bounded inference, real MuJoCo feedback; no task-gain claims."""
import importlib
import contextlib
import io
import json
import multiprocessing as mp
import runpy
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_embodied_provider import make_model, profile
from wmal.models.embodied_world import EmbodiedWorldModel


def stuck_worker(connection, checkpoint, device):
    connection.send({'ready': EmbodiedWorldModel.load(checkpoint).version})
    connection.recv()
    time.sleep(60)


def delayed_worker(connection, checkpoint, device):
    from wmal.agents.embodied_provider import predict_request
    model=EmbodiedWorldModel.load(checkpoint)
    connection.send({'ready':model.version})
    while True:
        request=connection.recv()
        time.sleep(.4)
        connection.send({'result':predict_request(model,request)})


class EmbodiedAgentTests(unittest.TestCase):
    def test_original_prediction_deadline_survives_plan_emit(self):
        self._dispatch_fence('wall')

    def test_simulation_time_revalidated_after_plan_emit(self):
        self._dispatch_fence('simulation')

    def _dispatch_fence(self, mode):
        from test_task_runtime import Session, Predictor
        from wmal.agents.task_graph import TaskGraph, TaskNode
        from wmal.agents.task_runtime import TaskRuntime, RuntimeConfig
        clock = [0.]; session = Session()
        class BoundPredictor(Predictor):
            family = 'wmal.embodied_world.v1'
            def prepare_decision(self, context):
                self.context = context
            def predict_candidate(self, rgb, state, past, actions, **kwargs):
                clock[0] = .06
                result = self.predict(rgb[-1], state[-1], actions)
                result['diagnostics'] = {'prediction_binding': {'request_id':'bound-request',
                    'decision_id':self.context.decision_id, 'deadline':self.context.deadline}}
                return result
        def emit(event, payload):
            if event == 'plan' and payload['status'] == 'execute':
                if mode == 'wall':
                    clock[0] = .11
                else:
                    session.data.time += .02  # pixels/positions unchanged, no valid new endpoint
        task = TaskGraph('hold', (TaskNode('hold', 'joint_hold', [.35,.87], hold_steps=1),))
        with patch('time.monotonic', side_effect=lambda:clock[0]):
            result = TaskRuntime(task, session, BoundPredictor(),
                config=RuntimeConfig(baseline='A2', horizon=1, planning_timeout_s=.1), emit=emit).run()
        self.assertEqual(result['state']['executed_cycles'], 0)
        self.assertEqual(session.step_id, 0)
        self.assertNotEqual(result['state']['status'], 'succeeded')

    def test_local_session_capacity_never_evicts_replay_ids(self):
        api = self.api()
        from test_embodied_service import request_for
        provider = api.BoundedInference(self.checkpoint)
        self.addCleanup(provider.close)
        for index in range(2048):
            if hasattr(provider._seen, 'add'):
                provider._seen.add(f'accepted-{index}')
            else:
                provider._seen.append(f'accepted-{index}')
        with self.assertRaisesRegex(RuntimeError, 'session limit'):
            provider.infer(request_for(self.model))

    def test_rejected_candidates_keep_prediction_and_bound_evidence(self):
        from test_task_runtime import Session, Predictor, Calibration
        from wmal.agents.task_graph import TaskGraph, TaskNode
        from wmal.agents.task_runtime import TaskRuntime, RuntimeConfig
        class BoundPredictor(Predictor):
            family = 'wmal.embodied_world.v1'
            def prepare_decision(self, context): self.context = context
            def predict_candidate(self, rgb, state, past, actions, **kwargs):
                result = self.predict(rgb[-1],state[-1],actions)
                result['diagnostics'] = {'prediction_binding': {'request_id':'candidate-request',
                    'decision_id':self.context.decision_id, 'protocol_hash':'fixture-protocol'},
                    'prediction_scope':{'max_action_delta':.06}}
                return result
        events = []
        task = TaskGraph('reach', (TaskNode('reach','joint_reach',[.43,.95]),))
        result = TaskRuntime(task, Session(), BoundPredictor(), Calibration(),
            config=RuntimeConfig(baseline='A3', horizon=1, error_budget=.001),
            emit=lambda event,payload:events.append((event,payload))).run()
        self.assertEqual(result['state']['executed_cycles'], 0)
        plan = next(payload for event,payload in events if event=='plan' and payload['status']=='reobserve')
        candidate = plan['evidence']['candidates'][0]
        self.assertIn('predicted_states', candidate)
        np.testing.assert_allclose(candidate['predicted_states'], [[.39,.91,.39,.91]], atol=1e-8)
        np.testing.assert_array_equal(candidate['error_bounds'], [[.02]*4])
        self.assertEqual(candidate['model_version'], 'test-dynamics')
        self.assertEqual(candidate['world_model_diagnostics']['prediction_binding']['protocol_hash'], 'fixture-protocol')
        self.assertEqual(candidate['world_model_diagnostics']['prediction_scope'], {'max_action_delta':.06})

    def test_prediction_deadline_stops_remaining_prefix(self):
        from test_task_runtime import Session
        from wmal.agents.task_execution import ExecutionManager
        clock = [0.]; session = Session(); manager = ExecutionManager(session)
        with patch('time.monotonic', side_effect=lambda:clock[0]):
            grant = manager.authorize(session.observe(), np.zeros((2,2)), {}, .1)
            def on_step(observed, offset):
                clock[0] = .11
            result = manager.execute(grant, on_step=on_step)
        self.assertEqual(result['executed'], 1)
        self.assertEqual(session.step_id, 1)
        self.assertEqual(result['reason'], 'execution_deadline')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = self.root / 'model.pt'
        self.model = make_model(importlib.import_module('wmal.models.embodied_world'))
        self.model.save(self.checkpoint)

    def api(self):
        try:
            return importlib.import_module('wmal.agents.embodied_provider')
        except ModuleNotFoundError:
            self.fail('Task5 bounded Agent provider is missing')

    def test_stuck_worker_terminated_and_locked(self):
        api = self.api()
        from test_embodied_service import request_for
        with patch.object(api, '_prediction_worker', stuck_worker):
            worker = api.BoundedInference(self.checkpoint)
            self.addCleanup(worker.close)
            process = worker.process
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                worker.infer(request_for(self.model, timeout=.1))
            self.assertLess(time.monotonic()-started, 1.5)
            self.assertFalse(process.is_alive())
            with self.assertRaises(RuntimeError):
                worker.infer(request_for(self.model))

    def test_late_after_reset_never_dispatches(self):
        api = self.api()
        predictor, _ = api.load_embodied_agent_provider(self.checkpoint, None)
        self.addCleanup(predictor.close)
        context = api.DecisionContext(run_id='real-run', task_id='reach', node_id='node',
            episode_id='ep0', context_step_ids=(0,), observation_times=(0.,),
            controller=self.model.profile.controller, deadline=time.monotonic()+5,
            rgb=np.full((1,3,16,16), .3), state=np.array([[12.,-16.,110.,220.]]),
            past_actions=np.empty((0,2)))
        predictor.prepare_decision(context)
        predictor.reset()
        with self.assertRaises(RuntimeError):
            predictor.predict_candidate(context.rgb, context.state, context.past_actions,
                np.zeros((1,2)), candidate_id='candidate0', skill='joint_hold')

    def test_decision_controller_is_a_detached_immutable_snapshot(self):
        api=self.api()
        context=api.DecisionContext(run_id='run',task_id='task',node_id='node',episode_id='ep',
            context_step_ids=(0,),observation_times=(0.,),controller={'gains':[4,5]},
            deadline=time.monotonic()+5,rgb=np.zeros((1,3,16,16)),state=np.zeros((1,4)),
            past_actions=np.empty((0,2)))
        context.controller['gains'][0]=99
        self.assertEqual(context.controller['gains'],[4,5])

    def test_inflight_reset_discards_real_response_and_reaps_old_worker(self):
        api=self.api()
        from test_embodied_service import request_for
        with patch.object(api,'_prediction_worker',delayed_worker):
            worker=api.BoundedInference(self.checkpoint)
            self.addCleanup(worker.close)
            old=worker.process
            results=[]
            def call():
                try:
                    results.append(worker.infer(request_for(self.model)))
                except Exception as exc:
                    results.append(exc)
            thread=threading.Thread(target=call);thread.start()
            time.sleep(.1);worker.reset();thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertFalse(old.is_alive())
            self.assertIsInstance(results[0],RuntimeError)
            self.assertFalse(worker.faulted)
            self.assertEqual(worker.infer(request_for(self.model))['physical_states'].shape,(1,8,2,2))

    def test_time_only_drift_invalidates_before_dispatch(self):
        from test_task_runtime import Session, Predictor, graph
        from wmal.agents.task_runtime import TaskRuntime, RuntimeConfig
        session=Session()
        class ClockShift(Predictor):
            family='wmal.embodied_world.v1'
            def prepare_decision(self, context):
                self.context=context
            def predict_candidate(self,rgb,state,past,actions,**kwargs):
                session.data.time+=.02  # identical sensor content, advanced simulation clock
                return self.predict(rgb[-1],state[-1],actions)
        result=TaskRuntime(graph(),session,ClockShift(),config=RuntimeConfig(baseline='A2',horizon=1)).run()
        self.assertEqual(result['state']['executed_cycles'],0)
        self.assertEqual(result['state']['status'],'needs_review')

    def test_cli_rejection_closes_owned_prediction_worker(self):
        script=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/g1_task_agent.py'))
        before={child.pid for child in mp.active_children()}
        try:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                script['main'](['--model-family','embodied','--checkpoint',str(self.checkpoint),
                    '--config',str(Path(__file__).resolve().parents[1]/'configs/tasks/g1_joint_sequence.json'),
                    '--baseline','A2','--output',str(self.root/'rejected')])
            self.assertFalse([child for child in mp.active_children() if child.pid not in before])
        finally:
            for child in mp.active_children():
                if child.pid not in before:
                    child.terminate();child.join(1)

    def test_repeated_task_and_quit_cleanup(self):
        api = self.api()
        predictor, _ = api.load_embodied_agent_provider(self.checkpoint, None)
        old = predictor.transport.process
        predictor.reset()
        self.assertFalse(old.is_alive())
        self.assertTrue(predictor.transport.process.is_alive())
        new = predictor.transport.process
        predictor.close()
        predictor.close()
        self.assertFalse(new.is_alive())
        with self.assertRaises(RuntimeError):
            predictor.reset()

    def test_true_mujoco_adapter_trace(self):
        api = self.api()
        from wmal.envs.visual_workcell import VisualWorkcellSession
        from wmal.agents.task_graph import TaskGraph
        from wmal.agents.task_runtime import RuntimeConfig, TaskRuntime
        with VisualWorkcellSession('stack_block', image_size=32, controller='gravity_compensated') as session:
            session.reset([.35,.87])
            p = profile(controller=session.controller_profile, event_names=[], image_size=32,
                camera_ids=[session.camera])
            model = make_model(importlib.import_module('wmal.models.embodied_world'),
                observation_profile=p, metadata={'max_horizon':3,'training_source':{'task':'stack_block'}})
            # Neutral physical residual test fixture, not a trained-success result.
            for parameter in model.members[0].physical_head.parameters():
                parameter.data.zero_()
            model = EmbodiedWorldModel(model.config,p,model.normalization,model.members,
                metadata=model.metadata)
            model.save(self.checkpoint)
            predictor,_ = api.load_embodied_agent_provider(self.checkpoint,None)
            self.addCleanup(predictor.close)
            graph = TaskGraph.from_dict({'schema':'wmal.task_graph.v1','task_id':'hold',
                'start':[.35,.87],'nodes':[{'node_id':'hold','skill':'joint_hold',
                'target':[.35,.87],'hold_steps':1}]})
            events=[]
            runtime=TaskRuntime(graph,session,predictor,config=RuntimeConfig(baseline='A2',horizon=1),
                emit=lambda event,payload:events.append((event,payload)))
            report=runtime.run()
            self.assertEqual(report['state']['status'],'succeeded')
            self.assertEqual(report['state']['executed_cycles'],1)
            plan=next(payload for event,payload in events if event=='plan' and payload['status']=='execute')
            binding=plan['evidence']['world_model_diagnostics']['prediction_binding']
            self.assertEqual(binding['task_id'],'hold')
            self.assertEqual(binding['node_id'],'hold')
            self.assertEqual(binding['context_step_ids'],[0])
            self.assertEqual(binding['member_count'],1)
            self.assertEqual(plan['prediction_request_id'],binding['request_id'])
            self.assertNotEqual(plan['decision_id'],binding['decision_id'])

    def test_timeout_latches_real_robot_without_an_action(self):
        api=self.api()
        from wmal.envs.visual_workcell import VisualWorkcellSession
        from wmal.agents.task_graph import TaskGraph
        from wmal.agents.task_runtime import RuntimeConfig, TaskRuntime
        with VisualWorkcellSession('stack_block',image_size=32,controller='gravity_compensated') as session:
            p=profile(controller=session.controller_profile,event_names=[],image_size=32,camera_ids=[session.camera])
            model=make_model(importlib.import_module('wmal.models.embodied_world'),observation_profile=p,
                metadata={'max_horizon':3,'training_source':{'task':'stack_block'}})
            model.save(self.checkpoint)
            with patch.object(api,'_prediction_worker',stuck_worker):
                predictor,_=api.load_embodied_agent_provider(self.checkpoint,None)
                self.addCleanup(predictor.close)
                process=predictor.transport.process
                graph=TaskGraph.from_dict({'schema':'wmal.task_graph.v1','task_id':'hold',
                    'start':[.35,.87],'nodes':[{'node_id':'hold','skill':'joint_hold',
                    'target':[.35,.87],'hold_steps':1}]})
                result=TaskRuntime(graph,session,predictor,
                    config=RuntimeConfig(baseline='A2',horizon=1,planning_timeout_s=.1)).run()
                self.assertEqual(result['state']['status'],'fault_latched')
                self.assertEqual(result['state']['executed_cycles'],0)
                self.assertEqual(session.step_id,0)
                self.assertFalse(process.is_alive())
                self.assertEqual(session.fault_reason,'prediction_transport_failed')

    def test_embodied_cli_manifest_binds_family_and_implementation(self):
        api=self.api()
        from wmal.envs.visual_workcell import VisualWorkcellSession
        with VisualWorkcellSession('stack_block',image_size=32,controller='gravity_compensated') as session:
            p=profile(controller=session.controller_profile,event_names=[],image_size=32,camera_ids=[session.camera])
        model=make_model(importlib.import_module('wmal.models.embodied_world'),observation_profile=p,
            metadata={'max_horizon':3,'training_source':{'task':'stack_block'}})
        for parameter in model.members[0].physical_head.parameters():
            parameter.data.zero_()
        EmbodiedWorldModel(model.config,p,model.normalization,model.members,metadata=model.metadata).save(self.checkpoint)
        config=self.root/'config.json'
        config.write_text(json.dumps({'scene':'stack_block','task':{'schema':'wmal.task_graph.v1',
            'task_id':'hold','start':[.35,.87],'nodes':[{'node_id':'hold','skill':'joint_hold',
            'target':[.35,.87],'hold_steps':1}]},'runtime':{'baseline':'A2','horizon':1}}))
        script=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/g1_task_agent.py'))
        with contextlib.redirect_stdout(io.StringIO()):
            code=script['main'](['--model-family','embodied','--checkpoint',str(self.checkpoint),
                '--config',str(config),'--controller','gravity_compensated','--output',str(self.root/'cli')])
        self.assertEqual(code,0)
        manifest=json.loads((self.root/'cli/run_manifest.json').read_text())
        self.assertIn('model_family',manifest['parameters'])
        self.assertEqual(manifest['parameters']['model_family'],'embodied')
        self.assertIn('src/wmal/agents/embodied_provider.py',manifest['implementation_hashes'])
        report=json.loads((self.root/'cli/task_000.json').read_text())
        self.assertEqual(report['state']['executed_cycles'],1)
