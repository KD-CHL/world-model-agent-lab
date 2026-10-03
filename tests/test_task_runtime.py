"""Deterministic physical-contract fixtures exercise real task scheduling."""
from dataclasses import replace
import unittest
import numpy as np
from wmal.agents.predictive_skill_agent import VisualObservation
from wmal.agents.task_graph import TaskNode, TaskGraph


class Session:
    semantics = {'action_order':['s','e'], 'state_order':['q1','q2','c1','c2'], 'period_s':.2}
    period_s = .2
    def __init__(self, perturb=False, invalid=False):
        self.q = np.array([.35,.87])
        self.target = self.q.copy()
        self.episode_id, self.step_id = 'episode', 0
        self.fault_reason = None
        self.perturb, self.invalid = perturb, invalid
        self.injected = False
        self.data = type('Data',(),{'time':0.})()
    def observe(self):
        return VisualObservation(self.episode_id,self.step_id,np.zeros((3,32,32)),
                                 np.concatenate([self.q,self.target]))
    def execute_actions(self, actions, *, episode_id, step_id):
        if self.fault_reason or episode_id!=self.episode_id or step_id!=self.step_id:
            raise RuntimeError('Invalid session authorization')
        for action in actions:
            self.target += action
            self.q = self.target.copy()
            self.step_id += 1
            self.data.time += .2
            if self.perturb and not self.injected and np.all(action==0):
                self.q[0] += .07
                self.injected = True
        if self.invalid:
            return VisualObservation(self.episode_id,self.step_id+1,np.zeros((3,32,32)),
                                     np.concatenate([self.q,self.target]))
        return self.observe()
    def abort(self, reason):
        self.fault_reason = reason


class Predictor:
    version = 'test-dynamics'
    semantics = Session.semantics
    metadata = {'max_horizon':4}
    normalization = {'state_scale':[1.,1.,1.,1.]}
    config = type('Config',(),{'context_steps':0})()
    def predict(self, rgb, state, actions):
        targets = state[2:] + np.cumsum(actions,axis=0)
        values = np.concatenate([targets,targets],axis=1)
        return {'model_version':self.version, 'states':np.stack([values,values]),
                'frames':np.broadcast_to(rgb,(2,len(actions),*rgb.shape)).copy()}


class Calibration:
    def validate_for(self, version, semantics):
        if version != 'test-dynamics':
            raise ValueError('version changed')
    def bounds(self, std, *, version):
        return np.full_like(std,.02)


def graph():
    return TaskGraph('sequence',(
        TaskNode('reach_A','joint_reach',[.43,.95]),
        TaskNode('hold_A','joint_hold',[.43,.95],('reach_A',),3),
        TaskNode('reach_B','joint_reach',[.27,.79],('hold_A',)),
        TaskNode('hold_B','joint_hold',[.27,.79],('reach_B',),3),
        TaskNode('return_start','joint_reach',[.35,.87],('hold_B',),3)))


class TaskRuntimeTests(unittest.TestCase):
    def run_task(self, session=None, **kwargs):
        from wmal.agents.task_runtime import TaskRuntime, RuntimeConfig
        cfg=RuntimeConfig(baseline='A1',horizon=1,**kwargs)
        return TaskRuntime(graph(),session or Session(),Predictor(),config=cfg).run()

    def test_real_five_node_scheduler_with_executed_hold_evidence(self):
        session=Session()
        report=self.run_task(session)
        self.assertEqual(report['state']['status'],'succeeded')
        self.assertEqual(report['state']['completed_subgoals'],
                         ['reach_A','hold_A','reach_B','hold_B','return_start'])
        self.assertEqual(report['state']['nodes']['hold_A']['hold_count'],3)
        self.assertEqual(report['state']['nodes']['return_start']['hold_count'],3)
        self.assertEqual(report['state']['executed_cycles'],session.step_id)
        self.assertGreaterEqual(session.data.time, 2.6)

    def test_hold_failure_r0_stops_r1_repairs_only_blocked_node(self):
        failed=self.run_task(Session(perturb=True),recovery=False)
        self.assertEqual(failed['state']['completed_subgoals'],['reach_A'])
        self.assertEqual(failed['state']['status'],'failed')
        fixed=self.run_task(Session(perturb=True),recovery=True)
        self.assertEqual(fixed['state']['status'],'succeeded')
        self.assertEqual(fixed['state']['recoveries_succeeded'],1)
        self.assertEqual(fixed['state']['completed_subgoals'].count('reach_A'),1)

    def test_budget_and_unknown_receipt_are_terminal_without_retry(self):
        report=self.run_task(max_actions=2)
        self.assertEqual(report['state']['status'],'budget_exhausted')
        self.assertEqual(report['state']['executed_cycles'],2)
        session=Session(invalid=True)
        failed=self.run_task(session,recovery=True)
        self.assertEqual(failed['state']['status'],'fault_latched')
        self.assertEqual(session.step_id,1)
        self.assertEqual(failed['state']['executed_cycles'],1)
        self.assertIsNotNone(session.fault_reason)

    def test_config_rejects_a0_recovery_boolean_budget_and_nan(self):
        from wmal.agents.task_runtime import RuntimeConfig
        for kwargs in ({'baseline':'A0','recovery':True},{'max_actions':True},
                       {'error_budget':float('nan')},{'horizon':5}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                RuntimeConfig(**kwargs)

    def test_one_time_authorization_and_header_tampering(self):
        import time
        from wmal.agents.task_execution import ExecutionManager
        session=Session()
        manager=ExecutionManager(session)
        token=manager.authorize(session.observe(),np.array([[.02,.02]]),{'task':'x'},time.monotonic()+2)
        token.actions.shape=(2,)
        with self.assertRaises(ValueError):
            manager.execute(token)
        self.assertEqual(session.step_id,0)

    def test_cooperative_stop_records_actual_prefix_and_cannot_replay(self):
        import time
        from wmal.agents.task_execution import ExecutionManager
        session=Session()
        manager=ExecutionManager(session)
        token=manager.authorize(session.observe(),np.full((3,2),.02),{},time.monotonic()+2)
        result=manager.execute(token,on_step=lambda obs,index:'check' if index==0 else None)
        self.assertEqual(result['status'],'cooperative_stop')
        self.assertEqual(result['executed'],1)
        self.assertEqual(len(result['trace']),1)
        with self.assertRaises(ValueError):
            manager.execute(token)
        self.assertEqual(session.step_id,1)

    def test_canceled_before_first_action(self):
        from wmal.agents.task_runtime import TaskRuntime, RuntimeConfig
        session=Session()
        report=TaskRuntime(graph(),session,Predictor(),config=RuntimeConfig(baseline='A1')).run(cancel=lambda:True)
        self.assertEqual(report['state']['status'],'canceled')
        self.assertEqual(session.step_id,0)

    def test_no_trusted_prefix_is_bounded_and_does_not_fake_new_evidence(self):
        from wmal.agents.task_runtime import TaskRuntime, RuntimeConfig
        session=Session()
        report=TaskRuntime(graph(),session,Predictor(),Calibration(),
                          config=RuntimeConfig(error_budget=.001,recovery=True)).run()
        self.assertEqual(report['state']['status'],'needs_review')
        self.assertEqual(report['state']['decision_count'],3)
        self.assertEqual(session.step_id,0)

    def test_k2_real_history_survives_task_node_transition(self):
        from wmal.agents.task_runtime import TaskRuntime, RuntimeConfig
        class Temporal(Predictor):
            config=type('Config',(),{'context_steps':2})()
            def predict_context(self,rgb,states,past_actions,future_actions):
                if len(states)>1:
                    np.testing.assert_allclose(states[-1,2:]-states[-2,2:],past_actions[-1])
                return super().predict(rgb[-1],states[-1],future_actions)
        report=TaskRuntime(graph(),Session(),Temporal(),config=RuntimeConfig(baseline='A2',horizon=1)).run()
        self.assertEqual(report['state']['status'],'succeeded')
        self.assertEqual(report['state']['belief']['context_steps_used'],2)

    def test_version_change_during_prediction_does_not_execute(self):
        from wmal.agents.task_runtime import TaskRuntime, RuntimeConfig
        class Changed(Predictor):
            def predict(self,*args):
                result=super().predict(*args)
                self.version='new-version'
                return result
        session=Session()
        report=TaskRuntime(graph(),session,Changed(),config=RuntimeConfig(baseline='A2')).run()
        self.assertEqual(report['state']['status'],'needs_review')
        self.assertEqual(session.step_id,0)

    def test_a0_frozen_program_no_model_queries(self):
        from wmal.agents.task_runtime import TaskRuntime, RuntimeConfig
        class Forbidden(Predictor):
            def predict(self,*args):
                raise AssertionError('Fixed plan queried model')
        report=TaskRuntime(graph(),Session(),Forbidden(),config=RuntimeConfig(baseline='A0',horizon=1)).run()
        self.assertEqual(report['state']['status'],'succeeded')

    def test_a3_prediction_mismatch_stops_before_second_authorized_action(self):
        from wmal.agents.task_runtime import TaskRuntime, RuntimeConfig
        class Biased(Predictor):
            def predict(self,*args):
                result=super().predict(*args)
                result['states'][:,:,0]+=.03
                return result
        session=Session()
        report=TaskRuntime(graph(),session,Biased(),Calibration(),config=RuntimeConfig()).run()
        self.assertEqual(report['state']['status'],'failed')
        self.assertEqual(session.step_id,1)
        self.assertEqual(report['execution_ledger'][0]['reason'],'prediction_mismatch')

    def test_stalled_execution_recovery_is_finite(self):
        from wmal.agents.task_runtime import TaskRuntime, RuntimeConfig
        class Stalled(Session):
            def execute_actions(self,*args,**kwargs):
                before=self.q.copy()
                super().execute_actions(*args,**kwargs)
                self.q=before
                return self.observe()
        report=TaskRuntime(graph(),Stalled(),Predictor(),
                          config=RuntimeConfig(baseline='A1',horizon=1,recovery=True)).run()
        self.assertEqual(report['state']['status'],'failed')
        self.assertEqual(report['state']['nodes']['reach_A']['recoveries'],2)
        self.assertLess(report['state']['executed_cycles'],80)

    def test_stale_authorization_is_consumed_and_does_not_execute(self):
        import time
        from wmal.agents.task_execution import ExecutionManager
        session=Session()
        manager=ExecutionManager(session)
        token=manager.authorize(session.observe(),np.full((2,2),.02),{},time.monotonic()+2)
        session.q[0]+=.001
        result=manager.execute(token)
        self.assertEqual(result['status'],'fault')
        self.assertEqual(session.step_id,0)
        with self.assertRaises(ValueError):
            manager.execute(token)

    def test_callback_exception_keeps_actual_progress_and_latches_session(self):
        import time
        from wmal.agents.task_execution import ExecutionManager
        session=Session()
        manager=ExecutionManager(session)
        token=manager.authorize(session.observe(),np.full((3,2),.02),{},time.monotonic()+2)
        def sink(*args):
            raise RuntimeError('feedback receiver failed')
        result=manager.execute(token,on_step=sink)
        self.assertEqual(result['status'],'fault')
        self.assertEqual(result['executed'],1)
        self.assertEqual(session.step_id,1)
        self.assertIsNotNone(session.fault_reason)
