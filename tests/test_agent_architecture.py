"""Behavioral checks for recovery, skill boundaries, and remote side effects."""
from dataclasses import asdict, replace
import unittest
import time
from uuid import uuid4

from wmal.agents.g1_coordinator import LanguageMissionPlanner
from wmal.communication.g1_session_protocol import SCHEMA, G1SessionOwner
from wmal.communication.g1_ros2 import G1RemoteSession
from wmal.locomotion.agent import G1TaskResult
from wmal.locomotion.contracts import G1State, G1Goal, G1VelocityAction
from wmal.locomotion.mission import MissionAgent
from wmal.models.calibration import fit_residual_calibration
from wmal.skills.registry import SkillSpec, SkillRegistry
from wmal.skills.executor import SkillExecutor
from wmal.envs.indoor_scene import IndoorScene


def state():
    return G1State(str(uuid4()), 0, 0., 0., 0., .8, 0., 0., 0., 0., 0., 0., 0., .8)


class Session:
    is_running = True
    def __init__(self):
        self.state, self.calls = state(), 0
    def observe(self): return self.state
    def step(self, action, duration_s):
        self.calls += 1
        self.state = replace(self.state, step_id=self.state.step_id+1,
                             sim_time_s=self.state.sim_time_s+duration_s)
        return self.state
    def close(self): self.is_running = False
    def set_goal_marker(self, goal): pass


class ArchitectureTests(unittest.TestCase):
    def test_dataset_calibration_reads_only_validation_and_rejects_leakage(self):
        import json
        import tempfile
        from pathlib import Path
        import numpy as np
        from scripts.calibrate_residuals import dataset_residuals
        from wmal.locomotion.learned import LearnedG1Dynamics
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'model.json'
            LearnedG1Dynamics(np.zeros((2,10,6)),.5,'fixture').save(checkpoint)
            rows = []
            for split in ('train','validation','validation','test'):
                before = state()
                after = replace(before,step_id=1,sim_time_s=.5,x=.1)
                rows.append({'split':split,'before':asdict(before),'after':asdict(after),
                             'action':asdict(G1VelocityAction(.1,0.,0.,.5))})
            dataset = root/'data.json'
            dataset.write_text(json.dumps({'schema':'wmal.g1.transitions.v1','rows':rows}))
            residuals, excluded, version, duration, digest = dataset_residuals(dataset,checkpoint)
            self.assertEqual(len(residuals),2)
            self.assertEqual(len(excluded),2)
            self.assertEqual(version,'fixture')
            rows.append({**rows[0], 'split':'validation'})
            dataset.write_text(json.dumps({'schema':'wmal.g1.transitions.v1','rows':rows}))
            with self.assertRaises(ValueError): dataset_residuals(dataset,checkpoint)

    def test_calibration_cli_roundtrip(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from scripts.calibrate_residuals import main
        from wmal.models.calibration import ResidualCalibration
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'episode_id':e, 'split':'validation', 'model_version':'m',
                     'duration_s':.5, 'position_error_m':v} for e,v in [('v1',.1),('v2',.2)]]
            (root/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (root/'excluded.json').write_text('["train1", "test1"]')
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--input',str(root/'rows.jsonl'), '--output',str(root/'calibration.json'),
                      '--excluded-episodes',str(root/'excluded.json'), '--model-version','m',
                      '--duration','.5','--dataset-id','fixture'])
            result = ResidualCalibration.load(root/'calibration.json')
            self.assertAlmostEqual(result.threshold_m, .2)
            self.assertEqual(result.episode_ids, ('v1','v2'))

    def test_summary_retains_incomplete_and_failed_tasks(self):
        from wmal.analysis.agent_summary import summarize_agent_events
        records = [
            {'event':'task_started','payload':{'task_id':'a','mode':'joint_goal'}},
            {'event':'task_started','payload':{'task_id':'b','mode':'g1_navigation'}},
            {'event':'task_result','payload':{'task_id':'b','status':'failed','cycles':None}},
            {'event':'replan_requested','payload':{'task_id':'b'}},
        ]
        report = summarize_agent_events(records)
        self.assertEqual(report['task_status_counts'], {'incomplete':1, 'failed':1})
        self.assertIsNone(report['position_residual_m']['rmse'])
        self.assertEqual(report['tasks'][1]['replans'], 1)

    def test_coordinator_reinvokes_language_after_typed_failure(self):
        from wmal.agents.g1_coordinator import G1Coordinator
        from wmal.locomotion.planner import G1Plan
        from wmal.planners.replan import NavigationStalled
        class Language:
            calls = 0
            def propose(self,*args): return [G1Goal(.3,0.)]
            def recover(self,instruction,goal,*args):
                self.calls += 1
                return [goal]
        class Planner:
            calls = 0
            def plan(self, observed, goal):
                self.calls += 1
                if self.calls == 1: raise NavigationStalled('fixture')
                action = G1VelocityAction(.4,0.,0.,.5)
                predicted = replace(observed, x=observed.x+.2, step_id=observed.step_id+1,
                                    sim_time_s=observed.sim_time_s+.5)
                return G1Plan('fixture', observed.step_id, action, predicted, predicted, .1)
        class MovingSession(Session):
            def step(self, action, duration_s):
                super().step(action, duration_s)
                self.state = replace(self.state, x=self.state.x+action.vx*duration_s)
                return self.state
        language, events = Language(), []
        coordinator = G1Coordinator(language, Planner(), IndoorScene(), 'fixture',
            max_replans=1, max_total_cycles=3, log=lambda e,p:events.append((e,p)))
        report = coordinator.run('go', MovingSession())
        self.assertEqual(report['status'], 'succeeded')
        self.assertEqual(language.calls, 1)
        self.assertEqual(report['cycles'], 1)
        self.assertEqual(sum(e=='task_result' for e,p in events), 1)

    def test_visual_adapter_checks_success_after_last_allowed_action(self):
        from types import SimpleNamespace
        from wmal.agents.skill_adapters import run_registered_task, visual_policy_skill
        from wmal.communication.contracts import RobotProfile, Observation, ExecutionResult
        profile = RobotProfile('arm', 'arm', {'j':(-1.,1.)})
        class Channel:
            step = 0
            def observe(self, **kw): return Observation('arm','ep',self.step,self.step*.5,{'j':self.step*.2})
            def observe_with_images(self, *args, **kw):
                return [SimpleNamespace(observation=self.observe(), width=1, height=1, rgb=bytes(3))]
            def execute(self,*a,**kw):
                self.step += 1
                return ExecutionResult('c', 'succeeded')
        class Client:
            version = 'policy'
            def propose(self,query): return SimpleNamespace(model_version='policy', actions=[[.2]])
        class Mapping:
            action_order = ['j']
            def command(self,*args): return SimpleNamespace(command_id='c', values={'j':.2})
        result = run_registered_task('move', Channel(), lambda emit: visual_policy_skill(
            Client(), profile, Mapping(), max_cycles=1, state_order=['j'], history_length=1,
            success_checker=lambda instruction, obs:obs.joints['j']==.2, log=emit),
            max_cycles=1, log=lambda *args:None)
        self.assertEqual(result.status, 'succeeded')
        self.assertEqual(result.cycles, 1)

    def test_joint_adapter_uses_shared_events_and_observed_success(self):
        from wmal.agents.skill_adapters import run_registered_task, joint_skill
        from wmal.communication.contracts import RobotProfile, Observation, Goal
        profile = RobotProfile('arm', 'arm', {'j':(-1.,1.)})
        class Client:
            def propose_goal(self,*a): return Goal('joint_goal', {'j':.2})
        class Channel:
            def observe(self,**kw): return Observation('arm','episode',0,0.,{'j':.2})
        events = []
        result = run_registered_task('move', Channel(),
            lambda emit: joint_skill(Client(), profile, max_cycles=2, log=emit),
            max_cycles=2, log=lambda e,p:events.append((e,p)))
        self.assertEqual(result.status, 'succeeded')
        self.assertEqual([e for e,p in events if e in ('task_started','skill_result','task_result')],
                         ['task_started','skill_result','task_result'])
        self.assertEqual(len({p['task_id'] for e,p in events}), 1)

    def test_skill_rejects_false_success_and_failed_precondition(self):
        registry, session = SkillRegistry(), Session()
        calls = []
        def execute(*args):
            calls.append(1)
            return G1TaskResult('succeeded', 0, 'unsupported claim', session.state)
        registry.register(SkillSpec('test', 'p', 's', lambda *a: True, execute, lambda *a: False))
        result = SkillExecutor(registry).run('test', {}, session, 3)
        self.assertEqual(result.status, 'success_unverified')
        registry.register(SkillSpec('blocked', 'p', 's', lambda *a: False, execute, lambda *a: True))
        self.assertEqual(SkillExecutor(registry).run('blocked', {}, session, 3).status, 'precondition_failed')
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError): registry.get('unknown')
        with self.assertRaises(ValueError): registry.register(registry.get('test'))

    def test_recovery_preserves_original_mission_and_budget(self):
        session = Session()
        seen, recoveries = [], []
        class Agent:
            def run_goal(self, goal, session, max_cycles):
                seen.append(goal.x)
                if len(seen) == 1:
                    return G1TaskResult('stalled', 1, 'stalled', session.state)
                session.state = replace(session.state, x=goal.x, y=goal.y)
                return G1TaskResult('succeeded', 1, 'observed', session.state)
        def recover(goal, *args):
            recoveries.append(goal)
            return [G1Goal(.5, 0.), goal]
        result = MissionAgent(Agent(), lambda *a: None, recover=recover,
                              max_replans=1, max_total_cycles=4).run([G1Goal(1.,0.), G1Goal(3.,0.)], session)
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(seen, [1., .5, 1., 3.])
        self.assertEqual(result['replans'], 1)
        self.assertEqual(result['cycles'], 4)

    def test_recovery_cannot_change_endpoint_or_retry_uncertain_failure(self):
        session = Session()
        class Agent:
            status = 'stalled'
            def run_goal(self, *args): return G1TaskResult(self.status, 0, self.status, session.state)
        agent = Agent()
        mission = MissionAgent(agent, lambda *a: None, recover=lambda *a: [G1Goal(9.,0.)], max_replans=1)
        with self.assertRaises(ValueError): mission.run([G1Goal(1.,0.)], session)
        agent.status = 'failed'
        self.assertEqual(mission.run([G1Goal(1.,0.)], session)['replans'], 0)

    def test_zero_progress_cannot_cause_unbounded_recovery(self):
        session = Session()
        class Agent:
            def run_goal(self, *args): return G1TaskResult('stalled', 0, 'stall', session.state)
        report = MissionAgent(Agent(), lambda *a: None, recover=lambda goal,*a:[goal],
                              max_replans=2, max_total_cycles=3).run([G1Goal(1.,0.)], session)
        self.assertEqual(report['status'], 'stalled')
        self.assertEqual(report['replans'], 2)
        self.assertEqual(len(report['goals']), 3)

    def test_total_budget_overrides_per_goal_budget(self):
        session = Session()
        budgets = []
        class Agent:
            def run_goal(self, goal, session, max_cycles):
                budgets.append(max_cycles)
                return G1TaskResult('budget_exhausted', max_cycles, 'budget', session.state)
        report = MissionAgent(Agent(), lambda *a:None, recover=lambda goal,*a:[goal],
                              max_replans=10, max_total_cycles=5).run([G1Goal(1.,0.)], session, max_cycles=3)
        self.assertEqual(budgets, [3,2])
        self.assertEqual(report['cycles'], 5)

    def test_language_recovery_context_and_endpoint_validation(self):
        class Client:
            model = 'fixture'
            reply = {'status':'planned','goals':[{'x':1.,'y':0.,'yaw':None}]}
            def request_json(self, body):
                self.body = body
                return self.reply
        client, goal = Client(), G1Goal(1.,0.)
        planner = LanguageMissionPlanner(client)
        result = planner.recover('go forward', goal, state(), IndoorScene(), 'model', 'stalled', 1, 0)
        self.assertEqual(result[-1], goal)
        self.assertIn('failure_reason', client.body['messages'][-1]['content'])
        client.reply = {'status':'planned','goals':[{'x':0.,'y':0.,'yaw':None}]}
        with self.assertRaises(ValueError):
            planner.recover('go', goal, state(), IndoorScene(), 'model', 'stalled', 1, 0)

    def test_calibration_is_episode_based_and_rejects_leakage(self):
        rows = [{'episode_id': e, 'position_error_m': v, 'split':'validation',
                 'model_version':'m', 'duration_s':.5}
                for e,v in [('a',.1),('a',.3),('b',.2),('c',.4)]]
        calibration = fit_residual_calibration(rows, model_version='m', duration_s=.5,
                                              dataset_id='validation-v1', quantile=.5)
        self.assertAlmostEqual(calibration.threshold_m, .3)
        with self.assertRaises(ValueError): calibration.validate_for('other', .5)
        with self.assertRaises(ValueError): calibration.validate_for('m', .25)
        with self.assertRaises(ValueError):
            fit_residual_calibration(rows, model_version='m', duration_s=.5, dataset_id='v', excluded_episode_ids=['a'])
        rows[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            fit_residual_calibration(rows, model_version='m', duration_s=.5, dataset_id='v')

    def test_observed_goal_success_takes_priority_over_residual_alarm(self):
        from wmal.locomotion.agent import G1Agent
        from wmal.locomotion.planner import G1Plan
        class Planner:
            def plan(self, observed, goal):
                predicted = replace(observed, step_id=1, sim_time_s=.5)
                return G1Plan('m',0,G1VelocityAction(.1,0.,0.,.5),predicted,predicted,1.)
        class Feedback:
            def update(self,*args): return {'replan_required':True}
        class ArrivingSession(Session):
            def step(self, action, duration_s):
                super().step(action,duration_s)
                self.state = replace(self.state,x=1.)
        events = []
        result = G1Agent(Planner(),feedback=Feedback(),log=lambda e,p:events.append(e)).run_goal(
            G1Goal(1.,0.), ArrivingSession(), max_cycles=1)
        self.assertEqual(result.status,'succeeded')
        self.assertIn('prediction_residual', events)

    def test_calibrated_alarms_are_consecutive_and_explicit(self):
        from wmal.locomotion.feedback import ResidualFeedback
        from wmal.models.calibration import ResidualCalibration
        calibration = ResidualCalibration('m', .5, .1, .95, 'validation', ('v1','v2'))
        feedback = ResidualFeedback(calibration=calibration, consecutive_alarms=2)
        class Planner: feedback_scale=1.
        before = state()
        after = replace(before, x=.2)
        self.assertFalse(feedback.update(before, after, Planner)['replan_required'])
        self.assertTrue(feedback.update(before, after, Planner)['replan_required'])
        self.assertFalse(feedback.update(before, after, Planner)['replan_required'])


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.owner = G1SessionOwner(Session, 'scene')
        self.addCleanup(self.owner.close)
        self.client = G1RemoteSession(self.owner.handle, 'scene')
        self.action = G1VelocityAction(.1,0.,0.,.5)

    def request(self, op='step'):
        observed = self.owner.session.observe()
        request = {'schema': SCHEMA, 'request_id':str(uuid4()), 'scene_id':'scene', 'operation':op,
                   'episode_id': observed.episode_id, 'step_id':observed.step_id, 'sim_time_s':observed.sim_time_s,
                   'expires_at_unix_s':time.time()+10}
        if op == 'step': request['action'] = asdict(self.action)
        return request

    def test_duplicate_does_not_execute_twice_and_reset_discards_old(self):
        request = self.request()
        first = self.owner.handle(request)
        self.assertEqual(first, self.owner.handle(request))
        self.assertEqual(self.owner.session.calls, 1)
        mutated = dict(request, action=asdict(G1VelocityAction(.2,0.,0.,.5)))
        with self.assertRaises(ValueError): self.owner.handle(mutated)
        self.owner.handle(self.request('reset'))
        with self.assertRaises(ValueError): self.owner.handle(request)
        with self.assertRaises(ValueError): self.owner.handle(dict(request,request_id=str(uuid4())))
        self.assertEqual(self.owner.session.calls, 0)

    def test_old_reset_receipt_is_rejected_after_a_later_reset(self):
        first = self.request('reset')
        self.owner.handle(first)
        self.owner.handle(self.request('reset'))
        with self.assertRaises(ValueError): self.owner.handle(first)

    def test_timeout_latches_client_and_no_retry(self):
        self.client.observe()
        calls = []
        def uncertain(request):
            calls.append(request)
            raise TimeoutError('lost receipt')
        self.client.rpc = uncertain
        with self.assertRaises(TimeoutError): self.client.step(self.action)
        with self.assertRaises(RuntimeError): self.client.step(self.action)
        self.assertEqual(len(calls), 1)

    def test_scene_mismatch_and_stale_step(self):
        with self.assertRaises(ValueError): G1RemoteSession(self.owner.handle, 'other').observe()
        self.client.observe()
        self.owner.handle(self.request())
        with self.assertRaises(ValueError): self.client.step(self.action)
        self.assertEqual(self.owner.session.calls, 1)

    def test_expired_command_never_executes(self):
        request = self.request()
        request['expires_at_unix_s'] = time.time()-1
        with self.assertRaises(ValueError): self.owner.handle(request)
        self.assertEqual(self.owner.session.calls, 0)

    def test_observe_cannot_clear_local_timeout_latch(self):
        self.client.observe()
        self.client.faulted = True
        self.client.observe()
        self.assertTrue(self.client.faulted)
        self.client.reset()
        self.assertFalse(self.client.faulted)

    def test_roundtrip_and_reset(self):
        before = self.client.observe()
        after = self.client.step(self.action)
        self.assertEqual(after.step_id, before.step_id+1)
        self.assertEqual(after.sim_time_s-before.sim_time_s, .5)
        reset = self.client.reset()
        self.assertNotEqual(reset.episode_id, before.episode_id)

    def test_partial_execution_fault_is_cached(self):
        request = self.request()
        calls = []
        def broken(*a):
            calls.append(1)
            raise RuntimeError('partial physics step')
        self.owner.session.step = broken
        self.assertEqual(self.owner.handle(request)['status'], 'uncertain')
        self.assertEqual(self.owner.handle(request)['status'], 'uncertain')
        self.assertEqual(len(calls), 1)
        with self.assertRaises(RuntimeError): self.owner.handle(self.request())


if __name__ == '__main__': unittest.main()
