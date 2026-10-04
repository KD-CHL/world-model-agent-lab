"""Real scheduler contracts with delayed, independently observed servo feedback."""
import unittest
import numpy as np

from test_task_runtime import Calibration, Predictor, Session, graph
from wmal.agents.task_graph import TaskGraph, TaskNode
from wmal.agents.task_runtime import RuntimeConfig, TaskRuntime


class DelayedServo(Session):
    """Commands advance every cycle; measured joints need stable real cycles."""

    def __init__(self, *, stable_samples=4, follow_after=None):
        super().__init__()
        self.stable_samples = stable_samples
        self.follow_after = follow_after
        self.stable_cycles = 0
        self.samples = []

    def execute_actions(self, actions, **kwargs):
        measured = self.q.copy()
        super().execute_actions(actions, **kwargs)
        self.stable_cycles = self.stable_cycles + 1 if np.all(actions == 0) else 0
        if (self.stable_cycles < self.stable_samples
                and (self.follow_after is None or self.step_id < self.follow_after)):
            self.q = measured
        self.samples.append(self.observe())
        return self.observe()


class BoundCalibration(Calibration):
    def __init__(self, profile):
        self.controller_profile = profile

    def validate_controller(self, profile):
        if profile != self.controller_profile:
            raise ValueError('Controller calibration mismatch')


class TaskConvergenceTests(unittest.TestCase):
    def run_task(self, session, task=None, **config):
        return TaskRuntime(task or graph(), session, Predictor(),
                           config=RuntimeConfig(baseline='A1', horizon=1, **config)).run()

    def test_command_progress_does_not_stall_unchanged_measured_joints(self):
        # Measured-only progress incorrectly stops at sample four, before six
        # .04-rad commands can reach this hand-chosen target.
        session = DelayedServo(follow_after=6)
        task = TaskGraph('command_progress', (TaskNode('reach', 'joint_reach', [.59, 1.11]),))
        result = self.run_task(session, task)
        self.assertEqual(result['state']['status'], 'succeeded')
        self.assertEqual(result['state']['recoveries'], 0)
        self.assertEqual(result['state']['executed_cycles'], 6)
        np.testing.assert_allclose(session.samples[3].state[:2], [.35, .87])
        np.testing.assert_allclose(session.samples[3].state[2:], [.51, 1.03])

    def test_command_in_tolerance_gets_four_real_stable_cycles_before_stall(self):
        # An old measured-only window stops at cycle four. The fourth stable
        # zero cycle is cycle six, and supplies the first measured success.
        session = DelayedServo()
        task = TaskGraph('servo_settle', (TaskNode('reach', 'joint_reach', [.43, .95]),))
        result = self.run_task(session, task)
        self.assertEqual(result['state']['status'], 'succeeded')
        self.assertEqual(result['state']['recoveries'], 0)
        self.assertEqual(result['state']['executed_cycles'], 6)
        self.assertEqual([s.step_id for s in session.samples], [1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(np.stack([s.state[2:] for s in session.samples[1:]]),
                                   np.tile([.43, .95], (5, 1)))
        self.assertEqual(result['state']['nodes']['reach']['evidence']['step_id'], 6)

    def test_five_node_dag_waits_for_servo_and_counts_only_executed_holds(self):
        session = DelayedServo()
        result = self.run_task(session)
        self.assertEqual(result['state']['status'], 'succeeded')
        self.assertEqual(result['state']['completed_subgoals'],
                         ['reach_A', 'hold_A', 'reach_B', 'hold_B', 'return_start'])
        self.assertEqual(result['state']['recoveries'], 0)
        self.assertEqual(result['state']['executed_cycles'], session.step_id)
        self.assertLessEqual(session.step_id, 80)
        for node in graph().nodes:
            entry = result['state']['nodes'][node.node_id]
            self.assertLessEqual(entry['actions'], 24)
            observed = np.asarray(entry['evidence']['observed_state'])
            self.assertLessEqual(np.linalg.norm(observed[:2] - node.target), .035)
            self.assertLessEqual(np.linalg.norm(observed[2:] - node.target), .015)
            if node.hold_steps:
                self.assertEqual(entry['hold_count'], node.hold_steps)

    def test_frozen_measured_pose_fails_after_bounded_settling_and_recovery(self):
        session = DelayedServo(stable_samples=1000)
        result = self.run_task(session, recovery=True)
        self.assertEqual(result['state']['status'], 'failed')
        self.assertEqual(result['state']['failure_reason'], 'stalled')
        self.assertEqual(result['state']['nodes']['reach_A']['recoveries'], 2)
        self.assertEqual(result['state']['recoveries_succeeded'], 0)
        self.assertEqual(result['state']['executed_cycles'], 14)
        self.assertEqual(result['state']['completed_subgoals'], [])

    def test_genuine_command_and_measured_no_progress_still_fails_finitely(self):
        class Frozen(Session):
            def execute_actions(self, actions, **kwargs):
                measured, command = self.q.copy(), self.target.copy()
                super().execute_actions(actions, **kwargs)
                self.q, self.target = measured, command
                return self.observe()

        session = Frozen()
        result = self.run_task(session, recovery=True)
        self.assertEqual(result['state']['status'], 'failed')
        self.assertEqual(result['state']['failure_reason'], 'stalled')
        self.assertEqual(result['state']['nodes']['reach_A']['recoveries'], 2)
        self.assertEqual(result['state']['executed_cycles'], 12)
        self.assertEqual(session.step_id, 12)

    def test_hold_failure_preserves_completed_node_and_bounded_repair(self):
        for recovery, status in ((False, 'failed'), (True, 'succeeded')):
            with self.subTest(recovery=recovery):
                result = self.run_task(Session(perturb=True), recovery=recovery)
                self.assertEqual(result['state']['status'], status)
                self.assertEqual(result['state']['completed_subgoals'].count('reach_A'), 1)
                if recovery:
                    self.assertEqual(result['state']['recoveries_succeeded'], 1)
                else:
                    self.assertEqual(result['state']['failure_reason'], 'hold_condition_lost')
                    self.assertEqual(result['state']['completed_subgoals'], ['reach_A'])


class ControllerBindingTests(unittest.TestCase):
    gravity = {'profile': 'g1_joint_pd_gravity_v2'}

    def test_nonlegacy_a3_requires_controller_bound_calibration(self):
        session = Session()
        session.controller_profile = self.gravity.copy()
        with self.assertRaises(ValueError):
            TaskRuntime(graph(), session, Predictor(), Calibration(), config=RuntimeConfig())
        self.assertEqual(session.step_id, 0)

    def test_a3_rejects_wrong_controller_calibration_before_execution(self):
        session = Session()
        with self.assertRaises(ValueError):
            TaskRuntime(graph(), session, Predictor(), BoundCalibration(self.gravity.copy()),
                        config=RuntimeConfig())
        self.assertEqual(session.step_id, 0)

    def test_controller_change_after_authorization_stops_before_dispatch(self):
        session = Session()
        session.controller_profile = self.gravity.copy()

        def emit(event, payload):
            if event == 'execution_started':
                session.controller_profile['profile'] = 'g1_joint_pd_v1'

        result = TaskRuntime(graph(), session, Predictor(),
                             BoundCalibration(self.gravity.copy()),
                             config=RuntimeConfig(), emit=emit).run()
        self.assertEqual(result['state']['status'], 'needs_review')
        self.assertEqual(result['state']['failure_reason'], 'model_or_semantics_changed')
        self.assertEqual(session.step_id, 0)

    def test_bound_controller_success_records_frozen_profile(self):
        session = Session()
        session.controller_profile = self.gravity.copy()
        result = TaskRuntime(graph(), session, Predictor(),
                             BoundCalibration(self.gravity.copy()), config=RuntimeConfig()).run()
        self.assertEqual(result['state']['status'], 'succeeded')
        self.assertEqual(result.get('controller_profile'), self.gravity)
        session.controller_profile['profile'] = 'changed_after_report'
        self.assertEqual(result['controller_profile'], self.gravity)


if __name__ == '__main__':
    unittest.main()
