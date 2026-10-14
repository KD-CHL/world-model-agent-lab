"""Regression: a fixed servo must track targets without relaxing task predicates."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'MuJoCo unavailable')
class WorkcellControllerTests(unittest.TestCase):
    def compensated(self):
        from wmal.envs.visual_workcell import VisualWorkcellSession
        try:
            return VisualWorkcellSession(image_size=32, controller='gravity_compensated')
        except TypeError as exc:
            self.fail(f'Fixed gravity-compensated controller is not available: {exc}')

    def test_b_target_tracks_with_gravity_and_unchanged_command(self):
        with self.compensated() as session:
            before = session.reset([.27, .79])
            for _ in range(8):
                before = session.execute_actions([[0., 0.]], episode_id=before.episode_id,
                                                 step_id=before.step_id)
                np.testing.assert_allclose(before.state[2:], [.27, .79], atol=1e-7)
            self.assertLess(np.linalg.norm(before.state[:2] - [.27, .79]), .035)
            np.testing.assert_allclose(session.model.opt.gravity, [0., 0., -9.81])
            self.assertTrue(np.all(np.abs(session.data.actuator_force) <=
                                   session.model.actuator_forcerange[:, 1] + 1e-8))

    def test_feedforward_and_pd_share_the_actuator_torque_limit(self):
        with self.compensated() as session:
            session.model.actuator_forcerange[session.aids] = [-.1, .1]
            before = session.reset([.27, .79])
            session.execute_actions([[.06, -.06]], episode_id=before.episode_id,
                                    step_id=before.step_id)
            self.assertTrue(np.all(np.abs(session.data.actuator_force[session.aids]) <= .1+1e-9))
            np.testing.assert_array_equal(session.data.qfrc_applied, np.zeros(session.model.nv))

    def test_profile_binds_physical_parameters_and_reset_clears_stale_feedforward(self):
        with self.compensated() as session:
            profile = session.controller_profile
            session.model.actuator_gainprm[session.aids[0], 0] += 1.
            self.assertNotEqual(session.controller_profile, profile)
            session.model.actuator_biasprm[:, 0] = 100.
            obs = session.reset([.35, .87])
            self.assertLess(np.linalg.norm(obs.state[:2] - [.35, .87]), .035)
            self.assertTrue(np.all(np.abs(session.model.actuator_biasprm[:, 0]) < 100.))

    def test_legacy_default_preserves_known_tracking_bias(self):
        from wmal.envs.visual_workcell import VisualWorkcellSession
        with VisualWorkcellSession(image_size=32) as session:
            obs = session.reset([.27, .79])
            for _ in range(8):
                obs = session.execute_actions([[0., 0.]], episode_id=obs.episode_id,
                                              step_id=obs.step_id)
            self.assertGreater(np.linalg.norm(obs.state[:2] - [.27, .79]), .035)

    def test_actual_five_node_task_keeps_all_measured_hold_predicates(self):
        from wmal.agents.task_graph import TaskGraph
        from wmal.agents.task_runtime import RuntimeConfig,TaskRuntime
        from types import SimpleNamespace
        from pathlib import Path
        import json
        task=TaskGraph.from_dict(json.loads((Path(__file__).resolve().parents[1]/
                                            'configs/tasks/g1_joint_sequence.json').read_text())['task'])
        with self.compensated() as session:
            predictor=SimpleNamespace(version='nominal-no-model-query',semantics=session.semantics,
                metadata={'max_horizon':4},normalization={'state_scale':[1.,1.,1.,1.]},
                config=SimpleNamespace(context_steps=0))
            started=float(session.data.time)
            result=TaskRuntime(task,session,predictor,config=RuntimeConfig(baseline='A1',horizon=4)).run()
            self.assertEqual(result['state']['status'],'succeeded')
            self.assertEqual(result['state']['completed_subgoals'],
                             ['reach_A','hold_A','reach_B','hold_B','return_start'])
            self.assertAlmostEqual(session.data.time-started,result['state']['executed_cycles']*.2,places=7)
            for node in task.nodes:
                actual=result['state']['nodes'][node.node_id]
                observed=np.asarray(actual['evidence']['observed_state'])
                self.assertLessEqual(np.linalg.norm(observed[:2]-node.target),.035)
                self.assertLessEqual(np.linalg.norm(observed[2:]-node.target),.015)
                if node.hold_steps:
                    self.assertEqual(actual['hold_count'],3)
