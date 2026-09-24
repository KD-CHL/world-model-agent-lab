import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / 'configs/robots/g1_fixed_base.json'


@unittest.skipUnless(__import__('importlib').util.find_spec('mujoco'), 'Optional MuJoCo not installed')
class G1MujocoTests(unittest.TestCase):
    def load_config(self):
        config = json.loads(CONFIG_PATH.read_text())
        config['model_path'] = str((CONFIG_PATH.parent / config['model_path']).resolve())
        return config

    def test_config_loads_all_29_limited_joints_and_renders_pack_camera(self):
        from wmal.communication.contracts import RobotProfile
        from wmal.envs.mujoco_backend import MujocoBackend

        config = self.load_config()
        profile = RobotProfile(**config['profile'])
        backend = MujocoBackend(config['model_path'], profile, config['actuator_map'])
        self.assertEqual(len(profile.joint_limits), 29)
        self.assertEqual(len(backend.mapping), 29)
        self.assertNotIn('base_velocity', profile.capabilities)
        self.assertEqual(backend.model.nq, 29)
        for name, (joint_id, _, _) in backend.mapping.items():
            self.assertEqual(list(profile.joint_limits[name]), list(backend.model.jnt_range[joint_id]))
        image = backend.render_rgb(camera=config['camera']['name'],
                                   width=config['camera']['width'], height=config['camera']['height'])
        self.assertEqual(image.shape, (240, 320, 3))
        self.assertEqual(image.dtype.name, 'uint8')
        self.assertGreater(float(image.std()), 10.0)

    def test_simulation_cli_runs_and_rejects_unknown_joint(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'src')
        cli = ROOT / 'scripts/simulate.py'
        base = [sys.executable, str(cli), '--config', str(CONFIG_PATH), '--steps', '5']
        result = subprocess.run(base, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('29 joints', result.stdout)

        moved = subprocess.run(base + ['--steps', '1000', '--joint', 'left_shoulder_pitch_joint=-0.2'],
                               cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(moved.returncode, 0, moved.stderr)
        joint_line = next(line for line in moved.stdout.splitlines()
                          if line.startswith('left_shoulder_pitch_joint='))
        actual = float(joint_line.split('=')[1].split()[0])
        self.assertLess(actual, -0.15)
        self.assertGreaterEqual(actual, -0.2)

        rejected = subprocess.run(base + ['--joint', 'not_a_joint=0.1'], cwd=ROOT, env=env,
                                  capture_output=True, text=True, timeout=60)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('Unknown joint', rejected.stderr)

    def test_keyboard_joint_input_drives_physics_with_rate_and_limit_bounds(self):
        from wmal.communication.contracts import RobotProfile
        from wmal.envs.mujoco_backend import MujocoBackend
        from wmal.envs.keyboard_control import KeyboardJointControl

        config = self.load_config()
        profile = RobotProfile(**config['profile'])
        backend = MujocoBackend(config['model_path'], profile, config['actuator_map'])
        control = KeyboardJointControl(backend, initial_joint='left_shoulder_pitch_joint')
        joint = control.selected_joint

        control.on_key(ord('['))
        control.update()
        self.assertAlmostEqual(control.requested_targets[joint], -0.1)
        self.assertLessEqual(abs(backend.targets[joint]),
                             profile.max_joint_velocity_rad_s * backend.model.opt.timestep + 1e-9)
        for _ in range(1000):
            control.update()
            backend.step()
        self.assertLess(backend.observe().joints[joint], -0.04)

        for _ in range(100):
            control.on_key(ord(']'))
        control.update()
        self.assertLessEqual(control.requested_targets[joint], profile.joint_limits[joint][1])

    @unittest.skipUnless(__import__('importlib').util.find_spec('onnxruntime'),
                         'Optional ONNX Runtime not installed')
    def test_persistent_g1_session_steps_and_remains_open_until_explicit_close(self):
        from wmal.envs.g1_session import G1MuJoCoSession
        from wmal.locomotion.agent import G1Agent
        from wmal.locomotion.contracts import (G1_ACTION_SCHEMA, G1_STATE_SCHEMA,
                                               G1Goal, G1VelocityAction)
        from wmal.locomotion.planner import G1RolloutPlanner

        class SmokePredictor:
            version = 'test-only-kinematic-fixture'
            state_schema = G1_STATE_SCHEMA
            action_schema = G1_ACTION_SCHEMA
            calls = 0
            def predict(self, state, action, duration_s):
                from dataclasses import replace
                import math
                self.calls += 1
                return replace(state, step_id=state.step_id + 1,
                               sim_time_s=state.sim_time_s + duration_s,
                               x=state.x + (action.vx * math.cos(state.yaw)
                                            - action.vy * math.sin(state.yaw)) * duration_s,
                               y=state.y + (action.vx * math.sin(state.yaw)
                                            + action.vy * math.cos(state.yaw)) * duration_s,
                               yaw=state.yaw + action.yaw_rate * duration_s)

        with G1MuJoCoSession(viewer=False, realtime=False) as session:
            initial = session.observe()
            moved = session.step(G1VelocityAction(0.0, 0.0, 0.0, 0.1), 0.1)
            self.assertEqual(moved.step_id, initial.step_id + 1)
            self.assertTrue(session.is_running)
            next_state = session.step(G1VelocityAction(0.0, 0.0, 0.0, 0.1), 0.1)
            self.assertEqual(next_state.episode_id, initial.episode_id)
            self.assertEqual(next_state.step_id, moved.step_id + 1)
            self.assertTrue(session.is_running)
            self.assertGreater(next_state.pelvis_height, 0.6)
            predictor = SmokePredictor()
            planner = G1RolloutPlanner(predictor, samples=4, horizon=1,
                                       action_duration_s=0.1, seed=2)
            agent = G1Agent(planner)
            completed = agent.run_goal(G1Goal(next_state.x, next_state.y), session,
                                       max_cycles=1)
            self.assertEqual(completed.status, 'succeeded')
            self.assertTrue(session.is_running)
            result = agent.run_goal(G1Goal(10.0, 0.0), session, max_cycles=1)
            self.assertEqual(result.status, 'budget_exhausted')
            self.assertGreater(predictor.calls, 0)
            self.assertTrue(session.is_running)
        self.assertFalse(session.is_running)


if __name__ == '__main__':
    unittest.main()
