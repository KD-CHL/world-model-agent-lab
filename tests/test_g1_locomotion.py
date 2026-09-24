import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'Optional MuJoCo not installed')
class G1LocomotionTests(unittest.TestCase):
    def test_walk_cli_exposes_mujoco_viewer_option(self):
        root = Path(__file__).parents[1]
        result = subprocess.run(
            [sys.executable, str(root / 'scripts/walk_g1.py'), '--help'],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--viewer', result.stdout)

    def test_floating_model_has_free_root_and_29_policy_actuators(self):
        from wmal.envs.g1_locomotion import build_g1_model

        model = build_g1_model()
        self.assertEqual(model.nq, 36)
        self.assertEqual(model.nv, 35)
        self.assertEqual(model.nu, 29)
        self.assertEqual(model.njnt, 30)
        import mujoco
        hip_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_hip_pitch_joint')
        hip_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_hip_roll_joint')
        self.assertAlmostEqual(model.dof_armature[model.jnt_dofadr[hip_pitch_id]],
                               0.0101775200413, places=10)
        self.assertAlmostEqual(model.dof_armature[model.jnt_dofadr[hip_roll_id]],
                               0.025101925, places=10)

    @unittest.skipUnless(importlib.util.find_spec('onnxruntime'), 'Optional ONNX Runtime not installed')
    def test_published_velocity_policy_completes_ten_measured_footsteps(self):
        from wmal.envs.g1_locomotion import simulate_g1_walk

        result = simulate_g1_walk(steps=10, velocity_x=0.2, max_duration_s=8.0)
        self.assertEqual(result['footsteps'], 10)
        self.assertGreater(result['forward_displacement_m'], 0.1)
        self.assertGreater(result['base_height_m'], 0.6)
        self.assertLess(result['tilt_rad'], 0.35)

    @unittest.skipUnless(importlib.util.find_spec('onnxruntime'), 'Optional ONNX Runtime not installed')
    def test_walk_command_prints_machine_readable_success_metrics(self):
        root = Path(__file__).parents[1]
        env = os.environ.copy()
        env['PYTHONPATH'] = str(root / 'src')
        result = subprocess.run(
            [sys.executable, str(root / 'scripts/walk_g1.py'), '--steps', '10'],
            cwd=root, env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'succeeded')
        self.assertEqual(report['footsteps'], 10)


if __name__ == '__main__':
    unittest.main()
