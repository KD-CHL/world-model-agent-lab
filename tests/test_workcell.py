import unittest
import numpy as np

from wmal.envs.workcell import TASKS, build_workcell


class WorkcellTests(unittest.TestCase):
    def test_all_worlds_compile_and_settle(self):
        import mujoco
        for task in TASKS:
            with self.subTest(task=task):
                scene,model,data = build_workcell(task,seed=7)
                mujoco.mj_step(model,data,nstep=250)
                self.assertTrue(np.isfinite(data.qpos).all())
                self.assertFalse(any(w.number for w in data.warning))
                expected_nu = 58 if task == 'dualrobot_clean_table' else 31 if task == 'stack_block' else 29
                self.assertEqual(model.nu, expected_nu)
                self.assertEqual(model.nflex,1 if task=='fold_towel' else 0)
                for name in scene.objects:
                    self.assertGreater(data.body(name).xpos[2],.70)
                for camera in scene.metadata()['cameras']:
                    self.assertGreaterEqual(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,camera),0)

    def test_seed_and_unknown_task(self):
        _,m,a = build_workcell('stack_block',seed=1)
        _,_,b = build_workcell('stack_block',seed=1)
        np.testing.assert_array_equal(a.qpos,b.qpos)
        with self.assertRaises(ValueError):
            build_workcell('not_a_task')

    def test_stack_block_uses_left_arm_reachable_side_fixture(self):
        scene, model, data = build_workcell('stack_block', seed=0)
        metadata = scene.metadata()
        self.assertEqual(metadata['worktable_center'], [0.0, 0.95])
        self.assertEqual(metadata['gripper']['asset_kind'], 'synthetic_parallel_jaw_proxy')
        self.assertFalse(metadata['gripper']['is_dex1'])
        self.assertLess(abs(data.body('block_0').xpos[0]), .25)
        self.assertGreater(data.body('block_0').xpos[1], .30)
        self.assertGreater(data.body('block_1').xpos[1], .30)
