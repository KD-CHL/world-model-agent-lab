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
                self.assertEqual(model.nu,58 if task=='dualrobot_clean_table' else 29)
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
