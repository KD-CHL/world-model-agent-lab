import importlib.util
from pathlib import Path
import unittest
from wmal.communication.contracts import RobotProfile, MotionCommand


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'Optional MuJoCo not installed')
class MujocoTests(unittest.TestCase):
    def test_joint_motion_uses_physics_and_cancel_holds_current_pose(self):
        self.assertIsNotNone(importlib.util.find_spec('wmal.envs.mujoco_backend'), 'Backend missing')
        from wmal.envs.mujoco_backend import MujocoBackend
        profile = RobotProfile('arm', 'arm', {'hinge': (-1, 1)})
        model = Path(__file__).parents[1] / 'robots/assets/interface_probe.xml'
        backend = MujocoBackend(str(model), profile, {'hinge': {'actuator': 'servo', 'mode': 'position'}})
        before = backend.observe()
        cmd = MotionCommand('test', 'arm', before.episode_id, before.step_id, 'joint_positions', {'hinge': .3}, 1)
        backend.begin(cmd)
        for _ in range(500): backend.step()
        after = backend.observe()
        self.assertGreater(after.sim_time_s, before.sim_time_s)
        self.assertAlmostEqual(after.joints['hinge'], .3, delta=.03)
        backend.stop()
        self.assertAlmostEqual(backend.data.ctrl[0], after.joints['hinge'])


if __name__ == '__main__': unittest.main()
