import unittest

import numpy as np

from wmal.envs.workcell import build_workcell
from wmal.robots.gripper import GripperSpec, MuJoCoParallelGripper


class ParallelGripperTests(unittest.TestCase):
    def setUp(self):
        self.scene, self.model, self.data = build_workcell('stack_block', seed=2)
        spec = GripperSpec.from_workcell_metadata(self.scene.metadata()['gripper'])
        self.gripper = MuJoCoParallelGripper(
            self.model, spec, object_geom_names=('block_0_geom', 'block_1_geom', 'block_2_geom'))

    def test_reset_opens_gripper_and_reports_measured_state(self):
        state = self.gripper.reset(self.data)
        self.assertGreater(state.opening_m, .07)
        self.assertTrue(state.at_target)
        self.assertFalse(state.contact_left)
        self.assertFalse(state.contact_right)

    def test_close_and_open_commands_move_independent_fingers_within_bounds(self):
        receipt = self.gripper.command(
            self.data, 'close', opening_m=0.0, speed_m_s=.02, force_limit_n=4.0)
        self.assertTrue(receipt.accepted)
        self.assertEqual(receipt.requested_mode, 'close')
        for _ in range(20):
            self.gripper.advance(self.data, .02)
        closed = self.gripper.observe(self.data)
        self.assertLess(closed.target_opening_m, .05)
        qpos = [self.data.qpos[index] for index in self.gripper.qpos_addresses]
        self.assertAlmostEqual(qpos[0], qpos[1], places=5)

        self.gripper.command(
            self.data, 'open', opening_m=.07, speed_m_s=.02, force_limit_n=4.0)
        for _ in range(20):
            self.gripper.advance(self.data, .02)
        self.assertGreater(self.gripper.observe(self.data).opening_m, .065)

    def test_hold_preserves_measured_opening(self):
        before = self.gripper.observe(self.data).opening_m
        self.gripper.command(
            self.data, 'hold', opening_m=before, speed_m_s=.02, force_limit_n=4.0)
        for _ in range(3):
            self.gripper.advance(self.data, .02)
        self.assertAlmostEqual(self.gripper.observe(self.data).target_opening_m, before, places=6)

    def test_rejects_unknown_mode_nonfinite_boolean_and_out_of_range_commands(self):
        bad_commands = (
            ('squeeze', .05, .02, 4.0),
            ('open', float('nan'), .02, 4.0),
            ('open', .05, float('inf'), 4.0),
            ('open', .05, .02, float('nan')),
            ('open', True, .02, 4.0),
            ('open', .2, .02, 4.0),
            ('open', .05, .2, 4.0),
            ('open', .05, .02, 100.0),
        )
        for command in bad_commands:
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.gripper.command(self.data, command[0], opening_m=command[1],
                                     speed_m_s=command[2], force_limit_n=command[3])

    def test_contact_flags_only_reflect_named_object_geometry(self):
        self.assertEqual(set(self.gripper.object_geom_ids), {
            self.model.geom('block_0_geom').id,
            self.model.geom('block_1_geom').id,
            self.model.geom('block_2_geom').id,
        })
        self.assertFalse(self.gripper.observe(self.data).contact_left)
        self.assertFalse(self.gripper.observe(self.data).contact_right)


if __name__ == '__main__':
    unittest.main()
