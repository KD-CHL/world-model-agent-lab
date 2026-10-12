import unittest
import numpy as np

from wmal.envs.manipulation_session import ActionCommand, ManipulationSession


class ManipulationSessionTests(unittest.TestCase):
    def setUp(self):
        self.session = ManipulationSession(render=False, control_period_s=.02)
        self.addCleanup(self.session.close)
        self.observation, self.audit = self.session.reset(11, 'nominal')

    def test_observation_and_agent_serialization_have_no_scene_truth(self):
        payload = self.observation.to_agent_dict()
        self.assertNotIn('object_states', payload)
        self.assertNotIn('contacts', payload)
        self.assertNotIn('reset_seed', payload)
        self.assertNotIn('success', payload)
        self.assertFalse(hasattr(self.observation, 'objects'))
        self.assertIsNone(self.observation.rgb)
        truth = self.session.truth_for_verifier()
        self.assertTrue(np.isfinite(truth.object('block_0').position).all())
        self.assertIs(self.session.last_observation, self.observation)

    def test_reset_reproduces_scene_robot_gripper_and_time_and_excludes_settle_steps(self):
        first = self.audit
        self.assertEqual(first.sim_time_s, 0.0)
        self.assertEqual(self.observation.step_id, 0)
        second_observation, second = self.session.reset(11, 'nominal')
        self.assertNotEqual(first.episode_id, second.episode_id)
        self.assertEqual(second.sim_time_s, 0.0)
        self.assertEqual(second_observation.step_id, 0)
        self.assertEqual(first.object_states, second.object_states)
        self.assertEqual(first.joint_positions, second.joint_positions)
        self.assertEqual(first.gripper_opening_m, second.gripper_opening_m)
        self.assertEqual(self.session.trace, ())

    def test_jittered_profile_is_repeatable_per_seed_and_varies_between_seeds(self):
        _, same = self.session.reset(22, 'jittered')
        _, replay = self.session.reset(22, 'jittered')
        _, different = self.session.reset(23, 'jittered')
        self.assertEqual(same.object_states, replay.object_states)
        self.assertNotEqual(same.object_states, different.object_states)
        same_states = {item.name: item for item in same.object_states}
        different_states = {item.name: item for item in different.object_states}
        self.assertEqual(same_states['block_1'], different_states['block_1'])
        self.assertEqual(same_states['block_2'], different_states['block_2'])

    def test_rejects_invalid_or_stale_action_before_advancing_physics(self):
        current_time = self.session.sim_time_s
        invalid = (
            {'joint_targets': {'unknown_joint': .1}},
            {'joint_targets': {'left_shoulder_pitch_joint': float('nan')}},
            {'joint_targets': {'left_shoulder_pitch_joint': True}},
            {'joint_targets': {'left_shoulder_pitch_joint': 99.0}},
            {'expected_step_id': 1},
            {'duration_steps': 0},
            {'duration_steps': 10000},
            {'action_id': ''},
        )
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.session.execute(self.action(**fields))
            self.assertEqual(self.session.sim_time_s, current_time)
        stale_episode = self.action(episode_id='stale-episode')
        with self.assertRaises(ValueError):
            self.session.execute(stale_episode)
        self.assertEqual(self.session.sim_time_s, current_time)

    def test_execution_receipt_preserves_contiguous_steps_and_control_time(self):
        receipt = self.session.execute(self.action(duration_steps=2))
        self.assertEqual(receipt.executed_steps, 2)
        self.assertTrue(hasattr(receipt, 'gripper_command_receipt'))
        self.assertIsNone(receipt.gripper_command_receipt)
        self.assertEqual(receipt.start_step_id, 0)
        self.assertEqual(receipt.end_step_id, 2)
        self.assertAlmostEqual(receipt.sim_duration_s, .04, places=8)
        self.assertEqual([obs.step_id for obs in receipt.observations], [1, 2])
        self.assertEqual(self.session.trace, (receipt,))
        truth_trace = self.session.truth_trace_for_verifier(receipt)
        self.assertEqual([truth.step_id for truth in truth_trace], [1, 2])
        self.assertEqual(self.session.truth_for_verifier().step_id, 2)

    def test_completed_gripper_action_receipt_is_bound_to_parent_action(self):
        command = self.action(action_id='close-block-1', duration_steps=2,
                              gripper_mode='close')
        receipt = self.session.execute(command)
        self.assertTrue(hasattr(receipt, 'gripper_command_receipt'))
        gripper_receipt = receipt.gripper_command_receipt
        self.assertEqual(gripper_receipt.command_id, 'close-block-1')
        self.assertEqual(gripper_receipt.requested_mode, 'close')
        self.assertAlmostEqual(gripper_receipt.requested_opening_m, .034, places=6)
        self.assertTrue(gripper_receipt.accepted)
        self.assertTrue(gripper_receipt.executed)
        self.assertIsNone(gripper_receipt.fault)
        self.assertEqual(receipt.executed_steps, 2)

    def test_duplicate_action_id_is_rejected_without_second_execution(self):
        command = self.action(duration_steps=1)
        self.session.execute(command)
        sim_time = self.session.sim_time_s
        with self.assertRaises(ValueError):
            self.session.execute(command)
        self.assertAlmostEqual(self.session.sim_time_s, sim_time)

    def test_missing_post_action_observation_returns_confirmed_prefix_and_latches_fault(self):
        original_observe = self.session.observe
        calls = {'count': 0}

        def fail_after_physics_step():
            calls['count'] += 1
            if calls['count'] == 1:
                return original_observe()
            raise RuntimeError('injected camera failure')

        self.session.observe = fail_after_physics_step
        receipt = self.session.execute(self.action(duration_steps=3, gripper_mode='close'))
        self.assertTrue(receipt.interrupted)
        self.assertEqual(receipt.executed_steps, 2)
        self.assertEqual(receipt.end_step_id, 2)
        self.assertEqual(len(receipt.observations), 1)
        self.assertEqual(receipt.fault, 'post_action_observation_failed')
        self.assertEqual(self.session.fault_reason, 'post_action_observation_failed')
        self.assertTrue(hasattr(receipt, 'gripper_command_receipt'))
        self.assertTrue(receipt.gripper_command_receipt.accepted)
        self.assertFalse(receipt.gripper_command_receipt.executed)
        self.assertEqual(receipt.gripper_command_receipt.fault,
                         'post_action_observation_failed')
        try:
            truth_trace = self.session.truth_trace_for_verifier(receipt)
        except KeyError as error:
            self.fail(f'completed physics steps must retain verifier truth: {error}')
        self.assertEqual([truth.step_id for truth in truth_trace], [1, 2])
        with self.assertRaises(RuntimeError):
            self.session.execute(self.action(action_id='must-not-retry'))

    def action(self, **changes):
        values = {
            'episode_id': self.observation.episode_id,
            'expected_step_id': self.observation.step_id,
            'action_id': 'test-action',
            'joint_targets': {'left_shoulder_pitch_joint': .1},
            'duration_steps': 1,
        }
        values.update(changes)
        return ActionCommand(**values)


if __name__ == '__main__':
    unittest.main()
