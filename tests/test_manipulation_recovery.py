import unittest
from math import cos, sin
from types import SimpleNamespace

from wmal.envs.manipulation_session import LEFT_ARM_JOINTS, ManipulationSession
from wmal.skills.cartesian_servo import ActuatorAwareCartesianServo
from wmal.skills.manipulation_recovery import (
    RGBXYPlacementMissAssessor, STACK_BLOCK_RECOVERY_MODEL_VERSION,
    StackBlockRGBXYPreReleaseAlignment, StackBlockRGBXYRecovery,
)


class FakeLocalizer:
    def __init__(self, predictions=None, *, radius=0.014724834364995557,
                 visible=True, calibrated=True, model_version='rgbxy-test-v1'):
        self.predictions = list(predictions or [(0.2725, 0.419)] * 3)
        self.visible = visible
        self.calibration = ({'radius_m': radius,
                             'scope': 'stationary_tabletop_episode_max_error'}
                            if calibrated else None)
        self.metadata = SimpleNamespace(model_version=model_version)
        self.checkpoint_sha256 = 'checkpoint-sha256'

    def predict(self, rgb):
        if not self.visible:
            raise ValueError('object is not confidently visible')
        position = self.predictions.pop(0)
        return {
            'position_xy_m': position,
            'position_xy_calibrated_radius_m': self.calibration['radius_m'],
            'position_xy_calibration_scope': self.calibration['scope'],
            'model_version': self.metadata.model_version,
        }


def observations(*, steps=(1408, 1440, 1472), opening=0.12,
                 contacts=(False, False), fault=None):
    return [SimpleNamespace(
        episode_id='episode-1', step_id=step, sim_time_s=step * .02,
        rgb=step,
        gripper=SimpleNamespace(opening_m=opening, target_opening_m=opening,
                                contact_left=contacts[0], contact_right=contacts[1],
                                fault=fault),
    ) for step in steps]


class RGBXYPlacementMissTests(unittest.TestCase):
    def assessor(self, localizer=None):
        return RGBXYPlacementMissAssessor(
            localizer or FakeLocalizer(), target_xy_m=(.20, .42),
            target_tolerance_m=(.055, .055), expected_model_version='rgbxy-test-v1')

    def test_only_confident_stable_off_target_estimate_triggers(self):
        result = self.assessor().assess(observations())

        self.assertTrue(result.trigger)
        self.assertEqual(result.reason, 'confidently_outside_target')
        self.assertEqual(result.observation_step_ids, (1408, 1440, 1472))
        self.assertEqual(result.model_version, 'rgbxy-test-v1')
        self.assertEqual(result.checkpoint_sha256, 'checkpoint-sha256')

    def test_confidently_on_target_does_not_trigger(self):
        localizer = FakeLocalizer(predictions=[(.205, .422)] * 3)

        result = self.assessor(localizer).assess(observations())

        self.assertFalse(result.trigger)
        self.assertEqual(result.reason, 'confidently_within_target')

    def test_calibration_interval_crossing_target_boundary_fails_closed(self):
        localizer = FakeLocalizer(predictions=[(.26, .42)] * 3)

        result = self.assessor(localizer).assess(observations())

        self.assertFalse(result.trigger)
        self.assertEqual(result.reason, 'uncertainty_overlaps_target_boundary')

    def test_inconsistent_frames_do_not_trigger(self):
        localizer = FakeLocalizer(predictions=[(.27, .42), (.282, .42), (.27, .42)])

        result = self.assessor(localizer).assess(observations())

        self.assertFalse(result.trigger)
        self.assertEqual(result.reason, 'localization_not_stationary')

    def test_missing_calibration_or_visible_object_fails_closed(self):
        no_calibration = self.assessor(FakeLocalizer(calibrated=False)).assess(observations())
        not_visible = self.assessor(FakeLocalizer(visible=False)).assess(observations())

        self.assertFalse(no_calibration.trigger)
        self.assertEqual(no_calibration.reason, 'calibration_unavailable')
        self.assertFalse(not_visible.trigger)
        self.assertEqual(not_visible.reason, 'object_not_confidently_visible')

    def test_gripper_must_be_open_fault_free_and_frames_must_be_same_episode(self):
        closed = self.assessor().assess(observations(opening=.04))
        faulted = self.assessor().assess(observations(fault='sensor_fault'))
        contact = self.assessor().assess(observations(contacts=(True, False)))
        mixed = observations()
        mixed[-1].episode_id = 'episode-2'
        mixed_episode = self.assessor().assess(mixed)

        for result in (closed, faulted, contact, mixed_episode):
            self.assertFalse(result.trigger)
        self.assertEqual(closed.reason, 'gripper_not_released')
        self.assertEqual(faulted.reason, 'gripper_fault')
        self.assertEqual(contact.reason, 'gripper_contact_remains')
        self.assertEqual(mixed_episode.reason, 'episode_mismatch')

    def test_too_close_or_too_few_observations_fails_closed(self):
        too_close = self.assessor().assess(observations(steps=(1408, 1416, 1424)))
        too_few = self.assessor().assess(observations(steps=(1408, 1440)))

        self.assertFalse(too_close.trigger)
        self.assertEqual(too_close.reason, 'observations_not_separated')
        self.assertFalse(too_few.trigger)
        self.assertEqual(too_few.reason, 'insufficient_observations')


class StackBlockRecoveryAnchorTests(unittest.TestCase):
    def test_recovery_anchor_uses_initial_rgb_and_measured_bilateral_grasp_pose(self):
        localizer = FakeLocalizer(
            predictions=[(.08, .42)], model_version=STACK_BLOCK_RECOVERY_MODEL_VERSION)
        recovery = StackBlockRGBXYRecovery(localizer)
        initial = SimpleNamespace(rgb='initial-visible-rgb')
        grasp = SimpleNamespace(
            gripper=SimpleNamespace(contact_left=True, contact_right=True, fault=None),
            tool_position_m=(.06, .42, .84),
            tool_quaternion_wxyz=(1., 0., 0., 0.),
        )

        anchor, reason = recovery.capture_grasp_anchor(initial, grasp)

        self.assertIsNone(reason)
        self.assertEqual(anchor['initial_object_xy_m'], (.08, .42))
        self.assertAlmostEqual(anchor['object_to_tool_offset_m'][0], .02)
        self.assertAlmostEqual(anchor['object_to_tool_offset_m'][1], 0.)
        self.assertAlmostEqual(anchor['object_to_tool_offset_m'][2], -.025)

    def test_recovery_anchor_declines_without_measured_bilateral_contact(self):
        localizer = FakeLocalizer(model_version=STACK_BLOCK_RECOVERY_MODEL_VERSION)
        recovery = StackBlockRGBXYRecovery(localizer)
        initial = SimpleNamespace(rgb='initial-visible-rgb')
        grasp = SimpleNamespace(
            gripper=SimpleNamespace(contact_left=True, contact_right=False, fault=None),
            tool_position_m=(.06, .42, .84),
            tool_quaternion_wxyz=(1., 0., 0., 0.),
        )

        anchor, reason = recovery.capture_grasp_anchor(initial, grasp)

        self.assertIsNone(anchor)
        self.assertEqual(reason, 'successful_bilateral_grasp_not_observed')


class FakePreReleaseServo:
    def __init__(self, *, xyz_delta=(.012, -.019, .04),
                 orientation_error=.05, status='failed', reason='convergence_iteration_limit',
                 final_contacts=None, position_tolerance_m=.008,
                 orientation_tolerance_rad=.08):
        self.xyz_delta = xyz_delta
        self.orientation_error = orientation_error
        self.status = status
        self.reason = reason
        self.final_contacts = final_contacts
        self.position_tolerance_m = position_tolerance_m
        self.orientation_tolerance_rad = orientation_tolerance_rad
        self.calls = []

    def move(self, session, *, target_position_m, target_quaternion_wxyz,
             budget_control_steps, cancel_event=None, observation_guard=None):
        self.calls.append((target_position_m, target_quaternion_wxyz,
                           budget_control_steps, cancel_event))
        observation = session.observe()
        observation.tool_position_m = tuple(
            target_position_m[i] + self.xyz_delta[i] for i in range(3))
        observation.tool_quaternion_wxyz = (
            cos(self.orientation_error / 2.),
            sin(self.orientation_error / 2.), 0., 0.)
        if self.final_contacts is not None:
            observation.gripper.contact_left = self.final_contacts[0]
            observation.gripper.contact_right = self.final_contacts[1]
        guard_reason = observation_guard(observation) if observation_guard else None
        if guard_reason:
            self.status, self.reason = 'failed', f'observation_guard:{guard_reason}'
        receipt = SimpleNamespace(executed_steps=16)
        return SimpleNamespace(
            status=self.status, reason=self.reason, receipts=(receipt,),
            control_steps=16, final_observation=observation,
            position_error_m=sum(value * value for value in self.xyz_delta) ** .5,
            orientation_error_rad=self.orientation_error,
        )


class FakePreReleaseSession:
    def __init__(self, *, contacts=(True, True), fault=None,
                 opening=.04, target_opening=.034):
        self.observation = SimpleNamespace(
            episode_id='align-episode', step_id=640,
            tool_position_m=(.20, .42, .95),
            tool_quaternion_wxyz=(1., 0., 0., 0.),
            gripper=SimpleNamespace(
                contact_left=contacts[0], contact_right=contacts[1],
                fault=fault, opening_m=opening,
                target_opening_m=target_opening),
        )

    def observe(self):
        return self.observation


class FakeHoldPreReleaseSession(FakePreReleaseSession):
    def __init__(self):
        super().__init__()
        self.hold_commands = []

    def actuator_targets_for(self, joint_names):
        return {name: 0.0 for name in joint_names}

    def execute(self, command):
        self.hold_commands.append(command)
        observations = []
        for _ in range(command.duration_steps):
            self.observation.step_id += 1
            observations.append(SimpleNamespace(
                episode_id=self.observation.episode_id,
                step_id=self.observation.step_id))
        return SimpleNamespace(
            observations=tuple(observations), executed_steps=command.duration_steps,
            interrupted=False, fault=None, end_step_id=self.observation.step_id)


class StackBlockPreReleaseAlignmentTests(unittest.TestCase):
    def make_alignment(self, *, servo=None):
        localizer = FakeLocalizer(
            predictions=[(.08, .42)], model_version=STACK_BLOCK_RECOVERY_MODEL_VERSION)
        return StackBlockRGBXYPreReleaseAlignment(
            localizer, servo=servo or FakePreReleaseServo())

    @staticmethod
    def anchor(alignment):
        initial = SimpleNamespace(rgb='initial-visible-rgb')
        grasp = SimpleNamespace(
            gripper=SimpleNamespace(contact_left=True, contact_right=True, fault=None),
            tool_position_m=(.06, .42, .84),
            tool_quaternion_wxyz=(1., 0., 0., 0.),
        )
        anchor, reason = alignment.capture_grasp_anchor(initial, grasp)
        if reason:
            raise AssertionError(reason)
        return anchor

    def test_default_alignment_servo_uses_available_control_budget(self):
        localizer = FakeLocalizer(
            predictions=[(.08, .42)], model_version=STACK_BLOCK_RECOVERY_MODEL_VERSION)
        alignment = StackBlockRGBXYPreReleaseAlignment(localizer)

        self.assertEqual(alignment.servo.steps_per_iteration, 12)
        self.assertEqual(alignment.servo.max_iterations, 42)

    def test_converged_alignment_with_bilateral_contact_authorizes_release(self):
        servo = FakePreReleaseServo(
            xyz_delta=(.002, .003, .004), orientation_error=.04,
            status='succeeded', reason=None)
        alignment = self.make_alignment(servo=servo)
        anchor = self.anchor(alignment)
        session = FakePreReleaseSession(contacts=(True, True))

        result = alignment.execute(session, anchor, budget_control_steps=512)

        self.assertEqual(result.status, 'ready_to_release')
        self.assertEqual(result.reason, 'within_release_envelope')
        self.assertEqual(result.target_tool_pose_m, (.18, .42, .91))
        self.assertEqual(servo.calls[0][0], (.18, .42, .91))
        self.assertEqual(servo.calls[0][2], 512)
        self.assertEqual(result.final_observation.gripper.contact_left, True)
        self.assertEqual(result.final_observation.gripper.contact_right, True)
        self.assertEqual(result.control_steps, 16)
        self.assertEqual(len(result.receipts), 1)

    def test_nonconverged_pose_inside_loose_envelope_does_not_authorize_release(self):
        servo = FakePreReleaseServo(
            xyz_delta=(.012, -.019, .04), orientation_error=.05,
            status='failed', reason='convergence_iteration_limit')
        alignment = self.make_alignment(servo=servo)

        result = alignment.execute(
            FakePreReleaseSession(contacts=(True, True)),
            self.anchor(alignment), budget_control_steps=512)

        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.reason, 'alignment_not_converged')

    def test_inconsistent_servo_success_outside_position_tolerance_does_not_release(self):
        servo = FakePreReleaseServo(
            xyz_delta=(.009, 0., 0.), status='succeeded', reason=None)
        alignment = self.make_alignment(servo=servo)

        result = alignment.execute(
            FakePreReleaseSession(contacts=(True, True)),
            self.anchor(alignment), budget_control_steps=512)

        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.reason, 'alignment_position_tolerance_not_met')

    def test_orientation_outside_servo_tolerance_does_not_authorize_release(self):
        servo = FakePreReleaseServo(
            xyz_delta=(.002, .003, .004), orientation_error=.081,
            status='succeeded', reason=None)
        alignment = self.make_alignment(servo=servo)

        result = alignment.execute(
            FakePreReleaseSession(contacts=(True, True)),
            self.anchor(alignment), budget_control_steps=512)

        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.reason, 'alignment_orientation_tolerance_not_met')

    def test_out_of_envelope_pose_and_gripper_fault_never_authorize_release(self):
        bad_xy = self.make_alignment(servo=FakePreReleaseServo(
            xyz_delta=(.021, 0., .02), status='succeeded', reason=None,
            position_tolerance_m=.05))
        result_xy = bad_xy.execute(FakePreReleaseSession(), self.anchor(bad_xy),
                                   budget_control_steps=512)
        bad_height = self.make_alignment(servo=FakePreReleaseServo(
            xyz_delta=(0., 0., .061), status='succeeded', reason=None,
            position_tolerance_m=.1))
        result_height = bad_height.execute(
            FakePreReleaseSession(), self.anchor(bad_height), budget_control_steps=512)
        faulted = self.make_alignment()
        result_fault = faulted.execute(
            FakePreReleaseSession(fault='gripper_sensor_fault'),
            self.anchor(faulted), budget_control_steps=512)

        self.assertEqual(result_xy.status, 'failed')
        self.assertEqual(result_xy.reason, 'tool_outside_release_envelope')
        self.assertEqual(result_height.reason, 'tool_outside_release_envelope')
        self.assertEqual(result_fault.reason, 'gripper_fault')
        self.assertNotEqual(result_fault.status, 'ready_to_release')

    def test_bad_anchor_declines_without_motion(self):
        servo = FakePreReleaseServo()
        alignment = self.make_alignment(servo=servo)

        result = alignment.execute(FakePreReleaseSession(), None,
                                   budget_control_steps=512)

        self.assertEqual(result.status, 'declined')
        self.assertEqual(result.reason, 'grasp_anchor_unavailable')
        self.assertEqual(servo.calls, [])

    def test_released_gripper_is_rejected_before_alignment_motion(self):
        servo = FakePreReleaseServo()
        alignment = self.make_alignment(servo=servo)

        result = alignment.execute(
            FakePreReleaseSession(opening=.12, target_opening=.12),
            self.anchor(alignment), budget_control_steps=512)

        self.assertEqual(result.status, 'declined')
        self.assertEqual(result.reason, 'gripper_not_closed_for_alignment')
        self.assertEqual(servo.calls, [])

    def test_missing_bilateral_contact_declines_before_alignment(self):
        servo = FakePreReleaseServo()
        alignment = self.make_alignment(servo=servo)

        result = alignment.execute(
            FakePreReleaseSession(contacts=(True, False)), self.anchor(alignment),
            budget_control_steps=512)

        self.assertEqual(result.status, 'declined')
        self.assertEqual(result.reason, 'bilateral_contact_missing_before_alignment')
        self.assertEqual(servo.calls, [])

    def test_contact_loss_during_alignment_stops_before_release_authorization(self):
        servo = FakePreReleaseServo(final_contacts=(True, False))
        alignment = self.make_alignment(servo=servo)

        result = alignment.execute(
            FakePreReleaseSession(), self.anchor(alignment),
            budget_control_steps=512)

        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.reason, 'bilateral_gripper_contact_lost')
        self.assertEqual(result.final_observation.gripper.contact_left, True)
        self.assertEqual(result.final_observation.gripper.contact_right, False)

    def test_contact_loss_runs_bounded_stationary_probe_and_remains_fail_closed(self):
        servo = FakePreReleaseServo(final_contacts=(False, False))
        localizer = FakeLocalizer(
            predictions=[(.08, .42)], model_version=STACK_BLOCK_RECOVERY_MODEL_VERSION)
        alignment = StackBlockRGBXYPreReleaseAlignment(
            localizer, servo=servo, contact_loss_hold_steps=96)
        session = FakeHoldPreReleaseSession()

        result = alignment.execute(
            session, self.anchor(alignment), budget_control_steps=608)

        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.reason, 'bilateral_gripper_contact_lost')
        self.assertEqual(result.control_steps, 112)
        self.assertEqual(len(result.receipts), 4)
        self.assertEqual([command.duration_steps for command in session.hold_commands],
                         [32, 32, 32])
        self.assertTrue(all(command.gripper_mode is None
                            for command in session.hold_commands))
        self.assertTrue(all(command.expected_step_id >= 640
                            for command in session.hold_commands))


class ManipulationJacobianTests(unittest.TestCase):
    def test_session_exposes_measured_left_arm_tool_pose_jacobian(self):
        session = ManipulationSession(render=False)
        try:
            observation = session.observe()
            jacobian = session.tool_pose_jacobian()
        finally:
            session.close()

        self.assertEqual(len(jacobian), 6)
        self.assertTrue(all(len(row) == 7 for row in jacobian))
        self.assertTrue(all(abs(value) < float('inf') for row in jacobian for value in row))
        self.assertEqual(len(observation.joint_names), 29)

    def test_session_exposes_finite_actuator_targets_for_left_arm_only(self):
        session = ManipulationSession(render=False)
        try:
            targets = session.actuator_targets_for(LEFT_ARM_JOINTS)
            self.assertEqual(set(targets), set(LEFT_ARM_JOINTS))
            self.assertTrue(all(abs(value) < float('inf') for value in targets.values()))
            with self.assertRaises(ValueError):
                session.actuator_targets_for(('waist_pitch_joint',))
            self.assertIsNotNone(session.gripper_spec)
        finally:
            session.close()


class FakeServoSession:
    joint_names = (
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint', 'left_elbow_joint',
        'left_wrist_roll_joint', 'left_wrist_pitch_joint',
        'left_wrist_yaw_joint',
    )

    def __init__(self, actuator_bias=None):
        self.joints = [0.] * 7
        self.actuator_bias = list(actuator_bias or [0.] * 7)
        self.actuator_targets = list(self.actuator_bias)
        self.position = [.20, .42, .90]
        self.quaternion = (1., 0., 0., 0.)
        self.step_id = 0
        self.action_count = 0

    def observe(self):
        return SimpleNamespace(
            episode_id='servo-episode', step_id=self.step_id,
            joint_names=self.joint_names, joint_position=tuple(self.joints),
            tool_position_m=tuple(self.position),
            tool_quaternion_wxyz=self.quaternion,
        )

    def tool_pose_jacobian(self):
        return tuple(tuple(1. if row == col else 0. for col in range(7))
                     for row in range(6))

    def joint_limits_for(self, names):
        return {name: (-1., 1.) for name in names}

    def actuator_targets_for(self, names):
        return {name: self.actuator_targets[self.joint_names.index(name)]
                for name in names}

    def execute(self, command):
        self.actuator_targets = [dict(command.joint_targets)[name]
                                 for name in self.joint_names]
        self.joints = [target - bias for target, bias in
                       zip(self.actuator_targets, self.actuator_bias)]
        self.position = [.20 + self.joints[0], .42 + self.joints[1],
                         .90 + self.joints[2]]
        self.step_id += command.duration_steps
        self.action_count += 1
        return SimpleNamespace(interrupted=False, executed_steps=command.duration_steps,
                               fault=None, end_step_id=self.step_id)


class CartesianServoTests(unittest.TestCase):
    def test_servo_accepts_convergence_reached_on_final_iteration(self):
        session = FakeServoSession()
        servo = ActuatorAwareCartesianServo(
            max_iterations=1, steps_per_iteration=1,
            max_joint_step_rad=.035, damping=.001)

        result = servo.move(
            session, target_position_m=(.212, .42, .90),
            target_quaternion_wxyz=(1., 0., 0., 0.),
            budget_control_steps=1)

        self.assertEqual(result.status, 'succeeded')
        self.assertLessEqual(result.position_error_m, servo.position_tolerance_m)
        self.assertLessEqual(result.orientation_error_rad,
                             servo.orientation_tolerance_rad)
        self.assertEqual(result.control_steps, 1)

    def test_servo_replans_from_measured_pose_until_converged(self):
        session = FakeServoSession()
        servo = ActuatorAwareCartesianServo(
            max_iterations=8, steps_per_iteration=8,
            max_joint_step_rad=.03, damping=.001)

        result = servo.move(
            session, target_position_m=(.28, .38, .95),
            target_quaternion_wxyz=(1., 0., 0., 0.),
            budget_control_steps=64)

        self.assertEqual(result.status, 'succeeded')
        self.assertLessEqual(result.position_error_m, .008)
        self.assertLessEqual(result.orientation_error_rad, .08)
        self.assertGreater(session.action_count, 1)
        self.assertLessEqual(result.control_steps, 64)

    def test_servo_preserves_actuator_hold_bias_while_applying_cartesian_updates(self):
        session = FakeServoSession(actuator_bias=(.12, -.14, .08, -.05,
                                                  .09, .13, -.10))
        servo = ActuatorAwareCartesianServo(
            max_iterations=12, steps_per_iteration=4,
            max_joint_step_rad=.03, damping=.001)

        result = servo.move(
            session, target_position_m=(.24, .40, .94),
            target_quaternion_wxyz=(1., 0., 0., 0.),
            budget_control_steps=48)

        self.assertEqual(result.status, 'succeeded')
        self.assertLessEqual(result.position_error_m, .008)
        self.assertLessEqual(result.orientation_error_rad, .08)

    def test_servo_rejects_out_of_workspace_target_without_dispatch(self):
        session = FakeServoSession()
        servo = ActuatorAwareCartesianServo()

        result = servo.move(
            session, target_position_m=(.5, .42, .9),
            target_quaternion_wxyz=(1., 0., 0., 0.),
            budget_control_steps=128)

        self.assertEqual(result.status, 'declined')
        self.assertEqual(result.reason, 'target_outside_workspace')
        self.assertEqual(session.action_count, 0)

    def test_servo_observation_guard_stops_after_first_confirmed_step(self):
        session = FakeServoSession()
        servo = ActuatorAwareCartesianServo(
            max_iterations=8, steps_per_iteration=8,
            max_joint_step_rad=.03, damping=.001)

        result = servo.move(
            session, target_position_m=(.28, .38, .95),
            target_quaternion_wxyz=(1., 0., 0., 0.),
            budget_control_steps=64,
            observation_guard=lambda observation:
                'guard_condition_met' if observation.step_id > 0 else None)

        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.reason, 'observation_guard:guard_condition_met')
        self.assertEqual(result.control_steps, 8)
        self.assertEqual(session.action_count, 1)


if __name__ == '__main__':
    unittest.main()
