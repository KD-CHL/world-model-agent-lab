"""Observation-only placement-miss decisions for the stack-block task."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
from statistics import median
from uuid import uuid4

from wmal.envs.manipulation_session import ActionCommand, LEFT_ARM_JOINTS
from wmal.skills.cartesian_servo import ActuatorAwareCartesianServo


_CALIBRATION_SCOPE = 'stationary_tabletop_episode_max_error'
STACK_BLOCK_RECOVERY_MODEL_VERSION = 'rgbxy-2411-24-9e1675b8f6df'
STACK_BLOCK_TARGET_XY_M = (.20, .42)
STACK_BLOCK_TARGET_TOLERANCE_M = (.055, .055)
STACK_BLOCK_TABLE_CENTER_Z_M = .815
STACK_BLOCK_STACKED_CENTER_Z_M = .885


@dataclass(frozen=True)
class PlacementAssessment:
    trigger: bool
    reason: str
    episode_id: str | None = None
    observation_step_ids: tuple[int, ...] = ()
    position_xy_m: tuple[float, float] | None = None
    target_xy_m: tuple[float, float] | None = None
    target_tolerance_m: tuple[float, float] | None = None
    calibrated_radius_m: float | None = None
    lower_error_bound_m: tuple[float, float] | None = None
    model_version: str | None = None
    visual_domain_sha256: str | None = None
    checkpoint_sha256: str | None = None
    calibration_sha256: str | None = None

    def to_dict(self):
        return asdict(self)


class RGBXYPlacementMissAssessor:
    """Trigger only when calibrated RGB observations prove a stable XY miss.

    The assessor consumes robot observations after the intentional release. It
    does not accept verifier output, object truth, reset seeds, or simulator
    internals, so none of those can become an action-selection shortcut.
    """

    def __init__(self, localizer, *, target_xy_m, target_tolerance_m,
                 expected_model_version, min_opening_m=.10,
                 min_frame_gap_steps=16, max_frame_span_steps=128,
                 max_position_spread_m=.008):
        if not callable(getattr(localizer, 'predict', None)):
            raise ValueError('A frozen RGB-XY localizer is required')
        if len(target_xy_m) != 2 or len(target_tolerance_m) != 2:
            raise ValueError('Placement target and tolerance must be XY pairs')
        target = tuple(float(v) for v in target_xy_m)
        tolerance = tuple(float(v) for v in target_tolerance_m)
        scalars = (min_opening_m, max_position_spread_m)
        if (not all(isfinite(v) for v in (*target, *tolerance, *scalars))
                or any(v <= 0 for v in tolerance)
                or min_opening_m <= 0 or max_position_spread_m <= 0
                or type(min_frame_gap_steps) is not int or min_frame_gap_steps < 1
                or type(max_frame_span_steps) is not int
                or max_frame_span_steps < min_frame_gap_steps * 2):
            raise ValueError('Invalid placement-miss assessment limits')
        if not isinstance(expected_model_version, str) or not expected_model_version:
            raise ValueError('The expected frozen model version is required')
        self.localizer = localizer
        self.target_xy_m = target
        self.target_tolerance_m = tolerance
        self.expected_model_version = expected_model_version
        self.min_opening_m = float(min_opening_m)
        self.min_frame_gap_steps = min_frame_gap_steps
        self.max_frame_span_steps = max_frame_span_steps
        self.max_position_spread_m = float(max_position_spread_m)

    def assess(self, observations):
        """Return a fail-closed decision from the three latest released views."""
        rows = tuple(observations or ())
        if len(rows) < 3:
            return self._decline('insufficient_observations')
        rows = rows[-3:]
        episode_id = getattr(rows[0], 'episode_id', None)
        steps = tuple(getattr(row, 'step_id', None) for row in rows)
        if not episode_id or any(getattr(row, 'episode_id', None) != episode_id
                                 for row in rows):
            return self._decline('episode_mismatch')
        if (any(type(step) is not int or step < 0 for step in steps)
                or any(b <= a for a, b in zip(steps, steps[1:]))):
            return self._decline('observation_order_invalid', episode_id, steps)
        gaps = tuple(b - a for a, b in zip(steps, steps[1:]))
        if min(gaps) < self.min_frame_gap_steps:
            return self._decline('observations_not_separated', episode_id, steps)
        if steps[-1] - steps[0] > self.max_frame_span_steps:
            return self._decline('observation_window_too_wide', episode_id, steps)

        for row in rows:
            gripper = getattr(row, 'gripper', None)
            if gripper is None:
                return self._decline('gripper_state_unavailable', episode_id, steps)
            if getattr(gripper, 'fault', None):
                return self._decline('gripper_fault', episode_id, steps)
            if getattr(gripper, 'contact_left', False) or getattr(gripper, 'contact_right', False):
                return self._decline('gripper_contact_remains', episode_id, steps)
            opening = getattr(gripper, 'opening_m', None)
            target_opening = getattr(gripper, 'target_opening_m', None)
            if (not self._finite(opening) or not self._finite(target_opening)
                    or opening < self.min_opening_m or target_opening < self.min_opening_m):
                return self._decline('gripper_not_released', episode_id, steps)
            if getattr(row, 'rgb', None) is None:
                return self._decline('object_not_confidently_visible', episode_id, steps)

        calibration = getattr(self.localizer, 'calibration', None)
        if not isinstance(calibration, dict):
            return self._decline('calibration_unavailable', episode_id, steps)
        if calibration.get('scope') != _CALIBRATION_SCOPE:
            return self._decline('calibration_scope_unsupported', episode_id, steps)
        radius = calibration.get('radius_m')
        if not self._finite(radius) or radius < 0:
            return self._decline('calibration_invalid', episode_id, steps)
        metadata = getattr(self.localizer, 'metadata', None)
        model_version = getattr(metadata, 'model_version', None)
        visual_domain = getattr(metadata, 'visual_domain_sha256', None)
        if model_version != self.expected_model_version:
            return self._decline('model_version_mismatch', episode_id, steps)

        predictions = []
        for row in rows:
            try:
                output = self.localizer.predict(row.rgb)
            except (ValueError, RuntimeError):
                return self._decline('object_not_confidently_visible', episode_id, steps)
            xy = output.get('position_xy_m')
            if (output.get('model_version') != self.expected_model_version
                    or output.get('position_xy_calibration_scope') != _CALIBRATION_SCOPE
                    or not self._finite_pair(xy)):
                return self._decline('localization_output_invalid', episode_id, steps)
            prediction_radius = output.get('position_xy_calibrated_radius_m')
            if (not self._finite(prediction_radius)
                    or abs(float(prediction_radius) - float(radius)) > 1e-9):
                return self._decline('calibration_output_mismatch', episode_id, steps)
            predictions.append(tuple(float(v) for v in xy))

        spread = tuple(max(point[i] for point in predictions)
                       - min(point[i] for point in predictions) for i in range(2))
        if max(spread) > self.max_position_spread_m:
            return self._decline('localization_not_stationary', episode_id, steps)
        position = tuple(float(median(point[i] for point in predictions)) for i in range(2))
        lower_error = tuple(max(0., abs(position[i] - self.target_xy_m[i]) - float(radius))
                            for i in range(2))
        upper_error = tuple(abs(position[i] - self.target_xy_m[i]) + float(radius)
                            for i in range(2))
        common = {
            'episode_id': episode_id,
            'observation_step_ids': steps,
            'position_xy_m': position,
            'target_xy_m': self.target_xy_m,
            'target_tolerance_m': self.target_tolerance_m,
            'calibrated_radius_m': float(radius),
            'lower_error_bound_m': lower_error,
            'model_version': model_version,
            'visual_domain_sha256': visual_domain,
            'checkpoint_sha256': getattr(self.localizer, 'checkpoint_sha256', None),
            'calibration_sha256': self._calibration_sha256(calibration),
        }
        if any(lower_error[i] > self.target_tolerance_m[i] for i in range(2)):
            return PlacementAssessment(True, 'confidently_outside_target', **common)
        if all(upper_error[i] <= self.target_tolerance_m[i] for i in range(2)):
            return PlacementAssessment(False, 'confidently_within_target', **common)
        return PlacementAssessment(False, 'uncertainty_overlaps_target_boundary', **common)

    @staticmethod
    def _finite(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and isfinite(float(value)))

    @classmethod
    def _finite_pair(cls, value):
        try:
            return len(value) == 2 and all(cls._finite(v) for v in value)
        except TypeError:
            return False

    @staticmethod
    def _calibration_sha256(calibration):
        content = json.dumps(calibration, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return sha256(content).hexdigest()

    @staticmethod
    def _decline(reason, episode_id=None, step_ids=()):
        return PlacementAssessment(False, reason, episode_id=episode_id,
                                   observation_step_ids=tuple(step_ids))


@dataclass(frozen=True)
class RegraspRecoveryRun:
    status: str
    reason: str | None
    receipts: tuple
    control_steps: int
    final_observation: object
    stages: tuple
    pick_tool_pose_m: tuple[float, float, float] | None = None
    place_tool_pose_m: tuple[float, float, float] | None = None

    def to_dict(self):
        return {
            'status': self.status,
            'reason': self.reason,
            'control_steps': self.control_steps,
            'start_step_id': (self.receipts[0].start_step_id if self.receipts else None),
            'end_step_id': (self.receipts[-1].end_step_id if self.receipts else
                            getattr(self.final_observation, 'step_id', None)),
            'stages': list(self.stages),
            'pick_tool_pose_m': self.pick_tool_pose_m,
            'place_tool_pose_m': self.place_tool_pose_m,
        }


class StackBlockRGBXYRecovery:
    """One post-release, observation-triggered regrasp and corrected place.

    It is deliberately scoped to the frozen RGB-XY model and this fixed stack
    workcell. The pose offsets are estimated from the reset RGB image and the
    measured successful grasp pose, then the same measured orientation is kept
    through the recovery. No simulator object state or verifier result enters
    the trigger or motion plan.
    """

    def __init__(self, localizer, *, max_control_steps=2048, servo=None):
        if not callable(getattr(localizer, 'predict', None)):
            raise ValueError('A frozen RGB-XY localizer is required')
        if type(max_control_steps) is not int or max_control_steps < 512:
            raise ValueError('Recovery requires a finite budget of at least 512 control steps')
        metadata = getattr(localizer, 'metadata', None)
        if getattr(metadata, 'model_version', None) != STACK_BLOCK_RECOVERY_MODEL_VERSION:
            raise ValueError('Recovery checkpoint is not the allow-listed frozen model')
        self.localizer = localizer
        self.assessor = RGBXYPlacementMissAssessor(
            localizer, target_xy_m=STACK_BLOCK_TARGET_XY_M,
            target_tolerance_m=STACK_BLOCK_TARGET_TOLERANCE_M,
            expected_model_version=STACK_BLOCK_RECOVERY_MODEL_VERSION)
        self.max_control_steps = max_control_steps
        self.servo = servo or ActuatorAwareCartesianServo()
        self._attempted = False

    def capture_grasp_anchor(self, initial_observation, grasp_observation):
        """Estimate object-to-tool offset using only the reset view and contact pose."""
        if initial_observation is None or initial_observation.rgb is None:
            return None, 'initial_rgb_unavailable'
        gripper = getattr(grasp_observation, 'gripper', None)
        if (gripper is None or not gripper.contact_left or not gripper.contact_right
                or gripper.fault):
            return None, 'successful_bilateral_grasp_not_observed'
        calibration = getattr(self.localizer, 'calibration', None)
        if not isinstance(calibration, dict) or calibration.get('scope') != _CALIBRATION_SCOPE:
            return None, 'calibration_unavailable'
        radius = calibration.get('radius_m')
        if not RGBXYPlacementMissAssessor._finite(radius) or radius > .02:
            return None, 'initial_localization_uncalibrated'
        try:
            estimate = self.localizer.predict(initial_observation.rgb)
            xy = estimate.get('position_xy_m')
            if (estimate.get('model_version') != STACK_BLOCK_RECOVERY_MODEL_VERSION
                    or estimate.get('position_xy_calibration_scope') != _CALIBRATION_SCOPE
                    or not RGBXYPlacementMissAssessor._finite_pair(xy)
                    or abs(float(estimate.get('position_xy_calibrated_radius_m', float('inf')))
                           - float(radius)) > 1e-9):
                return None, 'initial_localization_invalid'
        except (ValueError, RuntimeError, TypeError):
            return None, 'initial_object_not_confidently_visible'
        if not (.0 <= xy[0] <= .32 and .25 <= xy[1] <= .58):
            return None, 'initial_object_outside_recovery_workspace'
        try:
            tool_position = tuple(float(value) for value in grasp_observation.tool_position_m)
            quaternion = tuple(float(value) for value in grasp_observation.tool_quaternion_wxyz)
        except (TypeError, ValueError):
            return None, 'grasp_pose_unavailable'
        if (len(tool_position) != 3 or len(quaternion) != 4
                or not all(isfinite(value) for value in (*tool_position, *quaternion))):
            return None, 'grasp_pose_invalid'
        offset = (float(xy[0]) - tool_position[0],
                  float(xy[1]) - tool_position[1],
                  STACK_BLOCK_TABLE_CENTER_Z_M - tool_position[2])
        if any(abs(value) > .08 for value in offset):
            return None, 'measured_grasp_offset_out_of_range'
        norm = sqrt(sum(value * value for value in quaternion))
        if norm < 1e-8:
            return None, 'grasp_orientation_invalid'
        anchor = {
            'initial_object_xy_m': (float(xy[0]), float(xy[1])),
            'object_to_tool_offset_m': offset,
            'grasp_tool_quaternion_wxyz': tuple(value / norm for value in quaternion),
            'model_version': STACK_BLOCK_RECOVERY_MODEL_VERSION,
            'checkpoint_sha256': getattr(self.localizer, 'checkpoint_sha256', None),
            'calibrated_radius_m': float(radius),
        }
        return anchor, None

    def execute_once(self, session, assessment, anchor, *, on_receipt=None,
                     cancel_event=None):
        """Run one bounded regrasp and target correction after an accepted miss."""
        if self._attempted:
            return self._empty_run(session, 'attempt_budget_exhausted')
        if not isinstance(assessment, PlacementAssessment) or not assessment.trigger:
            return self._empty_run(session, 'placement_miss_not_confirmed')
        if not isinstance(anchor, dict) or anchor.get('model_version') != STACK_BLOCK_RECOVERY_MODEL_VERSION:
            return self._empty_run(session, 'grasp_anchor_unavailable')
        try:
            object_xy = self._finite_pair(assessment.position_xy_m)
            offset = tuple(float(value) for value in anchor['object_to_tool_offset_m'])
            quaternion = tuple(float(value) for value in anchor['grasp_tool_quaternion_wxyz'])
            if (not RGBXYPlacementMissAssessor._finite_pair(object_xy)
                    or len(offset) != 3 or len(quaternion) != 4
                    or not all(isfinite(value) for value in (*offset, *quaternion))):
                raise ValueError('invalid recovery pose anchor')
        except (KeyError, TypeError, ValueError):
            return self._empty_run(session, 'grasp_anchor_invalid')
        pick_tool = (object_xy[0] - offset[0], object_xy[1] - offset[1],
                     STACK_BLOCK_TABLE_CENTER_Z_M - offset[2])
        place_tool = (STACK_BLOCK_TARGET_XY_M[0] - offset[0],
                      STACK_BLOCK_TARGET_XY_M[1] - offset[1],
                      STACK_BLOCK_STACKED_CENTER_Z_M - offset[2])
        clear_z = 1.04
        if any(not self.servo._inside_workspace(point) for point in (
                (pick_tool[0], pick_tool[1], clear_z), pick_tool,
                (pick_tool[0], pick_tool[1], pick_tool[2] + .08),
                (place_tool[0], place_tool[1], clear_z), place_tool)):
            return self._empty_run(session, 'recovery_pose_outside_workspace')

        self._attempted = True

        receipts = []
        stages = []
        control_steps = 0

        def save_receipt(receipt, label):
            nonlocal control_steps
            receipts.append(receipt)
            control_steps += int(receipt.executed_steps)
            stages.append({'name': label, 'status': 'executed',
                           'start_step_id': receipt.start_step_id,
                           'end_step_id': receipt.end_step_id,
                           'control_steps': receipt.executed_steps,
                           'action_id': receipt.action_id})
            if on_receipt is not None:
                on_receipt(receipt, label)

        def move(label, point):
            remaining = self.max_control_steps - control_steps
            if remaining <= 0:
                return False, 'control_step_budget_exhausted', session.observe()
            result = self.servo.move(
                session, target_position_m=point,
                target_quaternion_wxyz=quaternion,
                budget_control_steps=remaining, cancel_event=cancel_event)
            for receipt in result.receipts:
                save_receipt(receipt, label)
            stages.append({'name': label, 'status': result.status,
                           'reason': result.reason,
                           'position_error_m': result.position_error_m,
                           'orientation_error_rad': result.orientation_error_rad})
            return result.status == 'succeeded', result.reason, result.final_observation

        current = session.observe()
        current_xy = current.tool_position_m[:2]
        stages.append({'name': 'trigger', 'status': 'accepted',
                       'reason': assessment.reason,
                       'observation_step_ids': assessment.observation_step_ids})
        ok, reason, current = move('clear_above_release',
                                   (current_xy[0], current_xy[1], clear_z))
        if not ok:
            return self._run('failed', reason or 'clearance_failed', receipts,
                             control_steps, current, stages, pick_tool, place_tool)
        ok, reason, current = move('approach_drop_pregrasp',
                                   (pick_tool[0], pick_tool[1], clear_z))
        if not ok:
            return self._run('failed', reason or 'drop_pregrasp_failed', receipts,
                             control_steps, current, stages, pick_tool, place_tool)
        ok, reason, current = move('descend_to_drop', pick_tool)
        if not ok:
            return self._run('failed', reason or 'drop_descent_failed', receipts,
                             control_steps, current, stages, pick_tool, place_tool)

        close_ok, reason, current = self._close_until_contact(
            session, on_receipt, save_receipt, cancel_event,
            self.max_control_steps - control_steps)
        if not close_ok:
            return self._run('failed', reason, receipts, control_steps,
                             current, stages, pick_tool, place_tool)
        lift_point = (pick_tool[0], pick_tool[1], pick_tool[2] + .08)
        ok, reason, current = move('lift_regrasped_block', lift_point)
        if not ok:
            return self._run('failed', reason or 'regrasp_lift_failed', receipts,
                             control_steps, current, stages, pick_tool, place_tool)
        if not (current.gripper.contact_left and current.gripper.contact_right):
            return self._run('failed', 'bilateral_contact_lost_during_lift', receipts,
                             control_steps, current, stages, pick_tool, place_tool)

        carry_high = (place_tool[0], place_tool[1], clear_z)
        ok, reason, current = move('transport_above_stack', carry_high)
        if not ok:
            return self._run('failed', reason or 'transport_failed', receipts,
                             control_steps, current, stages, pick_tool, place_tool)
        ok, reason, current = move('lower_to_stack_target', place_tool)
        if not ok:
            return self._run('failed', reason or 'target_descent_failed', receipts,
                             control_steps, current, stages, pick_tool, place_tool)

        if self.max_control_steps - control_steps < 128:
            return self._run('failed', 'control_step_budget_exhausted', receipts,
                             control_steps, session.observe(), stages, pick_tool,
                             place_tool)
        release_ok, reason, current = self._gripper_action(
            session, 'open', 128, on_receipt, save_receipt, cancel_event)
        if not release_ok:
            return self._run('failed', reason or 'release_failed', receipts,
                             control_steps, current, stages, pick_tool, place_tool)
        if self.max_control_steps - control_steps < 128:
            return self._run('failed', 'control_step_budget_exhausted', receipts,
                             control_steps, session.observe(), stages, pick_tool,
                             place_tool)
        settle_ok, reason, current = self._gripper_action(
            session, None, 128, on_receipt, save_receipt, cancel_event)
        if not settle_ok:
            return self._run('failed', reason or 'settle_failed', receipts,
                             control_steps, current, stages, pick_tool, place_tool)
        return self._run('executed', None, receipts, control_steps, current,
                         stages, pick_tool, place_tool)

    @staticmethod
    def _finite_pair(values):
        try:
            return tuple(float(value) for value in values)
        except (TypeError, ValueError):
            return ()

    def _close_until_contact(self, session, on_receipt, save_receipt, cancel_event,
                             budget_control_steps):
        remaining = budget_control_steps
        while remaining > 0:
            observation = session.observe()
            if observation.gripper.fault:
                return False, 'gripper_fault', observation
            if observation.gripper.contact_left and observation.gripper.contact_right:
                return True, None, observation
            if cancel_event is not None and cancel_event.is_set():
                return False, 'cancelled', observation
            duration = min(128, remaining)
            ok, reason, observation = self._gripper_action(
                session, 'close', duration, on_receipt, save_receipt,
                cancel_event, stage='close_regrasp')
            if not ok:
                return False, reason or 'close_action_failed', observation
            remaining -= duration
            if observation.gripper.contact_left and observation.gripper.contact_right:
                return True, None, observation
        return False, 'bilateral_contact_not_observed', session.observe()

    def _gripper_action(self, session, mode, duration, on_receipt, save_receipt,
                        cancel_event, stage='recovery_gripper'):
        observation = session.observe()
        if cancel_event is not None and cancel_event.is_set():
            return False, 'cancelled', observation
        targets = session.actuator_targets_for(LEFT_ARM_JOINTS)
        spec = session.gripper_spec
        command = ActionCommand(
            observation.episode_id, observation.step_id,
            f'recovery-gripper:{uuid4()}', targets, duration, mode,
            spec.max_opening_m, min(.04, spec.max_speed_m_s), spec.max_force_n)
        try:
            receipt = session.execute(command)
        except (RuntimeError, ValueError) as exc:
            return False, f'action_rejected:{type(exc).__name__}', observation
        save_receipt(receipt, stage)
        if (receipt.interrupted or receipt.fault
                or receipt.executed_steps != duration):
            current = getattr(session, 'last_observation', None) or observation
            return False, receipt.fault or 'partial_execution', current
        current = session.observe()
        if current.gripper.fault:
            return False, 'gripper_fault', current
        return True, None, current

    @staticmethod
    def _empty_run(session, reason):
        return RegraspRecoveryRun('declined', reason, (), 0, session.observe(), ())

    @staticmethod
    def _run(status, reason, receipts, control_steps, observation, stages,
             pick_tool, place_tool):
        return RegraspRecoveryRun(status, reason, tuple(receipts), control_steps,
                                  observation, tuple(stages), tuple(pick_tool),
                                  tuple(place_tool))


@dataclass(frozen=True)
class PreReleaseAlignmentRun:
    status: str
    reason: str
    receipts: tuple
    control_steps: int
    final_observation: object
    target_tool_pose_m: tuple[float, float, float] | None
    position_error_m: float | None
    orientation_error_rad: float | None
    servo_status: str | None
    contact_left: bool | None
    contact_right: bool | None
    contact_loss_hold_requested_steps: int = 0
    contact_loss_hold_completed_steps: int = 0

    def to_dict(self):
        return {
            'status': self.status,
            'reason': self.reason,
            'control_steps': self.control_steps,
            'target_tool_pose_m': self.target_tool_pose_m,
            'position_error_m': self.position_error_m,
            'orientation_error_rad': self.orientation_error_rad,
            'servo_status': self.servo_status,
            'contact_left': self.contact_left,
            'contact_right': self.contact_right,
            'contact_loss_hold_requested_steps': self.contact_loss_hold_requested_steps,
            'contact_loss_hold_completed_steps': self.contact_loss_hold_completed_steps,
        }


class StackBlockRGBXYPreReleaseAlignment:
    """Development-only direct alignment before the intentional release.

    The reset RGB estimate and measured grasp pose define the object-to-tool
    offset. The frozen offset is used to target the configured stack center.
    This policy deliberately remains separate from post-release regrasp and is
    not a real-robot safety controller. Release is authorized only after the
    Cartesian servo reports convergence within its measured-pose tolerances,
    the final observation still reports healthy bilateral contact, and the
    tool pose is inside the task's release envelope.
    """

    POLICY_ID = 'rgbxy_pre_release_align_strict_release_dev_v4'
    MAX_XY_RESIDUAL_M = .020
    MIN_VERTICAL_GAP_M = 0.
    MAX_VERTICAL_GAP_M = .060
    MAX_ORIENTATION_ERROR_RAD = .15

    def __init__(self, localizer, *, max_control_steps=512, servo=None,
                 contact_loss_hold_steps=0):
        if type(max_control_steps) is not int or max_control_steps < 1:
            raise ValueError('Pre-release alignment requires a positive finite budget')
        if (type(contact_loss_hold_steps) is not int
                or not 0 <= contact_loss_hold_steps <= 256):
            raise ValueError('Contact-loss observation hold must be in [0, 256] steps')
        # Reuse the recovery class's strict checkpoint and calibration checks
        # for its robot-observation-only grasp anchor.
        self._anchor_source = StackBlockRGBXYRecovery(localizer)
        self.alignment_max_control_steps = max_control_steps
        self.contact_loss_hold_steps = contact_loss_hold_steps
        self.max_control_steps = max_control_steps + contact_loss_hold_steps
        # The prior 24-iteration default stopped after 288 control steps even
        # though this skill reserves 512 steps. Use the full bounded budget.
        self.servo = servo or ActuatorAwareCartesianServo(max_iterations=42)

    def capture_grasp_anchor(self, initial_observation, grasp_observation):
        return self._anchor_source.capture_grasp_anchor(
            initial_observation, grasp_observation)

    def execute(self, session, anchor, *, budget_control_steps,
                cancel_event=None):
        if type(budget_control_steps) is not int or budget_control_steps < 1:
            raise ValueError('Alignment budget must be a positive integer')
        observation = session.observe()
        if not isinstance(anchor, dict) or anchor.get('model_version') != STACK_BLOCK_RECOVERY_MODEL_VERSION:
            return self._run('declined', 'grasp_anchor_unavailable', (), 0,
                             observation, None, None, None, None)
        try:
            offset = tuple(float(value) for value in anchor['object_to_tool_offset_m'])
            quaternion = tuple(float(value) for value in anchor['grasp_tool_quaternion_wxyz'])
            if (len(offset) != 3 or len(quaternion) != 4
                    or not all(isfinite(value) for value in (*offset, *quaternion))):
                raise ValueError('invalid grasp anchor')
            quaternion = ActuatorAwareCartesianServo._unit_quaternion(quaternion)
        except (KeyError, TypeError, ValueError):
            return self._run('declined', 'grasp_anchor_invalid', (), 0,
                             observation, None, None, None, None)
        gripper = getattr(observation, 'gripper', None)
        if gripper is None:
            return self._run('declined', 'gripper_state_unavailable', (), 0,
                             observation, None, None, None, None)
        if getattr(gripper, 'fault', None):
            return self._run('declined', 'gripper_fault', (), 0,
                             observation, None, None, None, None)
        if (not RGBXYPlacementMissAssessor._finite(getattr(gripper, 'opening_m', None))
                or not RGBXYPlacementMissAssessor._finite(getattr(gripper, 'target_opening_m', None))
                or gripper.opening_m >= .10 or gripper.target_opening_m >= .08):
            return self._run('declined', 'gripper_not_closed_for_alignment', (), 0,
                             observation, None, None, None, None)
        if not (gripper.contact_left and gripper.contact_right):
            return self._run('declined', 'bilateral_contact_missing_before_alignment',
                             (), 0, observation, None, None, None, None)

        target = (
            STACK_BLOCK_TARGET_XY_M[0] - offset[0],
            STACK_BLOCK_TARGET_XY_M[1] - offset[1],
            STACK_BLOCK_STACKED_CENTER_Z_M - offset[2],
        )
        available = min(self.alignment_max_control_steps, budget_control_steps)
        if available < 1:
            return self._run('declined', 'control_step_budget_insufficient', (), 0,
                             observation, target, None, None, None)
        result = self.servo.move(
            session, target_position_m=target,
            target_quaternion_wxyz=quaternion,
            budget_control_steps=available, cancel_event=cancel_event,
            observation_guard=self._alignment_observation_guard)
        final = result.final_observation
        final_gripper = getattr(final, 'gripper', None)
        contacts = (getattr(final_gripper, 'contact_left', None),
                    getattr(final_gripper, 'contact_right', None))
        if result.status == 'cancelled':
            return self._run('cancelled', result.reason or 'cancelled',
                             result.receipts, result.control_steps, final, target,
                             result.position_error_m, result.orientation_error_rad,
                             result.status, *contacts)
        if result.status == 'failed' and isinstance(result.reason, str) and \
                result.reason.startswith('observation_guard:'):
            guard_reason = result.reason.removeprefix('observation_guard:')
            hold_receipts = ()
            hold_steps = 0
            final_observation = final
            if guard_reason == 'bilateral_gripper_contact_lost':
                hold_receipts, hold_steps, final_observation = self._contact_loss_hold(
                    session, max(0, budget_control_steps - result.control_steps), final)
            return self._run('failed', guard_reason, result.receipts,
                             result.control_steps, final, target,
                             result.position_error_m, result.orientation_error_rad,
                             result.status, *contacts,
                             contact_loss_hold_requested_steps=(
                                 self.contact_loss_hold_steps if guard_reason ==
                                 'bilateral_gripper_contact_lost' else 0),
                             contact_loss_hold_completed_steps=hold_steps,
                             extra_receipts=hold_receipts,
                             final_observation_override=final_observation)
        if result.status not in ('succeeded', 'failed') or (
                result.status == 'failed'
                and result.reason != 'convergence_iteration_limit'):
            return self._run('failed', result.reason or 'alignment_servo_failed',
                             result.receipts, result.control_steps, final, target,
                             result.position_error_m, result.orientation_error_rad,
                             result.status, *contacts)
        if final_gripper is None or getattr(final_gripper, 'fault', None):
            reason = 'gripper_state_unavailable' if final_gripper is None else 'gripper_fault'
            return self._run('failed', reason, result.receipts, result.control_steps,
                             final, target, result.position_error_m,
                             result.orientation_error_rad, result.status, *contacts)
        if not (getattr(final_gripper, 'contact_left', False)
                and getattr(final_gripper, 'contact_right', False)):
            return self._run('failed', 'bilateral_gripper_contact_lost',
                             result.receipts, result.control_steps, final, target,
                             result.position_error_m, result.orientation_error_rad,
                             result.status, *contacts)
        if (getattr(final_gripper, 'opening_m', .10) >= .10
                or getattr(final_gripper, 'target_opening_m', .08) >= .08):
            return self._run('failed', 'gripper_opened_before_alignment',
                             result.receipts, result.control_steps, final, target,
                             result.position_error_m, result.orientation_error_rad,
                             result.status, *contacts)
        if result.status != 'succeeded':
            return self._run('failed', 'alignment_not_converged', result.receipts,
                             result.control_steps, final, target,
                             result.position_error_m, result.orientation_error_rad,
                             result.status, *contacts)

        try:
            dx = float(final.tool_position_m[0]) - target[0]
            dy = float(final.tool_position_m[1]) - target[1]
            vertical_gap = float(final.tool_position_m[2]) - target[2]
            orientation_error = ActuatorAwareCartesianServo._norm(
                ActuatorAwareCartesianServo._quaternion_error(
                    quaternion,
                    ActuatorAwareCartesianServo._unit_quaternion(
                        final.tool_quaternion_wxyz)))
        except (AttributeError, IndexError, TypeError, ValueError):
            return self._run('failed', 'measured_pose_invalid', result.receipts,
                             result.control_steps, final, target,
                             result.position_error_m, result.orientation_error_rad,
                             result.status, *contacts)
        measured_position_error = sqrt(dx * dx + dy * dy + vertical_gap * vertical_gap)
        position_tolerance = float(getattr(self.servo, 'position_tolerance_m', .008))
        orientation_tolerance = float(getattr(
            self.servo, 'orientation_tolerance_rad', .08))
        if (not isfinite(position_tolerance) or position_tolerance <= 0
                or not isfinite(orientation_tolerance) or orientation_tolerance <= 0):
            return self._run('failed', 'servo_tolerance_invalid', result.receipts,
                             result.control_steps, final, target,
                             result.position_error_m, orientation_error,
                             result.status, *contacts)
        if measured_position_error > position_tolerance:
            return self._run('failed', 'alignment_position_tolerance_not_met',
                             result.receipts, result.control_steps, final, target,
                             measured_position_error, orientation_error,
                             result.status, *contacts)
        if orientation_error > orientation_tolerance:
            return self._run('failed', 'alignment_orientation_tolerance_not_met',
                             result.receipts, result.control_steps, final, target,
                             measured_position_error, orientation_error,
                             result.status, *contacts)
        inside = (
            abs(dx) <= self.MAX_XY_RESIDUAL_M
            and abs(dy) <= self.MAX_XY_RESIDUAL_M
            and self.MIN_VERTICAL_GAP_M <= vertical_gap <= self.MAX_VERTICAL_GAP_M
            and orientation_error <= self.MAX_ORIENTATION_ERROR_RAD
        )
        return self._run(
            'ready_to_release' if inside else 'failed',
            'within_release_envelope' if inside else 'tool_outside_release_envelope',
            result.receipts, result.control_steps, final, target,
            result.position_error_m, orientation_error, result.status, *contacts)

    @staticmethod
    def _alignment_observation_guard(observation):
        gripper = getattr(observation, 'gripper', None)
        if gripper is None:
            return 'gripper_state_unavailable'
        if getattr(gripper, 'fault', None):
            return 'gripper_fault'
        if not (getattr(gripper, 'contact_left', False)
                and getattr(gripper, 'contact_right', False)):
            return 'bilateral_gripper_contact_lost'
        if (getattr(gripper, 'opening_m', .10) >= .10
                or getattr(gripper, 'target_opening_m', .08) >= .08):
            return 'gripper_opened_during_alignment'
        return None

    def _contact_loss_hold(self, session, budget_control_steps, final_observation):
        """Collect a short fixed-pose observation window without opening the gripper."""
        remaining = min(self.contact_loss_hold_steps, budget_control_steps)
        receipts = []
        while remaining > 0:
            try:
                observation = session.observe()
                gripper = getattr(observation, 'gripper', None)
                if (gripper is None or getattr(gripper, 'fault', None)
                        or getattr(gripper, 'opening_m', .10) >= .10
                        or getattr(gripper, 'target_opening_m', .08) >= .08):
                    break
                targets = session.actuator_targets_for(LEFT_ARM_JOINTS)
                duration = min(32, remaining)
                receipt = session.execute(ActionCommand(
                    episode_id=observation.episode_id,
                    expected_step_id=observation.step_id,
                    action_id=f'contact-loss-hold:{uuid4()}',
                    joint_targets=targets, duration_steps=duration))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                break
            receipts.append(receipt)
            completed = int(getattr(receipt, 'executed_steps', 0))
            remaining -= min(duration, max(0, completed))
            observations = getattr(receipt, 'observations', ())
            if observations:
                final_observation = observations[-1]
            if (getattr(receipt, 'interrupted', False) or getattr(receipt, 'fault', None)
                    or completed != duration):
                break
        completed_steps = sum(int(getattr(receipt, 'executed_steps', 0))
                              for receipt in receipts)
        return tuple(receipts), completed_steps, final_observation

    @staticmethod
    def _run(status, reason, receipts, control_steps, observation, target,
             position_error, orientation_error, servo_status,
             contact_left=None, contact_right=None,
             contact_loss_hold_requested_steps=0,
             contact_loss_hold_completed_steps=0,
             extra_receipts=(), final_observation_override=None):
        return PreReleaseAlignmentRun(
            status, reason, tuple(receipts) + tuple(extra_receipts),
            int(control_steps) + sum(int(getattr(receipt, 'executed_steps', 0))
                                     for receipt in extra_receipts),
            (observation if final_observation_override is None
             else final_observation_override),
            None if target is None else tuple(target), position_error,
            orientation_error, servo_status, contact_left, contact_right,
            contact_loss_hold_requested_steps, contact_loss_hold_completed_steps)
