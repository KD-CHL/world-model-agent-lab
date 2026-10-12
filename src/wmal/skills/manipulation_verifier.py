"""Independent stack-block predicates evaluated only from executed scene truth."""
from dataclasses import dataclass
from math import asin, atan2, hypot, isfinite


@dataclass(frozen=True)
class PredicateReport:
    predicate: str
    passed: bool
    episode_id: str
    step_id: int
    sim_time_s: float
    verifier_version: str
    thresholds: tuple
    measurements: tuple
    evidence: tuple
    failure: str | None = None


class StackBlockVerifier:
    VERSION = 'stack-block-verifier.v2'

    def __init__(self, *, target_xy_tolerance_m=.055, lift_height_m=.035,
                 max_linear_speed_m_s=.025, max_angular_speed_rad_s=.15,
                 max_tilt_rad=.20, max_grasp_offset_m=.075,
                 lift_stable_samples=3, stable_samples=8,
                 drop_reset_tolerance_m=.02, drop_stable_samples=2):
        if (target_xy_tolerance_m <= 0 or lift_height_m <= 0 or max_linear_speed_m_s <= 0
                or max_angular_speed_rad_s <= 0 or max_tilt_rad <= 0
                or max_grasp_offset_m <= 0 or type(lift_stable_samples) is not int
                or lift_stable_samples < 1 or type(stable_samples) is not int or stable_samples < 2
                or isinstance(drop_reset_tolerance_m, bool)
                or not isinstance(drop_reset_tolerance_m, (int, float))
                or not isfinite(drop_reset_tolerance_m) or drop_reset_tolerance_m <= 0
                or type(drop_stable_samples) is not int or drop_stable_samples < 1):
            raise ValueError('Invalid verifier thresholds')
        self.target_xy_tolerance_m = float(target_xy_tolerance_m)
        self.lift_height_m = float(lift_height_m)
        self.max_linear_speed_m_s = float(max_linear_speed_m_s)
        self.max_angular_speed_rad_s = float(max_angular_speed_rad_s)
        self.max_tilt_rad = float(max_tilt_rad)
        self.max_grasp_offset_m = float(max_grasp_offset_m)
        self.lift_stable_samples = lift_stable_samples
        self.stable_samples = stable_samples
        self.drop_reset_tolerance_m = float(drop_reset_tolerance_m)
        self.drop_stable_samples = drop_stable_samples
        self.reset()

    def reset(self):
        self.episode_id = None
        self.last_step_id = None
        self.initial_z = None
        self.stable_count = 0
        self.lift_count = 0
        self.lifted = False
        self.grasped = False
        self.placed = False
        self.stacked = False
        self.failed = False
        self.failure_reason = None
        self.drop_count = 0
        self._drop_latched = False

    @staticmethod
    def _has_contact(truth, a, b):
        return any(set(pair) == {a, b} for pair in truth.contacts)

    def update(self, truth, receipt, parameters=None):
        parameters = parameters or {}
        object_name = parameters.get('object_name', 'block_0')
        support_name = parameters.get('support_name', 'block_1')
        if self.episode_id is None:
            self.episode_id = truth.episode_id
            self.last_step_id = receipt.start_step_id
            self.initial_z = truth.object(object_name).position[2]
        if truth.episode_id != self.episode_id or receipt.episode_id != self.episode_id:
            self.reset()
            self.episode_id = truth.episode_id
            self.last_step_id = receipt.start_step_id
            self.initial_z = truth.object(object_name).position[2]
            self.failed = True
            self.failure_reason = 'episode_changed'
            return self._report(truth, 'sequence', False, 'episode_changed',
                                object_name=object_name)
        if (receipt.start_step_id != self.last_step_id or receipt.end_step_id < receipt.start_step_id
                or receipt.executed_steps != receipt.end_step_id - receipt.start_step_id
                or truth.step_id != receipt.end_step_id):
            self.stable_count = 0
            self.failed = True
            self.failure_reason = 'receipt_gap_or_mismatch'
            return self._report(truth, 'sequence', False, 'receipt_gap_or_mismatch',
                                object_name=object_name)
        snapshots = parameters.get('truth_snapshots')
        observations = getattr(receipt, 'observations', ())
        if (receipt.executed_steps > 1 and snapshots is None
                and (len(observations) != receipt.executed_steps or any(
                    obs.step_id != receipt.start_step_id + i + 1
                    for i, obs in enumerate(observations)))):
            self.stable_count = 0
            self.failed = True
            self.failure_reason = 'missing_or_noncontiguous_execution_samples'
            return self._report(truth, 'sequence', False,
                                'missing_or_noncontiguous_execution_samples',
                                object_name=object_name)
        if snapshots is not None:
            if len(snapshots) != receipt.executed_steps or any(
                    item.step_id != receipt.start_step_id + i + 1 for i, item in enumerate(snapshots)):
                self.stable_count = 0
                self.failed = True
                self.failure_reason = 'noncontiguous_truth_samples'
                return self._report(truth, 'sequence', False, 'noncontiguous_truth_samples',
                                    object_name=object_name)
            stream = snapshots
        else:
            stream = (truth,)
        if receipt.executed_steps == 0 or receipt.interrupted or receipt.fault:
            self.stable_count = 0
            self.failed = True
            self.failure_reason = 'incomplete_or_faulted_receipt'
            return self._report(truth, 'execution', False, 'incomplete_or_faulted_receipt',
                                object_name=object_name)

        reports = []
        drop_samples = []
        for sample in stream:
            obj = sample.object(object_name)
            state_values = (*obj.position, *obj.quaternion_wxyz,
                            *obj.linear_velocity, *obj.angular_velocity)
            if not all(isfinite(value) for value in state_values):
                self.failed = True
                self.failure_reason = 'nonfinite_object_state'
                return self._report(sample, 'failed', True, self.failure_reason,
                                    object_name=object_name)
            left_name = parameters.get('left_finger_geom', 'g1_0_stack_gripper_left_finger_geom')
            right_name = parameters.get('right_finger_geom', 'g1_0_stack_gripper_right_finger_geom')
            bilateral = (self._has_contact(sample, object_name + '_geom', left_name)
                         and self._has_contact(sample, object_name + '_geom', right_name))
            offset_ok = (sample.gripper_site_position is not None and
                         sum((obj.position[i]-sample.gripper_site_position[i])**2 for i in range(3))**.5
                         <= self.max_grasp_offset_m)
            self.grasped = self.grasped or (bilateral and offset_ok)
            raised = obj.position[2] >= self.initial_z + self.lift_height_m
            lift_ok = self.grasped and bilateral and offset_ok and raised
            self.lift_count = self.lift_count + 1 if lift_ok else 0
            self.lifted = self.lifted or self.lift_count >= self.lift_stable_samples
            region = next((r for r in sample.target_regions if r[0] == 'stack_target'), None)
            within_target = False
            if region:
                _, center, half = region
                within_target = (abs(obj.position[0]-center[0]) <= min(half[0], self.target_xy_tolerance_m)
                                 and abs(obj.position[1]-center[1]) <= min(half[1], self.target_xy_tolerance_m))
            support = self._has_contact(sample, object_name + '_geom', support_name + '_geom')
            slow = (hypot(*obj.linear_velocity[:2]) <= self.max_linear_speed_m_s
                    and abs(obj.linear_velocity[2]) <= self.max_linear_speed_m_s
                    and sum(v*v for v in obj.angular_velocity)**.5 <= self.max_angular_speed_rad_s)
            w,x,y,z = obj.quaternion_wxyz
            roll = atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
            pitch = asin(max(-1., min(1., 2*(w*y-z*x))))
            tilt = hypot(roll, pitch)
            upright = tilt <= self.max_tilt_rad
            on_target = within_target and support and slow and upright
            self.stable_count = self.stable_count + 1 if on_target else 0
            dropped = (self.lifted and not bilateral and not on_target
                       and obj.position[2] <= self.initial_z + self.drop_reset_tolerance_m)
            self.drop_count = self.drop_count + 1 if dropped else 0
            if self.drop_count >= self.drop_stable_samples and not self._drop_latched:
                self.failed = True
                self.failure_reason = ('object_fell_below_table' if
                                       obj.position[2] < self.initial_z - .05 else
                                       'object_dropped_after_lift')
                self._drop_latched = True
                drop_samples.append((sample, self.failure_reason, self.drop_count))
            if bilateral and raised:
                self._drop_latched = False
            # Placement/stacking are terminal-state predicates, not historical
            # events. A later retreat or collision must revoke success.
            self.placed = self.stable_count >= self.stable_samples
            self.stacked = self.placed
        self.last_step_id = receipt.end_step_id
        reports.extend((self._report(truth, 'grasped', self.grasped,
                                     None if self.grasped else 'bilateral_finger_contact_missing',
                                     object_name=object_name),
                        self._report(truth, 'lifted', self.lifted,
                                     None if self.lifted else 'object_not_raised_after_grasp',
                                     object_name=object_name),
                        self._report(truth, 'placed', self.placed,
                                     None if self.placed else 'stable_support_contact_not_observed',
                                     object_name=object_name),
                        self._report(truth, 'stacked', self.stacked,
                                     None if self.stacked else 'stable_stack_not_observed',
                                     object_name=object_name)))
        reports.extend(self._report(sample, 'failed', True, reason,
                                     drop_count=drop_count, object_name=object_name)
                       for sample, reason, drop_count in drop_samples)
        return tuple(reports)

    def _report(self, truth, predicate, passed, failure, *, drop_count=None,
                object_name='block_0'):
        try:
            obj = truth.object(object_name)
            measurements = (('object_name', object_name),
                            ('object_z_m', obj.position[2]),
                            ('initial_z_m', self.initial_z),
                            ('height_above_reset_m', obj.position[2] - self.initial_z),
                            ('linear_speed_m_s', sum(v*v for v in obj.linear_velocity) ** .5),
                            ('drop_consecutive_samples', (self.drop_count if drop_count is None
                                                          else drop_count)),
                            ('stable_samples', self.stable_count))
            evidence = (('object_name', object_name),
                        ('object_position_m', obj.position),
                        ('object_quaternion_wxyz', obj.quaternion_wxyz),
                        ('object_linear_velocity_m_s', obj.linear_velocity),
                        ('object_angular_velocity_rad_s', obj.angular_velocity),
                        ('contacts', truth.contacts),
                        ('gripper_site_position_m', truth.gripper_site_position))
        except (AttributeError, KeyError):
            measurements = ()
            evidence = ()
        return PredicateReport(predicate, bool(passed), truth.episode_id, truth.step_id,
            truth.sim_time_s, self.VERSION,
            (('target_xy_tolerance_m', self.target_xy_tolerance_m),
             ('lift_height_m', self.lift_height_m),
             ('max_linear_speed_m_s', self.max_linear_speed_m_s),
             ('max_angular_speed_rad_s', self.max_angular_speed_rad_s),
             ('max_tilt_rad', self.max_tilt_rad),
             ('max_grasp_offset_m', self.max_grasp_offset_m),
             ('lift_stable_samples', self.lift_stable_samples),
             ('stable_samples', self.stable_samples),
             ('drop_reset_tolerance_m', self.drop_reset_tolerance_m),
             ('drop_stable_samples', self.drop_stable_samples)), measurements, evidence, failure)
