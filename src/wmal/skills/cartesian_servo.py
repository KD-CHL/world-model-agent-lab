"""Bounded Cartesian motion closed around measured MuJoCo tool feedback."""

from dataclasses import dataclass
from math import atan2, isfinite, sqrt
from uuid import uuid4

from wmal.envs.manipulation_session import ActionCommand, LEFT_ARM_JOINTS


@dataclass(frozen=True)
class CartesianServoResult:
    status: str
    reason: str | None
    receipts: tuple
    control_steps: int
    final_observation: object
    position_error_m: float
    orientation_error_rad: float


class ActuatorAwareCartesianServo:
    """Move to a 6D tool pose with short commands and feedback after each one.

    The Jacobian supplies a local joint update, not an assumed endpoint. Every
    update is sent through the session position actuators and the next update
    is computed from the resulting measured tool pose.
    """

    def __init__(self, *, workspace=((-0.05, .35), (.25, .58), (.76, 1.16)),
                 position_tolerance_m=.008, orientation_tolerance_rad=.08,
                 max_iterations=24, steps_per_iteration=12,
                 max_joint_step_rad=.035, damping=.05):
        if (len(workspace) != 3 or any(len(bounds) != 2 for bounds in workspace)
                or any(not all(isfinite(float(v)) for v in bounds)
                       or float(bounds[0]) >= float(bounds[1]) for bounds in workspace)):
            raise ValueError('Workspace must contain three finite increasing bounds')
        if (not isfinite(position_tolerance_m) or position_tolerance_m <= 0
                or not isfinite(orientation_tolerance_rad) or orientation_tolerance_rad <= 0
                or type(max_iterations) is not int or max_iterations < 1
                or type(steps_per_iteration) is not int or not 1 <= steps_per_iteration <= 128
                or not isfinite(max_joint_step_rad) or not 0 < max_joint_step_rad <= .06
                or not isfinite(damping) or damping <= 0):
            raise ValueError('Invalid Cartesian servo bounds')
        self.workspace = tuple((float(low), float(high)) for low, high in workspace)
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.max_iterations = max_iterations
        self.steps_per_iteration = steps_per_iteration
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.damping = float(damping)

    def move(self, session, *, target_position_m, target_quaternion_wxyz,
             budget_control_steps, cancel_event=None, observation_guard=None):
        if type(budget_control_steps) is not int or budget_control_steps < 1:
            raise ValueError('Cartesian servo budget must be a positive integer')
        if observation_guard is not None and not callable(observation_guard):
            raise ValueError('Observation guard must be callable')
        target_position = self._finite_vector(target_position_m, 3, 'target position')
        target_quaternion = self._unit_quaternion(target_quaternion_wxyz)
        observation = session.observe()
        guard_reason = self._guard_reason(observation_guard, observation)
        if guard_reason:
            return self._result('failed', f'observation_guard:{guard_reason}', (), 0,
                                observation, target_position, target_quaternion)
        if not self._inside_workspace(target_position):
            return self._result('declined', 'target_outside_workspace', (), 0,
                                observation, target_position, target_quaternion)
        if not all(name in observation.joint_names for name in LEFT_ARM_JOINTS):
            return self._result('failed', 'left_arm_observation_incomplete', (), 0,
                                observation, target_position, target_quaternion)
        try:
            limits = session.joint_limits_for(LEFT_ARM_JOINTS)
        except (AttributeError, KeyError, ValueError):
            return self._result('failed', 'joint_limits_unavailable', (), 0,
                                observation, target_position, target_quaternion)
        try:
            actuator_targets = self._actuator_target_map(
                session.actuator_targets_for(LEFT_ARM_JOINTS))
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            return self._result('failed', 'actuator_targets_unavailable', (), 0,
                                observation, target_position, target_quaternion)

        receipts = []
        control_steps = 0
        last_position_error = float('inf')
        last_orientation_error = float('inf')
        for iteration in range(self.max_iterations):
            if cancel_event is not None and cancel_event.is_set():
                return self._result('cancelled', 'cancelled', receipts, control_steps,
                                    observation, target_position, target_quaternion)
            position_error, orientation_error = self._pose_error(
                observation, target_position, target_quaternion)
            last_position_error = self._norm(position_error)
            last_orientation_error = self._norm(orientation_error)
            if (last_position_error <= self.position_tolerance_m
                    and last_orientation_error <= self.orientation_tolerance_rad):
                return self._result('succeeded', None, receipts, control_steps,
                                    observation, target_position, target_quaternion)
            if control_steps >= budget_control_steps:
                return self._result('budget_exhausted', 'control_step_budget_exhausted',
                                    receipts, control_steps, observation,
                                    target_position, target_quaternion)
            if not self._inside_workspace(observation.tool_position_m):
                return self._result('failed', 'measured_pose_outside_workspace', receipts,
                                    control_steps, observation, target_position,
                                    target_quaternion)
            try:
                jacobian = self._jacobian(session.tool_pose_jacobian())
                joint_delta = self._joint_delta(jacobian,
                                                (*position_error, *orientation_error))
            except (AttributeError, ArithmeticError, TypeError, ValueError):
                return self._result('failed', 'jacobian_update_invalid', receipts,
                                    control_steps, observation, target_position,
                                    target_quaternion)
            peak = max(abs(value) for value in joint_delta)
            if peak < 1e-6:
                return self._result('failed', 'joint_update_too_small', receipts,
                                    control_steps, observation, target_position,
                                    target_quaternion)
            scale = min(1., self.max_joint_step_rad / peak)
            try:
                actuator_targets = self._actuator_target_map(
                    session.actuator_targets_for(LEFT_ARM_JOINTS))
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                return self._result('failed', 'actuator_targets_unavailable', receipts,
                                    control_steps, observation, target_position,
                                    target_quaternion)
            targets = {}
            for index, name in enumerate(LEFT_ARM_JOINTS):
                low, high = limits[name]
                requested = actuator_targets[name] + joint_delta[index] * scale
                targets[name] = min(float(high), max(float(low), requested))
            if all(abs(targets[name] - actuator_targets[name]) < 1e-6
                   for name in LEFT_ARM_JOINTS):
                return self._result('failed', 'joint_limits_block_motion', receipts,
                                    control_steps, observation, target_position,
                                    target_quaternion)
            duration = min(self.steps_per_iteration,
                           budget_control_steps - control_steps)
            action = ActionCommand(
                observation.episode_id, observation.step_id,
                f'cartesian-servo:{uuid4()}', targets, duration)
            try:
                receipt = session.execute(action)
            except (RuntimeError, ValueError) as exc:
                return self._result('failed', f'action_rejected:{type(exc).__name__}',
                                    receipts, control_steps, observation,
                                    target_position, target_quaternion)
            receipts.append(receipt)
            control_steps += int(getattr(receipt, 'executed_steps', 0))
            if (getattr(receipt, 'interrupted', False)
                    or getattr(receipt, 'fault', None)
                    or getattr(receipt, 'executed_steps', 0) != duration):
                observation = getattr(session, 'last_observation', None) or observation
                return self._result('failed', getattr(receipt, 'fault', None)
                                    or 'partial_execution', receipts, control_steps,
                                    observation, target_position, target_quaternion)
            try:
                observation = session.observe()
            except (RuntimeError, ValueError):
                observation = getattr(session, 'last_observation', None) or observation
                return self._result('failed', 'post_action_observation_failed',
                                    receipts, control_steps, observation,
                                    target_position, target_quaternion)
            guard_reason = self._guard_reason(observation_guard, observation)
            if guard_reason:
                return self._result(
                    'failed', f'observation_guard:{guard_reason}', receipts,
                    control_steps, observation, target_position, target_quaternion)

        position_error, orientation_error = self._pose_error(
            observation, target_position, target_quaternion)
        last_position_error = self._norm(position_error)
        last_orientation_error = self._norm(orientation_error)
        if (last_position_error <= self.position_tolerance_m
                and last_orientation_error <= self.orientation_tolerance_rad):
            return self._result('succeeded', None, receipts, control_steps,
                                observation, target_position, target_quaternion,
                                last_position_error, last_orientation_error)
        return self._result('failed', 'convergence_iteration_limit', receipts,
                            control_steps, observation, target_position,
                            target_quaternion, last_position_error,
                            last_orientation_error)

    def _result(self, status, reason, receipts, control_steps, observation,
                target_position, target_quaternion, position_error=None,
                orientation_error=None):
        if position_error is None or orientation_error is None:
            try:
                p_error, r_error = self._pose_error(observation, target_position,
                                                    target_quaternion)
                position_error = self._norm(p_error)
                orientation_error = self._norm(r_error)
            except (AttributeError, TypeError, ValueError):
                position_error = orientation_error = float('inf')
        return CartesianServoResult(status, reason, tuple(receipts), control_steps,
                                    observation, float(position_error),
                                    float(orientation_error))

    def _inside_workspace(self, position):
        try:
            return len(position) == 3 and all(
                low <= float(value) <= high
                for value, (low, high) in zip(position, self.workspace))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _guard_reason(guard, observation):
        if guard is None:
            return None
        try:
            reason = guard(observation)
        except Exception:
            return 'guard_evaluation_failed'
        if reason is None or reason is False:
            return None
        if reason is True:
            return 'guard_triggered'
        if isinstance(reason, str) and reason:
            return reason
        return 'guard_result_invalid'

    @staticmethod
    def _finite_vector(values, width, name):
        try:
            result = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be a finite vector') from exc
        if len(result) != width or not all(isfinite(value) for value in result):
            raise ValueError(f'{name} must be a finite vector of width {width}')
        return result

    @classmethod
    def _unit_quaternion(cls, values):
        quaternion = cls._finite_vector(values, 4, 'target quaternion')
        norm = cls._norm(quaternion)
        if norm < 1e-8:
            raise ValueError('Target quaternion must be nonzero')
        return tuple(value / norm for value in quaternion)

    @classmethod
    def _pose_error(cls, observation, target_position, target_quaternion):
        position = cls._finite_vector(observation.tool_position_m, 3, 'measured position')
        quaternion = cls._unit_quaternion(observation.tool_quaternion_wxyz)
        translation = tuple(target_position[i] - position[i] for i in range(3))
        rotation = cls._quaternion_error(target_quaternion, quaternion)
        return translation, rotation

    @classmethod
    def _quaternion_error(cls, target, current):
        w1, x1, y1, z1 = target
        w2, x2, y2, z2 = current
        delta = (w1*w2 + x1*x2 + y1*y2 + z1*z2,
                 -w1*x2 + x1*w2 - y1*z2 + z1*y2,
                 -w1*y2 + x1*z2 + y1*w2 - z1*x2,
                 -w1*z2 - x1*y2 + y1*x2 + z1*w2)
        if delta[0] < 0:
            delta = tuple(-value for value in delta)
        vector_norm = cls._norm(delta[1:])
        if vector_norm < 1e-10:
            return tuple(2. * value for value in delta[1:])
        angle = 2. * atan2(vector_norm, max(0., delta[0]))
        return tuple(value * angle / vector_norm for value in delta[1:])

    def _joint_delta(self, jacobian, pose_error):
        matrix = [[sum(jacobian[row][joint] * jacobian[col][joint]
                       for joint in range(7))
                   + (self.damping ** 2 if row == col else 0.)
                   for col in range(6)] for row in range(6)]
        dual = self._solve(matrix, list(pose_error))
        return tuple(sum(jacobian[row][joint] * dual[row] for row in range(6))
                     for joint in range(7))

    @staticmethod
    def _jacobian(values):
        rows = tuple(tuple(float(value) for value in row) for row in values)
        if (len(rows) != 6 or any(len(row) != 7 for row in rows)
                or any(not isfinite(value) for row in rows for value in row)):
            raise ValueError('Tool pose Jacobian must be finite 6x7')
        return rows

    @staticmethod
    def _actuator_target_map(values):
        if not isinstance(values, dict) or set(values) != set(LEFT_ARM_JOINTS):
            raise ValueError('Position actuator targets must cover the left arm')
        targets = {name: float(values[name]) for name in LEFT_ARM_JOINTS}
        if not all(isfinite(value) for value in targets.values()):
            raise ValueError('Position actuator targets must be finite')
        return targets

    @staticmethod
    def _solve(matrix, vector):
        size = len(vector)
        augmented = [list(matrix[row]) + [float(vector[row])] for row in range(size)]
        for column in range(size):
            pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
            if abs(augmented[pivot][column]) < 1e-12:
                raise ArithmeticError('Damped tool Jacobian is singular')
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            divisor = augmented[column][column]
            augmented[column] = [value / divisor for value in augmented[column]]
            for row in range(size):
                if row == column:
                    continue
                factor = augmented[row][column]
                augmented[row] = [augmented[row][i] - factor * augmented[column][i]
                                  for i in range(size + 1)]
        solution = tuple(augmented[row][-1] for row in range(size))
        if not all(isfinite(value) for value in solution):
            raise ArithmeticError('Damped tool update is nonfinite')
        return solution

    @staticmethod
    def _norm(values):
        return sqrt(sum(float(value) ** 2 for value in values))
