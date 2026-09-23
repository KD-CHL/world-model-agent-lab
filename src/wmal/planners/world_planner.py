"""Receding-horizon planning through the project's action-conditioned model."""
from uuid import uuid4
import numpy as np

from wmal.communication.contracts import MotionCommand, Plan, PredictionReport, finite
from wmal.planners.sequence_optimizer import SequenceOptimizer


class RolloutPlanner:
    """Search action chunks, roll them through model members, execute only step one."""

    def __init__(self, model, samples=128, horizon=4, duration_s=0.5, seed=0,
                 iterations=4, elites=12, uncertainty_weight=0.05, action_weight=0.001):
        if samples < 2 or horizon < 1 or duration_s <= 0 or uncertainty_weight < 0 or action_weight < 0:
            raise ValueError('Invalid planning budget')
        if not isinstance(getattr(model, 'version', None), str) or not model.version:
            raise ValueError('Model version required')
        self.model = model
        self.samples, self.horizon, self.duration_s = samples, horizon, duration_s
        self.uncertainty_weight, self.action_weight = uncertainty_weight, action_weight
        self.optimizer = SequenceOptimizer(candidates=samples, elites=min(elites, samples - 1),
                                           iterations=iterations, seed=seed)

    def _rollout(self, sequence, state, goal, profile):
        if hasattr(self.model, 'ensemble_size'):
            member_count = int(self.model.ensemble_size)
        elif hasattr(self.model, 'weights'):
            member_count = len(self.model.weights)
        else:
            member_count = 1
        trajectories = []
        for member_index in range(member_count):
            current = dict(state)
            first_state = None
            total = 0.0
            for action_index, proposed in enumerate(sequence):
                command = {}
                for index, joint in enumerate(profile.joint_limits):
                    low, high = profile.joint_limits[joint]
                    max_delta = profile.max_joint_velocity_rad_s * self.duration_s
                    command[joint] = min(high, max(low, min(current[joint] + max_delta,
                                                              max(current[joint] - max_delta,
                                                                  float(proposed[index])))))
                if hasattr(self.model, 'predict_member'):
                    predicted = self.model.predict_member(member_index, current, command, self.duration_s)
                elif hasattr(self.model, 'members') and hasattr(self.model, 'weights'):
                    predicted = self.model.members(current, command, self.duration_s)[member_index]
                else:
                    predicted = self.model.predict(current, command, self.duration_s)
                if not isinstance(predicted, dict) or set(predicted) != set(current):
                    raise ValueError('World model returned incompatible state')
                current = {key: finite(value) for key, value in predicted.items()}
                if action_index == 0:
                    first_state = dict(current)
                if any(not profile.joint_limits[key][0] <= value <= profile.joint_limits[key][1]
                       for key, value in current.items()):
                    return None
                total += sum((current[key] - value) ** 2 for key, value in goal.targets.items())
                total += self.action_weight * sum((command[key] - state[key]) ** 2 for key in command)
            trajectories.append((first_state, current, total))
        return trajectories

    def plan(self, profile, observation, goal):
        observation.validate(profile)
        goal.validate(profile)
        if self.duration_s > profile.max_duration_s:
            raise ValueError('Planning action duration exceeds robot limit')
        joints = tuple(profile.joint_limits)
        initial = np.tile(np.asarray([observation.joints[j] for j in joints]), (self.horizon, 1))
        max_delta = profile.max_joint_velocity_rad_s * self.duration_s
        for index, joint in enumerate(joints):
            if joint in goal.targets:
                desired = goal.targets[joint]
                initial[:, index] = np.linspace(observation.joints[joint], desired, self.horizon + 1)[1:]
        lower, upper = [], []
        for joint in joints:
            if joint not in goal.targets:
                lower.append(observation.joints[joint])
                upper.append(observation.joints[joint])
                continue
            lo, hi = profile.joint_limits[joint]
            lower.append(max(lo, observation.joints[joint] - max_delta * self.horizon))
            upper.append(min(hi, observation.joints[joint] + max_delta * self.horizon))
        lower, upper = np.asarray(lower), np.asarray(upper)

        def score(sequence):
            rollouts = self._rollout(sequence, observation.joints, goal, profile)
            if rollouts is None:
                return float('inf')
            costs = np.asarray([result[2] for result in rollouts], dtype=float)
            return float(costs.mean() + self.uncertainty_weight * costs.std())

        sequence, _, _ = self.optimizer.optimize(score, horizon=self.horizon, width=len(joints),
                                                  lower=lower, upper=upper, initial=initial)
        rollouts = self._rollout(sequence, observation.joints, goal, profile)
        if rollouts is None:
            raise ValueError('Selected trajectory became inadmissible')
        next_states = [result[0] for result in rollouts]
        final_states = [result[1] for result in rollouts]
        predicted_state = {joint: float(np.mean([state[joint] for state in next_states])) for joint in joints}
        terminal_state = {joint: float(np.mean([state[joint] for state in final_states])) for joint in joints}
        uncertainty = {joint: float(np.std([state[joint] for state in next_states])) for joint in joints}
        first_target = {}
        for index, joint in enumerate(joints):
            lo, hi = profile.joint_limits[joint]
            reach = profile.max_joint_velocity_rad_s * self.duration_s
            first_target[joint] = min(hi, max(lo, min(observation.joints[joint] + reach,
                                                       max(observation.joints[joint] - reach,
                                                           float(sequence[0, index])))))
        predicted_cost = float(np.mean([result[2] for result in rollouts]))
        command = MotionCommand(str(uuid4()), profile.robot_id, observation.episode_id,
                                observation.step_id, 'joint_positions', first_target, self.duration_s)
        prediction = PredictionReport(self.model.version, observation.step_id, self.horizon,
                                      predicted_state, predicted_cost,
                                      uncertainty_kind='ensemble_spread' if len(rollouts) > 1 else 'unavailable',
                                      uncertainty=uncertainty if len(rollouts) > 1 else {},
                                      predicted_terminal_state=terminal_state)
        result = Plan(str(uuid4()), self.model.version, [command], prediction)
        result.validate(profile, observation)
        return result
