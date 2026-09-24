"""Sampling-based, action-conditioned receding-horizon G1 planner."""
from dataclasses import dataclass
import math

import numpy as np

from wmal.locomotion.contracts import (G1_POLICY_PERIOD_S, G1Goal, G1State,
                                       G1VelocityAction)
from wmal.locomotion.world_model import WorldModelAdapter


@dataclass(frozen=True)
class G1Plan:
    model_version: str
    observation_step: int
    action: G1VelocityAction
    predicted_state: G1State
    predicted_terminal_state: G1State
    objective_cost: float


def _wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


class G1RolloutPlanner:
    """Evaluate bounded velocity sequences; execute only the first chunk."""

    def __init__(self, model, samples=48, horizon=3, seed=0, action_duration_s=0.5,
                 max_linear_velocity=0.45, max_yaw_rate=0.7, uncertainty_weight=0.02):
        if type(samples) is not int or samples < 4 or type(horizon) is not int or horizon < 1:
            raise ValueError('Planner requires at least four samples and one horizon step')
        if not 0.05 <= action_duration_s <= 1 or not 0 < max_linear_velocity <= 0.5:
            raise ValueError('Planner limits exceed the G1 policy-supported range')
        frame_count = round(action_duration_s / G1_POLICY_PERIOD_S)
        if frame_count < 3 or abs(frame_count * G1_POLICY_PERIOD_S - action_duration_s) > 1e-9:
            raise ValueError('Action duration must be an exact multiple of the G1 policy period')
        if not 0 < max_yaw_rate <= 1 or uncertainty_weight < 0:
            raise ValueError('Invalid planner bounds')
        self.model = model if isinstance(model, WorldModelAdapter) else WorldModelAdapter(model)
        self.samples, self.horizon, self.seed = samples, horizon, seed
        self.action_duration_s = float(action_duration_s)
        self.max_linear_velocity, self.max_yaw_rate = float(max_linear_velocity), float(max_yaw_rate)
        self.uncertainty_weight = float(uncertainty_weight)

    def _sequences(self, state, goal):
        rng = np.random.default_rng(self.seed + state.step_id)
        sequences = rng.uniform(-1.0, 1.0, size=(self.samples, self.horizon, 3))
        sequences[:, :, 0] *= self.max_linear_velocity
        sequences[:, :, 1] *= min(0.25, self.max_linear_velocity)
        sequences[:, :, 2] *= self.max_yaw_rate
        planar_speed = np.hypot(sequences[:, :, 0], sequences[:, :, 1])
        scale = np.minimum(1.0, self.max_linear_velocity / np.maximum(planar_speed, 1e-12))
        sequences[:, :, 0] *= scale
        sequences[:, :, 1] *= scale

        dx, dy = goal.x - state.x, goal.y - state.y
        distance = math.hypot(dx, dy)
        heading_error = _wrap_angle(math.atan2(dy, dx) - state.yaw) if distance else 0.0
        forward = min(self.max_linear_velocity, distance / max(self.action_duration_s * self.horizon, 1e-6))
        sequences[0, :, :] = (0.0, 0.0, 0.0)
        sequences[1, :, :] = (forward, 0.0,
                              float(np.clip(heading_error, -self.max_yaw_rate, self.max_yaw_rate)))
        if goal.yaw is not None:
            turn = float(np.clip(_wrap_angle(goal.yaw - state.yaw), -self.max_yaw_rate, self.max_yaw_rate))
            sequences[2, :, :] = (0.0, 0.0, turn)
        return sequences

    def _score(self, state, goal, sequence):
        current = state
        first_prediction = None
        total_uncertainty = 0.0
        control_cost = 0.0
        for index, values in enumerate(sequence):
            action = G1VelocityAction(float(values[0]), float(values[1]), float(values[2]),
                                     self.action_duration_s)
            prediction = self.model.predict(current, action, action.duration_s)
            current = prediction.state
            if (current.pelvis_height < 0.48 or abs(current.roll) > 0.65
                    or abs(current.pitch) > 0.65):
                return float('inf'), None, None
            if index == 0:
                first_prediction = current
            total_uncertainty += sum(prediction.uncertainty.values())
            control_cost += action.vx ** 2 + action.vy ** 2 + 0.3 * action.yaw_rate ** 2
        position_error = math.hypot(current.x - goal.x, current.y - goal.y)
        yaw_error = 0.0 if goal.yaw is None else abs(_wrap_angle(goal.yaw - current.yaw))
        unstable = max(0.0, abs(current.roll) - 0.35) ** 2 + max(0.0, abs(current.pitch) - 0.35) ** 2
        low_height = max(0.0, 0.58 - current.pelvis_height)
        cost = (position_error + 0.4 * yaw_error + 5.0 * unstable + 10.0 * low_height
                + 0.04 * control_cost + self.uncertainty_weight * total_uncertainty)
        return cost, first_prediction, current

    def plan(self, state, goal):
        if not isinstance(state, G1State) or not isinstance(goal, G1Goal):
            raise ValueError('G1 planner requires G1State and G1Goal')
        best = None
        for sequence in self._sequences(state, goal):
            result = self._score(state, goal, sequence)
            if best is None or result[0] < best[0]:
                best = result
                best_action = G1VelocityAction(*map(float, sequence[0]), self.action_duration_s)
        if best is None or not math.isfinite(best[0]):
            raise ValueError('World model predicts no safe candidate sequence')
        return G1Plan(self.model.version, state.step_id, best_action,
                      best[1], best[2], float(best[0]))
