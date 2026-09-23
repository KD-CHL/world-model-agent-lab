"""World-model rollout planning. Prediction is injected, never a hidden simulator."""
from typing import Protocol
from uuid import uuid4
import random
from wmal.communication.contracts import MotionCommand, Plan, PredictionReport, finite


class DynamicsModel(Protocol):
    version: str

    def predict(self, joints: dict, targets: dict, duration_s: float) -> dict:
        """Predict joint observations after holding a joint target for duration_s."""
        ...


class RolloutPlanner:
    def __init__(self, model: DynamicsModel, samples=64, horizon=3, duration_s=0.5, seed=0):
        if samples < 2 or horizon < 1 or duration_s <= 0:
            raise ValueError('Invalid rollout budget')
        if not isinstance(model.version, str) or not model.version:
            raise ValueError('Model version required')
        self.model = model
        self.samples, self.horizon, self.duration_s = samples, horizon, duration_s
        self.rng = random.Random(seed)

    def plan(self, profile, observation, goal):
        observation.validate(profile)
        goal.validate(profile)
        if self.duration_s > profile.max_duration_s:
            raise ValueError('Planning action duration exceeds robot limit')
        best = None
        for index in range(self.samples):
            state = dict(observation.joints)
            first = None
            cost = 0.0
            for depth in range(self.horizon):
                # Include hold and direct-target candidates; sample others within declared limits.
                targets = dict(state)
                for joint, desired in goal.targets.items():
                    lo, hi = profile.joint_limits[joint]
                    reachable_lo = max(lo, state[joint] - profile.max_joint_velocity_rad_s * self.duration_s)
                    reachable_hi = min(hi, state[joint] + profile.max_joint_velocity_rad_s * self.duration_s)
                    targets[joint] = state[joint] if index == 0 else min(reachable_hi, max(reachable_lo, desired)) if index == 1 else self.rng.uniform(reachable_lo, reachable_hi)
                targets = {key: min(profile.joint_limits[key][1], max(profile.joint_limits[key][0], value)) for key, value in targets.items()}
                if first is None:
                    first = dict(targets)
                predicted = self.model.predict(dict(state), dict(targets), self.duration_s)
                if not isinstance(predicted, dict) or set(predicted) != set(state):
                    raise ValueError('World model returned incompatible state')
                state = {key: finite(value) for key, value in predicted.items()}
                if any(not profile.joint_limits[k][0] <= v <= profile.joint_limits[k][1] for k, v in state.items()):
                    cost = float('inf')
                    break
                cost += sum((state[key] - value) ** 2 for key, value in goal.targets.items())
            if best is None or cost < best[0]:
                best = (cost, first, state, depth + 1)
        if best is None or best[0] == float('inf'):
            raise ValueError('No admissible predicted trajectory')
        command = MotionCommand(str(uuid4()), profile.robot_id, observation.episode_id,
                                observation.step_id, 'joint_positions', best[1], self.duration_s)
        prediction = PredictionReport(self.model.version, observation.step_id, best[3], best[2], best[0])
        result = Plan(str(uuid4()), self.model.version, [command], prediction)
        result.validate(profile, observation)
        return result
