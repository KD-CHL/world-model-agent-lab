"""Explicit WMA action-vector to named, bounded joint-command mapping."""
from dataclasses import dataclass
from uuid import uuid4

from wmal.communication.contracts import MotionCommand, finite


@dataclass(frozen=True)
class ActionMapping:
    action_order: tuple[str, ...]
    joint_order: tuple[str, ...]
    action_semantics: str
    units: str
    normalization: str
    duration_s: float
    source_min: tuple[float, ...] = ()
    source_max: tuple[float, ...] = ()

    @classmethod
    def from_config(cls, data, profile):
        if not isinstance(data, dict):
            raise ValueError("WMA action mapping must be an object")
        action_order, joint_order = data.get("action_order"), data.get("joint_order")
        if (not isinstance(action_order, list) or not action_order
                or not all(isinstance(item, str) and item for item in action_order)
                or len(set(action_order)) != len(action_order)):
            raise ValueError("action_order must contain unique action feature names")
        if (not isinstance(joint_order, list) or len(joint_order) != len(action_order)
                or not all(isinstance(item, str) and item for item in joint_order)
                or len(set(joint_order)) != len(joint_order)):
            raise ValueError("joint_order must explicitly map every action dimension")
        if not set(joint_order).issubset(profile.joint_limits):
            raise ValueError("Mapped joints are not present in robot profile")
        semantics, units = data.get("action_semantics"), data.get("units")
        normalization = data.get("normalization")
        if semantics != "joint_position" or units != "rad":
            raise ValueError("Only explicit absolute joint-position actions in radians are supported")
        if normalization not in ("identity", "min_max"):
            raise ValueError("normalization must be identity or min_max")
        low, high = tuple(data.get("source_min", ())), tuple(data.get("source_max", ()))
        if normalization == "min_max":
            if len(low) != len(action_order) or len(high) != len(action_order):
                raise ValueError("min_max normalization requires source_min/source_max per dimension")
            for lo, hi in zip(low, high):
                if finite(lo) >= finite(hi):
                    raise ValueError("Invalid action normalization range")
        elif low or high:
            raise ValueError("Identity normalization must not provide source bounds")
        duration = finite(data.get("duration_s"))
        if duration > profile.max_duration_s:
            raise ValueError("Action duration exceeds robot profile")
        return cls(tuple(action_order), tuple(joint_order), semantics, units, normalization,
                   duration, tuple(float(v) for v in low), tuple(float(v) for v in high))

    def command(self, action, observation, profile):
        observation.validate(profile)
        if not isinstance(action, (list, tuple)) or len(action) != len(self.action_order):
            raise ValueError("Model action width does not match configured order")
        values = [finite(item) for item in action]
        if self.normalization == "min_max":
            values = [lo + (value + 1.0) * 0.5 * (hi - lo)
                      for value, lo, hi in zip(values, self.source_min, self.source_max)]
        targets = dict(zip(self.joint_order, values))
        profile.validate_targets(targets)
        command = MotionCommand(str(uuid4()), profile.robot_id, observation.episode_id,
                                observation.step_id, "joint_positions", targets, self.duration_s)
        command.validate(profile)
        if max(abs(targets[joint] - observation.joints[joint]) / self.duration_s
               for joint in targets) > profile.max_joint_velocity_rad_s:
            raise ValueError("Mapped action exceeds speed limit from current observation")
        return command
