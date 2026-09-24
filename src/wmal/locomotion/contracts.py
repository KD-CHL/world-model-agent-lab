"""Validated contracts for floating-base locomotion experiments."""
from dataclasses import dataclass, field
import math

G1_STATE_SCHEMA = 'wmal.g1.base_state.v1'
G1_ACTION_SCHEMA = 'wmal.g1.body_velocity.v1'
G1_POLICY_PERIOD_S = 0.02


def _finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)


@dataclass(frozen=True)
class G1State:
    episode_id: str
    step_id: int
    sim_time_s: float
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    vz: float
    yaw_rate: float
    roll: float
    pitch: float
    pelvis_height: float
    joint_positions: dict = field(default_factory=dict)
    contacts: tuple = ()

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError('episode_id is required')
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError('step_id must be a nonnegative integer')
        if _finite('sim_time_s', self.sim_time_s) < 0:
            raise ValueError('sim_time_s must be nonnegative')
        for name in ('x', 'y', 'z', 'yaw', 'vx', 'vy', 'vz', 'yaw_rate', 'roll', 'pitch', 'pelvis_height'):
            _finite(name, getattr(self, name))
        if self.pelvis_height < 0 or not isinstance(self.joint_positions, dict):
            raise ValueError('Invalid G1 posture data')
        for name, value in self.joint_positions.items():
            if not isinstance(name, str) or not name:
                raise ValueError('Invalid joint name')
            _finite('joint position', value)
        if not isinstance(self.contacts, tuple) or any(not isinstance(contact, str) for contact in self.contacts):
            raise ValueError('contacts must be a tuple of names')


@dataclass(frozen=True)
class G1Goal:
    x: float
    y: float
    yaw: float | None = None
    position_tolerance_m: float = 0.15
    yaw_tolerance_rad: float = 0.15

    def __post_init__(self):
        for name in ('x', 'y', 'position_tolerance_m', 'yaw_tolerance_rad'):
            _finite(name, getattr(self, name))
        if self.yaw is not None:
            _finite('yaw', self.yaw)
        if self.position_tolerance_m <= 0 or self.yaw_tolerance_rad <= 0:
            raise ValueError('Goal tolerances must be positive')


@dataclass(frozen=True)
class G1VelocityAction:
    vx: float
    vy: float
    yaw_rate: float
    duration_s: float

    def __post_init__(self):
        for name in ('vx', 'vy', 'yaw_rate', 'duration_s'):
            _finite(name, getattr(self, name))
        if (math.hypot(self.vx, self.vy) > 0.5 or abs(self.yaw_rate) > 1.0):
            raise ValueError('Velocity action exceeds policy limit')
        frames = round(self.duration_s / G1_POLICY_PERIOD_S)
        if (not 0.05 <= self.duration_s <= 1.0 or frames < 3
                or abs(frames * G1_POLICY_PERIOD_S - self.duration_s) > 1e-9):
            raise ValueError('Action duration must be a policy-period multiple in [0.05, 1.0] seconds')


@dataclass(frozen=True)
class G1Prediction:
    state: G1State
    uncertainty: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.state, G1State):
            raise ValueError('Prediction must contain a G1State')
        if not isinstance(self.uncertainty, dict):
            raise ValueError('Prediction uncertainty must be a named mapping')
        for name, value in self.uncertainty.items():
            if not isinstance(name, str) or not name or _finite('uncertainty', value) < 0:
                raise ValueError('Prediction uncertainty must be finite and nonnegative')
