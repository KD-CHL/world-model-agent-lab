"""Validated, versioned wire contracts independent of ROS and model providers."""
from dataclasses import asdict, dataclass, field
import json
import math
import re


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite number')
    return float(value)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', value):
        raise ValueError('Invalid robot identifier')
    return value


def vector(values):
    if not isinstance(values, dict) or not values:
        raise ValueError('Expected nonempty named vector')
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Invalid joint name')
        finite(value)


@dataclass
class RobotProfile:
    robot_id: str
    kind: str
    joint_limits: dict
    capabilities: list = field(default_factory=lambda: ['joint_positions'])
    max_duration_s: float = 5.0
    max_linear_velocity: float = 0.3
    max_yaw_rate: float = 0.5

    def __post_init__(self):
        identifier(self.robot_id)
        if self.kind not in ('arm', 'go2', 'g1') or not self.joint_limits:
            raise ValueError('Unknown robot kind or missing joint limits')
        for name, bounds in self.joint_limits.items():
            if not isinstance(name, str) or not name or len(bounds) != 2 or finite(bounds[0]) >= finite(bounds[1]):
                raise ValueError('Invalid joint limits')
        if not isinstance(self.capabilities, list) or set(self.capabilities) - {'joint_positions', 'base_velocity'}:
            raise ValueError('Unknown capability')
        for value in (self.max_duration_s, self.max_linear_velocity, self.max_yaw_rate):
            if finite(value) <= 0:
                raise ValueError('Limits must be positive')

    def validate_targets(self, targets):
        vector(targets)
        for name, value in targets.items():
            if name not in self.joint_limits:
                raise ValueError('Unknown joint: ' + name)
            lower, upper = self.joint_limits[name]
            if not lower <= value <= upper:
                raise ValueError('Joint outside limits: ' + name)


@dataclass
class Observation:
    robot_id: str
    episode_id: str
    step_id: int
    sim_time_s: float
    joints: dict

    def __post_init__(self):
        identifier(self.robot_id)
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError('Missing episode')
        if type(self.step_id) is not int or self.step_id < 0 or finite(self.sim_time_s) < 0:
            raise ValueError('Invalid observation time')
        vector(self.joints)

    def validate(self, profile):
        if self.robot_id != profile.robot_id or set(self.joints) != set(profile.joint_limits):
            raise ValueError('Observation does not match robot profile')


@dataclass
class Goal:
    intent: str
    targets: dict

    def validate(self, profile):
        if self.intent != 'joint_goal':
            raise ValueError('Unsupported goal intent')
        profile.validate_targets(self.targets)


@dataclass
class MotionCommand:
    command_id: str
    robot_id: str
    episode_id: str
    expected_step: int
    mode: str
    values: dict
    duration_s: float

    def validate(self, profile):
        if not isinstance(self.command_id, str) or not self.command_id or not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError('Missing command/episode identifier')
        if self.robot_id != profile.robot_id or self.mode not in profile.capabilities:
            raise ValueError('Robot or capability mismatch')
        if type(self.expected_step) is not int or self.expected_step < 0:
            raise ValueError('Invalid expected step')
        if not 0 < finite(self.duration_s) <= profile.max_duration_s:
            raise ValueError('Invalid command duration')
        if self.mode == 'joint_positions':
            profile.validate_targets(self.values)
        elif self.mode == 'base_velocity':
            vector(self.values)
            if set(self.values) != {'vx', 'vy', 'yaw_rate'}:
                raise ValueError('Velocity requires vx, vy, yaw_rate')
            if math.hypot(self.values['vx'], self.values['vy']) > profile.max_linear_velocity or abs(self.values['yaw_rate']) > profile.max_yaw_rate:
                raise ValueError('Velocity exceeds configured limit')
        else:
            raise ValueError('Unknown command mode')


@dataclass
class Plan:
    plan_id: str
    model_version: str
    commands: list

    def validate(self, profile, observation):
        observation.validate(profile)
        if not isinstance(self.plan_id, str) or not self.plan_id or not isinstance(self.model_version, str) or not self.model_version:
            raise ValueError('Missing plan or model version')
        if not isinstance(self.commands, list) or len(self.commands) != 1:
            raise ValueError('Receding-horizon interface requires exactly one executable command')
        for cmd in self.commands:
            cmd.validate(profile)
            if cmd.episode_id != observation.episode_id or cmd.expected_step != observation.step_id:
                raise ValueError('Plan bound to stale observation')


@dataclass
class ExecutionResult:
    command_id: str
    status: str
    detail: str = ''

    def __post_init__(self):
        if self.status not in ('succeeded', 'failed', 'canceled', 'rejected', 'timeout'):
            raise ValueError('Invalid execution status')


def encode(value):
    payload = asdict(value) if hasattr(value, '__dataclass_fields__') else value
    return json.dumps({'schema_version': 1, 'payload': payload}, allow_nan=False)


def decode(text, cls):
    if not isinstance(text, str) or len(text) > 1_000_000:
        raise ValueError('Invalid message size')
    obj = json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(obj, dict) or obj.get('schema_version') != 1 or set(obj) != {'schema_version', 'payload'}:
        raise ValueError('Unsupported wire schema')
    payload = obj['payload']
    if cls is dict:
        if not isinstance(payload, dict):
            raise ValueError('Expected object')
        return payload
    if not isinstance(payload, dict):
        raise ValueError('Expected object payload')
    if cls is Plan:
        payload = dict(payload)
        payload['commands'] = [MotionCommand(**item) for item in payload['commands']]
    return cls(**payload)
