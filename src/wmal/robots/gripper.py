"""Bounded contracts and MuJoCo adapter for the stack-block gripper proxy."""
from dataclasses import dataclass
from math import isfinite
from uuid import uuid4


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    return float(value)


@dataclass(frozen=True)
class GripperSpec:
    name: str
    asset_kind: str
    is_dex1: bool
    left_joint: str
    right_joint: str
    left_actuator: str
    right_actuator: str
    left_geom: str
    right_geom: str
    min_opening_m: float
    max_opening_m: float
    max_speed_m_s: float
    max_force_n: float
    finger_max_slide_m: float

    @classmethod
    def from_workcell_metadata(cls, value):
        if not isinstance(value, dict):
            raise ValueError('Workcell has no gripper capability')
        try:
            return cls(
                name=value['name_prefix'], asset_kind=value['asset_kind'],
                is_dex1=value['is_dex1'], left_joint=value['left_joint'],
                right_joint=value['right_joint'], left_actuator=value['left_actuator'],
                right_actuator=value['right_actuator'], left_geom=value['left_geom'],
                right_geom=value['right_geom'], min_opening_m=value['min_opening_m'],
                max_opening_m=value['max_opening_m'], max_speed_m_s=value['max_speed_m_s'],
                max_force_n=value['max_force_n'], finger_max_slide_m=value['finger_max_slide_m'])
        except (KeyError, TypeError) as exc:
            raise ValueError('Incomplete gripper metadata') from exc

    def __post_init__(self):
        if (not self.name or self.asset_kind != 'synthetic_parallel_jaw_proxy'
                or type(self.is_dex1) is not bool or self.is_dex1):
            raise ValueError('The local gripper must be explicitly identified as a synthetic proxy')
        for name in ('left_joint', 'right_joint', 'left_actuator', 'right_actuator', 'left_geom', 'right_geom'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'{name} is required')
        for name in ('min_opening_m', 'max_opening_m', 'max_speed_m_s', 'max_force_n', 'finger_max_slide_m'):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        if (self.min_opening_m < 0 or self.max_opening_m <= self.min_opening_m
                or self.max_speed_m_s <= 0 or self.max_force_n <= 0 or self.finger_max_slide_m <= 0):
            raise ValueError('Invalid gripper limits')


@dataclass(frozen=True)
class GripperState:
    opening_m: float
    target_opening_m: float
    contact_left: bool
    contact_right: bool
    at_target: bool
    fault: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'opening_m', _finite_real(self.opening_m, 'opening_m'))
        object.__setattr__(self, 'target_opening_m', _finite_real(self.target_opening_m, 'target_opening_m'))
        if any(type(getattr(self, name)) is not bool for name in ('contact_left', 'contact_right', 'at_target')):
            raise ValueError('Gripper state flags must be booleans')
        if self.fault is not None and not isinstance(self.fault, str):
            raise ValueError('fault must be a string or None')


@dataclass(frozen=True)
class GripperCommandReceipt:
    command_id: str
    requested_mode: str
    requested_opening_m: float
    accepted: bool


class MuJoCoParallelGripper:
    """Moves two opposed slide fingers; contact feedback is measured from MuJoCo."""

    MODES = frozenset(('open', 'close', 'hold'))

    def __init__(self, model, spec, *, object_geom_names):
        if not isinstance(spec, GripperSpec):
            raise TypeError('spec must be a GripperSpec')
        import mujoco

        self.model, self.spec, self.mj = model, spec, mujoco
        self.joint_ids = tuple(self._required_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in
                               (spec.left_joint, spec.right_joint))
        self.qpos_addresses = tuple(int(model.jnt_qposadr[i]) for i in self.joint_ids)
        self.actuator_ids = tuple(self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in
                                  (spec.left_actuator, spec.right_actuator))
        self.finger_geom_ids = tuple(self._required_id(mujoco.mjtObj.mjOBJ_GEOM, n) for n in
                                     (spec.left_geom, spec.right_geom))
        names = tuple(object_geom_names)
        if not names or len(set(names)) != len(names):
            raise ValueError('At least one unique object geometry name is required')
        self.object_geom_ids = frozenset(self._required_id(mujoco.mjtObj.mjOBJ_GEOM, n) for n in names)
        self._target_opening = spec.max_opening_m
        self._commanded_opening = spec.max_opening_m
        self._speed = spec.max_speed_m_s
        self._fault = None
        self._counter = 0
        self._last_command_id = None

    def _required_id(self, obj_type, name):
        item_id = self.mj.mj_name2id(self.model, obj_type, name)
        if item_id < 0:
            raise ValueError(f'Missing MuJoCo object: {name}')
        return int(item_id)

    def reset(self, data):
        self._fault = None
        self._target_opening = self.spec.max_opening_m
        self._commanded_opening = self.spec.max_opening_m
        self._speed = self.spec.max_speed_m_s
        self._counter = 0
        self._last_command_id = None
        slide = (self._target_opening - self.spec.min_opening_m) / 2.
        for address, actuator_id in zip(self.qpos_addresses, self.actuator_ids):
            data.qpos[address] = slide
            data.ctrl[actuator_id] = slide
            self.model.actuator_forcerange[actuator_id] = [-self.spec.max_force_n, self.spec.max_force_n]
        self.mj.mj_forward(self.model, data)
        return self.observe(data)

    def command(self, data, mode, *, opening_m, speed_m_s, force_limit_n):
        if mode not in self.MODES:
            raise ValueError('Unknown gripper mode')
        requested = _finite_real(opening_m, 'opening_m')
        speed = _finite_real(speed_m_s, 'speed_m_s')
        force = _finite_real(force_limit_n, 'force_limit_n')
        if speed <= 0 or speed > self.spec.max_speed_m_s:
            raise ValueError('speed_m_s exceeds gripper capability')
        if force <= 0 or force > self.spec.max_force_n:
            raise ValueError('force_limit_n exceeds gripper capability')
        if mode == 'open':
            target = requested
            if target < self.spec.min_opening_m or target > self.spec.max_opening_m:
                raise ValueError('opening_m outside gripper range')
        elif mode == 'close':
            target = self.spec.min_opening_m
        else:
            target = self.observe(data).opening_m
        self._commanded_opening = self.observe(data).opening_m
        self._target_opening, self._speed = target, speed
        self._counter += 1
        self._last_command_id = str(uuid4())
        for actuator_id in self.actuator_ids:
            self.model.actuator_forcerange[actuator_id] = [-force, force]
        return GripperCommandReceipt(self._last_command_id, mode, target, True)

    def advance(self, data, period_s):
        period = _finite_real(period_s, 'period_s')
        if period <= 0:
            raise ValueError('period_s must be positive')
        delta = self._speed * period
        if self._target_opening > self._commanded_opening:
            next_opening = min(self._target_opening, self._commanded_opening + delta)
        else:
            # Ramp the commanded jaw position, not the measured aperture. If
            # the object blocks the fingers, a measured-position ramp stalls
            # at first contact and never develops a holding force.
            next_opening = max(self._target_opening, self._commanded_opening - delta)
        self._commanded_opening = next_opening
        slide = (next_opening - self.spec.min_opening_m) / 2.
        for actuator_id in self.actuator_ids:
            data.ctrl[actuator_id] = slide

    def observe(self, data):
        opening = self.spec.min_opening_m + sum(float(data.qpos[i]) for i in self.qpos_addresses)
        left = right = False
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = (int(contact.geom1), int(contact.geom2))
            if pair[0] in self.object_geom_ids:
                other = pair[1]
            elif pair[1] in self.object_geom_ids:
                other = pair[0]
            else:
                continue
            if other == self.finger_geom_ids[0]:
                left = True
            elif other == self.finger_geom_ids[1]:
                right = True
        return GripperState(
            opening_m=opening, target_opening_m=self._target_opening,
            contact_left=left, contact_right=right,
            at_target=abs(opening - self._target_opening) <= .002, fault=self._fault)
