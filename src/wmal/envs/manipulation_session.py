"""Single-owner MuJoCo session for truth-separated stack-block experiments."""
from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
from uuid import uuid4

import numpy as np

from wmal.envs.g1_locomotion import JOINT_NAMES
from wmal.robots.gripper import GripperCommandReceipt, GripperSpec, MuJoCoParallelGripper


ACTION_SCHEMA = 'wmal.manipulation.action.v1'
LEFT_ARM_JOINTS = (
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
)
OBJECT_NAMES = ('block_0', 'block_1', 'block_2')


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name} must be a finite real number')
    value = float(value)
    if not isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    return value


def _readonly_array(value, dtype=np.float32):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RobotObservation:
    episode_id: str
    step_id: int
    sim_time_s: float
    joint_names: tuple[str, ...]
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    tool_position_m: tuple[float, float, float]
    tool_quaternion_wxyz: tuple[float, float, float, float]
    gripper: object
    rgb: np.ndarray | None

    def __post_init__(self):
        if not self.episode_id or type(self.step_id) is not int or self.step_id < 0:
            raise ValueError('Invalid observation identity')
        object.__setattr__(self, 'sim_time_s', _finite(self.sim_time_s, 'sim_time_s'))
        if not (len(self.joint_names) == len(self.joint_position) == len(self.joint_velocity)):
            raise ValueError('Joint observation vectors do not align')
        if not all(isfinite(float(v)) for v in (*self.joint_position, *self.joint_velocity)):
            raise ValueError('Joint observation contains nonfinite values')
        if len(self.tool_position_m) != 3 or len(self.tool_quaternion_wxyz) != 4 or not all(
                isfinite(float(v)) for v in (*self.tool_position_m, *self.tool_quaternion_wxyz)):
            raise ValueError('Tool pose must contain finite position and quaternion values')
        if self.rgb is not None:
            image = _readonly_array(self.rgb)
            if image.ndim != 3 or image.shape[0] != 3 or not np.isfinite(image).all():
                raise ValueError('RGB observation must be finite CHW data')
            object.__setattr__(self, 'rgb', image)

    def to_agent_dict(self):
        """Serialize only robot-visible measurements; never scene truth or reset metadata."""
        return {
            'schema': 'wmal.robot_observation.v1', 'episode_id': self.episode_id,
            'step_id': self.step_id, 'sim_time_s': self.sim_time_s,
            'joint_names': list(self.joint_names), 'joint_position': list(self.joint_position),
            'joint_velocity': list(self.joint_velocity),
            'tool_position_m': list(self.tool_position_m),
            'tool_quaternion_wxyz': list(self.tool_quaternion_wxyz),
            'gripper': {
                'opening_m': self.gripper.opening_m,
                'target_opening_m': self.gripper.target_opening_m,
                'contact_left': self.gripper.contact_left,
                'contact_right': self.gripper.contact_right,
                'at_target': self.gripper.at_target,
                'fault': self.gripper.fault,
            },
            'rgb_shape': None if self.rgb is None else list(self.rgb.shape),
        }


@dataclass(frozen=True)
class ObjectTruth:
    name: str
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]


@dataclass(frozen=True)
class SceneTruth:
    episode_id: str
    step_id: int
    sim_time_s: float
    object_states: tuple[ObjectTruth, ...]
    contacts: tuple[tuple[str, str], ...]
    target_regions: tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...]
    gripper_site_position: tuple[float, float, float] | None = None

    def object(self, name):
        for item in self.object_states:
            if item.name == name:
                return item
        raise KeyError(name)


@dataclass(frozen=True)
class ResetAudit:
    episode_id: str
    reset_seed: int
    initial_state_profile: str
    sim_time_s: float
    object_states: tuple[ObjectTruth, ...]
    joint_positions: tuple[tuple[str, float], ...]
    gripper_opening_m: float
    model_signature: str


@dataclass(frozen=True)
class ActionCommand:
    episode_id: str
    expected_step_id: int
    action_id: str
    joint_targets: object
    duration_steps: int = 1
    gripper_mode: str | None = None
    gripper_opening_m: float = .074
    gripper_speed_m_s: float = .04
    gripper_force_limit_n: float = 8.0
    schema: str = ACTION_SCHEMA

    def __post_init__(self):
        if self.schema != ACTION_SCHEMA or not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError('Invalid action schema or episode')
        if type(self.expected_step_id) is not int or self.expected_step_id < 0:
            raise ValueError('expected_step_id must be a nonnegative integer')
        if not isinstance(self.action_id, str) or not self.action_id or len(self.action_id) > 128:
            raise ValueError('action_id must be a nonempty bounded string')
        if type(self.duration_steps) is not int or not 1 <= self.duration_steps <= 128:
            raise ValueError('duration_steps must be an integer in [1, 128]')
        if self.gripper_mode not in (None, 'open', 'close', 'hold'):
            raise ValueError('Unknown gripper mode')
        values = self.joint_targets.items() if isinstance(self.joint_targets, dict) else self.joint_targets
        try:
            pairs = tuple((name, _finite(target, f'joint target {name}')) for name, target in values)
        except (TypeError, ValueError) as exc:
            raise ValueError('joint_targets must contain finite named pairs') from exc
        if not pairs or len({name for name, _ in pairs}) != len(pairs):
            raise ValueError('joint_targets must contain unique named targets')
        if any(not isinstance(name, str) or not name for name, _ in pairs):
            raise ValueError('Joint names must be nonempty strings')
        object.__setattr__(self, 'joint_targets', tuple(sorted(pairs)))
        object.__setattr__(self, 'gripper_opening_m', _finite(self.gripper_opening_m, 'gripper_opening_m'))
        object.__setattr__(self, 'gripper_speed_m_s', _finite(self.gripper_speed_m_s, 'gripper_speed_m_s'))
        object.__setattr__(self, 'gripper_force_limit_n', _finite(self.gripper_force_limit_n, 'gripper_force_limit_n'))


@dataclass(frozen=True)
class ExecutionReceipt:
    action_id: str
    episode_id: str
    start_step_id: int
    end_step_id: int
    executed_steps: int
    sim_duration_s: float
    observations: tuple[RobotObservation, ...]
    limit_events: tuple[str, ...]
    interrupted: bool
    fault: str | None
    joint_targets: tuple[tuple[str, float], ...] = ()
    gripper_mode: str | None = None
    gripper_opening_m: float = .074
    gripper_speed_m_s: float = .04
    gripper_force_limit_n: float = 8.0
    gripper_command_receipt: GripperCommandReceipt | None = None


class ManipulationSession:
    """Owns physics, control, observations, and reset; skills never receive MjData."""

    def __init__(self, *, task='stack_block', image_size=64, camera='workcell_overview',
                 control_period_s=.2, max_episode_steps=2000, max_joint_delta_rad=.06,
                 render=True, reset_hooks=()):
        import mujoco
        from wmal.envs.workcell import build_workcell

        if task != 'stack_block':
            raise ValueError('ManipulationSession currently supports stack_block only')
        if type(image_size) is not int or not 16 <= image_size <= 512:
            raise ValueError('image_size must be an integer in [16, 512]')
        self._mj = mujoco
        self._scene, self._model, self._data = build_workcell(task, seed=0)
        if camera not in self._scene.metadata()['cameras']:
            raise ValueError('Unknown workcell camera')
        self._camera, self._image_size, self._render_enabled = camera, image_size, bool(render)
        self._control_period_s = _finite(control_period_s, 'control_period_s')
        if self._control_period_s <= 0 or self._control_period_s > 1:
            raise ValueError('Invalid control period')
        self._physics_steps = round(self._control_period_s / self._model.opt.timestep)
        if abs(self._physics_steps * self._model.opt.timestep - self._control_period_s) > 1e-8:
            raise ValueError('Control period must align with MuJoCo physics timestep')
        if type(max_episode_steps) is not int or not 1 <= max_episode_steps <= 100000:
            raise ValueError('Invalid episode action budget')
        self._max_episode_steps = max_episode_steps
        self._max_joint_delta = _finite(max_joint_delta_rad, 'max_joint_delta_rad')
        if not 0 < self._max_joint_delta <= .25:
            raise ValueError('Invalid joint slew limit')
        self._reset_hooks = tuple(reset_hooks)
        self._joint_names = tuple(JOINT_NAMES)
        self._joint_ids = {}
        self._actuator_ids = {}
        self._joint_limits = {}
        for name in self._joint_names:
            jid = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, 'g1_0_' + name)
            aid = self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, 'g1_0_' + name + '_hold')
            self._joint_ids[name] = jid
            self._actuator_ids[name] = aid
            self._joint_limits[name] = tuple(float(v) for v in self._model.jnt_range[jid])
        if any(name not in self._joint_ids for name in LEFT_ARM_JOINTS):
            raise ValueError('Model is missing required left-arm joints')
        self._object_joint_ids = {
            name: self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name + '_free') for name in OBJECT_NAMES
        }
        self._gripper_spec = GripperSpec.from_workcell_metadata(self._scene.metadata()['gripper'])
        self._gripper_site_id = self._required_id(
            mujoco.mjtObj.mjOBJ_SITE, 'g1_0_stack_gripper_center')
        self._gripper = MuJoCoParallelGripper(
            self._model, self._gripper_spec,
            object_geom_names=tuple(name + '_geom' for name in OBJECT_NAMES))
        self._renderer = None
        self._baseline_qpos = self._data.qpos.copy()
        self._baseline_ctrl = self._data.ctrl.copy()
        self._step_id = 0
        self._episode_id = None
        self._reset_seed = 0
        self._profile = 'nominal'
        self._used_steps = 0
        self._action_ids = set()
        self._trace = []
        self._truth_snapshots = {}
        self._fault_reason = None
        self.reset(0, 'nominal')

    def _required_id(self, obj_type, name):
        item_id = self._mj.mj_name2id(self._model, obj_type, name)
        if item_id < 0:
            raise ValueError(f'Missing required MuJoCo object: {name}')
        return int(item_id)

    @property
    def sim_time_s(self):
        return float(self._data.time)

    @property
    def fault_reason(self):
        return self._fault_reason

    @property
    def trace(self):
        return tuple(self._trace)

    @property
    def episode_id(self):
        return self._episode_id

    def reset(self, seed, initial_state_profile='nominal'):
        if type(seed) is not int or seed < 0:
            raise ValueError('reset seed must be a nonnegative integer')
        if initial_state_profile not in ('nominal', 'jittered', 'disturbed'):
            raise ValueError('Unknown initial_state_profile')
        self._mj.mj_resetData(self._model, self._data)
        self._data.qpos[:] = self._baseline_qpos
        self._data.ctrl[:] = self._baseline_ctrl
        rng = np.random.default_rng(seed)
        if initial_state_profile in ('jittered', 'disturbed'):
            # Perturb the manipulated object while keeping the support fixture
            # and distractor fixed; otherwise the fixed target waypoint would
            # silently become an invalid target for the episode.
            jid = self._object_joint_ids['block_0']
            qadr = int(self._model.jnt_qposadr[jid])
            xy_range = (.012, .015) if initial_state_profile == 'jittered' else (.03, .04)
            self._data.qpos[qadr] += float(rng.uniform(-xy_range[0], xy_range[0]))
            self._data.qpos[qadr + 1] += float(rng.uniform(-xy_range[1], xy_range[1]))
            if initial_state_profile == 'disturbed':
                yaw = float(rng.uniform(-.20, .20))
                self._data.qpos[qadr + 3:qadr + 7] = (np.cos(yaw/2), 0., 0., np.sin(yaw/2))
        for jid in self._object_joint_ids.values():
            dadr = int(self._model.jnt_dofadr[jid])
            self._data.qvel[dadr:dadr + 6] = 0.
        self._gripper.reset(self._data)
        self._mj.mj_forward(self._model, self._data)
        # Settle physics without counting these setup steps as task transitions.
        for _ in range(self._physics_steps):
            self._mj.mj_step(self._model, self._data)
        self._data.time = 0.
        self._mj.mj_forward(self._model, self._data)
        if not np.isfinite(self._data.qpos).all() or not np.isfinite(self._data.qvel).all():
            self._fault_reason = 'reset_nonfinite_state'
            raise RuntimeError('Nonfinite reset state')
        self._step_id, self._used_steps = 0, 0
        self._episode_id, self._reset_seed, self._profile = str(uuid4()), seed, initial_state_profile
        self._action_ids.clear()
        self._trace.clear()
        self._truth_snapshots.clear()
        self._fault_reason = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        for hook in self._reset_hooks:
            hook.reset()
        observation = self.observe()
        truth = self.truth_for_verifier()
        self._truth_snapshots[0] = truth
        audit = ResetAudit(
            episode_id=self._episode_id, reset_seed=seed,
            initial_state_profile=initial_state_profile, sim_time_s=float(self._data.time),
            object_states=truth.object_states,
            joint_positions=tuple((name, float(self._data.qpos[self._model.jnt_qposadr[self._joint_ids[name]]]))
                                  for name in self._joint_names),
            gripper_opening_m=observation.gripper.opening_m,
            model_signature=self._model_signature(),
        )
        return observation, audit

    def _model_signature(self):
        digest = sha256()
        digest.update(repr((self._scene.task, self._control_period_s,
                            self._scene.metadata()['worktable_center'], self._gripper_spec)).encode('utf-8'))
        for array in (self._model.qpos0, self._model.jnt_range, self._model.geom_pos,
                      self._model.geom_size, self._model.geom_friction,
                      self._model.geom_contype, self._model.geom_conaffinity,
                      self._model.actuator_ctrlrange, self._model.actuator_forcerange):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(repr((self._model.nq, self._model.nv, self._model.nu,
                            self._model.ngeom, self._model.opt.timestep)).encode('ascii'))
        return digest.hexdigest()

    def observe(self):
        if self._episode_id is None:
            raise RuntimeError('Session must be reset before observation')
        rgb = None
        if self._render_enabled:
            if self._renderer is None:
                self._renderer = self._mj.Renderer(
                    self._model, height=self._image_size, width=self._image_size)
            self._renderer.update_scene(self._data, camera=self._camera)
            rgb = self._renderer.render().transpose(2, 0, 1).astype(np.float32) / 255.
        positions = tuple(float(self._data.qpos[self._model.jnt_qposadr[jid]])
                          for jid in self._joint_ids.values())
        velocities = tuple(float(self._data.qvel[self._model.jnt_dofadr[jid]])
                           for jid in self._joint_ids.values())
        site_position = tuple(float(x) for x in self._data.site_xpos[self._gripper_site_id])
        site_quaternion = np.empty(4, dtype=float)
        self._mj.mju_mat2Quat(site_quaternion,
                              self._data.site_xmat[self._gripper_site_id].reshape(9))
        return RobotObservation(
            episode_id=self._episode_id, step_id=self._step_id, sim_time_s=float(self._data.time),
            joint_names=self._joint_names, joint_position=positions, joint_velocity=velocities,
            tool_position_m=site_position,
            tool_quaternion_wxyz=tuple(float(x) for x in site_quaternion),
            gripper=self._gripper.observe(self._data), rgb=rgb)

    def truth_for_verifier(self):
        if self._episode_id is None:
            raise RuntimeError('Session must be reset before reading truth')
        objects = []
        velocity = np.zeros(6, dtype=float)
        for name in OBJECT_NAMES:
            body_id = self._required_id(self._mj.mjtObj.mjOBJ_BODY, name)
            self._mj.mj_objectVelocity(
                self._model, self._data, self._mj.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
            body = self._data.body(body_id)
            objects.append(ObjectTruth(
                name=name, position=tuple(float(x) for x in body.xpos),
                quaternion_wxyz=tuple(float(x) for x in body.xquat),
                linear_velocity=tuple(float(x) for x in velocity[3:]),
                angular_velocity=tuple(float(x) for x in velocity[:3])))
        contacts = []
        for index in range(self._data.ncon):
            contact = self._data.contact[index]
            names = (self._mj.mj_id2name(self._model, self._mj.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                     self._mj.mj_id2name(self._model, self._mj.mjtObj.mjOBJ_GEOM, int(contact.geom2)))
            if any(name in (obj + '_geom' for obj in OBJECT_NAMES) for name in names):
                contacts.append(tuple(str(name or '') for name in names))
        regions = tuple((item['name'], tuple(float(x) for x in item['center']),
                         tuple(float(x) for x in item['half_size'])) for item in self._scene.regions)
        site_id = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_SITE,
                                     'g1_0_stack_gripper_center')
        gripper_site = (tuple(float(x) for x in self._data.site_xpos[site_id])
                        if site_id >= 0 else None)
        return SceneTruth(self._episode_id, self._step_id, float(self._data.time),
                          tuple(objects), tuple(contacts), regions, gripper_site)

    def truth_trace_for_verifier(self, receipt):
        """Read truth snapshots for a completed receipt through a verifier-only boundary."""
        if not isinstance(receipt, ExecutionReceipt) or not any(item is receipt for item in self._trace):
            raise ValueError('Receipt does not belong to this session')
        return tuple(self._truth_snapshots[step]
                     for step in range(receipt.start_step_id + 1, receipt.end_step_id + 1))

    def execute(self, command):
        if self._fault_reason:
            raise RuntimeError('Session is fault-latched; explicit reset is required')
        if not isinstance(command, ActionCommand):
            raise TypeError('command must be an ActionCommand')
        if command.episode_id != self._episode_id or command.expected_step_id != self._step_id:
            raise ValueError('Stale episode or observation step')
        if command.action_id in self._action_ids:
            raise ValueError('Duplicate action_id')
        if self._used_steps + command.duration_steps > self._max_episode_steps:
            raise ValueError('Action exceeds remaining episode budget')
        targets = dict(command.joint_targets)
        for name, target in targets.items():
            if name not in LEFT_ARM_JOINTS:
                raise ValueError(f'Joint is outside manipulation allowlist: {name}')
            lower, upper = self._joint_limits[name]
            if target < lower or target > upper:
                raise ValueError(f'Joint target outside model limits: {name}')
        gripper_command_receipt = None
        if command.gripper_mode is not None:
            gripper_command_receipt = self._gripper.command(
                self._data, command.gripper_mode,
                opening_m=command.gripper_opening_m,
                speed_m_s=command.gripper_speed_m_s,
                force_limit_n=command.gripper_force_limit_n,
                command_id=command.action_id)
        self._action_ids.add(command.action_id)
        start_step, start_time = self._step_id, float(self._data.time)
        observations, limits = [], []
        fault = None
        for _ in range(command.duration_steps):
            if self._fault_reason:
                fault = self._fault_reason
                break
            for name, target in targets.items():
                aid = self._actuator_ids[name]
                current = float(self._data.ctrl[aid])
                delta = target - current
                limited = float(np.clip(delta, -self._max_joint_delta, self._max_joint_delta))
                if abs(delta - limited) > 1e-12:
                    limits.append(f'{name}:slew_limited')
                self._data.ctrl[aid] = current + limited
            self._gripper.advance(self._data, self._control_period_s)
            try:
                for _physics_step in range(self._physics_steps):
                    self._mj.mj_step(self._model, self._data)
                self._step_id += 1
                self._used_steps += 1
            except Exception:
                self._fault_reason = 'physics_step_failed'
                fault = self._fault_reason
                break
            try:
                observation = self.observe()
                observations.append(observation)
                self._truth_snapshots[self._step_id] = self.truth_for_verifier()
            except Exception:
                self._fault_reason = 'post_action_observation_failed'
                fault = self._fault_reason
                break
        if gripper_command_receipt is not None:
            fully_executed = (self._step_id - start_step == command.duration_steps
                              and fault is None)
            gripper_command_receipt = replace(
                gripper_command_receipt, executed=fully_executed, fault=fault)
        receipt = ExecutionReceipt(
            action_id=command.action_id, episode_id=self._episode_id,
            start_step_id=start_step, end_step_id=self._step_id,
            executed_steps=self._step_id - start_step,
            sim_duration_s=float(self._data.time) - start_time,
            observations=tuple(observations), limit_events=tuple(limits),
            interrupted=fault is not None, fault=fault,
            joint_targets=command.joint_targets,
            gripper_mode=command.gripper_mode,
            gripper_opening_m=command.gripper_opening_m,
            gripper_speed_m_s=command.gripper_speed_m_s,
            gripper_force_limit_n=command.gripper_force_limit_n,
            gripper_command_receipt=gripper_command_receipt)
        self._trace.append(receipt)
        return receipt

    def abort(self, reason):
        if not isinstance(reason, str) or not reason:
            raise ValueError('Abort reason is required')
        self._fault_reason = reason

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
