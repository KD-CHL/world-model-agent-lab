"""Physics backend for configured robot assets; no synthetic gait or state teleporting."""
from uuid import uuid4
from wmal.communication.contracts import Observation, finite
from wmal.robots.interfaces import robot_interface


class MujocoBackend:
    """All calls must be serialized by the owner node's lock.

    actuator_map: joint -> {actuator, mode: position|pd_torque, kp?, kd?}.
    Supports limited scalar hinge joints with direct unit-gear actuators.
    A locomotion plugin, if configured, owns full ctrl while velocity mode is active.
    """
    def __init__(self, model_path, profile, actuator_map, locomotion=None):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.profile, self.interface = profile, robot_interface(profile)
        self.locomotion = locomotion
        if ('base_velocity' in profile.capabilities) != (locomotion is not None):
            raise ValueError('Velocity capability requires a configured locomotion controller')
        if profile.kind == 'arm' and locomotion is not None:
            raise ValueError('Arm profile does not support locomotion backend')
        if set(actuator_map) != set(profile.joint_limits):
            raise ValueError('Actuator map must cover configured joints exactly')
        self.mapping = {}
        used = set()
        for name, settings in actuator_map.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, settings['actuator'])
            if jid < 0 or aid < 0 or aid in used:
                raise ValueError('Missing or duplicated actuator/joint mapping')
            used.add(aid)
            if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE or not self.model.jnt_limited[jid]:
                raise ValueError('Only limited hinge joints are supported by this adapter')
            if self.model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT or self.model.actuator_trnid[aid, 0] != jid:
                raise ValueError('Actuator does not directly drive configured joint')
            if self.model.actuator_gear[aid, 0] != 1 or any(self.model.actuator_gear[aid, 1:] != 0):
                raise ValueError('Non-unit actuator gear requires a dedicated adapter')
            lo, hi = profile.joint_limits[name]
            actual_lo, actual_hi = self.model.jnt_range[jid]
            if lo < actual_lo - 1e-8 or hi > actual_hi + 1e-8:
                raise ValueError('Profile limits exceed asset joint limits')
            mode = settings['mode']
            if self.model.actuator_dyntype[aid] != mujoco.mjtDyn.mjDYN_NONE or self.model.actuator_gaintype[aid] != mujoco.mjtGain.mjGAIN_FIXED:
                raise ValueError('Stateful/nonlinear actuator needs a dedicated adapter')
            gain = self.model.actuator_gainprm[aid, 0]
            bias = self.model.actuator_biasprm[aid]
            if mode == 'position':
                if gain <= 0 or self.model.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_AFFINE or abs(bias[1] + gain) > 1e-6:
                    raise ValueError('Configured position mode does not match asset servo')
                if self.model.actuator_ctrllimited[aid]:
                    low, high = self.model.actuator_ctrlrange[aid]
                    if lo < low or hi > high:
                        raise ValueError('Profile exceeds servo command range')
            elif mode == 'pd_torque':
                if gain != 1 or self.model.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_NONE:
                    raise ValueError('PD torque mode requires a unit-gain motor')
                if not self.model.actuator_ctrllimited[aid]:
                    raise ValueError('Torque actuator must declare ctrlrange')
                if finite(settings.get('kp')) <= 0 or finite(settings.get('kd')) < 0:
                    raise ValueError('PD gains must be explicitly configured')
            else:
                raise ValueError('Unsupported actuator mode')
            self.mapping[name] = (jid, aid, settings)
        if len(used) != self.model.nu:
            raise ValueError('All model actuators must be mapped; partial actuator ownership is unsupported')
        self.episode_id, self.step_id = str(uuid4()), 0
        self.mode = 'joint_positions'
        mujoco.mj_forward(self.model, self.data)
        self.targets = dict(self.observe().joints)

    def observe(self):
        joints = {name: float(self.data.qpos[self.model.jnt_qposadr[jid]]) for name, (jid, _, _) in self.mapping.items()}
        return Observation(self.profile.robot_id, self.episode_id, self.step_id, float(self.data.time), joints)

    def begin(self, command):
        self.interface.validate_command(command)
        observation = self.observe()
        if command.episode_id != observation.episode_id or command.expected_step != observation.step_id:
            raise ValueError('Command is not bound to current simulation state')
        self.stop()
        self.mode = command.mode
        if command.mode == 'joint_positions':
            self.targets.update(command.values)
        else:
            self.locomotion.set_velocity(**command.values)

    def step(self):
        if self.mode == 'base_velocity':
            self.locomotion.step(self.model, self.data)
        else:
            for name, (jid, aid, settings) in self.mapping.items():
                target = self.targets[name]
                if settings['mode'] == 'position':
                    control = target
                else:
                    q = self.data.qpos[self.model.jnt_qposadr[jid]]
                    dq = self.data.qvel[self.model.jnt_dofadr[jid]]
                    control = settings['kp'] * (target - q) - settings['kd'] * dq
                if self.model.actuator_ctrllimited[aid]:
                    lo, hi = self.model.actuator_ctrlrange[aid]
                    control = min(hi, max(lo, control))
                self.data.ctrl[aid] = finite(control)
        for value in self.data.ctrl:
            finite(float(value))
        self.mj.mj_step(self.model, self.data)
        self.step_id += 1
        return self.observe()

    def stop(self):
        if self.locomotion is not None:
            self.locomotion.stop()
        self.mode = 'joint_positions'
        self.targets = self.observe().joints
        for name, (_, aid, settings) in self.mapping.items():
            self.data.ctrl[aid] = self.targets[name] if settings['mode'] == 'position' else 0.0

    def reset(self):
        self.stop()
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        self.episode_id, self.step_id = str(uuid4()), 0
        self.stop()
        return self.observe()
