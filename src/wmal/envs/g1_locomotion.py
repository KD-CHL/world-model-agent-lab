"""Unitree G1 velocity-policy playback on the matching Unitree RL Mjlab model."""
from pathlib import Path

import numpy as np

from wmal.locomotion.contracts import G1_POLICY_PERIOD_S

ROOT = Path(__file__).resolve().parents[3]
MODEL_XML = ROOT / 'robots/assets/unitree_g1_mjlab/xmls/g1.xml'
POLICY_ONNX = ROOT / 'models/unitree_g1_velocity_policy.onnx'
POLICY_DT = G1_POLICY_PERIOD_S
GAIT_PERIOD_S = 0.6

JOINT_NAMES = (
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint',
    'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
    'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
)

DEFAULT_JOINT_POS = np.asarray((
    -0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0,
    0, 0, 0, 0.35, 0.18, 0, 0.87, 0, 0, 0,
    0.35, -0.18, 0, 0.87, 0, 0, 0,
), dtype=np.float64)

ACTION_SCALE = np.asarray((
    0.55, 0.35, 0.55, 0.35, 0.44, 0.44, 0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
    0.55, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,
    0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,
), dtype=np.float32)

KP = np.asarray((
    40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
    40.2, 28.5, 28.5, 14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
    14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
), dtype=np.float64)

KD = np.asarray((
    2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
    2.6, 1.8, 1.8, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
    0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
), dtype=np.float64)

EFFORT_LIMIT = np.asarray((
    88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50,
    88, 50, 50, 25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5,
), dtype=np.float64)

ARMATURE = np.asarray((
    0.0101775200413, 0.025101925, 0.0101775200413, 0.025101925,
    0.00721945, 0.00721945, 0.0101775200413, 0.025101925, 0.0101775200413,
    0.025101925, 0.00721945, 0.00721945, 0.0101775200413, 0.00721945,
    0.00721945, 0.003609725, 0.003609725, 0.003609725, 0.003609725,
    0.003609725, 0.00017, 0.00017, 0.003609725, 0.003609725, 0.003609725,
    0.003609725, 0.003609725, 0.00017, 0.00017,
), dtype=np.float64)


def build_g1_model(xml_path=MODEL_XML):
    """Compile a floating-base model with official policy gains and contact shapes."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(name='wmal_ground', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], friction=[0.8, 0.01, 0.001])
    for index, name in enumerate(JOINT_NAMES):
        joint = spec.joint(name)
        if joint is None:
            raise ValueError('G1 policy joint is missing from model: ' + name)
        joint.armature = float(ARMATURE[index])
        actuator = spec.add_actuator(name=name.removesuffix('_joint'),
                                     trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                     target=name)
        actuator.set_to_position(float(KP[index]), float(KD[index]))
        actuator.forcelimited = True
        actuator.forcerange = [-float(EFFORT_LIMIT[index]), float(EFFORT_LIMIT[index])]
    model = spec.compile()
    if (model.nq, model.nv, model.nu, model.njnt) != (36, 35, 29, 30):
        raise ValueError('Unexpected G1 policy model dimensions')
    return model


class G1VelocityPolicy:
    """ONNX policy adapter matching Unitree's published G1 deployment schema."""
    def __init__(self, model, data, policy_path=POLICY_ONNX):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('Install the walking extra: python -m pip install -e ".[walking]"') from exc
        self.mj = __import__('mujoco')
        self.model, self.data = model, data
        self.session = ort.InferenceSession(str(policy_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.joint_qpos = np.asarray([
            model.jnt_qposadr[self._joint_id(name)] for name in JOINT_NAMES
        ], dtype=int)
        self.joint_dof = np.asarray([
            model.jnt_dofadr[self._joint_id(name)] for name in JOINT_NAMES
        ], dtype=int)
        self.pelvis_id = self.mj.mj_name2id(model, self.mj.mjtObj.mjOBJ_BODY, 'pelvis')
        self.command = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.phase = 0.0
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 2 or shape[-1] not in (98, '98', None):
            raise ValueError(f'Unexpected G1 policy input shape: {shape}')

    def _joint_id(self, name):
        jid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError('G1 policy joint is missing from model: ' + name)
        return jid

    def set_velocity(self, vx, vy=0.0, yaw_rate=0.0):
        command = np.asarray([vx, vy, yaw_rate], dtype=np.float32)
        if not np.isfinite(command).all() or abs(vx) > 1.0 or abs(vy) > 0.5 or abs(yaw_rate) > 1.0:
            raise ValueError('G1 velocity command is outside the policy training range')
        self.command[:] = command

    def infer(self):
        angular_velocity = np.zeros(6, dtype=np.float64)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY,
                                  self.pelvis_id, angular_velocity, 1)
        quat = self.data.qpos[3:7].copy()
        quat[1:] *= -1
        gravity_body = np.zeros(3, dtype=np.float64)
        self.mj.mju_rotVecQuat(gravity_body, np.asarray([0.0, 0.0, -1.0]), quat)
        command_norm = float(np.linalg.norm(self.command))
        phase = np.asarray([np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)], dtype=np.float32)
        if command_norm < 0.1:
            phase[:] = 0
        observation = np.concatenate((
            angular_velocity[:3], gravity_body, self.command, phase,
            self.data.qpos[self.joint_qpos] - DEFAULT_JOINT_POS,
            self.data.qvel[self.joint_dof], self.last_action,
        )).astype(np.float32, copy=False)
        action = self.session.run([self.output_name], {self.input_name: observation[None, :]})[0][0]
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError('G1 ONNX policy returned invalid action')
        self.last_action[:] = action
        self.phase = (self.phase + POLICY_DT / GAIT_PERIOD_S) % 1.0
        return DEFAULT_JOINT_POS + ACTION_SCALE * action


def simulate_g1_walk(steps=10, velocity_x=0.2, max_duration_s=12.0, viewer=False):
    """Walk until the requested alternating foot touchdown count, or fail safely."""
    import mujoco

    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError('steps must be an integer in [1, 100]')
    if not 0.05 <= float(velocity_x) <= 0.5:
        raise ValueError('velocity_x must be in [0.05, 0.5] m/s for a forward walk')
    if not np.isfinite(max_duration_s) or not 1.0 <= float(max_duration_s) <= 60.0:
        raise ValueError('max_duration_s must be in [1, 60] simulated seconds')
    if not POLICY_ONNX.is_file():
        raise FileNotFoundError(f'Pretrained Unitree G1 policy is missing: {POLICY_ONNX}')

    model = build_g1_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for index, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = DEFAULT_JOINT_POS[index]
    mujoco.mj_forward(model, data)
    policy = G1VelocityPolicy(model, data)

    if viewer:
        import time
        try:
            import mujoco.viewer
        except ImportError as exc:
            raise RuntimeError('MuJoCo viewer is unavailable in this environment') from exc
        try:
            with mujoco.viewer.launch_passive(model, data) as window:
                window.cam.distance = 3.2
                window.cam.azimuth = 135
                window.cam.elevation = -18
                window.cam.lookat[:] = [0.0, 0.0, 0.8]
                result = _run_g1_walk(model, data, policy, steps, velocity_x,
                                      max_duration_s, window)
        finally:
            # MuJoCo 3.x's passive viewer owns a daemon render thread; give it
            # time to destroy its GL context before Python tears down GLFW.
            time.sleep(1.0)
        return result
    return _run_g1_walk(model, data, policy, steps, velocity_x, max_duration_s)


def _run_g1_walk(model, data, policy, steps, velocity_x, max_duration_s, viewer=None):
    import time
    import mujoco

    frame_skip = round(POLICY_DT / model.opt.timestep)
    if abs(frame_skip * model.opt.timestep - POLICY_DT) > 1e-9:
        raise ValueError('MuJoCo timestep must divide the policy interval exactly')

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'wmal_ground')
    foot_geoms = {
        side: {
            geom_id for geom_id in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or '').startswith(side + '_foot')
        }
        for side in ('left', 'right')
    }

    def contacts():
        touching = {'left': False, 'right': False}
        for index in range(data.ncon):
            contact = data.contact[index]
            for side, ids in foot_geoms.items():
                if ((contact.geom1 == floor_id and contact.geom2 in ids)
                        or (contact.geom2 == floor_id and contact.geom1 in ids)):
                    touching[side] = True
        return touching

    wall_start = time.monotonic()
    sim_start = float(data.time)

    def control_interval():
        target = policy.infer()
        data.ctrl[:] = target
        for _ in range(frame_skip):
            mujoco.mj_step(model, data)
        if viewer is not None:
            if not viewer.is_running():
                raise RuntimeError('MuJoCo viewer was closed before the walk completed')
            viewer.sync()
            deadline = wall_start + (float(data.time) - sim_start)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        height = float(data.qpos[2])
        quat = data.qpos[3:7].copy()
        quat[1:] *= -1
        gravity_body = np.zeros(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(gravity_body, np.asarray([0.0, 0.0, -1.0]), quat)
        tilt = float(np.arccos(np.clip(-gravity_body[2], -1.0, 1.0)))
        if height < 0.48 or tilt > 0.85:
            raise RuntimeError(f'Safety stop: G1 lost upright posture at t={data.time:.2f}s')
        return tilt

    # Let the published zero-command policy settle into its standing posture.
    policy.set_velocity(0.0)
    for _ in range(round(0.4 / POLICY_DT)):
        control_interval()
    last_contacts = contacts()
    policy.phase = 0.0
    policy.set_velocity(float(velocity_x))
    accepted_side = None
    last_touchdown_time = {'left': -1e6, 'right': -1e6}
    touchdowns = 0
    initial_x = float(data.qpos[0])
    max_end_time = float(data.time) + float(max_duration_s)
    while touchdowns < steps and data.time < max_end_time:
        tilt = control_interval()
        current_contacts = contacts()
        for side in ('left', 'right'):
            rising = current_contacts[side] and not last_contacts[side]
            if (rising and side != accepted_side
                    and data.time - last_touchdown_time[side] >= 0.18):
                touchdowns += 1
                accepted_side = side
                last_touchdown_time[side] = float(data.time)
                if touchdowns >= steps:
                    break
        last_contacts = current_contacts

    if touchdowns < steps:
        raise RuntimeError(
            f'G1 did not complete {steps} measured footsteps within {max_duration_s:.1f}s; '
            f'completed {touchdowns}'
        )

    policy.set_velocity(0.0)
    for _ in range(round(0.6 / POLICY_DT)):
        tilt = control_interval()
    return {
        'status': 'succeeded',
        'requested_footsteps': steps,
        'footsteps': touchdowns,
        'forward_displacement_m': float(data.qpos[0] - initial_x),
        'base_height_m': float(data.qpos[2]),
        'tilt_rad': tilt,
        'simulated_time_s': float(data.time),
        'command_velocity_x_m_s': float(velocity_x),
        'model': str(MODEL_XML.relative_to(ROOT)),
        'policy': str(POLICY_ONNX.relative_to(ROOT)),
    }
