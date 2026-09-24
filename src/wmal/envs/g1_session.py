"""Persistent MuJoCo session for G1 world-model/Agent experiments."""
from contextlib import AbstractContextManager
import math
import time
from uuid import uuid4

import numpy as np

from wmal.envs.g1_locomotion import (DEFAULT_JOINT_POS, JOINT_NAMES, POLICY_DT,
                                    G1VelocityPolicy, build_g1_model)
from wmal.locomotion.contracts import G1State, G1VelocityAction


class G1MuJoCoSession(AbstractContextManager):
    """Long-lived simulator; task completion never owns or closes this session."""

    def __init__(self, *, viewer=True, realtime=True):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError('Install the simulation extra to use MuJoCo') from exc
        from wmal.envs.g1_locomotion import POLICY_ONNX
        if not POLICY_ONNX.is_file():
            raise FileNotFoundError(f'Pretrained G1 low-level policy is missing: {POLICY_ONNX}')
        self.mj = mujoco
        self.model = build_g1_model()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, 0.8]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.joint_qpos = {}
        for index, name in enumerate(JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_address = self.model.jnt_qposadr[jid]
            self.joint_qpos[name] = qpos_address
            self.data.qpos[qpos_address] = DEFAULT_JOINT_POS[index]
        mujoco.mj_forward(self.model, self.data)
        self.policy = G1VelocityPolicy(self.model, self.data)
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'wmal_ground')
        self.foot_geom_ids = {
            side: {geom_id for geom_id in range(self.model.ngeom)
                   if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or '')
                   .startswith(side + '_foot')}
            for side in ('left', 'right')}
        self.episode_id = str(uuid4())
        self.step_id = 0
        self.realtime = bool(realtime)
        self.closed = False
        self.viewer = None
        self.viewer_context = None
        if viewer:
            try:
                import mujoco.viewer
                self.viewer_context = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer = self.viewer_context.__enter__()
                self.viewer.cam.distance = 3.2
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -18
                self.viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            except (ImportError, RuntimeError) as exc:
                self.close()
                raise RuntimeError('MuJoCo viewer could not be started') from exc
        self.wall_start = time.monotonic()
        self.sim_start = float(self.data.time)
        self.policy.set_velocity(0.0)
        try:
            for _ in range(round(0.4 / POLICY_DT)):
                self._control_interval()
        except Exception:
            self.close()
            raise

    @property
    def is_running(self):
        return (not self.closed and
                (self.viewer is None or self.viewer.is_running()))

    def __enter__(self):
        if self.closed:
            raise RuntimeError('G1 session is closed')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _control_interval(self):
        target = self.policy.infer()
        self.data.ctrl[:] = target
        frame_skip = round(POLICY_DT / self.model.opt.timestep)
        for _ in range(frame_skip):
            self.mj.mj_step(self.model, self.data)
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise RuntimeError('MuJoCo viewer was closed')
            self.viewer.sync()
        if self.realtime:
            target_wall = self.wall_start + (float(self.data.time) - self.sim_start)
            remaining = target_wall - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        state = self.observe()
        if state.pelvis_height < 0.48 or abs(state.roll) > 0.85 or abs(state.pitch) > 0.85:
            raise RuntimeError(f'G1 safety stop at simulation time {self.data.time:.2f}s')

    def _contacts(self):
        touching = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            for side, geom_ids in self.foot_geom_ids.items():
                if ((contact.geom1 == self.floor_id and contact.geom2 in geom_ids)
                        or (contact.geom2 == self.floor_id and contact.geom1 in geom_ids)):
                    touching.append(side)
        return tuple(sorted(set(touching)))

    def observe(self):
        if self.closed:
            raise RuntimeError('G1 session is closed')
        if self.viewer is not None and not self.viewer.is_running():
            raise RuntimeError('MuJoCo viewer was closed')
        quat = self.data.qpos[3:7]
        qw, qx, qy, qz = map(float, quat)
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        sin_pitch = 2 * (qw * qy - qz * qx)
        pitch = math.asin(float(np.clip(sin_pitch, -1.0, 1.0)))
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        spatial_velocity = np.zeros(6, dtype=np.float64)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY,
                                  self.pelvis_id, spatial_velocity, 0)
        positions = {name: float(self.data.qpos[address])
                     for name, address in self.joint_qpos.items()}
        return G1State(
            self.episode_id, self.step_id, float(self.data.time),
            float(self.data.qpos[0]), float(self.data.qpos[1]), float(self.data.qpos[2]),
            yaw, float(spatial_velocity[3]), float(spatial_velocity[4]),
            float(spatial_velocity[5]), float(spatial_velocity[2]), roll, pitch,
            float(self.data.qpos[2]), positions, self._contacts())

    def step(self, action, duration_s=None):
        if self.closed:
            raise RuntimeError('G1 session is closed')
        if not self.is_running:
            raise RuntimeError('MuJoCo viewer is no longer running')
        if not isinstance(action, G1VelocityAction):
            raise ValueError('G1 session requires a validated G1VelocityAction')
        duration = action.duration_s if duration_s is None else float(duration_s)
        if not math.isfinite(duration) or abs(duration - action.duration_s) > 1e-9:
            raise ValueError('Session duration must match the planned action duration')
        intervals = round(duration / POLICY_DT)
        if intervals < 1 or abs(intervals * POLICY_DT - duration) > 1e-9:
            raise ValueError('Action duration must be an exact multiple of the policy period')
        self.policy.set_velocity(action.vx, action.vy, action.yaw_rate)
        for _ in range(intervals):
            self._control_interval()
        # Agent-visible step IDs count completed action chunks, not 20ms policy ticks.
        self.step_id += 1
        return self.observe()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.viewer_context is not None:
            try:
                self.viewer_context.__exit__(None, None, None)
            finally:
                # Allow MuJoCo's passive viewer thread to release its GL context.
                time.sleep(1.0)
                self.viewer_context = None
                self.viewer = None
