"""Local fixed-base G1 RGB session for supervised arm-motion experiments.

It exposes two bounded joint target deltas, NOT grasping or humanoid locomotion.
The controller is fixed and never learns alongside the world model.
"""
from uuid import uuid4
import time
import numpy as np

from wmal.agents.predictive_skill_agent import VisualObservation

JOINTS=('left_shoulder_pitch_joint','left_elbow_joint')
TARGET_LOW=np.array([.05,.55])
TARGET_HIGH=np.array([.65,1.15])
MAX_DELTA=.06
ACTION_SCHEMA='wmal.g1.workcell.target_delta.v1'


class VisualWorkcellSession:
    def __init__(self,task='stack_block',*,image_size=64,camera='workcell_overview',period_s=.2,seed=0,
                 frame_publisher=None,realtime=False):
        import mujoco
        from wmal.envs.workcell import build_workcell
        if image_size not in (32,64,128) or not np.isfinite(period_s) or period_s<=0 or period_s>1:
            raise ValueError('Invalid camera/control period')
        self.mj=mujoco
        self.scene,self.model,self.data=build_workcell(task,seed)
        if camera not in self.scene.metadata()['cameras']:
            raise ValueError('Unknown workcell camera')
        self.image_size,self.camera,self.period_s=image_size,camera,period_s
        self.physics_steps=round(period_s/self.model.opt.timestep)
        if abs(self.physics_steps*self.model.opt.timestep-period_s)>1e-8:
            raise ValueError('Control period must align with physics timestep')
        self.qadr=[self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'g1_0_'+joint)]
                   for joint in JOINTS]
        self.aids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,'g1_0_'+joint+'_hold') for joint in JOINTS]
        self._qpos=self.data.qpos.copy()
        self._ctrl=self.data.ctrl.copy()
        self.renderer=mujoco.Renderer(self.model,height=image_size,width=image_size)
        self.viewer=None
        self.frame_publisher=frame_publisher
        self.monitor_renderer=None
        self.realtime=bool(realtime)
        self.fault_reason=None
        self.reset()

    @property
    def semantics(self):
        return {'action_schema':ACTION_SCHEMA,'action_order':list(JOINTS),
                'state_order':list(JOINTS)+[joint+'_target' for joint in JOINTS],
                'camera':self.camera,'period_s':self.period_s,'image_size':self.image_size,'event_names':[]}

    def reset(self,targets=None):
        targets=np.array([.35,.87]) if targets is None else np.asarray(targets,dtype=float)
        if targets.shape!=(2,) or not np.isfinite(targets).all() or np.any(targets<TARGET_LOW) or np.any(targets>TARGET_HIGH):
            raise ValueError('Reset targets outside local joint envelope')
        self.mj.mj_resetData(self.model,self.data)
        self.data.qpos[:]=self._qpos
        self.data.ctrl[:]=self._ctrl
        self.data.qpos[self.qadr]=targets
        self.data.ctrl[self.aids]=targets
        self.mj.mj_forward(self.model,self.data)
        # Settle dynamic objects and servo without generating training transitions.
        self.mj.mj_step(self.model,self.data,nstep=self.physics_steps)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self.abort('reset_failed')
            raise RuntimeError('Nonfinite reset state; session remains locked')
        self.episode_id,self.step_id=str(uuid4()),0
        self.fault_reason=None
        self.publish_monitor_frame()
        return self.observe()

    def publish_monitor_frame(self):
        """Called only by the physics owner; render failure never alters action execution."""
        publisher=self.frame_publisher
        if publisher is None or not publisher.ready():
            return
        try:
            if self.monitor_renderer is None:
                self.monitor_renderer=self.mj.Renderer(self.model,height=publisher.height,width=publisher.width)
            self.monitor_renderer.update_scene(self.data,camera=self.camera)
            publisher.publish(self.monitor_renderer.render(),episode_id=self.episode_id,
                              step_id=self.step_id,sim_time_s=float(self.data.time))
        except Exception as exc:
            publisher.capture_error(exc)

    def observe(self):
        self.renderer.update_scene(self.data,camera=self.camera)
        rgb=self.renderer.render().transpose(2,0,1).astype('float32')/255.
        state=np.concatenate([self.data.qpos[self.qadr],self.data.ctrl[self.aids]]).astype('float32')
        return VisualObservation(self.episode_id,self.step_id,rgb,state)

    def execute_actions(self,actions,*,episode_id,step_id,on_observation=None):
        """Execute committed deltas; optional trace sink receives each actual endpoint.

        A failed sink aborts the session just like a controller failure; do not
        continue executing actions whose real feedback was not acknowledged.
        """
        if self.fault_reason is not None:
            raise RuntimeError('Session is fault-latched; explicit reset required')
        actions=np.asarray(actions,dtype=float)
        if (episode_id!=self.episode_id or step_id!=self.step_id or actions.ndim!=2
                or actions.shape[1]!=2 or not len(actions) or not np.isfinite(actions).all()
                or np.any(np.abs(actions)>MAX_DELTA+1e-8)):
            raise ValueError('Stale episode/step, wrong action semantics or excessive delta')
        targets=self.data.ctrl[self.aids]+np.cumsum(actions,axis=0)
        if np.any(targets<TARGET_LOW-1e-8) or np.any(targets>TARGET_HIGH+1e-8):
            raise ValueError('Full committed prefix violates local joint envelope')
        try:
            for target in targets:
                started=time.monotonic()
                self.data.ctrl[self.aids]=target
                self.mj.mj_step(self.model,self.data,nstep=self.physics_steps)
                if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                    raise RuntimeError('Nonfinite MuJoCo state; abort this session')
                self.step_id+=1
                if on_observation is not None:
                    on_observation(self.observe())
                self.publish_monitor_frame()
                if self.viewer is not None:
                    self.viewer.sync()
                if self.realtime:
                    time.sleep(max(0.,self.period_s-(time.monotonic()-started)))
            return self.observe()
        except (Exception,KeyboardInterrupt):
            self.abort('execution_failed')
            raise

    def abort(self,reason='execution_failed'):
        """Cancel pending servo targets and freeze physics until explicit scene reset."""
        self.fault_reason=reason
        measured=self.data.qpos[self.qadr]
        self.data.ctrl[self.aids]=np.where(np.isfinite(measured),measured,self._ctrl[self.aids])

    def idle(self):
        """A faulted viewer can render, but cannot integrate unacknowledged motion."""
        if self.fault_reason is None:
            try:
                self.mj.mj_step(self.model,self.data,nstep=10)
                if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                    raise RuntimeError('Nonfinite idle state')
            except (Exception,KeyboardInterrupt):
                self.abort('idle_failed')
                raise
        if self.viewer is not None:
            self.viewer.sync()
        self.publish_monitor_frame()

    def open_viewer(self):
        import mujoco.viewer
        if self.viewer is None:
            self.viewer=mujoco.viewer.launch_passive(self.model,self.data)
        return self.viewer

    def close(self):
        if self.frame_publisher is not None:
            self.frame_publisher.close()
        if self.monitor_renderer is not None:
            self.monitor_renderer.close()
            self.monitor_renderer=None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer=None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer=None

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()
