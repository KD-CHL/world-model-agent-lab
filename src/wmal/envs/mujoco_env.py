"""Synchronous MuJoCo facade for training, evaluation and visual observations."""
from uuid import uuid4
import math

from wmal.communication.contracts import MotionCommand


class MujocoEnvironment:
    """Owns action hold duration while the backend owns all physics mutations."""

    def __init__(self, backend, *, action_duration_s=0.25):
        if action_duration_s <= 0 or action_duration_s > backend.profile.max_duration_s:
            raise ValueError('Action duration exceeds robot profile')
        self.backend = backend
        self.profile = backend.profile
        timestep = float(backend.model.opt.timestep)
        self.physics_steps = round(action_duration_s / timestep)
        if self.physics_steps < 1:
            raise ValueError('Action duration is shorter than the physics step')
        self.action_duration_s = self.physics_steps * timestep
        self.rng = None

    def reset(self, *, seed=None):
        import numpy as np
        self.rng = np.random.default_rng(seed)
        return self.backend.reset()

    def step(self, command):
        command.validate(self.profile)
        before = self.backend.observe()
        if command.episode_id != before.episode_id or command.expected_step != before.step_id:
            raise ValueError('Command is bound to a stale simulator state')
        timestep = float(self.backend.model.opt.timestep)
        steps = round(command.duration_s / timestep)
        if steps < 1:
            raise ValueError('Command duration is shorter than the physics step')
        if not math.isclose(steps * timestep, command.duration_s, rel_tol=0.0, abs_tol=timestep * 1e-6):
            raise ValueError('Command duration must align with the physics timestep')
        self.backend.begin(command)
        controls, completed_steps = [], 0
        try:
            for _ in range(steps):
                self.backend.step()
                controls.append(self.backend.data.ctrl.tolist())
                completed_steps += 1
        finally:
            self.backend.stop()
        after = self.backend.observe()
        return after, {'command_id': command.command_id, 'physics_steps': completed_steps,
                       'sim_duration_s': after.sim_time_s - before.sim_time_s,
                       'applied_controls': controls}

    def step_targets(self, targets, *, duration_s=None):
        before = self.backend.observe()
        duration = self.action_duration_s if duration_s is None else duration_s
        command = MotionCommand(str(uuid4()), self.profile.robot_id, before.episode_id,
                                before.step_id, 'joint_positions', dict(targets), duration)
        return self.step(command)

    def render_rgb(self, *, camera=None, width=320, height=240):
        return self.backend.render_rgb(camera=camera, width=width, height=height)

    def snapshot(self):
        snapshot = self.backend.snapshot()
        snapshot['environment_rng_state'] = None if self.rng is None else self.rng.bit_generator.state
        snapshot['action_duration_s'] = self.action_duration_s
        return snapshot

    def restore(self, snapshot):
        if snapshot.get('action_duration_s') != self.action_duration_s:
            raise ValueError('Snapshot action timing does not match environment')
        observation = self.backend.restore(snapshot)
        if snapshot.get('environment_rng_state') is not None:
            import numpy as np
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = snapshot['environment_rng_state']
        return observation
