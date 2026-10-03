"""Task success is measured; read-only observations never count as held time."""
import numpy as np
from wmal.agents.task_graph import joint_target, positive_int


class JointVerifier:
    def __init__(self, target, required_steps=0):
        self.target = np.asarray(joint_target(target))
        self.required_steps = positive_int(required_steps, 'required_steps', minimum=0)
        self.count = 0
        self._episode = None
        self._step = None
        self.last = {}

    def matches(self, observation):
        if observation.state.shape != (4,):
            raise ValueError('Joint predicate requires four named state values')
        measured = float(np.linalg.norm(observation.state[:2]-self.target))
        command = float(np.linalg.norm(observation.state[2:]-self.target))
        return measured <= .035 and command <= .015, measured, command

    def update(self, observation, executed=False):
        if type(executed) is not bool:
            raise ValueError('Execution flag must be boolean')
        if self._episode is not None:
            if observation.episode_id != self._episode:
                raise ValueError('Predicate cannot cross episodes')
            expected = self._step + (1 if executed else 0)
            if observation.step_id != expected:
                raise ValueError('Predicate requires contiguous real execution samples')
        elif executed:
            raise ValueError('Predicate requires initial real observation')
        self._episode, self._step = observation.episode_id, observation.step_id
        matched, measured, command = self.matches(observation)
        if not matched:
            self.count = 0
        elif executed:
            self.count += 1
        self.last = {'matched':matched, 'measured_error_rad':measured, 'command_error_rad':command,
                     'hold_count':self.count, 'required_steps':self.required_steps,
                     'episode_id':self._episode, 'step_id':self._step}
        return matched and (not self.required_steps or self.count >= self.required_steps)
