"""Bounded high-level execution loop. Observations are the source of success."""
from dataclasses import dataclass


@dataclass
class TaskResult:
    status: str
    cycles: int
    detail: str


class AgentRunner:
    def __init__(self, llm, channel, profile, max_cycles=10, tolerance=0.03, timeout_s=30, log=None):
        if max_cycles < 1 or tolerance <= 0 or timeout_s <= 0:
            raise ValueError('Invalid runner budget')
        self.llm, self.channel, self.profile = llm, channel, profile
        self.max_cycles, self.tolerance, self.timeout_s = max_cycles, tolerance, timeout_s
        self.log = log or (lambda event, payload: None)

    def run(self, instruction):
        cycles = 0
        try:
            obs = self.channel.observe(timeout_s=self.timeout_s)
            obs.validate(self.profile)
            goal = self.llm.propose_goal(instruction, self.profile, obs)
            goal.validate(self.profile)
            self.log('goal', {'intent': goal.intent, 'targets': goal.targets})
            episode = obs.episode_id
            while True:
                obs.validate(self.profile)
                if obs.episode_id != episode:
                    raise ValueError('Episode changed during task')
                if max(abs(obs.joints[key] - value) for key, value in goal.targets.items()) <= self.tolerance:
                    return TaskResult('succeeded', cycles, 'Goal verified from robot observation')
                if cycles >= self.max_cycles:
                    return TaskResult('budget_exhausted', cycles, 'Goal not observed within cycle budget')
                plan = self.channel.plan(self.profile, obs, goal, timeout_s=self.timeout_s)
                plan.validate(self.profile, obs)
                self.log('plan', {'plan_id': plan.plan_id, 'model_version': plan.model_version,
                                  'observation_step': obs.step_id, 'command_id': plan.commands[0].command_id,
                                  'prediction': {'horizon_steps': plan.prediction.horizon_steps,
                                                 'predicted_state': plan.prediction.predicted_state,
                                                 'predicted_terminal_state': plan.prediction.predicted_terminal_state,
                                                 'objective_cost': plan.prediction.objective_cost,
                                                 'uncertainty_kind': plan.prediction.uncertainty_kind,
                                                 'uncertainty': plan.prediction.uncertainty}})
                command = plan.commands[0]
                result = self.channel.execute(command, timeout_s=self.timeout_s)
                cycles += 1
                self.log('execution', {'command_id': result.command_id, 'status': result.status})
                if result.command_id != command.command_id:
                    raise ValueError('Execution receipt mismatch')
                if result.status != 'succeeded':
                    return TaskResult('failed', cycles, result.status)
                next_obs = self.channel.observe(timeout_s=self.timeout_s)
                if next_obs.step_id <= obs.step_id and next_obs.episode_id == obs.episode_id:
                    raise ValueError('No fresh observation after execution')
                residual = {joint: next_obs.joints[joint] - plan.prediction.predicted_state[joint]
                            for joint in self.profile.joint_limits}
                residual_rmse = (sum(value * value for value in residual.values()) /
                                 len(residual)) ** 0.5
                self.log('prediction_residual', {
                    'episode_id': next_obs.episode_id, 'observation_step': next_obs.step_id,
                    'model_version': plan.model_version, 'residual_rad': residual,
                    'rmse_rad': residual_rmse})
                obs = next_obs
        except (ValueError, TypeError, KeyError, TimeoutError, RuntimeError, OSError) as exc:
            # Never include HTTP response bodies, API keys or provider prompts in error logs.
            self.log('failure', {'error_type': type(exc).__name__})
            return TaskResult('failed', cycles, type(exc).__name__)
