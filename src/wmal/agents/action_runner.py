"""Closed-loop runner for a language-conditioned action-chunk service."""
from dataclasses import dataclass

from wmal.communication.observation_history import rgb_to_chw
from wmal.models.action_service import ActionQuery


@dataclass
class PolicyTaskResult:
    status: str
    cycles: int
    detail: str


class ActionPolicyRunner:
    """Request an action chunk, execute one safe step, and require fresh frames."""

    def __init__(self, client, channel, profile, mapping, *, state_order, history_length=2,
                 conditioning_steps=1, success_checker=None, log=None):
        if not state_order or len(set(state_order)) != len(state_order):
            raise ValueError("state_order must be a unique ordered vector")
        if not set(state_order).issubset(profile.joint_limits):
            raise ValueError("state_order does not match robot profile")
        if type(history_length) is not int or history_length < 1 or history_length > 8:
            raise ValueError("history_length must be in [1, 8]")
        if type(conditioning_steps) is not int or conditioning_steps < 1 or conditioning_steps > 64:
            raise ValueError("conditioning_steps must be in [1, 64]")
        self.client, self.channel, self.profile, self.mapping = client, channel, profile, mapping
        self.state_order, self.history_length = tuple(state_order), history_length
        self.conditioning_steps = conditioning_steps
        self.success_checker = success_checker
        self.log = log or (lambda event, payload: None)

    def _query(self, instruction, history):
        state_width = len(self.state_order)
        action_width = len(self.mapping.action_order)
        states, images = [], []
        for item in history:
            observation = item.observation
            if any(name not in observation.joints for name in self.state_order):
                raise ValueError("Observation is missing configured state dimensions")
            states.append([observation.joints[name] for name in self.state_order])
            images.append(rgb_to_chw(item.rgb, item.width, item.height))
        query = ActionQuery(instruction, states, images,
                            [[0.0] * action_width for _ in range(self.conditioning_steps)],
                            state_width, action_width)
        query.validate()
        return query

    def run(self, instruction, *, max_cycles=20, timeout_s=30):
        cycles = 0
        try:
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Task instruction is required")
            if type(max_cycles) is not int or max_cycles < 1 or timeout_s <= 0:
                raise ValueError("Invalid policy execution budget")
            history = self.channel.observe_with_images(self.history_length, timeout_s=timeout_s)
            episode = history[-1].observation.episode_id
            while cycles < max_cycles:
                observation = history[-1].observation
                observation.validate(self.profile)
                if observation.episode_id != episode:
                    raise ValueError("Episode changed during policy task")
                if self.success_checker is not None and self.success_checker(instruction, observation):
                    return PolicyTaskResult("succeeded", cycles, "Success verified from robot observation")
                query = self._query(instruction, history)
                chunk = self.client.propose(query)
                if chunk.model_version != self.client.version:
                    raise ValueError("Action service model version mismatch")
                command = self.mapping.command(chunk.actions[0], observation, self.profile)
                receipt = self.channel.execute(command, timeout_s=timeout_s)
                cycles += 1
                self.log("policy_action", {"model_version": chunk.model_version,
                                           "episode_id": observation.episode_id,
                                           "observation_step": observation.step_id,
                                           "command_id": command.command_id,
                                           "command": command.values,
                                           "action_chunk_length": len(chunk.actions),
                                           "execution_status": receipt.status})
                if receipt.command_id != command.command_id:
                    raise ValueError("Execution receipt mismatch")
                if receipt.status != "succeeded":
                    return PolicyTaskResult("failed", cycles, receipt.status)
                history = self.channel.observe_with_images(self.history_length, timeout_s=timeout_s)
                if history[-1].observation.episode_id != episode:
                    raise ValueError("Episode changed after execution")
                if history[-1].observation.step_id <= observation.step_id:
                    raise ValueError("No fresh camera/state observation after action")
                self.log("policy_observation", {"episode_id": episode,
                                                "step_id": history[-1].observation.step_id,
                                                "model_version": chunk.model_version})
            return PolicyTaskResult("budget_exhausted", cycles, "No success detector or success not observed")
        except (ValueError, TypeError, KeyError, TimeoutError, RuntimeError, OSError) as exc:
            self.log("policy_failure", {"error_type": type(exc).__name__})
            return PolicyTaskResult("failed", cycles, type(exc).__name__)
