"""Constrained task-to-goal boundary for high-level agents.

An LLM or scripted policy may propose a goal, but this layer owns schema and
joint-limit validation before a proposal reaches the world-model planner.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TaskProposal:
    intent: str
    targets: dict[str, float]
    skill: str = "joint_goal"


class GoalParser(Protocol):
    def propose(self, instruction: str, joint_limits: dict[str, tuple[float, float]]) -> TaskProposal: ...


class ConstrainedTaskPlanner:
    def __init__(self, parser: GoalParser, joint_limits: dict[str, tuple[float, float]]):
        if not joint_limits or any(low > high for low, high in joint_limits.values()):
            raise ValueError("Invalid joint limits")
        self.parser, self.joint_limits = parser, dict(joint_limits)

    def propose(self, instruction: str) -> TaskProposal:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Task instruction is required")
        proposal = self.parser.propose(instruction, self.joint_limits)
        if not proposal.intent or not proposal.targets:
            raise ValueError("Agent returned an empty task proposal")
        if not set(proposal.targets).issubset(self.joint_limits):
            raise ValueError("Agent proposed an unknown joint")
        for joint, value in proposal.targets.items():
            low, high = self.joint_limits[joint]
            if not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f"Agent target outside limits: {joint}")
        return proposal
