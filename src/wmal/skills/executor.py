"""Skill execution requires observed success and a finite cycle budget."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    max_cycles: int
    max_replans: int = 0

    def __post_init__(self):
        if (not isinstance(self.task_id, str) or not self.task_id
                or not isinstance(self.instruction, str) or not self.instruction.strip()
                or type(self.max_cycles) is not int or self.max_cycles < 1
                or type(self.max_replans) is not int or self.max_replans < 0):
            raise ValueError('Invalid task specification')


@dataclass(frozen=True)
class ExecutionFeedback:
    skill: str
    status: str
    cycles: int
    detail: str
    final_state: Any = None

    def __post_init__(self):
        statuses = {'succeeded', 'failed', 'canceled', 'rejected', 'timeout', 'stopped',
                    'budget_exhausted', 'safety_stop', 'stalled', 'no_candidate',
                    'replan_required', 'precondition_failed', 'success_unverified'}
        if (not isinstance(self.skill, str) or not self.skill or self.status not in statuses
                or type(self.cycles) is not int or self.cycles < 0 or not isinstance(self.detail, str)):
            raise ValueError('Invalid skill feedback')


class SkillExecutor:
    def __init__(self, registry):
        self.registry = registry

    def run(self, name, parameters, session, budget):
        spec = self.registry.get(name)
        if type(budget) is not int or budget < 1:
            raise ValueError('Skill requires remaining cycle budget')
        before = session.observe()
        if not spec.precondition(before, parameters):
            return ExecutionFeedback(name, 'precondition_failed', 0, 'Precondition not satisfied', before)
        result = spec.execute(parameters, session, min(budget, spec.max_cycles))
        if type(result.cycles) is not int or not 0 <= result.cycles <= min(budget, spec.max_cycles):
            raise ValueError('Skill returned invalid cycle accounting')
        # Query independently: a policy cannot assert success through its return value.
        after = session.observe()
        if before.episode_id != after.episode_id:
            raise ValueError('Episode changed during skill')
        status = result.status
        if status == 'succeeded' and not spec.success(after, parameters):
            status = 'success_unverified'
        return ExecutionFeedback(name, status, result.cycles, result.detail, after)
