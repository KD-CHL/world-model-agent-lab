"""Adapt existing joint and visual-policy runners to the shared skill boundary."""
from uuid import uuid4
from wmal.skills.registry import SkillSpec, SkillRegistry
from wmal.skills.executor import TaskSpec, SkillExecutor


def run_registered_task(instruction, channel, spec_factory, *, max_cycles, log):
    task = TaskSpec(str(uuid4()), instruction, max_cycles)
    def emit(event, payload):
        log(event, {'task_id': task.task_id, **payload})
    registry = SkillRegistry()
    spec = spec_factory(emit)
    registry.register(spec)
    emit('task_started', {'schema': 'wmal.task.v1', 'mode': spec.name, 'max_cycles': max_cycles})
    try:
        result = SkillExecutor(registry).run(spec.name, instruction, channel, max_cycles)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        # Unknown accounting is explicit; never invent zero executed actions after failure.
        emit('task_result', {'schema': 'wmal.task.v1', 'status': 'failed',
                             'cycles': None, 'error_type': type(exc).__name__})
        raise
    emit('skill_result', {'skill': result.skill, 'status': result.status,
                         'cycles': result.cycles, 'detail': result.detail})
    emit('task_result', {'schema': 'wmal.task.v1', 'status': result.status, 'cycles': result.cycles})
    return result


def joint_skill(llm, profile, *, max_cycles=100, timeout_s=30, tolerance=.03, log):
    from wmal.agents.runner import AgentRunner
    proposed = {}
    class Parser:
        def propose_goal(self, instruction, profile, observation):
            goal = llm.propose_goal(instruction, profile, observation)
            goal.validate(profile)
            proposed['goal'] = goal
            return goal
    def ready(state, instruction):
        state.validate(profile)
        return bool(instruction.strip()) and 'joint_positions' in profile.capabilities
    def execute(instruction, channel, budget):
        proposed.clear()
        return AgentRunner(Parser(), channel, profile, max_cycles=budget, tolerance=tolerance,
                           timeout_s=timeout_s, log=log).run(instruction)
    def success(state, instruction):
        goal = proposed.get('goal')
        return goal is not None and all(abs(state.joints[k]-v) <= tolerance for k,v in goal.targets.items())
    return SkillSpec('joint_goal', 'wmal.language_instruction.v1', 'wmal.joint_observation.v1',
                     ready, execute, success, max_cycles)


def visual_policy_skill(client, profile, mapping, *, max_cycles=100, timeout_s=30, log,
                        state_order, history_length=2, conditioning_steps=1, success_checker=None):
    from wmal.agents.action_runner import ActionPolicyRunner
    def ready(state, instruction):
        state.validate(profile)
        return bool(instruction.strip())
    def execute(instruction, channel, budget):
        return ActionPolicyRunner(client, channel, profile, mapping, state_order=state_order,
            history_length=history_length, conditioning_steps=conditioning_steps,
            success_checker=success_checker, log=log).run(instruction, max_cycles=budget, timeout_s=timeout_s)
    return SkillSpec('visual_action_policy', 'wmal.language_instruction.v1',
        'wmal.rgb_joint_observation.v1', ready, execute,
        lambda state, instruction: success_checker is not None and success_checker(instruction, state), max_cycles)
