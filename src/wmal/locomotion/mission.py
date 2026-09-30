"""Skill-based missions with bounded recovery and immutable original objectives."""
from dataclasses import asdict
import math
import time

from wmal.planners.replan import RECOVERABLE
from wmal.skills.registry import SkillRegistry, SkillSpec
from wmal.skills.executor import SkillExecutor
from wmal.skills.predicates import navigation_ready, navigation_reached


class MissionAgent:
    def __init__(self, agent, log, *, recover=None, max_replans=0, max_total_cycles=None):
        if type(max_replans) is not int or max_replans < 0:
            raise ValueError('Invalid recovery budget')
        if max_total_cycles is not None and (type(max_total_cycles) is not int or max_total_cycles < 1):
            raise ValueError('Invalid mission cycle budget')
        self.agent, self.log = agent, log
        self.recover, self.max_replans = recover, max_replans
        self.max_total_cycles = max_total_cycles

    def run(self, goals, session, max_cycles=150):
        if not goals or type(max_cycles) is not int or max_cycles < 1:
            raise ValueError('Mission needs goals and a positive budget')
        total_budget = self.max_total_cycles or len(goals)*max_cycles
        registry = SkillRegistry()
        registry.register(SkillSpec('navigate', 'wmal.g1.goal.v1', 'wmal.g1.base_state.v1',
                                   navigation_ready, self.agent.run_goal, navigation_reached, max_cycles))
        executor = SkillExecutor(registry)
        completed, cycles, replans, results = 0, 0, 0, []
        start = time.perf_counter()
        status = 'budget_exhausted'
        for index, goal in enumerate(goals):
            pending = [goal]
            self.log('mission_goal', {'index': index, 'goal': asdict(goal)})
            while pending and cycles < total_budget:
                target = pending[0]
                session.set_goal_marker(target)
                result = executor.run('navigate', target, session, total_budget-cycles)
                cycles += result.cycles
                status = result.status
                item = {'skill': result.skill, 'status': status, 'cycles': result.cycles,
                        'goal_index': index,
                        'position_error_m': math.hypot(result.final_state.x-target.x,
                                                       result.final_state.y-target.y)}
                results.append(item)
                self.log('skill_result', item)
                if status == 'succeeded':
                    pending.pop(0)
                    continue
                if (status not in RECOVERABLE or self.recover is None or replans >= self.max_replans
                        or cycles >= total_budget or not session.is_running):
                    break
                replans += 1
                self.log('replan_requested', {'reason': status, 'attempt': replans,
                                              'remaining_cycles': total_budget-cycles,
                                              'completed_goals': completed})
                replacement = self.recover(goal, result.final_state, status, replans, completed)
                if not replacement:
                    break
                # Recovery may insert waypoints, but must retain the original goal and tolerances.
                if replacement[-1] != goal:
                    raise ValueError('Recovery changed original task objective')
                pending = list(replacement)
            if pending:
                if cycles >= total_budget:
                    status = 'budget_exhausted'
                break
            completed += 1
        report = {'status': 'succeeded' if completed == len(goals) else status,
                  'completed_goals': completed, 'total_goals': len(goals),
                  'cycles': cycles, 'replans': replans, 'cycle_budget': total_budget,
                  'goals': results, 'elapsed_s': time.perf_counter()-start}
        self.log('mission_result', report)
        return report
