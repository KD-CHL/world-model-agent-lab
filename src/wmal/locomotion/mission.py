"""Sequential mission execution above the receding-horizon G1 agent."""
from dataclasses import asdict
import math
import time


class MissionAgent:
    def __init__(self, agent, log):
        self.agent, self.log = agent, log

    def run(self, goals, session, max_cycles=150):
        if not goals:
            raise ValueError('Mission needs at least one goal')
        completed, cycles, results = 0, 0, []
        start = time.perf_counter()
        for index, goal in enumerate(goals):
            session.set_goal_marker(goal)
            self.log('mission_goal', {'index': index, 'goal': asdict(goal)})
            result = self.agent.run_goal(goal, session, max_cycles=max_cycles)
            cycles += result.cycles
            results.append({'status': result.status, 'cycles': result.cycles,
                            'position_error_m': math.hypot(result.final_state.x-goal.x, result.final_state.y-goal.y)
                            if result.final_state else None})
            if result.status != 'succeeded':
                break
            completed += 1
        report = {'status': 'succeeded' if completed == len(goals) else result.status,
                  'completed_goals': completed, 'total_goals': len(goals),
                  'cycles': cycles, 'goals': results, 'elapsed_s': time.perf_counter()-start}
        self.log('mission_result', report)
        return report
