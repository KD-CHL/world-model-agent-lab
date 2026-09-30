"""Task-scoped Agent loop; simulator-session lifetime remains with its caller."""
from dataclasses import dataclass, asdict
import math
import time

from wmal.locomotion.contracts import G1Goal, G1State
from wmal.planners.replan import NavigationStalled, NoFeasibleCandidate


@dataclass(frozen=True)
class G1TaskResult:
    status: str
    cycles: int
    detail: str
    final_state: G1State | None = None


class G1Agent:
    def __init__(self, planner, *, max_tilt_rad=0.65, min_pelvis_height_m=0.48,
                 log=None, feedback=None):
        if not 0 < max_tilt_rad < 1.2 or min_pelvis_height_m <= 0:
            raise ValueError('Invalid Agent safety limits')
        self.planner = planner
        self.max_tilt_rad, self.min_pelvis_height_m = max_tilt_rad, min_pelvis_height_m
        self.log = log or (lambda event, payload: None)
        self.feedback = feedback

    @staticmethod
    def _goal_reached(state, goal):
        if math.hypot(state.x - goal.x, state.y - goal.y) > goal.position_tolerance_m:
            return False
        return goal.yaw is None or abs((goal.yaw - state.yaw + math.pi) % (2 * math.pi) - math.pi) <= goal.yaw_tolerance_rad

    def _safe(self, state):
        return (state.pelvis_height >= self.min_pelvis_height_m
                and abs(state.roll) <= self.max_tilt_rad
                and abs(state.pitch) <= self.max_tilt_rad)

    def run_goal(self, goal, session, max_cycles=100):
        if not isinstance(goal, G1Goal) or type(max_cycles) is not int or max_cycles < 1:
            raise ValueError('A G1Goal and positive cycle budget are required')
        cycles, previous = 0, None
        try:
            state = session.observe()
            if not isinstance(state, G1State):
                raise ValueError('Simulator returned an incompatible G1 observation')
            episode = state.episode_id
            while True:
                if not getattr(session, 'is_running', True):
                    return G1TaskResult('stopped', cycles, 'Simulation session is not running', state)
                if not self._safe(state):
                    self.log('safety_stop', {'step_id': state.step_id})
                    return G1TaskResult('safety_stop', cycles, 'G1 posture outside safety limits', state)
                if previous is not None and (state.episode_id != episode or state.step_id <= previous.step_id
                                              or state.sim_time_s <= previous.sim_time_s):
                    raise ValueError('Simulator observation is stale or changed episode')
                if self._goal_reached(state, goal):
                    return G1TaskResult('succeeded', cycles, 'Goal observed within tolerance', state)
                if cycles >= max_cycles:
                    return G1TaskResult('budget_exhausted', cycles, 'Goal not reached within cycle budget', state)
                planning_start = time.perf_counter()
                plan = self.planner.plan(state, goal)
                planning_latency_s = time.perf_counter() - planning_start
                if self.feedback is not None and getattr(self.feedback, 'calibration', None) is not None:
                    self.feedback.calibration.validate_for(plan.model_version, plan.action.duration_s)
                if plan.observation_step != state.step_id:
                    raise ValueError('Planner returned a stale plan')
                self.log('plan', {'episode_id': episode, 'observation_step': state.step_id,
                                  'planning_latency_s': planning_latency_s,
                                  'model_version': plan.model_version,
                                  'action': {'vx': plan.action.vx, 'vy': plan.action.vy,
                                             'yaw_rate': plan.action.yaw_rate,
                                             'duration_s': plan.action.duration_s},
                                  'predicted_state': plan.predicted_state.__dict__,
                                  'predicted_terminal_state': plan.predicted_terminal_state.__dict__,
                                  'objective_cost': plan.objective_cost,
                                  'prediction': asdict(plan.prediction_report()),
                                  'planning_evidence': plan.evidence,
                                  'prediction_horizon_s': plan.horizon_steps * plan.action.duration_s,
                                  'prediction_state_schema': 'wmal.g1.base_state.v1'})
                session.step(plan.action, plan.action.duration_s)
                cycles += 1
                previous = state
                state = session.observe()
                if not isinstance(state, G1State):
                    raise ValueError('Simulator returned an incompatible G1 observation')
                if (state.episode_id != previous.episode_id or state.step_id <= previous.step_id
                        or state.sim_time_s <= previous.sim_time_s):
                    raise ValueError('Simulator observation is stale or changed episode')
                self.log('executed_transition', {'schema':'wmal.executed_transition.v1',
                    'before':asdict(previous),'action':asdict(plan.action),'after':asdict(state),
                    'actual_duration_s':state.sim_time_s-previous.sim_time_s,
                    'model_version':plan.model_version,
                    'predicted_state':asdict(plan.predicted_state)})
                replan_required = False
                if self.feedback is not None:
                    feedback = self.feedback.update(plan.predicted_state, state, self.planner)
                    self.log('feedback', feedback)
                    replan_required = feedback.get('replan_required', False)
                self.log('prediction_residual', {
                    'episode_id': state.episode_id, 'observation_step': state.step_id,
                    'model_version': plan.model_version,
                    'duration_s': plan.action.duration_s,
                    'position_error_m': math.hypot(state.x - plan.predicted_state.x,
                                                   state.y - plan.predicted_state.y),
                    'yaw_error_rad': abs((state.yaw - plan.predicted_state.yaw + math.pi)
                                         % (2 * math.pi) - math.pi)})
                if not self._safe(state):
                    self.log('safety_stop', {'step_id': state.step_id})
                    return G1TaskResult('safety_stop', cycles, 'G1 posture outside safety limits', state)
                if self._goal_reached(state, goal):
                    return G1TaskResult('succeeded', cycles, 'Goal observed within tolerance', state)
                if replan_required:
                    return G1TaskResult('replan_required', cycles, 'Calibrated residual alarm', state)
        except (NavigationStalled, NoFeasibleCandidate) as exc:
            status = 'stalled' if isinstance(exc, NavigationStalled) else 'no_candidate'
            self.log('planning_event', {'reason': status})
            return G1TaskResult(status, cycles, status, state)
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            self.log('failure', {'error_type': type(exc).__name__, 'detail': str(exc)})
            return G1TaskResult('failed', cycles, type(exc).__name__)
