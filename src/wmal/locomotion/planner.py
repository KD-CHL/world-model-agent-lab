"""Sampling-based, action-conditioned receding-horizon G1 planner."""
from dataclasses import dataclass, field
import math

import numpy as np

from wmal.locomotion.contracts import (G1_POLICY_PERIOD_S, G1Goal, G1State,
                                       G1VelocityAction)
from wmal.locomotion.world_model import WorldModelAdapter
from wmal.planners.replan import NoFeasibleCandidate


@dataclass(frozen=True)
class G1Plan:
    model_version: str
    observation_step: int
    action: G1VelocityAction
    predicted_state: G1State
    predicted_terminal_state: G1State
    objective_cost: float
    horizon_steps: int = 1
    uncertainty: dict = field(default_factory=dict)
    uncertainty_kind: str = 'unavailable'
    evidence: dict = field(default_factory=dict)

    def prediction_report(self):
        from wmal.communication.contracts import PredictionReport
        names = ('x', 'y', 'yaw', 'roll', 'pitch', 'pelvis_height')
        return PredictionReport(self.model_version, self.observation_step, self.horizon_steps,
            {key: getattr(self.predicted_state, key) for key in names}, self.objective_cost,
            self.uncertainty_kind, dict(self.uncertainty),
            {key: getattr(self.predicted_terminal_state, key) for key in names})


def _wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


class G1RolloutPlanner:
    """Evaluate bounded velocity sequences; execute only the first chunk."""

    def __init__(self, model, samples=48, horizon=3, seed=0, action_duration_s=0.5,
                 max_linear_velocity=0.45, max_yaw_rate=0.7, uncertainty_weight=0.02):
        if type(samples) is not int or samples < 4 or type(horizon) is not int or horizon < 1:
            raise ValueError('Planner requires at least four samples and one horizon step')
        if not 0.05 <= action_duration_s <= 1 or not 0 < max_linear_velocity <= 0.5:
            raise ValueError('Planner limits exceed the G1 policy-supported range')
        frame_count = round(action_duration_s / G1_POLICY_PERIOD_S)
        if frame_count < 3 or abs(frame_count * G1_POLICY_PERIOD_S - action_duration_s) > 1e-9:
            raise ValueError('Action duration must be an exact multiple of the G1 policy period')
        if not 0 < max_yaw_rate <= 1 or uncertainty_weight < 0:
            raise ValueError('Invalid planner bounds')
        self.model = model if isinstance(model, WorldModelAdapter) else WorldModelAdapter(model)
        self.samples, self.horizon, self.seed = samples, horizon, seed
        self.action_duration_s = float(action_duration_s)
        self.max_linear_velocity, self.max_yaw_rate = float(max_linear_velocity), float(max_yaw_rate)
        self.uncertainty_weight = float(uncertainty_weight)
        self.feedback_scale = 1.0
        self.scene = None
        self.last_evidence = {}

    def _batch_score(self, state, goal, sequences, rollout):
        """Use mean outcomes for progress and each member for modelled constraint screening."""
        paths = rollout.means()
        evaluations, results = [], []
        spread = rollout.members[...,:2].var(axis=0).sum(axis=-1)
        for index, (sequence, path) in enumerate(zip(sequences,paths)):
            reason = None
            members = rollout.members[:,index]
            if np.any(members[...,8]<.48) or np.any(np.abs(members[...,6:8])>.65):
                reason = 'predicted_posture_constraint'
            if self.scene is not None and reason is None:
                # Member paths can straddle an obstacle while their mean passes through it.
                for member in np.concatenate([members,path[None]],axis=0):
                    previous = (state.x,state.y)
                    for point in member:
                        if not self.scene.segment_free(previous,tuple(point[:2])):
                            reason = 'predicted_scene_constraint'
                            break
                        previous = tuple(point[:2])
                    if reason:
                        break
            current = path[-1]
            position_error = math.hypot(current[0]-goal.x,current[1]-goal.y)
            yaw_error = 0. if goal.yaw is None else abs(_wrap_angle(goal.yaw-current[2]))
            unstable = max(0.,abs(current[6])-.35)**2+max(0.,abs(current[7])-.35)**2
            low_height = max(0.,.58-current[8])
            control_cost = float(np.sum(sequence[:,:2]**2)+.3*np.sum(sequence[:,2]**2))
            total_spread = float(np.sqrt(spread[index]).sum())
            cost = (position_error+.4*yaw_error+5.*unstable+10.*low_height
                    +.04*control_cost+self.uncertainty_weight*total_spread)
            evaluations.append({'candidate_id':index,'cost':float(cost) if reason is None else None,
                'rejection':reason,'terminal_distance_m':position_error,
                'terminal_position_spread_m':float(np.sqrt(spread[index,-1])),
                'first_action':sequence[0].tolist()})
            results.append((float(cost) if reason is None else float('inf'),index))
        self.last_evidence = {'schema':'wmal.planning_evidence.v1','mode':'member_preserving_batch',
            'model_version':self.model.version,'episode_id':state.episode_id,'observation_step':state.step_id,
            'horizon_steps':self.horizon,'action_duration_s':self.action_duration_s,
            'candidate_count':len(sequences),'ensemble_members':len(rollout.members),
            'network_forward_calls':len(rollout.members)*self.horizon,
            'model_imagination_steps':rollout.imagination_steps,
            'uncertainty_kind':'ensemble_spread','success_probability':None,
            'candidates':evaluations}
        cost,index = min(results)
        if not math.isfinite(cost):
            raise NoFeasibleCandidate('World model predicts no safe candidate sequence')
        self.last_evidence['selected_candidate'] = index
        first,terminal = rollout.prediction(index,0),rollout.prediction(index,self.horizon-1)
        return G1Plan(self.model.version,state.step_id,rollout.candidates[index][0],first.state,
                      terminal.state,cost,self.horizon,dict(first.uncertainty),first.uncertainty_kind,
                      dict(self.last_evidence))

    def _sequences(self, state, goal):
        rng = np.random.default_rng(self.seed + state.step_id)
        sequences = rng.uniform(-1.0, 1.0, size=(self.samples, self.horizon, 3))
        sequences[:, :, 0] *= self.max_linear_velocity
        sequences[:, :, 1] *= min(0.25, self.max_linear_velocity)
        sequences[:, :, 2] *= self.max_yaw_rate
        planar_speed = np.hypot(sequences[:, :, 0], sequences[:, :, 1])
        scale = np.minimum(1.0, self.max_linear_velocity / np.maximum(planar_speed, 1e-12))
        sequences[:, :, 0] *= scale
        sequences[:, :, 1] *= scale

        dx, dy = goal.x - state.x, goal.y - state.y
        distance = math.hypot(dx, dy)
        heading_error = _wrap_angle(math.atan2(dy, dx) - state.yaw) if distance else 0.0
        forward = min(self.max_linear_velocity, distance / max(self.action_duration_s * self.horizon, 1e-6))
        sequences[0, :, :] = (0.0, 0.0, 0.0)
        sequences[1, :, :] = (forward, 0.0,
                              float(np.clip(heading_error, -self.max_yaw_rate, self.max_yaw_rate)))
        # Include a body-frame translation proposal for reverse/lateral goals.
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        gain = min(self.max_linear_velocity / max(distance, 1e-9),
                   1. / (self.action_duration_s * self.horizon))
        sequences[3, :, :] = ((c*dx+s*dy)*gain, (-s*dx+c*dy)*gain, 0.)
        if goal.yaw is not None:
            turn = float(np.clip(_wrap_angle(goal.yaw - state.yaw), -self.max_yaw_rate, self.max_yaw_rate))
            sequences[2, :, :] = (0.0, 0.0, turn)
        return sequences * self.feedback_scale

    def _score(self, state, goal, sequence):
        current = state
        first_prediction = None
        total_uncertainty = 0.0
        control_cost = 0.0
        for index, values in enumerate(sequence):
            action = G1VelocityAction(float(values[0]), float(values[1]), float(values[2]),
                                     self.action_duration_s)
            prediction = self.model.predict(current, action, action.duration_s)
            if self.scene is not None and not self.scene.segment_free(
                    (current.x, current.y), (prediction.state.x, prediction.state.y)):
                return float('inf'), None, None
            current = prediction.state
            if (current.pelvis_height < 0.48 or abs(current.roll) > 0.65
                    or abs(current.pitch) > 0.65):
                return float('inf'), None, None
            if index == 0:
                first_prediction = prediction
            # The learned adapter defines position_variance in square metres.
            # Unknown provider spread and mixed posture units are not added to distance cost.
            if prediction.uncertainty_kind == 'ensemble_spread':
                total_uncertainty += math.sqrt(prediction.uncertainty.get('position_variance', 0.))
            control_cost += action.vx ** 2 + action.vy ** 2 + 0.3 * action.yaw_rate ** 2
        position_error = math.hypot(current.x - goal.x, current.y - goal.y)
        yaw_error = 0.0 if goal.yaw is None else abs(_wrap_angle(goal.yaw - current.yaw))
        unstable = max(0.0, abs(current.roll) - 0.35) ** 2 + max(0.0, abs(current.pitch) - 0.35) ** 2
        low_height = max(0.0, 0.58 - current.pelvis_height)
        cost = (position_error + 0.4 * yaw_error + 5.0 * unstable + 10.0 * low_height
                + 0.04 * control_cost + self.uncertainty_weight * total_uncertainty)
        return cost, first_prediction, current

    def plan(self, state, goal):
        if not isinstance(state, G1State) or not isinstance(goal, G1Goal):
            raise ValueError('G1 planner requires G1State and G1Goal')
        sequences = self._sequences(state,goal)
        actions = [[G1VelocityAction(*map(float,values),self.action_duration_s) for values in row]
                   for row in sequences]
        rollout = self.model.rollout(state,actions)
        if rollout is not None:
            return self._batch_score(state,goal,sequences,rollout)
        best = None
        self.last_evidence = {'schema':'wmal.planning_evidence.v1','mode':'scalar_mean_feedback',
                              'model_version':self.model.version,'episode_id':state.episode_id,
                              'observation_step':state.step_id,'candidate_count':len(sequences),
                              'success_probability':None,'candidates':[]}
        for index,sequence in enumerate(sequences):
            result = self._score(state, goal, sequence)
            self.last_evidence['candidates'].append({'candidate_id':index,
                'cost':float(result[0]) if math.isfinite(result[0]) else None,
                'rejection':None if math.isfinite(result[0]) else 'predicted_constraint'})
            if best is None or result[0] < best[0]:
                best = result
                best_action = G1VelocityAction(*map(float, sequence[0]), self.action_duration_s)
        if best is None or not math.isfinite(best[0]):
            raise NoFeasibleCandidate('World model predicts no safe candidate sequence')
        return G1Plan(self.model.version, state.step_id, best_action,
                      best[1].state, best[2], float(best[0]), self.horizon,
                      dict(best[1].uncertainty), best[1].uncertainty_kind if best[1].uncertainty else 'unavailable',
                      dict(self.last_evidence))
