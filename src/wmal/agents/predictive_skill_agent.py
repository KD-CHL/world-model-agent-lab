"""Inspectable fixed-skill Agent: consequence prediction is not action execution.

A0 fixed candidate, A1 nominal feedback correction, A2 learned consequence
ranking, A3 calibrated execution prefixes. Only real feedback completes goals.
"""
from dataclasses import dataclass, field, replace
from copy import deepcopy
import math
import time
from uuid import uuid4

import numpy as np


def _snapshot(value):
    """Immutable bytes backing also prevents callers re-enabling write flags."""
    array=np.asarray(value,dtype=float)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class VisualObservation:
    episode_id: str
    step_id: int
    rgb: np.ndarray
    state: np.ndarray

    def __post_init__(self):
        rgb,state=np.asarray(self.rgb),np.asarray(self.state)
        if (not self.episode_id or type(self.step_id) is not int or self.step_id<0
                or rgb.ndim!=3 or rgb.shape[0]!=3 or not np.isfinite(rgb).all()
                or rgb.min()<0 or rgb.max()>1 or state.ndim!=1 or not np.isfinite(state).all()):
            raise ValueError('Invalid visual observation')
        object.__setattr__(self,'rgb',_snapshot(rgb))
        object.__setattr__(self,'state',_snapshot(state))


@dataclass(frozen=True)
class TaskStage:
    name: str
    target: np.ndarray

    def __post_init__(self):
        target=np.asarray(self.target)
        if not isinstance(self.name,str) or not self.name or target.ndim!=1 or not np.isfinite(target).all():
            raise ValueError('Invalid task stage')
        object.__setattr__(self,'target',_snapshot(target))


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    goal: np.ndarray
    callable_skills: tuple
    tolerance: float = .05
    max_cycles: int = 100
    stages: tuple = ()

    def __post_init__(self):
        goal=np.asarray(self.goal)
        if (not self.task_id or goal.ndim!=1 or not goal.size or not np.isfinite(goal).all()
                or not self.callable_skills or len(set(self.callable_skills))!=len(self.callable_skills)
                or any(not isinstance(s,str) or not s for s in self.callable_skills)
                or not math.isfinite(self.tolerance) or self.tolerance<=0
                or type(self.max_cycles) is not int or self.max_cycles<1):
            raise ValueError('Invalid Agent task')
        if self.stages:
            if len({s.name for s in self.stages})!=len(self.stages):
                raise ValueError('Duplicate task stage')
            for stage in self.stages:
                if not stage.name or np.asarray(stage.target).shape!=goal.shape or not np.isfinite(stage.target).all():
                    raise ValueError('Invalid stage target')
            if not np.array_equal(self.stages[-1].target,goal):
                raise ValueError('Final stage must match task goal')
        object.__setattr__(self,'goal',_snapshot(goal))
        object.__setattr__(self,'stages',tuple(self.stages))
        object.__setattr__(self,'callable_skills',tuple(self.callable_skills))


@dataclass(frozen=True)
class SkillCandidate:
    skill: str
    actions: np.ndarray
    nominal_terminal: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self,'actions',_snapshot(self.actions))
        if self.nominal_terminal is not None:
            object.__setattr__(self,'nominal_terminal',_snapshot(self.nominal_terminal))


@dataclass
class ResearchAgentState:
    task_goal: list
    callable_skills: list
    completed_subgoals: list = field(default_factory=list)
    current_candidates: list = field(default_factory=list)
    recent_feedback: dict = field(default_factory=dict)
    failure_reason: str | None = None
    status: str = 'running'
    executed_cycles: int = 0
    replans: int = 0
    active_subgoal: str | None = None
    decision_count: int = 0
    reobservations: int = 0
    belief: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SkillDecision:
    decision_id: str
    episode_id: str
    step_id: int
    skill: str | None
    actions: np.ndarray
    prefix_length: int
    model_version: str
    predicted_states: np.ndarray | None
    predicted_frames: np.ndarray | None
    error_bounds: np.ndarray | None
    evidence: dict
    status: str = 'execute'

    def __post_init__(self):
        for name in ('actions','predicted_states','predicted_frames','error_bounds'):
            value=getattr(self,name)
            if value is not None:
                object.__setattr__(self,name,_snapshot(value))


class PredictiveSkillAgent:
    def __init__(self,task,predictor,calibration=None,*,baseline='A3',error_budget=.5,
                 action_lower,action_upper,state_lower=None,state_upper=None,max_reobservations=3):
        if baseline not in ('A0','A1','A2','A3') or not math.isfinite(error_budget) or error_budget<=0:
            raise ValueError('Invalid Agent baseline or error budget')
        if type(max_reobservations) is not int or max_reobservations<1:
            raise ValueError('Invalid reobservation budget')
        self.max_reobservations=max_reobservations
        semantics=predictor.semantics
        lower,upper=np.asarray(action_lower),np.asarray(action_upper)
        if (lower.shape!=(len(semantics['action_order']),) or upper.shape!=lower.shape
                or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>=upper)
                or np.asarray(task.goal).shape!=(len(semantics['state_order']),)):
            raise ValueError('Task/actions incompatible with world-model coordinates')
        if baseline=='A3':
            if calibration is None:
                raise ValueError('A3 requires independent held-out calibration')
            calibration.validate_for(predictor.version,semantics)
        self.task,self.predictor,self.calibration=task,predictor,calibration
        self.baseline,self.error_budget=baseline,error_budget
        self.lower,self.upper=_snapshot(lower),_snapshot(upper)
        self.state_lower=None if state_lower is None else np.asarray(state_lower,dtype=float)
        self.state_upper=None if state_upper is None else np.asarray(state_upper,dtype=float)
        if (self.state_lower is None)!=(self.state_upper is None):
            raise ValueError('Provide both state bounds')
        if self.state_lower is not None and (self.state_lower.shape!=np.asarray(task.goal).shape
                or self.state_upper.shape!=self.state_lower.shape or np.any(self.state_lower>=self.state_upper)
                or not np.isfinite(self.state_lower).all() or not np.isfinite(self.state_upper).all()):
            raise ValueError('Invalid named state bounds')
        normalization=getattr(predictor,'normalization',{})
        self.scale=np.asarray(normalization.get('state_scale',np.ones(len(task.goal))))
        self.state=ResearchAgentState(np.asarray(task.goal).tolist(),list(task.callable_skills))
        self.state.active_subgoal=self._target().name
        self._pending=None
        self._pending_payload=None
        self._episode=None
        self._step=None
        self._force_reobserve=False
        self.context_steps=getattr(getattr(predictor,'config',None),'context_steps',0)
        self._history=None
        if self.context_steps:
            from wmal.agents.observation_history import ObservationHistory
            self._history=ObservationHistory(self.context_steps,len(self.lower))

    def _target(self):
        stages=self.task.stages or (TaskStage(self.task.task_id,self.task.goal),)
        return stages[min(len(self.state.completed_subgoals),len(stages)-1)]

    @property
    def active_stage(self):
        """Controller proposals must target this stage, not skip to the final goal."""
        return self._target()

    def _update_belief(self,observation):
        self.state.belief.update(episode_id=observation.episode_id,step_id=observation.step_id,
                                model_version=self.predictor.version,observed_state=observation.state.tolist(),
                                conditioning='current_real_observation',latent_memory='not_enabled')
        if self._history is not None:
            self.state.belief.update(self._history.describe())

    def _reobserve(self,observation,reason,started):
        self.state.failure_reason=reason
        if self.state.status=='running':
            self.state.reobservations+=1
            if self.state.reobservations>=self.max_reobservations:
                self.state.status='needs_review'
        return SkillDecision(str(uuid4()),observation.episode_id,observation.step_id,None,
                np.empty((0,len(self.lower))),0,self.predictor.version,None,None,None,
                {'baseline':self.baseline,'reason':reason,
                 'reobservations_remaining':max(0,self.max_reobservations-self.state.reobservations),
                 'planning_ms':(time.perf_counter()-started)*1000},'reobserve')

    def plan(self,observation,candidates):
        started=time.perf_counter()
        if self._pending is not None or self.state.status!='running':
            raise ValueError('Agent already has an outstanding execution or is terminal')
        if (np.asarray(observation.state).shape!=np.asarray(self.task.goal).shape
                or (self._episode is not None and observation.episode_id!=self._episode)
                or (self._step is not None and observation.step_id!=self._step)):
            raise ValueError('Stale observation or unexpected episode')
        if self._history is not None:
            self._history.observe_current(observation)
        self._episode,self._step=observation.episode_id,observation.step_id
        self.state.decision_count+=1
        self._update_belief(observation)
        if not candidates:
            return self._reobserve(observation,'no_candidate',started)
        for candidate in candidates:
            actions=np.asarray(candidate.actions)
            if (candidate.skill not in self.task.callable_skills or actions.ndim!=2 or not len(actions)
                    or actions.shape[1]!=len(self.lower) or not np.isfinite(actions).all()
                    or np.any(actions<self.lower) or np.any(actions>self.upper)):
                raise ValueError('Unknown skill or candidate outside explicit action constraints')
        self.state.current_candidates=[c.skill for c in candidates]
        if self._force_reobserve:
            self._force_reobserve=False
            return self._reobserve(observation,'prediction_mismatch_reobserve',started)
        remaining=self.task.max_cycles-self.state.executed_cycles
        if remaining<=0:
            self.state.status='budget_exhausted'
            return self._reobserve(observation,'budget_exhausted',started)
        target=np.asarray(self._target().target)
        from wmal.agents.selection_policy import select_candidates
        ranked,evidence=select_candidates(self.predictor,observation,candidates,target,
                baseline=self.baseline,calibration=self.calibration,error_budget=self.error_budget,
                scale=self.scale,remaining=remaining,state_lower=self.state_lower,
                state_upper=self.state_upper,history=self._history)
        self.state.replans+=1
        if not ranked:
            return self._reobserve(observation,'no_trusted_prefix',started)
        score,index,candidate,actions,prefix,mean,frames,bounds,diagnostics=min(ranked,key=lambda r:(r[0],r[1]))
        decision=SkillDecision(str(uuid4()),observation.episode_id,observation.step_id,candidate.skill,
                actions[:prefix].copy(),prefix,self.predictor.version,mean,frames,bounds,
                {'baseline':self.baseline,'selected_candidate':index,'score':score,'candidates':evidence,
                 'planning_ms':(time.perf_counter()-started)*1000,'target_stage':self._target().name,
                 'world_model_diagnostics':diagnostics,
                 'trust_mechanism':'held_out_calibrated_error_bound' if bounds is not None else 'not_calibrated'})
        self._pending=decision
        self._pending_payload=replace(decision,evidence=deepcopy(decision.evidence))
        self.state.failure_reason=None
        return decision

    def _receipt(self,decision):
        """Public ndarray headers are mutable even when their bytes aren't."""
        if decision is not self._pending or self._pending_payload is None:
            raise ValueError('Receipt is not the pending authoritative decision')
        expected=self._pending_payload
        for name in ('actions','predicted_states','predicted_frames','error_bounds'):
            public,private=getattr(decision,name),getattr(expected,name)
            if (public is None)!=(private is None):
                raise ValueError('Execution receipt payload changed')
            if public is not None and (public.dtype!=private.dtype or public.shape!=private.shape
                                      or public.tobytes()!=private.tobytes()):
                raise ValueError('Execution receipt array metadata/content changed')
        return expected

    def record_feedback(self,decision,observation,*,observations=None):
        decision=self._receipt(decision)
        if (decision.model_version!=self.predictor.version or observation.episode_id!=decision.episode_id
                or observation.step_id!=decision.step_id+decision.prefix_length
                or np.asarray(observation.state).shape!=np.asarray(self.task.goal).shape):
            raise ValueError('Feedback is stale, duplicated, incomplete or from another episode/model')
        trace=None if self._history is None else self._history.validate_execution(
            observation,decision.actions,observations)
        feedback={'decision_id':decision.decision_id,'skill':decision.skill,
                  'target_stage':self._target().name,
                  'executed_prefix':decision.prefix_length,'model_version':decision.model_version,
                  'observed_state':np.asarray(observation.state).tolist()}
        if decision.predicted_states is not None:
            expected=decision.predicted_states[decision.prefix_length-1]
            error=np.abs(observation.state-expected)
            feedback.update(predicted_state=expected.tolist(),state_error=float(np.sqrt(np.mean(error**2))),
                            frame_mse=float(np.mean((observation.rgb-decision.predicted_frames[decision.prefix_length-1])**2)))
            if trace is not None:
                trace_states=np.stack([v.state for v in trace])
                trace_rgb=np.stack([v.rgb for v in trace])
                trace_error=np.abs(trace_states-decision.predicted_states[:decision.prefix_length])
                feedback.update(observed_states_by_step=trace_states.tolist(),
                                state_rmse_by_step=np.sqrt(np.mean(trace_error**2,axis=-1)).tolist(),
                                frame_mse_by_step=np.mean((trace_rgb-decision.predicted_frames[:decision.prefix_length])**2,
                                                         axis=(1,2,3)).tolist())
            if decision.error_bounds is not None:
                bound=decision.error_bounds[decision.prefix_length-1]
                mismatch=bool(np.any(error>bound))
                if trace is not None:
                    violations=np.any(trace_error>decision.error_bounds[:decision.prefix_length],axis=-1)
                    mismatch=bool(violations.any())
                    feedback.update(outside_calibration_by_step=violations.tolist(),
                                    first_mismatch_step_id=trace[int(np.flatnonzero(violations)[0])].step_id
                                    if mismatch else None)
                feedback.update(error_bound=bound.tolist(),outside_calibration=mismatch)
                self._force_reobserve=mismatch
                if mismatch:
                    self.state.failure_reason='prediction_mismatch'
        if self._history is not None:
            self._history.commit_execution(trace,decision.actions)
            feedback['history_step_ids']=[v.step_id for v in trace]
        self.state.executed_cycles+=decision.prefix_length
        self.state.reobservations=0
        self.state.recent_feedback=feedback
        self._pending=None
        self._pending_payload=None
        self._step=observation.step_id
        self._update_belief(observation)
        self.state.belief.update(last_prediction_error=feedback.get('state_error'),
                                outside_calibration=feedback.get('outside_calibration'),
                                world_model_diagnostics=decision.evidence.get('world_model_diagnostics'))
        stage=self._target()
        if np.linalg.norm(observation.state-stage.target)<=self.task.tolerance:
            self.state.completed_subgoals.append(stage.name)
            if len(self.state.completed_subgoals)==len(self.task.stages or (stage,)):
                self.state.status='succeeded'
        self.state.active_subgoal=None if self.state.status=='succeeded' else self._target().name
        if self.state.status=='running' and self.state.executed_cycles>=self.task.max_cycles:
            self.state.status='budget_exhausted'
        return feedback

    def record_execution_failure(self,decision,observation,reason):
        """Known partial receipt: account real progress, stop; never retry uncertain actions."""
        decision=self._receipt(decision)
        if (observation.episode_id!=decision.episode_id
                or not decision.step_id<=observation.step_id<=decision.step_id+decision.prefix_length
                or not isinstance(reason,str) or not reason):
            raise ValueError('Invalid execution failure receipt')
        executed=observation.step_id-decision.step_id
        self.state.executed_cycles+=executed
        self.state.status='execution_failed'
        self.state.failure_reason=reason
        self.state.recent_feedback={'decision_id':decision.decision_id,'executed_prefix':executed,
                                    'observed_state':np.asarray(observation.state).tolist(),
                                    'status':'execution_failed','reason':reason}
        self._pending=None
        self._pending_payload=None
        self._step=observation.step_id
        self._update_belief(observation)
        if self._history is not None:
            # A partial failure cannot manufacture missing intermediate frames.
            self.state.belief['context_valid']=False
