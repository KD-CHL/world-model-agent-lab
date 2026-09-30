"""Inspectable fixed-skill Agent: consequence prediction is not action execution.

A0 fixed candidate, A1 nominal feedback correction, A2 learned consequence
ranking, A3 calibrated execution prefixes. Only real feedback completes goals.
"""
from dataclasses import dataclass, field
import math
import time
from uuid import uuid4

import numpy as np


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


@dataclass(frozen=True)
class TaskStage:
    name: str
    target: np.ndarray


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


@dataclass(frozen=True)
class SkillCandidate:
    skill: str
    actions: np.ndarray
    nominal_terminal: np.ndarray | None = None


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


class PredictiveSkillAgent:
    def __init__(self,task,predictor,calibration=None,*,baseline='A3',error_budget=.5,
                 action_lower,action_upper,state_lower=None,state_upper=None):
        if baseline not in ('A0','A1','A2','A3') or not math.isfinite(error_budget) or error_budget<=0:
            raise ValueError('Invalid Agent baseline or error budget')
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
        self.lower,self.upper=lower,upper
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
        self._pending=None
        self._episode=None
        self._step=None
        self._force_reobserve=False

    def _target(self):
        stages=self.task.stages or (TaskStage(self.task.task_id,self.task.goal),)
        return stages[min(len(self.state.completed_subgoals),len(stages)-1)]

    def _reobserve(self,observation,reason,started):
        self.state.failure_reason=reason
        return SkillDecision(str(uuid4()),observation.episode_id,observation.step_id,None,
                np.empty((0,len(self.lower))),0,self.predictor.version,None,None,None,
                {'baseline':self.baseline,'reason':reason,'planning_ms':(time.perf_counter()-started)*1000},'reobserve')

    def plan(self,observation,candidates):
        started=time.perf_counter()
        if self._pending is not None or self.state.status!='running':
            raise ValueError('Agent already has an outstanding execution or is terminal')
        if (np.asarray(observation.state).shape!=np.asarray(self.task.goal).shape
                or (self._episode is not None and observation.episode_id!=self._episode)
                or (self._step is not None and observation.step_id!=self._step)):
            raise ValueError('Stale observation or unexpected episode')
        self._episode,self._step=observation.episode_id,observation.step_id
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
        ranked,evidence=[],[]
        for index,candidate in enumerate(candidates):
            actions=np.asarray(candidate.actions)[:remaining]
            mean,frames,bounds=None,None,None
            prefix=len(actions)
            score=float(index)  # A0: fixed plan ordering, no model query.
            if self.baseline=='A1':
                if candidate.nominal_terminal is None:
                    raise ValueError('A1 requires an explicit controller-provided nominal terminal state')
                terminal=np.asarray(candidate.nominal_terminal)
                if terminal.shape!=target.shape or not np.isfinite(terminal).all():
                    raise ValueError('Invalid nominal outcome')
                score=float(np.linalg.norm((terminal-target)/self.scale))
            elif self.baseline in ('A2','A3'):
                result=self.predictor.predict(observation.rgb,observation.state,actions)
                values=np.asarray(result['states'])
                video=np.asarray(result['frames'])
                if (result['model_version']!=self.predictor.version or values.ndim!=3
                        or values.shape[1:]!=(len(actions),len(target)) or len(values)<2
                        or video.ndim!=5 or video.shape[:2]!=values.shape[:2]
                        or video.shape[2:]!=np.asarray(observation.rgb).shape
                        or not np.isfinite(values).all() or not np.isfinite(video).all()):
                    raise ValueError('Malformed/stale world-model response')
                mean,frames=values.mean(0),video.mean(0)
                if self.baseline=='A3':
                    self.calibration.validate_for(self.predictor.version,self.predictor.semantics)
                    bounds=self.calibration.bounds(values.std(0),version=self.predictor.version)
                    trusted=(bounds/self.scale).max(-1)<=self.error_budget
                    if self.state_lower is not None:
                        trusted &= np.all((mean-bounds>=self.state_lower)&(mean+bounds<=self.state_upper),axis=-1)
                    prefix=int(np.cumprod(trusted).sum())
                elif self.state_lower is not None:
                    if np.any(values<self.state_lower) or np.any(values>self.state_upper):
                        prefix=0
                if prefix:
                    score=float(np.linalg.norm((mean[prefix-1]-target)/self.scale))
                    if bounds is not None:
                        score+=float(np.linalg.norm(bounds[prefix-1]/self.scale))
            evidence.append({'candidate':index,'skill':candidate.skill,'trusted_prefix':prefix,
                             'score':score if prefix else None})
            if prefix:
                ranked.append((score,index,candidate,actions,prefix,mean,frames,bounds))
        self.state.replans+=1
        if not ranked:
            return self._reobserve(observation,'no_trusted_prefix',started)
        score,index,candidate,actions,prefix,mean,frames,bounds=min(ranked,key=lambda r:(r[0],r[1]))
        decision=SkillDecision(str(uuid4()),observation.episode_id,observation.step_id,candidate.skill,
                actions[:prefix].copy(),prefix,self.predictor.version,mean,frames,bounds,
                {'baseline':self.baseline,'selected_candidate':index,'score':score,'candidates':evidence,
                 'planning_ms':(time.perf_counter()-started)*1000,'target_stage':self._target().name})
        self._pending=decision
        self.state.failure_reason=None
        return decision

    def record_feedback(self,decision,observation):
        if (self._pending is None or decision.decision_id!=self._pending.decision_id
                or decision.model_version!=self.predictor.version or observation.episode_id!=decision.episode_id
                or observation.step_id!=decision.step_id+decision.prefix_length
                or np.asarray(observation.state).shape!=np.asarray(self.task.goal).shape):
            raise ValueError('Feedback is stale, duplicated, incomplete or from another episode/model')
        feedback={'decision_id':decision.decision_id,'skill':decision.skill,
                  'executed_prefix':decision.prefix_length,'model_version':decision.model_version,
                  'observed_state':np.asarray(observation.state).tolist()}
        if decision.predicted_states is not None:
            expected=decision.predicted_states[decision.prefix_length-1]
            error=np.abs(observation.state-expected)
            feedback.update(predicted_state=expected.tolist(),state_error=float(np.sqrt(np.mean(error**2))),
                            frame_mse=float(np.mean((observation.rgb-decision.predicted_frames[decision.prefix_length-1])**2)))
            if decision.error_bounds is not None:
                bound=decision.error_bounds[decision.prefix_length-1]
                mismatch=bool(np.any(error>bound))
                feedback.update(error_bound=bound.tolist(),outside_calibration=mismatch)
                self._force_reobserve=mismatch
                if mismatch:
                    self.state.failure_reason='prediction_mismatch'
        self.state.executed_cycles+=decision.prefix_length
        self.state.recent_feedback=feedback
        self._pending=None
        self._step=observation.step_id
        stage=self._target()
        if np.linalg.norm(observation.state-stage.target)<=self.task.tolerance:
            self.state.completed_subgoals.append(stage.name)
            if len(self.state.completed_subgoals)==len(self.task.stages or (stage,)):
                self.state.status='succeeded'
        if self.state.status=='running' and self.state.executed_cycles>=self.task.max_cycles:
            self.state.status='budget_exhausted'
        return feedback

    def record_execution_failure(self,decision,observation,reason):
        """Known partial receipt: account real progress, stop; never retry uncertain actions."""
        if (self._pending is None or decision.decision_id!=self._pending.decision_id
                or observation.episode_id!=decision.episode_id
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
        self._step=observation.step_id
