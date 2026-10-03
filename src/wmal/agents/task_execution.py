"""Single-use grants and actual partial execution; never retry uncertain actions."""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import time
from uuid import uuid4
import numpy as np
from wmal.agents.predictive_skill_agent import VisualObservation, _snapshot
from wmal.envs.visual_workcell import TARGET_LOW, TARGET_HIGH, MAX_DELTA


def observation_digest(value):
    return sha256(value.episode_id.encode()+str(value.step_id).encode()
                  +value.state.tobytes()+value.rgb.tobytes()).hexdigest()


@dataclass(frozen=True)
class ExecutionGrant:
    decision_id: str
    episode_id: str
    step_id: int
    observation_hash: str
    actions: np.ndarray
    binding_json: str
    deadline: float


class ExecutionManager:
    def __init__(self, session, timeout_s=2.):
        self.session = session
        self.timeout_s = timeout_s
        self._pending = None
        self._authority = None

    def authorize(self, observation, actions, binding, deadline):
        if self._pending is not None or getattr(self.session,'fault_reason',None):
            raise ValueError('Execution outstanding or session fault-latched')
        actions = np.asarray(actions,dtype=float)
        if (actions.ndim!=2 or actions.shape[1]!=2 or not 1<=len(actions)<=4
                or not np.isfinite(actions).all() or np.any(np.abs(actions)>MAX_DELTA+1e-8)
                or not np.isfinite(deadline) or deadline<=time.monotonic()):
            raise ValueError('Invalid committed action prefix or deadline')
        targets = observation.state[2:]+np.cumsum(actions,axis=0)
        if np.any(targets<TARGET_LOW-1e-8) or np.any(targets>TARGET_HIGH+1e-8):
            raise ValueError('Full committed prefix leaves target envelope')
        if observation_digest(self.session.observe())!=observation_digest(observation):
            raise ValueError('Authorization observation is stale')
        args = (str(uuid4()),observation.episode_id,observation.step_id,observation_digest(observation),
                _snapshot(actions),json.dumps(binding,sort_keys=True,allow_nan=False),float(deadline))
        self._pending = ExecutionGrant(*args)
        self._authority = ExecutionGrant(*args[:4],_snapshot(actions),*args[5:])
        return self._pending

    def execute(self, token, stop=None, on_step=None):
        expected = self._authority
        if token is not self._pending or expected is None:
            raise ValueError('Grant is not the outstanding authoritative decision')
        if (any(getattr(token,n)!=getattr(expected,n) for n in
                ('decision_id','episode_id','step_id','observation_hash','binding_json','deadline'))
                or token.actions.dtype!=expected.actions.dtype or token.actions.shape!=expected.actions.shape
                or token.actions.tobytes()!=expected.actions.tobytes()):
            raise ValueError('Grant payload/metadata changed')
        self._pending = self._authority = None  # consumed even on failure
        trace, status, reason, unresolved = [], 'complete', None, False
        current = self.session.observe()
        try:
            if (time.monotonic()>expected.deadline or getattr(self.session,'fault_reason',None)
                    or observation_digest(current)!=expected.observation_hash):
                raise ValueError('Expired, faulted or stale execution grant')
            for index, action in enumerate(expected.actions):
                if stop is not None and stop():
                    status,reason = 'cooperative_stop','canceled'
                    break
                started = time.monotonic()
                returned = self.session.execute_actions(action[None],episode_id=expected.episode_id,
                                                         step_id=expected.step_id+index)
                actual = self.session.observe()  # controller cannot assert its own result
                if (actual.episode_id!=expected.episode_id or actual.step_id!=expected.step_id+index+1
                        or actual.state.shape!=(4,) or actual.rgb.shape!=current.rgb.shape):
                    unresolved = True
                    raise ValueError('Missing/cross-episode physical feedback')
                actual = VisualObservation(actual.episode_id,actual.step_id,actual.rgb,actual.state)
                trace.append(actual)
                current = actual
                if observation_digest(returned)!=observation_digest(actual):
                    raise ValueError('Controller receipt differs from independent observation')
                if np.any(actual.state[:2]<TARGET_LOW-.15) or np.any(actual.state[:2]>TARGET_HIGH+.15):
                    raise ValueError('Actual joints left experimental envelope')
                if time.monotonic()-started>self.timeout_s:
                    status,reason = 'cooperative_stop','execution_deadline'
                    break
                reason = on_step(actual,index) if on_step is not None else None
                if reason:
                    status = 'cooperative_stop'
                    break
        except (Exception,KeyboardInterrupt) as exc:
            status,reason = 'fault',f'{type(exc).__name__}: {exc}'
            # An acknowledgement may fail AFTER physics advanced. Observe once,
            # never resend. Only one additional contiguous endpoint is admissible.
            try:
                final=self.session.observe()
                if (final.episode_id==expected.episode_id
                        and final.step_id==expected.step_id+len(trace)+1
                        and len(trace)<len(expected.actions)
                        and final.state.shape==(4,) and final.rgb.shape==current.rgb.shape):
                    final=VisualObservation(final.episode_id,final.step_id,final.rgb,final.state)
                    trace.append(final)
                    current=final
                elif final.episode_id!=expected.episode_id or final.step_id!=expected.step_id+len(trace):
                    unresolved=True
            except Exception:
                unresolved=True
            self.session.abort('task_execution_fault')
        return {'decision_id':expected.decision_id,'status':status,'reason':reason,
                'executed':len(trace),'authorized':len(expected.actions),'trace':tuple(trace),
                'unresolved':unresolved,'actual_total':None if unresolved else len(trace),'final':current}
