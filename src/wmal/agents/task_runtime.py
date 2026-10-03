"""Inspectable task graph, fixed skills, short imagination, measured recovery.

This scheduler is not a trained policy or an open-world language agent.
"""
from dataclasses import asdict, dataclass, replace
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np
from wmal.agents.task_graph import positive_int
from wmal.agents.task_state import TaskState
from wmal.agents.observation_history import ObservationHistory
from wmal.agents.selection_policy import select_candidates
from wmal.agents.task_execution import ExecutionManager
from wmal.agents.recovery import RecoveryPolicy
from wmal.agents.predictive_skill_agent import SkillCandidate
from wmal.skills.research_contracts import JointSkills
from wmal.skills.task_verifier import JointVerifier
from wmal.envs.visual_workcell import TARGET_LOW, TARGET_HIGH


@dataclass(frozen=True)
class RuntimeConfig:
    baseline: str = 'A3'
    recovery: bool = False
    horizon: int = 4
    error_budget: float = 1.
    seed: int = 0
    max_actions: int = 80
    max_decisions: int = 120
    max_recoveries: int = 4
    node_recoveries: int = 2
    max_rejections: int = 3
    planning_timeout_s: float = 5.
    execution_timeout_s: float = 2.

    def __post_init__(self):
        if (self.baseline not in ('A0','A1','A2','A3') or type(self.recovery) is not bool
                or (self.baseline=='A0' and self.recovery)):
            raise ValueError('Invalid baseline/recovery combination; A0 requires R0')
        for name in ('horizon','max_actions','max_decisions','max_recoveries','node_recoveries','max_rejections'):
            positive_int(getattr(self,name),name)
        positive_int(self.seed,'seed',minimum=0)
        if self.horizon>4:
            raise ValueError('Research horizon must not exceed four steps')
        for name in ('error_budget','planning_timeout_s','execution_timeout_s'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
                raise ValueError(f'Invalid {name}')


class TaskRuntime:
    def __init__(self, graph, session, predictor, calibration=None, *, config, emit=None, artifact_dir=None):
        if not isinstance(config,RuntimeConfig):
            raise ValueError('RuntimeConfig is required')
        if session.semantics!=predictor.semantics:
            raise ValueError('Robot and world-model semantics differ')
        if len(predictor.semantics.get('action_order',[]))!=2 or len(predictor.semantics.get('state_order',[]))!=4:
            raise ValueError('This runtime requires two actions/four task-state values')
        if config.horizon>predictor.metadata.get('max_horizon',0):
            raise ValueError('Requested horizon exceeds trained horizon')
        if config.baseline=='A3':
            if calibration is None:
                raise ValueError('A3 requires independent frozen calibration')
            calibration.validate_for(predictor.version,predictor.semantics)
        self.graph,self.session,self.predictor,self.calibration=graph,session,predictor,calibration
        self.config,self.emit_sink=config,emit
        self.artifact_dir=Path(artifact_dir) if artifact_dir is not None else None
        self.state=TaskState(graph)
        self.model_version=predictor.version
        self.semantics=deepcopy(predictor.semantics)
        self.calibration_hash=sha256(json.dumps(vars(calibration),sort_keys=True,default=str).encode()).hexdigest() if calibration else None
        self.scale=np.asarray(predictor.normalization.get('state_scale',np.ones(4)),dtype=float)
        if self.scale.shape!=(4,) or not np.isfinite(self.scale).all() or np.any(self.scale<=0):
            raise ValueError('Invalid model state normalization')
        self.context_steps=getattr(predictor.config,'context_steps',0)
        self.history=ObservationHistory(max(1,self.context_steps),2)
        self.manager=ExecutionManager(session,max(config.execution_timeout_s,5*session.period_s))
        self.rng=np.random.default_rng(config.seed)
        self._observed=None
        self._started=False
        self._ledger=[]
        self._cancel=None
        self._residuals=[]
        self._frozen={}
        previous=np.asarray(graph.start)
        # A0's complete nominal program is frozen before any execution.
        for node in graph.nodes:
            if node.skill=='joint_hold':
                actions=np.zeros((node.hold_steps,2))
            else:
                steps=max(1,int(np.ceil(np.max(np.abs(np.asarray(node.target)-previous))/.04)))
                target=previous.copy()
                commands=[]
                for _ in range(steps):
                    delta=np.clip(np.asarray(node.target)-target,-.04,.04)
                    commands.append(delta)
                    target+=delta
                actions=np.concatenate([commands,np.zeros((1+node.hold_steps,2))])
            self._frozen[node.node_id]=actions
            previous=np.asarray(node.target)

    def _emit(self,event,payload):
        if self.emit_sink is None:
            return
        observation=self._observed
        record={'task_id':self.graph.task_id,'node_id':self.state.active_node,
                'episode_id':observation.episode_id if observation else None,
                'step_id':observation.step_id if observation else None,
                'sim_time_s':float(self.session.data.time),**payload}
        self.emit_sink(event,record)

    def _budgets(self,node=None):
        self.state.budget_remaining={'actions':self.config.max_actions-self.state.executed_cycles,
            'decisions':self.config.max_decisions-self.state.decision_count,
            'recoveries':self.config.max_recoveries-self.state.recoveries}
        if node is not None:
            self.state.budget_remaining['node_actions']=node.max_actions-self.state.nodes[node.node_id]['actions']

    def _terminate(self,status,reason):
        self.state.status,self.state.failure_reason=status,reason

    def _versions_valid(self):
        if self.predictor.version!=self.model_version or self.predictor.semantics!=self.semantics:
            return False
        if self.calibration is not None:
            current=sha256(json.dumps(vars(self.calibration),sort_keys=True,default=str).encode()).hexdigest()
            if current!=self.calibration_hash:
                return False
        return self.session.semantics==self.semantics

    def _recover(self,node,reason):
        entry=self.state.nodes[node.node_id]
        if not RecoveryPolicy.allow(reason,self.state,node.node_id,self.config):
            self._terminate('failed',reason)
            entry['status']='failed'
            return False
        self.state.status='recovering'
        self.state.recoveries+=1
        entry['recoveries']+=1
        entry['status']='blocked'
        self._emit('recovery_started',{'reason':reason,'attempt':entry['recoveries'],
                    'target':list(node.target),'completed_subgoals':list(self.state.completed_subgoals)})
        return True

    def _save_artifact(self,decision_id,before,actions,mean,frames,bounds,trace):
        if self.artifact_dir is None or frames is None or not trace:
            return None
        name=f'prediction_{self.config.seed}_{self.state.decision_count:04d}.npz'
        count=len(trace)
        values={'before_rgb':before.rgb,'actions':actions[:count],'predicted_rgb':frames[:count],
                'observed_rgb':trace[-1].rgb,'predicted_state':mean[:count],
                'observed_state':trace[-1].state,'observed_rgb_sequence':np.stack([v.rgb for v in trace]),
                'observed_state_sequence':np.stack([v.state for v in trace]),
                'observed_step_ids':np.array([v.step_id for v in trace]),
                'episode_id':np.array(before.episode_id),'before_step':np.array(before.step_id),
                'after_step':np.array(trace[-1].step_id),'decision_id':np.array(decision_id)}
        if bounds is not None:
            values['error_bounds']=bounds[:count]
        np.savez_compressed(self.artifact_dir/name,**values)
        return name

    def run(self,cancel=None):
        if self._started:
            raise ValueError('TaskRuntime is single-use; reset creates a new task')
        self._started=True
        self._cancel=cancel
        started=time.monotonic()
        self._observed=self.session.observe()
        self.history.observe_current(self._observed)
        self.state.status='running'
        self._emit('task_started',{'schema':'wmal.monitor.task.v1','goal':list(self.graph.nodes[-1].target),
              'graph':self.graph.to_dict(),'graph_hash':self.graph.digest,'baseline':self.config.baseline,
              'recovery':self.config.recovery,'max_cycles':self.config.max_actions,
              'callable_skills':list(JointSkills.names),'model_version':self.model_version})
        start_state=self._observed.state
        if getattr(self.session,'fault_reason',None):
            self._terminate('fault_latched','session_fault')
        elif (start_state.shape!=(4,) or np.linalg.norm(start_state[2:]-self.graph.start)>.015
              or np.any(start_state[:2]<TARGET_LOW-.15) or np.any(start_state[:2]>TARGET_HIGH+.15)):
            self._terminate('needs_review','initial_pose_mismatch')
        while self.state.status=='running':
            if cancel is not None and cancel():
                self._terminate('canceled','canceled')
                break
            node=next((n for n in self.graph.nodes if self.state.nodes[n.node_id]['status']!='succeeded'
                       and all(self.state.nodes[d]['status']=='succeeded' for d in n.dependencies)),None)
            if node is None:
                self._terminate('succeeded',None)
                break
            self._run_node(node)
        self._budgets()
        self._emit('agent_state',{'state':self.state.snapshot()})
        report={'schema':'wmal.task_result.v1','baseline':self.config.baseline,'recovery':self.config.recovery,
                'seed':self.config.seed,'model_version':self.model_version,'graph_hash':self.graph.digest,
                'elapsed_s':time.monotonic()-started,'state':self.state.snapshot(),
                'execution_ledger':deepcopy(self._ledger),'prediction_residuals':deepcopy(self._residuals)}
        self._emit('task_result',report)
        return report

    def _run_node(self,node):
        state,entry=self.state,self.state.nodes[node.node_id]
        state.active_node=node.node_id
        entry['status']='running'
        verifier=JointVerifier(node.target,node.hold_steps)
        verifier.update(self._observed)
        mode='hold' if node.skill=='joint_hold' else 'reach'
        repairing=False
        progress=[]
        frozen_offset=0
        consecutive_rejections=0
        self._emit('node_state',{'node':deepcopy(entry),'target':list(node.target),'skill':node.skill})
        while state.status in ('running','recovering'):
            self._budgets(node)
            if self._cancel is not None and self._cancel():
                self._terminate('canceled','canceled')
                return
            if any(state.budget_remaining[key]<=0 for key in ('actions','node_actions','decisions')):
                self._terminate('budget_exhausted','finite_budget')
                return
            if not self._versions_valid():
                self._terminate('needs_review','model_or_semantics_changed')
                return
            matched=verifier.matches(self._observed)[0]
            if mode=='hold' and (not matched or not JointSkills.ready(self._observed,replace(node,skill='joint_hold',hold_steps=max(1,node.hold_steps)))):
                verifier.count=0
                entry['hold_count']=0
                if not self._recover(node,'hold_condition_lost'):
                    return
                repairing,mode=True,'reach'
            if mode=='reach' and matched and self.config.baseline!='A0':
                if repairing:
                    repairing=False
                    state.recoveries_succeeded+=1
                    state.status='running'
                    entry['status']='running'
                    self._emit('recovery_result',{'status':'succeeded','reason':'condition_reestablished'})
                if node.hold_steps:
                    mode='hold'
                    verifier.count=0
                else:
                    self._complete(node,entry)
                    return
            horizon=min(self.config.horizon,state.budget_remaining['actions'],state.budget_remaining['node_actions'])
            if mode=='hold':
                horizon=min(horizon,max(1,node.hold_steps-verifier.count))
                active=replace(node,skill='joint_hold',hold_steps=max(1,node.hold_steps))
            else:
                active=replace(node,skill='joint_reach')
            if self.config.baseline=='A0':
                frozen=self._frozen[node.node_id]
                if frozen_offset>=len(frozen):
                    self._terminate('failed','fixed_program_did_not_verify')
                    return
                actions=frozen[frozen_offset:frozen_offset+horizon]
                candidates=[SkillCandidate(active.skill,actions)]
            else:
                candidates=JointSkills.candidates(self._observed,active,horizon,self.rng)
            state.decision_count+=1
            state.replans+=1
            planned_at=time.monotonic()
            try:
                ranked,evidence=select_candidates(self.predictor,self._observed,candidates,np.tile(node.target,2),
                    baseline=self.config.baseline,calibration=self.calibration,error_budget=self.config.error_budget,
                    scale=self.scale,remaining=horizon,state_lower=np.concatenate([TARGET_LOW-.15,TARGET_LOW]),
                    state_upper=np.concatenate([TARGET_HIGH+.15,TARGET_HIGH]),
                    history=self.history if self.context_steps else None)
                if not self._versions_valid():
                    raise ValueError('Model/semantics changed during prediction')
            except (ValueError,RuntimeError,KeyError,TypeError) as exc:
                self._terminate('needs_review',f'prediction_invalid: {exc}')
                return
            if time.monotonic()-planned_at>self.config.planning_timeout_s:
                self._terminate('needs_review','planning_deadline')
                return
            if not ranked:
                consecutive_rejections+=1
                state.rejections+=1
                self._emit('plan',{'status':'reobserve','prefix_length':0,'evidence':{'candidates':evidence,
                     'reason':'no_trusted_prefix','new_evidence':False,'planning_ms':(time.monotonic()-planned_at)*1000}})
                if consecutive_rejections>=self.config.max_rejections:
                    self._terminate('needs_review','no_trusted_prefix')
                    return
                continue
            consecutive_rejections=0
            score,index,candidate,actions,prefix,mean,frames,bounds,diagnostics=min(ranked,key=lambda r:(r[0],r[1]))
            actions=actions[:prefix].copy()
            before=self._observed
            binding={'task':self.graph.task_id,'node':node.node_id,'graph':self.graph.digest,
                     'skill':candidate.skill,'model':self.model_version,'calibration':self.calibration_hash}
            try:
                grant=self.manager.authorize(before,actions,binding,time.monotonic()+self.config.planning_timeout_s)
            except ValueError as exc:
                self._terminate('needs_review',f'authorization_rejected: {exc}')
                return
            self._emit('plan',{'decision_id':grant.decision_id,'skill':candidate.skill,'actions':actions.tolist(),
                 'prefix_length':prefix,'model_version':self.model_version,'status':'execute',
                 'predicted_states':None if mean is None else mean.tolist(),
                 'error_bounds':None if bounds is None else bounds.tolist(),
                 'evidence':{'baseline':self.config.baseline,'selected_candidate':index,'score':score,
                     'candidates':evidence,'planning_ms':(time.monotonic()-planned_at)*1000,
                     'world_model_diagnostics':diagnostics}})
            self._emit('execution_started',{'decision_id':grant.decision_id,'skill':candidate.skill,
                                           'prefix_length':prefix,'actions':actions.tolist()})
            def on_step(observed,offset):
                nonlocal mode
                self._observed=observed
                verified=verifier.update(observed,executed=True)
                if mode=='reach' and node.hold_steps:
                    verifier.count=0  # reaching sample is not a zero-action hold
                    verified=False
                entry['hold_count']=verifier.count
                self._emit('predicate_result',dict(verifier.last,verified=verified))
                if not self._versions_valid():
                    return 'model_or_semantics_changed'
                if mode=='hold' and not verifier.last['matched']:
                    return 'hold_condition_lost'
                if bounds is not None and np.any(np.abs(observed.state-mean[offset])>bounds[offset]):
                    return 'prediction_mismatch'
                if verified and not repairing and (self.config.baseline!='A0' or frozen_offset+offset+1>=len(self._frozen[node.node_id])):
                    return 'node_verified'
                if mode=='reach' and verifier.last['matched'] and (node.hold_steps or repairing):
                    return 'reach_verified'
                return None
            result=self.manager.execute(grant,stop=self._cancel,on_step=on_step)
            count=result['executed']
            state.executed_cycles+=count
            entry['actions']+=count
            frozen_offset+=count
            ledger={k:v for k,v in result.items() if k not in ('trace','final')}
            ledger.update(node_id=node.node_id,before_step=before.step_id,
                          after_step=result['final'].step_id)
            self._ledger.append(ledger)
            trace=result['trace']
            if trace:
                self._observed=trace[-1]
            if result['status']=='fault':
                state.belief['context_valid']=False
                self._terminate('fault_latched',result['reason'])
                self._emit('execution_failed',ledger)
                return
            if trace:
                try:
                    validated=self.history.validate_execution(trace[-1],actions[:count],trace)
                    self.history.commit_execution(validated,actions[:count])
                except ValueError as exc:
                    self.session.abort('feedback_invalid')
                    state.belief['context_valid']=False
                    self._terminate('fault_latched',str(exc))
                    return
            state.belief.update(self.history.describe(),model_version=self.model_version)
            feedback={'decision_id':grant.decision_id,'skill':candidate.skill,'executed_prefix':count,
                      'authorized_prefix':prefix,'observed_state':self._observed.state.tolist(),
                      'actual_status':result['status'],'reason':result['reason']}
            if mean is not None and trace:
                actual=np.stack([v.state for v in trace])
                errors=np.abs(actual-mean[:count])
                rms=np.sqrt(np.mean(errors**2,axis=1))
                feedback.update(state_error=float(rms[-1]),state_rmse_by_step=rms.tolist(),
                                predicted_state=mean[count-1].tolist())
                if bounds is not None:
                    violations=np.any(errors>bounds[:count],axis=1)
                    feedback.update(outside_calibration_by_step=violations.tolist(),
                                    outside_calibration=bool(violations.any()),error_bound=bounds[count-1].tolist())
                self._residuals.append({'node_id':node.node_id,'rmse':rms.tolist(),
                      'bounds':None if bounds is None else bounds[:count].tolist(),
                      'absolute_error':errors.tolist()})
            artifact=self._save_artifact(grant.decision_id,before,actions,mean,frames,bounds,trace)
            if artifact:
                feedback['artifact']=artifact
            state.recent_feedback=feedback
            self._emit('feedback',feedback)
            if result['status']=='cooperative_stop':
                self._emit('execution_stopped',ledger)
            self._budgets(node)
            self._emit('agent_state',{'state':state.snapshot()})
            reason=result['reason']
            if reason=='canceled':
                self._terminate('canceled','canceled')
                return
            if reason in ('execution_deadline','model_or_semantics_changed'):
                self._terminate('needs_review',reason)
                return
            if reason=='hold_condition_lost':
                if not self._recover(node,reason):
                    return
                mode,repairing='reach',True
                continue
            if reason=='prediction_mismatch':
                if not self._recover(node,reason):
                    return
                if not repairing:
                    state.status='running'
                    entry['status']='running'
                    self._emit('recovery_result',{'status':'replanned','reason':reason})
                continue
            matched=verifier.matches(self._observed)[0]
            if mode=='reach' and matched:
                if repairing:
                    repairing=False
                    state.status='running'
                    entry['status']='running'
                    state.recoveries_succeeded+=1
                    self._emit('recovery_result',{'status':'succeeded','reason':'condition_reestablished'})
                if node.hold_steps:
                    mode='hold'
                    verifier.count=0
                    entry['hold_count']=0
                elif self.config.baseline!='A0' or frozen_offset>=len(self._frozen[node.node_id]):
                    self._complete(node,entry)
                    return
            elif mode=='hold' and verifier.count>=node.hold_steps:
                self._complete(node,entry)
                return
            if mode=='reach' and trace:
                progress.extend(float(np.linalg.norm(v.state[:2]-node.target)) for v in trace)
                if len(progress)>=4 and progress[-4]-progress[-1]<.005:
                    if not self._recover(node,'stalled'):
                        return
                    progress=[]
                    # Same finite candidate set, next seeded proposals; no action replay.
                    state.status='running'
                    entry['status']='running'
                    self._emit('recovery_result',{'status':'replanned','reason':'stalled'})

    def _complete(self,node,entry):
        entry['status']='succeeded'
        entry['evidence']={'episode_id':self._observed.episode_id,'step_id':self._observed.step_id,
                           'observed_state':self._observed.state.tolist(),'sim_time_s':float(self.session.data.time)}
        self.state.completed_subgoals.append(node.node_id)
        self.state.status='running'
        self._emit('node_state',{'node':deepcopy(entry),'target':list(node.target),'skill':node.skill})
