#!/usr/bin/env python3
"""Keep the G1 viewer alive after bounded research tasks; no LLM/controller training."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from contextlib import ExitStack
import json
from pathlib import Path
import select
import sys
import time

import numpy as np

from wmal.agents.predictive_skill_agent import AgentTask, TaskStage, PredictiveSkillAgent, SkillCandidate
from wmal.envs.visual_workcell import VisualWorkcellSession, TARGET_LOW, TARGET_HIGH, MAX_DELTA
from wmal.logging.manifest import atomic_json, build_manifest
from wmal.models.horizon_calibration import HorizonCalibration
from wmal.models.visual_latent import VisualWorldModel


def candidates(observation, goal, horizon, rng):
    """Fixed, explicit arm_delta capability; nominal outcomes are not learned predictions."""
    result=[]
    current=observation.state[2:]
    direction=np.sign(np.asarray(goal[:2])-current)
    proposals=[np.tile(direction*.04*gain,(horizon,1)) for gain in (1.,.5,-.5,0.)]
    proposals.extend(rng.uniform(-.04,.04,(horizon,2)) for _ in range(8))
    for proposal in proposals:
        target=current.copy()
        actions=[]
        for delta in proposal:
            updated=np.clip(target+delta,TARGET_LOW+.005,TARGET_HIGH-.005)
            actions.append(updated-target)
            target=updated
        result.append(SkillCandidate('arm_delta',np.array(actions),np.concatenate([target,target])))
    return result


def fixed_joint_plan(start,goal,budget):
    """Precompute a finite A0 plan once; consume it without state-dependent correction."""
    target=np.asarray(start,dtype=float).copy()
    goal=np.asarray(goal,dtype=float)
    result=[]
    for _ in range(budget):
        delta=np.clip(goal-target,-.04,.04)
        result.append(delta)
        target+=delta
    return np.asarray(result)


def run_task(session,model,calibration,goal,*,baseline,horizon,max_cycles,error_budget,seed,log,artifact_dir,
             waypoints=()):
    goal=np.asarray(goal,dtype=float)
    if goal.shape!=(2,) or not np.isfinite(goal).all() or np.any(goal<TARGET_LOW) or np.any(goal>TARGET_HIGH):
        raise ValueError('Goal must contain shoulder/elbow targets within the local envelope')
    state_goal=np.concatenate([goal,goal])
    points=[np.asarray(point,dtype=float) for point in waypoints]
    if any(p.shape!=(2,) or not np.isfinite(p).all() or np.any(p<TARGET_LOW) or np.any(p>TARGET_HIGH) for p in points):
        raise ValueError('Waypoints must contain shoulder/elbow targets within the local envelope')
    stages=tuple(TaskStage(f'waypoint_{i+1}',np.tile(p,2)) for i,p in enumerate(points))
    if stages:
        stages+= (TaskStage('final_goal',state_goal),)
    if baseline=='A0' and points and horizon!=1:
        raise ValueError('A0 waypoint experiment requires horizon=1 to audit each intermediate target')
    task=AgentTask(f'arm_goal_{seed}',state_goal,('arm_delta',),tolerance=.035,max_cycles=max_cycles,stages=stages)
    agent=PredictiveSkillAgent(task,model,calibration,baseline=baseline,error_budget=error_budget,
            action_lower=[-MAX_DELTA]*2,action_upper=[MAX_DELTA]*2,
            state_lower=np.concatenate([TARGET_LOW-.15,TARGET_LOW]),
            state_upper=np.concatenate([TARGET_HIGH+.15,TARGET_HIGH]))
    rng=np.random.default_rng(seed)
    observed=session.observe()
    def emit(event,payload):
        record={'event':event,'wall_time_utc':datetime.now(timezone.utc).isoformat(),
                'task_id':task.task_id,'episode_id':observed.episode_id,'step_id':observed.step_id,
                'sim_time_s':float(session.data.time),**payload}
        log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        log.flush()
    emit('task_started',{'schema':'wmal.monitor.task.v1','goal':state_goal.tolist(),
                        'baseline':baseline,'max_cycles':max_cycles,'callable_skills':list(task.callable_skills),
                        'model_version':model.version,'semantics':session.semantics,
                        'stages':[{'name':s.name,'target':s.target.tolist()} for s in stages]})
    emit('agent_state',{'state':asdict(agent.state)})
    frozen=fixed_joint_plan(observed.state[2:],goal,max_cycles) if baseline=='A0' else None
    if baseline=='A0' and points:
        # Fixed once from the initial nominal state. No actual-state correction.
        previous=observed.state[2:]
        segments=[]
        for point in points+[goal]:
            steps=max(1,int(np.ceil(np.max(np.abs(point-previous))/.04)))
            segments.append(fixed_joint_plan(previous,point,steps))
            segments.append(np.zeros((1,2)))  # nominal settle, not a synthetic success
            previous=point
        frozen=np.concatenate(segments+[np.zeros((max_cycles,2))])[:max_cycles]
    started=time.perf_counter()
    reobservations=0
    for cycle in range(max_cycles+3):
        if agent.state.status!='running':
            break
        if frozen is not None:
            offset=agent.state.executed_cycles
            available=[SkillCandidate('arm_delta',frozen[offset:offset+horizon])]
        else:
            available=candidates(observed,agent.active_stage.target,horizon,rng)
        decision=agent.plan(observed,available)
        record={'cycle':cycle,'episode_id':observed.episode_id,'step_id':observed.step_id,
                'decision_id':decision.decision_id,'status':decision.status,'prefix_length':decision.prefix_length,
                'skill':decision.skill,'actions':decision.actions.tolist(),'model_version':decision.model_version,
                'evidence':decision.evidence,'predicted_states':None if decision.predicted_states is None else decision.predicted_states.tolist(),
                'error_bounds':None if decision.error_bounds is None else decision.error_bounds.tolist(),
                'uncertainty_kind':'calibrated_error_bound' if decision.error_bounds is not None else 'unavailable'}
        emit('plan',record)
        emit('agent_state',{'state':asdict(agent.state)})
        if decision.prefix_length==0:
            reobservations+=1
            if reobservations>=3:
                agent.state.status='needs_review'
                break
            observed=session.observe()
            continue
        reobservations=0
        before=observed
        emit('execution_started',{'decision_id':decision.decision_id,'skill':decision.skill,
                                  'prefix_length':decision.prefix_length,'actions':decision.actions.tolist()})
        trace=[]
        try:
            options={'on_observation':trace.append} if agent.context_steps else {}
            observed=session.execute_actions(decision.actions,episode_id=decision.episode_id,step_id=decision.step_id,
                                             **options)
        except (ValueError,RuntimeError):
            session.abort('controller_rejected_or_failed')
            observed=session.observe()
            agent.record_execution_failure(decision,observed,'controller_rejected_or_failed')
            emit('execution_failed',{'decision_id':decision.decision_id,'reason':'controller_rejected_or_failed',
                                     'state':asdict(agent.state)})
            break
        feedback=agent.record_feedback(decision,observed,observations=trace if agent.context_steps else None)
        # Concrete RGB evidence makes prediction/observation alignment inspectable.
        if decision.predicted_frames is not None:
            artifact=f'prediction_{seed}_{cycle:04d}.npz'
            arrays=dict(
                before_rgb=before.rgb,actions=decision.actions,predicted_rgb=decision.predicted_frames[:decision.prefix_length],
                observed_rgb=observed.rgb,predicted_state=decision.predicted_states[:decision.prefix_length],
                observed_state=observed.state,episode_id=np.array(observed.episode_id),
                before_step=np.array(before.step_id),after_step=np.array(observed.step_id),
                decision_id=np.array(decision.decision_id))
            if decision.error_bounds is not None:
                arrays['error_bounds']=decision.error_bounds[:decision.prefix_length]
            if trace:
                arrays.update(observed_rgb_sequence=np.stack([v.rgb for v in trace]),
                              observed_state_sequence=np.stack([v.state for v in trace]),
                              observed_step_ids=np.array([v.step_id for v in trace]))
            np.savez_compressed(artifact_dir/artifact,**arrays)
            feedback['artifact']=artifact
        emit('feedback',feedback)
        emit('observation',{'observed_state':observed.state.tolist(),'state_order':session.semantics['state_order']})
        emit('agent_state',{'state':asdict(agent.state)})
    result={'baseline':baseline,'seed':seed,'elapsed_s':time.perf_counter()-started,
            'model_version':model.version,'state':asdict(agent.state),
            'goal_error_rad':float(np.linalg.norm(observed.state-state_goal))}
    emit('task_result',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--calibration')
    parser.add_argument('--task',default='stack_block')
    parser.add_argument('--goal',nargs=2,type=float,default=[.45,.95])
    parser.add_argument('--waypoints',nargs='+',type=float,default=[],
                        help='Ordered shoulder/elbow pairs before the final --goal')
    parser.add_argument('--baseline',choices=['A0','A1','A2','A3'],default='A3')
    parser.add_argument('--horizon',type=int,default=4)
    parser.add_argument('--max-cycles',type=int,default=30)
    parser.add_argument('--error-budget',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',required=True)
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--monitor',action='store_true',help='Serve a persistent read-only browser monitor')
    parser.add_argument('--monitor-port',type=int,default=8765)
    parser.add_argument('--monitor-runs-root',default='runs')
    parser.add_argument('--realtime',action='store_true',help='Pace action execution at simulation time')
    args=parser.parse_args()
    if len(args.waypoints)%2:
        parser.error('Waypoints must be shoulder/elbow pairs')
    waypoints=np.asarray(args.waypoints,dtype=float).reshape(-1,2).tolist()
    if any(not np.isfinite(p).all() or np.any(p<TARGET_LOW) or np.any(p>TARGET_HIGH) for p in map(np.asarray,waypoints)):
        parser.error('Waypoints outside local shoulder/elbow envelope')
    if args.baseline=='A0' and waypoints and args.horizon!=1:
        parser.error('A0 waypoint experiment requires --horizon 1')
    import torch
    torch.set_num_threads(1)
    model=VisualWorldModel.load(args.checkpoint,device=args.device)
    if model.metadata.get('training_source',{}).get('task')!=args.task:
        parser.error('Checkpoint scene scope differs or is unknown; scene transfer needs separate validation/calibration')
    calibration=HorizonCalibration.load(args.calibration) if args.calibration else None
    if args.horizon<1 or args.horizon>model.metadata['max_horizon']:
        parser.error('Horizon must not exceed training horizon')
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be new/empty')
    output.mkdir(parents=True,exist_ok=True)
    inputs={'checkpoint':args.checkpoint} if Path(args.checkpoint).is_file() else {}
    if args.calibration:
        inputs['calibration']=args.calibration
    atomic_json(output/'run_manifest.json',build_manifest('visual_skill_agent',args.seed,inputs,
                parameters={'task':args.task,'goal':args.goal,'baseline':args.baseline,'max_cycles':args.max_cycles,
                            'horizon':args.horizon,'error_budget':args.error_budget,'realtime':args.realtime,
                            'waypoints':waypoints},
                semantics=model.semantics,model_version=model.version))
    try:
        with ExitStack() as stack:
            monitor=None
            if args.monitor:
                from wmal.monitor.runtime import MonitorRuntime
                monitor=stack.enter_context(MonitorRuntime(output,args.monitor_runs_root,args.monitor_port,
                                                           camera=model.semantics['camera']))
                print(f'Agent monitor: {monitor.url}',flush=True)
            session=stack.enter_context(VisualWorkcellSession(args.task,image_size=model.config.image_size,
                         camera=model.semantics['camera'],period_s=model.semantics['period_s'],seed=args.seed,
                         frame_publisher=monitor.publisher if monitor else None,realtime=args.realtime))
            if session.semantics!=model.semantics:
                raise ValueError('Model was trained with different robot/camera/action coordinates')
            viewer=session.open_viewer() if args.viewer else None
            with (output/'events.jsonl').open('w') as log:
                goal,task_index=args.goal,0
                stdin_active=True
                while True:
                    report=run_task(session,model,calibration,goal,baseline=args.baseline,horizon=args.horizon,
                        max_cycles=args.max_cycles,error_budget=args.error_budget,seed=args.seed+task_index,
                        log=log,artifact_dir=output,waypoints=waypoints if task_index==0 else ())
                    atomic_json(output/f'task_{task_index:03d}.json',report)
                    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
                    if viewer is None and monitor is None:
                        break
                    print('观测会话保持打开。输入 shoulder elbow 开始新目标，reset 显式重置故障会话，quit 退出。',flush=True)
                    goal=None
                    while viewer is None or viewer.is_running():
                        # Hold the last command; do not run unbudgeted Agent actions.
                        if viewer is not None:
                            session.idle()
                        else:
                            session.publish_monitor_frame()
                        if stdin_active and sys.stdin in select.select([sys.stdin],[],[],0)[0]:
                            line=sys.stdin.readline()
                            if not line:
                                stdin_active=False
                                print('终端输入已结束；观测会话保持打开，Ctrl+C 或关闭 MuJoCo 窗口退出。',flush=True)
                                continue
                            line=line.strip()
                            if line in ('quit','exit'):
                                return
                            if line=='reset':
                                session.reset()
                                print('场景已重置，请输入新目标。',flush=True)
                                continue
                            if session.fault_reason is not None:
                                print('执行异常，会话已锁定；请 reset 显式重置或 quit 退出。',flush=True)
                                continue
                            try:
                                values=[float(v) for v in line.split()]
                                if len(values)!=2 or not np.isfinite(values).all() or np.any(values<TARGET_LOW) or np.any(values>TARGET_HIGH):
                                    raise ValueError()
                                goal=values
                                break
                            except ValueError:
                                print('请输入 .05≤shoulder≤.65、.55≤elbow≤1.15，例如 0.4 0.9',flush=True)
                        time.sleep(.02)
                    if goal is None:
                        break
                    task_index+=1
    except KeyboardInterrupt:
        print('会话已停止；日志和预测保留。')


if __name__=='__main__':
    main()
