#!/usr/bin/env python3
"""G1 task graphs: fixed skills, learned consequence evaluation, real feedback."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import select
import sys
import time
import numpy as np
from wmal.agents.task_graph import TaskGraph
from wmal.agents.task_runtime import RuntimeConfig, TaskRuntime
from wmal.logging.manifest import atomic_json, build_manifest, sha256_file


class ShoulderPulse:
    """Explicit experimental intervention. No disturbance labels enter the Agent."""
    def __init__(self, session):
        self.session = session
        self.node_id = None
        self.first_hold_seen = False
        self.fired = False
        self.record = None

    def __getattr__(self, name):
        return getattr(self.session,name)

    def reset(self, targets=None):
        self.first_hold_seen, self.fired, self.record = False, False, None
        return self.session.reset(targets)

    def execute_actions(self, actions, **kwargs):
        eligible=self.node_id=='hold_A' and np.all(np.asarray(actions)==0)
        fire=eligible and self.first_hold_seen and not self.fired
        if eligible:
            self.first_hold_seen=True
        if not fire:
            return self.session.execute_actions(actions,**kwargs)
        session=self.session
        joint=session.mj.mj_name2id(session.model,session.mj.mjtObj.mjOBJ_JOINT,
                                  'g1_0_left_shoulder_pitch_joint')
        if joint<0:
            raise ValueError('Perturbation joint missing')
        dof=int(session.model.jnt_dofadr[joint])
        previous=float(session.data.qfrc_applied[dof])
        self.record={'kind':'shoulder-pulse','torque_nm':.5,'duration_s':session.period_s,
                     'before_step':session.step_id,'episode_id':session.episode_id}
        self.fired=True
        try:
            session.data.qfrc_applied[dof]=previous+.5
            return session.execute_actions(actions,**kwargs)
        finally:
            session.data.qfrc_applied[dof]=previous


def load_config(path, overrides):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict) or set(value)-{'scene','task','runtime'}:
        raise ValueError('Unknown experiment config fields')
    if not isinstance(value.get('scene'),str) or not value['scene']:
        raise ValueError('Scene is required')
    graph=TaskGraph.from_dict(value.get('task'))
    parameters=value.get('runtime',{})
    if not isinstance(parameters,dict):
        raise ValueError('Runtime configuration must be an object')
    parameters={**parameters,**{k:v for k,v in overrides.items() if v is not None}}
    try:
        config=RuntimeConfig(**parameters)
    except TypeError as exc:
        raise ValueError('Unknown runtime configuration fields') from exc
    return graph,config,value['scene']


def disturbance_summary(session, report, configured):
    return {'configured':configured,'injected':getattr(session,'fired',False),
            'record':getattr(session,'record',None),
            'recovery_triggered':report['state']['nodes'].get('hold_A',{}).get('recoveries',0)>0,
            'any_recovery':report['state']['recoveries']>0,
            'interpretation':'temporal association at hold_A, not a causal effect estimate'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--model-family',choices=('visual','embodied'),default='visual')
    parser.add_argument('--inference-url',help='Optional inference-only URL for embodied family')
    parser.add_argument('--calibration')
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--baseline',choices=('A0','A1','A2','A3'))
    parser.add_argument('--recovery',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--horizon',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--controller',choices=('legacy','gravity_compensated'),default='legacy',
                        help='Fixed low-level profile; A3 requires calibration bound to this controller')
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--monitor',action='store_true')
    parser.add_argument('--monitor-port',type=int,default=8766)
    parser.add_argument('--monitor-runs-root',default='runs')
    parser.add_argument('--realtime',action='store_true')
    parser.add_argument('--disturbance',choices=('none','shoulder-pulse'),default='none')
    args=parser.parse_args(argv)
    owned_provider=None
    try:
        graph,config,scene=load_config(args.config,{k:getattr(args,k) for k in ('baseline','recovery','horizon','seed')})
        if config.baseline=='A3' and not args.calibration:
            raise ValueError('A3 requires --calibration')
        import torch
        from wmal.models.visual_latent import VisualWorldModel
        from wmal.models.horizon_calibration import HorizonCalibration
        from wmal.envs.visual_workcell import VisualWorkcellSession
        torch.set_num_threads(1)
        if args.model_family=='embodied':
            from wmal.agents.embodied_provider import load_embodied_agent_provider
            model,calibration=load_embodied_agent_provider(args.checkpoint,args.calibration,
                device=args.device,remote_url=args.inference_url)
            owned_provider=model
        else:
            if args.inference_url:
                raise ValueError('Remote inference requires explicit embodied model family')
            model=VisualWorldModel.load(args.checkpoint,device=args.device)
            calibration=HorizonCalibration.load(args.calibration) if args.calibration else None
        if model.metadata.get('training_source',{}).get('task')!=scene:
            raise ValueError('Checkpoint scene scope differs; transfer requires separate validation')
        if config.horizon>model.metadata['max_horizon']:
            raise ValueError('Horizon exceeds trained model')
        if config.baseline=='A3':
            calibration.validate_for(model.version,model.semantics)
        output=Path(args.output).resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('Output must be new/empty; do not overwrite an experiment')
        output.mkdir(parents=True,exist_ok=True)
        inputs={'checkpoint':args.checkpoint,'config':args.config}
        if args.calibration:
            inputs['calibration']=args.calibration
        root=Path(__file__).resolve().parents[1]
        implementations={name:sha256_file(root/name) for name in (
            'src/wmal/agents/task_runtime.py','src/wmal/agents/task_execution.py',
            'src/wmal/agents/selection_policy.py','src/wmal/skills/research_contracts.py',
            'src/wmal/skills/task_verifier.py','src/wmal/envs/visual_workcell.py',
            'src/wmal/models/horizon_calibration.py','src/wmal/models/visual_latent.py',
            'src/wmal/models/categorical_rssm.py','scripts/g1_task_agent.py')}
        if args.model_family=='embodied':
            implementations.update({name:sha256_file(root/name) for name in (
                'src/wmal/agents/embodied_provider.py','src/wmal/agents/prediction_contracts.py',
                'src/wmal/models/embodied_world.py','src/wmal/models/embodied_calibration.py',
                'src/wmal/models/embodied_service.py')})
        with ExitStack() as stack:
            if args.model_family=='embodied':
                stack.callback(model.close)
            monitor=None
            if args.monitor:
                from wmal.monitor.runtime import MonitorRuntime
                monitor=stack.enter_context(MonitorRuntime(output,args.monitor_runs_root,args.monitor_port,
                                                           camera=model.semantics['camera']))
                print(f'Agent monitor: {monitor.url}',flush=True)
            physical=stack.enter_context(VisualWorkcellSession(scene,image_size=model.config.image_size,
                 camera=model.semantics['camera'],period_s=model.semantics['period_s'],seed=config.seed,
                 frame_publisher=monitor.publisher if monitor else None,realtime=args.realtime,
                 controller=args.controller))
            if config.baseline=='A3':
                calibration.validate_controller(physical.controller_profile)
            atomic_json(output/'run_manifest.json',build_manifest('g1_task_agent',config.seed,inputs,
                 parameters={**asdict(config),'scene':scene,'disturbance':args.disturbance,
                             'controller':args.controller,
                             **({'model_family':'embodied','prediction_samples':model.samples,
                                 'prediction_mode':'sample','inference_backend':
                                 'remote' if args.inference_url else 'local'} if args.model_family=='embodied' else {})},
                 graph=graph.to_dict(),graph_hash=graph.digest,implementation_hashes=implementations,
                 semantics=model.semantics,model_version=model.version,
                 controller_profile=physical.controller_profile,
                 checkpoint_training_controller=model.metadata.get('training_source',{}).get(
                     'controller_profile',{'profile':'g1_joint_pd_v1'}),
                 controller_transfer=physical.controller_profile!=model.metadata.get('training_source',{}).get(
                     'controller_profile',{'profile':'g1_joint_pd_v1'})))
            physical.reset(graph.start)
            session=ShoulderPulse(physical) if args.disturbance=='shoulder-pulse' else physical
            viewer=physical.open_viewer() if args.viewer else None
            with (output/'events.jsonl').open('w') as log:
                def emit(event,payload):
                    # Only test wrapper gets execution context; never return its hidden labels to Agent.
                    if event=='execution_started' and isinstance(session,ShoulderPulse):
                        session.node_id=payload.get('node_id')
                    log.write(json.dumps({'event':event,'wall_time_utc':datetime.now(timezone.utc).isoformat(),
                                         **payload},ensure_ascii=False,allow_nan=False)+'\n')
                    log.flush()
                task_index=0
                while True:
                    instance=replace(graph,task_id=f'{graph.task_id}_{task_index:03d}')
                    runtime=TaskRuntime(instance,session,model,calibration,config=config,emit=emit,artifact_dir=output)
                    report=runtime.run(cancel=(lambda: not viewer.is_running()) if viewer else None)
                    report['disturbance']=disturbance_summary(session,report,args.disturbance)
                    atomic_json(output/f'task_{task_index:03d}.json',report)
                    print(json.dumps({'task':instance.task_id,'status':report['state']['status'],
                        'completed':report['state']['completed_subgoals'],'actions':report['state']['executed_cycles'],
                        'failure_reason':report['state']['failure_reason']},ensure_ascii=False),flush=True)
                    if viewer is None and monitor is None:
                        break
                    print('任务已结束，窗口保持打开。输入 run 显式重置并开始新任务；quit 退出。',flush=True)
                    repeat=False
                    stdin_active=True
                    while viewer is None or viewer.is_running():
                        if viewer is not None:
                            physical.idle()
                        else:
                            physical.publish_monitor_frame()
                        if stdin_active and select.select([sys.stdin],[],[],.03)[0]:
                            line=sys.stdin.readline()
                            if not line:
                                stdin_active=False
                            elif line.strip()=='quit':
                                break
                            elif line.strip()=='run':
                                if args.model_family=='embodied':
                                    model.reset()
                                session.reset(graph.start)
                                task_index+=1
                                repeat=True
                                break
                        if not stdin_active:
                            time.sleep(.03)
                    if not repeat:
                        break
                emit('session_exit',{'task_id':instance.task_id,'reason':'closed'})
        return 0
    except (ValueError,TypeError,KeyError,OSError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        return 130
    finally:
        if owned_provider is not None:
            owned_provider.close()


if __name__=='__main__':
    raise SystemExit(main())
