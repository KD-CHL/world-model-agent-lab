"""Train/resume or explicitly evaluate a planning-compatible motion network."""
import argparse
import json
from pathlib import Path
from wmal.logging.manifest import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode',required=True)
    collect = sub.add_parser('collect')
    collect.add_argument('--output',required=True)
    collect.add_argument('--train-episodes',type=int,default=8)
    collect.add_argument('--validation-episodes',type=int,default=3)
    collect.add_argument('--test-episodes',type=int,default=3)
    collect.add_argument('--steps',type=int,default=100)
    collect.add_argument('--duration',type=float,default=.5)
    collect.add_argument('--seed',type=int,default=0)
    export = sub.add_parser('export-execution')
    export.add_argument('--logs',nargs='+',required=True)
    export.add_argument('--split-map',required=True,help='JSON mapping whole episode IDs to train/validation/test')
    export.add_argument('--source',choices=('mujoco_interaction',),required=True)
    export.add_argument('--output',required=True)
    train = sub.add_parser('train')
    train.add_argument('--dataset',required=True)
    train.add_argument('--checkpoint',required=True)
    train.add_argument('--config',default='configs/training/motion_network.json')
    train.add_argument('--resume')
    train.add_argument('--epochs',type=int,help='Total epochs, including already completed epochs')
    train.add_argument('--device')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--dataset',required=True)
    evaluate.add_argument('--checkpoint',required=True)
    evaluate.add_argument('--split',choices=('validation','test'),default='test')
    evaluate.add_argument('--horizon',type=int,default=3)
    evaluate.add_argument('--device',default='cpu')
    evaluate.add_argument('--output',required=True)
    args = parser.parse_args(argv)
    if args.mode == 'export-execution':
        from wmal.datasets.execution_export import export_execution_logs
        report = export_execution_logs(args.logs,args.output,json.loads(Path(args.split_map).read_text()),
                                       source=args.source)
        print(json.dumps(report))
        return 0
    if args.mode == 'collect':
        from wmal.training.motion_collector import collect_motion
        report = collect_motion(args.output,train_episodes=args.train_episodes,
            validation_episodes=args.validation_episodes,test_episodes=args.test_episodes,
            steps=args.steps,duration_s=args.duration,seed=args.seed,
            progress=lambda row:print(json.dumps(row),flush=True))
        print(json.dumps(report))
        return 0
    from wmal.training.motion_trainer import TrainingConfig, train_motion, evaluate_motion
    if args.mode == 'train':
        config = json.loads(Path(args.config).read_text())
        if args.epochs is not None: config['epochs'] = args.epochs
        if args.device is not None: config['device'] = args.device
        report = train_motion(args.dataset,args.checkpoint,TrainingConfig(**config),resume=args.resume,
                              progress=lambda row:print(json.dumps({'phase':'training',**row}),flush=True))
        print(json.dumps({key:report[key] for key in ('model_version','checkpoint','best_epoch','best_validation_loss')}))
    else:
        report = evaluate_motion(args.dataset,args.checkpoint,split=args.split,horizon=args.horizon,device=args.device)
        atomic_json(args.output,report)
        print(json.dumps(report,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
