"""Reproducible collect -> train/fine-tune -> calibrate -> evaluate commands."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from wmal.logging.manifest import atomic_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    collect=sub.add_parser('collect')
    collect.add_argument('--output',required=True)
    collect.add_argument('--task',default='stack_block')
    collect.add_argument('--episodes',type=int,default=40)
    collect.add_argument('--steps',type=int,default=16)
    collect.add_argument('--image-size',type=int,default=64)
    collect.add_argument('--seed',type=int,default=0)
    importer=sub.add_parser('import-lerobot')
    importer.add_argument('--source-manifest',required=True)
    importer.add_argument('--output',required=True)
    importer.add_argument('--action-schema',required=True)
    importer.add_argument('--image-size',type=int,default=64)
    importer.add_argument('--max-frames',type=int,default=64)
    importer.add_argument('--max-episodes',type=int)
    train=sub.add_parser('train')
    train.add_argument('--manifest',required=True)
    train.add_argument('--output',required=True)
    train.add_argument('--config',default='configs/training/visual_world.json')
    train.add_argument('--pretrained')
    train.add_argument('--freeze-encoder',action='store_true')
    train.add_argument('--device')
    train.add_argument('--epochs',type=int)
    for name in ('calibrate','evaluate'):
        command=sub.add_parser(name)
        command.add_argument('--manifest',required=True)
        command.add_argument('--checkpoint',required=True)
        command.add_argument('--output',required=True)
        command.add_argument('--horizon',type=int,default=4)
        command.add_argument('--device',default='cpu')
        if name=='calibrate':
            command.add_argument('--alpha',type=float,default=.2)
            command.add_argument('--std-floor',type=float,default=.05)
        else:
            command.add_argument('--calibration')
    args=parser.parse_args(argv)
    if args.command in ('calibrate','evaluate'):
        import torch
        torch.set_num_threads(1)
    if args.command=='collect':
        from wmal.training.visual_collector import collect_visual_workcell
        result={'manifest':str(collect_visual_workcell(args.output,task=args.task,episodes=args.episodes,
                    steps=args.steps,image_size=args.image_size,seed=args.seed))}
    elif args.command=='import-lerobot':
        from wmal.datasets.visual_lerobot import convert_lerobot_v21
        result={'manifest':str(convert_lerobot_v21(args.source_manifest,args.output,
                  action_schema=args.action_schema,image_size=args.image_size,
                  max_frames=args.max_frames,max_episodes=args.max_episodes))}
    elif args.command=='train':
        from wmal.training.visual_trainer import VisualTrainingConfig, train_visual
        values=json.loads(Path(args.config).read_text())
        for key in ('device','epochs'):
            if getattr(args,key) is not None:
                values[key]=getattr(args,key)
        report=train_visual(args.manifest,args.output,VisualTrainingConfig(**values),
                            pretrained=args.pretrained,freeze_encoder=args.freeze_encoder)
        result={key:report[key] for key in ('checkpoint','model_version','best_validation_loss','elapsed_s')}
    elif args.command=='calibrate':
        from wmal.models.horizon_calibration import calibrate_visual
        result=asdict(calibrate_visual(args.manifest,args.checkpoint,args.output,horizon=args.horizon,
                  alpha=args.alpha,std_floor=args.std_floor,device=args.device))
    else:
        from wmal.training.visual_trainer import evaluate_visual
        result=evaluate_visual(args.manifest,args.checkpoint,horizon=args.horizon,
                               device=args.device,calibration=args.calibration)
        atomic_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
