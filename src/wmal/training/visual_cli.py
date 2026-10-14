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
    collect.add_argument('--controller',choices=('legacy','gravity_compensated'),default='legacy')
    collect.add_argument('--target-low',type=float,nargs=2,help='Declared local shoulder/elbow command lower bounds')
    collect.add_argument('--target-high',type=float,nargs=2,help='Declared local shoulder/elbow command upper bounds')
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
    train.add_argument('--no-progress',action='store_true',help='Disable terminal progress; JSON/report output remains')
    for name in ('calibrate','calibrate-object','evaluate'):
        command=sub.add_parser(name)
        command.add_argument('--manifest',required=True)
        command.add_argument('--checkpoint',required=True)
        command.add_argument('--output',required=True)
        command.add_argument('--horizon',type=int,default=4)
        command.add_argument('--device',default='cpu')
        if name in ('calibrate','evaluate'):
            command.add_argument('--allow-controller-transfer',action='store_true',
                                 help='Explicit frozen-weight evaluation or calibration on a different fixed controller')
        if name in ('calibrate','calibrate-object'):
            command.add_argument('--alpha',type=float,default=.2)
            command.add_argument('--std-floor',type=float,default=.05)
            command.add_argument('--window-stride',type=int,default=32,
                                 help='Calibrate on one window every N control steps per episode')
        else:
            command.add_argument('--calibration')
            command.add_argument('--object-calibration')
            command.add_argument('--split',choices=('validation','calibration','test'),default='test')
            command.add_argument('--window-stride',type=int,default=1,
                                 help='Evaluate one window every N control steps within each test episode')
    fit_localizer=sub.add_parser('fit-stack-localizer')
    fit_localizer.add_argument('--manifest',required=True)
    fit_localizer.add_argument('--output',required=True)
    fit_localizer.add_argument('--frame-stride',type=int,default=32)
    fit_localizer.add_argument('--tabletop-only',action='store_true',
                               help='Fit only stationary frames at the reset table height')
    eval_localizer=sub.add_parser('evaluate-stack-localizer')
    eval_localizer.add_argument('--manifest',required=True)
    eval_localizer.add_argument('--localizer',required=True)
    eval_localizer.add_argument('--output',required=True)
    eval_localizer.add_argument('--split',choices=('validation','calibration','test'),default='validation')
    eval_localizer.add_argument('--frame-stride',type=int,default=32)
    eval_localizer.add_argument('--tabletop-only',action='store_true',
                                help='Evaluate only stationary frames at the reset table height')
    fit_rgb_xy=sub.add_parser('fit-rgb-xy-localizer')
    fit_rgb_xy.add_argument('--manifest',required=True)
    fit_rgb_xy.add_argument('--output',required=True,help='New or empty checkpoint directory')
    fit_rgb_xy.add_argument('--epochs',type=int,default=20)
    fit_rgb_xy.add_argument('--batch-size',type=int,default=128)
    fit_rgb_xy.add_argument('--frame-stride',type=int,default=16)
    fit_rgb_xy.add_argument('--max-samples-per-episode',type=int,default=32)
    fit_rgb_xy.add_argument('--learning-rate',type=float,default=3e-4)
    fit_rgb_xy.add_argument('--seed',type=int,default=2411)
    fit_rgb_xy.add_argument('--device',default='cuda')
    calibrate_rgb_xy=sub.add_parser('calibrate-rgb-xy-localizer')
    calibrate_rgb_xy.add_argument('--manifest',required=True)
    calibrate_rgb_xy.add_argument('--checkpoint',required=True)
    calibrate_rgb_xy.add_argument('--output',required=True)
    calibrate_rgb_xy.add_argument('--alpha',type=float,default=.2)
    calibrate_rgb_xy.add_argument('--frame-stride',type=int,default=16)
    calibrate_rgb_xy.add_argument('--max-samples-per-episode',type=int,default=32)
    calibrate_rgb_xy.add_argument('--batch-size',type=int,default=128)
    calibrate_rgb_xy.add_argument('--device',default='cuda')
    eval_rgb_xy=sub.add_parser('evaluate-rgb-xy-localizer')
    eval_rgb_xy.add_argument('--manifest',required=True)
    eval_rgb_xy.add_argument('--checkpoint',required=True)
    eval_rgb_xy.add_argument('--output',required=True)
    eval_rgb_xy.add_argument('--split',choices=('validation','calibration','test'),default='validation')
    eval_rgb_xy.add_argument('--frame-stride',type=int,default=16)
    eval_rgb_xy.add_argument('--max-samples-per-episode',type=int,default=32)
    eval_rgb_xy.add_argument('--batch-size',type=int,default=128)
    eval_rgb_xy.add_argument('--device',default='cuda')
    eval_rgb_xy.add_argument('--calibration',help='Frozen episode-level localizer radius JSON')
    args=parser.parse_args(argv)
    if args.command in ('calibrate','calibrate-object','evaluate',
                        'fit-rgb-xy-localizer','calibrate-rgb-xy-localizer',
                        'evaluate-rgb-xy-localizer'):
        import torch
        torch.set_num_threads(1)
    if args.command=='collect':
        from wmal.training.visual_collector import collect_visual_workcell
        result={'manifest':str(collect_visual_workcell(args.output,task=args.task,episodes=args.episodes,
                    steps=args.steps,image_size=args.image_size,seed=args.seed,controller=args.controller,
                    target_low=args.target_low,target_high=args.target_high))}
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
                            pretrained=args.pretrained,freeze_encoder=args.freeze_encoder,
                            show_progress=not args.no_progress)
        result={key:report[key] for key in ('checkpoint','model_version','best_validation_loss','elapsed_s')}
    elif args.command=='fit-stack-localizer':
        from wmal.perception.stack_block_localizer import fit_stack_block_localizer
        result=asdict(fit_stack_block_localizer(args.manifest,args.output,
                                                 frame_stride=args.frame_stride,
                                                 tabletop_only=args.tabletop_only))
    elif args.command=='evaluate-stack-localizer':
        from wmal.perception.stack_block_localizer import evaluate_stack_block_localizer
        result=evaluate_stack_block_localizer(args.manifest,args.localizer,split=args.split,
                                              frame_stride=args.frame_stride,
                                              tabletop_only=args.tabletop_only)
        atomic_json(args.output,result)
    elif args.command=='fit-rgb-xy-localizer':
        from wmal.perception.rgb_xy_regressor import train_rgb_xy_localizer
        result=train_rgb_xy_localizer(args.manifest,args.output,epochs=args.epochs,
            batch_size=args.batch_size,frame_stride=args.frame_stride,
            max_samples_per_episode=args.max_samples_per_episode,
            learning_rate=args.learning_rate,seed=args.seed,device=args.device)
    elif args.command=='calibrate-rgb-xy-localizer':
        from wmal.perception.rgb_xy_regressor import calibrate_rgb_xy_localizer
        result=calibrate_rgb_xy_localizer(args.manifest,args.checkpoint,args.output,
            alpha=args.alpha,frame_stride=args.frame_stride,
            max_samples_per_episode=args.max_samples_per_episode,
            batch_size=args.batch_size,device=args.device)
    elif args.command=='evaluate-rgb-xy-localizer':
        from wmal.perception.rgb_xy_regressor import evaluate_rgb_xy_localizer
        result=evaluate_rgb_xy_localizer(args.manifest,args.checkpoint,args.output,
            split=args.split,frame_stride=args.frame_stride,
            max_samples_per_episode=args.max_samples_per_episode,
            batch_size=args.batch_size,device=args.device,calibration_path=args.calibration)
    elif args.command=='calibrate':
        from wmal.models.horizon_calibration import calibrate_visual
        result=asdict(calibrate_visual(args.manifest,args.checkpoint,args.output,horizon=args.horizon,
                  alpha=args.alpha,std_floor=args.std_floor,device=args.device,
                  window_stride=args.window_stride,
                  allow_controller_transfer=args.allow_controller_transfer))
    elif args.command=='calibrate-object':
        from wmal.models.horizon_calibration import calibrate_object_visual
        result=asdict(calibrate_object_visual(args.manifest,args.checkpoint,args.output,horizon=args.horizon,
                  alpha=args.alpha,std_floor=args.std_floor,device=args.device,
                  window_stride=args.window_stride))
    else:
        from wmal.training.visual_trainer import evaluate_visual
        result=evaluate_visual(args.manifest,args.checkpoint,horizon=args.horizon,
                               device=args.device,calibration=args.calibration,
                               object_calibration=args.object_calibration,
                               window_stride=args.window_stride,split=args.split,
                               allow_controller_transfer=args.allow_controller_transfer)
        atomic_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
