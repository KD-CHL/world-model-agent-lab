"""Independent embodied CLI; network JSON and training JSON are distinct inputs."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import torch

from wmal.datasets.embodied_sequences import EmbodiedDataset, from_visual_manifest
from wmal.logging.manifest import atomic_json
from wmal.models.embodied_calibration import EmbodiedCalibration
from wmal.models.embodied_config import ScalableConfig
from wmal.models.embodied_rssm import ScalableRSSMMember, parameter_report
from wmal.models.embodied_world import EmbodiedWorldModel
from .embodied_distributed import DistributedRuntime, optimizer_update
from .embodied_checkpoint import deterministic_execution, environment_identity
from .embodied_loss import EmbodiedLossForward
from .embodied_trainer import (EmbodiedTrainingConfig, EpisodeSampler, hardware_report,
    resolve_config, train_embodied, _new_member)


def _json(path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError('Configuration JSON must be an object')
    return payload


def _manifest(args):
    physical, commands, rule = args.legacy_physical_indices, args.legacy_command_indices, args.legacy_command_rule
    if any(value is not None for value in (physical, commands, rule)):
        if any(value is None for value in (physical, commands, rule)):
            raise ValueError('Legacy adaptation requires physical/command indices and command rule explicitly')
        indices = lambda value: [int(x) for x in value.split(',')] if value else []
        return from_visual_manifest(args.manifest, physical_indices=indices(physical), command_indices=indices(commands),
            command_rule=rule, profile=_json(args.profile) if args.profile else None)
    if args.profile:
        raise ValueError('--profile is only for explicit legacy mapping; native manifests contain profiles')
    return args.manifest


def _config(args):
    values = _json(args.training_config) if args.training_config else {}
    if args.network_config:
        if values.get('model') is not None:
            raise ValueError('Use model in training config or --network-config, not both')
        network = ScalableConfig(**_json(args.network_config))
        values['model'] = network
        # Standalone network files bind H/K; the training JSON still owns T and updates.
        values.setdefault('horizon', network.horizon)
        values.setdefault('context_steps', network.context_steps)
    for name in ('device', 'distributed', 'epochs', 'members', 'batch_size', 'accumulation', 'precision'):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    try:
        return EmbodiedTrainingConfig(**values)
    except TypeError as exc:
        raise ValueError(f'Unknown training configuration field (network files use --network-config): {exc}') from exc


def inspect_manifest(manifest):
    # Construction checks metadata/file hashes, but reads no episode arrays.
    dataset = EmbodiedDataset(manifest, 'train', horizon=1)
    splits = {}
    for split in ('train', 'validation', 'calibration', 'test'):
        rows = [row for row in dataset.manifest['episodes'] if row['split'] == split]
        splits[split] = dict(episodes=len(rows), transitions=sum(row['steps'] for row in rows),
                            observations=sum(row['steps']+1 for row in rows))
    return dict(schema=dataset.manifest['schema'], profile=dataset.profile.to_dict(),
                source=dataset.manifest['source'], splits=splits,
                train_original_groups=len(dataset.episode_window_indices), arrays_read=False)


@deterministic_execution()
def profile_embodied(manifest, output, config):
    """Measure real complete accumulation update + M/S8 provider wall time, unchanged sizes."""
    runtime = DistributedRuntime(config.distributed, config.device)
    output = Path(output).resolve()
    try:
        def new_output():
            if output.exists():
                raise ValueError('Profile output must be new')
        runtime.rank_zero(new_output)
        data = EmbodiedDataset(manifest, 'train', horizon=config.sequence_length-1)
        config = resolve_config(config, data.profile)
        capacity = parameter_report(config.model)
        if config.model.preset != 'smoke' and not capacity['within_budget']:
            raise ValueError('Formal network preset is outside its declared parameter budget')
        norm = data.normalization()
        torch.manual_seed(config.seed)
        sampler = EpisodeSampler(data, config, config.seed)
        if runtime.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(runtime.device)
        member = ScalableRSSMMember(config.model).to(runtime.device)
        if config.freeze_encoder:
            from .embodied_trainer import FROZEN_ENCODERS
            for name, parameter in member.named_parameters():
                if name.startswith(FROZEN_ENCODERS):
                    parameter.requires_grad_(False)
        wrapped = runtime.wrap(EmbodiedLossForward(member, norm, config))
        optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=config.lr)
        scaler = torch.amp.GradScaler('cuda', enabled=config.precision == 'float16')
        group = []
        for _ in range(config.accumulation):
            draws = sampler.draw(config.batch_size*runtime.world_size)
            batch = []
            for draw in draws[runtime.rank*config.batch_size:(runtime.rank+1)*config.batch_size]:
                sample = data[draw['index']]; sample.update(origin=draw['origin'], seed=draw['seed'])
                batch.append(sample)
            group.append(batch)
        if runtime.device.type == 'cuda':
            torch.cuda.synchronize(runtime.device)
        start = time.perf_counter()
        optimizer_update(wrapped, optimizer, group, wrapped, scaler=scaler)
        if runtime.device.type == 'cuda':
            torch.cuda.synchronize(runtime.device)
        update_seconds = time.perf_counter()-start
        weights, _ = runtime.snapshot(wrapped, member, optimizer)
        # Actual configured M independent parameter objects for inference. Other
        # members are random probes, not trained-capability or selection results.
        members = [_new_member(config.model, weights)]
        for _ in range(1, config.members):
            members.append(ScalableRSSMMember(config.model))
        model = EmbodiedWorldModel(config.model, data.profile, norm, members,
            metadata={'max_horizon': config.horizon}, device=runtime.device)
        inference_data = EmbodiedDataset(manifest, 'train', horizon=config.horizon, context_steps=config.context_steps)
        sample = inference_data[min(config.context_steps, len(inference_data)-1)]
        if runtime.device.type == 'cuda':
            torch.cuda.synchronize(runtime.device)
        start = time.perf_counter()
        prediction = model.predict_sample(sample, samples=8, mode='sample', seed=0)
        if runtime.device.type == 'cuda':
            torch.cuda.synchronize(runtime.device)
        inference_seconds = time.perf_counter()-start  # includes assert_frozen, validation, hashes and copies
        measured = runtime.gather(dict(update_seconds=update_seconds, inference_seconds=inference_seconds,
                                       environment=environment_identity(runtime.device),
                                       resources=hardware_report(runtime.device)))
        report = dict(schema='wmal.embodied_profile.v1', config=asdict(config), world_size=runtime.world_size,
            optimizer_steps=1, update_seconds=max(row['update_seconds'] for row in measured),
            inference_seconds=max(row['inference_seconds'] for row in measured),
            inference_samples=8, inference_members=config.members,
            inference_shape=list(prediction['physical_states'].shape), ranks=measured,
            actual_effective_batch=config.batch_size*config.accumulation*runtime.world_size,
            sequence_length=config.sequence_length, actions=config.sequence_length-1,
            provider_validation_and_hash_included=True, measured=True, gpu8_tested=False,
            limitations=['One local probe, not trained-capability or server-throughput evidence',
                         'No automatic batch, sequence, network or sample-count reduction'])
        report['samples_per_second'] = report['actual_effective_batch']/report['update_seconds']
        report['parameter_counts'] = dict(per_member=sum(p.numel() for p in model.members[0].parameters()),
            independent_members=len(model.members), total=sum(p.numel() for m in model.members for p in m.parameters()),
            precision_master_parameters='float32', capacity_claim='smoke only' if config.model.preset == 'smoke' else config.model.preset)
        report['budget_status'] = ('not_applicable_cpu' if runtime.device.type == 'cpu' else
            ('within_80_percent' if all(row['resources']['within_budget'] for row in measured) else 'exceeded_80_percent'))
        runtime.rank_zero(lambda: atomic_json(output, report))
        if report['budget_status'] == 'exceeded_80_percent':
            raise RuntimeError('Measured allocated/reserved exceeds 80% reported hardware budget; see profile report')
        return report
    finally:
        runtime.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent embodied world model training and probes')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'train', 'calibrate', 'profile', 'evaluate'):
        command = sub.add_parser(name)
        command.add_argument('--manifest', required=True)
        command.add_argument('--legacy-physical-indices')
        command.add_argument('--legacy-command-indices')
        command.add_argument('--legacy-command-rule', choices=['joint_delta', 'none'])
        command.add_argument('--profile', help='Explicit legacy observation profile JSON')
        if name != 'inspect':
            command.add_argument('--output', required=True)
        if name in ('train', 'profile'):
            command.add_argument('--training-config', help='Training protocol JSON; fields of EmbodiedTrainingConfig')
            command.add_argument('--network-config', help='Network-only ScalableConfig JSON')
            command.add_argument('--device')
            command.add_argument('--distributed', choices=['none', 'ddp', 'fsdp'])
            command.add_argument('--precision', choices=['float32', 'float16', 'bfloat16'])
            for field in ('epochs', 'members', 'batch-size', 'accumulation'):
                command.add_argument('--'+field, type=int)
        if name == 'train':
            command.add_argument('--pretrained'); command.add_argument('--resume')
            command.add_argument('--no-progress', action='store_true')
        if name in ('calibrate', 'evaluate'):
            command.add_argument('--model', required=True)
            command.add_argument('--device', default='cpu')
        if name == 'evaluate':
            command.add_argument('--calibration')
            command.add_argument('--horizon', type=int)
        if name == 'calibrate':
            command.add_argument('--alpha', type=float, default=.2)
            command.add_argument('--samples', type=int, default=8)
            command.add_argument('--mode', choices=['sample', 'soft'], default='sample')
            command.add_argument('--scope', help='Raw-coordinate calibration scope JSON')
    args = parser.parse_args(argv)
    try:
        manifest = _manifest(args)
        if args.command == 'inspect':
            result = inspect_manifest(manifest)
        elif args.command == 'train':
            result = train_embodied(manifest, args.output, _config(args), pretrained=args.pretrained,
                                   resume=args.resume, show_progress=not args.no_progress)
        elif args.command == 'profile':
            result = profile_embodied(manifest, args.output, _config(args))
        elif args.command == 'evaluate':
            from .embodied_evaluation import evaluate_embodied
            output = Path(args.output)
            if output.exists():
                raise ValueError('Evaluation output must be new')
            result = evaluate_embodied(manifest, args.model, calibration=args.calibration,
                                      horizon=args.horizon, device=args.device)
            atomic_json(output, result)
        else:
            output = Path(args.output)
            if output.exists():
                raise ValueError('Calibration output must be new')
            model = EmbodiedWorldModel.load(args.model, device=args.device)
            data = EmbodiedDataset(manifest, 'calibration', horizon=model.config.horizon,
                                   context_steps=model.config.context_steps)
            calibration = EmbodiedCalibration.fit(data, model, alpha=args.alpha, samples=args.samples,
                mode=args.mode, scope=_json(args.scope) if args.scope else model.metadata.get('prediction_scope'))
            calibration.save(output)
            result = dict(output=str(output), version=calibration.version, episodes=calibration.episode_count)
        if int(os.environ.get('RANK', '0')) == 0:
            print(json.dumps(result, allow_nan=False))
    except (ValueError, RuntimeError, NotImplementedError, TypeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0
