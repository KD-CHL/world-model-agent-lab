"""Independent sequential RSSM training, detached selection and exact replay."""
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from wmal.agents.prediction_contracts import content_hash, validate_scope
from wmal.datasets.embodied_sequences import EmbodiedDataset, validate_lineage
from wmal.logging.manifest import atomic_json, sha256_file
from wmal.models.embodied_config import ScalableConfig, network_preset
from wmal.models.embodied_rssm import ScalableRSSMMember, parameter_report
from wmal.models.embodied_world import EmbodiedWorldModel, trajectory_seed
from .embodied_checkpoint import (capture_rng, deterministic_execution, environment_identity, load_training_checkpoint,
                                  restore_rng, save_training_checkpoint)
from .embodied_distributed import DistributedRuntime, optimizer_update, unique_validation_metrics
from .embodied_loss import EmbodiedLossForward, imagine_sample, prepare_sample
from .progress import epoch_status, progress_bar, training_status

SELECTION_METRIC = 'mean_per_horizon_dimension_normalized_rmse'
FROZEN_ENCODERS = ('encoder.', 'physical_encoder.', 'command_encoder.', 'fusion.')


@dataclass(frozen=True)
class EmbodiedTrainingConfig:
    model: ScalableConfig | dict | None = None
    model_preset: str = 'smoke'
    epochs: int = 1
    members: int = 3
    batch_size: int = 2
    accumulation: int = 4
    lr: float = 3e-4
    seed: int = 0
    device: str = 'cpu'
    precision: str = 'float32'
    sequence_length: int = 32
    horizon: int = 8
    context_steps: int = 8
    distributed: str = 'none'
    state_weight: float = 1.
    frame_weight: float = 1.
    change_weight: float = 1.
    reconstruction_weight: float = 1.
    dyn_weight: float = .5
    rep_weight: float = .1
    event_weight: float = 1.
    free_nats: float = 1.
    change_threshold: float = 1/255
    mode: str = 'sample'
    sample_count: int = 8
    freeze_encoder: bool = False
    task_sampling: dict | None = None
    command_rule: str = 'joint_delta'

    def __post_init__(self):
        if isinstance(self.model, dict):
            object.__setattr__(self, 'model', ScalableConfig(**self.model))
        if self.model is not None and not isinstance(self.model, ScalableConfig):
            raise ValueError('model must be a ScalableConfig/network dictionary')
        for name in ('epochs', 'members', 'batch_size', 'accumulation', 'sequence_length', 'horizon', 'sample_count'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.context_steps) is not int or self.context_steps < 0:
            raise ValueError('context_steps must be nonnegative')
        if self.sequence_length <= self.horizon:
            raise ValueError('sequence_length must contain origin and entire horizon')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be in [0,2**63)')
        for name in ('lr', 'state_weight', 'frame_weight', 'change_weight', 'reconstruction_weight',
                     'dyn_weight', 'rep_weight', 'event_weight', 'free_nats', 'change_threshold'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.lr == 0 or self.change_threshold > 1:
            raise ValueError('lr must be positive; change_threshold must be in [0,1]')
        if self.distributed not in ('none', 'ddp', 'fsdp') or self.precision not in ('float32', 'float16', 'bfloat16'):
            raise ValueError('Unknown distributed/precision protocol')
        if self.mode not in ('sample', 'soft') or type(self.freeze_encoder) is not bool:
            raise ValueError('Invalid sampling/freeze protocol')
        if self.command_rule not in ('joint_delta', 'none'):
            raise ValueError('Unknown command_rule')
        if self.task_sampling is not None:
            if not isinstance(self.task_sampling, dict) or not self.task_sampling or any(
                    type(k) is not str or not k or type(v) not in (int, float) or
                    not math.isfinite(v) or v <= 0 for k, v in self.task_sampling.items()):
                raise ValueError('task_sampling must map recorded task names to positive weights')


def resolve_config(config, profile):
    if not isinstance(config, EmbodiedTrainingConfig):
        raise ValueError('Expected EmbodiedTrainingConfig')
    dimensions = dict(physical_dim=len(profile.physical_order), command_dim=len(profile.command_order),
        action_dim=len(profile.action_order), views=len(profile.camera_ids), image_size=profile.image_size,
        event_dim=len(profile.event_names), context_steps=config.context_steps, horizon=config.horizon)
    model = config.model or network_preset(config.model_preset, **dimensions)
    if any(getattr(model, name) != value for name, value in dimensions.items()):
        raise ValueError('Network dimensions/context/horizon differ from training/data profile')
    if profile.command_order and profile.command_rule != 'joint_delta':
        raise ValueError('Cannot train action-only priors with unknown future commands')
    if config.precision == 'float16' and torch.device(config.device).type != 'cuda':
        raise ValueError('float16 training requires CUDA GradScaler')
    return replace(config, model=model, command_rule=profile.command_rule)


class EpisodeSampler:
    """Same original-group bootstrap on every rank, then global draw and rank shard."""
    def __init__(self, dataset, config, seed):
        self.groups = dataset.episode_window_indices
        self.random = random.Random(seed)
        self.bootstrap = self.random.choices(sorted(self.groups), k=len(self.groups))
        self.position = 0
        self.config = config
        self.tasks = {}
        by_episode = {row['episode_id']: row.get('task', dataset.manifest['source'].get('task')) for row in dataset.rows}
        for group, indices in self.groups.items():
            task_names = {by_episode[dataset.rows[dataset.index[index][0]]['episode_id']] for index in indices}
            if len(task_names) != 1:
                raise ValueError('Original group has conflicting task identities')
            self.tasks[group] = task_names.pop()
        if config.task_sampling and set(config.task_sampling) - set(self.tasks.values()):
            raise ValueError('task_sampling refers to absent recorded tasks')
        if config.task_sampling:
            # Stratify bootstrap only for explicit task weighting so a requested
            # task cannot disappear due to a small global bootstrap draw.
            self.bootstrap = []
            for task in sorted(set(self.tasks.values()), key=str):
                population = sorted(group for group in self.groups if self.tasks[group] == task)
                self.bootstrap.extend(self.random.choices(population, k=len(population)))

    def draw(self, count):
        result = []
        for _ in range(count):
            candidates = self.bootstrap
            if self.config.task_sampling:
                task = self.random.choices(list(self.config.task_sampling), weights=list(self.config.task_sampling.values()))[0]
                candidates = [key for key in candidates if self.tasks[key] == task]
                if not candidates:
                    raise ValueError('Bootstrap omitted a requested task; choose a different seed')
            group = self.random.choice(candidates)
            result.append(dict(index=self.random.choice(self.groups[group]),
                origin=self.random.randrange(self.config.sequence_length-self.config.horizon),
                seed=self.random.randrange(2**63), original_group=group))
            self.position += 1
        return result

    def state_dict(self):
        return dict(bootstrap=list(self.bootstrap), position=self.position, rng=self.random.getstate())

    def load_state_dict(self, state):
        if len(state['bootstrap']) != len(self.groups) or set(state['bootstrap']) - self.groups.keys():
            raise ValueError('Sampler origin groups changed')
        self.bootstrap, self.position = list(state['bootstrap']), state['position']
        self.random.setstate(state['rng'])


def _new_member(config, weights=None, device='cpu'):
    # Export/validation clones never consume training RNG or share optimizer storage.
    with torch.random.fork_rng(devices=[]):
        member = ScalableRSSMMember(config)
    if weights is not None:
        member.load_state_dict(weights, strict=True)
    return member.to(device=device, dtype=torch.float32)


def validate_member(member, dataset, normalization, config, runtime, member_index):
    records, baseline = [], []
    member.eval()
    with torch.inference_mode():
        for index in range(runtime.rank, len(dataset), runtime.world_size):
            sample = dataset[index]
            values = prepare_sample(sample, normalization, config, runtime.device)
            predicted = [imagine_sample(member, values, config,
                seed=trajectory_seed(0, member_index, i))['physical'].float() for i in range(config.sample_count)]
            origin = values['origin']
            target = values['physical'][:, origin+1:origin+config.horizon+1]
            mean = torch.stack(predicted).mean(0)
            records.append((sample['sample_id'], (mean-target).square()[0].cpu().tolist()))
            baseline.append((sample['sample_id'], (values['physical'][:, origin:origin+1]-target).square()[0].cpu().tolist()))
    combined = [row for rank in runtime.gather(records) for row in rank]
    result = unique_validation_metrics(combined, normalization['physical_scale'])
    baselines = [row for rank in runtime.gather(baseline) for row in rank]
    result['baseline_rmse'] = unique_validation_metrics(baselines, normalization['physical_scale'])['physical_rmse']
    member.train()
    return result


def hardware_report(device):
    result = dict(device=str(device), torch=str(torch.__version__), cuda=torch.version.cuda,
                  allocated_peak_bytes=None, reserved_peak_bytes=None, total_memory_bytes=None,
                  budget_fraction=.8, within_budget=None, gpu8_tested=False)
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
        info = torch.cuda.get_device_properties(device)
        allocated, reserved = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
        result.update(name=info.name, total_memory_bytes=info.total_memory,
            allocated_peak_bytes=allocated, reserved_peak_bytes=reserved,
            allocated_fraction=allocated/info.total_memory, reserved_fraction=reserved/info.total_memory,
            within_budget=allocated <= .8*info.total_memory and reserved <= .8*info.total_memory)
    return result


def _output_directory(output, resume):
    if resume is None:
        if output.exists():
            raise ValueError('Output must be new unless explicitly resuming this same run')
        output.mkdir(parents=True)
    elif not output.is_dir() or Path(resume).resolve() != output / 'training.pt':
        raise ValueError('Exact resume requires the same run output/training.pt')


def _export(run, config, dataset, normalization, output):
    metadata = dict(episode_lineage=run['lineage'],
        event_supervised_by_member=[row['selected_event_seen'] for row in run['members']],
        max_horizon=config.horizon, training_source=dataset.manifest['source'],
        prediction_scope=dataset.manifest['source'].get('prediction_scope', dataset.manifest['source'].get('scope')),
        training_config=asdict(config), parent_model_version=run['parent_model_version'],
        selection_metric=SELECTION_METRIC, selected_epochs=[row['selected_epoch'] for row in run['members']],
        training_sampling=dict(mode=config.mode, samples=config.sample_count,
            origin_policy='uniform valid origin in T; bounded real K prefix',
            sequence_length=config.sequence_length, change_threshold=config.change_threshold),
        transfer=run['transfer'])
    members = [_new_member(config.model, row['weights']) for row in run['members']]
    model = EmbodiedWorldModel(config.model, dataset.profile, normalization, members, metadata=metadata)
    model.save(output / 'model.pt')
    return model.version


@deterministic_execution()
def train_embodied(manifest, output, config, *, pretrained=None, resume=None, show_progress=True):
    """Train independent members sequentially; no calibration or test arrays read.

    `output` is a new run directory. Exact resume accepts only its training.pt.
    Pretrained accepts the new provider artifact with exactly matching network and
    robot semantics; coordinates and recorded exposure are inherited, optimizer reset.
    """
    if pretrained is not None and resume is not None:
        raise ValueError('pretrained and exact resume are mutually exclusive')
    runtime = DistributedRuntime(config.distributed, config.device)
    output = Path(output).resolve()
    started = time.perf_counter()
    try:
        train = EmbodiedDataset(manifest, 'train', horizon=config.sequence_length-1, context_steps=0)
        config = resolve_config(config, train.profile)
        capacity = parameter_report(config.model)
        if config.model.preset != 'smoke' and not capacity['within_budget']:
            raise ValueError('Formal network preset is outside its declared parameter budget')
        validate_scope(train.manifest['source'].get('prediction_scope', train.manifest['source'].get('scope')), train.profile)
        validation = EmbodiedDataset(manifest, 'validation', horizon=config.horizon, context_steps=config.context_steps)
        if runtime.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(runtime.device)
        identity = dict(config=asdict(config), data=content_hash(train.manifest),
            manifest_sha256=sha256_file(train.path), profile=train.profile.to_dict(),
            world_size=runtime.world_size, environments=runtime.gather(environment_identity(runtime.device)),
            output=str(output))
        identity['code_sha256'] = {str(file.relative_to(Path(__file__).parents[1])): sha256_file(file)
            for file in [Path(__file__), Path(__file__).with_name('embodied_checkpoint.py'),
                Path(__file__).with_name('embodied_distributed.py'), Path(__file__).with_name('embodied_loss.py'),
                Path(__file__).parents[1] / 'models/embodied_rssm.py',
                Path(__file__).parents[1] / 'models/embodied_blocks.py',
                Path(__file__).parents[1] / 'models/embodied_world.py',
                Path(__file__).parents[1] / 'models/categorical_rssm.py',
                Path(__file__).parents[1] / 'datasets/embodied_sequences.py',
                Path(__file__).parents[1] / 'datasets/embodied_profiles.py',
                Path(__file__).parents[1] / 'agents/prediction_contracts.py']}
        snapshot = load_training_checkpoint(resume, identity) if resume else None
        if snapshot:
            restore_rng(snapshot['rank_states'][runtime.rank]['rng'], runtime.device)
        parent = EmbodiedWorldModel.load(pretrained) if pretrained is not None else None
        if parent:
            if parent.config != config.model or len(parent.members) != config.members:
                raise ValueError('Pretrained requires an exactly matching architecture and member count')
            parent.validate_profile(train.profile)
            if 'episode_lineage' not in parent.metadata:
                raise ValueError('Unknown pretrained episode lineage refuses fine-tuning')
        lineage = validate_lineage(train.manifest['episodes'],
            parent.metadata['episode_lineage'] if parent else None, context='fine-tune' if parent else 'training')
        norm = snapshot['normalization'] if snapshot else (parent.normalization if parent else train.normalization())
        runtime.rank_zero(lambda: _output_directory(output, resume))
        run = snapshot['run'] if snapshot else dict(members=[], current=None, lineage=lineage,
            parent_model_version=parent.version if parent else None,
            initial_members=[dict(weights={key: value.detach().cpu().clone() for key, value in member.state_dict().items()},
                event_seen=parent.metadata.get('event_supervised_by_member',
                    [[False]*config.model.event_dim for _ in parent.members])[index])
                for index, member in enumerate(parent.members)] if parent else [None]*config.members,
            transfer=dict(policy='exact network and semantic match; all layers' if parent else 'random initialization',
                          matching_layers=list(parent.members[0].state_dict()) if parent else [],
                          random_layers=[] if parent else ['all']))
        del parent
        position = snapshot['position'] if snapshot else dict(member=0, epoch=0, batch=0, step=0)
        microbatches = math.ceil(len(train)/(config.batch_size*runtime.world_size))
        enabled = bool(show_progress and runtime.rank == 0)
        training_status(f'Independent RSSM: {config.members} members, real T{config.sequence_length}, '
                        f'H{config.horizon}/K{config.context_steps}, {runtime.world_size} rank(s)', enabled=enabled)
        for member_index in range(position['member'], config.members):
            member_seed = (config.seed + member_index*1_000_003) % 2**63
            random.seed(member_seed); np.random.seed(member_seed % 2**32); torch.manual_seed(member_seed)
            member = ScalableRSSMMember(config.model).to(device=runtime.device, dtype=torch.float32)
            initial = run['initial_members'][member_index]
            if initial is not None:
                member.load_state_dict(initial['weights'], strict=True)
            continuing = (snapshot is not None and run['current'] is not None and
                          member_index == snapshot['position']['member'])
            if continuing:
                member.load_state_dict(snapshot['model'], strict=True)
            if config.freeze_encoder:
                for name, parameter in member.named_parameters():
                    if name.startswith(FROZEN_ENCODERS):
                        parameter.requires_grad_(False)
            frozen = {name: value.detach().cpu().clone() for name, value in member.state_dict().items()
                      if config.freeze_encoder and name.startswith(FROZEN_ENCODERS)}
            count = sum(p.numel() for p in member.parameters())
            trainable = sum(p.numel() for p in member.parameters() if p.requires_grad)
            wrapped = runtime.wrap(EmbodiedLossForward(member, norm, config))
            optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=config.lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
            scaler = torch.amp.GradScaler('cuda', enabled=config.precision == 'float16')
            sampler = EpisodeSampler(train, config, member_seed)
            if run['current'] is None:
                inherited_events = initial['event_seen'] if initial is not None else [False]*config.model.event_dim
                run['current'] = dict(history=[], selected_epoch=None, best_metric=None, weights=None,
                    event_seen=list(inherited_events), new_event_seen=[False]*config.model.event_dim,
                    effective_batch_sizes=[], optimizer_steps=0,
                    total_parameters=count, trainable_parameters=trainable, frozen_parameters=count-trainable,
                    exposures=[], epoch_metrics=[], bootstrap=list(sampler.bootstrap))
            current = run['current']
            run['initial_members'][member_index] = None
            del initial
            if continuing:
                runtime.restore_optimizer(wrapped, optimizer, snapshot['optimizer'])
                scheduler.load_state_dict(snapshot['scheduler']); scaler.load_state_dict(snapshot['scaler'])
                rank_state = snapshot['rank_states'][runtime.rank]
                sampler.load_state_dict(rank_state['sampler']); restore_rng(rank_state['rng'], runtime.device)
            else:
                position.update(member=member_index, epoch=0, batch=0)
            def save_boundary():
                model_state, optim_state = runtime.snapshot(wrapped, member, optimizer)
                rank_states = runtime.gather(dict(rng=capture_rng(runtime.device), sampler=sampler.state_dict()))
                payload = dict(identity=identity, model=model_state, optimizer=optim_state,
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rank_states=rank_states,
                    position=dict(position), run=run, normalization=norm)
                # All ranks call the writer (which writes on rank zero), and surface failures together.
                error = None
                try:
                    save_training_checkpoint(output / 'training.pt', payload)
                except Exception as exc:
                    error = exc
                errors = runtime.gather(None if error is None else f'{type(error).__name__}: {error}')
                if any(errors):
                    if error is not None:
                        raise error
                    raise RuntimeError(f'Checkpoint failed on peer: {errors}')
                return model_state
            for epoch in range(position['epoch'], config.epochs):
                start_batch = position['batch'] if position['epoch'] == epoch else 0
                bar = progress_bar(microbatches, f'Member {member_index+1} epoch {epoch+1}',
                                   initial=start_batch, enabled=enabled)
                try:
                    for first in range(start_batch, microbatches, config.accumulation):
                        group = []
                        group_length = min(config.accumulation, microbatches-first)
                        for _ in range(group_length):
                            drawn = sampler.draw(config.batch_size*runtime.world_size)
                            local = drawn[runtime.rank*config.batch_size:(runtime.rank+1)*config.batch_size]
                            batch = []
                            for draw in local:
                                sample = train[draw['index']]
                                sample.update(origin=draw['origin'], seed=draw['seed'], original_group=draw['original_group'])
                                batch.append(sample)
                            group.append(batch)
                        update = optimizer_update(wrapped, optimizer, group, wrapped, scaler=scaler)
                        scheduler.step()
                        position.update(member=member_index, epoch=epoch, batch=first+group_length,
                                        step=position['step']+1)
                        all_metrics = [item for rank in runtime.gather(update['metrics']) for item in rank]
                        current['epoch_metrics'].extend(all_metrics)
                        for micro in all_metrics:
                            for sample_metrics in micro['samples']:
                                if config.event_weight > 0:
                                    current['event_seen'] = [a or b for a, b in zip(current['event_seen'], sample_metrics['event_seen'])]
                                    current['new_event_seen'] = [a or b for a, b in zip(current['new_event_seen'], sample_metrics['event_seen'])]
                                current['exposures'].append({key: sample_metrics[key] for key in
                                    ('sample_id', 'episode_id', 'source_episode_id', 'original_group',
                                     'offset', 'seed', 'origin', 'sequence_length')})
                        current['optimizer_steps'] += 1
                        current['effective_batch_sizes'].append(sum(len(micro['samples']) for micro in all_metrics))
                        save_boundary(); bar.update(group_length)
                finally:
                    bar.close()
                weights, _ = runtime.snapshot(wrapped, member, optimizer)
                # FSDP validation uses a detached full-state clone; no sharded-method bypass.
                validation_member = (_new_member(config.model, weights, runtime.device)
                                     if runtime.mode == 'fsdp' else member)
                selected = validate_member(validation_member, validation, norm, config, runtime, member_index)
                if runtime.mode == 'fsdp':
                    del validation_member
                train_loss = float(np.mean([row['loss'] for row in current['epoch_metrics']]))
                sample_metrics = [sample for row in current['epoch_metrics'] for sample in row['samples']]
                metrics = {name: float(np.mean([row[name] for row in sample_metrics])) for name in
                    ('state', 'frame', 'changed_region', 'reconstruction', 'event', 'raw_kl', 'clipped_kl',
                     'kl_clipping_rate', 'prior_entropy', 'posterior_entropy', 'changed_fraction')}
                for name in ('physical_rmse', 'baseline_rmse'):
                    metrics[name] = np.sqrt(np.square([row[name] for row in sample_metrics]).mean(0)).tolist()
                current['history'].append(dict(epoch=epoch+1, train_loss=train_loss,
                    training_metrics=metrics, **selected))
                if current['best_metric'] is None or selected['selection_metric'] < current['best_metric']:
                    current.update(best_metric=selected['selection_metric'], selected_epoch=epoch+1, weights=weights,
                                   selected_event_seen=list(current['event_seen']))
                current['validation_unique_samples'] = selected['unique_samples']
                current['epoch_metrics'] = []
                epoch_status(epoch+1, config.epochs, train_loss, selected['selection_metric'],
                             current['best_metric'], enabled=enabled)
                position.update(epoch=epoch+1, batch=0)
                save_boundary()
            final_weights, _ = runtime.snapshot(wrapped, member, optimizer)
            if any(not torch.equal(value, final_weights[name]) for name, value in frozen.items()):
                raise ValueError('Frozen encoder tensors changed')
            current['frozen_verified'] = bool(config.freeze_encoder)
            run['members'].append(current); run['current'] = None
            position.update(member=member_index+1, epoch=0, batch=0)
            save_boundary()
            snapshot = None
            del optimizer, scheduler, scaler, wrapped, member
        version = runtime.rank_zero(lambda: _export(run, config, train, norm, output))
        resources = runtime.gather(hardware_report(runtime.device))
        report = dict(schema='wmal.embodied_training_report.v1', model_version=version,
            output=str(output), world_size=runtime.world_size, config=asdict(config),
            selection_metric=SELECTION_METRIC, members=[{key: value for key, value in row.items() if key != 'weights'}
                for row in run['members']], episode_lineage=run['lineage'],
            training_source=train.manifest['source'], normalization=norm, identity=identity,
            sampling=dict(original_groups=len(train.episode_window_indices), raw_sequences=len(train),
                sequence_length=config.sequence_length, actions=config.sequence_length-1,
                origin_policy='uniform original bootstrap group, uniform sequence start, uniform valid origin',
                rank_sharding='one common global draw, contiguous per-rank microbatch',
                task_sampling=config.task_sampling, change_threshold=config.change_threshold,
                changed_region_normalization='changed pixels only; zero if empty',
                requested_effective_batch=config.batch_size*config.accumulation*runtime.world_size),
            resources=resources, elapsed_seconds=time.perf_counter()-started, transfer=run['transfer'],
            capacity=dict(per_member=capacity, total_independent_parameters=capacity['total']*config.members,
                capacity_claim='smoke only' if config.model.preset == 'smoke' else config.model.preset),
            limitations=['CUDA8 performance unverified', 'FSDP CUDA 2.11.0 path unverified on multiple GPUs'])
        runtime.rank_zero(lambda: atomic_json(output / 'report.json', report))
        return report
    finally:
        runtime.close()
