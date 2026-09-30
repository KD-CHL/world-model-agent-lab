"""Episode-bootstrap training, differentiable rollouts and resumable checkpoints."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

from wmal.datasets.motion_sequences import load_motion_episodes, contiguous_sequences
from wmal.logging.manifest import atomic_json, build_manifest, sha256_file
from wmal.models.motion_network import (SCHEMA, LATENT_SCHEMA, STATE_SCALES, MotionNetwork, atomic_torch_save,
    build_network, content_version, cpu_states, features, integrate, state_vector, target_outputs)
from wmal.models.neural_dynamics import _torch_modules


@dataclass(frozen=True)
class TrainingConfig:
    hidden: tuple = (128,128,128)
    members: int = 5
    epochs: int = 100
    horizon: int = 3
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    rollout_weight: float = .2
    seed: int = 0
    device: str = 'cpu'
    threads: int = 1
    architecture: str = 'physical'
    latent_weight: float = .1
    horizon_decay: float = .9

    def __post_init__(self):
        for name in ('members','epochs','horizon','batch_size','threads'):
            if type(getattr(self,name)) is not int or getattr(self,name) < (2 if name=='members' else 1):
                raise ValueError('Invalid training integer: '+name)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('Invalid training seed')
        if not self.hidden or any(type(v) is not int or v<4 for v in self.hidden):
            raise ValueError('Invalid network hidden widths')
        for name in ('learning_rate','weight_decay','rollout_weight','latent_weight','horizon_decay'):
            value = getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Invalid training scalar: '+name)
        if self.learning_rate == 0:
            raise ValueError('Learning rate must be positive')
        if self.architecture not in ('physical','latent_state') or not 0<self.horizon_decay<=1:
            raise ValueError('Invalid training architecture or horizon discount')


def tensors(windows, device):
    torch, _ = _torch_modules()
    before = torch.tensor([[state_vector(s) for s,a,n in window] for window in windows],
                          dtype=torch.float32,device=device)
    actions = torch.tensor([[[a.vx,a.vy,a.yaw_rate,a.duration_s] for s,a,n in window] for window in windows],
                           dtype=torch.float32,device=device)
    after = torch.tensor([[state_vector(n) for s,a,n in window] for window in windows],
                         dtype=torch.float32,device=device)
    return before, actions, after


def normalization(train_episodes, architecture='physical'):
    torch, _ = _torch_modules()
    windows = [[row] for rows in train_episodes.values() for row in rows]
    before, action, after = tensors(windows,'cpu')
    x = features(before[:,0],action[:,0])
    y = target_outputs(before[:,0],after[:,0],action[:,0],architecture)
    return {'x_mean':x.mean(0).tolist(),'x_scale':x.std(0,unbiased=False).clamp_min(.01).tolist(),
            'y_mean':y.mean(0).tolist(),'y_scale':y.std(0,unbiased=False).clamp_min(.01).tolist()}


def sequence_loss(model, before, actions, after, stats, rollout_weight, latent_weight=.1, horizon_decay=.9):
    torch, _ = _torch_modules()
    normalized = (features(before,actions)-stats['x_mean'])/stats['x_scale']
    latent_model = callable(getattr(model,'advance',None))
    architecture = 'latent_state' if latent_model else 'physical'
    target = (target_outputs(before,after,actions,architecture)-stats['y_mean'])/stats['y_scale']
    one_step = (model(normalized)-target).square().mean()
    current, losses = before[:,0], []
    scales = torch.tensor(STATE_SCALES,device=before.device,dtype=before.dtype)
    latent, consistency = None, []
    if latent_model:
        encoded = model.encoder(normalized[...,:6])
        reconstruction = (model.reconstruction(encoded)-normalized[...,:6]).square().mean()
        latent = encoded[:,0]
        target_x = (features(after,actions)-stats['x_mean'])/stats['x_scale']
        target_latents = model.encoder(target_x[...,:6]).detach()
    for step in range(actions.shape[1]):
        x = (features(current,actions[:,step])-stats['x_mean'])/stats['x_scale']
        if latent_model:
            latent, output = model.advance(latent,x[...,6:])
            consistency.append((latent-target_latents[:,step]).square().mean())
        else:
            output = model(x)
        output = output*stats['y_scale']+stats['y_mean']
        current = integrate(current,output,actions[:,step])
        difference = current-after[:,step]
        # Avoid an in-place edit of tensors needed for autograd.
        yaw = torch.atan2(torch.sin(difference[:,2]),torch.cos(difference[:,2]))
        difference = torch.cat([difference[:,:2],yaw[:,None],difference[:,3:]],dim=-1)
        losses.append((difference/scales).square().mean())
    # Preserve the original v1 objective; apply discount only to the latent architecture.
    weights = torch.tensor([horizon_decay**i if latent_model else 1. for i in range(len(losses))],
                           device=before.device,dtype=before.dtype)
    rollout = (torch.stack(losses)*weights).sum()/weights.sum()
    auxiliary = ((torch.stack(consistency)*weights).sum()/weights.sum()+reconstruction) if latent_model else 0.
    return one_step+rollout_weight*rollout+latent_weight*auxiliary, one_step, rollout


def validation_loss(models, arrays, stats, config):
    torch, _ = _torch_modules()
    total = 0.
    with torch.inference_mode():
        for model in models:
            model.eval()
            for start in range(0,len(arrays[0]),config.batch_size):
                batch = tuple(a[start:start+config.batch_size] for a in arrays)
                loss, _, _ = sequence_loss(model,*batch,stats,config.rollout_weight,config.latent_weight,config.horizon_decay)
                total += float(loss)*len(batch[0])
    return total/(len(models)*len(arrays[0]))


def train_motion(dataset, checkpoint, config=None, *, resume=None, progress=None):
    torch, _ = _torch_modules()
    config = config or TrainingConfig()
    if Path(dataset).resolve() == Path(checkpoint).resolve() or Path(checkpoint).suffix != '.pt':
        raise ValueError('Checkpoint must be a separate .pt file')
    torch.set_num_threads(config.threads)
    splits, duration = load_motion_episodes(dataset)
    if any(not episodes for episodes in splits.values()):
        raise ValueError('Training requires disjoint train, validation and reserved test episodes')
    groups = contiguous_sequences(splits['train'],config.horizon)
    validation = contiguous_sequences(splits['validation'],config.horizon)
    train_windows, group_indices = [], []
    for windows in groups.values():
        group_indices.append(list(range(len(train_windows),len(train_windows)+len(windows))))
        train_windows.extend(windows)
    arrays = tensors(train_windows,config.device)
    valid_arrays = tensors([window for windows in validation.values() for window in windows],config.device)
    norm = normalization(splits['train'],config.architecture)
    stats = {key:torch.tensor(values,dtype=torch.float32,device=config.device) for key,values in norm.items()}
    params = asdict(config)
    params['hidden'] = list(params['hidden'])
    dataset_hash = sha256_file(dataset)
    torch.manual_seed(config.seed)
    models = [build_network(config.hidden,config.architecture).to(config.device) for _ in range(config.members)]
    optimizers = [torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
                  for model in models]
    samples = []
    for member in range(config.members):
        generator = torch.Generator().manual_seed(config.seed+1009*(member+1))
        selected = torch.randint(len(group_indices),(len(group_indices),),generator=generator).tolist()
        samples.append([index for group in selected for index in group_indices[group]])
    start_epoch, best_epoch, best_loss, history = 0, -1, float('inf'), []
    best_states = None
    checkpoint = Path(checkpoint)
    latest_path = checkpoint.with_name(checkpoint.stem+'.latest.pt')
    if resume:
        if Path(resume).resolve() == checkpoint.resolve():
            raise ValueError('Resume needs latest snapshot; output must be a distinct best-model file')
        snapshot = torch.load(resume,map_location='cpu',weights_only=True)
        old_config = snapshot.get('config',{})
        # Old v1 snapshots predate the architecture switches; their defaults preserve v1 math.
        for key in ('architecture','latent_weight','horizon_decay'):
            old_config.setdefault(key,asdict(TrainingConfig())[key])
        unchanged = {k:v for k,v in params.items() if k!='epochs'}
        if (snapshot.get('schema') != 'wmal.motion_training.v1' or snapshot.get('dataset_sha256') != dataset_hash
                or {k:v for k,v in old_config.items() if k!='epochs'} != unchanged
                or snapshot.get('normalization') != norm):
            raise ValueError('Resume dataset, configuration or normalization mismatch')
        if any(len(snapshot[key]) != config.members for key in ('state_dicts','optimizers','best_states')):
            raise ValueError('Resume ensemble size mismatch')
        for model, optimizer, weights, opt_state in zip(models,optimizers,snapshot['state_dicts'],snapshot['optimizers']):
            model.load_state_dict(weights)
            optimizer.load_state_dict(opt_state)
        start_epoch = snapshot['epoch']+1
        best_epoch, best_loss = snapshot['best_epoch'], snapshot['best_loss']
        best_states, history = snapshot['best_states'], snapshot['history']
    elif checkpoint.exists() or latest_path.exists():
        raise FileExistsError('Output already exists; choose another path or resume explicitly')
    if start_epoch >= config.epochs:
        raise ValueError('Total epochs must exceed resumed epoch')
    emit = progress or (lambda item:None)
    started = time.perf_counter()
    for epoch in range(start_epoch,config.epochs):
        training_total, count = 0., 0
        for member, (model,optimizer,indices) in enumerate(zip(models,optimizers,samples)):
            model.train()
            generator = torch.Generator().manual_seed(config.seed+epoch*100003+member*101)
            order = torch.randperm(len(indices),generator=generator).tolist()
            for start in range(0,len(order),config.batch_size):
                selected = torch.tensor([indices[i] for i in order[start:start+config.batch_size]],device=config.device)
                batch = tuple(a[selected] for a in arrays)
                loss, _, _ = sequence_loss(model,*batch,stats,config.rollout_weight,config.latent_weight,config.horizon_decay)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step()
                training_total += float(loss.detach())*len(selected)
                count += len(selected)
        valid_loss = validation_loss(models,valid_arrays,stats,config)
        if not math.isfinite(valid_loss):
            raise ValueError('Nonfinite validation loss')
        if valid_loss < best_loss:
            best_loss, best_epoch, best_states = valid_loss, epoch, cpu_states(models)
        row = {'epoch':epoch,'train_loss':training_total/count,'validation_loss':valid_loss,
               'best_epoch':best_epoch,'elapsed_s':time.perf_counter()-started}
        history.append(row)
        snapshot = {'schema':'wmal.motion_training.v1','config':params,'dataset_sha256':dataset_hash,
                    'normalization':norm,'epoch':epoch,'state_dicts':cpu_states(models),
                    'optimizers':[optimizer.state_dict() for optimizer in optimizers],
                    'best_epoch':best_epoch,'best_loss':best_loss,'best_states':best_states,'history':history}
        atomic_torch_save(latest_path,snapshot)
        emit(row)
    metadata = {'dataset_sha256':dataset_hash,'config':params,'best_epoch':best_epoch,
                'training_episodes':list(splits['train']), 'validation_episodes':list(splits['validation']),
                'reserved_test_episodes':list(splits['test']),
                'joint_order':sorted(next(iter(splits['train'].values()))[0][0].joint_positions)}
    payload = {'schema':LATENT_SCHEMA if config.architecture=='latent_state' else SCHEMA,
               'hidden':list(config.hidden),'duration_s':duration,'normalization':norm,
               'state_dicts':best_states,'metadata':metadata}
    if config.architecture=='latent_state':
        payload['architecture'] = config.architecture
    payload['version'] = content_version(payload)
    atomic_torch_save(checkpoint,payload)
    import wmal.models.motion_network as network_module
    report = {**build_manifest('neural_motion_training',config.seed,
              {'dataset':dataset,'training_source':__file__,'network_source':network_module.__file__},params),
              'model_version':payload['version'],'checkpoint':str(checkpoint),
              'latest_checkpoint':str(latest_path),'checkpoint_sha256':sha256_file(checkpoint),
              'best_epoch':best_epoch,'best_validation_loss':best_loss,'history':history,
              'torch_version':str(torch.__version__), 'duration_s':duration,
              'sequences':{'train':len(train_windows),'validation':len(valid_arrays[0])},
              'episodes':{key:len(value) for key,value in splits.items()},
              'test_used_for_selection':False}
    report['architecture'] = config.architecture
    report['trainable_parameters'] = sum(p.numel() for m in models for p in m.parameters())
    atomic_json(str(checkpoint)+'.report.json',report)
    return report


def evaluate_motion(dataset, checkpoint, *, split='test', horizon=3, device='cpu'):
    from wmal.models.motion_network import load
    if split not in ('validation','test'):
        raise ValueError('Evaluation split must be validation or test')
    splits, duration = load_motion_episodes(dataset)
    model = load({'checkpoint':str(checkpoint),'device':device})
    if abs(model.duration_s-duration)>1e-9:
        raise ValueError('Evaluation duration mismatch')
    if set(splits[split]) & set(model.metadata.get('training_episodes',[])):
        raise ValueError('Evaluation overlaps training episodes')
    if split == 'test' and set(splits[split]) & set(model.metadata.get('validation_episodes',[])):
        raise ValueError('Test overlaps checkpoint-selection episodes')
    windows = contiguous_sequences(splits[split],horizon)
    errors, velocity_errors, yaw_errors = ([[] for _ in range(horizon)] for _ in range(3))
    episode_metrics, episode_velocity_metrics, episode_yaw_metrics = {}, {}, {}
    for episode, sequences in windows.items():
        own, own_velocity, own_yaw = ([[] for _ in range(horizon)] for _ in range(3))
        for window in sequences:
            current = window[0][0]
            rollout = model.rollout(current, [[row[1] for row in window]])
            for step, (_,action,actual) in enumerate(window):
                current = rollout.prediction(0,step).state
                error = (current.x-actual.x)**2+(current.y-actual.y)**2
                own[step].append(error)
                errors[step].append(error)
                velocity_error = (current.vx-actual.vx)**2+(current.vy-actual.vy)**2
                yaw_error = math.atan2(math.sin(current.yaw-actual.yaw),math.cos(current.yaw-actual.yaw))**2
                own_velocity[step].append(velocity_error)
                own_yaw[step].append(yaw_error)
                velocity_errors[step].append(velocity_error)
                yaw_errors[step].append(yaw_error)
        episode_metrics[episode] = [math.sqrt(sum(e)/len(e)) for e in own]
        episode_velocity_metrics[episode] = [math.sqrt(sum(e)/len(e)) for e in own_velocity]
        episode_yaw_metrics[episode] = [math.sqrt(sum(e)/len(e)) for e in own_yaw]
    return {'schema':'wmal.motion_evaluation.v1','model_version':model.version,'split':split,
            'dataset_sha256':sha256_file(dataset),'horizon':horizon,
            'position_rmse_m_by_step':[math.sqrt(sum(e)/len(e)) for e in errors],
            'terminal_velocity_rmse_m_s_by_step':[math.sqrt(sum(e)/len(e)) for e in velocity_errors],
            'yaw_rmse_rad_by_step':[math.sqrt(sum(e)/len(e)) for e in yaw_errors],
            'episode_position_rmse_m_by_step':episode_metrics,
            'episode_terminal_velocity_rmse_m_s_by_step':episode_velocity_metrics,
            'episode_yaw_rmse_rad_by_step':episode_yaw_metrics,
            'rollout_mode':'member_preserving_batch','architecture':model.architecture,
            'statistical_unit':'episode; overlapping windows are not independent repeats'}
