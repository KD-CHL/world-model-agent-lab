"""Independent visual world-model training, fine-tuning and held-out evaluation."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import math
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler, Subset

from wmal.datasets.visual_sequences import VisualDataset, read_episode_lineage, assert_unexposed_episodes
from wmal.logging.manifest import atomic_json, sha256_file
from wmal.models.visual_latent import NetworkConfig, make_member, VisualWorldModel
from wmal.training.progress import progress_bar, training_status, epoch_status


def _average_ranks(values):
    """Stable average ranks for Spearman metrics, including tied predictions."""
    values=np.asarray(values)
    order=np.argsort(values,kind='mergesort')
    sorted_values=values[order]
    ranks=np.empty(len(order),dtype=float)
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and sorted_values[end]==sorted_values[start]:
            end+=1
        ranks[order[start:end]]=(start+end-1)/2.
        start=end
    return ranks


def _rank_correlation(first, second):
    """Pearson correlation of rank vectors without invoking a BLAS backend."""
    first=tuple(float(value) for value in first)
    second=tuple(float(value) for value in second)
    if len(first)!=len(second):
        raise ValueError('Rank vectors must have the same length')
    if not first or not all(math.isfinite(value) for value in first+second):
        return None
    first_mean=math.fsum(first)/len(first)
    second_mean=math.fsum(second)/len(second)
    first_centered=tuple(value-first_mean for value in first)
    second_centered=tuple(value-second_mean for value in second)
    first_ss=math.fsum(value*value for value in first_centered)
    second_ss=math.fsum(value*value for value in second_centered)
    denominator=math.sqrt(first_ss*second_ss)
    if denominator==0:
        return None
    covariance=math.fsum(a*b for a,b in zip(first_centered,second_centered))
    return covariance/denominator


def _event_sample_scores(probabilities, targets, target_mask, origin, origin_mask):
    """Score event prediction and persistence only where the origin label is known."""
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    target_mask = np.asarray(target_mask, dtype=bool)
    origin = np.asarray(origin)
    origin_mask = np.asarray(origin_mask, dtype=bool)
    if (probabilities.shape != targets.shape or target_mask.shape != targets.shape
            or origin.shape != targets.shape[1:] or origin_mask.shape != origin.shape):
        raise ValueError('Event labels, masks, and prediction-origin state do not align')
    mask = target_mask & origin_mask[None, :]
    persistence = np.broadcast_to(origin[None, :], targets.shape)
    return (np.square(probabilities-targets),
            np.square(persistence-targets),
            ((probabilities >= .5) == (targets >= .5)).astype(float), mask)


def _masked_horizon_mean(values, masks):
    values, masks = np.asarray(values), np.asarray(masks, dtype=bool)
    counts = masks.sum(axis=(0, 2))
    totals = (values * masks).sum(axis=(0, 2))
    means = [float(total/count) if count else None for total, count in zip(totals, counts)]
    scored_windows = masks.any(axis=2).sum(axis=0).astype(int).tolist()
    return means, scored_windows, counts.astype(int).tolist()


class EpisodeGroupedSampler(Sampler):
    """Shuffle windows while keeping each episode locally cached during a pass."""
    def __init__(self, episode_ids, seed):
        self.episode_ids = tuple(int(value) for value in episode_ids)
        self.seed = int(seed)
        self.epoch = 0
        self.groups = {}
        for index, episode_id in enumerate(self.episode_ids):
            self.groups.setdefault(episode_id, []).append(index)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        episodes = np.asarray(list(self.groups))
        rng.shuffle(episodes)
        for episode_id in episodes:
            indices = np.asarray(self.groups[int(episode_id)], dtype=np.int64)
            rng.shuffle(indices)
            yield from indices.tolist()

    def __len__(self):
        return len(self.episode_ids)


@dataclass(frozen=True)
class VisualTrainingConfig:
    epochs: int = 30
    members: int = 3
    batch_size: int = 16
    horizon: int = 4
    latent_dim: int = 64
    hidden_dim: int = 128
    learning_rate: float = .0003
    weight_decay: float = .00001
    frame_weight: float = 1.
    state_weight: float = 1.
    latent_weight: float = .05
    event_weight: float = .2
    change_weight: float = 4.
    gradient_clip: float = 5.
    seed: int = 0
    device: str = 'cpu'
    threads: int = 1
    architecture: str = 'deterministic'
    stoch: int = 8
    classes: int = 8
    unimix: float = .01
    dyn_weight: float = 1.
    rep_weight: float = .1
    free_nats: float = 1.
    reconstruction_weight: float = 1.
    context_steps: int = 0
    object_weight: float = 1.0
    object_observation_weight: float = 0.0
    object_state_mode: str = 'absolute'
    object_spatial_encoder: bool = False
    window_stride: int = 1

    def __post_init__(self):
        if (any(type(v) is not int or v<1 for v in (self.epochs,self.members,self.batch_size,
                self.horizon,self.latent_dim,self.hidden_dim,self.threads)) or self.members<2
                or type(self.seed) is not int or self.seed<0
                or any(not math.isfinite(v) or v<0 for v in (self.learning_rate,self.weight_decay,
                   self.frame_weight,self.state_weight,self.latent_weight,self.event_weight,self.change_weight))
                or self.learning_rate<=0 or not math.isfinite(self.gradient_clip) or self.gradient_clip<=0
                or self.frame_weight<=0 or self.state_weight<=0
                or any(not math.isfinite(v) or v<0 for v in
                       (self.object_weight,self.object_observation_weight))
                or self.object_state_mode not in ('absolute','residual')
                or type(self.window_stride) is not int or self.window_stride<1
                or type(self.object_spatial_encoder) is not bool):
            raise ValueError('Invalid visual training configuration')
        NetworkConfig(1,1,architecture=self.architecture,stoch=self.stoch,
                      classes=self.classes,unimix=self.unimix,context_steps=self.context_steps)
        if any(not math.isfinite(v) or v<0 for v in
               (self.dyn_weight,self.rep_weight,self.free_nats,self.reconstruction_weight)):
            raise ValueError('Invalid RSSM loss configuration')


def batch_loss(member, batch, normalization, config):
    device=next(member.parameters()).device
    rgb,state,action=[batch[key].to(device) for key in ('rgb','states','actions')]
    norm={key:torch.as_tensor(value,dtype=torch.float32,device=device) for key,value in normalization.items()}
    state=(state-norm['state_mean'])/norm['state_scale']
    action=(action-norm['action_mean'])/norm['action_scale']
    if member.config.architecture=='categorical_rssm':
        from wmal.training.rssm_loss import rssm_loss
        return rssm_loss(member,rgb,state,action,batch,config,norm)
    prediction=member.imagine(rgb[:,0],state[:,0],action)
    # Dynamic pixels are upweighted, but are targets only (not rollout inputs).
    weights=1.+config.change_weight*(rgb[:,1:]-rgb[:,:1]).abs().mean(2,keepdim=True)
    pixel=((prediction['rgb']-rgb[:,1:]).square()*weights).mean()
    state_loss=F.mse_loss(prediction['state'],state[:,1:])
    target_latents=member.encoder(rgb[:,1:].reshape(-1,*rgb.shape[2:])).detach().reshape_as(prediction['latent'])
    latent=F.mse_loss(prediction['latent'],target_latents)
    event=torch.zeros((),device=device)
    if prediction['event_logits'] is not None:
        mask=batch['event_mask'].to(device)
        loss=F.binary_cross_entropy_with_logits(prediction['event_logits'],batch['events'].to(device),reduction='none')
        event=(loss*mask).sum()/mask.sum().clamp_min(1.)
    object_loss=torch.zeros((),device=device)
    object_observation_loss=torch.zeros((),device=device)
    if prediction.get('object_state') is not None:
        if 'object_truth' not in batch or 'object_mean' not in norm or 'object_scale' not in norm:
            raise ValueError('Object prediction requires future object truth and training normalization')
        object_target=(batch['object_truth'].to(device)[:,1:]-norm['object_mean'])/norm['object_scale']
        object_loss=F.mse_loss(prediction['object_state'],object_target)
        if prediction.get('object_observation_state') is not None:
            initial=(batch['object_truth'].to(device)[:,:1]-norm['object_mean'])/norm['object_scale']
            object_observation_loss=F.mse_loss(prediction['object_observation_state'][:,None],initial)
    total=(config.frame_weight*pixel+config.state_weight*state_loss+config.latent_weight*latent+
           config.event_weight*event+config.object_weight*object_loss+
           config.object_observation_weight*object_observation_loss)
    return total,{'frame':float(pixel.detach()),'state':float(state_loss.detach()),
                  'latent':float(latent.detach()),'event':float(event.detach()),
                  'object_state':float(object_loss.detach()),
                  'object_observation':float(object_observation_loss.detach())}


def _fine_tune_parent(parent, network, semantics, device):
    """Validate a fine-tune parent, allowing named event channels to append."""
    if len(parent.members)<2:
        raise ValueError('Fine-tuning checkpoint has an invalid ensemble')
    old_names=parent.semantics.get('event_names',[])
    new_names=semantics.get('event_names',[])
    if (not isinstance(old_names,list) or not isinstance(new_names,list)
            or len(old_names)!=parent.config.event_dim or len(new_names)!=network.event_dim
            or any(not isinstance(name,str) or not name for name in old_names+new_names)
            or len(set(old_names))!=len(old_names) or len(set(new_names))!=len(new_names)):
        raise ValueError('Fine-tuning event names do not match event dimensions')
    old_core={key:value for key,value in parent.semantics.items() if key!='event_names'}
    new_core={key:value for key,value in semantics.items() if key!='event_names'}
    if old_core!=new_core:
        raise ValueError('Fine-tuning architecture/control semantics do not match pretrained checkpoint')
    if replace(parent.config,event_dim=network.event_dim)!=network:
        raise ValueError('Fine-tuning architecture/control semantics do not match pretrained checkpoint')
    if parent.config.event_dim>network.event_dim or new_names[:len(old_names)]!=old_names:
        raise ValueError('Fine-tuning may only append named events to an otherwise identical checkpoint')

    if network.event_dim==parent.config.event_dim:
        members=parent.members
    else:
        event_keys={'event_head.weight','event_head.bias'}
        members=[]
        for previous in parent.members:
            current=make_member(network).to(device)
            old_state=previous.state_dict()
            new_state=current.state_dict()
            added=set(new_state)-set(old_state)
            removed=set(old_state)-set(new_state)
            expected_added=event_keys if parent.config.event_dim==0 else set()
            if added!=expected_added or removed:
                raise ValueError('Fine-tuning checkpoint layers differ beyond the event head')
            with torch.no_grad():
                for name,old_value in old_state.items():
                    if name in event_keys:
                        new_value=new_state[name]
                        if (old_value.shape[0]!=parent.config.event_dim
                                or new_value.shape[0]!=network.event_dim
                                or old_value.shape[1:]!=new_value.shape[1:]):
                            raise ValueError('Fine-tuning event head dimensions are inconsistent')
                        new_value[:parent.config.event_dim].copy_(old_value)
                    else:
                        if new_state[name].shape!=old_value.shape:
                            raise ValueError('Fine-tuning checkpoint layer dimensions do not match')
                        new_state[name].copy_(old_value)
            current.load_state_dict(new_state,strict=True)
            members.append(current)

    masks=parent.metadata.get('event_supervised_by_member')
    if masks is None:
        masks=[[False]*parent.config.event_dim for _ in parent.members]
    if (not isinstance(masks,list) or len(masks)!=len(parent.members)
            or any(not isinstance(row,list) or len(row)!=parent.config.event_dim
                   or any(type(value) is not bool for value in row) for row in masks)):
        raise ValueError('Fine-tuning checkpoint has invalid event supervision evidence')
    event_masks=[row+[False]*(network.event_dim-parent.config.event_dim) for row in masks]
    return members,event_masks


def train_visual(manifest, output_dir, config=VisualTrainingConfig(), *, pretrained=None, freeze_encoder=False,
                 show_progress=False):
    output=Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Training output is nonempty; use a new directory')
    if freeze_encoder and pretrained is None:
        raise ValueError('Cannot freeze a randomly initialized encoder')
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    if config.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    training_status(f'Checking training/validation dataset hashes: {Path(manifest).resolve()}',enabled=show_progress)
    train=VisualDataset(manifest,'train',horizon=config.horizon,context_steps=config.context_steps)
    validation=VisualDataset(manifest,'validation',horizon=config.horizon,context_steps=config.context_steps)
    def sampled_indices(dataset):
        terminal={i:row['steps']-config.horizon for i,row in enumerate(dataset.rows)}
        return [index for index,(episode,offset) in enumerate(dataset.index)
                if offset%config.window_stride==0 or offset==terminal[episode]]
    training_indices=sampled_indices(train)
    validation_indices=sampled_indices(validation)
    if not training_indices or not validation_indices:
        raise ValueError('Window stride removed all train or validation sequences')
    source=train.manifest['source']
    source_kind=source.get('kind','unknown') if isinstance(source,dict) else (
        source if isinstance(source,str) else 'unknown')
    training_status(f'source={source_kind} '
                    f'train_episodes={len(train.rows)} train_windows={len(train)} '
                    f'validation_episodes={len(validation.rows)} validation_windows={len(validation)} '
                    f'device={config.device}',enabled=show_progress)
    semantics=train.semantics
    network=NetworkConfig(len(semantics['state_order']),len(semantics['action_order']),
                          image_size=semantics['image_size'],latent_dim=config.latent_dim,
                          hidden_dim=config.hidden_dim,event_dim=len(semantics.get('event_names',[])),
                          architecture=config.architecture,stoch=config.stoch,
                          classes=config.classes,unimix=config.unimix,context_steps=config.context_steps,
                          object_dim=len(semantics.get('object_state_order',[])),
                          object_state_mode=config.object_state_mode,
                          object_spatial_encoder=config.object_spatial_encoder)
    normalization=train.normalization()
    lineage={'schema':'wmal.episode_lineage.v1',
             'train':{'episode_ids':[],'sha256':[]},'selection':{'episode_ids':[],'sha256':[]}}
    if pretrained is None:
        members=[make_member(network).to(config.device) for _ in range(config.members)]
        parent_version=None
        parent_event_masks=[[False]*network.event_dim for _ in members]
    else:
        parent=VisualWorldModel.load(pretrained,device=config.device)
        if len(parent.members)!=config.members:
            raise ValueError('Fine-tuning architecture/control semantics do not match pretrained checkpoint')
        members,parent_event_masks=_fine_tune_parent(parent,network,semantics,config.device)
        lineage=read_episode_lineage(parent.metadata)
        assert_unexposed_episodes(validation.rows,[lineage['train']],context='Validation')
        normalization=parent.normalization  # Freeze coordinates as well as learned backbone.
        parent_version=parent.version
    for kind,rows in (('train',train.rows),('selection',validation.rows)):
        for key in ('episode_ids','sha256'):
            field='episode_id' if key=='episode_ids' else 'sha256'
            lineage[kind][key]=sorted(set(lineage[kind][key])|{r[field] for r in rows})
    for member in members:
        encoders=[member.encoder]
        if hasattr(member,'state_encoder'):
            encoders.append(member.state_encoder)
        for encoder in encoders:
            for parameter in encoder.parameters():
                parameter.requires_grad_(not freeze_encoder)
    optimizers=[torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                 lr=config.learning_rate,weight_decay=config.weight_decay) for m in members]
    # Resample independent EPISODES, preserving all within-episode windows.
    by_episode={i:[] for i in range(len(train.rows))}
    for j in training_indices:
        by_episode[train.index[j][0]].append(j)
    if any(not indices for indices in by_episode.values()):
        raise ValueError('Window stride removed all sequences from a training episode')
    rng=np.random.default_rng(config.seed)
    loaders=[]
    event_masks=[]
    for i in range(config.members):
        episode_indices=rng.integers(0,len(train.rows),len(train.rows))
        indices=[j for episode in episode_indices for j in by_episode[int(episode)]]
        if not indices:
            raise ValueError('Bootstrap contains no valid training windows')
        event_supervised=np.asarray(parent_event_masks[i],dtype=bool).copy()
        if config.event_weight>0 and network.event_dim:
            for episode in set(map(int,episode_indices)):
                if by_episode[episode]:
                    event_supervised |= train.event_supervision_for_episode(episode)
        event_masks.append(event_supervised.tolist())
        episode_ids=[train.index[j][0] for j in indices]
        sampler=EpisodeGroupedSampler(episode_ids,config.seed+i)
        loaders.append(DataLoader(Subset(train,indices),batch_size=config.batch_size,sampler=sampler,
                                  num_workers=0))
    val_loader=DataLoader(Subset(validation,validation_indices),batch_size=config.batch_size,
                          shuffle=False,num_workers=0)
    output.mkdir(parents=True,exist_ok=True)
    metadata={'max_horizon':config.horizon,'dataset_manifest_sha256':sha256_file(manifest),
              'training_source':train.manifest['source'],
              'training_episode_ids':[r['episode_id'] for r in train.rows],
              'selection_episode_ids':[r['episode_id'] for r in validation.rows],
              'parent_model_version':parent_version,'episode_lineage':lineage,'freeze_encoder':freeze_encoder,
              'event_supervision':bool(np.asarray(event_masks,dtype=bool).all(0).any()),
              'event_supervised_by_member':event_masks,
              'event_supervised_channels':np.asarray(event_masks,dtype=bool).all(0).tolist(),
              'config':asdict(config)}
    history,best,started=[],float('inf'),time.perf_counter()
    label='Fine-tune epochs' if pretrained is not None else 'Visual epochs'
    with progress_bar(config.epochs,label,enabled=show_progress,unit='epoch') as epochs:
        for epoch in range(config.epochs):
            train_total,train_count=0.,0
            with progress_bar(sum(len(loader) for loader in loaders),f'Train {epoch+1}/{config.epochs}',
                              enabled=show_progress,position=1,leave=False) as batches:
                for index,(member,optimizer,loader) in enumerate(zip(members,optimizers,loaders)):
                    member.train()
                    for batch in loader:
                        optimizer.zero_grad(set_to_none=True)
                        loss,_=batch_loss(member,batch,normalization,config)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite training loss; checkpoint not overwritten')
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(member.parameters(),config.gradient_clip,error_if_nonfinite=True)
                        optimizer.step()
                        size=len(batch['rgb'])
                        train_total+=float(loss.detach())*size
                        train_count+=size
                        batches.set_postfix(member=f'{index+1}/{len(members)}',
                                            loss=f'{train_total/train_count:.4g}',refresh=False)
                        batches.update()
            val_total,val_count=0.,0
            components={}
            with progress_bar(len(members)*len(val_loader),f'Validation {epoch+1}/{config.epochs}',
                              enabled=show_progress,position=1,leave=False) as batches, torch.inference_mode():
                for index,member in enumerate(members):
                    member.eval()
                    for batch in val_loader:
                        loss,parts=batch_loss(member,batch,normalization,config)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite validation loss; checkpoint not overwritten')
                        size=len(batch['rgb'])
                        val_total+=float(loss)*size
                        val_count+=size
                        for key,value in parts.items():
                            components[key]=components.get(key,0.)+value*size
                        batches.set_postfix(member=f'{index+1}/{len(members)}',
                                            loss=f'{val_total/val_count:.4g}',refresh=False)
                        batches.update()
            val_loss=val_total/val_count
            row={'epoch':epoch+1,'training_loss':train_total/train_count,'validation_loss':val_loss,
                 'validation_components':{key:value/val_count for key,value in components.items()}}
            history.append(row)
            model=VisualWorldModel(network,semantics,normalization,members,device=config.device,metadata=metadata)
            if val_loss<best:
                best=val_loss
                model.save(output/'best.pt')
            model.save(output/'last.pt')
            atomic_json(output/'history.json',history)
            epochs.set_postfix(train=f'{row["training_loss"]:.4g}',val=f'{val_loss:.4g}',refresh=False)
            epochs.update()
            epoch_status(epoch+1,config.epochs,row['training_loss'],val_loss,best,enabled=show_progress)
    selected=VisualWorldModel.load(output/'best.pt')
    report={'schema':'wmal.visual_training_report.v1','model_version':selected.version,
            'best_validation_loss':best,'history':history,'elapsed_s':time.perf_counter()-started,
            'checkpoint':str(output/'best.pt'),'metadata':metadata,
            'parameters_per_member':sum(p.numel() for p in members[0].parameters()),
            'trainable_parameters_per_member':sum(p.numel() for p in members[0].parameters() if p.requires_grad)}
    atomic_json(output/'training_report.json',report)
    return report


def evaluate_visual(manifest, checkpoint, *, horizon=4, device='cpu', calibration=None,
                    object_calibration=None, window_stride=1, split='test',
                    allow_controller_transfer=False):
    from wmal.models.horizon_calibration import HorizonCalibration, ObjectHorizonCalibration
    if split not in ('validation','calibration','test'):
        raise ValueError('Evaluation split must be validation/calibration/test')
    if split!='test' and (calibration is not None or object_calibration is not None):
        raise ValueError('Final calibration artifacts may only be applied to test evaluation')
    model=VisualWorldModel.load(checkpoint,device=device)
    dataset=VisualDataset(manifest,split,horizon=horizon,context_steps=model.config.context_steps)
    if type(window_stride) is not int or window_stride < 1:
        raise ValueError('window_stride must be a positive integer')
    if dataset.semantics!=model.semantics:
        raise ValueError('Evaluation dataset/control semantics mismatch')
    controller=dataset.manifest.get('source',{}).get('controller_profile',{'profile':'g1_joint_pd_v1'})
    training_source=model.metadata.get('training_source',{})
    trained_controller=training_source.get('controller_profile',{'profile':'g1_joint_pd_v1'}) if isinstance(training_source,dict) else {'profile':'g1_joint_pd_v1'}
    if controller!=trained_controller and not allow_controller_transfer:
        raise ValueError('Evaluation controller transfer requires explicit --allow-controller-transfer')
    lineage=read_episode_lineage(model.metadata)
    evaluation_ids={row['episode_id'] for row in dataset.rows}
    if split=='validation':
        assert_unexposed_episodes(dataset.rows,[lineage['train']],context='Model selection evaluation')
        if not evaluation_ids<=set(lineage['selection']['episode_ids']):
            raise ValueError('Validation evaluation contains episodes outside the model-selection lineage')
    else:
        assert_unexposed_episodes(dataset.rows,[lineage['train'],lineage['selection']],context='Evaluation')
    bound_model=HorizonCalibration.load(calibration) if calibration else None
    if bound_model:
        bound_model.validate_for(model.version,model.semantics)
        bound_model.validate_controller(controller)
        if bound_model.prediction_scope!=dataset.manifest.get('source',{}).get('prediction_scope'):
            raise ValueError('Evaluation/calibration deployment scope mismatch')
        if set(bound_model.episode_ids).intersection(r['episode_id'] for r in dataset.rows):
            raise ValueError('Calibration overlaps evaluation episodes')
    object_bound_model=(ObjectHorizonCalibration.load(object_calibration) if object_calibration else None)
    if object_bound_model:
        if controller!=trained_controller:
            raise ValueError('Object calibration has no controller provenance; cannot apply after controller transfer')
        object_bound_model.validate_for(model.version,model.semantics)
        if set(object_bound_model.episode_ids).intersection(r['episode_id'] for r in dataset.rows):
            raise ValueError('Object calibration overlaps evaluation episodes')
    frame,state,persist_frame,persist_state,sensitivity=[],[],[],[],[]
    frame_sensitivity,change_error,change_persistence=[],[],[]
    event_brier,event_accuracy,event_persistence_brier,event_masks=[],[],[],[]
    event_uncertainty,event_actual_error=[],[]
    object_error,object_persistence,object_error_dimensions,object_persistence_dimensions=[],[],[],[]
    object_physical_error_dimensions,object_physical_persistence_dimensions=[],[]
    object_quaternion_angle_errors={}
    object_observation_errors,object_observation_physical_errors=[],[]
    object_uncertainty,object_actual_error=[],[]
    calibrated_width=[]
    object_coverage,object_calibrated_width={},[]
    object_group_coverage,object_group_calibrated_width={},{}
    object_group_indices={}
    for index,name in enumerate(model.semantics.get('object_state_order',[])):
        group=':'.join(name.split(':')[:2])
        object_group_indices.setdefault(group,[]).append(index)
    event_names=[]
    coverage={}
    selected_indices=[]
    offset=0
    for row in dataset.rows:
        count=row['steps']-horizon+1
        selected_indices.extend(offset+i for i in range(0,count,window_stride))
        offset+=count
    for i in selected_indices:
        sample=dataset[i]
        predicted=model.predict_sample(sample)
        counter=model.predict_sample(sample,actions=np.zeros_like(sample['actions'][model.config.context_steps:]))
        # Keep context away from target metrics and persistence baselines.
        sample=dict(sample,rgb=sample['rgb'][model.config.context_steps:],
                    states=sample['states'][model.config.context_steps:])
        if 'object_truth' in sample:
            sample['object_truth']=sample['object_truth'][model.config.context_steps:]
        mean=predicted['states'].mean(0)
        frames=predicted['frames'].mean(0)
        frame.append(((frames-sample['rgb'][1:])**2).mean((1,2,3)))
        state.append(((mean-sample['states'][1:])**2).mean(-1))
        persist_frame.append(((sample['rgb'][:1]-sample['rgb'][1:])**2).mean((1,2,3)))
        persist_state.append(((sample['states'][:1]-sample['states'][1:])**2).mean(-1))
        if predicted.get('object_states') is not None:
            object_mean=predicted['object_states'].mean(0)
            object_target=sample['object_truth'][1:]
            object_scale=np.asarray(model.normalization['object_scale'])
            normalized_error=(object_mean-object_target)/object_scale
            normalized_persistence=(sample['object_truth'][:1]-object_target)/object_scale
            object_error_dimensions.append(normalized_error**2)
            object_persistence_dimensions.append(normalized_persistence**2)
            object_physical_error_dimensions.append((object_mean-object_target)**2)
            object_physical_persistence_dimensions.append((sample['object_truth'][:1]-object_target)**2)
            if predicted.get('object_observations') is not None:
                current_target=sample['object_truth'][0]
                current_mean=predicted['object_observations'].mean(0)
                object_observation_errors.append(((current_mean-current_target)/object_scale)**2)
                object_observation_physical_errors.append((current_mean-current_target)**2)
            quaternion_groups={}
            for index,name in enumerate(model.semantics['object_state_order']):
                parts=name.split(':')
                if len(parts)==3 and parts[1]=='quaternion_wxyz':
                    quaternion_groups.setdefault(parts[0],[]).append(index)
            for obj,indices in quaternion_groups.items():
                if len(indices)!=4:
                    continue
                predicted_q=object_mean[:,indices]
                target_q=object_target[:,indices]
                persistence_q=np.repeat(sample['object_truth'][:1,indices],len(object_target),axis=0)
                predicted_q=predicted_q/np.maximum(np.linalg.norm(predicted_q,axis=-1,keepdims=True),1e-8)
                target_q=target_q/np.maximum(np.linalg.norm(target_q,axis=-1,keepdims=True),1e-8)
                persistence_q=persistence_q/np.maximum(np.linalg.norm(persistence_q,axis=-1,keepdims=True),1e-8)
                predicted_angle=2*np.arccos(np.clip(np.abs((predicted_q*target_q).sum(-1)),0.,1.))
                persistence_angle=2*np.arccos(np.clip(np.abs((persistence_q*target_q).sum(-1)),0.,1.))
                object_quaternion_angle_errors.setdefault(obj,[]).append(
                    (predicted_angle**2,persistence_angle**2))
            object_error.append((normalized_error**2).mean(-1))
            object_persistence.append((normalized_persistence**2).mean(-1))
            object_uncertainty.append((predicted['object_states'].std(0)/object_scale).mean(-1))
            object_actual_error.append((np.abs(object_mean-object_target)/object_scale).mean(-1))
            if object_bound_model:
                bounds=object_bound_model.bounds(predicted['object_states'].std(0),version=model.version,
                                                 object_state_order=model.semantics['object_state_order'])
                absolute_error=np.abs(object_mean-object_target)
                covered=np.all(absolute_error<=bounds,axis=-1)
                key=sample['episode_id']
                object_coverage[key]=covered if key not in object_coverage else object_coverage[key]&covered
                object_calibrated_width.append((bounds/object_scale[None,:]).mean(-1))
                for group,indices in object_group_indices.items():
                    group_covered=np.all(absolute_error[:,indices]<=bounds[:,indices],axis=-1)
                    prior=object_group_coverage.setdefault(group,{})
                    prior[key]=group_covered if key not in prior else prior[key]&group_covered
                    object_group_calibrated_width.setdefault(group,[]).append(
                        (bounds[:,indices]/object_scale[None,indices]).mean(-1))
        # Counterfactual sensitivity, not an assertion that zero-action targets are known.
        sensitivity.append(np.abs(mean-counter['states'].mean(0)).mean(-1))
        frame_sensitivity.append(np.abs(frames-counter['frames'].mean(0)).mean((1,2,3)))
        changed=(np.abs(sample['rgb'][1:]-sample['rgb'][:1]).max(1,keepdims=True)>.02)
        count=np.broadcast_to(changed,frames.shape).sum((1,2,3))
        change_error.append((((frames-sample['rgb'][1:])**2)*changed).sum((1,2,3))/np.maximum(count,1))
        change_persistence.append((((sample['rgb'][:1]-sample['rgb'][1:])**2)*changed).sum((1,2,3))/np.maximum(count,1))
        if bound_model:
            state_std=predicted['states'].std(0)
            bounds=bound_model.bounds(state_std,version=model.version)
            covered=np.all(np.abs(mean-sample['states'][1:])<=bounds,axis=-1)
            key=sample['episode_id']
            coverage[key]=covered if key not in coverage else coverage[key]&covered
            calibrated_width.append((bounds/np.asarray(bound_model.state_scale)[None,:]).mean(-1))
        if predicted.get('event_probabilities') is not None:
            event_names=predicted.get('event_probability_names',event_names)
            probabilities=predicted['event_probabilities'].mean(0)
            targets=sample['events']
            if probabilities.shape != targets.shape:
                raise ValueError('Event prediction dimensions do not match held-out labels')
            scores = _event_sample_scores(
                probabilities, targets, sample['event_mask'],
                sample['event_context'], sample['event_context_mask'])
            model_brier, persistence_brier, accuracy, score_mask = scores
            event_brier.append(model_brier)
            event_accuracy.append(accuracy)
            event_persistence_brier.append(persistence_brier)
            event_masks.append(score_mask)
            valid_horizon = score_mask.any(-1)
            channel_count = np.maximum(score_mask.sum(-1), 1)
            event_uncertainty.append(predicted['event_probabilities'].std(0).mean(-1)[valid_horizon])
            event_actual_error.append(
                ((np.abs(probabilities-targets)*score_mask).sum(-1)/channel_count)[valid_horizon])
    event_metrics=None
    if event_brier:
        errors=np.concatenate(event_actual_error)
        uncertainty=np.concatenate(event_uncertainty)
        error_ranks,uncertainty_ranks=_average_ranks(errors),_average_ranks(uncertainty)
        rank_correlation=_rank_correlation(error_ranks,uncertainty_ranks) if errors.size else None
        brier_by_horizon, scored_windows, scored_labels = _masked_horizon_mean(event_brier,event_masks)
        persistence_by_horizon, _, _ = _masked_horizon_mean(event_persistence_brier,event_masks)
        accuracy_by_horizon, _, _ = _masked_horizon_mean(event_accuracy,event_masks)
        event_metrics={
            'event_names':event_names,
            'brier_by_horizon':brier_by_horizon,
            'persistence_brier_by_horizon':persistence_by_horizon,
            'accuracy_by_horizon':accuracy_by_horizon,
            'scored_windows_by_horizon':scored_windows,
            'scored_labels_by_horizon':scored_labels,
            'persistence_origin':'last known event state at prediction origin',
            'reset_event_context_policy':(
                'use stored reset event state; omit offset-zero scoring for legacy archives without one'),
            'ensemble_spread_vs_absolute_error_spearman':rank_correlation,
        }
    report={'schema':'wmal.visual_evaluation.v1','split':split,'model_version':model.version,
            'controller_profile':controller,'controller_transfer':controller!=trained_controller,
            'prediction_scope':dataset.manifest.get('source',{}).get('prediction_scope'),
            'context_steps':model.config.context_steps,
            'manifest_sha256':sha256_file(manifest),'episodes':len(dataset.rows),
            'windows':len(selected_indices),'window_stride_control_steps':window_stride,
            'frame_mse_by_horizon':np.mean(frame,0).tolist(),
            'state_rmse_by_horizon':np.sqrt(np.mean(state,0)).tolist(),
            'persistence_frame_mse_by_horizon':np.mean(persist_frame,0).tolist(),
            'persistence_state_rmse_by_horizon':np.sqrt(np.mean(persist_state,0)).tolist(),
            'zero_action_sensitivity_by_horizon':np.mean(sensitivity,0).tolist(),
            'zero_action_frame_sensitivity_by_horizon':np.mean(frame_sensitivity,0).tolist(),
            'changed_region_frame_mse_by_horizon':np.mean(change_error,0).tolist(),
            'persistence_changed_region_frame_mse_by_horizon':np.mean(change_persistence,0).tolist(),
            'episode_simultaneous_coverage_by_horizon':np.mean(list(coverage.values()),0).tolist() if coverage else None,
            'calibrated_state_mean_half_width_by_horizon':np.mean(calibrated_width,0).tolist() if calibrated_width else None,
            'event_prediction_metrics':event_metrics}
    if object_error:
        errors=np.concatenate(object_actual_error)
        uncertainty=np.concatenate(object_uncertainty)
        error_ranks,uncertainty_ranks=_average_ranks(errors),_average_ranks(uncertainty)
        correlation=_rank_correlation(error_ranks,uncertainty_ranks)
        report['object_state_metrics']={
            'state_order':model.semantics['object_state_order'],
            'normalized_rmse_by_horizon':np.sqrt(np.mean(object_error,0)).tolist(),
            'persistence_normalized_rmse_by_horizon':np.sqrt(np.mean(object_persistence,0)).tolist(),
            'ensemble_spread_vs_absolute_error_spearman':correlation,
            'episode_simultaneous_coverage_by_horizon':(
                np.mean(list(object_coverage.values()),0).tolist() if object_coverage else None),
            'calibrated_mean_half_width_by_horizon':(
                np.mean(object_calibrated_width,0).tolist() if object_calibrated_width else None),
            'calibrated_group_half_width_by_horizon':{
                group:np.mean(widths,0).tolist() for group,widths in object_group_calibrated_width.items()},
            'group_episode_simultaneous_coverage_by_horizon':{
                group:np.mean(list(episodes.values()),0).tolist()
                for group,episodes in object_group_coverage.items()},
        }
        errors_by_dimension=np.stack(object_error_dimensions,axis=0)
        persistence_by_dimension=np.stack(object_persistence_dimensions,axis=0)
        names=model.semantics['object_state_order']
        groups={}
        for index,name in enumerate(names):
            parts=name.split(':')
            group=':'.join(parts[:2])
            groups.setdefault(group,[]).append(index)
        report['object_state_metrics']['normalized_rmse_by_group']={
            group:{'model':np.sqrt(errors_by_dimension[:,:,indices].mean((0,2))).tolist(),
                   'persistence':np.sqrt(persistence_by_dimension[:,:,indices].mean((0,2))).tolist()}
            for group,indices in groups.items()}
        physical_errors=np.stack(object_physical_error_dimensions,axis=0)
        physical_persistence=np.stack(object_physical_persistence_dimensions,axis=0)
        report['object_state_metrics']['physical_rmse_by_group']={
            group:np.sqrt(physical_errors[:,:,indices].mean((0,2))).tolist()
            for group,indices in groups.items()}
        report['object_state_metrics']['physical_persistence_rmse_by_group']={
            group:np.sqrt(physical_persistence[:,:,indices].mean((0,2))).tolist()
            for group,indices in groups.items()}
        report['object_state_metrics']['quaternion_geodesic_rmse_deg_by_object']={
            obj:{'model':np.rad2deg(np.sqrt(np.mean(np.stack([row[0] for row in values]),axis=0))).tolist(),
                 'persistence':np.rad2deg(np.sqrt(np.mean(np.stack([row[1] for row in values]),axis=0))).tolist()}
            for obj,values in object_quaternion_angle_errors.items()}
        if object_observation_errors:
            observation_error=np.stack(object_observation_errors,axis=0)
            observation_physical=np.stack(object_observation_physical_errors,axis=0)
            report['object_state_metrics']['current_image_localization']={
                'normalized_rmse':float(np.sqrt(observation_error.mean())),
                'normalized_rmse_by_group':{
                    group:float(np.sqrt(observation_error[:,indices].mean()))
                    for group,indices in groups.items()},
                'physical_rmse_by_group':{
                    group:float(np.sqrt(observation_physical[:,indices].mean()))
                    for group,indices in groups.items()}}
    return report
