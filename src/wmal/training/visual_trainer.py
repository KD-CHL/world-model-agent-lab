"""Independent visual world-model training, fine-tuning and held-out evaluation."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from wmal.datasets.visual_sequences import VisualDataset, read_episode_lineage, assert_unexposed_episodes
from wmal.logging.manifest import atomic_json, sha256_file
from wmal.models.visual_latent import NetworkConfig, VisualLatentMember, VisualWorldModel


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

    def __post_init__(self):
        if (any(type(v) is not int or v<1 for v in (self.epochs,self.members,self.batch_size,
                self.horizon,self.latent_dim,self.hidden_dim,self.threads)) or self.members<2
                or type(self.seed) is not int or self.seed<0
                or any(not math.isfinite(v) or v<0 for v in (self.learning_rate,self.weight_decay,
                   self.frame_weight,self.state_weight,self.latent_weight,self.event_weight,self.change_weight))
                or self.learning_rate<=0 or not math.isfinite(self.gradient_clip) or self.gradient_clip<=0
                or self.frame_weight<=0 or self.state_weight<=0):
            raise ValueError('Invalid visual training configuration')


def batch_loss(member, batch, normalization, config):
    device=next(member.parameters()).device
    rgb,state,action=[batch[key].to(device) for key in ('rgb','states','actions')]
    norm={key:torch.as_tensor(value,dtype=torch.float32,device=device) for key,value in normalization.items()}
    state=(state-norm['state_mean'])/norm['state_scale']
    action=(action-norm['action_mean'])/norm['action_scale']
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
    total=config.frame_weight*pixel+config.state_weight*state_loss+config.latent_weight*latent+config.event_weight*event
    return total,{'frame':float(pixel.detach()),'state':float(state_loss.detach()),
                  'latent':float(latent.detach()),'event':float(event.detach())}


def train_visual(manifest, output_dir, config=VisualTrainingConfig(), *, pretrained=None, freeze_encoder=False):
    output=Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Training output is nonempty; use a new directory')
    if freeze_encoder and pretrained is None:
        raise ValueError('Cannot freeze a randomly initialized encoder')
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    if config.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    train=VisualDataset(manifest,'train',horizon=config.horizon)
    validation=VisualDataset(manifest,'validation',horizon=config.horizon)
    semantics=train.semantics
    network=NetworkConfig(len(semantics['state_order']),len(semantics['action_order']),
                          image_size=semantics['image_size'],latent_dim=config.latent_dim,
                          hidden_dim=config.hidden_dim,event_dim=len(semantics.get('event_names',[])))
    normalization=train.normalization()
    lineage={'schema':'wmal.episode_lineage.v1',
             'train':{'episode_ids':[],'sha256':[]},'selection':{'episode_ids':[],'sha256':[]}}
    if pretrained is None:
        members=[VisualLatentMember(network).to(config.device) for _ in range(config.members)]
        parent_version=None
    else:
        parent=VisualWorldModel.load(pretrained,device=config.device)
        if parent.config!=network or parent.semantics!=semantics or len(parent.members)!=config.members:
            raise ValueError('Fine-tuning architecture/control semantics do not match pretrained checkpoint')
        lineage=read_episode_lineage(parent.metadata)
        assert_unexposed_episodes(validation.rows,[lineage['train']],context='Validation')
        members=parent.members
        normalization=parent.normalization  # Freeze coordinates as well as learned backbone.
        parent_version=parent.version
    for kind,rows in (('train',train.rows),('selection',validation.rows)):
        for key in ('episode_ids','sha256'):
            field='episode_id' if key=='episode_ids' else 'sha256'
            lineage[kind][key]=sorted(set(lineage[kind][key])|{r[field] for r in rows})
    for member in members:
        for parameter in member.encoder.parameters():
            parameter.requires_grad_(not freeze_encoder)
    optimizers=[torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                 lr=config.learning_rate,weight_decay=config.weight_decay) for m in members]
    # Resample independent EPISODES, preserving all within-episode windows.
    by_episode={i:[j for j,(episode,_) in enumerate(train.index) if episode==i] for i in range(len(train.rows))}
    rng=np.random.default_rng(config.seed)
    loaders=[]
    for i in range(config.members):
        episode_indices=rng.integers(0,len(train.rows),len(train.rows))
        indices=[j for episode in episode_indices for j in by_episode[int(episode)]]
        if not indices:
            raise ValueError('Bootstrap contains no valid training windows')
        loaders.append(DataLoader(Subset(train,indices),batch_size=config.batch_size,shuffle=True,
                                  generator=torch.Generator().manual_seed(config.seed+i),num_workers=0))
    val_loader=DataLoader(validation,batch_size=config.batch_size,shuffle=False,num_workers=0)
    output.mkdir(parents=True,exist_ok=True)
    metadata={'max_horizon':config.horizon,'dataset_manifest_sha256':sha256_file(manifest),
              'training_source':train.manifest['source'],
              'training_episode_ids':[r['episode_id'] for r in train.rows],
              'selection_episode_ids':[r['episode_id'] for r in validation.rows],
              'parent_model_version':parent_version,'episode_lineage':lineage,'freeze_encoder':freeze_encoder,
              'event_supervision':bool(network.event_dim),'config':asdict(config)}
    history,best,started=[],float('inf'),time.perf_counter()
    for epoch in range(config.epochs):
        train_total,train_count=0.,0
        for member,optimizer,loader in zip(members,optimizers,loaders):
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
        val_total,val_count=0.,0
        components={'frame':0.,'state':0.,'latent':0.,'event':0.}
        with torch.inference_mode():
            for member in members:
                member.eval()
                for batch in val_loader:
                    loss,parts=batch_loss(member,batch,normalization,config)
                    size=len(batch['rgb'])
                    val_total+=float(loss)*size
                    val_count+=size
                    for key,value in parts.items():
                        components[key]+=value*size
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
    selected=VisualWorldModel.load(output/'best.pt')
    report={'schema':'wmal.visual_training_report.v1','model_version':selected.version,
            'best_validation_loss':best,'history':history,'elapsed_s':time.perf_counter()-started,
            'checkpoint':str(output/'best.pt'),'metadata':metadata,
            'parameters_per_member':sum(p.numel() for p in members[0].parameters()),
            'trainable_parameters_per_member':sum(p.numel() for p in members[0].parameters() if p.requires_grad)}
    atomic_json(output/'training_report.json',report)
    return report


def evaluate_visual(manifest, checkpoint, *, horizon=4, device='cpu', calibration=None):
    from wmal.models.horizon_calibration import HorizonCalibration
    model=VisualWorldModel.load(checkpoint,device=device)
    dataset=VisualDataset(manifest,'test',horizon=horizon)
    if dataset.semantics!=model.semantics:
        raise ValueError('Evaluation dataset/control semantics mismatch')
    lineage=read_episode_lineage(model.metadata)
    assert_unexposed_episodes(dataset.rows,[lineage['train'],lineage['selection']],context='Evaluation')
    bound_model=HorizonCalibration.load(calibration) if calibration else None
    if bound_model:
        bound_model.validate_for(model.version,model.semantics)
        if set(bound_model.episode_ids).intersection(r['episode_id'] for r in dataset.rows):
            raise ValueError('Calibration overlaps test episodes')
    frame,state,persist_frame,persist_state,sensitivity=[],[],[],[],[]
    frame_sensitivity,change_error,change_persistence=[],[],[]
    coverage={}
    for i in range(len(dataset)):
        sample=dataset[i]
        predicted=model.predict(sample['rgb'][0],sample['states'][0],sample['actions'])
        mean=predicted['states'].mean(0)
        frames=predicted['frames'].mean(0)
        frame.append(((frames-sample['rgb'][1:])**2).mean((1,2,3)))
        state.append(((mean-sample['states'][1:])**2).mean(-1))
        persist_frame.append(((sample['rgb'][:1]-sample['rgb'][1:])**2).mean((1,2,3)))
        persist_state.append(((sample['states'][:1]-sample['states'][1:])**2).mean(-1))
        # Counterfactual sensitivity, not an assertion that zero-action targets are known.
        counter=model.predict(sample['rgb'][0],sample['states'][0],np.zeros_like(sample['actions']))
        sensitivity.append(np.abs(mean-counter['states'].mean(0)).mean(-1))
        frame_sensitivity.append(np.abs(frames-counter['frames'].mean(0)).mean((1,2,3)))
        changed=(np.abs(sample['rgb'][1:]-sample['rgb'][:1]).max(1,keepdims=True)>.02)
        count=np.broadcast_to(changed,frames.shape).sum((1,2,3))
        change_error.append((((frames-sample['rgb'][1:])**2)*changed).sum((1,2,3))/np.maximum(count,1))
        change_persistence.append((((sample['rgb'][:1]-sample['rgb'][1:])**2)*changed).sum((1,2,3))/np.maximum(count,1))
        if bound_model:
            bounds=bound_model.bounds(predicted['states'].std(0),version=model.version)
            covered=np.all(np.abs(mean-sample['states'][1:])<=bounds,axis=-1)
            key=sample['episode_id']
            coverage[key]=covered if key not in coverage else coverage[key]&covered
    report={'schema':'wmal.visual_evaluation.v1','split':'test','model_version':model.version,
            'manifest_sha256':sha256_file(manifest),'episodes':len(dataset.rows),'windows':len(dataset),
            'frame_mse_by_horizon':np.mean(frame,0).tolist(),
            'state_rmse_by_horizon':np.sqrt(np.mean(state,0)).tolist(),
            'persistence_frame_mse_by_horizon':np.mean(persist_frame,0).tolist(),
            'persistence_state_rmse_by_horizon':np.sqrt(np.mean(persist_state,0)).tolist(),
            'zero_action_sensitivity_by_horizon':np.mean(sensitivity,0).tolist(),
            'zero_action_frame_sensitivity_by_horizon':np.mean(frame_sensitivity,0).tolist(),
            'changed_region_frame_mse_by_horizon':np.mean(change_error,0).tolist(),
            'persistence_changed_region_frame_mse_by_horizon':np.mean(change_persistence,0).tolist(),
            'episode_simultaneous_coverage_by_horizon':np.mean(list(coverage.values()),0).tolist() if coverage else None}
    return report
