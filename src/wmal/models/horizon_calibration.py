"""Frozen episode-level task-state prediction intervals, NOT safety probabilities."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import numpy as np
from wmal.logging.manifest import atomic_json, sha256_file


@dataclass(frozen=True)
class HorizonCalibration:
    model_version: str
    dataset_id: str
    alpha: float
    quantiles: tuple
    state_scale: tuple
    std_floor: float
    episode_ids: tuple
    semantics: dict
    schema: str = 'wmal.horizon_calibration.v1'

    def __post_init__(self):
        if (self.schema!='wmal.horizon_calibration.v1' or not self.model_version or not self.dataset_id
                or not 0<self.alpha<1 or not self.quantiles or not self.state_scale
                or not self.episode_ids or len(set(self.episode_ids))!=len(self.episode_ids)
                or not np.isfinite(self.std_floor) or self.std_floor<=0
                or not np.isfinite(self.quantiles).all() or min(self.quantiles)<0
                or not np.isfinite(self.state_scale).all() or min(self.state_scale)<=0):
            raise ValueError('Invalid horizon calibration')

    def validate_for(self,version,semantics):
        if version!=self.model_version or semantics!=self.semantics:
            raise ValueError('Calibration model/control semantics mismatch')

    def bounds(self, std, *, version):
        std=np.asarray(std)
        if (version!=self.model_version or std.ndim!=2 or not 1<=len(std)<=len(self.quantiles)
                or std.shape[1]!=len(self.state_scale) or not np.isfinite(std).all() or np.any(std<0)):
            raise ValueError('Invalid horizon uncertainty or model version')
        floor=self.std_floor*np.asarray(self.state_scale)
        return np.asarray(self.quantiles[:len(std)])[:,None]*np.maximum(std,floor)

    def save(self,path):
        atomic_json(path,asdict(self))

    @classmethod
    def load(cls,path):
        payload=json.loads(Path(path).read_text())
        for key in ('quantiles','state_scale','episode_ids'):
            payload[key]=tuple(payload[key])
        return cls(**payload)


@dataclass(frozen=True)
class ObjectHorizonCalibration:
    """Episode-level conformal bounds for named predicted object states."""
    model_version: str
    dataset_id: str
    alpha: float
    quantiles: tuple
    object_scale: tuple
    std_floor: float
    episode_ids: tuple
    semantics: dict
    group_quantiles: dict = field(default_factory=dict)
    schema: str = 'wmal.object_horizon_calibration.v2'

    def __post_init__(self):
        if (self.schema not in ('wmal.object_horizon_calibration.v1','wmal.object_horizon_calibration.v2')
                or not self.model_version or not self.dataset_id
                or not 0<self.alpha<1 or not self.quantiles or not self.object_scale
                or not self.episode_ids or len(set(self.episode_ids))!=len(self.episode_ids)
                or not np.isfinite(self.std_floor) or self.std_floor<=0
                or not np.isfinite(self.quantiles).all() or min(self.quantiles)<0
                or not np.isfinite(self.object_scale).all() or min(self.object_scale)<=0):
            raise ValueError('Invalid object horizon calibration')
        if len(self.object_scale)!=len(self.semantics.get('object_state_order',[])):
            raise ValueError('Object calibration dimensions do not match named semantics')
        for name,values in self.group_quantiles.items():
            if (not isinstance(name,str) or not name or len(values)!=len(self.quantiles)
                    or not np.isfinite(values).all() or min(values)<0):
                raise ValueError('Invalid group-specific object calibration quantile')
        if self.schema.endswith('.v2') and not self.group_quantiles:
            raise ValueError('Group-wise object calibration quantiles are required')

    def validate_for(self,version,semantics):
        if version!=self.model_version or semantics!=self.semantics:
            raise ValueError('Object calibration model/target semantics mismatch')

    def bounds(self,std,*,version,object_state_order=None):
        std=np.asarray(std)
        if (version!=self.model_version or std.ndim!=2 or not 1<=len(std)<=len(self.quantiles)
                or std.shape[1]!=len(self.object_scale) or not np.isfinite(std).all() or np.any(std<0)):
            raise ValueError('Invalid object uncertainty or model version')
        floor=self.std_floor*np.asarray(self.object_scale)
        if self.group_quantiles:
            names=object_state_order or self.semantics['object_state_order']
            if len(names)!=len(self.object_scale):
                raise ValueError('Object calibration named dimensions mismatch')
            bounds=np.empty_like(std,dtype=float)
            for group,quantiles in self.group_quantiles.items():
                indices=[i for i,name in enumerate(names) if ':'.join(name.split(':')[:2])==group]
                if not indices:
                    raise ValueError('Object calibration group absent from target semantics')
                bounds[:,indices]=np.asarray(quantiles[:len(std)])[:,None]*np.maximum(std[:,indices],floor[indices])
            return bounds
        return np.asarray(self.quantiles[:len(std)])[:,None]*np.maximum(std,floor)

    def save(self,path):
        atomic_json(path,asdict(self))

    @classmethod
    def load(cls,path):
        payload=json.loads(Path(path).read_text())
        for key in ('quantiles','object_scale','episode_ids'):
            payload[key]=tuple(payload[key])
        return cls(**payload)


def fit_horizon_calibration(rows, *, model_version, dataset_id, alpha, state_scale,
                            std_floor, semantics, excluded_episode_ids=()):
    if not np.isfinite(alpha) or not 0<alpha<1:
        raise ValueError('Invalid miscoverage level')
    maxima={}
    excluded=set(excluded_episode_ids)
    width=None
    for row in rows:
        scores=np.asarray(row['scores'],dtype=float)
        episode=row['episode_id']
        if (row['split']!='calibration' or episode in excluded or not isinstance(episode,str) or not episode
                or scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all() or np.any(scores<0)):
            raise ValueError('Invalid/overlapping calibration record')
        width=len(scores) if width is None else width
        if len(scores)!=width:
            raise ValueError('Calibration horizons differ')
        maxima[episode]=np.maximum(maxima.get(episode,np.zeros(width)),scores)
    n=len(maxima)
    rank=math.ceil((n+1)*(1-alpha))
    if n<2 or rank>n:
        raise ValueError('Not enough independent calibration episodes for finite requested quantile')
    quantiles=np.sort(np.stack(list(maxima.values())),axis=0)[rank-1]
    # Nondecreasing envelope: reliable prefixes cannot become reliable again after a failed step.
    quantiles=np.maximum.accumulate(quantiles)
    return HorizonCalibration(model_version,dataset_id,alpha,tuple(quantiles.tolist()),
                              tuple(state_scale),std_floor,tuple(sorted(maxima)),semantics)


def calibrate_visual(manifest, checkpoint, output, *, horizon=4, alpha=.1, std_floor=.05,
                     device='cpu', window_stride=1):
    from wmal.datasets.visual_sequences import VisualDataset, read_episode_lineage, assert_unexposed_episodes
    from wmal.models.visual_latent import VisualWorldModel
    model=VisualWorldModel.load(checkpoint,device=device)
    if type(window_stride) is not int or window_stride<1:
        raise ValueError('State calibration window stride must be a positive integer')
    dataset=VisualDataset(manifest,'calibration',horizon=horizon,context_steps=model.config.context_steps)
    if dataset.semantics!=model.semantics:
        raise ValueError('Calibration control semantics mismatch')
    lineage=read_episode_lineage(model.metadata)
    assert_unexposed_episodes(dataset.rows,[lineage['train'],lineage['selection']],context='Calibration')
    if not np.isfinite(std_floor) or std_floor<=0:
        raise ValueError('Invalid standard deviation floor')
    scale=np.asarray(model.normalization['state_scale'])
    rows=[]
    for i in range(len(dataset)):
        sample=dataset[i]
        if sample['offset']%window_stride:
            continue
        result=model.predict_sample(sample)
        denominator=np.maximum(result['states'].std(0),std_floor*scale)
        scores=(np.abs(result['states'].mean(0)-sample['states'][model.config.context_steps+1:])/denominator).max(-1)
        rows.append({'episode_id':sample['episode_id'],'split':'calibration','scores':scores.tolist()})
    excluded=lineage['train']['episode_ids']+lineage['selection']['episode_ids']
    excluded += [r['episode_id'] for r in dataset.manifest['episodes'] if r['split']=='test']
    calibration=fit_horizon_calibration(rows,model_version=model.version,dataset_id=sha256_file(manifest),
                 alpha=alpha,state_scale=scale.tolist(),std_floor=std_floor,semantics=model.semantics,
                 excluded_episode_ids=excluded)
    calibration.save(output)
    return calibration


def calibrate_object_visual(manifest,checkpoint,output,*,horizon=4,alpha=.2,
                            std_floor=.05,device='cpu',window_stride=32):
    """Calibrate object prediction intervals with episode maxima, never framewise splits."""
    from wmal.datasets.visual_sequences import VisualDataset, read_episode_lineage, assert_unexposed_episodes
    from wmal.models.visual_latent import VisualWorldModel
    model=VisualWorldModel.load(checkpoint,device=device)
    if type(window_stride) is not int or window_stride<1:
        raise ValueError('Object calibration window stride must be a positive integer')
    if not model.config.object_dim:
        raise ValueError('Checkpoint has no named object-state prediction head')
    dataset=VisualDataset(manifest,'calibration',horizon=horizon,context_steps=model.config.context_steps)
    if dataset.semantics!=model.semantics:
        raise ValueError('Calibration object semantics mismatch')
    lineage=read_episode_lineage(model.metadata)
    assert_unexposed_episodes(dataset.rows,[lineage['train'],lineage['selection']],context='Object calibration')
    if not np.isfinite(std_floor) or std_floor<=0:
        raise ValueError('Invalid object standard-deviation floor')
    scale=np.asarray(model.normalization['object_scale'])
    names=model.semantics['object_state_order']
    groups={}
    for index,name in enumerate(names):
        group=':'.join(name.split(':')[:2])
        groups.setdefault(group,[]).append(index)
    maxima={row['episode_id']:{group:np.zeros(horizon,dtype=float) for group in groups}
            for row in dataset.rows}
    for index in range(len(dataset)):
        sample=dataset[index]
        if sample['offset']%window_stride:
            continue
        prediction=model.predict_sample(sample)
        mean=prediction['object_states'].mean(0)
        std=prediction['object_states'].std(0)
        k=model.config.context_steps
        target=sample['object_truth'][k+1:]
        denominator=np.maximum(std,std_floor*scale)
        normalized=np.abs(mean-target)/denominator
        for group,indices in groups.items():
            scores=normalized[:,indices].max(-1)
            maxima[sample['episode_id']][group]=np.maximum(maxima[sample['episode_id']][group],scores)
    n=len(maxima)
    rank=math.ceil((n+1)*(1-alpha))
    if n<2 or rank>n:
        raise ValueError('Not enough independent calibration episodes for requested object quantile')
    group_quantiles={group:np.maximum.accumulate(np.sort(np.stack(
        [entry[group] for entry in maxima.values()]),axis=0)[rank-1]) for group in groups}
    quantiles=np.maximum.reduce(list(group_quantiles.values()))
    excluded=lineage['train']['episode_ids']+lineage['selection']['episode_ids']
    excluded += [row['episode_id'] for row in dataset.manifest['episodes'] if row['split']=='test']
    if set(maxima).intersection(excluded):
        raise ValueError('Object calibration episodes overlap model exposure/test')
    result=ObjectHorizonCalibration(
        model.version,sha256_file(manifest),alpha,tuple(quantiles.tolist()),tuple(scale.tolist()),
        std_floor,tuple(sorted(maxima)),model.semantics,
        group_quantiles={key:tuple(value.tolist()) for key,value in group_quantiles.items()})
    result.save(output)
    return result
