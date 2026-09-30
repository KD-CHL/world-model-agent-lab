"""Frozen episode-level task-state prediction intervals, NOT safety probabilities."""
from dataclasses import asdict, dataclass
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


def calibrate_visual(manifest, checkpoint, output, *, horizon=4, alpha=.1, std_floor=.05, device='cpu'):
    from wmal.datasets.visual_sequences import VisualDataset, read_episode_lineage, assert_unexposed_episodes
    from wmal.models.visual_latent import VisualWorldModel
    model=VisualWorldModel.load(checkpoint,device=device)
    dataset=VisualDataset(manifest,'calibration',horizon=horizon)
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
        result=model.predict(sample['rgb'][0],sample['states'][0],sample['actions'])
        denominator=np.maximum(result['states'].std(0),std_floor*scale)
        scores=(np.abs(result['states'].mean(0)-sample['states'][1:])/denominator).max(-1)
        rows.append({'episode_id':sample['episode_id'],'split':'calibration','scores':scores.tolist()})
    excluded=lineage['train']['episode_ids']+lineage['selection']['episode_ids']
    excluded += [r['episode_id'] for r in dataset.manifest['episodes'] if r['split']=='test']
    calibration=fit_horizon_calibration(rows,model_version=model.version,dataset_id=sha256_file(manifest),
                 alpha=alpha,state_scale=scale.tolist(),std_floor=std_floor,semantics=model.semantics,
                 excluded_episode_ids=excluded)
    calibration.save(output)
    return calibration
