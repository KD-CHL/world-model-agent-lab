"""Episode-isolated RGB-to-stack-block XY perception model."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from wmal.datasets.visual_sequences import _validate_manifest
from wmal.logging.manifest import atomic_json, sha256_file


def _visual_domain_sha256(payload):
    semantics=payload['semantics']
    signature={
        'camera_semantics':{key:semantics[key] for key in
            ('camera','camera_fovy_deg','image_size','period_s','object_state_order')},
        'collection_source':payload['source'],
    }
    encoded=json.dumps(signature,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _red_pixel_count(rgb):
    image=np.asarray(rgb)
    red,green,blue=(image[:,:,i].astype(np.int16) for i in range(3))
    mask=(red>=70)&(red-green>=35)&(red-blue>=35)&(red>=1.7*green)&(red>=1.7*blue)
    return int(mask.sum())


class RGBXYNetwork(nn.Module):
    """Preserve coarse image coordinates while regressing the world XY plane."""
    def __init__(self,image_size):
        super().__init__()
        if type(image_size) is not int or image_size not in (64,128):
            raise ValueError('RGB XY network supports 64 or 128 px images')
        channels=(16,32,64,96)
        layers=[]
        previous=3
        for width in channels:
            layers.extend((nn.Conv2d(previous,width,5 if previous==3 else 3,2,
                                     padding=2 if previous==3 else 1,bias=False),
                           nn.GroupNorm(8,width),nn.SiLU()))
            previous=width
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool2d((8,8)),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(previous*64,192),nn.SiLU(),nn.Dropout(.1),
                                nn.Linear(192,64),nn.SiLU(),nn.Linear(64,2))
        self.image_size=image_size

    def forward(self,rgb):
        if rgb.ndim!=4 or rgb.shape[1:]!=(3,self.image_size,self.image_size):
            raise ValueError('RGB XY model input shape mismatch')
        return self.head(self.encoder(rgb))


@dataclass(frozen=True)
class RGBXYMetadata:
    model_version: str
    dataset_sha256: str
    image_size: int
    camera: str
    camera_fovy_deg: float
    world_mean_xy_m: tuple
    world_scale_xy_m: tuple
    training_episode_ids: tuple
    training_episode_sha256: tuple
    validation_episode_ids: tuple
    validation_episode_sha256: tuple
    frame_stride: int
    max_samples_per_episode: int
    red_pixel_range: tuple
    visual_domain_sha256: str
    schema: str='wmal.rgb_stack_block_xy.v2'

    def __post_init__(self):
        mean=np.asarray(self.world_mean_xy_m,dtype=float)
        scale=np.asarray(self.world_scale_xy_m,dtype=float)
        if (self.schema!='wmal.rgb_stack_block_xy.v2' or not self.model_version
                or not self.dataset_sha256 or mean.shape!=(2,) or scale.shape!=(2,)
                or not isinstance(self.visual_domain_sha256,str)
                or len(self.visual_domain_sha256)!=64
                or any(char not in '0123456789abcdef' for char in self.visual_domain_sha256.lower())
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or np.any(scale<=0) or self.image_size not in (64,128)
                or not self.camera or not math.isfinite(self.camera_fovy_deg)
                or not 1<=self.camera_fovy_deg<179
                or type(self.frame_stride) is not int or self.frame_stride<1
                or type(self.max_samples_per_episode) is not int or self.max_samples_per_episode<1
                or not self.training_episode_ids or not self.validation_episode_ids
                or set(self.training_episode_ids)&set(self.validation_episode_ids)
                or len(self.training_episode_ids)!=len(self.training_episode_sha256)
                or len(self.validation_episode_ids)!=len(self.validation_episode_sha256)
                or tuple(self.red_pixel_range)!=(8,4096)):
            raise ValueError('Invalid RGB XY model metadata')


class RGBBlockXYLocalizer:
    """Load a frozen RGB-only XY regressor for supervised perception or analysis."""
    def __init__(self,checkpoint,*,device='cpu',calibration_path=None):
        self.checkpoint_sha256=sha256_file(checkpoint)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.metadata=RGBXYMetadata(**{**payload['metadata'],
            'world_mean_xy_m':tuple(payload['metadata']['world_mean_xy_m']),
            'world_scale_xy_m':tuple(payload['metadata']['world_scale_xy_m']),
            'training_episode_ids':tuple(payload['metadata']['training_episode_ids']),
            'training_episode_sha256':tuple(payload['metadata']['training_episode_sha256']),
            'validation_episode_ids':tuple(payload['metadata']['validation_episode_ids']),
            'validation_episode_sha256':tuple(payload['metadata']['validation_episode_sha256']),
            'red_pixel_range':tuple(payload['metadata']['red_pixel_range'])})
        self.device=device
        self.network=RGBXYNetwork(self.metadata.image_size).to(device)
        self.network.load_state_dict(payload['state_dict'],strict=True)
        self.network.eval()
        self.calibration=None
        if calibration_path is not None:
            calibration=json.loads(Path(calibration_path).read_text())
            if (calibration.get('schema')!='wmal.rgb_stack_block_xy_calibration.v1'
                    or calibration.get('model_version')!=self.metadata.model_version
                    or calibration.get('checkpoint_sha256')!=self.checkpoint_sha256
                    or calibration.get('training_manifest_sha256')!=self.metadata.dataset_sha256
                    or calibration.get('visual_domain_sha256')!=self.metadata.visual_domain_sha256
                    or not math.isfinite(calibration.get('radius_m',float('nan')))
                    or calibration['radius_m']<0):
                raise ValueError('RGB XY calibration does not match the frozen model')
            self.calibration=calibration

    def predict(self,rgb):
        image=np.asarray(rgb)
        if image.dtype==np.uint8 and image.shape==(self.metadata.image_size,self.metadata.image_size,3):
            if not 8<=_red_pixel_count(image)<=4096:
                raise ValueError('Red block is not confidently visible')
            image=image.transpose(2,0,1).astype(np.float32)/255.
        elif (image.ndim==3 and image.shape==(3,self.metadata.image_size,self.metadata.image_size)
              and np.issubdtype(image.dtype,np.floating) and np.isfinite(image).all()
              and image.min()>=0 and image.max()<=1):
            hwc=np.rint(image.transpose(1,2,0)*255).astype(np.uint8)
            if not 8<=_red_pixel_count(hwc)<=4096:
                raise ValueError('Red block is not confidently visible')
            image=image.astype(np.float32)
        else:
            raise ValueError('RGB XY localizer expects HWC uint8 or CHW float RGB')
        with torch.inference_mode():
            normalized=self.network(torch.as_tensor(image,device=self.device)[None]).cpu().numpy()[0]
        position=normalized*np.asarray(self.metadata.world_scale_xy_m)+np.asarray(self.metadata.world_mean_xy_m)
        result={'position_xy_m':position,'model_version':self.metadata.model_version}
        if self.calibration is not None:
            result['position_xy_calibrated_radius_m']=self.calibration['radius_m']
            result['position_xy_calibration_scope']='stationary_tabletop_episode_max_error'
        return result


class _EpisodeFrameDataset(torch.utils.data.Dataset):
    def __init__(self,manifest_path,split,*,frame_stride,max_samples_per_episode,
                 image_size,position_indices,z_index,linear_indices,angular_indices):
        self.manifest_path=Path(manifest_path).resolve()
        self.root=self.manifest_path.parent
        self.rows=[]
        self.samples=[]
        self.candidate_frames=0
        self.episode_candidate_counts={}
        self.episode_visible_counts={}
        payload=json.loads(self.manifest_path.read_text())
        self.semantics=payload['semantics']
        self.image_size=image_size
        for row in payload['episodes']:
            if row['split']!=split:
                continue
            file=(self.root/row['path']).resolve()
            if not file.is_relative_to(self.root) or sha256_file(file)!=row['sha256']:
                raise ValueError('Episode escapes dataset root or content hash changed')
            self.rows.append(row)
            with np.load(file,allow_pickle=False) as arrays:
                rgb=arrays['rgb']
                truth=arrays['object_truth']
                if (len(rgb)!=len(truth)
                        or rgb.shape[1:]!=(image_size,image_size,3)):
                    raise ValueError('RGB and object labels do not align with image semantics')
                initial_z=float(truth[0,z_index])
                eligible=[]
                for frame in range(0,len(truth),frame_stride):
                    if (abs(float(truth[frame,z_index])-initial_z)>.003
                            or np.linalg.norm(truth[frame,linear_indices])>.01
                            or np.linalg.norm(truth[frame,angular_indices])>.1):
                        continue
                    eligible.append(frame)
                if not eligible:
                    continue
                selected=np.unique(np.rint(np.linspace(0,len(eligible)-1,
                    min(max_samples_per_episode,len(eligible)))).astype(int))
                episode_id=row['episode_id']
                self.candidate_frames+=len(selected)
                self.episode_candidate_counts[episode_id]=len(selected)
                visible=0
                for offset in selected:
                    frame=eligible[int(offset)]
                    image=rgb[frame].copy()
                    if 8<=_red_pixel_count(image)<=4096:
                        self.samples.append((episode_id,image,
                                             truth[frame,position_indices].astype(np.float32)))
                        visible+=1
                self.episode_visible_counts[episode_id]=visible
        if len(self.rows)<6 or len({episode_id for episode_id,_,_ in self.samples})<6:
            raise ValueError(f'At least six episodes with visible tabletop frames are required for {split}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        _,image,target=self.samples[index]
        return image.transpose(2,0,1),target.copy()


def _pixel_dataset(manifest_path,split,*,frame_stride,max_samples_per_episode):
    path=Path(manifest_path).resolve()
    payload=json.loads(path.read_text())
    _validate_manifest(payload)
    semantics=payload['semantics']
    if (semantics.get('camera')!='workcell_overview'
            or 'camera_fovy_deg' not in semantics or 'object_state_order' not in semantics):
        raise ValueError('Manifest lacks stack-block RGB/object camera semantics')
    order=semantics['object_state_order']
    indices=[order.index(f'block_0:position_m:{axis}') for axis in 'xy']
    return payload,_EpisodeFrameDataset(path,split,frame_stride=frame_stride,
        max_samples_per_episode=max_samples_per_episode,image_size=semantics['image_size'],
        position_indices=indices,z_index=order.index('block_0:position_m:z'),
        linear_indices=[order.index(f'block_0:linear_velocity_m_s:{a}') for a in 'xyz'],
        angular_indices=[order.index(f'block_0:angular_velocity_rad_s:{a}') for a in 'xyz'])


def _evaluate(network,dataset,mean,scale,device,batch_size):
    network.eval()
    errors={}
    with torch.inference_mode():
        for start in range(0,len(dataset),batch_size):
            images,targets=zip(*(dataset[i] for i in range(start,min(start+batch_size,len(dataset)))))
            inputs=torch.as_tensor(np.stack(images),dtype=torch.float32,device=device)/255.
            expected=np.stack(targets)
            predicted=network(inputs).cpu().numpy()*scale+mean
            for key,err in zip((dataset.samples[i][0]
                                for i in range(start,min(start+batch_size,len(dataset)))),
                               predicted-expected):
                errors.setdefault(key,[]).append(err)
    per_episode={key:np.stack(values) for key,values in errors.items()}
    frame_errors=np.concatenate(list(per_episode.values()),axis=0)
    ep_planar=[np.linalg.norm(value,axis=-1).mean() for value in per_episode.values()]
    return {'episodes':len(per_episode),'frames':len(frame_errors),
            'candidate_episodes':len(dataset.episode_candidate_counts),
            'candidate_frames':dataset.candidate_frames,
            'visible_frames':len(dataset.samples),
            'visible_frame_detection_rate':float(len(dataset.samples)/dataset.candidate_frames),
            'rmse_xy_m':np.sqrt(np.mean(frame_errors**2,axis=0)).tolist(),
            'mean_planar_error_m':float(np.linalg.norm(frame_errors,axis=-1).mean()),
            'median_planar_error_m':float(np.median(np.linalg.norm(frame_errors,axis=-1))),
            'p95_planar_error_m':float(np.quantile(np.linalg.norm(frame_errors,axis=-1),.95)),
            'max_planar_error_m':float(np.linalg.norm(frame_errors,axis=-1).max()),
            'episode_mean_planar_error_m':float(np.mean(ep_planar)),
            'fraction_within_10mm':float(np.mean(np.linalg.norm(frame_errors,axis=-1)<=.01)),
            'per_episode_mean_planar_error_m':{key:float(np.linalg.norm(value,axis=-1).mean())
                                               for key,value in sorted(per_episode.items())},
            'per_episode_max_planar_error_m':{key:float(np.linalg.norm(value,axis=-1).max())
                                              for key,value in sorted(per_episode.items())},
            'per_episode_detection_rate':{key:float(dataset.episode_visible_counts.get(key,0)/count)
                for key,count in sorted(dataset.episode_candidate_counts.items()) if count}}


def train_rgb_xy_localizer(manifest_path,output_dir,*,epochs=20,batch_size=128,
                           frame_stride=16,max_samples_per_episode=32,learning_rate=3e-4,
                           seed=2411,device='cuda'):
    if (type(epochs) is not int or epochs<1 or type(batch_size) is not int or batch_size<1
            or type(frame_stride) is not int or frame_stride<1
            or type(max_samples_per_episode) is not int or max_samples_per_episode<1
            or type(seed) is not int or seed<0 or not math.isfinite(learning_rate)
            or learning_rate<=0):
        raise ValueError('Invalid RGB XY training parameters')
    output=Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('RGB XY training output must be empty')
    payload,train=_pixel_dataset(manifest_path,'train',frame_stride=frame_stride,
                                 max_samples_per_episode=max_samples_per_episode)
    _,validation=_pixel_dataset(manifest_path,'validation',frame_stride=frame_stride,
                                max_samples_per_episode=max_samples_per_episode)
    semantics=payload['semantics']
    if train.image_size!=validation.image_size:
        raise ValueError('Train and validation image sizes differ')
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    train_targets=np.stack([target for _,_,target in train.samples]).astype(np.float64)
    mean=train_targets.mean(axis=0)
    scale=np.maximum(train_targets.std(axis=0),.01)
    normalized_train=(train_targets-mean)/scale
    validation_targets=np.stack([target for _,_,target in validation.samples]).astype(np.float64)
    normalized_validation=(validation_targets-mean)/scale
    network=RGBXYNetwork(semantics['image_size']).to(device)
    optimizer=torch.optim.AdamW(network.parameters(),lr=learning_rate,weight_decay=1e-5)
    output.mkdir(parents=True,exist_ok=True)
    history=[]
    best_validation=float('inf')
    best_path=output/'best.pt'
    for epoch in range(epochs):
        network.train()
        permutation=rng.permutation(len(train))
        train_loss=[]
        for start in range(0,len(permutation),batch_size):
            indices=permutation[start:start+batch_size]
            batch=[train[int(i)] for i in indices]
            images=torch.as_tensor(np.stack([row[0] for row in batch]),dtype=torch.float32,device=device)/255.
            targets=torch.as_tensor(normalized_train[indices],dtype=torch.float32,device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction=network(images)
            loss=F.smooth_l1_loss(prediction,targets,beta=.5)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite RGB XY training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            train_loss.append(float(loss.detach()))
        metrics=_evaluate(network,validation,mean,scale,device,batch_size)
        row={'epoch':epoch+1,'training_smooth_l1':float(np.mean(train_loss)),
             'validation':metrics}
        history.append(row)
        if metrics['episode_mean_planar_error_m']<best_validation:
            best_validation=metrics['episode_mean_planar_error_m']
            train_ids=sorted({episode_id for episode_id,_,_ in train.samples})
            validation_ids=sorted({episode_id for episode_id,_,_ in validation.samples})
            train_hashes={row['episode_id']:row['sha256'] for row in train.rows}
            validation_hashes={row['episode_id']:row['sha256'] for row in validation.rows}
            metadata=RGBXYMetadata(
                model_version=f'rgbxy-{seed}-{epoch+1}-{sha256_file(manifest_path)[:12]}',
                dataset_sha256=sha256_file(manifest_path),image_size=semantics['image_size'],
                camera=semantics['camera'],camera_fovy_deg=semantics['camera_fovy_deg'],
                world_mean_xy_m=tuple(mean.tolist()),world_scale_xy_m=tuple(scale.tolist()),
                training_episode_ids=tuple(train_ids),
                training_episode_sha256=tuple(train_hashes[key] for key in train_ids),
                validation_episode_ids=tuple(validation_ids),
                validation_episode_sha256=tuple(validation_hashes[key] for key in validation_ids),
                frame_stride=frame_stride,max_samples_per_episode=max_samples_per_episode,
                red_pixel_range=(8,4096),visual_domain_sha256=_visual_domain_sha256(payload))
            torch.save({'state_dict':network.cpu().state_dict(),'metadata':asdict(metadata),
                        'training_config':{'epochs':epochs,'batch_size':batch_size,
                            'frame_stride':frame_stride,'max_samples_per_episode':max_samples_per_episode,
                            'learning_rate':learning_rate,'seed':seed},
                        'selected_epoch':epoch+1},best_path)
            network.to(device)
        atomic_json(output/'history.json',history)
    return {'checkpoint':str(best_path),'best_validation_episode_mean_planar_error_m':best_validation,
            'selected_epoch':torch.load(best_path,map_location='cpu',weights_only=False)['selected_epoch'],
            'training_samples':len(train),'validation_samples':len(validation),
            'training_episodes':len(train.rows),'validation_episodes':len(validation.rows),
            'dataset_sha256':sha256_file(manifest_path)}


def _load_localizer_split(manifest_path,checkpoint_path,split,*,frame_stride,
                          max_samples_per_episode,device):
    localizer=RGBBlockXYLocalizer(checkpoint_path,device=device)
    payload,dataset=_pixel_dataset(manifest_path,split,frame_stride=frame_stride,
        max_samples_per_episode=max_samples_per_episode)
    meta=localizer.metadata
    if (payload['semantics']['camera']!=meta.camera
            or payload['semantics']['image_size']!=meta.image_size
            or payload['semantics']['camera_fovy_deg']!=meta.camera_fovy_deg
            or _visual_domain_sha256(payload)!=meta.visual_domain_sha256):
        raise ValueError('Evaluation camera semantics differ from the frozen localizer')
    evaluation_ids={row['episode_id'] for row in dataset.rows}
    if evaluation_ids&set(meta.training_episode_ids):
        raise ValueError('Localizer evaluation overlaps training episodes')
    if split in ('calibration','test') and evaluation_ids&set(meta.validation_episode_ids):
        raise ValueError('Final localizer evaluation overlaps model-selection episodes')
    return localizer,payload,dataset


def calibrate_rgb_xy_localizer(manifest_path,checkpoint_path,output_path,*,alpha=.2,
                               frame_stride=16,max_samples_per_episode=32,
                               device='cuda',batch_size=128):
    if not math.isfinite(alpha) or not 0<alpha<1:
        raise ValueError('Conformal alpha must lie strictly between 0 and 1')
    localizer,payload,dataset=_load_localizer_split(manifest_path,checkpoint_path,'calibration',
        frame_stride=frame_stride,max_samples_per_episode=max_samples_per_episode,device=device)
    meta=localizer.metadata
    result=_evaluate(localizer.network,dataset,np.asarray(meta.world_mean_xy_m),
                     np.asarray(meta.world_scale_xy_m),device,batch_size)
    maxima=result['per_episode_max_planar_error_m']
    ordered=sorted(maxima.values())
    rank=math.ceil((len(ordered)+1)*(1-alpha))
    if rank>len(ordered):
        raise ValueError('Too few calibration episodes for a finite conformal radius at this alpha')
    result.update({'schema':'wmal.rgb_stack_block_xy_calibration.v1',
        'method':'episode_max_split_conformal','alpha':alpha,'target_episode_coverage':1-alpha,
        'scope':'stationary_tabletop_episode_max_error',
        'checkpoint_sha256':localizer.checkpoint_sha256,
        'radius_m':float(ordered[rank-1]),'calibration_manifest_sha256':sha256_file(manifest_path),
        'visual_domain_sha256':localizer.metadata.visual_domain_sha256,
        'calibration_episode_ids':sorted(dataset.episode_candidate_counts),
        'calibration_score_episode_ids':sorted(maxima),
        'model_version':localizer.metadata.model_version,
        'training_manifest_sha256':localizer.metadata.dataset_sha256,
        'training_episode_ids':list(localizer.metadata.training_episode_ids),
        'model_selection_episode_ids':list(localizer.metadata.validation_episode_ids),
        'candidate_episodes':len(dataset.episode_candidate_counts),
        'visible_episodes':result['episodes'],
        'coverage_rank_one_based':rank})
    atomic_json(output_path,result)
    return result


def evaluate_rgb_xy_localizer(manifest_path,checkpoint_path,output_path,*,split='validation',
                              frame_stride=16,max_samples_per_episode=32,device='cuda',batch_size=128,
                              calibration_path=None):
    if split not in ('validation','calibration','test'):
        raise ValueError('Evaluation split must be validation/calibration/test')
    localizer,payload,dataset=_load_localizer_split(manifest_path,checkpoint_path,split,
        frame_stride=frame_stride,max_samples_per_episode=max_samples_per_episode,device=device)
    meta=localizer.metadata
    result=_evaluate(localizer.network,dataset,np.asarray(meta.world_mean_xy_m),
                     np.asarray(meta.world_scale_xy_m),device,batch_size)
    result.update({'schema':'wmal.rgb_stack_block_xy_evaluation.v1','split':split,
        'model_version':meta.model_version,'manifest_sha256':sha256_file(manifest_path),
        'training_episode_ids':list(meta.training_episode_ids),
        'model_selection_episode_ids':list(meta.validation_episode_ids),
        'training_manifest_sha256':meta.dataset_sha256})
    if calibration_path is not None:
        calibration=json.loads(Path(calibration_path).read_text())
        if (calibration.get('schema')!='wmal.rgb_stack_block_xy_calibration.v1'
                or calibration.get('model_version')!=meta.model_version
                or calibration.get('checkpoint_sha256')!=localizer.checkpoint_sha256
                or calibration.get('training_manifest_sha256')!=meta.dataset_sha256
                or calibration.get('visual_domain_sha256')!=meta.visual_domain_sha256):
            raise ValueError('Calibration artifact does not match the frozen localizer')
        evaluation_ids=set(dataset.episode_candidate_counts)
        calibration_ids=set(calibration['calibration_episode_ids'])
        if evaluation_ids&calibration_ids:
            raise ValueError('Final evaluation overlaps calibration episodes')
        episode_errors=result['per_episode_max_planar_error_m']
        accepted_ids=set(episode_errors)
        within={key for key,value in episode_errors.items() if value<=calibration['radius_m']}
        candidate_count=len(evaluation_ids)
        result.update({'calibration_radius_m':calibration['radius_m'],
            'calibration_alpha':calibration['alpha'],
            'calibration_episode_ids':sorted(calibration_ids),
            'bound_coverage_on_visible_episodes':float(len(within)/len(accepted_ids)),
            'usable_episode_fraction_within_bound':float(len(within)/candidate_count),
            'candidate_episodes':candidate_count,'visible_episodes':len(accepted_ids)})
    atomic_json(output_path,result)
    return result
