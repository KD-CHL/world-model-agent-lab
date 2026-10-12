"""Train-split-calibrated RGB-only localization for the red stack block."""
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from wmal.datasets.visual_sequences import _validate_manifest
from wmal.logging.manifest import atomic_json, sha256_file


def red_block_centroid(rgb):
    """Return the largest compact red component center in pixel (u, v) coordinates."""
    image=np.asarray(rgb)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1]!=3:
        raise ValueError('Expected HWC uint8 RGB')
    red,green,blue=(image[:,:,i].astype(np.int16) for i in range(3))
    mask=(red>=70)&(red-green>=35)&(red-blue>=35)&(red>=1.7*green)&(red>=1.7*blue)
    height,width=mask.shape
    visited=np.zeros_like(mask)
    components=[]
    for y,x in zip(*np.nonzero(mask)):
        if visited[y,x]:
            continue
        queue=deque([(int(y),int(x))])
        visited[y,x]=True
        points=[]
        while queue:
            cy,cx=queue.popleft()
            points.append((cy,cx))
            for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1),
                          (cy-1,cx-1),(cy-1,cx+1),(cy+1,cx-1),(cy+1,cx+1)):
                if 0<=ny<height and 0<=nx<width and mask[ny,nx] and not visited[ny,nx]:
                    visited[ny,nx]=True
                    queue.append((ny,nx))
        if 4<=len(points)<=512:
            components.append(points)
    if not components:
        raise ValueError('Red manipulated block not detected')
    points=max(components,key=len)
    return np.asarray([np.mean([x for _,x in points]),np.mean([y for y,_ in points])],dtype=float)


def _normalized_pixel(centroid,image_size):
    uv=np.asarray(centroid,dtype=float)
    return (uv-(image_size-1.)/2.)/max((image_size-1.)/2.,1.)


def _project(homography,centroid,image_size):
    u,v=_normalized_pixel(centroid,image_size)
    point=np.asarray([u,v,1.])@np.asarray(homography,dtype=float).reshape(3,3).T
    if not np.isfinite(point).all() or abs(point[2])<.05:
        raise ValueError('Projective localizer is singular at detected pixel')
    return point[:2]/point[2]


@dataclass(frozen=True)
class StackBlockLocalizer:
    homography: tuple
    world_center_xy_m: tuple
    world_scale_xy_m: tuple
    residual_scale_m: tuple
    image_size: int
    camera: str
    camera_fovy_deg: float
    manifest_sha256: str
    training_episode_ids: tuple
    training_episode_sha256: tuple
    world_bounds_xy_m: tuple
    schema: str='wmal.stack_block_rgb_localizer.v2'

    def __post_init__(self):
        homography=np.asarray(self.homography,dtype=float)
        center=np.asarray(self.world_center_xy_m,dtype=float)
        scale_xy=np.asarray(self.world_scale_xy_m,dtype=float)
        bounds=np.asarray(self.world_bounds_xy_m,dtype=float)
        scale=np.asarray(self.residual_scale_m,dtype=float)
        if (self.schema!='wmal.stack_block_rgb_localizer.v2' or homography.shape!=(3,3)
                or center.shape!=(2,) or scale_xy.shape!=(2,) or np.any(scale_xy<=0)
                or bounds.shape!=(2,2) or np.any(bounds[:,0]>=bounds[:,1])
                or scale.shape!=(2,) or not np.isfinite(homography).all()
                or not np.isfinite(center).all() or not np.isfinite(scale_xy).all()
                or not np.isfinite(scale).all() or np.any(scale<=0)
                or self.image_size not in (32,64,128) or not self.camera
                or not np.isfinite(self.camera_fovy_deg) or not 1<=self.camera_fovy_deg<179
                or not self.manifest_sha256 or not self.training_episode_ids
                or len(self.training_episode_ids)!=len(self.training_episode_sha256)):
            raise ValueError('Invalid stack-block visual localizer calibration')

    def predict(self,rgb):
        image=np.asarray(rgb)
        if image.shape!=(self.image_size,self.image_size,3):
            raise ValueError('RGB image size does not match localizer calibration')
        centroid=red_block_centroid(image)
        normalized=_project(self.homography,centroid,self.image_size)
        xy=normalized*np.asarray(self.world_scale_xy_m)+np.asarray(self.world_center_xy_m)
        bounds=np.asarray(self.world_bounds_xy_m)
        if np.any(xy<bounds[:,0]) or np.any(xy>bounds[:,1]):
            raise ValueError('Detected block projects outside the calibrated work area')
        return {'position_xy_m':xy,'pixel_centroid_uv':centroid,
                'residual_scale_m':np.asarray(self.residual_scale_m)}

    def save(self,path):
        atomic_json(path,asdict(self))

    @classmethod
    def load(cls,path):
        payload=json.loads(Path(path).read_text())
        for key in ('homography','world_center_xy_m','world_scale_xy_m','residual_scale_m',
                    'training_episode_ids','training_episode_sha256','world_bounds_xy_m'):
            payload[key]=tuple(tuple(v) if isinstance(v,list) and key in
                               ('homography','world_bounds_xy_m') else v
                               for v in payload[key])
        return cls(**payload)


def fit_stack_block_localizer(manifest_path,output_path,*,frame_stride=32,tabletop_only=False):
    """Robustly fit an RGB projective ground-plane map on train episodes."""
    if type(frame_stride) is not int or frame_stride<1:
        raise ValueError('frame_stride must be a positive integer')
    manifest_path=Path(manifest_path).resolve()
    payload=json.loads(manifest_path.read_text())
    _validate_manifest(payload)
    semantics=payload['semantics']
    if (semantics['camera']!='workcell_overview' or 'camera_fovy_deg' not in semantics
            or 'object_state_order' not in semantics):
        raise ValueError('Manifest lacks calibrated stack-block camera/object semantics')
    order=semantics['object_state_order']
    x_index=order.index('block_0:position_m:x')
    y_index=order.index('block_0:position_m:y')
    z_index=order.index('block_0:position_m:z')
    linear_indices=[order.index(f'block_0:linear_velocity_m_s:{axis}') for axis in 'xyz']
    angular_indices=[order.index(f'block_0:angular_velocity_rad_s:{axis}') for axis in 'xyz']
    features,targets,ids,hashes=[],[],[],[]
    for row in payload['episodes']:
        if row['split']!='train':
            continue
        path=(manifest_path.parent/row['path']).resolve()
        start=len(features)
        with np.load(path,allow_pickle=False) as arrays:
            if len(arrays['rgb'])!=len(arrays['object_truth']):
                raise ValueError('RGB and object labels do not align')
            for frame in range(0,len(arrays['rgb']),frame_stride):
                if tabletop_only:
                    truth=arrays['object_truth'][frame]
                    baseline_z=float(arrays['object_truth'][0,z_index])
                    if (abs(float(truth[z_index])-baseline_z)>.003
                            or np.linalg.norm(truth[linear_indices])>.01
                            or np.linalg.norm(truth[angular_indices])>.1):
                        continue
                try:
                    centroid=red_block_centroid(arrays['rgb'][frame])
                except ValueError:
                    continue
                features.append(centroid)
                targets.append(arrays['object_truth'][frame,[x_index,y_index]].astype(float))
        if len(features)>start:
            ids.append(row['episode_id'])
            hashes.append(row['sha256'])
    if len(features)<6 or len(ids)<6:
        raise ValueError('At least six independent train episodes are required')
    pixels=np.stack(features)
    pixels=(pixels-(semantics['image_size']-1.)/2.)/max((semantics['image_size']-1.)/2.,1.)
    world=np.stack(targets)
    world_center=np.median(world,axis=0)
    world_scale=np.maximum(np.ptp(world,axis=0)/2.,.005)
    values=(world-world_center)/world_scale
    # Fit an 8-parameter homography in normalized image/world coordinates.
    rows=[]
    for (u,v),(x,y) in zip(pixels,values):
        rows.extend(([u,v,1.,0.,0.,0.,-x*u,-x*v],
                     [0.,0.,0.,u,v,1.,-y*u,-y*v]))
    design=np.asarray(rows,dtype=float)
    rhs=values.reshape(-1)
    # Deterministic four-point RANSAC prevents a few failed red detections from
    # making the projective denominator explode outside one episode's pixels.
    rng=np.random.default_rng(20261013)
    point_count=len(pixels)
    best_inliers=None
    best_count=-1
    for _ in range(768):
        selected=rng.choice(point_count,4,replace=False)
        sample_rows=np.concatenate([design[2*i:2*i+2] for i in selected],axis=0)
        sample_rhs=np.concatenate([rhs[2*i:2*i+2] for i in selected],axis=0)
        try:
            candidate=np.linalg.solve(sample_rows,sample_rhs)
        except np.linalg.LinAlgError:
            continue
        error=(design@candidate-rhs).reshape(-1,2)*world_scale
        inliers=np.linalg.norm(error,axis=1)<=.015
        count=int(inliers.sum())
        if count>best_count:
            best_count,best_inliers=count,inliers
    if best_inliers is None or best_count<max(8,int(.1*point_count)):
        raise ValueError('Projective localizer could not find a stable inlier set')
    selected_points=np.flatnonzero(best_inliers)
    selected_rows=np.concatenate([design[2*i:2*i+2] for i in selected_points],axis=0)
    selected_rhs=np.concatenate([rhs[2*i:2*i+2] for i in selected_points],axis=0)
    coefficients=np.linalg.lstsq(selected_rows,selected_rhs,rcond=None)[0]
    # Huber IRLS limits occlusion and off-plane samples in ground-plane fits.
    for _ in range(12):
        residual=(selected_rows@coefficients-selected_rhs).reshape(-1,2)
        norms=np.linalg.norm(residual,axis=1)
        center=np.median(norms)
        robust_scale=max(1.4826*np.median(np.abs(norms-center)),.001)
        cutoff=1.345*robust_scale
        weights=np.minimum(1.,cutoff/np.maximum(norms,1e-12))
        row_weights=np.repeat(np.sqrt(weights),2)
        weighted_design=selected_rows*row_weights[:,None]
        weighted_values=selected_rhs*row_weights
        updated=np.linalg.lstsq(weighted_design,weighted_values,rcond=None)[0]
        if np.max(np.abs(updated-coefficients))<1e-10:
            coefficients=updated
            break
        coefficients=updated
    residual=(selected_rows@coefficients-selected_rhs).reshape(-1,2)*world_scale
    homography=np.asarray([[coefficients[0],coefficients[1],coefficients[2]],
                           [coefficients[3],coefficients[4],coefficients[5]],
                           [coefficients[6],coefficients[7],1.]])
    scale=np.sqrt(np.mean(residual**2,axis=0)).clip(.001)
    bounds=np.stack([world.min(axis=0)-.02,world.max(axis=0)+.02],axis=1)
    result=StackBlockLocalizer(tuple(map(tuple,homography.tolist())),tuple(world_center.tolist()),
        tuple(world_scale.tolist()),tuple(scale.tolist()),
        semantics['image_size'],semantics['camera'],semantics['camera_fovy_deg'],
        sha256_file(manifest_path),tuple(ids),tuple(hashes),tuple(map(tuple,bounds.tolist())))
    result.save(output_path)
    return result


def evaluate_stack_block_localizer(manifest_path,localizer_path,*,split='validation',frame_stride=32,
                                   tabletop_only=False):
    """Evaluate trajectory frames on a disjoint split; test is never the default."""
    if split not in ('validation','calibration','test'):
        raise ValueError('Localizer evaluation split must be validation/calibration/test')
    if type(frame_stride) is not int or frame_stride<1:
        raise ValueError('frame_stride must be a positive integer')
    manifest_path=Path(manifest_path).resolve()
    payload=json.loads(manifest_path.read_text())
    _validate_manifest(payload)
    localizer=StackBlockLocalizer.load(localizer_path)
    semantics=payload['semantics']
    if (localizer.manifest_sha256!=sha256_file(manifest_path)
            or localizer.camera!=semantics['camera']
            or localizer.camera_fovy_deg!=semantics.get('camera_fovy_deg')
            or localizer.image_size!=semantics['image_size']):
        raise ValueError('Localizer and evaluation dataset camera identities differ')
    if {row['episode_id'] for row in payload['episodes'] if row['split']==split}.intersection(
            localizer.training_episode_ids):
        raise ValueError('Localizer evaluation overlaps its fitting episodes')
    order=semantics['object_state_order']
    indices=[order.index('block_0:position_m:'+axis) for axis in 'xy']
    z_index=order.index('block_0:position_m:z')
    linear_indices=[order.index(f'block_0:linear_velocity_m_s:{axis}') for axis in 'xyz']
    angular_indices=[order.index(f'block_0:angular_velocity_rad_s:{axis}') for axis in 'xyz']
    errors=[]
    per_episode={}
    failures=candidates=0
    for row in payload['episodes']:
        if row['split']!=split:
            continue
        with np.load(manifest_path.parent/row['path'],allow_pickle=False) as arrays:
            if len(arrays['rgb'])!=len(arrays['object_truth']):
                raise ValueError('RGB and object labels do not align')
            for frame in range(0,len(arrays['rgb']),frame_stride):
                truth=arrays['object_truth'][frame]
                if tabletop_only:
                    baseline_z=float(arrays['object_truth'][0,z_index])
                    if (abs(float(truth[z_index])-baseline_z)>.003
                            or np.linalg.norm(truth[linear_indices])>.01
                            or np.linalg.norm(truth[angular_indices])>.1):
                        continue
                candidates+=1
                try:
                    prediction=localizer.predict(arrays['rgb'][frame])['position_xy_m']
                except ValueError:
                    failures+=1
                    continue
                target=truth[indices]
                error=prediction-target
                errors.append(error)
                per_episode.setdefault(row['episode_id'],[]).append(error)
    if not errors:
        raise ValueError('No episodes available in localizer evaluation split')
    error=np.stack(errors)
    return {'schema':'wmal.stack_block_rgb_localizer_evaluation.v1','split':split,
            'tabletop_only':tabletop_only,'episodes':len(per_episode),
            'candidate_frames':candidates,'evaluated_frames':len(error),'frame_stride':frame_stride,
            'detection_failures':failures,
            'detection_rate':len(error)/candidates,
            'manifest_sha256':sha256_file(manifest_path),
            'training_episode_ids':list(localizer.training_episode_ids),
            'rmse_xy_m':np.sqrt(np.mean(error**2,axis=0)).tolist(),
            'mean_euclidean_error_m':float(np.linalg.norm(error,axis=-1).mean()),
            'max_euclidean_error_m':float(np.linalg.norm(error,axis=-1).max()),
            'episode_rmse_xy_m':{key:np.sqrt(np.mean(np.stack(values)**2,axis=0)).tolist()
                                 for key,values in sorted(per_episode.items())}}
