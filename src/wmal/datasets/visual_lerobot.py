"""V2.1 parquet/video import; V3 packed videos require a separate adapter.

Original train/test assignments are retained. Validation episodes are partitioned
by episode into model selection/calibration, never by highly correlated frames.
Action semantics must be explicitly declared: names/dimension alone aren't enough.
"""
import json
from pathlib import Path
import numpy as np

from wmal.datasets.visual_sequences import write_episode, write_manifest
from wmal.logging.manifest import sha256_file


def convert_lerobot_v21(source_manifest,output,*,action_schema,image_size=64,
                        max_frames=64,max_episodes=None):
    import av
    import pyarrow.parquet as pq
    source_path=Path(source_manifest).resolve()
    source=json.loads(source_path.read_text())
    output=Path(output).resolve()
    if source.get('format')!='lerobot_v2.1_wma_source':
        raise ValueError('Expected validated LeRobot V2.1 source manifest, not V3')
    if not isinstance(action_schema,str) or not action_schema or any(ch.isspace() for ch in action_schema):
        raise ValueError('Explicit nonempty action schema required')
    if type(max_frames) is not int or max_frames<3 or image_size not in (32,64,128):
        raise ValueError('Invalid frame limit/resolution')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Conversion output must be empty')
    fps=source['fps']
    if not np.isfinite(fps) or fps<=0:
        raise ValueError('Invalid source frame rate')
    grouped={name:[] for name in ('train','validation','calibration','test')}
    validation=[r for r in source['episodes'] if r['split']=='validation']
    validation.sort(key=lambda r:r['episode_index'])
    calibration_ids={r['episode_index'] for r in validation[len(validation)//2:]}
    seen=set()
    for row in source['episodes']:
        split=row['split']
        if split not in ('train','validation','test') or row['episode_index'] in seen:
            raise ValueError('Invalid/duplicate source split or episode index')
        seen.add(row['episode_index'])
        if split=='validation' and row['episode_index'] in calibration_ids:
            split='calibration'
        grouped[split].append((row,split))
    if any(not group for group in grouped.values()):
        raise ValueError('Source requires independent train/validation/calibration/test episodes')
    if max_episodes is None:
        selected=[item for group in grouped.values() for item in group]
    else:
        if type(max_episodes) is not int or max_episodes<20:
            raise ValueError('Import subset needs >=20 episodes')
        counts=[int(max_episodes*.6),int(max_episodes*.1),int(max_episodes*.2)]
        counts.append(max_episodes-sum(counts))
        if any(len(group)<count for group,count in zip(grouped.values(),counts)):
            raise ValueError('Requested subset exceeds available episodes in an independent split')
        selected=[item for group,count in zip(grouped.values(),counts) for item in group[:count]]
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    source_records=[]
    for row,split in selected:
        parquet=Path(row['parquet']['path']).resolve()
        video=Path(row['video']['path']).resolve()
        table=pq.read_table(parquet,columns=['observation.state','action','timestamp','frame_index']).to_pydict()
        states=np.asarray(table['observation.state'],dtype='float32')[:max_frames]
        actions=np.asarray(table['action'],dtype='float32')[:max_frames]
        timestamps=np.asarray(table['timestamp'],dtype=float)[:max_frames]
        indices=np.asarray(table['frame_index'])[:max_frames]
        n=len(states)
        if (n<3 or len(actions)!=n or len(timestamps)!=n or len(indices)!=n
                or not np.array_equal(indices,np.arange(n))
                or not np.allclose(timestamps,np.arange(n)/fps,atol=.01/fps,rtol=.001)):
            raise ValueError('Source timestamps are not contiguous episode-local camera frames')
        decoder=av.open(str(video))
        stream=decoder.streams.video[0]
        images=[]
        try:
            if stream.average_rate is None or not np.isclose(float(stream.average_rate),fps,rtol=.01):
                raise ValueError('Video/metadata FPS differ; explicit resampling required')
            frames=iter(decoder.decode(stream))
            for index in range(n):
                try:
                    frame=next(frames)
                except StopIteration:
                    raise ValueError('Video ends before aligned state frames')
                if frame.time is None or abs(frame.time-timestamps[index])>.1/fps:
                    raise ValueError('Decoded video PTS and state timestamp are misaligned')
                # Explicit square resize; no claim of retaining calibrated intrinsics.
                images.append(frame.reformat(width=image_size,height=image_size,format='rgb24').to_ndarray())
        finally:
            decoder.close()
        name=f'episode_{row["episode_index"]:06d}.npz'
        write_episode(output/name,np.stack(images),states,actions[:-1])
        rows.append({'episode_id':f'lerobot_{row["episode_index"]}','split':split,'path':name})
        source_records.append({'episode_index':row['episode_index'],'original_split':row['split'],
                               'frames':n,'parquet_sha256':sha256_file(parquet),
                               'video_sha256':sha256_file(video)})
    provenance={'kind':'lerobot_v2.1_import','dataset':source['dataset'],
                'source_manifest_sha256':sha256_file(source_path),'source_episodes':source_records,
                'action_semantics_declaration':action_schema,
                'spatial_transform':'square_resize; intrinsics not preserved',
                'validation_rule':'sorted original validation episodes; second half calibration',
                'limitations':'Offline only: 16D Dex1 actions are NOT the 2D MuJoCo joint-delta interface; no task labels'}
    return write_manifest(output/'manifest.json',rows,action_schema=action_schema,
            action_order=source['action']['order'],state_order=source['state']['order'],
            camera=source['camera']['key'],period_s=1./fps,image_size=image_size,source=provenance)
