"""Episode-disjoint RGB/action data with explicit control semantics.

RGB[t] and state[t] precede action[t]; RGB[t+1] is its observed consequence.
Event[t] labels the post-action event state aligned to RGB/state[t+1].
Training, model selection, calibration and final test episodes never overlap.
"""
from collections import OrderedDict
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from wmal.logging.manifest import atomic_json, sha256_file

SPLITS = ('train', 'validation', 'calibration', 'test')
SCHEMA = 'wmal.visual_episodes.v1'


def read_episode_lineage(metadata):
    """Cumulative exposure survives fine-tuning; unknown fine-tune ancestry is unsafe."""
    lineage=metadata.get('episode_lineage')
    if lineage is None:
        if metadata.get('parent_model_version'):
            raise ValueError('Unknown checkpoint episode lineage; rebuild from a documented base')
        # Legacy base checkpoints retain ID exclusions, but have no historical file hashes.
        lineage={'schema':'wmal.episode_lineage.v1',
                 'train':{'episode_ids':metadata['training_episode_ids'],'sha256':[]},
                 'selection':{'episode_ids':metadata['selection_episode_ids'],'sha256':[]}}
    if lineage.get('schema')!='wmal.episode_lineage.v1':
        raise ValueError('Unsupported checkpoint episode lineage')
    result={'schema':lineage['schema']}
    for kind in ('train','selection'):
        entry=lineage.get(kind,{})
        result[kind]={}
        for key in ('episode_ids','sha256'):
            values=entry.get(key)
            if not isinstance(values,list) or any(not isinstance(v,str) or not v for v in values):
                raise ValueError('Invalid checkpoint episode lineage')
            result[kind][key]=sorted(set(values))
    return result


def assert_unexposed_episodes(rows, exposures, *, context):
    ids={v for entry in exposures for v in entry['episode_ids']}
    hashes={v for entry in exposures for v in entry['sha256']}
    if any(row['episode_id'] in ids or row['sha256'] in hashes for row in rows):
        raise ValueError(f'{context} episode overlap with ancestor/current model exposure')


def validate_arrays(rgb, states, actions, events=None, event_mask=None):
    if (rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3
            or rgb.shape[1] != rgb.shape[2] or rgb.shape[1] not in (32, 64, 128)):
        raise ValueError('RGB must be uint8 [T+1,size,size,3], size 32/64/128')
    if (states.ndim != 2 or actions.ndim != 2 or len(actions) < 1
            or len(states) != len(actions)+1 or len(rgb) != len(states)
            or min(states.shape[1], actions.shape[1]) < 1
            or not np.isfinite(states).all() or not np.isfinite(actions).all()):
        raise ValueError('Invalid or temporally misaligned state/action arrays')
    if (events is None) != (event_mask is None):
        raise ValueError('Event supervision requires explicit observation masks')
    if events is not None and (events.ndim != 2 or len(events) != len(actions)
            or event_mask.shape != events.shape or not np.isfinite(events).all()
            or np.any((events < 0) | (events > 1))
            or np.any((event_mask != 0) & (event_mask != 1))):
        raise ValueError('Invalid event targets/masks')


def validate_event_state_arrays(arrays):
    """Validate event labels aligned to observations, including the reset state."""
    has_states = 'event_states' in arrays
    has_mask = 'event_state_mask' in arrays
    if has_states != has_mask:
        raise ValueError('Event state labels require an explicit state mask')
    if not has_states:
        return
    events = arrays.get('events')
    event_mask = arrays.get('event_mask')
    states, mask = arrays['event_states'], arrays['event_state_mask']
    if events is None or event_mask is None:
        raise ValueError('Event state labels require transition event labels')
    expected = (len(arrays['states']), events.shape[1])
    if (states.shape != expected or mask.shape != expected
            or not np.isfinite(states).all() or not np.isfinite(mask).all()
            or np.any((states < 0) | (states > 1))
            or np.any((mask != 0) & (mask != 1))):
        raise ValueError('Event state labels/masks do not match observation alignment')
    known = event_mask.astype(bool)
    if (not np.array_equal(mask[1:].astype(bool), known)
            or not np.array_equal(states[1:][known], events[known])):
        raise ValueError('Post-action event states do not match transition labels/masks')


def write_episode(path, rgb, states, actions, *, events=None, event_mask=None,
                  extra_arrays=None, overwrite=False):
    rgb, states, actions = np.asarray(rgb), np.asarray(states), np.asarray(actions)
    events = None if events is None else np.asarray(events)
    event_mask = None if event_mask is None else np.asarray(event_mask)
    validate_arrays(rgb, states, actions, events, event_mask)
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise ValueError('Episode already exists; choose a new output directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {'rgb': rgb, 'states': states.astype('float32'), 'actions': actions.astype('float32')}
    if events is not None:
        arrays.update(events=events.astype('float32'), event_mask=event_mask.astype('float32'))
    reserved = set(arrays)
    for name, value in (extra_arrays or {}).items():
        extra = np.asarray(value)
        if (not isinstance(name, str) or not name or name in reserved or extra.dtype == object
                or extra.ndim < 1 or len(extra) != len(states)):
            raise ValueError('Extra arrays must have unique names and one row per state')
        if np.issubdtype(extra.dtype, np.number) and not np.isfinite(extra).all():
            raise ValueError('Extra arrays must contain only finite numeric values')
        arrays[name] = extra
    validate_event_state_arrays(arrays)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_manifest(path, episodes, *, action_schema, action_order, state_order, camera,
                   period_s, image_size, source, event_names=(), object_state_order=(),
                   camera_fovy_deg=None):
    path = Path(path).resolve()
    rows = []
    for row in episodes:
        file = (path.parent/row['path']).resolve()
        with np.load(file, allow_pickle=False) as arrays:
            validate_arrays(arrays['rgb'], arrays['states'], arrays['actions'],
                            arrays.get('events'), arrays.get('event_mask'))
            validate_event_state_arrays(arrays)
            if (arrays['rgb'].shape[1] != image_size or arrays['states'].shape[1] != len(state_order)
                    or arrays['actions'].shape[1] != len(action_order)
                    or len(event_names) != (arrays['events'].shape[1] if 'events' in arrays else 0)
                    or object_state_order and ('object_truth' not in arrays
                        or arrays['object_truth'].ndim != 2
                        or arrays['object_truth'].shape[1] != len(object_state_order))):
                raise ValueError('Episode dimensions do not match manifest semantics')
            steps = len(arrays['actions'])
        rows.append({**row, 'sha256': sha256_file(file), 'steps': steps})
    payload = {'schema': SCHEMA, 'semantics': {'action_schema': action_schema,
               'action_order': list(action_order), 'state_order': list(state_order),
               'camera': camera, 'period_s': period_s, 'image_size': image_size,
               **({'camera_fovy_deg': float(camera_fovy_deg)} if camera_fovy_deg is not None else {}),
               'event_names': list(event_names),
               **({'object_state_order': list(object_state_order)} if object_state_order else {})},
               'source': source, 'episodes': rows}
    _validate_manifest(payload)
    atomic_json(path, payload)
    return path


def _validate_manifest(payload):
    if payload.get('schema') != SCHEMA:
        raise ValueError('Unsupported visual dataset schema')
    semantics = payload['semantics']
    for key in ('action_schema', 'camera'):
        if not isinstance(semantics.get(key), str) or not semantics[key]:
            raise ValueError('Missing control/camera semantics')
    for key in ('action_order', 'state_order'):
        names = semantics[key]
        if not names or len(names) != len(set(names)) or any(not isinstance(n,str) or not n for n in names):
            raise ValueError('Vector order must be unique named dimensions')
    if 'object_state_order' in semantics:
        names = semantics['object_state_order']
        if (not isinstance(names, list) or not names or len(names) != len(set(names))
                or any(not isinstance(n, str) or not n for n in names)):
            raise ValueError('Object state order must be unique named dimensions')
    if (not np.isfinite(semantics['period_s']) or semantics['period_s'] <= 0
            or semantics['image_size'] not in (32,64,128)):
        raise ValueError('Invalid period or image size')
    if ('camera_fovy_deg' in semantics and
            (not np.isfinite(semantics['camera_fovy_deg']) or
             not 1.<=semantics['camera_fovy_deg']<179.)):
        raise ValueError('Invalid camera field of view')
    ids, contents = set(), set()
    for row in payload['episodes']:
        if (row['split'] not in SPLITS or not isinstance(row['episode_id'], str)
                or not row['episode_id'] or row['episode_id'] in ids or row['sha256'] in contents
                or type(row['steps']) is not int or row['steps'] < 1):
            raise ValueError('Duplicate/invalid episode or content leakage between splits')
        ids.add(row['episode_id'])
        contents.add(row['sha256'])
    if set(row['split'] for row in payload['episodes']) != set(SPLITS):
        raise ValueError('All four independent splits are required')


class VisualDataset:
    """Lazy windows with a bounded episode cache; no padded or cross-episode targets."""
    def __init__(self, manifest, split, *, horizon, context_steps=0):
        self.path = Path(manifest).resolve()
        self.manifest = json.loads(self.path.read_text())
        _validate_manifest(self.manifest)
        if (split not in SPLITS or type(horizon) is not int or horizon < 1
                or type(context_steps) is not int or context_steps < 0):
            raise ValueError('Invalid split or horizon')
        self.semantics = self.manifest['semantics']
        self.split, self.horizon = split, horizon
        self.context_steps = context_steps
        self.rows = [r for r in self.manifest['episodes'] if r['split'] == split]
        self.index, self._cache = [], OrderedDict()
        for i,row in enumerate(self.rows):
            file = (self.path.parent/row['path']).resolve()
            if not file.is_relative_to(self.path.parent) or sha256_file(file) != row['sha256']:
                raise ValueError('Episode escapes dataset root or content hash changed')
            self.index.extend((i,t) for t in range(row['steps']-horizon+1))
        if not self.index:
            raise ValueError('No full-horizon windows in selected split')

    def _episode(self, index):
        if index not in self._cache:
            row = self.rows[index]
            with np.load(self.path.parent/row['path'], allow_pickle=False) as handle:
                arrays = {key: handle[key] for key in handle.files}
            validate_arrays(arrays['rgb'], arrays['states'], arrays['actions'],
                            arrays.get('events'), arrays.get('event_mask'))
            validate_event_state_arrays(arrays)
            semantics=self.semantics
            if (len(arrays['actions']) != row['steps'] or arrays['states'].shape[1]!=len(semantics['state_order'])
                    or arrays['actions'].shape[1]!=len(semantics['action_order'])
                    or arrays['rgb'].shape[1]!=semantics['image_size']):
                raise ValueError('Manifest dimensions do not match episode')
            if 'object_state_order' in semantics:
                if ('object_truth' not in arrays or arrays['object_truth'].shape !=
                        (row['steps']+1, len(semantics['object_state_order']))):
                    raise ValueError('Object truth dimensions do not match manifest')
            self._cache[index] = arrays
            while len(self._cache)>2:
                self._cache.popitem(last=False)
        self._cache.move_to_end(index)
        return self._cache[index]

    def event_supervision_for_episode(self, index):
        """Return supervised channels without loading RGB/state episode arrays."""
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise IndexError('Episode index out of range')
        names=self.semantics.get('event_names',[])
        if not names:
            return np.zeros(0,dtype=bool)
        row=self.rows[index]
        path=(self.path.parent/row['path']).resolve()
        with np.load(path,allow_pickle=False) as arrays:
            if 'event_mask' not in arrays.files:
                raise ValueError('Event semantics require an event mask in every episode')
            mask=arrays['event_mask']
        if (mask.shape!=(row['steps'],len(names)) or not np.isfinite(mask).all()
                or np.any((mask!=0)&(mask!=1))):
            raise ValueError('Event supervision mask does not match episode semantics')
        return mask.astype(bool).any(0)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode,t = self.index[index]
        arrays = self._episode(episode)
        h = self.horizon
        events = arrays.get('events', np.zeros((len(arrays['actions']),0),dtype='float32'))
        mask = arrays.get('event_mask', np.zeros_like(events))
        if 'event_states' in arrays:
            event_context = arrays['event_states'][t]
            event_context_mask = arrays['event_state_mask'][t]
        elif t > 0:
            # Legacy archives align event labels only to post-action frames.
            # The previous transition is the last known label at this origin.
            event_context = events[t-1]
            event_context_mask = mask[t-1]
        else:
            # Do not infer reset labels from the first future target.
            event_context = np.zeros(events.shape[1], dtype=np.float32)
            event_context_mask = np.zeros(events.shape[1], dtype=np.float32)
        k=self.context_steps
        length=min(t,k)
        padding=k-length
        frames=arrays['rgb'][t-length:t+h+1]
        states=arrays['states'][t-length:t+h+1]
        object_truth=(arrays.get('object_truth') if 'object_state_order' in self.semantics else None)
        if object_truth is not None:
            object_truth=object_truth[t-length:t+h+1]
        actions=arrays['actions'][t-length:t+h]
        if padding:
            frames=np.concatenate([np.repeat(frames[:1],padding,axis=0),frames])
            states=np.concatenate([np.repeat(states[:1],padding,axis=0),states])
            if object_truth is not None:
                object_truth=np.concatenate([np.repeat(object_truth[:1],padding,axis=0),object_truth])
            actions=np.concatenate([np.zeros((padding,actions.shape[1]),dtype=actions.dtype),actions])
        is_first=np.zeros(k+h+1,dtype=bool)
        is_first[:padding+1]=True
        result={'rgb': np.transpose(frames,(0,3,1,2)).astype('float32')/255.,
                'states': states.astype('float32'),
                'actions': actions.astype('float32'),
                'events': events[t:t+h], 'event_mask': mask[t:t+h],
                'event_context': np.asarray(event_context, dtype=np.float32),
                'event_context_mask': np.asarray(event_context_mask, dtype=np.float32),
                'is_first':is_first,'context_length':length,'prediction_offset':k,
                'episode_id': self.rows[episode]['episode_id'], 'offset': t}
        if object_truth is not None:
            result['object_truth']=object_truth.astype('float32')
        return result

    def normalization(self):
        if self.split != 'train':
            raise ValueError('Normalization must be fitted on training episodes only')
        fields=[('state','states'),('action','actions')]
        if 'object_state_order' in self.semantics:
            fields.append(('object','object_truth'))
        counts={name:0 for name,_ in fields}
        totals={name:None for name,_ in fields}
        squares={name:None for name,_ in fields}
        # Read only the small arrays needed for moments. `_episode` materializes
        # every NPZ member, including large RGB sequences that normalization
        # never uses; doing that for every training episode is wasteful.
        for row in self.rows:
            path=(self.path.parent/row['path']).resolve()
            with np.load(path,allow_pickle=False) as arrays:
                available=set(arrays.files)
                for name,key in fields:
                    if key not in available:
                        raise ValueError(f'Training episode lacks normalization array: {key}')
                    values=arrays[key].astype('float64')
                    if values.ndim!=2 or not np.isfinite(values).all():
                        raise ValueError(f'Invalid normalization array: {key}')
                    counts[name]+=len(values)
                    total=values.sum(0)
                    square=(values*values).sum(0)
                    totals[name]=total if totals[name] is None else totals[name]+total
                    squares[name]=square if squares[name] is None else squares[name]+square
        result={}
        for name,_ in fields:
            mean=totals[name]/counts[name]
            scale=np.sqrt(np.maximum(squares[name]/counts[name]-mean*mean,0.)).clip(.01)
            result[name+'_mean'],result[name+'_scale']=mean.tolist(),scale.tolist()
        return result


def assign_four_splits(count, seed):
    if type(count) is not int or count < 20:
        raise ValueError('Need >=20 episodes for four independent splits')
    order = np.random.default_rng(seed).permutation(count)
    train, validation, calibration = int(count*.6), int(count*.1), int(count*.2)
    labels = (['train']*train+['validation']*validation+['calibration']*calibration
              +['test']*(count-train-validation-calibration))
    return {int(i): label for i,label in zip(order,labels)}
