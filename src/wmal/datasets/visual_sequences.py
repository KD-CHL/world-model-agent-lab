"""Episode-disjoint RGB/action data with explicit control semantics.

RGB[t] and state[t] precede action[t]; RGB[t+1] is its observed consequence.
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


def write_episode(path, rgb, states, actions, *, events=None, event_mask=None, overwrite=False):
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
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_manifest(path, episodes, *, action_schema, action_order, state_order, camera,
                   period_s, image_size, source, event_names=()):
    path = Path(path).resolve()
    rows = []
    for row in episodes:
        file = (path.parent/row['path']).resolve()
        with np.load(file, allow_pickle=False) as arrays:
            validate_arrays(arrays['rgb'], arrays['states'], arrays['actions'],
                            arrays.get('events'), arrays.get('event_mask'))
            if (arrays['rgb'].shape[1] != image_size or arrays['states'].shape[1] != len(state_order)
                    or arrays['actions'].shape[1] != len(action_order)
                    or len(event_names) != (arrays['events'].shape[1] if 'events' in arrays else 0)):
                raise ValueError('Episode dimensions do not match manifest semantics')
            steps = len(arrays['actions'])
        rows.append({**row, 'sha256': sha256_file(file), 'steps': steps})
    payload = {'schema': SCHEMA, 'semantics': {'action_schema': action_schema,
               'action_order': list(action_order), 'state_order': list(state_order),
               'camera': camera, 'period_s': period_s, 'image_size': image_size,
               'event_names': list(event_names)}, 'source': source, 'episodes': rows}
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
    if (not np.isfinite(semantics['period_s']) or semantics['period_s'] <= 0
            or semantics['image_size'] not in (32,64,128)):
        raise ValueError('Invalid period or image size')
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
    def __init__(self, manifest, split, *, horizon):
        self.path = Path(manifest).resolve()
        self.manifest = json.loads(self.path.read_text())
        _validate_manifest(self.manifest)
        if split not in SPLITS or type(horizon) is not int or horizon < 1:
            raise ValueError('Invalid split or horizon')
        self.semantics = self.manifest['semantics']
        self.split, self.horizon = split, horizon
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
            semantics=self.semantics
            if (len(arrays['actions']) != row['steps'] or arrays['states'].shape[1]!=len(semantics['state_order'])
                    or arrays['actions'].shape[1]!=len(semantics['action_order'])
                    or arrays['rgb'].shape[1]!=semantics['image_size']):
                raise ValueError('Manifest dimensions do not match episode')
            self._cache[index] = arrays
            while len(self._cache)>2:
                self._cache.popitem(last=False)
        self._cache.move_to_end(index)
        return self._cache[index]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode,t = self.index[index]
        arrays = self._episode(episode)
        h = self.horizon
        events = arrays.get('events', np.zeros((len(arrays['actions']),0),dtype='float32'))
        mask = arrays.get('event_mask', np.zeros_like(events))
        return {'rgb': np.transpose(arrays['rgb'][t:t+h+1],(0,3,1,2)).astype('float32')/255.,
                'states': arrays['states'][t:t+h+1].astype('float32'),
                'actions': arrays['actions'][t:t+h].astype('float32'),
                'events': events[t:t+h], 'event_mask': mask[t:t+h],
                'episode_id': self.rows[episode]['episode_id'], 'offset': t}

    def normalization(self):
        if self.split != 'train':
            raise ValueError('Normalization must be fitted on training episodes only')
        result = {}
        for name, key in [('state','states'),('action','actions')]:
            # Streaming moments, bounded memory for real robot datasets.
            count, total, square = 0, None, None
            for i in range(len(self.rows)):
                values = self._episode(i)[key].astype('float64')
                count += len(values)
                total = values.sum(0) if total is None else total+values.sum(0)
                square = (values*values).sum(0) if square is None else square+(values*values).sum(0)
            mean = total/count
            scale = np.sqrt(np.maximum(square/count-mean*mean,0.)).clip(.01)
            result[name+'_mean'], result[name+'_scale'] = mean.tolist(), scale.tolist()
        return result


def assign_four_splits(count, seed):
    if type(count) is not int or count < 20:
        raise ValueError('Need >=20 episodes for four independent splits')
    order = np.random.default_rng(seed).permutation(count)
    train, validation, calibration = int(count*.6), int(count*.1), int(count*.2)
    labels = (['train']*train+['validation']*validation+['calibration']*calibration
              +['test']*(count-train-validation-calibration))
    return {int(i): label for i,label in zip(order,labels)}
