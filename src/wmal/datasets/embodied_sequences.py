"""Aligned T-observation/T-1-action windows with original-episode provenance.

NPZ layout: uint8 rgb[T,V,S,S,3], physical[T,P], commands[T,C],
actions[T-1,A]; optional timestamps[T], view_mask[T,V], events/event_mask[T-1,E].
Events are transition labels, returned only for the H prediction transitions.
Missing event labels have zero masks, never inferred labels. Missing timestamps
are explicitly identified as period-derived episode-local time.

Manifests bind files with sha256 and original trajectories with source_episode_id
and source_sha256 (plus optional source_hashes for multi-file originals). All four
splits are required. Re-exports can share a split, but never cross a source group.
"""
from collections import OrderedDict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .embodied_profiles import ObservationProfile, serializable_copy
from .visual_sequences import SPLITS, _validate_manifest as _validate_visual_manifest, validate_arrays
from wmal.logging.manifest import atomic_json, sha256_file

SCHEMA = 'wmal.embodied_episodes.v1'
ADAPTER_SCHEMA = 'wmal.embodied_visual_adapter.v1'
LINEAGE_SCHEMA = 'wmal.episode_lineage.v1'
_G1_JOINTS = ('left_shoulder_pitch_joint', 'left_elbow_joint')
_G1_ACTION_SCHEMA = 'wmal.g1.workcell.target_delta.v1'


def _hash(value):
    if (type(value) is not str or len(value) != 64
            or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError('Invalid SHA256 identity')
    return value


def _identity(value):
    if type(value) is not str or not value.strip():
        raise ValueError('Missing episode identity')
    return value


def _identities(row):
    ids = {_identity(row.get('episode_id')), _identity(row.get('source_episode_id'))}
    hashes = {_hash(row.get('sha256')), _hash(row.get('source_sha256'))}
    extras = row.get('source_hashes', [])
    if not isinstance(extras, list):
        raise ValueError('source_hashes must be an explicit list')
    hashes.update(_hash(value) for value in extras)
    return ids, hashes


def validate_lineage(episodes, lineage=None, *, context='dataset'):
    """Validate split/origin isolation and inherit train/selection exposure.

    Ancestor training/selection may be reused as training during fine-tuning.
    Any validation/calibration/test overlap is rejected, by exported OR original
    ID/hash. The returned cumulative lineage is JSON-serializable and detached
    from caller values. Unknown or incomplete ancestor lineage fails closed.
    """
    rows = serializable_copy(episodes)
    if not isinstance(rows, list):
        raise ValueError('Episode lineage requires a list of rows')
    result = {'schema': LINEAGE_SCHEMA, 'train': {'episode_ids': [], 'sha256': []},
              'selection': {'episode_ids': [], 'sha256': []}}
    if lineage is not None:
        ancestor = serializable_copy(lineage)
        if not isinstance(ancestor, dict) or ancestor.get('schema') != LINEAGE_SCHEMA:
            raise ValueError('Unsupported or unknown ancestor lineage')
        for kind in ('train', 'selection'):
            entry = ancestor.get(kind)
            if not isinstance(entry, dict):
                raise ValueError('Incomplete ancestor exposure lineage')
            for key in ('episode_ids', 'sha256'):
                values = entry.get(key)
                if not isinstance(values, list):
                    raise ValueError('Incomplete ancestor exposure lineage')
                result[kind][key] = sorted(set((_identity if key == 'episode_ids' else _hash)(v) for v in values))
            if bool(result[kind]['episode_ids']) != bool(result[kind]['sha256']):
                raise ValueError('Ancestor exposure must retain both episode IDs and content hashes')
    ancestor_ids = {v for kind in ('train', 'selection') for v in result[kind]['episode_ids']}
    ancestor_hashes = {v for kind in ('train', 'selection') for v in result[kind]['sha256']}
    cumulative = {kind: {key: set(values) for key, values in result[kind].items()}
                  for kind in ('train', 'selection')}
    seen_ids, seen_hashes, exports = {}, {}, set()
    for row in rows:
        if not isinstance(row, dict) or row.get('split') not in SPLITS:
            raise ValueError('Invalid episode split')
        split = row['split']
        ids, hashes = _identities(row)
        if row['episode_id'] in exports:
            raise ValueError('Duplicate exported episode identity')
        exports.add(row['episode_id'])
        for values, seen in [(ids, seen_ids), (hashes, seen_hashes)]:
            for value in values:
                if value in seen and seen[value] != split:
                    raise ValueError(f'{context}: original episode/content overlap across splits')
                seen[value] = split
        if split != 'train' and (ids & ancestor_ids or hashes & ancestor_hashes):
            raise ValueError(f'{context}: held-out overlap with ancestor model exposure')
        kind = {'train': 'train', 'validation': 'selection'}.get(split)
        if kind:
            cumulative[kind]['episode_ids'].update(ids)
            cumulative[kind]['sha256'].update(hashes)
    for kind, entry in cumulative.items():
        result[kind] = {key: sorted(values) for key, values in entry.items()}
    return result


def _file(root, row):
    name = row.get('path')
    if type(name) is not str or not name or Path(name).is_absolute():
        raise ValueError('Episode path must be relative to dataset root')
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Episode path escapes dataset root')
    return path


def _read_npz(path):
    try:
        with np.load(path, allow_pickle=False) as handle:
            return {key: handle[key] for key in handle.files}
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f'Invalid episode NPZ: {path.name}') from exc


def _numeric(array, shape, name):
    if (array is None or array.shape != shape or array.dtype.kind not in 'fiu'
            or not np.isfinite(array).all()):
        raise ValueError(f'Missing, nonfinite or misaligned {name}')
    if name != 'timestamps' and np.any(np.abs(array.astype('float64')) > np.finfo('float32').max):
        raise ValueError(f'{name} cannot be represented as finite float32 coordinates')


def _mask(array, shape, name):
    if (array.shape != shape or array.dtype.kind not in 'bifu'
            or not np.isin(array, (0, 1)).all()):
        raise ValueError(f'Invalid or misaligned {name}')


def _aligned(arrays, profile, row):
    rgb = arrays.get('rgb')
    if (rgb is None or rgb.ndim != 5 or rgb.dtype != np.uint8 or len(rgb) < 2
            or rgb.shape[1:] != (len(profile.camera_ids), profile.image_size, profile.image_size, 3)):
        raise ValueError('Missing or misaligned RGB views/profile')
    count = len(rgb)
    for key, width, length in [('physical', len(profile.physical_order), count),
                               ('commands', len(profile.command_order), count),
                               ('actions', len(profile.action_order), count - 1)]:
        _numeric(arrays.get(key), (length, width), key)
    if 'steps' in row and (type(row['steps']) is not int or row['steps'] != count - 1):
        raise ValueError('Manifest steps do not match episode modalities')
    if profile.command_rule == 'joint_delta' and not np.allclose(
            arrays['commands'][1:].astype('float64'),
            arrays['commands'][:-1].astype('float64') + arrays['actions'].astype('float64'),
            rtol=0, atol=1e-6):
        raise ValueError('Known command transitions do not match declared joint_delta actions')
    arrays = dict(arrays)
    arrays['view_mask'] = arrays.get('view_mask', np.ones((count, len(profile.camera_ids)), dtype=bool))
    _mask(arrays['view_mask'], (count, len(profile.camera_ids)), 'view_mask')
    if not arrays['view_mask'].any(axis=1).all():
        raise ValueError('Every real observation requires at least one observed camera')
    if ('events' in arrays) != ('event_mask' in arrays):
        raise ValueError('Event targets require explicit event masks')
    shape = (count - 1, len(profile.event_names))
    arrays['events'] = arrays.get('events', np.zeros(shape, dtype='float32'))
    arrays['event_mask'] = arrays.get('event_mask', np.zeros(shape, dtype=bool))
    _numeric(arrays['events'], shape, 'events')
    _mask(arrays['event_mask'], shape, 'event_mask')
    if np.any((arrays['events'] < 0) | (arrays['events'] > 1)):
        raise ValueError('Event targets must be in [0,1]')
    if 'timestamps' in arrays:
        _numeric(arrays['timestamps'], (count,), 'timestamps')
        if not np.allclose(np.diff(arrays['timestamps'].astype('float64')), profile.period_s, rtol=1e-5, atol=1e-7):
            raise ValueError('Episode timestamps do not align with declared control period')
        arrays['timing_source'] = 'recorded'
    else:
        start = row.get('start_time_s', 0.)
        if type(start) not in (int, float) or not np.isfinite(start):
            raise ValueError('Invalid declared episode start time')
        arrays['timestamps'] = start + np.arange(count, dtype='float64') * profile.period_s
        arrays['timing_source'] = 'period_derived'
    return arrays


def _validate_manifest(payload):
    if not isinstance(payload, dict) or payload.get('schema') not in (SCHEMA, ADAPTER_SCHEMA):
        raise ValueError('Expected embodied manifest or explicit visual adapter')
    profile = ObservationProfile.from_dict(payload.get('profile'))
    if not isinstance(payload.get('source'), dict) or not payload['source']:
        raise ValueError('Dataset source identity required')
    rows = payload.get('episodes')
    validate_lineage(rows)
    if {row['split'] for row in rows} != set(SPLITS):
        raise ValueError('All four independent splits are required')
    for row in rows:
        if type(row.get('steps')) is not int or row['steps'] < 1:
            raise ValueError('Episode steps must be positive integers')
    return profile


def write_embodied_manifest(path, episodes, profile, source):
    """Validate all aligned files, record exported hashes, then atomically write.

    Source identities must be supplied; this writer never substitutes a new
    compressed NPZ hash for an original trajectory hash.
    """
    path = Path(path).resolve()
    profile = ObservationProfile.from_dict(profile.to_dict() if isinstance(profile, ObservationProfile) else profile)
    rows = serializable_copy(episodes)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Episodes must be manifest rows')
    for row in rows:
        file = _file(path.parent, row)
        arrays = _aligned(_read_npz(file), profile, row)
        row.update(sha256=sha256_file(file), steps=len(arrays['actions']))
    payload = dict(schema=SCHEMA, profile=profile.to_dict(), source=serializable_copy(source), episodes=rows)
    _validate_manifest(payload)
    atomic_json(path, payload)
    return path


def _mapping(indices, size, name, *, empty=False):
    if (not isinstance(indices, (list, tuple)) or (not indices and not empty)
            or any(type(index) is not int or not 0 <= index < size for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError(f'Invalid explicit {name} mapping')
    return list(indices)


def _legacy_origins(rows, source):
    """Bind the existing LeRobot converter's separate original-file records.

    Parquet and video identify the trajectory independently; neither can be
    replaced by the newly compressed NPZ hash during adaptation/re-export.
    """
    records = None
    if source.get('kind') == 'lerobot_v2.1_import':
        raw = source.get('source_episodes')
        dataset = source.get('dataset')
        name = dataset.get('repo_id') if isinstance(dataset, dict) else dataset
        if not isinstance(raw, list) or type(name) is not str or not name.strip():
            raise ValueError('Missing original LeRobot source identity/records')
        records = {}
        for entry in raw:
            if (not isinstance(entry, dict) or type(entry.get('episode_index')) is not int
                    or entry['episode_index'] < 0):
                raise ValueError('Invalid original LeRobot source episode record')
            episode_id = f'lerobot_{entry["episode_index"]}'
            if episode_id in records:
                raise ValueError('Duplicate original source episode records')
            records[episode_id] = entry
    for row in rows:
        if records is not None:
            entry = records.get(row['episode_id'])
            if entry is None:
                raise ValueError('Missing original source episode hash record')
            parquet, video = _hash(entry.get('parquet_sha256')), _hash(entry.get('video_sha256'))
            canonical_id = f'{name}:episode:{entry["episode_index"]}'
            if 'source_episode_id' in row and row['source_episode_id'] != canonical_id:
                raise ValueError('Declared original source episode ID conflicts with canonical identity')
            row['source_episode_id'] = canonical_id
            row.setdefault('source_sha256', parquet)
            row['source_hashes'] = sorted(set(row.get('source_hashes', [])) | {parquet, video})
        else:
            row.setdefault('source_episode_id', row['episode_id'])
            row.setdefault('source_sha256', row['sha256'])


def from_visual_manifest(manifest, *, physical_indices, command_indices, command_rule, profile=None):
    """Return an in-memory read-only legacy descriptor retaining the source root.

    Known G1 units follow its original 4D state/2D action schema independently
    of the selected coordinates. Only the explicit [0,1]/[2,3] mapping justifies
    known command inference. Other mappings require an explicit offline profile.
    Other robots require an explicit profile (including units,
    controller and origin); all action dimensions are retained. ``joint_delta``
    is justified only for the verified G1 legacy interface. ``none`` is offline.
    """
    path = Path(manifest).resolve()
    payload = json.loads(path.read_text())
    _validate_visual_manifest(payload)
    semantics, source = payload['semantics'], payload['source']
    physical = _mapping(physical_indices, len(semantics['state_order']), 'physical')
    commands = _mapping(command_indices, len(semantics['state_order']), 'command', empty=True)
    if set(physical) & set(commands):
        raise ValueError('Physical and command mappings cannot overlap')
    known_g1 = (semantics['action_schema'] == _G1_ACTION_SCHEMA
             and tuple(semantics['action_order']) == _G1_JOINTS
             and tuple(semantics['state_order']) == _G1_JOINTS + tuple(j + '_target' for j in _G1_JOINTS))
    is_g1 = known_g1 and physical == [0, 1] and commands == [2, 3]
    if command_rule not in ('joint_delta', 'none') or (command_rule == 'joint_delta' and not is_g1):
        raise ValueError('No justified legacy known-command semantics for this mapping')
    controller = source.get('controller_profile')
    if profile is None:
        if not is_g1:
            raise ValueError('Arbitrary legacy data requires an explicit units/controller/origin profile')
        profile = ObservationProfile(physical_order=[semantics['state_order'][i] for i in physical],
            command_order=[semantics['state_order'][i] for i in commands],
            action_order=semantics['action_order'], action_schema=semantics['action_schema'],
            camera_ids=[semantics['camera']], image_size=semantics['image_size'], period_s=semantics['period_s'],
            event_names=semantics.get('event_names', []), physical_units=['rad'] * 2,
            command_units=['rad'] * 2, action_units=['rad'] * 2, command_rule=command_rule,
            controller=controller, origin={**source, 'manifest_sha256': sha256_file(path)})
    profile = ObservationProfile.from_dict(profile.to_dict() if isinstance(profile, ObservationProfile) else profile)
    if known_g1 and (profile.physical_units != ('rad',) * len(physical)
                     or profile.command_units != ('rad',) * len(commands)
                     or profile.action_units != ('rad',) * len(semantics['action_order'])):
        raise ValueError('The verified G1 coordinates are radians; explicit conversion is required')
    if controller is not None and profile.controller != controller:
        raise ValueError('Declared controller differs from recorded legacy controller')
    if (profile.physical_order != tuple(semantics['state_order'][i] for i in physical)
            or profile.command_order != tuple(semantics['state_order'][i] for i in commands)
            or profile.action_order != tuple(semantics['action_order'])
            or profile.action_schema != semantics['action_schema']
            or profile.camera_ids != (semantics['camera'],) or profile.image_size != semantics['image_size']
            or profile.period_s != semantics['period_s'] or profile.command_rule != command_rule
            or profile.event_names != tuple(semantics.get('event_names', []))):
        raise ValueError('Declared profile does not match legacy semantics/mapping')
    rows = serializable_copy(payload['episodes'])
    _legacy_origins(rows, source)
    adapter = dict(schema=ADAPTER_SCHEMA, root=str(path.parent),
                   visual_manifest=str(path), visual_manifest_sha256=sha256_file(path),
                   physical_indices=physical, command_indices=commands, profile=profile.to_dict(),
                   source=serializable_copy(source), episodes=rows,
                   legacy_semantics=serializable_copy(semantics))
    _validate_manifest(adapter)
    return adapter


class EmbodiedDataset:
    """Lazy selected-split windows, with at most two decompressed episodes cached.

    All split identities, paths and file hashes are checked at construction;
    selected episode arrays are validated on first access. NPZ compression needs
    one whole episode in memory; cache memory never grows with episode count.
    Returned arrays and metadata are independent from the cache/manifest.
    """
    def __init__(self, manifest, split, *, horizon, context_steps=0):
        if (split not in SPLITS or type(horizon) is not int or horizon < 1
                or type(context_steps) is not int or context_steps < 0):
            raise ValueError('Invalid split, horizon or context_steps')
        if isinstance(manifest, dict):
            payload = serializable_copy(manifest)
            if payload.get('schema') != ADAPTER_SCHEMA:
                raise ValueError('Only explicit legacy descriptors may specify an in-memory root')
            self.path = Path(payload['visual_manifest']).resolve()
            self.root = Path(payload['root']).resolve()
            if self.path.parent != self.root or sha256_file(self.path) != payload['visual_manifest_sha256']:
                raise ValueError('Legacy manifest root/hash changed')
            # Rebuild to bind descriptor mappings/rows/profile to the original manifest.
            verified = from_visual_manifest(self.path, physical_indices=payload['physical_indices'],
                command_indices=payload['command_indices'], command_rule=payload['profile']['command_rule'],
                profile=payload['profile'])
            if verified != payload:
                raise ValueError('Legacy descriptor identity or mapping changed')
        else:
            self.path = Path(manifest).resolve()
            self.root = self.path.parent
            payload = json.loads(self.path.read_text())
            if payload.get('schema') != SCHEMA:
                raise ValueError('Legacy manifests require an explicit visual adapter/profile')
        self.profile = _validate_manifest(payload)
        self._profile_identity = sha256(json.dumps(self.profile.to_dict(), sort_keys=True,
                                                   separators=(',', ':')).encode()).hexdigest()
        self._manifest = serializable_copy(payload)
        self.split, self.horizon, self.context_steps = split, horizon, context_steps
        for row in payload['episodes']:
            if sha256_file(_file(self.root, row)) != row['sha256']:
                raise ValueError('Episode content hash changed')
        self._rows = [row for row in self._manifest['episodes'] if row['split'] == split]
        self.index, self._cache = [], OrderedDict()
        self.episode_window_indices = {}
        # Union original identities and hashes to group re-exports for bootstrap.
        parents = list(range(len(self._rows)))
        def find(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i
        identities = {}
        for i, row in enumerate(self._rows):
            source_keys = [('id', row['source_episode_id']), ('hash', row['source_sha256'])]
            source_keys += [('hash', value) for value in row.get('source_hashes', [])]
            for key in source_keys:
                if key in identities:
                    parents[find(i)] = find(identities[key])
                identities[key] = i
        names = {}
        for i, row in enumerate(self._rows):
            group = names.setdefault(find(i), row['source_episode_id'])
            indices = self.episode_window_indices.setdefault(group, [])
            for start in range(row['steps'] - horizon + 1):
                indices.append(len(self.index))
                self.index.append((i, start))
        self.episode_window_indices = {key: values for key, values in self.episode_window_indices.items() if values}
        if not self.index:
            raise ValueError('No full-horizon windows in selected split')

    @property
    def manifest(self):
        return serializable_copy(self._manifest)

    @property
    def rows(self):
        return serializable_copy(self._rows)

    def _episode(self, index):
        if index not in self._cache:
            row = self._rows[index]
            file = _file(self.root, row)
            if sha256_file(file) != row['sha256']:
                raise ValueError('Episode content hash changed')
            arrays = _read_npz(file)
            if self._manifest['schema'] == ADAPTER_SCHEMA:
                required = ('rgb', 'states', 'actions')
                if any(key not in arrays for key in required):
                    raise ValueError('Missing legacy modalities')
                validate_arrays(arrays['rgb'], arrays['states'], arrays['actions'],
                                arrays.get('events'), arrays.get('event_mask'))
                if arrays['states'].shape[1] != len(self._manifest['legacy_semantics']['state_order']):
                    raise ValueError('Legacy state dimensions changed')
                arrays['rgb'] = arrays['rgb'][:, None]
                arrays['physical'] = arrays['states'][:, self._manifest['physical_indices']]
                arrays['commands'] = arrays['states'][:, self._manifest['command_indices']]
                del arrays['states']
            aligned = _aligned(arrays, self.profile, row)
            if len(self._cache) >= 2:
                self._cache.popitem(last=False)
            self._cache[index] = aligned
        self._cache.move_to_end(index)
        return self._cache[index]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode, start = self.index[index]
        arrays, row = self._episode(episode), self._rows[episode]
        k, h = self.context_steps, self.horizon
        length = min(start, k)
        padding = k - length
        first = start - length
        sample = {}
        for key in ('rgb', 'physical', 'commands', 'timestamps', 'view_mask'):
            values = arrays[key][first:start + h + 1].copy()
            if padding:
                pad = (np.zeros((padding, values.shape[1]), dtype=bool) if key == 'view_mask'
                       else np.repeat(values[:1], padding, axis=0))
                values = np.concatenate([pad, values])
            sample[key] = values
        actions = arrays['actions'][first:start + h].astype('float32', copy=True)
        if padding:
            actions = np.concatenate([np.zeros((padding, actions.shape[1]), dtype='float32'), actions])
        sample['rgb'] = sample['rgb'].transpose(0, 1, 4, 2, 3).astype('float32') / 255.
        for key in ('physical', 'commands'):
            sample[key] = sample[key].astype('float32')
        sample['view_mask'] = sample['view_mask'].astype(bool)
        is_first = np.zeros(k + h + 1, dtype=bool)
        is_first[:padding + 1] = True
        identity = json.dumps([row['episode_id'], row['sha256'], row['source_episode_id'],
                               row['source_sha256'], row.get('source_hashes', []),
                               self._profile_identity, start, h, k], separators=(',', ':'))
        sample.update(actions=actions, events=arrays['events'][start:start + h].astype('float32', copy=True),
                      event_mask=arrays['event_mask'][start:start + h].astype(bool, copy=True),
                      is_first=is_first, context_length=length, prediction_offset=k,
                      sample_id=sha256(identity.encode()).hexdigest(), offset=start,
                      episode_id=row['episode_id'], source_episode_id=row['source_episode_id'],
                      source_sha256=row['source_sha256'], source_hashes=list(row.get('source_hashes', [])),
                      period_s=self.profile.period_s, timing_source=arrays['timing_source'])
        return sample

    def normalization(self):
        """Training-only streaming population moments over real episode coordinates.

        Fits each exported episode once, irrespective of horizon, padding or
        overlapping window counts. A 0.01 scale floor matches existing training.
        """
        if self.split != 'train':
            raise ValueError('Normalization must be fitted on training episodes only')
        result = {}
        for name, key in [('physical', 'physical'), ('command', 'commands'), ('action', 'actions')]:
            count, mean, m2 = 0, None, None
            for i in range(len(self._rows)):
                values = self._episode(i)[key].astype('float64')
                n = len(values)
                batch_mean = values.mean(axis=0)
                batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
                if count == 0:
                    count, mean, m2 = n, batch_mean, batch_m2
                else:
                    delta = batch_mean - mean
                    m2 += batch_m2 + delta ** 2 * count * n / (count + n)
                    mean += delta * n / (count + n)
                    count += n
            result[name + '_mean'] = mean.tolist()
            result[name + '_scale'] = np.sqrt(np.maximum(m2 / count, 0)).clip(.01).tolist()
        return result
