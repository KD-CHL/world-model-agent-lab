"""Independent episode-level simultaneous physical residual calibration.

Default scores use only training-fixed physical scales. Every episode contributes
one maximum across all windows, horizons and physical dimensions. Commands never
participate. Bounds are empirical prediction intervals, not safety probabilities.
The optional ensemble-scaled contrast has a separate frozen bound protocol.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from wmal.agents.prediction_contracts import (allows_scope, array_snapshot, content_hash,
    known_commands, profile_semantics, validate_scope)
from wmal.datasets.embodied_profiles import ObservationProfile, serializable_copy
from wmal.datasets.embodied_sequences import EmbodiedDataset, validate_lineage
from wmal.logging.manifest import atomic_json, sha256_file

SCHEMA = 'wmal.embodied_calibration.v1'
BOUND_PROTOCOL = 'wmal.embodied_simultaneous_physical_bounds.v1'


def _groups(rows):
    """Original/export IDs and all content hashes link re-exports transitively."""
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        keys = [('id', row['episode_id']), ('id', row['source_episode_id'])]
        keys += [('hash', value) for value in [row['sha256'], row['source_sha256']] + row.get('source_hashes', [])]
        for key in keys:
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    return [root(i) for i in range(len(rows))]


@dataclass(frozen=True)
class EmbodiedCalibration:
    """Immutable calibration payload; metadata properties return detached copies."""
    _payload_json: str

    @property
    def payload(self):
        return json.loads(self._payload_json)

    @property
    def model_version(self):
        return self.payload['model_version']

    @property
    def protocol_hash(self):
        return self.payload['protocol_hash']

    @property
    def quantile(self):
        return self.payload['quantile']

    @property
    def episode_count(self):
        return self.payload['episode_count']

    @property
    def episode_records(self):
        return self.payload['episode_records']

    @property
    def scope(self):
        return self.payload['scope']

    @property
    def version(self):
        return 'embodied-calibration-' + content_hash(self.payload)

    @classmethod
    def fit(cls, records, model, *, alpha=.2, samples=8, mode='sample', scope=None,
            bound_mode='fixed_scale', std_floor=.05):
        """Fit from EmbodiedDataset or iterable explicit residual records.

        Explicit records require split/episode_id/sha256/source_episode_id/
        source_sha256/source_hashes, profile, prediction (provider result), and
        target[H,P] in raw units. Dataset records are predicted with seed=0.
        Model metadata must have cumulative Task2 ``episode_lineage``; unknown
        exposure cannot be independently calibrated. Event-mask evidence is
        unrelated to this physical-only calibration.
        """
        model.assert_frozen()
        if (type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha < 1
                or bound_mode not in ('fixed_scale', 'ensemble_scaled')
                or type(std_floor) not in (int, float) or not math.isfinite(std_floor) or std_floor <= 0):
            raise ValueError('Invalid calibration miscoverage/bound protocol')
        if bound_mode == 'ensemble_scaled' and len(model.members) < 2:
            raise ValueError('Ensemble-scaled calibration requires genuine multiple members')
        protocol_hash = model.protocol_hash(samples=samples, mode=mode)
        lineage = model.metadata.get('episode_lineage')
        if lineage is None:
            raise ValueError('Independent calibration requires complete ancestor episode_lineage')
        validate_lineage([], lineage, context='calibration')
        scope = validate_scope(scope, model.profile)
        origins, identities, maxima, horizon = [], {}, {}, None
        collection_origins = {}
        manifest_hash = None
        if isinstance(records, EmbodiedDataset):
            dataset = records
            if dataset.split != 'calibration' or dataset.context_steps != model.config.context_steps:
                raise ValueError('Calibration requires independent calibration-split model-context windows')
            model.validate_profile(dataset.profile)
            # Task2 already validated complete manifest split isolation. The
            # model's own validation/selection rows are legitimately exposed;
            # only selected calibration origins must exclude ancestor exposure.
            validate_lineage(dataset.rows, lineage, context='calibration')
            manifest_hash = sha256_file(dataset.path)
            def dataset_records():
                by_id = {row['episode_id']: row for row in dataset.rows}
                for i in range(len(dataset)):
                    sample = dataset[i]
                    k = sample['prediction_offset']; low = k-sample['context_length']
                    future = known_commands(model.profile, sample['commands'][low:k+1], sample['actions'][k:])
                    if not allows_scope(scope, model.profile, sample['physical'][low:k+1],
                                        sample['commands'][low:k+1], sample['actions'][k:], future):
                        raise ValueError('Calibration window outside declared scope')
                    yield dict(**by_id[sample['episode_id']], profile=dataset.profile.to_dict(),
                        prediction=model.predict_sample(sample, samples=samples, mode=mode, seed=0),
                        target=sample['physical'][k+1:])
            iterator = dataset_records()
        else:
            iterator = iter(records)
        scales = np.asarray(model.normalization['physical_scale'], dtype='float64')
        for record in iterator:
            if not isinstance(record, dict) or record.get('split') != 'calibration':
                raise ValueError('Only independent calibration records allowed')
            required_record = {'episode_id', 'source_episode_id', 'sha256', 'source_sha256',
                               'profile', 'prediction', 'target'}
            if not required_record.issubset(record) or not isinstance(record['prediction'], dict):
                raise ValueError('Incomplete calibration record provenance/targets/prediction')
            row = {key: record[key] for key in ('episode_id', 'source_episode_id', 'sha256', 'source_sha256')}
            row.update(split='calibration', source_hashes=record.get('source_hashes', []))
            row = serializable_copy(row)
            validate_lineage([row], lineage, context='calibration')
            model.validate_profile(record['profile'])
            profile = record['profile']
            origin = (profile if isinstance(profile, ObservationProfile) else ObservationProfile.from_dict(profile)).origin
            collection_origins[content_hash(origin)] = origin
            if scope is not None:
                context = record.get('context')
                if context is None and not isinstance(records, EmbodiedDataset):
                    raise ValueError('Explicit scoped calibration requires raw origin/action context')
                if context is not None:
                    physical, commands, actions = [array_snapshot(context.get(key)) for key in
                                                  ('physical', 'commands', 'actions')]
                    if (physical.ndim != 2 or physical.shape[1] != model.config.physical_dim or not len(physical)
                            or commands.shape != (len(physical), model.config.command_dim)
                            or actions.ndim != 2 or actions.shape[1] != model.config.action_dim or not len(actions)):
                        raise ValueError('Invalid explicit raw scope context')
                    future = known_commands(model.profile, commands, actions)
                    if not allows_scope(scope, model.profile, physical, commands, actions, future):
                        raise ValueError('Explicit calibration record outside declared raw scope')
            if row['episode_id'] not in identities:
                identities[row['episode_id']] = row
                origins.append(row)
            elif identities[row['episode_id']] != row:
                raise ValueError('Inconsistent provenance between windows of one episode')
            prediction = record['prediction']
            if (prediction.get('model_version') != model.version
                    or prediction.get('protocol_hash') != protocol_hash):
                raise ValueError('Calibration prediction model/sampling protocol mismatch')
            predicted = array_snapshot(prediction['physical_states']).astype('float64')
            target = array_snapshot(record['target']).astype('float64')
            h = len(target) if target.ndim == 2 else 0
            if (not 1 <= h <= model.metadata.get('max_horizon', model.config.horizon)
                    or target.shape != (h, model.config.physical_dim)
                    or predicted.shape != (len(model.members), samples, h, model.config.physical_dim)
                    or (horizon is not None and horizon != h)):
                raise ValueError('Calibration requires matching bounded raw physical targets')
            horizon = h
            member_means = predicted.mean(1)
            denominator = (scales if bound_mode == 'fixed_scale' else
                           np.maximum(member_means.std(0), std_floor*scales))
            score = float((np.abs(member_means.mean(0)-target)/denominator).max())
            if not math.isfinite(score):
                raise ValueError('Nonfinite physical residual score')
            maxima[row['episode_id']] = max(maxima.get(row['episode_id'], 0.), score)
        validate_lineage(origins, lineage, context='calibration')
        grouped = {}
        groups = _groups(origins)
        for row, group in zip(origins, groups):
            grouped[group] = max(grouped.get(group, 0.), maxima[row['episode_id']])
        n = len(grouped)
        rank = math.ceil((n+1)*(1-alpha))
        if n < 2 or rank > n:
            raise ValueError('Not enough independent episodes for a finite requested quantile')
        bound_protocol = dict(schema=BOUND_PROTOCOL, bound_mode=bound_mode,
            denominator='training_physical_scale' if bound_mode == 'fixed_scale' else 'member_mean_std_with_training_scale_floor',
            aggregation='episode_max_over_windows_horizons_dimensions', std_floor=std_floor,
            physical_scale=scales.tolist(), horizon=horizon, protocol_hash=protocol_hash)
        ordered_origins = sorted(origins, key=lambda x: x['episode_id'])
        payload = dict(schema=SCHEMA, model_version=model.version, protocol_hash=protocol_hash,
            profile=model.profile.to_dict(), semantics=model.semantics, controller=model.profile.controller,
            alpha=alpha, samples=samples, mode=mode, seed=0, bound_protocol=bound_protocol,
            bound_protocol_hash=content_hash(bound_protocol), quantile=float(np.sort(list(grouped.values()))[rank-1]),
            episode_count=n, episode_records=ordered_origins,
            episode_groups=_groups(ordered_origins), calibration_manifest_sha256=manifest_hash,
            calibration_origins=[collection_origins[key] for key in sorted(collection_origins)],
            lineage=lineage, scope=scope)
        return cls._from_payload(payload)

    @classmethod
    def _from_payload(cls, payload):
        payload = serializable_copy(payload)
        required = {'schema', 'model_version', 'protocol_hash', 'profile', 'semantics', 'controller',
            'alpha', 'samples', 'mode', 'seed', 'bound_protocol', 'bound_protocol_hash', 'quantile',
            'episode_count', 'episode_records', 'episode_groups', 'calibration_manifest_sha256',
            'calibration_origins', 'lineage', 'scope'}
        if not isinstance(payload, dict) or payload.keys() != required or payload['schema'] != SCHEMA:
            raise ValueError('Expected complete independent embodied calibration')
        p = ObservationProfile.from_dict(payload['profile'])
        bp = payload['bound_protocol']
        if (payload['semantics'] != profile_semantics(p) or payload['controller'] != p.controller
                or content_hash(bp) != payload['bound_protocol_hash']
                or bp.get('schema') != BOUND_PROTOCOL or bp.get('bound_mode') not in ('fixed_scale', 'ensemble_scaled')
                or bp.get('protocol_hash') != payload['protocol_hash']
                or type(bp.get('horizon')) is not int or bp['horizon'] < 1
                or np.asarray(bp.get('physical_scale')).shape != (len(p.physical_order),)
                or np.any(np.asarray(bp['physical_scale']) <= 0)
                or type(payload['quantile']) not in (int, float) or payload['quantile'] < 0
                or type(payload['alpha']) not in (int, float) or not 0 < payload['alpha'] < 1
                or type(payload['episode_count']) is not int or payload['episode_count'] < 2):
            raise ValueError('Invalid calibration bounds/profile/protocol')
        validate_scope(payload['scope'], p)
        validate_lineage(payload['episode_records'], payload['lineage'], context='calibration load')
        if len(set(_groups(payload['episode_records']))) != payload['episode_count']:
            raise ValueError('Calibration independent episode count mismatch')
        return cls(json.dumps(payload, sort_keys=True, allow_nan=False))

    def validate(self, model, *, samples=8, mode='sample', profile=None, controller=None, scope=None):
        model.assert_frozen()
        p = self.payload
        if (model.version != p['model_version'] or model.protocol_hash(samples=samples, mode=mode) != p['protocol_hash']
                or model.semantics != p['semantics'] or model.profile.controller != p['controller']):
            raise ValueError('Calibration model/protocol/profile/controller mismatch')
        if profile is not None:
            model.validate_profile(profile)
        if controller is not None and controller != p['controller']:
            raise ValueError('Calibration controller mismatch')
        if scope is not None and validate_scope(scope, model.profile) != p['scope']:
            raise ValueError('Calibration deployment scope mismatch')
        return self

    def allows_scope(self, physical, commands, actions, *, model):
        self.validate(model, samples=self.payload['samples'], mode=self.payload['mode'])
        values = [array_snapshot(x) for x in (physical, commands, actions)]
        physical, commands, actions = values
        if (physical.ndim != 2 or physical.shape[1] != model.config.physical_dim or not len(physical)
                or commands.shape != (len(physical), model.config.command_dim)
                or actions.ndim != 2 or actions.shape[1] != model.config.action_dim or not len(actions)):
            raise ValueError('Invalid raw calibration scope input')
        future = known_commands(model.profile, commands, actions)
        return allows_scope(self.scope, model.profile, physical, commands, actions, future)

    def bounds(self, prediction=None, *, horizon=None, model, samples=8, mode='sample'):
        self.validate(model, samples=samples, mode=mode)
        p = self.payload; bp = p['bound_protocol']
        if prediction is not None:
            if (prediction.get('model_version') != model.version or prediction.get('protocol_hash') != self.protocol_hash):
                raise ValueError('Bound prediction identity mismatch')
            values = array_snapshot(prediction['physical_states'])
            if values.ndim != 4 or values.shape[:2] != (len(model.members), samples) or values.shape[-1] != model.config.physical_dim:
                raise ValueError('Malformed bound prediction samples')
            if horizon is not None and horizon != values.shape[2]:
                raise ValueError('Bound horizon differs from prediction')
            horizon = values.shape[2]
        if type(horizon) is not int or not 1 <= horizon <= bp['horizon']:
            raise ValueError('Horizon outside frozen joint calibration')
        scales = np.asarray(bp['physical_scale'])
        if bp['bound_mode'] == 'fixed_scale':
            denominator = np.broadcast_to(scales, (horizon, len(scales)))
        else:
            if prediction is None or len(model.members) < 2:
                raise ValueError('Ensemble-scaled bounds require genuine member trajectories')
            denominator = np.maximum(values.mean(1).std(0), bp['std_floor']*scales)
        return p['quantile'] * denominator

    def save(self, path):
        payload = self.payload
        atomic_json(path, dict(payload=payload, content_hash=content_hash(payload)))

    @classmethod
    def load(cls, path):
        saved = json.loads(Path(path).read_text())
        if (not isinstance(saved, dict) or saved.keys() != {'payload', 'content_hash'}
                or content_hash(saved['payload']) != saved['content_hash']):
            raise ValueError('Calibration complete content hash mismatch (legacy files unsupported)')
        return cls._from_payload(saved['payload'])


class VisualCalibrationAdapter:
    """Legacy raw bounds: physical interval plus exact-command zero errors."""
    def __init__(self, calibration, predictor):
        self.calibration, self.predictor = calibration, predictor

    def validate_for(self, version, semantics):
        if version != self.predictor.version or semantics != self.predictor.semantics:
            raise ValueError('Legacy calibration adapter identity mismatch')
        self.calibration.validate(self.predictor.model, samples=self.predictor.samples)

    def validate_controller(self, profile):
        self.calibration.validate(self.predictor.model, samples=self.predictor.samples, controller=profile)

    def allows_scope(self, states, actions):
        states = array_snapshot(states)
        if states.ndim != 2 or states.shape[1] != 4:
            raise ValueError('Legacy scope requires four named G1 state coordinates')
        return self.calibration.allows_scope(states[:, :2], states[:, 2:], actions, model=self.predictor.model)

    def bounds(self, std, *, version):
        self.validate_for(version, self.predictor.semantics)
        std = array_snapshot(std)
        if std.ndim != 2 or std.shape[1] != 4 or np.any(std < 0):
            raise ValueError('Legacy bound input requires finite nonnegative [H,4] spread')
        bp = self.calibration.payload['bound_protocol']
        if bp['bound_mode'] == 'fixed_scale':
            physical = self.calibration.bounds(horizon=len(std), model=self.predictor.model,
                                               samples=self.predictor.samples)
        else:
            if not 1 <= len(std) <= bp['horizon']:
                raise ValueError('Horizon outside frozen calibration')
            physical = self.calibration.quantile*np.maximum(std[:, :2],
                bp['std_floor']*np.asarray(bp['physical_scale']))
        return np.concatenate([physical, np.zeros((len(std), 2))], -1)
