"""Raw-coordinate, versioned provider for Task1's normalized RSSM family.

The provider rebuilds a posterior from bounded real history for every candidate.
Command integration is float64 in raw units, before any float32 normalization.
Single-member and sampled-trajectory axes are distinct and never duplicated.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from wmal.agents.prediction_contracts import (PREDICTION_SCHEMA, allows_scope, array_hash,
    content_hash, known_commands, profile_semantics, sampling_options, validate_context, validate_scope)
from wmal.datasets.embodied_profiles import ObservationProfile, serializable_copy
from wmal.datasets.embodied_sequences import validate_lineage
from .embodied_config import ScalableConfig
from .embodied_rssm import ScalableRSSMMember

SCHEMA = 'wmal.embodied_world.v1'
SEED_RULE = 'sha256-json-[embodied-v1,decision_seed,member_index,sample_index]-first8-be-mod2**63'


def trajectory_seed(seed, member, sample):
    sampling_options(seed)
    digest = sha256(json.dumps(['embodied-v1', seed, member, sample], separators=(',', ':')).encode()).digest()
    return int.from_bytes(digest[:8], 'big') % 2**63


def _weights_hash(weights):
    return content_hash({key: array_hash(value.detach().cpu().numpy())
                         for key, value in sorted(weights.items())})


class EmbodiedWorldModel:
    def __init__(self, config, profile, normalization, members, *, metadata=None, device='cpu'):
        if not isinstance(config, ScalableConfig):
            raise ValueError('Expected scalable embodied configuration')
        profile = profile if isinstance(profile, ObservationProfile) else ObservationProfile.from_dict(profile)
        dimensions = (len(profile.physical_order), len(profile.command_order), len(profile.action_order),
                      len(profile.camera_ids), profile.image_size, len(profile.event_names))
        if dimensions != (config.physical_dim, config.command_dim, config.action_dim,
                          config.views, config.image_size, config.event_dim):
            raise ValueError('Model dimensions differ from explicit observation profile')
        if (not isinstance(members, (list, tuple)) or not members
                or any(not isinstance(m, ScalableRSSMMember) or m.config != config for m in members)
                or len({id(m) for m in members}) != len(members)):
            raise ValueError('At least one genuine independent matching member required')
        norm = serializable_copy(normalization)
        expected = {prefix+'_'+kind for prefix in ('physical', 'command', 'action') for kind in ('mean', 'scale')}
        if not isinstance(norm, dict) or norm.keys() != expected:
            raise ValueError('Complete physical/command/action normalization required')
        for prefix, width in [('physical', config.physical_dim), ('command', config.command_dim),
                              ('action', config.action_dim)]:
            for kind in ('mean', 'scale'):
                value = np.asarray(norm[prefix+'_'+kind])
                if (value.dtype.kind not in 'fiu' or value.shape != (width,)
                        or not np.isfinite(value).all() or (kind == 'scale' and np.any(value <= 0))):
                    raise ValueError('Invalid training-fixed normalization')
        meta = serializable_copy(metadata if metadata is not None else {})
        if not isinstance(meta, dict):
            raise ValueError('Model metadata must be a dictionary')
        evidence = meta.get('event_supervised_by_member', [[False]*config.event_dim for _ in members])
        if (not isinstance(evidence, list) or len(evidence) != len(members)
                or any(not isinstance(row, list) or len(row) != config.event_dim
                       or any(type(x) is not bool for x in row) for row in evidence)):
            raise ValueError('Invalid per-member recorded event-mask evidence')
        if 'episode_lineage' in meta:
            meta['episode_lineage'] = validate_lineage([], meta['episode_lineage'], context='model')
        validate_scope(meta.get('prediction_scope'), profile)
        limit = meta.get('max_horizon', config.horizon)
        if type(limit) is not int or not 1 <= limit <= config.horizon:
            raise ValueError('Invalid trained horizon bound')
        self.config, self.profile, self.device = config, profile, str(device)
        self._normalization_json = json.dumps(norm, sort_keys=True)
        self._metadata_json = json.dumps(meta, sort_keys=True)
        self.members = tuple(m.to(device=device, dtype=torch.float32).eval() for m in members)
        self._event_channels = tuple(np.asarray(evidence, dtype=bool).all(0).tolist())
        hashes = [_weights_hash(m.state_dict()) for m in self.members]
        if len(set(hashes)) != len(hashes):
            raise ValueError('Duplicated member weights do not form a genuine ensemble')
        if any(not torch.isfinite(v).all() for m in self.members for v in m.state_dict().values()):
            raise ValueError('Nonfinite member weights')
        self._version = self._content_version()

    @property
    def normalization(self):
        return json.loads(self._normalization_json)

    @property
    def metadata(self):
        return json.loads(self._metadata_json)

    @property
    def semantics(self):
        return profile_semantics(self.profile)

    @property
    def version(self):
        return self._version

    @property
    def event_supervised_channels(self):
        return list(self._event_channels)

    def _identity(self):
        return dict(schema=SCHEMA, network=asdict(self.config), profile=self.profile.to_dict(),
            normalization=self.normalization, metadata=self.metadata,
            sampling=dict(schema=PREDICTION_SCHEMA, seed_rule=SEED_RULE, dtype='float32',
                          default_samples=8, default_mode='sample'),
            member_hashes=[_weights_hash(m.state_dict()) for m in self.members])

    def _content_version(self):
        return 'embodied-' + content_hash(self._identity())

    def assert_frozen(self):
        if self._content_version() != self.version:
            raise ValueError('Frozen model content changed; export a new model and recalibrate')

    def protocol(self, *, samples=8, mode='sample'):
        sampling_options(samples=samples, mode=mode)
        return dict(schema=PREDICTION_SCHEMA, model_version=self.version, samples=samples, mode=mode,
            seed_rule=SEED_RULE, dtype='float32', normalization=self.normalization,
            profile=self.profile.to_dict(), members=len(self.members),
            event_supervised_channels=self.event_supervised_channels,
            context_steps=self.config.context_steps, max_horizon=self.metadata.get('max_horizon', self.config.horizon))

    def protocol_hash(self, *, samples=8, mode='sample'):
        return content_hash(self.protocol(samples=samples, mode=mode))

    def validate_profile(self, profile):
        if profile_semantics(profile) != self.semantics:
            raise ValueError('Robot/control/camera semantics differ; independent calibration required')

    def predict_context(self, rgb, physical, commands, past_actions, actions, *, seed=0,
                        samples=8, mode='sample', view_mask=None):
        sampling_options(seed, samples, mode)
        self.assert_frozen()
        start = time.perf_counter()
        rgb, physical, commands, past_actions, actions, mask = validate_context(self.profile,
            rgb, physical, commands, past_actions, actions, view_mask, self.config.context_steps,
            self.metadata.get('max_horizon', self.config.horizon))
        future = known_commands(self.profile, commands, actions)
        if not allows_scope(self.metadata.get('prediction_scope'), self.profile, physical, commands, actions, future):
            raise ValueError('Candidate outside raw command/physical/action prediction scope')
        norm = self.normalization
        def normalize(values, prefix):
            return (values.astype('float64') - norm[prefix+'_mean']) / norm[prefix+'_scale']
        values = [rgb, normalize(physical, 'physical'), normalize(commands, 'command'),
            normalize(past_actions, 'action'), normalize(actions, 'action'), normalize(future, 'command')]
        if any(np.any(np.abs(x) > np.finfo('float32').max) for x in values):
            raise ValueError('Normalized coordinates cannot be represented as finite float32')
        tensors = [torch.as_tensor(np.array(x, copy=True), dtype=torch.float32, device=self.device)[None]
                   for x in values]
        mask_tensor = torch.as_tensor(np.array(mask, copy=True), dtype=torch.bool, device=self.device)[None]
        physicals, frames, events, details, seeds = [], [], [], [], []
        with torch.inference_mode():
            for index, member in enumerate(self.members):
                mp, mf, me, md, ms = [], [], [], [], []
                for sample in range(samples):
                    sample_seed = trajectory_seed(seed, index, sample)
                    output = member.imagine_context(*tensors, view_mask=mask_tensor,
                        sample=mode == 'sample', seed=sample_seed)
                    mp.append(output['physical'][0].cpu().numpy().astype('float64') *
                              norm['physical_scale'] + norm['physical_mean'])
                    mf.append(output['rgb'][0].cpu().numpy())
                    if any(self._event_channels):
                        me.append(output['event_logits'][0].sigmoid().cpu().numpy()[:, self._event_channels])
                    md.append(dict(prior_entropy=output['prior_entropy'][0].cpu().tolist(),
                                   posterior_entropy=float(output['posterior_entropy'][0])))
                    ms.append(sample_seed)
                physicals.append(np.stack(mp)); frames.append(np.stack(mf))
                if me: events.append(np.stack(me))
                details.append(md); seeds.append(ms)
        physicals, frames = np.stack(physicals), np.stack(frames)
        probabilities = np.stack(events) if events else None
        if (not np.isfinite(physicals).all() or not np.isfinite(frames).all()
                or (probabilities is not None and not np.isfinite(probabilities).all())):
            raise ValueError('Nonfinite world-model output')
        return dict(schema=PREDICTION_SCHEMA, model_version=self.version,
            protocol_hash=self.protocol_hash(samples=samples, mode=mode),
            protocol=self.protocol(samples=samples, mode=mode), seed=seed,
            physical_states=physicals, known_commands=future, frames=frames,
            event_probabilities=probabilities, event_probability_names=[name for name, enabled in
                zip(self.profile.event_names, self._event_channels) if enabled],
            event_supervised_channels=self.event_supervised_channels,
            view_mask=np.array(mask, copy=True), semantics=self.semantics,
            diagnostics=dict(context_steps_used=len(physical)-1, context_steps_limit=self.config.context_steps,
                inference_mode=mode, trajectory_seeds=seeds, members=details,
                within_member_std=physicals.std(1).tolist(),
                between_member_std=physicals.mean(1).std(0).tolist(),
                uncertainty_kind='raw_sample_and_member_spread_not_safety_probabilities',
                inference_seconds=time.perf_counter()-start))

    def predict_sample(self, sample, *, actions=None, **kwargs):
        k, length = self.config.context_steps, sample.get('context_length')
        if (sample.get('prediction_offset') != k or type(length) is not int or not 0 <= length <= k):
            raise ValueError('Dataset startup context protocol differs from model')
        low = k-length
        if 'view_mask' in kwargs:
            raise ValueError('Dataset sample supplies its own observed-view masks')
        return self.predict_context(sample['rgb'][low:k+1], sample['physical'][low:k+1],
            sample['commands'][low:k+1], sample['actions'][low:k],
            sample['actions'][k:] if actions is None else actions,
            view_mask=sample['view_mask'][low:k+1], **kwargs)

    def save(self, path):
        self.assert_frozen()
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._identity()
        payload.update(model_version=self.version,
                       members=[{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in self.members])
        handle = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
        temporary = Path(handle.name); handle.close()
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, device='cpu'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        keys = {'schema', 'network', 'profile', 'normalization', 'metadata', 'sampling',
                'member_hashes', 'model_version', 'members'}
        if not isinstance(payload, dict) or payload.keys() != keys or payload['schema'] != SCHEMA:
            raise ValueError('Expected complete independent embodied checkpoint')
        try:
            config = ScalableConfig(**payload['network'])
            if (not isinstance(payload['members'], list) or not payload['members']
                    or [_weights_hash(weights) for weights in payload['members']] != payload['member_hashes']):
                raise ValueError('Stored member tensor content/dtype hash mismatch')
            members = []
            # Initialization must not alter the caller's RNG state.
            with torch.random.fork_rng(devices=[]):
                for weights in payload['members']:
                    member = ScalableRSSMMember(config)
                    member.load_state_dict(weights, strict=True)
                    members.append(member)
            result = cls(config, payload['profile'], payload['normalization'], members,
                         metadata=payload['metadata'], device=device)
            actual = result._identity()
            if (result.version != payload['model_version']
                    or any(content_hash(actual[k]) != content_hash(payload[k]) for k in actual)):
                raise ValueError('Checkpoint complete content/version mismatch')
            return result
        except (KeyError, TypeError, RuntimeError) as exc:
            raise ValueError('Malformed embodied checkpoint') from exc

    def as_visual_predictor(self, calibration=None, samples=8):
        return EmbodiedVisualPredictor(self, calibration, samples)


class EmbodiedVisualPredictor:
    """Exact G1 legacy mapping; retains M real members, averages only S samples."""
    family = SCHEMA

    def __init__(self, model, calibration, samples):
        p = model.profile
        joints = ('left_shoulder_pitch_joint', 'left_elbow_joint')
        if (p.physical_order != joints or p.action_order != joints
                or p.command_order != tuple(j+'_target' for j in joints)
                or p.action_schema != 'wmal.g1.workcell.target_delta.v1' or p.command_rule != 'joint_delta'
                or p.physical_units != ('rad',)*2 or p.command_units != ('rad',)*2
                or p.action_units != ('rad',)*2):
            raise ValueError('Legacy Agent requires the exact G1 measured-joint/target-delta mapping')
        sampling_options(samples=samples)
        self.model, self.samples = model, samples
        if calibration is not None:
            from .embodied_calibration import VisualCalibrationAdapter
            calibration.validate(model, samples=samples)
            self.calibration = VisualCalibrationAdapter(calibration, self)
        else:
            self.calibration = None

    @property
    def version(self):
        return self.model.version

    @property
    def config(self):
        return self.model.config

    @property
    def semantics(self):
        p = self.model.profile
        return dict(state_order=list(p.physical_order+p.command_order), action_order=list(p.action_order),
            action_schema=p.action_schema, image_size=p.image_size, camera=p.camera_ids[0],
            period_s=p.period_s, event_names=list(p.event_names))

    @property
    def normalization(self):
        norm = self.model.normalization
        # The old selector divides bounds by state_scale. Physical scales must
        # remain training physical scales; exact commands have zero bounds.
        return dict(state_mean=norm['physical_mean']+norm['command_mean'],
            state_scale=norm['physical_scale']+norm['command_scale'],
            action_mean=norm['action_mean'], action_scale=norm['action_scale'])

    def predict_context(self, rgb, state, past_actions, actions, *, seed=0):
        rgb = np.asarray(rgb)
        if rgb.ndim == 4 and len(self.model.profile.camera_ids) == 1:
            rgb = rgb[:, None]
        elif rgb.ndim != 5:
            raise ValueError('Multi-view model requires explicit real views; legacy camera cannot invent missing views')
        state = np.asarray(state)
        if state.ndim != 2 or state.shape[1] != 4:
            raise ValueError('Expected legacy four-state real history')
        output = self.model.predict_context(rgb, state[:, :2], state[:, 2:],
            past_actions, actions, seed=seed, samples=self.samples)
        physical = output['physical_states'].mean(1)
        commands = np.broadcast_to(output['known_commands'], (*physical.shape[:-1], 2))
        output = dict(output)
        output.update(states=np.concatenate([physical, commands], -1), frames=output['frames'].mean(1)[:, :, 0])
        return output

    def predict(self, rgb, state, actions, *, seed=0):
        return self.predict_context(np.asarray(rgb)[None], np.asarray(state)[None],
                                    np.empty((0, 2)), actions, seed=seed)
