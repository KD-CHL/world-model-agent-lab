"""Local, immutable prediction requests. No transport or execution authority.

Arrays retain caller dtype, with canonical little-endian content hashes including
shape. Bytes-backed snapshots cannot be made writable. JSON snapshots are exposed
as detached copies. Deadlines use the client's monotonic clock. Remote servers
must derive their own local deadline from a remaining transport budget, preserve
the echoed client binding, and leave client-clock expiry validation to the client.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import time

import numpy as np

from wmal.datasets.embodied_profiles import ObservationProfile, serializable_copy

PREDICTION_SCHEMA = 'wmal.embodied_prediction.v1'


def content_hash(value):
    return sha256(json.dumps(serializable_copy(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_snapshot(value):
    array = np.asarray(value)
    if array.dtype.kind not in 'bifu' or not np.isfinite(array).all():
        raise ValueError('Expected finite numeric or boolean array')
    dtype = array.dtype.newbyteorder('<')
    array = np.ascontiguousarray(array.astype(dtype, copy=False))
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


def array_hash(value):
    array = array_snapshot(value)
    header = json.dumps([array.dtype.str, list(array.shape)], separators=(',', ':')).encode()
    return sha256(header + b'\0' + array.tobytes()).hexdigest()


def profile_semantics(profile):
    """Robot/control/camera compatibility excludes collection provenance only."""
    result = (profile.to_dict() if isinstance(profile, ObservationProfile)
              else ObservationProfile.from_dict(profile).to_dict())
    result.pop('origin')
    return result


def sampling_options(seed=0, samples=8, mode='sample'):
    if (type(seed) is not int or not 0 <= seed < 2**63
            or type(samples) is not int or not 1 <= samples <= 1024
            or mode not in ('sample', 'soft')):
        raise ValueError('Invalid bounded sampling protocol')


def validate_context(profile, rgb, physical, commands, past_actions, actions,
                     view_mask, context_steps, max_horizon):
    p = profile
    rgb, physical, commands, past_actions, actions = [array_snapshot(x) for x in
        (rgb, physical, commands, past_actions, actions)]
    t = len(physical) if physical.ndim == 2 else 0
    h = len(actions) if actions.ndim == 2 else 0
    if (type(context_steps) is not int or context_steps < 0
            or type(max_horizon) is not int or max_horizon < 1
            or not 1 <= t <= context_steps+1 or not 1 <= h <= max_horizon
            or physical.shape != (t, len(p.physical_order))
            or commands.shape != (t, len(p.command_order))
            or past_actions.shape != (t-1, len(p.action_order))
            or actions.shape != (h, len(p.action_order))
            or rgb.shape != (t, len(p.camera_ids), 3, p.image_size, p.image_size)
            or np.any(rgb < 0) or np.any(rgb > 1)):
        raise ValueError('Expected bounded aligned real context and candidate actions')
    if view_mask is None:
        view_mask = np.ones((t, len(p.camera_ids)), dtype=bool)
    view_mask = array_snapshot(view_mask)
    if (view_mask.dtype.kind != 'b' or view_mask.shape != (t, len(p.camera_ids))
            or not view_mask.any(-1).all()):
        raise ValueError('Each real observation requires a boolean observed-view mask')
    if p.command_rule == 'joint_delta' and t > 1 and not np.allclose(
            commands[1:].astype('float64'), commands[:-1].astype('float64') +
            past_actions.astype('float64'), rtol=0, atol=1e-6):
        raise ValueError('Past executed deltas do not align with recorded commands')
    return rgb, physical, commands, past_actions, actions, view_mask


def known_commands(profile, commands, actions):
    if not profile.command_order:
        return np.empty((len(actions), 0), dtype='float64')
    if profile.command_rule != 'joint_delta':
        raise ValueError('Future commands are unknown for this offline action profile')
    result = commands[-1].astype('float64') + np.cumsum(actions.astype('float64'), axis=0)
    if not np.isfinite(result).all():
        raise ValueError('Raw command integration overflow')
    return result


def validate_scope(scope, profile):
    if scope is None:
        return None
    result = serializable_copy(scope)
    allowed = {'command_lower', 'command_upper', 'physical_lower', 'physical_upper',
               'action_lower', 'action_upper', 'max_action_delta', 'measured_margin'}
    if not isinstance(result, dict) or not result or result.keys() - allowed:
        raise ValueError('Unsupported or empty prediction scope')
    for name, width in [('command', len(profile.command_order)),
                        ('physical', len(profile.physical_order)), ('action', len(profile.action_order))]:
        low, high = result.get(name+'_lower'), result.get(name+'_upper')
        if low is not None or high is not None:
            if (low is None or high is None or width == 0
                    or np.asarray(low).shape != (width,) or np.asarray(high).shape != (width,)
                    or np.asarray(low).dtype.kind not in 'fiu' or np.asarray(high).dtype.kind not in 'fiu'
                    or np.any(np.asarray(low) >= np.asarray(high))):
                raise ValueError('Invalid raw coordinate scope')
    for key in ('max_action_delta', 'measured_margin'):
        if key in result and (type(result[key]) not in (int, float)
                or result[key] < 0 or (key == 'max_action_delta' and result[key] == 0)):
            raise ValueError('Invalid scope scalar')
    if 'measured_margin' in result and ('command_lower' not in result
            or len(profile.command_order) != len(profile.physical_order)
            or profile.command_order != tuple(x+'_target' for x in profile.physical_order)
            or profile.command_units != profile.physical_units):
        raise ValueError('Measured margin requires corresponding physical and command coordinates')
    return result


def allows_scope(scope, profile, physical, commands, actions, future):
    scope = validate_scope(scope, profile)
    if scope is None:
        return True
    for name, values in [('command', np.concatenate([commands, future])),
                          ('physical', physical), ('action', actions)]:
        if name+'_lower' in scope and (np.any(values < scope[name+'_lower'])
                                      or np.any(values > scope[name+'_upper'])):
            return False
    if 'max_action_delta' in scope and np.any(np.abs(actions) > scope['max_action_delta']):
        return False
    if 'measured_margin' in scope:
        margin = scope['measured_margin']
        if (np.any(physical < np.asarray(scope['command_lower'])-margin)
                or np.any(physical > np.asarray(scope['command_upper'])+margin)):
            return False
    return True


@dataclass(frozen=True, kw_only=True)
class PredictionRequest:
    request_id: str
    decision_id: str
    run_id: str
    task_id: str
    node_id: str
    episode_id: str
    step_id: int
    context_step_ids: tuple
    observation_times: tuple
    candidate_id: str
    skill_version: str
    profile: ObservationProfile
    controller: dict
    model_version: str
    protocol_hash: str
    rgb: np.ndarray
    physical: np.ndarray
    commands: np.ndarray
    past_actions: np.ndarray
    actions: np.ndarray
    deadline: float
    timeout_s: float
    context_steps: int
    max_horizon: int
    member_count: int
    view_mask: np.ndarray | None = None
    samples: int = 8
    mode: str = 'sample'
    seed: int = 0
    event_supervised_channels: tuple | None = None
    context_hash: str = field(init=False)
    actions_hash: str = field(init=False)
    request_hash: str = field(init=False)

    def __post_init__(self):
        sampling_options(self.seed, self.samples, self.mode)
        for key in ('request_id', 'decision_id', 'run_id', 'task_id', 'node_id', 'episode_id',
                    'candidate_id', 'skill_version', 'model_version', 'protocol_hash'):
            value = getattr(self, key)
            if type(value) is not str or not value.strip():
                raise ValueError('Request identities must be explicit')
        p = self.profile if isinstance(self.profile, ObservationProfile) else ObservationProfile.from_dict(self.profile)
        controller = serializable_copy(self.controller)
        if controller != p.controller:
            raise ValueError('Request controller differs from observation profile')
        # Store controller as canonical JSON; property access is a detached copy.
        object.__setattr__(self, 'profile', p)
        object.__setattr__(self, 'controller', json.dumps(controller, sort_keys=True))
        evidence = self.event_supervised_channels
        if evidence is None:
            evidence = (False,)*len(p.event_names)
        if (not isinstance(evidence, (list, tuple)) or len(evidence) != len(p.event_names)
                or any(type(x) is not bool for x in evidence)):
            raise ValueError('Request requires explicit artifact-bound event supervision availability')
        object.__setattr__(self, 'event_supervised_channels', tuple(evidence))
        arrays = validate_context(p, self.rgb, self.physical, self.commands, self.past_actions,
            self.actions, self.view_mask, self.context_steps, self.max_horizon)
        for key, value in zip(('rgb', 'physical', 'commands', 'past_actions', 'actions', 'view_mask'), arrays):
            object.__setattr__(self, key, value)
        known_commands(p, self.commands, self.actions)
        ids, times = tuple(self.context_step_ids), tuple(self.observation_times)
        if (type(self.step_id) is not int or self.step_id < 0 or len(ids) != len(self.physical)
                or any(type(x) is not int or x < 0 for x in ids)
                or ids != tuple(range(self.step_id-len(ids)+1, self.step_id+1))
                or len(times) != len(ids) or any(type(x) not in (int, float) or not math.isfinite(x) for x in times)
                or (len(times) > 1 and not np.allclose(np.diff(times), p.period_s, rtol=1e-5, atol=1e-7))
                or type(self.member_count) is not int or self.member_count < 1
                or type(self.deadline) not in (int, float) or not math.isfinite(self.deadline)
                or type(self.timeout_s) not in (int, float) or not math.isfinite(self.timeout_s)
                or self.timeout_s <= 0 or self.deadline <= time.monotonic()):
            raise ValueError('Invalid contiguous request context, limits or client deadline')
        object.__setattr__(self, 'context_step_ids', ids)
        object.__setattr__(self, 'observation_times', times)
        hashes = {key: array_hash(getattr(self, key)) for key in
                  ('rgb', 'physical', 'commands', 'past_actions', 'view_mask')}
        object.__setattr__(self, 'context_hash', content_hash(hashes))
        object.__setattr__(self, 'actions_hash', array_hash(self.actions))
        object.__setattr__(self, 'request_hash', content_hash(self.binding(include_hash=False)))

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        return json.loads(value) if name == 'controller' and isinstance(value, str) else value

    @property
    def horizon(self):
        return len(self.actions)

    def binding(self, *, include_hash=True):
        keys = ('request_id', 'decision_id', 'run_id', 'task_id', 'node_id', 'episode_id',
                'step_id', 'candidate_id', 'skill_version', 'model_version', 'protocol_hash',
                'context_hash', 'actions_hash', 'deadline', 'timeout_s', 'samples', 'mode', 'seed',
                'context_steps', 'max_horizon', 'member_count')
        result = {key: getattr(self, key) for key in keys}
        result.update(schema=PREDICTION_SCHEMA, profile=self.profile.to_dict(), controller=self.controller,
            controller_hash=content_hash(self.controller), profile_hash=content_hash(self.profile.to_dict()),
            context_step_ids=list(self.context_step_ids), observation_times=list(self.observation_times),
            horizon=self.horizon, event_supervised_channels=list(self.event_supervised_channels))
        if include_hash:
            result['request_hash'] = self.request_hash
        return result

    def bind_response(self, prediction):
        """Attach local request binding without overwriting a mismatched model/protocol."""
        if (prediction.get('model_version') != self.model_version
                or prediction.get('protocol_hash') != self.protocol_hash
                or prediction.get('seed', self.seed) != self.seed
                or prediction.get('event_supervised_channels') != list(self.event_supervised_channels)):
            raise ValueError('Prediction differs from requested model/protocol/seed')
        response = dict(prediction)
        response.update(self.binding())
        validate_response(self, response)
        return response


def validate_response(request, response, *, now=None):
    if not isinstance(request, PredictionRequest) or not isinstance(response, dict):
        raise ValueError('Expected immutable request and prediction dictionary')
    now = time.monotonic() if now is None else now
    if not np.isfinite(now) or now >= request.deadline:
        raise ValueError('Prediction received after client deadline')
    binding = request.binding()
    if content_hash({key: response.get(key) for key in binding}) != content_hash(binding):
        raise ValueError('Prediction request/content/reset/deadline binding mismatch')
    protocol = response.get('protocol')
    if (not isinstance(protocol, dict) or content_hash(protocol) != request.protocol_hash
            or protocol.get('schema') != PREDICTION_SCHEMA or protocol.get('model_version') != request.model_version
            or protocol.get('profile') != request.profile.to_dict()
            or protocol.get('samples') != request.samples or protocol.get('mode') != request.mode
            or protocol.get('members') != request.member_count
            or protocol.get('context_steps') != request.context_steps or protocol.get('max_horizon') != request.max_horizon
            or protocol.get('event_supervised_channels') != list(request.event_supervised_channels)):
        raise ValueError('Prediction content differs from requested frozen protocol')
    serializable_copy(response.get('diagnostics', {}))
    mask = array_snapshot(response.get('view_mask'))
    if (content_hash(response.get('semantics')) != content_hash(profile_semantics(request.profile))
            or mask.dtype.kind != 'b' or not np.array_equal(mask, request.view_mask)):
        raise ValueError('Prediction units/semantics or observed-view availability changed')
    p, h, m, s = request.profile, request.horizon, request.member_count, request.samples
    shapes = {'physical_states': (m, s, h, len(p.physical_order)),
        'known_commands': (h, len(p.command_order)),
        'frames': (m, s, h, len(p.camera_ids), 3, p.image_size, p.image_size)}
    for key, shape in shapes.items():
        value = array_snapshot(response.get(key))
        if value.shape != shape or (key == 'frames' and (np.any(value < 0) or np.any(value > 1))):
            raise ValueError('Malformed prediction output shape/range')
    if not np.array_equal(response['known_commands'], known_commands(p, request.commands, request.actions)):
        raise ValueError('Prediction changed exact raw commands')
    names = response.get('event_probability_names', [])
    evidence = response.get('event_supervised_channels', [False]*len(p.event_names))
    if (not isinstance(evidence, list) or len(evidence) != len(p.event_names)
            or any(type(x) is not bool for x in evidence)
            or names != [name for name, enabled in zip(p.event_names, evidence) if enabled]):
        raise ValueError('Event output lacks consistent supervision evidence')
    events = response.get('event_probabilities')
    if names:
        value = array_snapshot(events)
        if value.shape != (m, s, h, len(names)) or np.any(value < 0) or np.any(value > 1):
            raise ValueError('Malformed supervised event probabilities')
    elif events is not None:
        raise ValueError('Unsupervised event probabilities forbidden')
    return response
