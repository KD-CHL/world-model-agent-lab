"""Explicit, serializable observation/control semantics for embodied datasets.

Positions and controller targets are separate coordinates. Missing measurements
(for example G1 velocities) are never filled in by a profile or legacy mapping.
``none`` declares an offline action profile with no justified command rollout.
"""
from dataclasses import dataclass
import json
import math


def serializable_copy(value):
    """Snapshot JSON values without coercing arrays, nonfinite floats or keys."""
    def check(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) in (list, tuple):
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError('Metadata must contain finite serializable JSON values')
    check(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _names(values, name, *, empty=False, unique=True):
    if (not isinstance(values, (list, tuple)) or (not values and not empty)
            or any(type(value) is not str or not value.strip() for value in values)
            or (unique and len(set(values)) != len(values))):
        raise ValueError(f'{name} must contain explicit named dimensions')
    return tuple(values)


@dataclass(frozen=True, init=False)
class ObservationProfile:
    """Immutable dimensions plus independent controller and origin snapshots.

    Units are mandatory per coordinate. ``joint_delta`` means the next command
    equals the current command plus its corresponding raw action; target names
    and units must make that correspondence unambiguous. Other action semantics
    use ``none`` and cannot be used for deterministic inference by a provider.
    """
    physical_order: tuple[str, ...]
    command_order: tuple[str, ...]
    action_order: tuple[str, ...]
    action_schema: str
    camera_ids: tuple[str, ...]
    image_size: int
    period_s: float
    physical_units: tuple[str, ...]
    command_units: tuple[str, ...]
    action_units: tuple[str, ...]
    command_rule: str
    event_names: tuple[str, ...]
    _controller_json: str
    _origin_json: str

    def __init__(self, *, physical_order, command_order, action_order, action_schema,
                 camera_ids, image_size, period_s, physical_units, command_units,
                 action_units, controller, origin, command_rule, event_names=()):
        values = {}
        for name, value in [('physical_order', physical_order), ('command_order', command_order),
                            ('action_order', action_order), ('camera_ids', camera_ids),
                            ('event_names', event_names)]:
            values[name] = _names(value, name, empty=name in ('command_order', 'event_names'))
        for name, value, order in [('physical_units', physical_units, 'physical_order'),
                                   ('command_units', command_units, 'command_order'),
                                   ('action_units', action_units, 'action_order')]:
            values[name] = _names(value, name, empty=order == 'command_order', unique=False)
            if len(values[name]) != len(values[order]):
                raise ValueError(f'{name} must specify every coordinate unit')
        if type(action_schema) is not str or not action_schema.strip():
            raise ValueError('Explicit action_schema required')
        if type(image_size) is not int or image_size < 16 or image_size & (image_size - 1):
            raise ValueError('image_size must be a power of two >= 16')
        if (type(period_s) not in (int, float) or not math.isfinite(period_s) or period_s <= 0):
            raise ValueError('period_s must be finite and positive')
        if command_rule not in ('joint_delta', 'none'):
            raise ValueError('Unsupported known-command rule')
        if command_rule == 'joint_delta' and (
                values['command_order'] != tuple(name + '_target' for name in values['action_order'])
                or values['command_units'] != values['action_units']):
            raise ValueError('joint_delta requires aligned target/action names and units')
        for name, value in [('controller', controller), ('origin', origin)]:
            snapshot = serializable_copy(value)
            if not isinstance(snapshot, dict) or not snapshot:
                raise ValueError(f'Explicit {name} identity required')
            if name == 'controller' and (type(snapshot.get('profile')) is not str or not snapshot['profile'].strip()):
                raise ValueError('Explicit controller profile identity required')
            if name == 'origin':
                identities = [snapshot.get(key) for key in ('dataset', 'repo_id', 'source_id', 'kind')]
                dataset = snapshot.get('dataset')
                if isinstance(dataset, dict):
                    identities.append(dataset.get('repo_id'))
                if not any(type(identity) is str and identity.strip() for identity in identities):
                    raise ValueError('Origin must identify its dataset, repository or collection source')
            values['_' + name + '_json'] = json.dumps(snapshot, sort_keys=True, allow_nan=False)
        values.update(action_schema=action_schema, image_size=image_size, period_s=float(period_s),
                      command_rule=command_rule)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def controller(self):
        return json.loads(self._controller_json)

    @property
    def origin(self):
        return json.loads(self._origin_json)

    def to_dict(self):
        result = {name: list(getattr(self, name)) for name in (
            'physical_order', 'command_order', 'action_order', 'camera_ids', 'event_names',
            'physical_units', 'command_units', 'action_units')}
        result.update(action_schema=self.action_schema, image_size=self.image_size,
                      period_s=self.period_s, command_rule=self.command_rule,
                      controller=self.controller, origin=self.origin)
        return result

    @classmethod
    def from_dict(cls, payload):
        values = serializable_copy(payload)
        if not isinstance(values, dict):
            raise ValueError('Observation profile must be a dictionary')
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError('Missing or unknown observation profile fields') from exc
