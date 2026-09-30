"""ROS-independent protocol core for a single-owner G1 simulator session."""
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import math
import time
from threading import Lock

from wmal.locomotion.contracts import G1State, G1VelocityAction

SCHEMA = 'wmal.g1.session.v1'


def scene_digest(scene):
    payload = {'bounds': scene.bounds, 'robot_radius': scene.robot_radius,
               'boxes': [asdict(box) for box in scene.boxes]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def state_from_wire(payload):
    data = dict(payload)
    data['contacts'] = tuple(data.get('contacts', ()))
    return G1State(**data)


class G1SessionOwner:
    """Serialize physics and reset. Cache commands, reject reuse with new content."""
    def __init__(self, factory, scene_id, cache_size=256):
        if type(cache_size) is not int or cache_size < 1:
            raise ValueError('Invalid receipt cache size')
        self.factory, self.scene_id, self.cache_size = factory, scene_id, cache_size
        self.session = factory()
        self.lock = Lock()
        self.receipts = OrderedDict()
        self.faulted = False

    def handle(self, request):
        with self.lock:
            if not isinstance(request, dict) or request.get('schema') != SCHEMA:
                raise ValueError('Invalid session schema')
            request_id = request.get('request_id')
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                raise ValueError('Invalid request ID')
            if request.get('scene_id') != self.scene_id:
                raise ValueError('Client and server scenes differ')
            op = request.get('operation')
            common = {'schema', 'request_id', 'scene_id', 'operation'}
            required = common if op == 'observe' else common | {'episode_id', 'step_id', 'sim_time_s', 'expires_at_unix_s'}
            if op == 'step':
                required |= {'action'}
            if op not in ('observe', 'step', 'reset') or set(request) != required:
                raise ValueError('Invalid session operation fields')
            canonical = json.dumps(request, sort_keys=True, allow_nan=False)
            if request_id in self.receipts:
                old, receipt = self.receipts[request_id]
                if old != canonical:
                    raise ValueError('Request ID reused with different content')
                cached_state = receipt.get('state')
                cached_episode = cached_state['episode_id'] if cached_state else request['episode_id']
                if cached_episode != self.session.observe().episode_id:
                    raise ValueError('Receipt belongs to an old episode')
                return receipt
            if op != 'observe':
                expires = request['expires_at_unix_s']
                if (isinstance(expires, bool) or not isinstance(expires, (float, int))
                        or not math.isfinite(expires) or expires <= time.time()):
                    raise ValueError('Expired command')
                state = self.session.observe()
                if type(request['step_id']) is not int or isinstance(request['sim_time_s'], bool):
                    raise ValueError('Invalid command time types')
                if ((request['episode_id'], request['step_id'], request['sim_time_s']) !=
                        (state.episode_id, state.step_id, state.sim_time_s)):
                    raise ValueError('Stale action or reset request')
            if op == 'step':
                if self.faulted:
                    raise RuntimeError('Session is faulted; explicit reset required')
                action = G1VelocityAction(**request['action'])
                try:
                    self.session.step(action, action.duration_s)
                    after = self.session.observe()
                    if (after.episode_id != state.episode_id or after.step_id != state.step_id+1
                            or abs(after.sim_time_s-state.sim_time_s-action.duration_s) > 1e-6):
                        raise ValueError('Invalid simulator acknowledgment')
                except Exception:
                    self.faulted = True
                    # Cache uncertain outcome, so duplicate delivery never re-executes it.
                    receipt = {'schema': SCHEMA, 'request_id': request_id, 'scene_id': self.scene_id,
                               'status': 'uncertain', 'faulted': True, 'state': None}
                    self._cache(request_id, canonical, receipt)
                    return receipt
            elif op == 'reset':
                try:
                    replacement = self.factory()
                    if replacement.observe().episode_id == self.session.observe().episode_id:
                        replacement.close()
                        raise ValueError('Reset must generate a new episode')
                    self.session.close()
                    self.session = replacement
                    self.faulted = False
                except Exception:
                    self.faulted = True
                    raise
            receipt = {'schema': SCHEMA, 'request_id': request_id, 'scene_id': self.scene_id,
                       'status': 'ok', 'faulted': self.faulted,
                       'state': asdict(self.session.observe())}
            if op != 'observe':
                self._cache(request_id, canonical, receipt)
            return receipt

    def _cache(self, request_id, canonical, receipt):
        self.receipts[request_id] = (canonical, receipt)
        while len(self.receipts) > self.cache_size:
            self.receipts.popitem(last=False)

    def close(self):
        with self.lock:
            self.session.close()
