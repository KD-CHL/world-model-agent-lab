"""Bounded HTTP adapter for a language-conditioned action-sequence service.

The service proposes actions; it does not predict future robot states.
"""
from dataclasses import dataclass
import json
import math
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError('Inference redirect refused')


def _finite_vector(values, width):
    if not isinstance(values, list) or len(values) != width:
        raise ValueError('Invalid vector width')
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError('Nonfinite vector')


@dataclass(frozen=True)
class ActionQuery:
    instruction: str
    states: list
    images_chw: list
    zero_actions: list
    state_dim: int
    action_dim: int

    def validate(self):
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError('Missing language instruction')
        if not 0 < len(self.states) == len(self.images_chw) <= 8:
            raise ValueError('Invalid observation history')
        if not 0 < self.state_dim <= 128 or not 0 < self.action_dim <= 128:
            raise ValueError('Invalid model dimensions')
        for state in self.states:
            _finite_vector(state, self.state_dim)
        for action in self.zero_actions:
            _finite_vector(action, self.action_dim)
            if any(value != 0 for value in action):
                raise ValueError('Action conditioning must be zero')
        if not self.zero_actions or len(self.zero_actions) > 64:
            raise ValueError('Invalid action conditioning horizon')
        shape = None
        for image in self.images_chw:
            if not isinstance(image, list) or len(image) != 3:
                raise ValueError('Expected RGB CHW image')
            current = tuple((len(channel), len(channel[0]) if channel else 0) for channel in image if isinstance(channel, list))
            if len(current) != 3 or current[0] != current[1] or current[1] != current[2] or current[0][0] == 0 or current[0][1] == 0 or current[0][0] > 2048 or current[0][1] > 2048:
                raise ValueError('Invalid image shape')
            if shape is not None and current != shape:
                raise ValueError('Image history shape changed')
            shape = current
            width = current[0][1]
            for channel in image:
                if any(not isinstance(row, list) or len(row) != width or any(type(px) is not int or not 0 <= px <= 255 for px in row) for row in channel):
                    raise ValueError('Expected uint8 RGB pixels')


@dataclass(frozen=True)
class ActionChunk:
    model_version: str
    actions: list


class ActionServiceClient:
    def __init__(self, base_url, model_version, timeout_s=10, max_response_bytes=2_000_000, opener=None):
        parsed = urlparse(base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
            raise ValueError('Invalid inference service URL')
        if parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('Remote inference requires HTTPS or an SSH tunnel')
        if not isinstance(model_version, str) or not model_version or timeout_s <= 0 or max_response_bytes < 1024:
            raise ValueError('Invalid inference configuration')
        self.url = base_url.rstrip('/') + '/predict_action'
        self.version = model_version
        self.timeout_s = timeout_s
        self.max_response_bytes = max_response_bytes
        self.opener = opener or build_opener(_NoRedirect())

    def propose(self, query):
        query.validate()
        payload = {
            'language_instruction': query.instruction,
            'observation.state': query.states,
            'observation.images.top': query.images_chw,
            'action': query.zero_actions,
        }
        request = Request(self.url, data=json.dumps(payload, allow_nan=False).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with self.opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise ValueError('Inference response too large')
            result = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError('Inference service unavailable or malformed') from None
        if not isinstance(result, dict) or result.get('result') != 'ok':
            raise RuntimeError('Inference service rejected request')
        actions = result.get('action')
        if not isinstance(actions, list) or not 0 < len(actions) <= 64:
            raise ValueError('Invalid action horizon')
        for action in actions:
            _finite_vector(action, query.action_dim)
        return ActionChunk(self.version, actions)
