"""Bootstrap ridge dynamics for policy-conditioned G1 base motion.

This lightweight learned baseline implements the same plugin contract as a
larger pretrained world model. It does not predict pixels or joint trajectories.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from wmal.logging.manifest import atomic_json
from wmal.locomotion.contracts import G1_ACTION_SCHEMA, G1_STATE_SCHEMA, G1Prediction


def wrap(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def features(state, action):
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    return np.array([1, c * state.vx + s * state.vy,
                     -s * state.vx + c * state.vy, state.yaw_rate,
                     state.roll, state.pitch, state.pelvis_height,
                     action.vx, action.vy, action.yaw_rate], dtype=float)


def target(before, after, duration):
    c, s = math.cos(before.yaw), math.sin(before.yaw)
    dx, dy = after.x - before.x, after.y - before.y
    return np.array([(c * dx + s * dy) / duration,
                     (-s * dx + c * dy) / duration,
                     wrap(after.yaw - before.yaw) / duration,
                     after.roll, after.pitch, after.pelvis_height])


class LearnedG1Dynamics:
    state_schema, action_schema = G1_STATE_SCHEMA, G1_ACTION_SCHEMA

    def __init__(self, weights, duration_s, version):
        self.weights = np.asarray(weights, dtype=float)
        if (self.weights.ndim != 3 or self.weights.shape[1:] != (10, 6)
                or len(self.weights) < 2 or not np.isfinite(self.weights).all()):
            raise ValueError('Expected finite ensemble weights [members, 10, 6]')
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('Invalid trained action duration')
        self.duration_s, self.version = duration_s, version

    def predict(self, state, action, duration_s):
        if abs(duration_s - self.duration_s) > 1e-9:
            raise ValueError('Action duration differs from training duration')
        outputs = features(state, action) @ self.weights
        mean, variance = outputs.mean(axis=0), outputs.var(axis=0)
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        vx, vy = c * mean[0] - s * mean[1], s * mean[0] + c * mean[1]
        predicted = replace(state, step_id=state.step_id + 1,
                            sim_time_s=state.sim_time_s + duration_s,
                            x=float(state.x + vx * duration_s),
                            y=float(state.y + vy * duration_s),
                            yaw=wrap(state.yaw + float(mean[2]) * duration_s),
                            vx=float(vx), vy=float(vy), yaw_rate=float(mean[2]),
                            roll=float(mean[3]), pitch=float(mean[4]),
                            z=max(0., float(mean[5])), pelvis_height=max(0., float(mean[5])))
        return G1Prediction(predicted, {'position_variance': float(sum(variance[:2]) * duration_s**2),
                                        'posture_variance': float(sum(variance[3:]))},
                            uncertainty_kind='ensemble_spread')

    def save(self, path):
        atomic_json(path, {'schema': 'wmal.g1.ridge.v1', 'weights': self.weights.tolist(),
                           'duration_s': self.duration_s, 'version': self.version})

    @classmethod
    def fit(cls, transitions, *, members=5, seed=0, ridge=0.01, prior=None,
            normalize=False, episode_bootstrap=False):
        if len(transitions) < 10 or members < 2 or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError('Need >=10 transitions, >=2 members and positive ridge')
        duration = transitions[0][1].duration_s
        if any(abs(a.duration_s - duration) > 1e-9 for _, a, _ in transitions):
            raise ValueError('Mixed action durations are not supported')
        x = np.stack([features(s, a) for s, a, _ in transitions])
        y = np.stack([target(s, n, duration) for s, _, n in transitions])
        if prior is not None and abs(prior.duration_s - duration) > 1e-9:
            raise ValueError('Prior duration mismatch')
        center = np.zeros((10, 6)) if prior is None else prior.weights.mean(axis=0)
        # Solve in normalized coordinates, then store raw-coordinate coefficients
        # so existing checkpoints and inference plugins remain compatible.
        transform = np.eye(10)
        if normalize:
            scale = x[:, 1:].std(axis=0)
            scale[scale < 1e-6] = 1.
            transform[0, 1:] = -x[:, 1:].mean(axis=0) / scale
            transform[np.arange(1, 10), np.arange(1, 10)] = 1 / scale
        normalized_x = x @ transform
        normalized_center = np.linalg.solve(transform, center)
        groups = {}
        for index, (before, _, _) in enumerate(transitions):
            groups.setdefault(before.episode_id, []).append(index)
        group_values = list(groups.values())
        rng, weights = np.random.default_rng(seed), []
        for _ in range(members):
            if episode_bootstrap:
                indices = np.concatenate([group_values[i] for i in rng.integers(0, len(groups), len(groups))])
            else:
                indices = rng.integers(0, len(x), len(x))
            bx, by = normalized_x[indices], y[indices]
            weights.append(transform @ np.linalg.solve(bx.T @ bx + ridge * np.eye(10),
                                                       bx.T @ by + ridge * normalized_center))
        version = 'g1-ridge-' + hashlib.sha256(np.array(weights).tobytes()).hexdigest()[:12]
        return cls(weights, duration, version)


def load(config):
    payload = json.loads(Path(config['checkpoint']).read_text())
    if payload['schema'] != 'wmal.g1.ridge.v1':
        raise ValueError('Unknown G1 checkpoint schema')
    return LearnedG1Dynamics(payload['weights'], payload['duration_s'], payload['version'])
