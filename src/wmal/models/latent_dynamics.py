"""Action-conditioned joint dynamics baseline for state-observation planning."""
from pathlib import Path
import json
import os
import tempfile

import numpy as np


class JointDynamics:
    """Bootstrap ridge regressors predict joint deltas; spread is not failure risk."""

    def __init__(self, joints, weights, version):
        self.joints = tuple(joints)
        self.weights = np.asarray(weights, dtype=float)
        self.version = version
        self.ensemble_size = len(self.weights)
        if not self.joints or len(set(self.joints)) != len(self.joints):
            raise ValueError('Invalid joint order')
        if self.weights.ndim != 3 or self.weights.shape[1:] != (2 + 2 * len(self.joints), len(self.joints)) or not np.isfinite(self.weights).all():
            raise ValueError('Invalid dynamics weights')

    def _features(self, state, targets, duration_s):
        if set(state) != set(self.joints) or not set(targets).issubset(self.joints) or duration_s <= 0:
            raise ValueError('State/action mismatch')
        values = np.asarray([1.0, duration_s, *[state[j] for j in self.joints],
                             *[targets.get(j, state[j]) for j in self.joints]], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite model input')
        return values

    def members(self, state, targets, duration_s):
        start = np.asarray([state[j] for j in self.joints], dtype=float)
        values = self._features(state, targets, duration_s) @ self.weights
        return [dict(zip(self.joints, row)) for row in start + values]

    def predict(self, state, targets, duration_s):
        candidates = self.members(state, targets, duration_s)
        return {joint: float(np.mean([candidate[joint] for candidate in candidates])) for joint in self.joints}

    def predict_member(self, index, state, targets, duration_s):
        if not 0 <= index < self.ensemble_size:
            raise IndexError('Unknown ensemble member')
        return self.members(state, targets, duration_s)[index]

    @classmethod
    def fit(cls, records, joints, *, members=5, ridge=1e-3, seed=0, version='joint-dynamics-v1'):
        if members < 1 or ridge <= 0 or not records:
            raise ValueError('Invalid training data or parameters')
        order = tuple(joints)
        provisional = cls(order, np.zeros((1, 2 + 2 * len(order), len(order))), version)
        features, deltas = [], []
        for row in records:
            before, after, target = row['before'], row['after'], row['target']
            if set(after) != set(order):
                raise ValueError('Next observation mismatch')
            features.append(provisional._features(before, target, row['duration_s']))
            delta = np.asarray([after[j] - before[j] for j in order], dtype=float)
            if not np.isfinite(delta).all():
                raise ValueError('Nonfinite training target')
            deltas.append(delta)
        x, y = np.asarray(features), np.asarray(deltas)
        generator, coefficients = np.random.default_rng(seed), []
        for _ in range(members):
            sample = generator.integers(0, len(x), len(x)) if members > 1 else np.arange(len(x))
            xs, ys = x[sample], y[sample]
            penalty = np.eye(x.shape[1]) * ridge
            penalty[0, 0] = 0
            coefficients.append(np.linalg.solve(xs.T @ xs + penalty, xs.T @ ys))
        return cls(order, coefficients, version)

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({'schema_version': 1, 'joints': self.joints,
                              'weights': self.weights.tolist(), 'version': self.version}, allow_nan=False)
        with tempfile.NamedTemporaryFile('w', dir=destination.parent, delete=False) as handle:
            handle.write(payload)
            temporary = handle.name
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get('schema_version') != 1:
            raise ValueError('Unknown checkpoint schema')
        return cls(data['joints'], data['weights'], data['version'])
