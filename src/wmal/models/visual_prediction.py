"""Adapter contract for feature-prediction models and goal-image planning.

The project owns this interface and planner. A backend may wrap a local model
or a separately installed inference service; no third-party model code is vendored.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from wmal.planners.sequence_optimizer import SequenceOptimizer


class FeaturePredictor(Protocol):
    version: str

    def encode(self, rgb):
        """Encode an HWC uint8 RGB image into a one-dimensional feature vector."""
        ...

    def predict_features(self, start_features, action_sequence):
        """Return [horizon, feature_dim] or [ensemble, horizon, feature_dim]."""
        ...


@dataclass
class VisualPlan:
    model_version: str
    action_sequence: list
    predicted_features: list
    goal_distance: float
    feature_spread: list


class GoalImagePlanner:
    """Plan action sequences by predicted feature distance to a goal image."""

    def __init__(self, predictor, *, horizon, lower, upper, candidates=96,
                 elites=10, iterations=4, seed=0, terminal_weight=2.0,
                 ensemble_penalty=0.05):
        if (horizon < 1 or terminal_weight <= 0 or ensemble_penalty < 0
                or not getattr(predictor, 'version', None)):
            raise ValueError('Invalid visual planner configuration')
        self.predictor = predictor
        self.horizon = horizon
        self.lower, self.upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if self.lower.ndim != 1 or self.upper.shape != self.lower.shape or np.any(self.lower >= self.upper):
            raise ValueError('Action bounds must be matching one-dimensional vectors')
        self.terminal_weight = terminal_weight
        self.ensemble_penalty = ensemble_penalty
        self.optimizer = SequenceOptimizer(candidates=candidates, elites=min(elites, candidates - 1),
                                           iterations=iterations, seed=seed)

    @staticmethod
    def _image(image):
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3 or not array.size:
            raise ValueError('Expected nonempty HWC uint8 RGB image')
        return array

    def plan(self, current_rgb, goal_rgb):
        current = np.asarray(self.predictor.encode(self._image(current_rgb)), dtype=float).reshape(-1)
        goal = np.asarray(self.predictor.encode(self._image(goal_rgb)), dtype=float).reshape(-1)
        if current.shape != goal.shape or not np.isfinite(current).all() or not np.isfinite(goal).all():
            raise ValueError('Predictor returned invalid image features')

        def predict(sequence):
            values = np.asarray(self.predictor.predict_features(current, sequence), dtype=float)
            if values.ndim == 2:
                values = values[None, ...]
            if values.ndim != 3 or values.shape[1:] != (self.horizon, current.size) or not np.isfinite(values).all():
                raise ValueError('Feature model returned invalid rollout shape')
            return values

        def cost(sequence):
            features = predict(sequence)
            distances = np.linalg.norm(features - goal[None, None, :], axis=-1)
            temporal = np.arange(1, self.horizon + 1, dtype=float) / self.horizon
            member_costs = (distances * temporal[None, :]).sum(axis=1)
            member_costs += self.terminal_weight * distances[:, -1]
            return float(member_costs.mean() + self.ensemble_penalty * member_costs.std())

        initial = np.tile((self.lower + self.upper) * 0.5, (self.horizon, 1))
        sequence, score, _ = self.optimizer.optimize(
            cost, horizon=self.horizon, width=len(self.lower), lower=self.lower,
            upper=self.upper, initial=initial)
        rollouts = predict(sequence)
        return VisualPlan(
            self.predictor.version, sequence.tolist(), rollouts.mean(axis=0).tolist(), score,
            rollouts.std(axis=0)[-1].tolist() if len(rollouts) > 1 else [])
