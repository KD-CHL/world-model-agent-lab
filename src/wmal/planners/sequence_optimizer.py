"""Bounded cross-entropy search over action sequences.

This module is an original implementation using the project's model contract.
It does not import code from external planning repositories.
"""
import numpy as np


class SequenceOptimizer:
    """Optimize continuous action sequences against a caller supplied rollout cost."""

    def __init__(self, *, candidates=128, elites=12, iterations=4, smoothing=0.2,
                 initial_std=0.35, seed=0):
        if candidates < 2 or not 1 <= elites < candidates or iterations < 1:
            raise ValueError('Invalid search budget')
        if not 0 <= smoothing < 1 or initial_std <= 0:
            raise ValueError('Invalid search distribution')
        self.candidates = candidates
        self.elites = elites
        self.iterations = iterations
        self.smoothing = smoothing
        self.initial_std = initial_std
        self.rng = np.random.default_rng(seed)

    def optimize(self, score, *, horizon, width, lower, upper, initial=None):
        """Return the best sequence, scalar cost, and per-iteration diagnostics."""
        if horizon < 1 or width < 1:
            raise ValueError('Sequence shape must be positive')
        low = np.broadcast_to(np.asarray(lower, dtype=float), (horizon, width))
        high = np.broadcast_to(np.asarray(upper, dtype=float), (horizon, width))
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
            raise ValueError('Invalid sequence bounds')
        center = ((low + high) * 0.5 if initial is None
                  else np.clip(np.asarray(initial, dtype=float), low, high))
        if center.shape != (horizon, width):
            raise ValueError('Initial sequence shape mismatch')
        spread = np.maximum((high - low) * self.initial_std, 1e-6)
        best_sequence, best_cost, history = None, float('inf'), []
        for _ in range(self.iterations):
            population = np.clip(self.rng.normal(center, spread, (self.candidates, horizon, width)), low, high)
            costs = np.asarray([score(sequence) for sequence in population], dtype=float)
            costs[~np.isfinite(costs)] = np.finfo(float).max
            order = np.argsort(costs)[:self.elites]
            elite = population[order]
            elite_costs = costs[order]
            if float(elite_costs[0]) < best_cost:
                best_cost = float(elite_costs[0])
                best_sequence = elite[0].copy()
            next_center = elite.mean(axis=0)
            next_spread = np.maximum(elite.std(axis=0), (high - low) * 0.015)
            center = self.smoothing * center + (1.0 - self.smoothing) * next_center
            spread = self.smoothing * spread + (1.0 - self.smoothing) * next_spread
            history.append({'best_cost': best_cost, 'elite_cost_mean': float(elite_costs.mean()),
                            'elite_cost_std': float(elite_costs.std())})
        if best_sequence is None or best_cost == float(np.finfo(float).max):
            raise ValueError('No finite candidate action sequence')
        return best_sequence, best_cost, history
