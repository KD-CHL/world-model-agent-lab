"""Bounded action-aligned REAL history, with atomic execution-trace validation.

This is not retrieval memory. No predicted observations, learned updates or
controller-generated nominal states belong here. A new Agent starts a fresh
context at the task's initial real observation, even within a persistent viewer.
"""
import numpy as np


def _copy_observation(observation):
    # Runtime import avoids a dependency cycle with the Agent's public DTOs.
    from wmal.agents.predictive_skill_agent import VisualObservation
    return VisualObservation(observation.episode_id, observation.step_id, observation.rgb, observation.state)


class ObservationHistory:
    def __init__(self, context_steps, action_dim):
        if type(context_steps) is not int or context_steps < 1 or type(action_dim) is not int or action_dim < 1:
            raise ValueError('Temporal history requires positive integer context/action dimensions')
        self.context_steps, self.action_dim = context_steps, action_dim
        self._observations = ()
        self._actions = np.empty((0, action_dim))

    def observe_current(self, observation):
        current = _copy_observation(observation)
        if self._observations:
            previous = self._observations[-1]
            if (current.episode_id != previous.episode_id or current.step_id != previous.step_id
                    or current.state.shape != previous.state.shape or current.rgb.shape != previous.rgb.shape):
                raise ValueError('Cannot skip history or cross an episode during reobservation')
            self._observations = self._observations[:-1] + (current,)
        else:
            self._observations = (current,)

    def inputs(self):
        if not self._observations:
            raise ValueError('Real history has not been initialized')
        # Return independent arrays; candidate backends cannot corrupt future branches.
        return (np.stack([v.rgb for v in self._observations]),
                np.stack([v.state for v in self._observations]), self._actions.copy())

    def describe(self):
        return dict(conditioning='bounded_real_history', latent_memory='fixed_context_recomputed',
                    context_steps_limit=self.context_steps, context_steps_used=len(self._actions),
                    history_step_ids=[v.step_id for v in self._observations], context_valid=True)

    def validate_execution(self, observation, actions, observations=None):
        """Snapshot the complete trace before changing any Agent or history state."""
        if not self._observations:
            raise ValueError('Uninitialized execution history')
        previous = self._observations[-1]
        actions = np.asarray(actions)
        if (actions.ndim != 2 or actions.shape[1] != self.action_dim or not len(actions)
                or not np.isfinite(actions).all()):
            raise ValueError('Invalid committed actions for execution history')
        if observations is None:
            if len(actions) != 1:
                raise ValueError('Temporal context requires every intermediate real execution observation')
            observations = (observation,)
        trace = tuple(_copy_observation(v) for v in observations)
        final = _copy_observation(observation)
        if len(trace) != len(actions):
            raise ValueError('Execution trace must contain one real observation per action')
        for i, value in enumerate(trace):
            if (value.episode_id != previous.episode_id or value.step_id != previous.step_id+i+1
                    or value.state.shape != previous.state.shape or value.rgb.shape != previous.rgb.shape):
                raise ValueError('Execution history has missing, duplicate or cross-episode feedback')
        last = trace[-1]
        if (final.episode_id != last.episode_id or final.step_id != last.step_id
                or not np.array_equal(final.state, last.state) or not np.array_equal(final.rgb, last.rgb)):
            raise ValueError('Final execution observation differs from the final trace entry')
        return trace

    def commit_execution(self, trace, actions):
        """Accept only the internally snapshotted trace validated before feedback computation."""
        values = self._observations + trace
        commands = np.concatenate([self._actions, actions])
        self._observations = values[-self.context_steps-1:]
        self._actions = commands[-self.context_steps:].copy()
