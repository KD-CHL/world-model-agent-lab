"""Explicit adapter for local or remote action-conditioned G1 world models."""
import importlib
import math

from wmal.locomotion.contracts import (G1_ACTION_SCHEMA, G1_STATE_SCHEMA,
                                       G1Prediction, G1State, G1VelocityAction)


class WorldModelAdapter:
    def __init__(self, model):
        if not callable(getattr(model, 'predict', None)):
            raise ValueError('World model must implement predict(state, action, duration_s)')
        version = getattr(model, 'version', None)
        if not isinstance(version, str) or not version.strip():
            raise ValueError('World model must expose a nonempty version')
        if getattr(model, 'state_schema', None) != G1_STATE_SCHEMA:
            raise ValueError(f'World model state_schema must be {G1_STATE_SCHEMA}')
        if getattr(model, 'action_schema', None) != G1_ACTION_SCHEMA:
            raise ValueError(f'World model action_schema must be {G1_ACTION_SCHEMA}')
        self.model, self.version = model, version

    def predict(self, state, action, duration_s=None):
        if not isinstance(state, G1State) or not isinstance(action, G1VelocityAction):
            raise ValueError('World model received an incompatible G1 state or action')
        duration = action.duration_s if duration_s is None else float(duration_s)
        if not math.isfinite(duration) or abs(duration - action.duration_s) > 1e-9:
            raise ValueError('World-model duration must match the action duration')
        try:
            prediction = self.model.predict(state, action, duration)
        except Exception as exc:
            raise RuntimeError(f'World-model inference failed ({type(exc).__name__})') from None
        if isinstance(prediction, G1State):
            prediction = G1Prediction(prediction)
        if not isinstance(prediction, G1Prediction):
            raise ValueError('World model must return G1Prediction or G1State')
        result = prediction.state
        if result.episode_id != state.episode_id or result.step_id <= state.step_id:
            raise ValueError('World-model prediction has stale episode or step provenance')
        if result.sim_time_s <= state.sim_time_s:
            raise ValueError('World-model prediction time must advance')
        if bool(state.joint_positions) != bool(result.joint_positions):
            raise ValueError('World-model prediction state schema mismatch')
        if set(state.joint_positions) != set(result.joint_positions):
            raise ValueError('World-model prediction state schema mismatch')
        return prediction


def load_world_model(factory_path, config):
    """Load `module:function` factory with a config mapping; never use a fallback."""
    if not isinstance(factory_path, str) or ':' not in factory_path:
        raise ValueError('World-model factory must use module:function syntax')
    module_name, function_name = factory_path.split(':', 1)
    if not module_name or not function_name:
        raise ValueError('World-model factory must use module:function syntax')
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise ValueError('Configured world-model factory is not callable')
    try:
        model = factory(dict(config))
    except Exception as exc:
        raise RuntimeError(f'World-model factory failed ({type(exc).__name__})') from None
    return WorldModelAdapter(model)
