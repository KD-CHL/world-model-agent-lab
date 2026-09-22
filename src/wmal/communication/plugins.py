"""Load explicit local configuration plugins; never take import paths from the LLM."""
from importlib import import_module


def load_factory(spec, **kwargs):
    if not isinstance(spec, str) or ':' not in spec:
        raise ValueError('Plugin must be module:factory')
    module, name = spec.split(':', 1)
    return getattr(import_module(module), name)(**kwargs)
