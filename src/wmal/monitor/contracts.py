"""Versionable, validated telemetry values shared by simulator and monitor."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RunDescriptor:
    run_id: str
    run_path: str
    run_kind: str | None = None
    started_at_utc: str | None = None

    def __post_init__(self):
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError('run_id must be a nonempty string')
        if not isinstance(self.run_path, str) or not self.run_path.strip():
            raise ValueError('run_path must be a nonempty root-relative path')
        if self.run_kind is not None and not isinstance(self.run_kind, str):
            raise TypeError('run_kind must be a string or None')
        if self.started_at_utc is not None and not isinstance(self.started_at_utc, str):
            raise TypeError('started_at_utc must be a string or None')


@dataclass(frozen=True)
class SimFrame:
    run_id: str
    episode_id: str
    step_id: int
    sim_time_s: float
    captured_at_utc: str
    camera: str
    rgb: np.ndarray

    def __post_init__(self):
        for name in ('run_id', 'episode_id', 'captured_at_utc', 'camera'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a nonempty string')
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError('step_id must be a nonnegative integer')
        if isinstance(self.sim_time_s, bool) or not math.isfinite(self.sim_time_s) or self.sim_time_s < 0:
            raise ValueError('sim_time_s must be finite and nonnegative')
        image = np.asarray(self.rgb)
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
                or not 1 <= image.shape[0] <= 4096 or not 1 <= image.shape[1] <= 4096):
            raise ValueError('rgb must be a nonempty HxWx3 uint8 image no larger than 4096x4096')
        image = image.copy()
        image.flags.writeable = False
        object.__setattr__(self, 'rgb', image)
