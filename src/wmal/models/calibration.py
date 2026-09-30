"""Empirical episode-level residual alarms; not failure probabilities."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ResidualCalibration:
    model_version: str
    duration_s: float
    threshold_m: float
    quantile: float
    dataset_id: str
    episode_ids: tuple
    schema: str = 'wmal.residual_calibration.v1'

    def __post_init__(self):
        if (self.schema != 'wmal.residual_calibration.v1' or not self.model_version
                or not self.dataset_id or not self.episode_ids
                or any(not isinstance(e, str) or not e for e in self.episode_ids)
                or len(set(self.episode_ids)) != len(self.episode_ids)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in (self.duration_s, self.threshold_m, self.quantile))
                or self.duration_s <= 0 or self.threshold_m <= 0 or not 0 < self.quantile <= 1):
            raise ValueError('Invalid residual calibration')

    def validate_for(self, version, duration_s):
        if version != self.model_version or abs(duration_s-self.duration_s) > 1e-9:
            raise ValueError('Calibration model version or action duration mismatch')

    def save(self, path):
        from wmal.logging.manifest import atomic_json
        atomic_json(path, asdict(self))

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        payload['episode_ids'] = tuple(payload['episode_ids'])
        return cls(**payload)


def fit_residual_calibration(rows, *, model_version, duration_s, dataset_id,
                             quantile=.95, excluded_episode_ids=()):
    """Rows explicitly marked validation; reject train/test episode overlap."""
    if not 0 < quantile <= 1:
        raise ValueError('Invalid quantile')
    maxima = {}
    excluded = set(excluded_episode_ids)
    for row in rows:
        episode = row['episode_id']
        value = row['position_error_m']
        if (row['split'] != 'validation' or episode in excluded or not isinstance(episode, str)
                or not episode or row['model_version'] != model_version
                or abs(row['duration_s']-duration_s) > 1e-9
                or isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value < 0):
            raise ValueError('Invalid or overlapping calibration record')
        maxima[episode] = max(maxima.get(episode, 0.), value)
    if len(maxima) < 2:
        raise ValueError('Need at least two independent validation episodes')
    ordered = sorted(maxima.values())
    threshold = max(1e-9, ordered[math.ceil(quantile*len(ordered))-1])
    return ResidualCalibration(model_version, duration_s, threshold, quantile,
                               dataset_id, tuple(sorted(maxima)))
