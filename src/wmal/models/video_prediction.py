"""Protocol boundary for external visual world-model inference plugins."""
from typing import Protocol, Iterable


class VideoPredictionProvider(Protocol):
    version: str

    def predict_video(self, observation_history, action_sequence):
        """Return a uint8 RGB sequence shaped [time, height, width, 3]."""
        ...

    def iter_evaluation_samples(self, dataset_path: str) -> Iterable[dict]:
        """Yield sample_id, observation_history, action_sequence, target_frames."""
        ...
