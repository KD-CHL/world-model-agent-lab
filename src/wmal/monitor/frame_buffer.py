"""Thread-safe bounded frame history; slow clients cannot backpressure simulation."""
from collections import deque
from threading import Lock

from wmal.monitor.contracts import SimFrame


class LatestFrameBuffer:
    def __init__(self, max_history=1):
        if type(max_history) is not int or max_history < 1:
            raise ValueError('max_history must be a positive integer')
        self._frames = deque(maxlen=max_history)
        self._lock = Lock()
        self._dropped_frames = 0

    def publish(self, frame):
        if not isinstance(frame, SimFrame):
            raise TypeError('frame must be a SimFrame')
        with self._lock:
            if len(self._frames) == self._frames.maxlen:
                self._dropped_frames += 1
            self._frames.append(frame)

    def latest(self, run_id=None):
        with self._lock:
            for frame in reversed(self._frames):
                if run_id is None or frame.run_id == run_id:
                    return frame
        return None

    def history(self, run_id, episode_id=None):
        with self._lock:
            return [frame for frame in self._frames
                    if frame.run_id == run_id
                    and (episode_id is None or frame.episode_id == episode_id)]

    @property
    def dropped_frames(self):
        with self._lock:
            return self._dropped_frames
