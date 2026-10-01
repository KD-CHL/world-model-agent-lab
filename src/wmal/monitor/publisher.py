"""Bounded telemetry publication; no simulator references or network I/O."""
from datetime import datetime, timezone
import math
import time
from threading import Lock

from wmal.monitor.contracts import SimFrame


class FramePublisher:
    def __init__(self, buffer, run_id, camera, max_fps=5, width=320, height=240, clock=time.monotonic):
        if not isinstance(run_id, str) or not run_id or not isinstance(camera, str) or not camera:
            raise ValueError('Publisher requires run and camera identifiers')
        if not math.isfinite(max_fps) or not .1 <= max_fps <= 30:
            raise ValueError('max_fps must be between .1 and 30')
        if type(width) is not int or type(height) is not int or not 1 <= width <= 2048 or not 1 <= height <= 2048:
            raise ValueError('Invalid monitor frame dimensions')
        self.buffer, self.run_id, self.camera = buffer, run_id, camera
        self.width, self.height, self.max_fps = width, height, max_fps
        self._clock, self._last = clock, float('-inf')
        self._lock = Lock()
        self._closed, self._errors, self._published, self._last_error = False, 0, 0, None

    def ready(self, force=False):
        with self._lock:
            return (not self._closed and self._errors < 3
                    and (force or self._clock() - self._last >= 1 / self.max_fps))

    def publish(self, rgb, *, episode_id, step_id, sim_time_s, force=False):
        if not self.ready(force):
            return False
        try:
            frame = SimFrame(self.run_id, episode_id, step_id, sim_time_s,
                             datetime.now(timezone.utc).isoformat(), self.camera, rgb)
            self.buffer.publish(frame)
            with self._lock:
                self._last = self._clock()
                self._published += 1
            return True
        except Exception as exc:
            self.capture_error(exc)
            return False

    def capture_error(self, error):
        with self._lock:
            self._errors += 1
            self._last_error = type(error).__name__

    def status(self):
        with self._lock:
            return {'closed': self._closed, 'published_frames': self._published,
                    'error_count': self._errors, 'last_error': self._last_error,
                    'max_fps': self.max_fps, 'render_disabled': self._errors >= 3}

    def close(self):
        with self._lock:
            self._closed = True
