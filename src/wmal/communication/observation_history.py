"""Bounded, step-exact RGB/state history for vision-conditioned inference."""
from dataclasses import dataclass
from time import monotonic

from wmal.communication.contracts import Observation


@dataclass(frozen=True)
class CameraFrame:
    observation: Observation
    camera: str
    width: int
    height: int
    rgb: bytes
    received_at: float


def decode_rgb(encoding, width, height, step_bytes, data):
    """Convert packed/row-padded rgb8 or bgr8 sensor data to contiguous RGB bytes."""
    if encoding not in ("rgb8", "bgr8"):
        raise ValueError("Camera encoding must be rgb8 or bgr8")
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError("Invalid camera dimensions")
    row_width = width * 3
    if type(step_bytes) is not int or step_bytes < row_width or len(data) != step_bytes * height:
        raise ValueError("Camera payload does not match row stride and dimensions")
    packed = bytearray(row_width * height)
    for row in range(height):
        source_start, target_start = row * step_bytes, row * row_width
        source = data[source_start:source_start + row_width]
        if encoding == "rgb8":
            packed[target_start:target_start + row_width] = source
        else:
            for col in range(width):
                b, g, r = source[col * 3:col * 3 + 3]
                packed[target_start + col * 3:target_start + col * 3 + 3] = bytes((r, g, b))
    return bytes(packed)


def rgb_to_chw(rgb, width, height):
    if len(rgb) != width * height * 3:
        raise ValueError("RGB frame size mismatch")
    channels = [bytearray(width * height) for _ in range(3)]
    for pixel in range(width * height):
        offset = pixel * 3
        for channel in range(3):
            channels[channel][pixel] = rgb[offset + channel]
    return [[list(channel[row * width:(row + 1) * width])
             for row in range(height)] for channel in channels]


class ObservationHistory:
    """Store exact same-step state/image pairs; reset and stale frames cannot leak."""

    def __init__(self, capacity=8, max_skew_s=1e-6, camera=None):
        if type(capacity) is not int or capacity < 1 or max_skew_s < 0:
            raise ValueError("Invalid observation history settings")
        self.capacity, self.max_skew_s, self.camera = capacity, float(max_skew_s), camera
        self._items = []

    def clear(self):
        self._items.clear()

    def append_pair(self, observation, *, camera, width, height, rgb, received_at=None):
        if not isinstance(observation, Observation):
            raise ValueError("Expected a validated robot observation")
        if not isinstance(camera, str) or not camera or (self.camera is not None and camera != self.camera):
            raise ValueError("Unexpected camera stream")
        if type(width) is not int or type(height) is not int or width < 1 or height < 1:
            raise ValueError("Invalid image dimensions")
        if not isinstance(rgb, (bytes, bytearray)) or len(rgb) != width * height * 3:
            raise ValueError("Invalid RGB payload")
        if self._items:
            previous = self._items[-1].observation
            if observation.episode_id != previous.episode_id:
                self.clear()
            elif observation.step_id <= previous.step_id or observation.sim_time_s < previous.sim_time_s:
                raise ValueError("Out-of-order observation/image pair")
        item = CameraFrame(observation, camera, width, height, bytes(rgb),
                           monotonic() if received_at is None else float(received_at))
        self._items.append(item)
        del self._items[:-self.capacity]
        return item

    def matched_history(self, count, *, max_age_s=None, now=None):
        if type(count) is not int or count < 1 or len(self._items) < count:
            raise TimeoutError("Insufficient synchronized camera/state history")
        history = self._items[-count:]
        if any((a.observation.episode_id != b.observation.episode_id
                or b.observation.step_id <= a.observation.step_id
                or b.observation.sim_time_s < a.observation.sim_time_s)
               for a, b in zip(history, history[1:])):
            raise ValueError("Camera/state history is not synchronized")
        if max_age_s is not None and (now if now is not None else monotonic()) - history[-1].received_at > max_age_s:
            raise TimeoutError("Synchronized camera/state history is stale")
        return list(history)
