"""Local read-only monitoring for robot-agent experiments."""

from wmal.monitor.contracts import RunDescriptor, SimFrame
from wmal.monitor.frame_buffer import LatestFrameBuffer

__all__ = ['LatestFrameBuffer', 'RunDescriptor', 'SimFrame']
