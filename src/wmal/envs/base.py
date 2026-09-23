"""Robot simulator interface consumed by data collection and evaluation."""
from typing import Protocol


class RobotEnvironment(Protocol):
    profile: object

    def reset(self):
        """Reset one episode and return the first observation."""
        ...

    def step(self, command):
        """Apply one bounded command and return (observation, execution metadata)."""
        ...

    def render_rgb(self, *, camera=None, width=320, height=240):
        ...
