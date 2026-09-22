"""Robot-side replay and observation binding guard; caller holds owner lock."""


class EpisodeGuard:
    def __init__(self):
        self._episode = None
        self._seen = set()

    def accept(self, command, observation):
        if command.robot_id != observation.robot_id or command.episode_id != observation.episode_id:
            raise ValueError('Robot or episode mismatch')
        if command.expected_step != observation.step_id:
            raise ValueError('Stale command step')
        if self._episode != observation.episode_id:
            self._episode = observation.episode_id
            self._seen.clear()
        if command.command_id in self._seen:
            raise ValueError('Duplicate physical command')
        if len(self._seen) >= 100000:
            raise ValueError('Episode command budget exhausted')
        self._seen.add(command.command_id)
