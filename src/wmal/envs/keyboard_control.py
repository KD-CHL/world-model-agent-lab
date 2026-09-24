"""Keyboard-to-joint target bridge for the interactive MuJoCo viewer."""
from queue import SimpleQueue, Empty


class KeyboardJointControl:
    """Queue viewer key events and slew bounded joint targets at physics rate."""

    _KEY_ACTIONS = {
        ord('N'): 'next',
        ord('P'): 'previous',
        ord('['): 'decrease',
        ord(']'): 'increase',
        ord('0'): 'hold',
    }

    def __init__(self, backend, *, initial_joint=None, step_rad=0.1):
        if step_rad <= 0:
            raise ValueError('Keyboard joint step must be positive')
        self.backend = backend
        self.joint_names = tuple(backend.profile.joint_limits)
        if initial_joint is None:
            initial_joint = self.joint_names[0]
        if initial_joint not in self.joint_names:
            raise ValueError('Unknown initial keyboard joint: ' + initial_joint)
        self.selected_index = self.joint_names.index(initial_joint)
        self.step_rad = float(step_rad)
        self.requested_targets = dict(backend.targets)
        self._events = SimpleQueue()

    @property
    def selected_joint(self):
        return self.joint_names[self.selected_index]

    def on_key(self, keycode):
        """GLFW callback: enqueue only; physics state is changed by update()."""
        action = self._KEY_ACTIONS.get(keycode)
        if action is not None:
            self._events.put(action)

    def update(self):
        """Apply queued input and rate-limit all target changes for one physics step."""
        messages = []
        while True:
            try:
                action = self._events.get_nowait()
            except Empty:
                break
            if action == 'next':
                self.selected_index = (self.selected_index + 1) % len(self.joint_names)
                name = self.selected_joint
                messages.append(f'Selected {name}; target={self.requested_targets[name]:.3f} rad')
                continue
            if action == 'previous':
                self.selected_index = (self.selected_index - 1) % len(self.joint_names)
                name = self.selected_joint
                messages.append(f'Selected {name}; target={self.requested_targets[name]:.3f} rad')
                continue

            name = self.selected_joint
            low, high = self.backend.profile.joint_limits[name]
            if action == 'decrease':
                self.requested_targets[name] = max(low, self.requested_targets[name] - self.step_rad)
            elif action == 'increase':
                self.requested_targets[name] = min(high, self.requested_targets[name] + self.step_rad)
            else:
                self.requested_targets[name] = self.backend.observe().joints[name]
            messages.append(f'{name} target={self.requested_targets[name]:.3f} rad')

        max_delta = self.backend.profile.max_joint_velocity_rad_s * float(self.backend.model.opt.timestep)
        for name in self.joint_names:
            current = self.backend.targets[name]
            difference = self.requested_targets[name] - current
            self.backend.targets[name] = current + min(max_delta, max(-max_delta, difference))
        return messages
