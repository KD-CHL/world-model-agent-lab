"""Robot-specific command boundaries, independent of ROS vendor messages."""
from wmal.communication.contracts import MotionCommand
from uuid import uuid4


class RobotInterface:
    kind = None

    def __init__(self, profile):
        if profile.kind != self.kind:
            raise ValueError('Robot kind mismatch')
        self.profile = profile

    def validate_command(self, command):
        command.validate(self.profile)

    def joint_command(self, observation, targets, duration_s=1.0):
        observation.validate(self.profile)
        command = MotionCommand(str(uuid4()), self.profile.robot_id, observation.episode_id,
                                observation.step_id, 'joint_positions', dict(targets), duration_s)
        self.validate_command(command)
        return command


class ArmInterface(RobotInterface):
    kind = 'arm'


class LeggedInterface(RobotInterface):
    def velocity_command(self, observation, vx, vy, yaw_rate, duration_s=0.5):
        observation.validate(self.profile)
        command = MotionCommand(str(uuid4()), self.profile.robot_id, observation.episode_id,
                                observation.step_id, 'base_velocity',
                                {'vx': vx, 'vy': vy, 'yaw_rate': yaw_rate}, duration_s)
        self.validate_command(command)
        return command


class Go2Interface(LeggedInterface):
    kind = 'go2'


class G1Interface(LeggedInterface):
    kind = 'g1'


def robot_interface(profile):
    return {'arm': ArmInterface, 'go2': Go2Interface, 'g1': G1Interface}[profile.kind](profile)
