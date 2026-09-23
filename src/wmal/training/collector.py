"""Collect episode-separated MuJoCo transitions for a joint-state baseline."""
from pathlib import Path
import json
import random

from wmal.communication.contracts import MotionCommand, RobotProfile
from wmal.envs.mujoco_backend import MujocoBackend


def build_backend(config_file):
    path = Path(config_file).resolve()
    config = json.loads(path.read_text())
    profile = RobotProfile(**config['profile'])
    if config.get('locomotion_plugin'):
        raise ValueError('This collector requires joint-only control')
    backend = MujocoBackend(str((path.parent / config['model_path']).resolve()),
                            profile, config['actuator_map'])
    return backend


def collect(config_file, output, *, episodes=10, steps_per_episode=20, duration_s=0.5, seed=0):
    if episodes < 5 or steps_per_episode < 1 or duration_s <= 0:
        raise ValueError('Invalid collection budget')
    backend = build_backend(config_file)
    if duration_s > backend.profile.max_duration_s:
        raise ValueError('Duration exceeds robot limit')
    physics_steps = round(duration_s / backend.model.opt.timestep)
    if physics_steps < 1:
        raise ValueError('Duration shorter than physics step')
    actual_duration = physics_steps * float(backend.model.opt.timestep)
    generator = random.Random(seed)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w') as handle:
        for episode_index in range(episodes):
            backend.reset()
            split = 'train' if episode_index % 5 < 3 else 'validation' if episode_index % 5 == 3 else 'test'
            for action_index in range(steps_per_episode):
                before = backend.observe()
                target = {}
                for joint, value in before.joints.items():
                    low, high = backend.profile.joint_limits[joint]
                    reach = backend.profile.max_joint_velocity_rad_s * actual_duration
                    target[joint] = generator.uniform(max(low, value - reach), min(high, value + reach))
                command = MotionCommand(f'collect_{episode_index}_{action_index}', backend.profile.robot_id,
                                        before.episode_id, before.step_id, 'joint_positions', target, actual_duration)
                backend.begin(command)
                for _ in range(physics_steps):
                    backend.step()
                backend.stop()
                after = backend.observe()
                row = {'source': 'mujoco_interaction', 'split': split, 'seed': seed,
                       'episode_index': episode_index, 'episode_id': before.episode_id,
                       'action_index': action_index, 'before': before.joints,
                       'target': target, 'after': after.joints,
                       'duration_s': actual_duration, 'sim_time_s': before.sim_time_s,
                       'next_sim_time_s': after.sim_time_s, 'step_id': before.step_id,
                       'next_step_id': after.step_id}
                handle.write(json.dumps(row, allow_nan=False) + '\n')
    return destination
