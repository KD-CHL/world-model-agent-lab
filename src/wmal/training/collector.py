"""Collect episode-separated MuJoCo transitions for a joint-state baseline."""
from pathlib import Path
import json
import random
from time import monotonic

import numpy as np

from wmal.communication.contracts import MotionCommand, RobotProfile
from wmal.envs.mujoco_backend import MujocoBackend
from wmal.envs.mujoco_env import MujocoEnvironment
from wmal.logging.manifest import atomic_json, build_manifest, related_path


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
    environment = MujocoEnvironment(backend, action_duration_s=actual_duration)
    generator = random.Random(seed)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_file).resolve()
    config = json.loads(config_path.read_text())
    asset_path = (config_path.parent / config['model_path']).resolve()
    manifest = build_manifest('mujoco_collection', seed, {'config': config_path, 'asset': asset_path},
                              {'episodes': episodes, 'steps_per_episode': steps_per_episode,
                               'duration_s': actual_duration, 'physics_timestep_s': float(backend.model.opt.timestep)},
                              robot_id=backend.profile.robot_id, robot_kind=backend.profile.kind)
    atomic_json(related_path(destination, 'manifest'), manifest)
    count = 0
    started = monotonic()
    state_flag = backend.mj.mjtState.mjSTATE_FULLPHYSICS

    def full_state():
        values = np.empty(backend.mj.mj_stateSize(backend.model, state_flag), dtype=float)
        backend.mj.mj_getState(backend.model, backend.data, values, state_flag)
        return values.tolist()

    with destination.open('w', buffering=1) as handle:
        for episode_index in range(episodes):
            environment.reset(seed=seed + episode_index)
            split = 'train' if episode_index % 5 < 3 else 'validation' if episode_index % 5 == 3 else 'test'
            for action_index in range(steps_per_episode):
                before = backend.observe()
                before_state = full_state()
                target = {}
                for joint, value in before.joints.items():
                    low, high = backend.profile.joint_limits[joint]
                    reach = backend.profile.max_joint_velocity_rad_s * actual_duration
                    target[joint] = generator.uniform(max(low, value - reach), min(high, value + reach))
                command = MotionCommand(f'collect_{episode_index}_{action_index}', backend.profile.robot_id,
                                        before.episode_id, before.step_id, 'joint_positions', target, actual_duration)
                after, execution = environment.step(command)
                applied_controls = execution['applied_controls']
                after_state = full_state()
                row = {'source': 'mujoco_interaction', 'split': split, 'seed': seed,
                       'episode_index': episode_index, 'episode_id': before.episode_id,
                       'action_index': action_index, 'before': before.joints,
                       'target': target, 'after': after.joints,
                       'duration_s': actual_duration, 'sim_time_s': before.sim_time_s,
                       'next_sim_time_s': after.sim_time_s, 'step_id': before.step_id,
                       'next_step_id': after.step_id, 'physics_steps': execution['physics_steps'],
                       'full_physics_state_before': before_state,
                       'full_physics_state_after': after_state,
                       'control_target': target, 'applied_controls': applied_controls,
                       'terminal_reason': 'action_duration_elapsed'}
                handle.write(json.dumps(row, allow_nan=False) + '\n')
                count += 1
            print(json.dumps({'phase': 'collection', 'episode': episode_index, 'split': split,
                              'transitions_written': count, 'output': str(destination)}, ensure_ascii=False), flush=True)
    atomic_json(related_path(destination, 'summary'), {
        'schema_version': 1, 'run_kind': 'mujoco_collection', 'source': 'mujoco_interaction',
        'episodes': episodes, 'transitions': count, 'physics_steps': count * physics_steps,
        'sim_seconds': count * actual_duration, 'wall_seconds': monotonic() - started,
        'split_transitions': {name: sum(steps_per_episode for episode in range(episodes)
                                        if ('train' if episode % 5 < 3 else 'validation' if episode % 5 == 3 else 'test') == name)
                              for name in ('train', 'validation', 'test')},
        'trajectory_path': str(destination), 'manifest_path': str(related_path(destination, 'manifest'))})
    return destination
