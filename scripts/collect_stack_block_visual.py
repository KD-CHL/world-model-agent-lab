"""Collect episode-disjoint, synchronized RGB/state/action/object-truth data."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from wmal.datasets.visual_sequences import SPLITS, write_episode, write_manifest
from wmal.logging.manifest import sha256_file
from wmal.envs.g1_locomotion import JOINT_NAMES
from wmal.envs.manipulation_session import LEFT_ARM_JOINTS
from wmal.envs.workcell import STACK_WORKCELL_CAMERA_FOVY_DEG
from wmal.skills.manipulation_verifier import StackBlockVerifier
from stack_block_baseline import load_config, run_episode


STATE_ORDER = tuple(
    [f'joint_position:{name}' for name in JOINT_NAMES]
    + [f'joint_velocity:{name}' for name in JOINT_NAMES]
    + [f'tool_position_m:{axis}' for axis in 'xyz']
    + [f'tool_quaternion_wxyz:{axis}' for axis in 'wxyz']
    + ['gripper:opening_m', 'gripper:target_opening_m', 'gripper:contact_left',
       'gripper:contact_right', 'gripper:at_target'])
ACTION_ORDER = tuple([f'joint_target:{name}' for name in LEFT_ARM_JOINTS]
                     + ['gripper_mode:none', 'gripper_mode:open',
                        'gripper_mode:close', 'gripper_mode:hold',
                        'gripper:opening_m', 'gripper:speed_m_s',
                        'gripper:force_limit_n'])
DROP_RESET_TOLERANCE_M = .02
DROP_STABLE_SAMPLES = 2
EVENT_NAMES = ('bilateral_grasp_contact', 'object_lifted', 'stable_stack',
               'object_dropped_after_lift')
TRUTH_ORDER = tuple(
    f'{obj}:{field}:{axis}'
    for obj in ('block_0', 'block_1', 'block_2')
    for field, axes in (('position_m', 'xyz'), ('quaternion_wxyz', 'wxyz'),
                        ('linear_velocity_m_s', 'xyz'), ('angular_velocity_rad_s', 'xyz'))
    for axis in axes)


def _robot_state(observation):
    g = observation.gripper
    values = (*observation.joint_position, *observation.joint_velocity,
              *observation.tool_position_m, *observation.tool_quaternion_wxyz,
              g.opening_m, g.target_opening_m, float(g.contact_left),
              float(g.contact_right), float(g.at_target))
    return np.asarray(values, dtype=np.float32)


def _rgb_u8(observation):
    if observation.rgb is None:
        raise RuntimeError('Visual collection requires a rendered RGB frame at every control step')
    return np.clip(np.rint(observation.rgb.transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)


def _truth_vector(truth):
    row = []
    for name in ('block_0', 'block_1', 'block_2'):
        obj = truth.object(name)
        row.extend(obj.position)
        row.extend(obj.quaternion_wxyz)
        row.extend(obj.linear_velocity)
        row.extend(obj.angular_velocity)
    return np.asarray(row, dtype=np.float32)


def _event_vector(observation, truth, initial_z, *, was_lifted=False):
    object_state = truth.object('block_0')
    contacts = {frozenset(pair) for pair in truth.contacts}
    left = frozenset(('block_0_geom', 'g1_0_stack_gripper_left_finger_geom')) in contacts
    right = frozenset(('block_0_geom', 'g1_0_stack_gripper_right_finger_geom')) in contacts
    lifted = object_state.position[2] >= initial_z + .035
    region = next(r for r in truth.target_regions if r[0] == 'stack_target')
    _, center, half = region
    in_xy = (abs(object_state.position[0] - center[0]) <= min(half[0], .055)
             and abs(object_state.position[1] - center[1]) <= min(half[1], .055))
    support_contact = frozenset(('block_0_geom', 'block_1_geom')) in contacts
    speed = np.linalg.norm(object_state.linear_velocity) <= .025
    angular_speed = np.linalg.norm(object_state.angular_velocity) <= .15
    w, x, y, z = object_state.quaternion_wxyz
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1, 1))
    upright = np.hypot(roll, pitch) <= .2
    stable = in_xy and support_contact and speed and angular_speed and upright
    bilateral = left and right
    dropped = (was_lifted and not bilateral and not stable
               and object_state.position[2] <= initial_z + DROP_RESET_TOLERANCE_M)
    return np.asarray([bilateral, lifted, stable], dtype=np.float32), bool(
        bilateral and lifted), bool(dropped)


class EpisodeBuffer:
    def __init__(self):
        self.rgb, self.states, self.truth = [], [], []
        self.times, self.steps, self.incoming_action_ids = [], [], []
        self.actions, self.events = [], []
        self.initial_z = None

    def reset(self, observation, truth):
        self.initial_z = truth.object('block_0').position[2]
        self.lifted_seen = False
        self.drop_count = 0
        self.rgb = [_rgb_u8(observation)]
        self.states = [_robot_state(observation)]
        self.truth = [_truth_vector(truth)]
        self.times = [observation.sim_time_s]
        self.steps = [observation.step_id]
        self.incoming_action_ids = ['reset']

    def step(self, observation, receipt, truth):
        if observation.rgb is None:
            raise RuntimeError('Rendering unexpectedly returned no RGB observation')
        if observation.step_id != self.steps[-1] + 1:
            raise RuntimeError('Noncontiguous episode observations')
        target = dict(receipt.joint_targets)
        if set(target) != set(LEFT_ARM_JOINTS):
            raise RuntimeError('Action target joints do not match the frozen skill schema')
        mode = receipt.gripper_mode or 'none'
        action = [target[name] for name in LEFT_ARM_JOINTS]
        action.extend(float(mode == candidate) for candidate in ('none', 'open', 'close', 'hold'))
        action.extend((receipt.gripper_opening_m, receipt.gripper_speed_m_s,
                       receipt.gripper_force_limit_n))
        self.actions.append(action)
        event, lifted_now, dropped_now = _event_vector(
            observation, truth, self.initial_z, was_lifted=self.lifted_seen)
        self.lifted_seen = self.lifted_seen or lifted_now
        self.drop_count = self.drop_count + 1 if dropped_now else 0
        self.events.append(np.concatenate((event, np.asarray([
            self.drop_count >= DROP_STABLE_SAMPLES], dtype=np.float32))))
        self.rgb.append(_rgb_u8(observation))
        self.states.append(_robot_state(observation))
        self.truth.append(_truth_vector(truth))
        self.times.append(observation.sim_time_s)
        self.steps.append(observation.step_id)
        self.incoming_action_ids.append(receipt.action_id)

    def arrays(self):
        if len(self.actions) < 1 or len(self.states) != len(self.actions) + 1:
            raise RuntimeError('Episode does not contain a complete transition sequence')
        states = np.stack(self.states)
        actions = np.asarray(self.actions, dtype=np.float32)
        events = np.asarray(self.events, dtype=np.float32)
        event_mask = np.ones_like(events, dtype=np.float32)
        extra = {
            'object_truth': np.stack(self.truth),
            'sim_times_s': np.asarray(self.times, dtype=np.float64),
            'step_ids': np.asarray(self.steps, dtype=np.int64),
            'incoming_action_ids': np.asarray(self.incoming_action_ids, dtype='U128'),
        }
        return np.stack(self.rgb), states, actions, events, event_mask, extra


def collect(config_path, output_dir, *, seed_start, episode_count, profile, image_size=128,
            recovery_policy='none'):
    config = load_config(config_path)
    if type(seed_start) is not int or seed_start < 0 or type(episode_count) is not int or episode_count < 4:
        raise ValueError('Use a nonnegative seed start and at least four episodes for four-way splits')
    config['initial_state_profile'] = profile
    if recovery_policy not in ('none', 'repeat_place_once'):
        raise ValueError('Unsupported recovery policy')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw_root, data_root = output / 'raw_runs', output / 'episodes'
    raw_root.mkdir()
    data_root.mkdir()
    rows = []
    n_validation = max(1, int(episode_count * .10))
    n_calibration = max(1, int(episode_count * .10))
    n_train = min(max(1, int(episode_count * .60)),
                  episode_count - n_validation - n_calibration - 1)
    for i in range(episode_count):
        seed = seed_start + i
        config['seeds'] = [seed]
        raw_run = raw_root / f'seed_{seed:06d}'
        raw_run.mkdir()
        (raw_run / 'events.jsonl').write_text('', encoding='utf-8')
        buffer = EpisodeBuffer()
        result = run_episode(config, seed, raw_run, render=True, render_image_size=image_size,
                             on_reset=buffer.reset, on_step=buffer.step,
                             recovery_policy=recovery_policy)
        rgb, states, actions, events, mask, extra = buffer.arrays()
        episode_file = data_root / f'episode_{seed:06d}.npz'
        write_episode(episode_file, rgb, states, actions, events=events,
                      event_mask=mask, extra_arrays=extra)
        split = ('train' if i < n_train else
                 'validation' if i < n_train + n_validation else
                 'calibration' if i < n_train + n_validation + n_calibration else 'test')
        rows.append({'episode_id': result['episode_id'], 'split': split,
                     'path': f'episodes/{episode_file.name}', 'seed': seed,
                     'initial_state_profile': profile, 'task_success': result['success'],
                     'failure_cause': result['failure_cause'],
                     'recovery_policy': recovery_policy,
                     'recovery_attempts': result['recovery_attempts'],
                     'recovery_succeeded': result['recovery_succeeded']})
        print(json.dumps({'seed': seed, 'split': split, 'success': result['success'],
                          'transitions': len(actions)}, sort_keys=True), flush=True)
    manifest_path = output / 'manifest.json'
    code_files = ('scripts/collect_stack_block_visual.py', 'scripts/stack_block_baseline.py',
                  'src/wmal/envs/workcell.py', 'src/wmal/envs/manipulation_session.py',
                  'src/wmal/robots/gripper.py', 'src/wmal/skills/manipulation_contracts.py',
                  'src/wmal/skills/manipulation_verifier.py')
    write_manifest(manifest_path, rows, action_schema='wmal.manipulation.action.v1',
                   action_order=ACTION_ORDER, state_order=STATE_ORDER,
                   camera='workcell_overview', period_s=config['control_period_s'],
                   image_size=image_size,
                   object_state_order=TRUTH_ORDER,
                   camera_fovy_deg=STACK_WORKCELL_CAMERA_FOVY_DEG,
                   source={'task': 'stack_block', 'profile': profile,
                           'world_model': 'disabled', 'recovery_policy': recovery_policy,
                           'truth_array': 'object_truth (labels only; excluded from model inputs)',
                           'code_sha256': {name: sha256_file(ROOT / name) for name in code_files}},
                   event_names=EVENT_NAMES)
    counts = {split: sum(row['split'] == split for row in rows) for split in SPLITS}
    print(json.dumps({'manifest': str(manifest_path), 'episodes': len(rows),
                      'split_counts': counts}, sort_keys=True), flush=True)
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/tasks/stack_block_baseline.json'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--profile', choices=('nominal', 'jittered', 'disturbed'), default='jittered')
    parser.add_argument('--recovery-policy', choices=('none', 'repeat_place_once'), default='none')
    parser.add_argument('--image-size', choices=(32, 64, 128), type=int, default=64)
    args = parser.parse_args(argv)
    collect(args.config, args.output_dir, seed_start=args.seed_start,
            episode_count=args.episodes, profile=args.profile,image_size=args.image_size,
            recovery_policy=args.recovery_policy)


if __name__ == '__main__':
    main()
