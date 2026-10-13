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
from wmal.skills.manipulation_recovery import StackBlockRGBXYPreReleaseAlignment
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
        self.event_states, self.event_state_mask = [], []
        self.initial_z = None

    def reset(self, observation, truth):
        self.initial_z = truth.object('block_0').position[2]
        self.lifted_seen = False
        self.drop_count = 0
        event, _, _ = _event_vector(observation, truth, self.initial_z)
        self.event_states = [np.concatenate((event, np.asarray([0.], dtype=np.float32)))]
        self.event_state_mask = [np.ones(len(EVENT_NAMES), dtype=np.float32)]
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
        event_state = np.concatenate((event, np.asarray([
            self.drop_count >= DROP_STABLE_SAMPLES], dtype=np.float32)))
        self.events.append(event_state)
        self.event_states.append(event_state.copy())
        self.event_state_mask.append(np.ones(len(EVENT_NAMES), dtype=np.float32))
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
            'event_states': np.stack(self.event_states),
            'event_state_mask': np.stack(self.event_state_mask),
        }
        return np.stack(self.rgb), states, actions, events, event_mask, extra


def _episode_split_labels(episode_count, split_seed):
    """Assign reproducible 60/10/10/20 splits after shuffling episode indices."""
    if type(episode_count) is not int or episode_count < 4:
        raise ValueError('At least four episodes are required for four-way splits')
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError('split_seed must be a nonnegative integer')
    n_validation = max(1, int(episode_count * .10))
    n_calibration = max(1, int(episode_count * .10))
    n_train = min(max(1, int(episode_count * .60)),
                  episode_count - n_validation - n_calibration - 1)
    labels = (['train'] * n_train + ['validation'] * n_validation
              + ['calibration'] * n_calibration
              + ['test'] * (episode_count - n_train - n_validation - n_calibration))
    order = np.random.default_rng(split_seed).permutation(episode_count)
    assigned = [''] * episode_count
    for index, label in zip(order, labels):
        assigned[int(index)] = label
    return tuple(assigned)


def collect(config_path, output_dir, *, seed_start, episode_count, profile, image_size=128,
            recovery_policy='none', recovery_checkpoint=None,
            recovery_calibration=None, recovery_device='cpu',
            recovery_max_control_steps=2048, placement_policy='fixed_waypoint',
            placement_checkpoint=None, placement_calibration=None,
            placement_device='cpu', placement_max_control_steps=512,
            placement_contact_loss_hold_steps=0):
    config = load_config(config_path)
    if type(seed_start) is not int or seed_start < 0 or type(episode_count) is not int or episode_count < 4:
        raise ValueError('Use a nonnegative seed start and at least four episodes for four-way splits')
    config['initial_state_profile'] = profile
    if recovery_policy not in ('none', 'repeat_place_once', 'rgbxy_regrasp_once'):
        raise ValueError('Unsupported recovery policy')
    if (type(placement_contact_loss_hold_steps) is not int
            or not 0 <= placement_contact_loss_hold_steps <= 256):
        raise ValueError('Placement contact-loss hold must be in [0, 256] steps')
    if placement_policy == 'fixed_waypoint' and placement_contact_loss_hold_steps:
        raise ValueError('Placement contact-loss hold requires the RGB-XY pre-release policy')
    recovery_localizer = None
    recovery_artifact = None
    if recovery_policy == 'rgbxy_regrasp_once':
        if not recovery_checkpoint or not recovery_calibration:
            raise ValueError('rgbxy_regrasp_once requires checkpoint and calibration paths')
        from wmal.perception.rgb_xy_regressor import RGBBlockXYLocalizer
        recovery_localizer = RGBBlockXYLocalizer(
            recovery_checkpoint, device=recovery_device,
            calibration_path=recovery_calibration)
        if image_size != recovery_localizer.metadata.image_size:
            raise ValueError('Collector image_size must match the frozen recovery model')
        if config['max_episode_steps'] < 1536 + recovery_max_control_steps:
            raise ValueError('Recovery config must budget the baseline plus one bounded attempt')
        recovery_artifact = {
            'model_version': recovery_localizer.metadata.model_version,
            'checkpoint_sha256': recovery_localizer.checkpoint_sha256,
            'visual_domain_sha256': recovery_localizer.metadata.visual_domain_sha256,
            'calibrated_radius_m': recovery_localizer.calibration['radius_m'],
            'calibration_scope': recovery_localizer.calibration['scope'],
            'calibration_sha256': sha256_file(recovery_calibration),
            'device': recovery_device,
        }
    elif recovery_checkpoint or recovery_calibration:
        raise ValueError('Recovery artifacts require rgbxy_regrasp_once')
    placement_localizer = None
    placement_artifact = None
    if placement_policy == StackBlockRGBXYPreReleaseAlignment.POLICY_ID:
        if not placement_checkpoint or not placement_calibration:
            raise ValueError('RGB-XY placement collection requires a frozen RGB-XY localizer and calibration')
        if recovery_policy != 'none' or recovery_localizer is not None:
            raise ValueError('Pre-release placement collection must be separate from post-release recovery')
        if type(placement_max_control_steps) is not int or placement_max_control_steps < 1:
            raise ValueError('Placement control-step budget must be positive')
        if (type(placement_contact_loss_hold_steps) is not int
                or not 0 <= placement_contact_loss_hold_steps <= 256):
            raise ValueError('Placement contact-loss hold must be in [0, 256] steps')
        if config['max_episode_steps'] < (1536 + placement_max_control_steps
                                          + placement_contact_loss_hold_steps + 256):
            raise ValueError('Placement config must budget alignment, release, and settle')
        from wmal.perception.rgb_xy_regressor import RGBBlockXYLocalizer
        placement_localizer = RGBBlockXYLocalizer(
            placement_checkpoint, device=placement_device,
            calibration_path=placement_calibration)
        if image_size != placement_localizer.metadata.image_size:
            raise ValueError('Collector image_size must match the frozen placement model')
        placement_artifact = {
            'policy': placement_policy,
            'model_version': placement_localizer.metadata.model_version,
            'checkpoint_sha256': placement_localizer.checkpoint_sha256,
            'visual_domain_sha256': placement_localizer.metadata.visual_domain_sha256,
            'calibrated_radius_m': placement_localizer.calibration['radius_m'],
            'calibration_scope': placement_localizer.calibration['scope'],
            'calibration_sha256': sha256_file(placement_calibration),
            'device': placement_device,
            'max_control_steps': placement_max_control_steps,
        }
    elif placement_policy != 'fixed_waypoint':
        raise ValueError('Unsupported placement policy')
    elif placement_checkpoint or placement_calibration:
        raise ValueError('Placement artifacts require the RGB-XY pre-release policy')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw_root, data_root = output / 'raw_runs', output / 'episodes'
    raw_root.mkdir()
    data_root.mkdir()
    rows = []
    split_labels = _episode_split_labels(episode_count, seed_start)
    for i in range(episode_count):
        seed = seed_start + i
        config['seeds'] = [seed]
        raw_run = raw_root / f'seed_{seed:06d}'
        raw_run.mkdir()
        (raw_run / 'events.jsonl').write_text('', encoding='utf-8')
        buffer = EpisodeBuffer()
        result = run_episode(config, seed, raw_run, render=True, render_image_size=image_size,
                             on_reset=buffer.reset, on_step=buffer.step,
                             recovery_policy=recovery_policy,
                             recovery_localizer=recovery_localizer,
                             recovery_max_control_steps=recovery_max_control_steps,
                             placement_policy=placement_policy,
                             placement_localizer=placement_localizer,
                             placement_max_control_steps=placement_max_control_steps,
                             placement_contact_loss_hold_steps=(
                                 placement_contact_loss_hold_steps))
        rgb, states, actions, events, mask, extra = buffer.arrays()
        episode_file = data_root / f'episode_{seed:06d}.npz'
        write_episode(episode_file, rgb, states, actions, events=events,
                      event_mask=mask, extra_arrays=extra)
        split = split_labels[i]
        rows.append({'episode_id': result['episode_id'], 'split': split,
                     'path': f'episodes/{episode_file.name}', 'seed': seed,
                     'initial_state_profile': profile, 'task_success': result['success'],
                     'failure_cause': result['failure_cause'],
                     'placement_policy': placement_policy,
                     'recovery_policy': recovery_policy,
                     'recovery_attempts': result['recovery_attempts'],
                     'recovery_succeeded': result['recovery_succeeded']})
        print(json.dumps({'seed': seed, 'split': split, 'success': result['success'],
                          'transitions': len(actions)}, sort_keys=True), flush=True)
    manifest_path = output / 'manifest.json'
    code_files = ('scripts/collect_stack_block_visual.py', 'scripts/stack_block_baseline.py',
                  'src/wmal/envs/workcell.py', 'src/wmal/envs/manipulation_session.py',
                  'src/wmal/robots/gripper.py', 'src/wmal/skills/manipulation_contracts.py',
                  'src/wmal/skills/manipulation_verifier.py',
                  'src/wmal/skills/cartesian_servo.py',
                  'src/wmal/skills/manipulation_recovery.py')
    write_manifest(manifest_path, rows, action_schema='wmal.manipulation.action.v1',
                   action_order=ACTION_ORDER, state_order=STATE_ORDER,
                   camera='workcell_overview', period_s=config['control_period_s'],
                   image_size=image_size,
                   object_state_order=TRUTH_ORDER,
                   camera_fovy_deg=STACK_WORKCELL_CAMERA_FOVY_DEG,
                   source={'task': 'stack_block', 'profile': profile,
                           'split_seed': seed_start,
                           'world_model': 'disabled', 'placement_policy': placement_policy,
                           'placement_artifact': placement_artifact,
                           'placement_contact_loss_hold_steps': (
                               placement_contact_loss_hold_steps),
                           'recovery_policy': recovery_policy,
                           'recovery_artifact': recovery_artifact,
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
    parser.add_argument('--recovery-policy',
                        choices=('none', 'repeat_place_once', 'rgbxy_regrasp_once'),
                        default='none')
    parser.add_argument('--recovery-checkpoint')
    parser.add_argument('--recovery-calibration')
    parser.add_argument('--recovery-device', default='cpu')
    parser.add_argument('--recovery-max-control-steps', type=int, default=2048)
    parser.add_argument('--placement-policy',
                        choices=('fixed_waypoint',
                                 StackBlockRGBXYPreReleaseAlignment.POLICY_ID),
                        default='fixed_waypoint')
    parser.add_argument('--placement-checkpoint')
    parser.add_argument('--placement-calibration')
    parser.add_argument('--placement-device', default='cpu')
    parser.add_argument('--placement-max-control-steps', type=int, default=512)
    parser.add_argument('--placement-contact-loss-hold-steps', type=int, default=0)
    parser.add_argument('--image-size', choices=(32, 64, 128), type=int, default=64)
    args = parser.parse_args(argv)
    collect(args.config, args.output_dir, seed_start=args.seed_start,
            episode_count=args.episodes, profile=args.profile,image_size=args.image_size,
            recovery_policy=args.recovery_policy,
            recovery_checkpoint=args.recovery_checkpoint,
            recovery_calibration=args.recovery_calibration,
            recovery_device=args.recovery_device,
            recovery_max_control_steps=args.recovery_max_control_steps,
            placement_policy=args.placement_policy,
            placement_checkpoint=args.placement_checkpoint,
            placement_calibration=args.placement_calibration,
            placement_device=args.placement_device,
            placement_max_control_steps=args.placement_max_control_steps,
            placement_contact_loss_hold_steps=(
                args.placement_contact_loss_hold_steps))


if __name__ == '__main__':
    main()
