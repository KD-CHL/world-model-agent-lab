"""Run the fixed no-world-model stack-block skill baseline."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from wmal.envs.manipulation_session import ManipulationSession
from wmal.envs.g1_locomotion import MODEL_XML
from wmal.skills.manipulation_contracts import SKILL_ORDER, StackBlockSkills, default_waypoint_profile
from wmal.skills.manipulation_verifier import StackBlockVerifier
from wmal.skills.manipulation_recovery import (
    StackBlockRGBXYPreReleaseAlignment, StackBlockRGBXYRecovery,
)


_GENERIC_SKILL_FAILURES = {'independent_predicate_not_passed'}
_TASK_SUCCESS_DEFINITION = (
    'independent stable stack predicate AND target_block_released '
    '(healthy gripper fully open and at target, with no target-block-to-finger contact)')
_PRIMARY_FAILURE_PREDICATES = {
    'grasp_block': ('grasped',),
    'lift_block': ('lifted',),
    'move_to_stack_target': ('lifted',),
    'place_block': ('placed', 'stacked'),
    'retreat': ('stacked',),
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _target_block_released(observation, truth, *, max_opening_m):
    """Require a fully open, healthy gripper and no target-to-finger contact."""
    gripper = getattr(observation, 'gripper', None)
    if gripper is None or getattr(gripper, 'fault', None):
        return False
    try:
        opening = float(gripper.opening_m)
        target_opening = float(gripper.target_opening_m)
        maximum = float(max_opening_m)
        contacts = tuple(truth.contacts)
    except (AttributeError, TypeError, ValueError):
        return False
    if (not all(isfinite(value) for value in (opening, target_opening, maximum))
            or maximum <= 0 or opening < maximum - .002
            or target_opening < maximum - .002
            or getattr(gripper, 'at_target', False) is not True):
        return False
    finger_pairs = (
        {'block_0_geom', 'g1_0_stack_gripper_left_finger_geom'},
        {'block_0_geom', 'g1_0_stack_gripper_right_finger_geom'},
    )
    return not any(set(pair) in finger_pairs for pair in contacts)


def resolve_failure_cause(verifier_failure, stage_results):
    """Prefer latched verifier or specific skill evidence to generic status text."""
    if verifier_failure:
        return verifier_failure
    for stage in reversed(stage_results):
        failure=stage.get('failure')
        if failure and failure not in _GENERIC_SKILL_FAILURES:
            return failure
    for stage in reversed(stage_results):
        reports=stage.get('predicates',())
        by_predicate={report.get('predicate'):report for report in reports}
        for predicate in _PRIMARY_FAILURE_PREDICATES.get(stage.get('skill'),()):
            report=by_predicate.get(predicate)
            if report and report.get('passed') is False and report.get('failure'):
                return report['failure']
        for report in reports:
            if report.get('passed') is False and report.get('failure'):
                return report['failure']
    for stage in reversed(stage_results):
        failure=stage.get('failure')
        if failure:
            return failure
    return 'stack_predicate_not_satisfied'


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {'schema', 'task', 'world_model', 'initial_state_profile', 'seeds',
                'control_period_s', 'max_episode_steps', 'skill_budget_control_steps',
                'skill_timeout_s', 'waypoint_profile', 'verifier'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Baseline config has unknown or missing fields')
    if config['schema'] != 'wmal.stack_block_baseline.v1' or config['task'] != 'stack_block':
        raise ValueError('Unsupported baseline config schema/task')
    if config['world_model'] != 'disabled':
        raise ValueError('Stage-one baseline must not load a world model')
    if config['initial_state_profile'] not in ('nominal', 'jittered', 'disturbed'):
        raise ValueError('Invalid initial-state profile')
    if not isinstance(config['seeds'], list) or not config['seeds'] or any(type(s) is not int or s < 0 for s in config['seeds']):
        raise ValueError('seeds must be a nonempty list of nonnegative integers')
    if len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('seeds must be unique')
    return config


def _receipt_action_trace(session, receipts):
    rows = []
    for receipt in receipts:
        snapshots = session.truth_trace_for_verifier(receipt)
        final_truth = ({
            obj.name: {'position': obj.position,
                       'quaternion_wxyz': obj.quaternion_wxyz,
                       'linear_velocity': obj.linear_velocity,
                       'angular_velocity': obj.angular_velocity}
            for obj in snapshots[-1].object_states
        } if snapshots else None)
        rows.append({
            'action_id': receipt.action_id,
            'start_step_id': receipt.start_step_id,
            'end_step_id': receipt.end_step_id,
            'observation_step_ids': [item.step_id for item in receipt.observations],
            'control_steps': receipt.executed_steps,
            'sim_duration_s': receipt.sim_duration_s,
            'joint_targets': receipt.joint_targets,
            'gripper_mode': receipt.gripper_mode,
            'gripper_opening_m': receipt.gripper_opening_m,
            'gripper_speed_m_s': receipt.gripper_speed_m_s,
            'gripper_force_limit_n': receipt.gripper_force_limit_n,
            'gripper_command_receipt': (
                None if receipt.gripper_command_receipt is None else
                asdict(receipt.gripper_command_receipt)),
            'final_observation': (receipt.observations[-1].to_agent_dict()
                                  if receipt.observations else None),
            'final_truth': final_truth,
        })
    return rows


def _record_step_observations(session, receipts, on_step):
    if on_step is None:
        return
    for receipt in receipts:
        snapshots = session.truth_trace_for_verifier(receipt)
        truth_by_step = {truth.step_id: truth for truth in snapshots}
        for step_observation in receipt.observations:
            truth = truth_by_step.get(step_observation.step_id)
            if truth is not None:
                on_step(step_observation, receipt, truth)


def _post_release_observations(receipts):
    """Sample the final settle receipt, strictly after the intentional open action."""
    release_indices = [i for i, receipt in enumerate(receipts)
                       if receipt.gripper_mode == 'open']
    if not release_indices:
        return ()
    settle = [receipt for receipt in receipts[max(release_indices) + 1:]
              if receipt.gripper_mode is None]
    if not settle:
        return ()
    observations = settle[-1].observations
    if len(observations) < 65:
        return ()
    return observations[-65], observations[-33], observations[-1]


def run_episode(config, seed, run_dir, *, render=False, render_image_size=32,
                on_reset=None, on_step=None, recovery_policy='none',
                recovery_localizer=None, recovery_max_control_steps=2048,
                placement_policy='fixed_waypoint', placement_localizer=None,
                placement_max_control_steps=512,
                placement_contact_loss_hold_steps=0):
    if recovery_policy not in ('none', 'repeat_place_once', 'rgbxy_regrasp_once'):
        raise ValueError('Unsupported recovery policy')
    if placement_policy not in ('fixed_waypoint',
                                StackBlockRGBXYPreReleaseAlignment.POLICY_ID):
        raise ValueError('Unsupported placement policy')
    if type(placement_max_control_steps) is not int or placement_max_control_steps < 1:
        raise ValueError('Placement control-step budget must be positive')
    if (type(placement_contact_loss_hold_steps) is not int
            or not 0 <= placement_contact_loss_hold_steps <= 256):
        raise ValueError('Placement contact-loss hold must be in [0, 256] steps')
    if placement_policy == 'fixed_waypoint' and placement_contact_loss_hold_steps:
        raise ValueError('Placement contact-loss hold requires the RGB-XY pre-release policy')
    if placement_policy == StackBlockRGBXYPreReleaseAlignment.POLICY_ID:
        if placement_localizer is None:
            raise ValueError('rgbxy_pre_release_align_strict_release_dev_v4 requires a frozen RGB-XY localizer')
        if recovery_policy != 'none' or recovery_localizer is not None:
            raise ValueError('Pre-release alignment must be evaluated separately from post-release recovery')
        if config['max_episode_steps'] < (1536 + placement_max_control_steps
                                          + placement_contact_loss_hold_steps + 256):
            raise ValueError('Episode step budget must cover placement alignment, release, and settle')
        render = True
        render_image_size = placement_localizer.metadata.image_size
    elif placement_localizer is not None:
        raise ValueError('A placement localizer is only valid with rgbxy_pre_release_align_strict_release_dev_v4')
    if recovery_policy == 'rgbxy_regrasp_once':
        if recovery_localizer is None:
            raise ValueError('rgbxy_regrasp_once requires a frozen RGB-XY localizer')
        if config['max_episode_steps'] < 1536 + recovery_max_control_steps:
            raise ValueError('Episode step budget must cover the baseline and one bounded recovery')
        render = True
        render_image_size = recovery_localizer.metadata.image_size
    elif recovery_localizer is not None:
        raise ValueError('A recovery localizer is only valid with rgbxy_regrasp_once')
    session = ManipulationSession(render=render, image_size=render_image_size,
                                  control_period_s=config['control_period_s'],
                                  max_episode_steps=config['max_episode_steps'])
    try:
        observation, audit = session.reset(seed, config['initial_state_profile'])
        reset_observation = observation
        initial_observation = observation.to_agent_dict()
        verifier = StackBlockVerifier(**config['verifier'])
        baseline_truth = session.truth_for_verifier()
        if on_reset is not None:
            on_reset(observation, baseline_truth)
        verifier.episode_id = baseline_truth.episode_id
        verifier.last_step_id = baseline_truth.step_id
        verifier.initial_z = baseline_truth.object('block_0').position[2]
        skills = StackBlockSkills(default_waypoint_profile())
        recovery = (StackBlockRGBXYRecovery(
            recovery_localizer, max_control_steps=recovery_max_control_steps)
            if recovery_policy == 'rgbxy_regrasp_once' else None)
        placement_alignment = (
            StackBlockRGBXYPreReleaseAlignment(
                placement_localizer,
                max_control_steps=placement_max_control_steps,
                contact_loss_hold_steps=placement_contact_loss_hold_steps)
            if placement_policy == StackBlockRGBXYPreReleaseAlignment.POLICY_ID
            else None)
        recovery_anchor = None
        recovery_anchor_reason = ('disabled' if recovery is None else
                                  'awaiting_measured_bilateral_grasp')
        placement_anchor = None
        placement_anchor_reason = ('disabled' if placement_alignment is None else
                                   'awaiting_measured_bilateral_grasp')
        results = []
        recovery_attempts = 0
        with (run_dir / 'events.jsonl').open('a', encoding='utf-8') as events:
            for _ in range(len(SKILL_ORDER) + 1):
                proposals = skills.candidates(observation)
                if not proposals:
                    break
                command = proposals[0]
                verifier_rows = []

                def verify_receipts(receipts, skill_name):
                    for receipt in receipts:
                        snapshots = session.truth_trace_for_verifier(receipt)
                        if not snapshots:
                            continue
                        report_rows = verifier.update(
                            snapshots[-1], receipt, {'truth_snapshots': snapshots})
                        verifier_rows.extend(report_rows if isinstance(report_rows, tuple) else (report_rows,))
                    needed = {
                        'grasp_block': {'grasped'}, 'lift_block': {'lifted'},
                        'move_to_stack_target': {'lifted'},
                        'place_block': {'placed', 'stacked'},
                    }.get(skill_name, set())
                    # Intermediate receipts may precede the predicate becoming
                    # true (for example the first close chunk). Judge the final
                    # cumulative state for each required predicate.
                    final_by_predicate = {}
                    for row in verifier_rows:
                        if row.predicate in needed:
                            final_by_predicate[row.predicate] = row
                    return tuple(final_by_predicate[name] for name in sorted(needed)
                                 if name in final_by_predicate)

                execute_options = {
                    'timeout_s': config['skill_timeout_s'],
                    'verifier': verify_receipts,
                }
                if command.name == 'place_block' and placement_alignment is not None:
                    execute_options['pre_release_alignment'] = placement_alignment
                    execute_options['pre_release_anchor'] = placement_anchor
                skill_run = skills.execute(
                    session, command, config['skill_budget_control_steps'],
                    **execute_options)
                observation = skill_run.final_observation
                if recovery is not None and command.name == 'grasp_block':
                    recovery_anchor, recovery_anchor_reason = recovery.capture_grasp_anchor(
                        reset_observation, skill_run.final_observation)
                event = {
                    'episode_id': observation.episode_id, 'seed': seed,
                    'skill': skill_run.name, 'status': skill_run.status,
                    'control_steps': skill_run.control_steps,
                    'observation_final': observation.to_agent_dict(),
                    'observation_complete': (not skill_run.receipts or
                        observation.step_id >= max(receipt.end_step_id for receipt in skill_run.receipts)),
                    'skill_evidence': [item.__dict__ if hasattr(item, '__dict__') else item
                                       for item in skill_run.evidence],
                    'start_step_id': command.expected_step_id,
                    'end_step_id': max((receipt.end_step_id for receipt in skill_run.receipts),
                                       default=observation.step_id),
                    'action_ids': [r.action_id for r in skill_run.receipts],
                    'action_trace': _receipt_action_trace(session, skill_run.receipts),
                    'failure': skill_run.failure,
                    'predicates': [report.__dict__ for report in verifier_rows],
                    'recovery_policy': recovery_policy,
                    'placement_policy': placement_policy,
                    'recovery_trigger': (recovery_policy == 'repeat_place_once'
                                         and command.name == 'place_block'
                                         and skill_run.status == 'success_unverified'
                                         and recovery_attempts == 0),
                    'is_recovery_action': recovery_attempts > 0,
                }
                if skill_run.details and 'placement_alignment' in skill_run.details:
                    event['placement_alignment'] = skill_run.details['placement_alignment']
                if placement_alignment is not None and command.name == 'grasp_block':
                    placement_anchor, placement_anchor_reason = (
                        placement_alignment.capture_grasp_anchor(
                            reset_observation, skill_run.final_observation))
                    event['placement_anchor_reason'] = placement_anchor_reason
                    event['placement_anchor'] = placement_anchor
                _record_step_observations(session, skill_run.receipts, on_step)
                assessment = None
                if recovery is not None and command.name == 'place_block':
                    assessment = recovery.assessor.assess(
                        _post_release_observations(skill_run.receipts))
                    event['recovery_assessment'] = assessment.to_dict()
                    event['recovery_anchor_reason'] = recovery_anchor_reason
                    event['recovery_trigger_candidate'] = assessment.trigger
                    event['recovery_trigger'] = bool(assessment.trigger and recovery_anchor)
                    if assessment.trigger and recovery_anchor is None:
                        event['recovery_decline_reason'] = recovery_anchor_reason
                elif recovery is not None and command.name == 'grasp_block':
                    event['recovery_anchor_reason'] = recovery_anchor_reason
                    event['recovery_anchor'] = recovery_anchor
                events.write(json.dumps(event, sort_keys=True) + '\n')
                results.append(event)
                if (recovery is not None and command.name == 'place_block'
                        and assessment is not None and assessment.trigger
                        and recovery_anchor is not None):
                    recovery_attempts += 1

                    def log_recovery_receipt(receipt, _stage):
                        # This path only persists measurements; it does not feed
                        # verifier truth back to the recovery controller.
                        return None

                    recovery_run = recovery.execute_once(
                        session, assessment, recovery_anchor,
                        on_receipt=log_recovery_receipt)
                    _record_step_observations(session, recovery_run.receipts, on_step)
                    action_trace = _receipt_action_trace(session, recovery_run.receipts)
                    verifier_rows = []
                    post_recovery_predicates = verify_receipts(
                        recovery_run.receipts, 'place_block')
                    recovery_event = {
                        'episode_id': recovery_run.final_observation.episode_id,
                        'seed': seed, 'skill': 'rgbxy_regrasp_once',
                        'status': recovery_run.status,
                        'control_steps': recovery_run.control_steps,
                        'observation_final': recovery_run.final_observation.to_agent_dict(),
                        'observation_complete': (not recovery_run.receipts or
                            recovery_run.final_observation.step_id >=
                            max(receipt.end_step_id for receipt in recovery_run.receipts)),
                        'start_step_id': (recovery_run.receipts[0].start_step_id
                                          if recovery_run.receipts else
                                          recovery_run.final_observation.step_id),
                        'end_step_id': recovery_run.final_observation.step_id,
                        'action_ids': [receipt.action_id for receipt in recovery_run.receipts],
                        'action_trace': action_trace,
                        'failure': recovery_run.reason,
                        'skill_evidence': [],
                        'predicates': [row.__dict__ for row in post_recovery_predicates],
                        'recovery_policy': recovery_policy,
                        'recovery_trigger': False,
                        'recovery_assessment': assessment.to_dict(),
                        'recovery_anchor': recovery_anchor,
                        'recovery_motion': recovery_run.to_dict(),
                        'is_recovery_action': True,
                    }
                    events.write(json.dumps(recovery_event, sort_keys=True) + '\n')
                    results.append(recovery_event)
                    observation = recovery_run.final_observation
                    break
                if skill_run.status not in ('executed', 'succeeded'):
                    if event['recovery_trigger']:
                        recovery_attempts += 1
                        continue
                    break
                # The task predicate is already terminal after a stable place.
                # Do not run the still-uncertified withdrawal profile: the
                # trial evidence shows it can contact and overturn the stack.
                if command.name == 'place_block' and verifier.stacked:
                    break
            stacked = verifier.stacked
            final_truth = session.truth_for_verifier()
            release_confirmed = _target_block_released(
                observation, final_truth,
                max_opening_m=session.gripper_spec.max_opening_m)
            task_success = stacked and release_confirmed
            cause = (None if task_success else
                     'target_block_not_released' if stacked else
                     resolve_failure_cause(verifier.failure_reason, results))
            return {
                'seed': seed, 'episode_id': audit.episode_id,
                'initial_state_profile': config['initial_state_profile'],
                'model_signature': audit.model_signature,
                'initial_observation': initial_observation,
                'initial_object_states': {
                    obj.name: {'position': obj.position,
                               'quaternion_wxyz': obj.quaternion_wxyz}
                    for obj in audit.object_states
                },
                'status': 'success' if task_success else 'failure',
                'success': task_success,
                'stacked_predicate': stacked,
                'release_confirmed': release_confirmed,
                'recovery_policy': recovery_policy,
                'placement_policy': placement_policy,
                'recovery_attempts': recovery_attempts,
                'recovery_succeeded': bool(recovery_attempts and stacked),
                'failure_cause': cause,
                'control_steps': session.trace[-1].end_step_id if session.trace else 0,
                'sim_time_s': session.sim_time_s,
                'verifier_truth_final': {
                    'objects': {obj.name: {'position': obj.position,
                                           'quaternion_wxyz': obj.quaternion_wxyz,
                                           'linear_velocity': obj.linear_velocity,
                                           'angular_velocity': obj.angular_velocity}
                                for obj in session.truth_for_verifier().object_states},
                    'contacts': final_truth.contacts,
                    'gripper': observation.gripper.__dict__,
                },
                'stage_events': results,
            }
    finally:
        session.close()


def run_batch(config, output_dir, config_path=None, *, recovery_policy='none',
              recovery_localizer=None, recovery_max_control_steps=2048,
              placement_policy='fixed_waypoint', placement_localizer=None,
              placement_max_control_steps=512,
              placement_contact_loss_hold_steps=0):
    if recovery_policy == 'rgbxy_regrasp_once':
        if recovery_localizer is None:
            raise ValueError('rgbxy_regrasp_once requires a frozen RGB-XY localizer')
        if config['max_episode_steps'] < 1536 + recovery_max_control_steps:
            raise ValueError('Episode step budget must cover the baseline and one bounded recovery')
    elif recovery_policy not in ('none', 'repeat_place_once') or recovery_localizer is not None:
        raise ValueError('Unsupported recovery policy/localizer combination')
    if placement_policy not in ('fixed_waypoint',
                                StackBlockRGBXYPreReleaseAlignment.POLICY_ID):
        raise ValueError('Unsupported placement policy')
    if (type(placement_contact_loss_hold_steps) is not int
            or not 0 <= placement_contact_loss_hold_steps <= 256):
        raise ValueError('Placement contact-loss hold must be in [0, 256] steps')
    if placement_policy == 'fixed_waypoint' and placement_contact_loss_hold_steps:
        raise ValueError('Placement contact-loss hold requires the RGB-XY pre-release policy')
    if placement_policy == StackBlockRGBXYPreReleaseAlignment.POLICY_ID:
        if placement_localizer is None:
            raise ValueError('rgbxy_pre_release_align_strict_release_dev_v4 requires a frozen RGB-XY localizer')
        if recovery_policy != 'none' or recovery_localizer is not None:
            raise ValueError('Pre-release alignment must be evaluated separately from post-release recovery')
        if type(placement_max_control_steps) is not int or placement_max_control_steps < 1:
            raise ValueError('Placement control-step budget must be positive')
        if config['max_episode_steps'] < (1536 + placement_max_control_steps
                                          + placement_contact_loss_hold_steps + 256):
            raise ValueError('Episode step budget must cover placement alignment, release, and settle')
    elif placement_localizer is not None:
        raise ValueError('A placement localizer is only valid with rgbxy_pre_release_align_strict_release_dev_v4')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        'schema': 'wmal.stack_block_run_manifest.v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'config': config, 'config_sha256': digest(config_path or ROOT / 'configs/tasks/stack_block_baseline.json'),
        'code_sha256': {name: digest(ROOT / name) for name in (
            'scripts/stack_block_baseline.py',
            'src/wmal/envs/workcell.py', 'src/wmal/envs/manipulation_session.py',
            'src/wmal/robots/gripper.py', 'src/wmal/skills/manipulation_contracts.py',
            'src/wmal/skills/manipulation_verifier.py',
            'src/wmal/skills/cartesian_servo.py',
            'src/wmal/skills/manipulation_recovery.py')},
        'robot_asset_sha256': digest(MODEL_XML),
        'world_model_loaded': False,
        'task_success_definition': _TASK_SUCCESS_DEFINITION,
        'gripper_asset': 'synthetic_parallel_jaw_proxy',
        'gripper_is_dex1': False,
        'controller': 'G1 position actuators with bounded per-control-step joint slew',
        'verifier_version': StackBlockVerifier.VERSION,
        'recovery_policy': recovery_policy,
        'recovery_artifact': (None if recovery_localizer is None else {
            'model_version': recovery_localizer.metadata.model_version,
            'checkpoint_sha256': recovery_localizer.checkpoint_sha256,
            'visual_domain_sha256': recovery_localizer.metadata.visual_domain_sha256,
            'calibration': recovery_localizer.calibration,
        }),
        'placement_policy': placement_policy,
        'placement_artifact': (None if placement_localizer is None else {
            'model_version': placement_localizer.metadata.model_version,
            'checkpoint_sha256': placement_localizer.checkpoint_sha256,
            'visual_domain_sha256': placement_localizer.metadata.visual_domain_sha256,
            'calibration': placement_localizer.calibration,
        }),
        'placement_max_control_steps': placement_max_control_steps,
        'placement_contact_loss_hold_steps': placement_contact_loss_hold_steps,
        'seed_count': len(config['seeds']),
    }
    (output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    (output_dir / 'events.jsonl').write_text('', encoding='utf-8')
    episodes = []
    for seed in config['seeds']:
        episodes.append(run_episode(
            config, seed, output_dir, recovery_policy=recovery_policy,
            recovery_localizer=recovery_localizer,
            recovery_max_control_steps=recovery_max_control_steps,
            placement_policy=placement_policy,
            placement_localizer=placement_localizer,
            placement_max_control_steps=placement_max_control_steps,
            placement_contact_loss_hold_steps=placement_contact_loss_hold_steps))
    successes = sum(bool(x['success']) for x in episodes)
    result = {
        'schema': 'wmal.stack_block_results.v2', 'task': config['task'],
        'world_model': 'disabled', 'placement_policy': placement_policy,
        'task_success_definition': _TASK_SUCCESS_DEFINITION,
        'sample_count': len(episodes),
        'success_count': successes, 'failure_count': len(episodes)-successes,
        'success_rate': successes / len(episodes),
        'failure_denominator_includes_interrupted_episodes': True,
        'episodes': episodes,
    }
    (output_dir / 'results.json').write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/tasks/stack_block_baseline.json'))
    parser.add_argument('--output-dir', required=True)
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
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        localizer = None
        if args.recovery_policy == 'rgbxy_regrasp_once':
            if not args.recovery_checkpoint or not args.recovery_calibration:
                raise ValueError('rgbxy_regrasp_once requires checkpoint and calibration paths')
            from wmal.perception.rgb_xy_regressor import RGBBlockXYLocalizer
            localizer = RGBBlockXYLocalizer(
                args.recovery_checkpoint, device=args.recovery_device,
                calibration_path=args.recovery_calibration)
        elif args.recovery_checkpoint or args.recovery_calibration:
            raise ValueError('Recovery artifacts require rgbxy_regrasp_once')
        placement_localizer = None
        if args.placement_policy == StackBlockRGBXYPreReleaseAlignment.POLICY_ID:
            if not args.placement_checkpoint or not args.placement_calibration:
                raise ValueError('rgbxy_pre_release_align_strict_release_dev_v4 requires checkpoint and calibration paths')
            from wmal.perception.rgb_xy_regressor import RGBBlockXYLocalizer
            placement_localizer = RGBBlockXYLocalizer(
                args.placement_checkpoint, device=args.placement_device,
                calibration_path=args.placement_calibration)
        elif args.placement_checkpoint or args.placement_calibration:
            raise ValueError('Placement artifacts require rgbxy_pre_release_align_strict_release_dev_v4')
        result = run_batch(config, args.output_dir, args.config,
                           recovery_policy=args.recovery_policy,
                           recovery_localizer=localizer,
                           recovery_max_control_steps=args.recovery_max_control_steps,
                           placement_policy=args.placement_policy,
                           placement_localizer=placement_localizer,
                           placement_max_control_steps=args.placement_max_control_steps,
                           placement_contact_loss_hold_steps=(
                               args.placement_contact_loss_hold_steps))
    except (ValueError, FileExistsError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: result[key] for key in
                      ('sample_count','success_count','failure_count','success_rate')}, sort_keys=True))
    return 0 if result['failure_count'] == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
