"""Run the fixed no-world-model stack-block skill baseline."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from wmal.envs.manipulation_session import ManipulationSession
from wmal.envs.g1_locomotion import MODEL_XML
from wmal.skills.manipulation_contracts import SKILL_ORDER, StackBlockSkills, default_waypoint_profile
from wmal.skills.manipulation_verifier import StackBlockVerifier


_GENERIC_SKILL_FAILURES = {'independent_predicate_not_passed'}
_PRIMARY_FAILURE_PREDICATES = {
    'grasp_block': ('grasped',),
    'lift_block': ('lifted',),
    'move_to_stack_target': ('lifted',),
    'place_block': ('placed', 'stacked'),
    'retreat': ('stacked',),
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


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


def run_episode(config, seed, run_dir, *, render=False, render_image_size=32,
                on_reset=None, on_step=None, recovery_policy='none'):
    if recovery_policy not in ('none', 'repeat_place_once'):
        raise ValueError('Unsupported recovery policy')
    session = ManipulationSession(render=render, image_size=render_image_size,
                                  control_period_s=config['control_period_s'],
                                  max_episode_steps=config['max_episode_steps'])
    try:
        observation, audit = session.reset(seed, config['initial_state_profile'])
        initial_observation = observation.to_agent_dict()
        verifier = StackBlockVerifier(**config['verifier'])
        baseline_truth = session.truth_for_verifier()
        if on_reset is not None:
            on_reset(observation, baseline_truth)
        verifier.episode_id = baseline_truth.episode_id
        verifier.last_step_id = baseline_truth.step_id
        verifier.initial_z = baseline_truth.object('block_0').position[2]
        skills = StackBlockSkills(default_waypoint_profile())
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

                skill_run = skills.execute(
                    session, command, config['skill_budget_control_steps'],
                    timeout_s=config['skill_timeout_s'], verifier=verify_receipts)
                observation = skill_run.final_observation
                event = {
                    'episode_id': observation.episode_id, 'seed': seed,
                    'skill': skill_run.name, 'status': skill_run.status,
                    'control_steps': skill_run.control_steps,
                    'observation_final': observation.to_agent_dict(),
                    'skill_evidence': [item.__dict__ if hasattr(item, '__dict__') else item
                                       for item in skill_run.evidence],
                    'start_step_id': command.expected_step_id,
                    'end_step_id': observation.step_id,
                    'action_ids': [r.action_id for r in skill_run.receipts],
                    'action_trace': [
                        {
                            'action_id': receipt.action_id,
                            'start_step_id': receipt.start_step_id,
                            'end_step_id': receipt.end_step_id,
                            'control_steps': receipt.executed_steps,
                            'sim_duration_s': receipt.sim_duration_s,
                            'joint_targets': receipt.joint_targets,
                            'gripper_mode': receipt.gripper_mode,
                            'gripper_opening_m': receipt.gripper_opening_m,
                            'gripper_speed_m_s': receipt.gripper_speed_m_s,
                            'gripper_force_limit_n': receipt.gripper_force_limit_n,
                            'final_observation': (receipt.observations[-1].to_agent_dict()
                                                  if receipt.observations else None),
                            'final_truth': {
                                obj.name: {
                                    'position': obj.position,
                                    'quaternion_wxyz': obj.quaternion_wxyz,
                                    'linear_velocity': obj.linear_velocity,
                                    'angular_velocity': obj.angular_velocity,
                                }
                                for obj in session.truth_trace_for_verifier(receipt)[-1].object_states
                            } if session.truth_trace_for_verifier(receipt) else None,
                        }
                        for receipt in skill_run.receipts
                    ],
                    'failure': skill_run.failure,
                    'predicates': [report.__dict__ for report in verifier_rows],
                    'recovery_policy': recovery_policy,
                    'recovery_trigger': (recovery_policy == 'repeat_place_once'
                                         and command.name == 'place_block'
                                         and skill_run.status == 'success_unverified'
                                         and recovery_attempts == 0),
                    'is_recovery_action': recovery_attempts > 0,
                }
                if on_step is not None:
                    for receipt in skill_run.receipts:
                        snapshots = session.truth_trace_for_verifier(receipt)
                        if len(snapshots) != len(receipt.observations):
                            raise RuntimeError('Observation/truth trace length mismatch')
                        for step_observation, truth in zip(receipt.observations, snapshots):
                            on_step(step_observation, receipt, truth)
                events.write(json.dumps(event, sort_keys=True) + '\n')
                results.append(event)
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
            cause = None if stacked else resolve_failure_cause(verifier.failure_reason, results)
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
                'status': 'success' if stacked else 'failure', 'success': stacked,
                'recovery_policy': recovery_policy,
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
                    'contacts': session.truth_for_verifier().contacts,
                    'gripper': observation.gripper.__dict__,
                },
                'stage_events': results,
            }
    finally:
        session.close()


def run_batch(config, output_dir, config_path=None):
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
            'src/wmal/skills/manipulation_verifier.py')},
        'robot_asset_sha256': digest(MODEL_XML),
        'world_model_loaded': False,
        'gripper_asset': 'synthetic_parallel_jaw_proxy',
        'gripper_is_dex1': False,
        'controller': 'G1 position actuators with bounded per-control-step joint slew',
        'verifier_version': StackBlockVerifier.VERSION,
        'seed_count': len(config['seeds']),
    }
    (output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    (output_dir / 'events.jsonl').write_text('', encoding='utf-8')
    episodes = []
    for seed in config['seeds']:
        episodes.append(run_episode(config, seed, output_dir))
    successes = sum(bool(x['success']) for x in episodes)
    result = {
        'schema': 'wmal.stack_block_results.v1', 'task': config['task'],
        'world_model': 'disabled', 'sample_count': len(episodes),
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
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        result = run_batch(config, args.output_dir, args.config)
    except (ValueError, FileExistsError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: result[key] for key in
                      ('sample_count','success_count','failure_count','success_rate')}, sort_keys=True))
    return 0 if result['failure_count'] == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
