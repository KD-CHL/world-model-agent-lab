"""Evaluate held-out prediction error and MuJoCo goal control on probe assets."""
import argparse
import json
import random
from pathlib import Path
from time import monotonic
import numpy as np
from wmal.analysis.metrics import control_summary
from wmal.communication.contracts import Goal, MotionCommand
from wmal.logging.manifest import atomic_json, build_manifest, related_path, sha256_file
from wmal.models.loader import load_dynamics
from wmal.planners.world_planner import RolloutPlanner
from wmal.training.collector import build_backend
from wmal.training.trainer import load_rows, prediction_error
from wmal.envs.mujoco_env import MujocoEnvironment


class CountedModel:
    def __init__(self, model):
        self.model, self.version, self.calls = model, model.version, 0
        self.ensemble_size = getattr(model, 'ensemble_size', 1)

    def predict(self, *args):
        self.calls += 1
        return self.model.predict(*args)

    def predict_member(self, *args):
        self.calls += 1
        if hasattr(self.model, 'predict_member'):
            return self.model.predict_member(*args)
        return self.model.predict(*args[1:])


def evaluate_run(config, dataset, checkpoint, output, *, episodes=10, seed=100, experiment_id='joint_probe'):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    model = load_dynamics(checkpoint)
    train_report_path = related_path(checkpoint, 'train')
    training_seed = None
    if train_report_path.exists():
        train_report = json.loads(train_report_path.read_text())
        if train_report.get('checkpoint_sha256') != sha256_file(checkpoint):
            raise ValueError('Training report and checkpoint differ')
        training_seed = train_report.get('training_seed')
    test_rows = [row for row in load_rows(dataset) if row['split'] == 'test']
    test_rmse = prediction_error(model, test_rows)
    backend = build_backend(config)
    environment = MujocoEnvironment(backend, action_duration_s=0.5)
    if set(model.joints) != set(backend.profile.joint_limits):
        raise ValueError('Checkpoint does not match robot joints')
    counted = CountedModel(model)
    planner = RolloutPlanner(counted, samples=32, horizon=3, duration_s=0.5, seed=seed)
    generator = random.Random(seed)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    step_path = destination.with_name(destination.stem + '.steps.jsonl')
    config_path = Path(config).resolve()
    asset_path = (config_path.parent / json.loads(config_path.read_text())['model_path']).resolve()
    manifest_path = related_path(destination, 'manifest')
    atomic_json(manifest_path, build_manifest('mujoco_evaluation', seed,
                {'config': config_path, 'asset': asset_path, 'dataset': dataset, 'checkpoint': checkpoint},
                {'episodes': episodes, 'max_control_steps': 8, 'action_duration_s': 0.5,
                 'success_tolerance_rad': 0.03, 'planner_samples': 32, 'planner_horizon': 3},
                robot_id=backend.profile.robot_id, model_version=model.version,
                experiment_id=experiment_id,
                training_seed=training_seed, agent_calls=0,
                input_data_source='mujoco_interaction', observation_mode='joint_state'))
    results, step_events = [], []
    started = monotonic()
    state_flag = backend.mj.mjtState.mjSTATE_FULLPHYSICS

    def full_state():
        values = np.empty(backend.mj.mj_stateSize(backend.model, state_flag), dtype=float)
        backend.mj.mj_getState(backend.model, backend.data, values, state_flag)
        return values.tolist()

    with destination.open('w', buffering=1) as episode_file, step_path.open('w', buffering=1) as step_file:
        for episode in range(episodes):
            targets = {joint: generator.uniform(*bounds) for joint, bounds in backend.profile.joint_limits.items()}
            for method in ('direct', 'model_planner'):
                environment.reset(seed=seed + episode)
                steps = 0
                for cycle in range(8):
                    obs = backend.observe()
                    if max(abs(obs.joints[j] - target) for j, target in targets.items()) <= 0.03:
                        break
                    before_state = full_state()
                    planning_started = monotonic()
                    objective_cost = None
                    if method == 'model_planner':
                        counted.calls = 0
                        plan = planner.plan(backend.profile, obs, Goal('joint_goal', targets))
                        command = plan.commands[0]
                        objective_cost = plan.prediction.objective_cost
                        imagined_steps = counted.calls
                    else:
                        reach = backend.profile.max_joint_velocity_rad_s * 0.5
                        clamped = {j: max(obs.joints[j] - reach, min(obs.joints[j] + reach, target)) for j, target in targets.items()}
                        command = MotionCommand(f'eval_{episode}_{cycle}', backend.profile.robot_id,
                                                obs.episode_id, obs.step_id, 'joint_positions', clamped, 0.5)
                        imagined_steps = 0
                    planning_wall_s = monotonic() - planning_started
                    predicted_next = model.predict(obs.joints, command.values, command.duration_s)
                    execution_started = monotonic()
                    after, execution = environment.step(command)
                    physics_steps = execution['physics_steps']
                    applied_controls = execution['applied_controls']
                    execution_wall_s = monotonic() - execution_started
                    event = {'schema_version': 1, 'source': 'mujoco_interaction',
                             'method': method, 'evaluation_episode': episode, 'evaluation_seed': seed,
                             'robot_id': backend.profile.robot_id, 'episode_id': obs.episode_id,
                             'cycle': cycle, 'command_id': command.command_id,
                             'step_id': obs.step_id, 'next_step_id': after.step_id,
                             'sim_time_s': obs.sim_time_s, 'next_sim_time_s': after.sim_time_s,
                             'observation': obs.joints, 'task_target': targets,
                             'commanded_action': command.values, 'duration_s': command.duration_s,
                             'physics_steps': physics_steps, 'applied_controls': applied_controls,
                             'held_control_after_stop': backend.data.ctrl.tolist(),
                             'full_physics_state_before': before_state, 'full_physics_state_after': full_state(),
                             'predicted_next': predicted_next, 'observed_next': after.joints,
                             'prediction_error_rad': {j: predicted_next[j] - after.joints[j] for j in model.joints},
                             'model_version': model.version, 'objective_cost': objective_cost,
                             'imagined_model_steps': imagined_steps, 'diagnostic_model_calls': 1,
                             'planning_wall_s': planning_wall_s, 'execution_wall_s': execution_wall_s,
                             'termination_reason': 'action_duration_elapsed'}
                    step_events.append(event)
                    step_file.write(json.dumps(event, allow_nan=False) + '\n')
                    steps += 1
                final = backend.observe()
                error = max(abs(final.joints[j] - target) for j, target in targets.items())
                record = {'method': method, 'evaluation_episode': episode, 'seed': seed,
                          'episode_id': final.episode_id, 'model_version': model.version if method == 'model_planner' else None,
                          'target': targets, 'final': final.joints, 'control_steps': steps,
                          'success': error <= 0.03, 'termination_reason': 'goal_observed' if error <= 0.03 else 'control_budget_exhausted',
                          'final_error_rad': error, 'test_transition_rmse_rad': test_rmse}
                results.append(record)
                episode_file.write(json.dumps(record, allow_nan=False) + '\n')
                print(json.dumps({'phase': 'evaluation', 'episode': episode, 'method': method,
                                  'success': record['success'], 'control_steps': steps,
                                  'final_error_rad': error}, ensure_ascii=False), flush=True)
    summary = {'schema_version': 1, 'run_kind': 'mujoco_evaluation', 'experiment_id': experiment_id,
               'evaluation_seed': seed, 'training_seed': training_seed, 'model_version': model.version,
               'scenario_scope': 'joint_goal_probe', 'agent_api_calls': 0,
               'test_transition_rmse_rad': test_rmse,
               'test_transition_count': len(test_rows), 'wall_seconds': monotonic() - started,
               'methods': control_summary(results, step_events),
               'episode_path': str(destination), 'step_path': str(step_path),
               'manifest_path': str(manifest_path),
               'not_measured': ['long_task_success', 'skill_success', 'disturbance_recovery',
                                'uncertainty_calibration', 'ros_deadline_miss_rate'],
               'statistical_note': 'Single checkpoint and evaluation seed; no across-training-seed confidence interval.'}
    atomic_json(related_path(destination, 'summary'), summary)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--experiment-id', default='joint_probe')
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.config, args.dataset, args.checkpoint, args.output,
                                  episodes=args.episodes, seed=args.seed,
                                  experiment_id=args.experiment_id), ensure_ascii=False))

if __name__ == "__main__":
    main()
