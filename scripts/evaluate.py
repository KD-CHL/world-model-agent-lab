"""Evaluate held-out prediction error and MuJoCo goal control on probe assets."""
import argparse
import json
import random
from pathlib import Path
from wmal.communication.contracts import Goal, MotionCommand
from wmal.models.latent_dynamics import JointDynamics
from wmal.planners.world_planner import RolloutPlanner
from wmal.training.collector import build_backend
from wmal.training.trainer import load_rows, prediction_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    model = JointDynamics.load(args.checkpoint)
    test_rows = [row for row in load_rows(args.dataset) if row['split'] == 'test']
    test_rmse = prediction_error(model, test_rows)
    backend = build_backend(args.config)
    if set(model.joints) != set(backend.profile.joint_limits):
        raise ValueError('Checkpoint does not match robot joints')
    planner = RolloutPlanner(model, samples=32, horizon=3, duration_s=0.5, seed=args.seed)
    generator = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with output.open('w') as handle:
        for episode in range(args.episodes):
            targets = {joint: generator.uniform(*bounds) for joint, bounds in backend.profile.joint_limits.items()}
            for method in ('direct', 'model_planner'):
                backend.reset()
                steps = 0
                for cycle in range(8):
                    obs = backend.observe()
                    if max(abs(obs.joints[j] - target) for j, target in targets.items()) <= 0.03:
                        break
                    if method == 'model_planner':
                        command = planner.plan(backend.profile, obs, Goal('joint_goal', targets)).commands[0]
                    else:
                        reach = backend.profile.max_joint_velocity_rad_s * 0.5
                        clamped = {j: max(obs.joints[j] - reach, min(obs.joints[j] + reach, target)) for j, target in targets.items()}
                        command = MotionCommand(f'eval_{episode}_{cycle}', backend.profile.robot_id,
                                                obs.episode_id, obs.step_id, 'joint_positions', clamped, 0.5)
                    backend.begin(command)
                    for _ in range(round(0.5 / backend.model.opt.timestep)):
                        backend.step()
                    backend.stop()
                    steps += 1
                final = backend.observe()
                error = max(abs(final.joints[j] - target) for j, target in targets.items())
                record = {'method': method, 'evaluation_episode': episode, 'seed': args.seed,
                          'model_version': model.version if method == 'model_planner' else None,
                          'target': targets, 'final': final.joints, 'control_steps': steps,
                          'success': error <= 0.03, 'final_error_rad': error,
                          'test_transition_rmse_rad': test_rmse}
                results.append(record)
                handle.write(json.dumps(record, allow_nan=False) + '\n')
    print(json.dumps({'test_transition_rmse_rad': test_rmse,
                      'episodes_per_method': args.episodes,
                      'successes': {method: sum(row['success'] for row in results if row['method'] == method)
                                    for method in ('direct', 'model_planner')},
                      'output': str(output)}))

if __name__ == "__main__":
    main()
