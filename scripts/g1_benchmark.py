"""Run reproducible G1 missions with actual MuJoCo and learned-model planning."""
import argparse
import json
from pathlib import Path
import platform

import numpy as np

from wmal.envs.indoor_scene import IndoorScene, Box
from wmal.envs.g1_session import G1MuJoCoSession
from wmal.locomotion.agent import G1Agent
from wmal.locomotion.contracts import G1Goal
from wmal.locomotion.feedback import ResidualFeedback
from wmal.locomotion.learned import load
from wmal.locomotion.mission import MissionAgent
from wmal.locomotion.navigation import NavigationPlanner
from wmal.locomotion.planner import G1RolloutPlanner
from wmal.logging.events import EventLog
from wmal.logging.manifest import atomic_json, sha256_file


def make_scene(name):
    scene = IndoorScene()
    if name == 'open':
        scene.boxes = tuple(b for b in scene.boxes if b.name.startswith('wall_'))
    elif name == 'corridor':
        scene.boxes = tuple(b for b in scene.boxes if b.name.startswith('wall_')) + (
            Box('aisle_left', 2.5, 1., 1., .25, 1.),
            Box('aisle_right', 2.5, -1., 1., .25, 1.))
    elif name != 'indoor':
        raise ValueError('Unknown scene')
    return scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='configs/g1_experiments.json')
    parser.add_argument('--checkpoint', default='runs/g1_research/model.json')
    parser.add_argument('--output', required=True, help='New directory; existing results are not overwritten')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--experiments', nargs='+', help='IDs to run, default all ten')
    parser.add_argument('--no-feedback', action='store_true')
    parser.add_argument('--viewer', action='store_true')
    args = parser.parse_args(argv)
    suite = json.loads(Path(args.suite).read_text())
    if suite.get('schema') != 'wmal.g1.experiments.v1':
        raise ValueError('Unknown experiment schema')
    experiments = suite['experiments']
    if args.experiments:
        if set(args.experiments) - {e['id'] for e in experiments}:
            raise ValueError('Unknown experiment ID')
        experiments = [e for e in experiments if e['id'] in args.experiments]
    # Validate targets and geometric feasibility before launching simulations.
    for exp in experiments:
        scene, previous = make_scene(exp['scene']), (0., 0.)
        scene.robot_radius += .15
        for coordinates in exp['goals']:
            goal = G1Goal(*coordinates)
            scene.route(previous, (goal.x, goal.y))
            previous = goal.x, goal.y
    model = load({'checkpoint': args.checkpoint})
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    results = []
    for exp in experiments:
        for seed in dict.fromkeys(args.seeds):
            latencies, errors = [], []
            event_log = EventLog(destination / f"{exp['id']}-seed{seed}.jsonl")
            def log(event, payload):
                event_log(event, payload)
                if event == 'plan':
                    latencies.append(payload['planning_latency_s'])
                if event == 'prediction_residual':
                    errors.append(payload['position_error_m'])
            scene = make_scene(exp['scene'])
            local = G1RolloutPlanner(model, samples=96, seed=seed,
                                    action_duration_s=model.duration_s,
                                    max_linear_velocity=.3, max_yaw_rate=.3)
            planner = NavigationPlanner(local, scene, log)
            agent = G1Agent(planner, log=log, feedback=None if args.no_feedback else ResidualFeedback())
            try:
                with G1MuJoCoSession(viewer=args.viewer, realtime=args.viewer, scene=scene) as session:
                    result = MissionAgent(agent, log).run([G1Goal(*g) for g in exp['goals']],
                                                          session, exp['cycles'])
            except (ValueError, RuntimeError, OSError) as exc:
                result = {'status': 'failed', 'error_type': type(exc).__name__, 'detail': str(exc)}
                log('experiment_failure', result)
            result.update(experiment=exp['id'], seed=seed,
                          planning_latency_mean_s=float(np.mean(latencies)) if latencies else None,
                          planning_latency_p95_s=float(np.percentile(latencies, 95)) if latencies else None,
                          prediction_rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None)
            results.append(result)
            atomic_json(destination / 'results.json', {
                'checkpoint_sha256': sha256_file(args.checkpoint), 'suite_sha256': sha256_file(args.suite),
                'model_version': model.version, 'python': platform.python_version(),
                'feedback': not args.no_feedback, 'results': results,
                'success_rate': sum(r['status'] == 'succeeded' for r in results)/len(results)})
            print(json.dumps(result), flush=True)
    return 0 if all(r['status'] == 'succeeded' for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
