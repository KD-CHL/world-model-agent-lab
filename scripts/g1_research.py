"""Collect, train/adapt and evaluate G1 policy-conditioned world models."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from wmal.logging.manifest import atomic_json, sha256_file
from wmal.locomotion.contracts import G1State, G1VelocityAction, G1Goal
from wmal.locomotion.learned import LearnedG1Dynamics, load


def state(value):
    return G1State(**{**value, 'contacts': tuple(value.get('contacts', []))})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['collect', 'train', 'evaluate'])
    parser.add_argument('--dataset', default='data/g1_locomotion/transitions.json')
    parser.add_argument('--checkpoint', default='runs/g1_research/model.json')
    parser.add_argument('--prior', help='Existing checkpoint to adapt with ridge regularization')
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--feedback', action='store_true')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--optimize', action='store_true', help='Select ridge/normalization using validation rollouts')
    args = parser.parse_args(argv)
    if args.episodes < 3 or args.steps < 1:
        parser.error('Need >=3 episodes and positive steps')
    if args.mode == 'collect':
        from wmal.envs.g1_session import G1MuJoCoSession
        rng, rows, failures = np.random.default_rng(args.seed), [], []
        for episode in range(args.episodes):
            split = ('validation' if episode == args.episodes - 2 else
                     'test' if episode == args.episodes - 1 else 'train')
            with G1MuJoCoSession(viewer=args.viewer, realtime=args.viewer) as session:
                for _ in range(args.steps):
                    before = session.observe()
                    action = G1VelocityAction(float(rng.uniform(-0.2, 0.35)),
                                              float(rng.uniform(-0.1, 0.1)),
                                              float(rng.uniform(-0.3, 0.3)), 0.5)
                    try:
                        after = session.step(action)
                    except RuntimeError as exc:
                        failures.append({'episode': episode, 'reason': str(exc)})
                        break
                    rows.append({'split': split, 'before': asdict(before),
                                 'action': asdict(action), 'after': asdict(after)})
            atomic_json(args.dataset, {'schema': 'wmal.g1.transitions.v1', 'seed': args.seed,
                                        'rows': rows, 'failed_episodes': failures})
        print(json.dumps({'transitions': len(rows), 'failed_episodes': len(failures)}))
    elif args.mode == 'train':
        payload = json.loads(Path(args.dataset).read_text())
        if payload['schema'] != 'wmal.g1.transitions.v1':
            raise ValueError('Incompatible dataset: G1 base states and body velocities required')
        splits = {name: [] for name in ('train', 'validation', 'test')}
        episodes = {}
        for row in payload['rows']:
            before, after, action = state(row['before']), state(row['after']), G1VelocityAction(**row['action'])
            if (before.episode_id != after.episode_id or after.step_id != before.step_id + 1
                    or abs(after.sim_time_s - before.sim_time_s - action.duration_s) > 1e-6):
                raise ValueError('Invalid transition provenance')
            if episodes.setdefault(before.episode_id, row['split']) != row['split']:
                raise ValueError('Episode leakage between splits')
            splits[row['split']].append((before, action, after))
        if any(not rows for rows in splits.values()):
            raise ValueError('Train, validation and test episodes are required')
        prior = load({'checkpoint': args.prior}) if args.prior else None
        selection = None
        if args.optimize:
            from wmal.training.g1_selection import select_model
            model, selection = select_model(splits['train'], splits['validation'], seed=args.seed, prior=prior)
        else:
            model = LearnedG1Dynamics.fit(splits['train'], seed=args.seed, prior=prior)
        metrics = {}
        for split in ('validation', 'test'):
            errors = []
            for before, action, after in splits[split]:
                pred = model.predict(before, action, action.duration_s).state
                errors.append((pred.x - after.x)**2 + (pred.y - after.y)**2)
            metrics[split + '_position_rmse_m'] = float(np.sqrt(np.mean(errors)))
        model.save(args.checkpoint)
        report = {'model_version': model.version, 'seed': args.seed, 'metrics': metrics,
                  'selection': selection,
                  'dataset_sha256': sha256_file(args.dataset),
                  'prior_sha256': sha256_file(args.prior) if args.prior else None,
                  'transitions': {key: len(value) for key, value in splits.items()}}
        if args.optimize:
            from wmal.training.g1_selection import rollout_metrics
            report['test_three_step'] = rollout_metrics(model, splits['test'])
        atomic_json(args.checkpoint + '.report.json', report)
        print(json.dumps(report, indent=2))
    else:
        from wmal.envs.g1_session import G1MuJoCoSession
        from wmal.locomotion.agent import G1Agent
        from wmal.locomotion.planner import G1RolloutPlanner
        from wmal.locomotion.feedback import ResidualFeedback
        from wmal.logging.events import EventLog
        model = load({'checkpoint': args.checkpoint})
        planner = G1RolloutPlanner(model, seed=args.seed, action_duration_s=model.duration_s)
        log = EventLog('runs/g1_research/evaluation.jsonl')
        agent = G1Agent(planner, feedback=ResidualFeedback() if args.feedback else None, log=log)
        start = time.perf_counter()
        with G1MuJoCoSession(viewer=args.viewer, realtime=args.viewer) as session:
            initial = session.observe()
            result = agent.run_goal(G1Goal(initial.x + 1, initial.y), session, max_cycles=args.steps)
        report = {**asdict(result), 'elapsed_s': time.perf_counter() - start,
                  'feedback': args.feedback, 'seed': args.seed, 'model_version': model.version}
        log('evaluation_result', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
