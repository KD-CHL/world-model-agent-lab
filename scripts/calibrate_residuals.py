"""Fit an empirical residual alarm from held-out G1 transitions or JSONL."""
import argparse
import json
import math
from pathlib import Path
from wmal.models.calibration import fit_residual_calibration


def dataset_residuals(path, checkpoint, factory='wmal.locomotion.learned:load', device='cpu'):
    from wmal.communication.g1_session_protocol import state_from_wire
    from wmal.locomotion.contracts import G1VelocityAction
    from wmal.locomotion.world_model import load_world_model
    from wmal.logging.manifest import sha256_file
    payload = json.loads(Path(path).read_text())
    if payload.get('schema') != 'wmal.g1.transitions.v1':
        raise ValueError('Invalid transition dataset')
    model = load_world_model(factory, {'checkpoint': str(checkpoint), 'device': device})
    assignments, transitions = {}, []
    for row in payload['rows']:
        before, after = state_from_wire(row['before']), state_from_wire(row['after'])
        action = G1VelocityAction(**row['action'])
        split = row['split']
        if (split not in ('train','validation','test')
                or assignments.setdefault(before.episode_id, split) != split
                or before.episode_id != after.episode_id or after.step_id != before.step_id+1
                or abs(after.sim_time_s-before.sim_time_s-action.duration_s) > 1e-6):
            raise ValueError('Invalid transition or episode split leakage')
        if split == 'validation':
            transitions.append((before, action, after))
    training_ids = set(getattr(model.model, 'metadata', {}).get('training_episodes', []))
    if any(before.episode_id in training_ids for before, _, _ in transitions):
        raise ValueError('Calibration episodes were used for training this checkpoint')
    rows = []
    for before, action, after in transitions:
        prediction = model.predict(before, action, action.duration_s).state
        rows.append({'episode_id': before.episode_id, 'split':'validation',
                     'model_version': model.version, 'duration_s': action.duration_s,
                     'position_error_m': math.hypot(prediction.x-after.x, prediction.y-after.y)})
    excluded = [episode for episode, split in assignments.items() if split != 'validation']
    return rows, excluded, model.version, model.model.duration_s, sha256_file(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', help='Residual JSONL with explicit validation split')
    source.add_argument('--dataset', help='Existing wmal.g1.transitions.v1 dataset')
    parser.add_argument('--checkpoint', help='G1 prediction checkpoint used with --dataset')
    parser.add_argument('--factory', default='wmal.locomotion.learned:load')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-version')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--dataset-id')
    parser.add_argument('--quantile', type=float, default=.95)
    parser.add_argument('--excluded-episodes', help='Training/test episode IDs JSON; required for --input')
    args = parser.parse_args(argv)
    if args.dataset:
        if not args.checkpoint:
            parser.error('--dataset requires --checkpoint')
        rows, excluded, version, duration, dataset_id = dataset_residuals(args.dataset, args.checkpoint, args.factory, args.device)
    else:
        if any(value is None for value in (args.model_version,args.duration,args.dataset_id,args.excluded_episodes)):
            parser.error('--input requires model-version, duration, dataset-id and excluded-episodes')
        rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
        excluded = json.loads(Path(args.excluded_episodes).read_text())
        if not isinstance(excluded, list) or any(not isinstance(e, str) for e in excluded):
            parser.error('excluded-episodes must contain a JSON string list')
        version, duration, dataset_id = args.model_version, args.duration, args.dataset_id
    calibration = fit_residual_calibration(rows, model_version=version, duration_s=duration,
        dataset_id=dataset_id, quantile=args.quantile, excluded_episode_ids=excluded)
    calibration.save(args.output)
    print(json.dumps({'threshold_m': calibration.threshold_m,
                      'validation_episodes': len(calibration.episode_ids),
                      'semantics': 'empirical_episode_max_residual_quantile'}))


if __name__ == '__main__':
    main()
