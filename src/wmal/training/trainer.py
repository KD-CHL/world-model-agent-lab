"""Train and validate a small dynamics model from separated episodes."""
from pathlib import Path
import json
import math

from wmal.models.latent_dynamics import JointDynamics
from wmal.logging.manifest import atomic_json, build_manifest, related_path, sha256_file


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(row.get('source') != 'mujoco_interaction' for row in rows):
        raise ValueError('Expected MuJoCo interaction records')
    episode_splits = {}
    for row in rows:
        episode = row['episode_id']
        split = row['split']
        if split not in ('train', 'validation', 'test') or (episode in episode_splits and episode_splits[episode] != split):
            raise ValueError('Episode split leakage')
        episode_splits[episode] = split
    return rows


def prediction_error(model, rows):
    if not rows:
        raise ValueError('Empty evaluation split')
    errors = []
    for row in rows:
        predicted = model.predict(row['before'], row['target'], row['duration_s'])
        errors.extend((predicted[j] - row['after'][j]) ** 2 for j in model.joints)
    return math.sqrt(sum(errors) / len(errors))


def train(dataset, checkpoint, *, members=5, seed=0):
    rows = load_rows(dataset)
    splits = {name: [row for row in rows if row['split'] == name] for name in ('train', 'validation', 'test')}
    if any(not split for split in splits.values()):
        raise ValueError('All three episode splits are required')
    joints = tuple(sorted(splits['train'][0]['before']))
    model = JointDynamics.fit(splits['train'], joints, members=members, seed=seed,
                              version=f'joint-dynamics-seed{seed}-n{len(splits["train"])}')
    validation_rmse = prediction_error(model, splits['validation'])
    model.save(checkpoint)
    report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint),
            'model_version': model.version, 'training_seed': seed,
            'train_transitions': len(splits['train']), 'validation_transitions': len(splits['validation']),
            'test_transitions_reserved': len(splits['test']), 'validation_rmse_rad': validation_rmse}
    atomic_json(related_path(checkpoint, 'train'), {
        **build_manifest('dynamics_training', seed, {'dataset': dataset},
                         {'members': members, 'ridge': 1e-3, 'joint_order': joints}),
        **report})
    return report
