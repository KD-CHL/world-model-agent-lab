"""Train the optional nonlinear transition ensemble on episode-separated data."""
import math

from wmal.logging.manifest import atomic_json, build_manifest, related_path, sha256_file
from wmal.models.neural_dynamics import NeuralJointDynamics
from wmal.training.trainer import load_rows


def _rmse(model, rows):
    if not rows:
        raise ValueError('Empty evaluation split')
    squared, count = 0.0, 0
    for row in rows:
        predicted = model.predict(row['before'], row['target'], row['duration_s'])
        for joint, actual in row['after'].items():
            squared += (predicted[joint] - actual) ** 2
            count += 1
    return math.sqrt(squared / count)


def train_neural(dataset, checkpoint, *, members=5, epochs=100, batch_size=256,
                 learning_rate=3e-4, seed=0, device='cpu'):
    rows = load_rows(dataset)
    splits = {name: [row for row in rows if row['split'] == name]
              for name in ('train', 'validation', 'test')}
    if any(not values for values in splits.values()):
        raise ValueError('All episode-separated data splits are required')
    joints = tuple(sorted(splits['train'][0]['before']))
    model, member_metrics = NeuralJointDynamics.fit(
        splits['train'], joints, validation_rows=splits['validation'], members=members,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        seed=seed, device=device)
    validation_rmse = _rmse(model, splits['validation'])
    model.save(checkpoint)
    report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint),
              'model_version': model.version, 'training_seed': seed, 'device': device,
              'train_transitions': len(splits['train']),
              'validation_transitions': len(splits['validation']),
              'test_transitions_reserved': len(splits['test']),
              'validation_rmse_rad': validation_rmse, 'members': member_metrics}
    atomic_json(related_path(checkpoint, 'train'), {
        **build_manifest('neural_dynamics_training', seed, {'dataset': dataset},
                         {'members': members, 'epochs': epochs, 'batch_size': batch_size,
                          'learning_rate': learning_rate, 'joint_order': joints, 'device': device}),
        **report})
    return report
