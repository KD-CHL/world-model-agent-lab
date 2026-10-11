"""Frozen test-only evaluation: physical predictions are not target-command gains."""
import math
import time

import numpy as np

from wmal.datasets.embodied_sequences import EmbodiedDataset, validate_lineage
from wmal.logging.manifest import sha256_file
from wmal.models.embodied_calibration import EmbodiedCalibration
from wmal.models.embodied_world import EmbodiedWorldModel
from wmal.models.embodied_rssm import parameter_report
from wmal.agents.prediction_contracts import allows_scope, known_commands
from .embodied_trainer import hardware_report


def _canonical_windows(dataset):
    """One deterministic first export per original group; never count re-exports twice."""
    selected = []
    for group, indices in sorted(dataset.episode_window_indices.items()):
        first = min({dataset.index[i][0] for i in indices},
                    key=lambda i: dataset.rows[i]['episode_id'])
        selected.extend((i, group) for i in indices if dataset.index[i][0] == first)
    return selected


def _features(physical, commands, actions, normalization):
    parts = [np.ones((*physical.shape[:-1], 1))]
    for values, name in ((physical, 'physical'), (commands, 'command'), (actions, 'action')):
        parts.append((values-np.asarray(normalization[name+'_mean'])) /
                     np.asarray(normalization[name+'_scale']))
    return np.concatenate(parts, axis=-1)


def _fit_linear(dataset, normalization):
    """Streaming train-only one-step least squares delta model, fixed before test reads."""
    physical_dim = len(dataset.profile.physical_order)
    width = 1+physical_dim+len(dataset.profile.command_order)+len(dataset.profile.action_order)
    gram, rhs = np.zeros((width, width)), np.zeros((width, physical_dim))
    count = 0
    for index, _ in _canonical_windows(dataset):
        sample = dataset[index]
        x = _features(sample['physical'][0].astype('float64'), sample['commands'][0].astype('float64'),
                      sample['actions'][0].astype('float64'), normalization)
        delta = sample['physical'][1].astype('float64')-sample['physical'][0]
        gram += np.outer(x, x); rhs += np.outer(x, delta); count += 1
    # Fixed tolerance, no selection/calibration/test hyperparameter tuning.
    coefficients = np.linalg.pinv(gram, rcond=1e-10) @ rhs
    return coefficients, dict(method='train-only normalized linear one-step delta; recursive rollout',
                              transitions=count, rank=int(np.linalg.matrix_rank(gram)), rcond=1e-10)


def _metrics(squared, count):
    mse = squared/count
    return dict(rmse_per_horizon_dim=np.sqrt(mse).tolist(),
                rmse_per_horizon=np.sqrt(mse.mean(-1)).tolist(),
                rmse_per_dim=np.sqrt(mse.mean(0)).tolist(), rmse=float(np.sqrt(mse.mean())))


def _wilson(success, total):
    z = 1.959963984540054; proportion = success/total
    denominator = 1+z*z/total
    center = (proportion+z*z/(2*total))/denominator
    radius = z*math.sqrt(proportion*(1-proportion)/total+z*z/(4*total*total))/denominator
    return [max(0., center-radius), min(1., center+radius)]


def evaluate_embodied(manifest, checkpoint, *, calibration=None, horizon=None, device='cpu'):
    """Evaluate a frozen S8/sample/seed0 protocol, no selection or online adaptation.

    Confidence uses original episodes, not overlapping windows. It assumes
    independent episodes and is descriptive, not an online safety guarantee.
    All test windows must be within a supplied calibration's declared scope.
    """
    model = EmbodiedWorldModel.load(checkpoint, device=device)
    if model.config.preset != 'smoke' and not parameter_report(model.config)['within_budget']:
        raise ValueError('Formal network preset is outside its declared parameter budget')
    h = model.config.horizon if horizon is None else horizon
    if type(h) is not int or not 1 <= h <= min(model.config.horizon, model.metadata.get('max_horizon', model.config.horizon)):
        raise ValueError('Evaluation horizon outside frozen model horizon')
    lineage = model.metadata.get('episode_lineage')
    if lineage is None:
        raise ValueError('Independent evaluation requires complete ancestor episode_lineage')
    data = EmbodiedDataset(manifest, 'test', horizon=h, context_steps=model.config.context_steps)
    model.validate_profile(data.profile)
    validate_lineage(data.rows, lineage, context='test evaluation')
    cal = None
    if calibration is not None:
        cal = EmbodiedCalibration.load(calibration)
        cal.validate(model, samples=8, mode='sample', profile=data.profile)
        # Classify interval-fitting origins as exposed, retaining every alias/hash.
        exposed = validate_lineage([dict(row, split='train') for row in cal.episode_records])
        validate_lineage(data.rows, exposed, context='test versus interval calibration')
        if h > cal.payload['bound_protocol']['horizon']:
            raise ValueError('Evaluation horizon outside frozen calibration')
    train = EmbodiedDataset(manifest, 'train', horizon=1)
    model.validate_profile(train.profile)
    coefficients, baseline = _fit_linear(train, model.normalization)
    windows = _canonical_windows(data)
    sums = {name: np.zeros((h, model.config.physical_dim)) for name in ('model', 'persistence', 'train_linear')}
    rgb = dict(model=0., persistence=0., model_changed=0., persistence_changed=0., pixels=0, changed=0)
    sensitivity = dict(zero=0., shuffled=0., frame_zero=0., frame_shuffled=0.)
    control_counts = {name: dict(evaluated_windows=0, out_of_scope_windows=0) for name in ('zero', 'shuffled')}
    width = np.zeros_like(sums['model']); joint_windows = 0; episodes = {}
    point_covered = np.zeros_like(width)
    events = {}; seed = 0
    threshold = model.metadata.get('training_sampling', {}).get('change_threshold', 1/255)
    if not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('Invalid frozen changed-pixel threshold')
    start = time.perf_counter()
    for index, group in windows:
        sample = data[index]; k = sample['prediction_offset']; low = k-sample['context_length']
        actions = sample['actions'][k:]
        if cal is not None and not cal.allows_scope(sample['physical'][low:k+1], sample['commands'][low:k+1], actions, model=model):
            raise ValueError('Test window outside frozen calibration scope; do not silently filter')
        prediction = model.predict_sample(sample, samples=8, mode='sample', seed=seed)
        mean = prediction['physical_states'].mean((0, 1)).astype('float64')
        target = sample['physical'][k+1:].astype('float64')
        origin = sample['physical'][k].astype('float64')
        linear = []; state = origin.copy(); command = sample['commands'][k].astype('float64')
        for step, action in enumerate(actions):
            state = state+_features(state, command, action, model.normalization) @ coefficients
            linear.append(state.copy())
            if model.profile.command_rule == 'joint_delta':
                command = prediction['known_commands'][step]
        for name, estimate in (('model', mean), ('persistence', origin[None]), ('train_linear', np.array(linear))):
            sums[name] += (estimate-target)**2
        frame = prediction['frames'].mean((0, 1)).astype('float64')
        truth = sample['rgb'][k+1:].astype('float64'); old = sample['rgb'][k].astype('float64')
        measured = np.broadcast_to(sample['view_mask'][k+1:, :, None, None], truth.shape[:2]+truth.shape[-2:])
        changed = (np.abs(truth-old).mean(-3) > threshold) & measured
        for name, estimate in (('model', frame), ('persistence', old[None])):
            error = ((estimate-truth)**2).mean(-3)
            rgb[name] += float(error[measured].sum()); rgb[name+'_changed'] += float(error[changed].sum())
        rgb['pixels'] += int(measured.sum()); rgb['changed'] += int(changed.sum())
        for name, alternate in (('zero', np.zeros_like(actions)), ('shuffled', actions[np.random.default_rng(0).permutation(h)])):
            future = known_commands(model.profile, sample['commands'][low:k+1], alternate)
            if not allows_scope(model.metadata.get('prediction_scope'), model.profile,
                    sample['physical'][low:k+1], sample['commands'][low:k+1], alternate, future):
                control_counts[name]['out_of_scope_windows'] += 1
                continue
            control = model.predict_sample(sample, actions=alternate, samples=8, mode='sample', seed=seed)
            control_counts[name]['evaluated_windows'] += 1
            sensitivity[name] += float(np.abs(control['physical_states'].mean((0, 1))-mean).mean())
            sensitivity['frame_'+name] += float(((control['frames'].mean((0, 1))-frame)**2).mean())
        if cal is not None:
            bound = cal.bounds(prediction, model=model)
            covered = np.abs(mean-target) <= bound
            point_covered += covered; width += 2*bound
            joint = bool(covered.all()); joint_windows += joint
            episodes[group] = episodes.get(group, True) and joint
        names = prediction['event_probability_names']
        if names:
            probability = prediction['event_probabilities'].mean((0, 1))
            for column, name in enumerate(names):
                original = model.profile.event_names.index(name); mask = sample['event_mask'][:, original]
                p, y = probability[:, column][mask], sample['events'][:, original][mask]
                if not len(y):
                    continue
                p = np.clip(p.astype('float64'), 1e-7, 1-1e-7)
                row = events.setdefault(name, dict(count=0, brier_sum=0., log_loss_sum=0.))
                row['count'] += len(y); row['brier_sum'] += float(((p-y)**2).sum())
                row['log_loss_sum'] += float((-y*np.log(p)-(1-y)*np.log(1-p)).sum())
    count = len(windows); model.assert_frozen()
    intervals = None
    if cal is not None:
        total, success = len(episodes), sum(episodes.values())
        intervals = dict(calibration_version=cal.version, calibration_episodes=cal.episode_count,
            independent_episodes=total, covered_episodes=success, joint_episode_coverage=success/total,
            joint_window_coverage=joint_windows/count, point_coverage_per_horizon_dim=(point_covered/count).tolist(),
            mean_full_width_per_horizon_dim=(width/count).tolist(), wilson_95=_wilson(success, total),
            formal_sample_size_met=total >= 100 and cal.episode_count >= 100,
            confidence_assumption='independent original episodes; all evaluated windows and physical coordinates jointly',
            bound_mode=cal.payload['bound_protocol']['bound_mode'], alpha=cal.payload['alpha'], scope=cal.scope)
    return dict(schema='wmal.embodied_evaluation.v1', manifest_sha256=sha256_file(data.path),
        checkpoint_sha256=sha256_file(checkpoint), model_version=model.version,
        protocol_hash=model.protocol_hash(samples=8, mode='sample'), protocol=dict(samples=8, mode='sample', seed=0, horizon=h),
        counts=dict(windows=count, original_episodes=len(data.episode_window_indices), exported_episodes=len(data.rows),
                    aggregation='window weighted, canonical lexicographic first export per original group'),
        physical=dict(order=list(model.profile.physical_order), units=list(model.profile.physical_units),
                      **{name: _metrics(value, count) for name, value in sums.items()}, train_linear_fit=baseline),
        commands=dict(order=list(model.profile.command_order), rule=model.profile.command_rule, scored_as_prediction_gain=False),
        rgb=dict(model_mse=rgb['model']/rgb['pixels'], persistence_mse=rgb['persistence']/rgb['pixels'],
            model_changed_mse=rgb['model_changed']/rgb['changed'] if rgb['changed'] else None,
            persistence_changed_mse=rgb['persistence_changed']/rgb['changed'] if rgb['changed'] else None,
            changed_pixel_fraction=rgb['changed']/rgb['pixels'], changed_pixels=rgb['changed'],
            measured_pixels=rgb['pixels'], threshold=threshold, units='RGB [0,1], channel-mean squared error'),
        sensitivity=dict(mean_physical_delta_zero_action=sensitivity['zero']/control_counts['zero']['evaluated_windows'] if control_counts['zero']['evaluated_windows'] else None,
            mean_physical_delta_shuffled_action=sensitivity['shuffled']/control_counts['shuffled']['evaluated_windows'] if control_counts['shuffled']['evaluated_windows'] else None,
            rgb_mse_delta_zero_action=sensitivity['frame_zero']/control_counts['zero']['evaluated_windows'] if control_counts['zero']['evaluated_windows'] else None,
            rgb_mse_delta_shuffled_action=sensitivity['frame_shuffled']/control_counts['shuffled']['evaluated_windows'] if control_counts['shuffled']['evaluated_windows'] else None,
            zero_action=control_counts['zero'], shuffled_action=control_counts['shuffled'],
            counterfactual_accuracy=False, method='same observations and common random seeds; permuted actions only, no branch ground truth'),
        events={name: dict(masked_transitions=row['count'], brier=row['brier_sum']/row['count'],
                          log_loss=row['log_loss_sum']/row['count']) for name, row in events.items()} or None,
        intervals=intervals, labels=dict(capacity='smoke only' if model.config.preset == 'smoke' else model.config.preset,
            parameters_per_member=sum(p.numel() for p in model.members[0].parameters()), independent_members=len(model.members),
            scene=model.metadata.get('training_source', {}).get('task'), profile=model.profile.to_dict(),
            prediction_scope=model.metadata.get('prediction_scope'), transfer=model.metadata.get('transfer'), gpu8_tested=False),
        hardware=hardware_report(device), evaluation_seconds=time.perf_counter()-start,
        limitations=['Test evaluated only after frozen selection; no evaluation-driven tuning',
            'No grasp, stack, locomotion or research-gain claim from joint-only G1 scene name',
            'Sample spread is not an independent ensemble or safety probability',
            'Canonical export avoids duplicate votes; overlapping windows are not independent trials',
            'Train-linear baseline is fit on this manifest train split only',
            'Eight-card/server capacity and physical robot deployment are unverified'])
