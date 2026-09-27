"""Validation-only model selection using episode-bounded multistep rollouts."""
import math
from wmal.locomotion.learned import LearnedG1Dynamics, wrap


def rollout_metrics(model, rows, horizon=3):
    groups = {}
    for row in rows:
        groups.setdefault(row[0].episode_id, []).append(row)
    position, heading = [], []
    for episode in groups.values():
        episode.sort(key=lambda row: row[0].step_id)
        for start in range(len(episode) - horizon + 1):
            window = episode[start:start+horizon]
            if any(a[2] != b[0] for a, b in zip(window, window[1:])):
                raise ValueError('Noncontiguous episode transitions')
            prediction = window[0][0]
            for _, action, _ in window:
                prediction = model.predict(prediction, action, action.duration_s).state
            actual = window[-1][2]
            position.append((prediction.x-actual.x)**2 + (prediction.y-actual.y)**2)
            heading.append(wrap(prediction.yaw-actual.yaw)**2)
    if not position:
        raise ValueError('No complete evaluation windows')
    return {'position_rmse_m': math.sqrt(sum(position)/len(position)),
            'yaw_rmse_rad': math.sqrt(sum(heading)/len(heading)), 'windows': len(position)}


def select_model(train, validation, *, seed=0, prior=None):
    if {r[0].episode_id for r in train} & {r[0].episode_id for r in validation}:
        raise ValueError('Episode leakage')
    best, reports = None, []
    for normalize in (False, True):
        for ridge in (.001, .01, .1, 1.):
            model = LearnedG1Dynamics.fit(train, seed=seed, prior=prior, ridge=ridge,
                                         normalize=normalize, episode_bootstrap=True)
            metrics = rollout_metrics(model, validation)
            score = metrics['position_rmse_m'] + .2 * metrics['yaw_rmse_rad']
            reports.append({'normalize': normalize, 'ridge': ridge, 'score': score, **metrics})
            if best is None or score < best[0]:
                best = score, model, reports[-1]
    return best[1], {'selected': best[2], 'candidates': reports, 'selection_horizon': 3}
