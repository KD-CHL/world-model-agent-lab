"""Shared consequence ranking; has no execution or task-completion authority."""
import numpy as np


def select_candidates(predictor, observation, candidates, target, *, baseline,
                      calibration, error_budget, scale, remaining,
                      state_lower, state_upper, history=None):
    ranked, evidence = [], []
    for index, candidate in enumerate(candidates):
        actions = np.asarray(candidate.actions)[:remaining]
        mean, frames, bounds, diagnostics = None, None, None, None
        prefix = len(actions)
        score = float(index)
        if baseline == 'A1':
            if candidate.nominal_terminal is None:
                raise ValueError('A1 requires an explicit controller-provided nominal terminal state')
            terminal = np.asarray(candidate.nominal_terminal)
            if terminal.shape != target.shape or not np.isfinite(terminal).all():
                raise ValueError('Invalid nominal outcome')
            score = float(np.linalg.norm((terminal-target)/scale))
        elif baseline in ('A2','A3'):
            if history is not None:
                result = predictor.predict_context(*history.inputs(), actions.copy())
            else:
                result = predictor.predict(observation.rgb.copy(), observation.state.copy(), actions.copy())
            values, video = np.asarray(result['states']), np.asarray(result['frames'])
            if (result['model_version'] != predictor.version or values.ndim != 3
                    or values.shape[1:] != (len(actions),len(target)) or len(values) < 2
                    or video.ndim != 5 or video.shape[:2] != values.shape[:2]
                    or video.shape[2:] != observation.rgb.shape
                    or not np.isfinite(values).all() or not np.isfinite(video).all()):
                raise ValueError('Malformed/stale world-model response')
            mean, frames = values.mean(0), video.mean(0)
            diagnostics = result.get('diagnostics')
            if baseline == 'A3':
                calibration.validate_for(predictor.version, predictor.semantics)
                bounds = calibration.bounds(values.std(0),version=predictor.version)
                trusted = (bounds/scale).max(-1) <= error_budget
                if state_lower is not None:
                    trusted &= np.all((mean-bounds >= state_lower) & (mean+bounds <= state_upper),axis=-1)
                prefix = int(np.cumprod(trusted).sum())
            elif state_lower is not None:
                if np.any(values < state_lower) or np.any(values > state_upper):
                    prefix = 0
            if prefix:
                score = float(np.linalg.norm((mean[prefix-1]-target)/scale))
                if bounds is not None:
                    score += float(np.linalg.norm(bounds[prefix-1]/scale))
        evidence.append({'candidate':index, 'skill':candidate.skill, 'trusted_prefix':prefix,
                         'score':score if prefix else None})
        if prefix:
            ranked.append((score,index,candidate,actions,prefix,mean,frames,bounds,diagnostics))
    return ranked, evidence
