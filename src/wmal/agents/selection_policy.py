"""Shared consequence ranking; has no execution or task-completion authority."""
import numpy as np


def select_candidates(predictor, observation, candidates, target, *, baseline,
                      calibration, error_budget, scale, remaining,
                      state_lower, state_upper, history=None, joint_target=None):
    """Optional task commands gate eligibility; measured outcomes remain learned.

    A0 and callers omitting joint_target retain the legacy selection policy.
    No deterministic command calculation changes predictions or trust bounds.
    """
    task_commands = joint_target is not None and baseline != 'A0'
    if task_commands:
        joint_target = np.asarray(joint_target, dtype=float)
        if (joint_target.shape != (2,) or np.asarray(observation.state).shape != (4,)
                or np.asarray(target).shape != (4,) or not np.isfinite(joint_target).all()):
            raise ValueError('Task joint ranking requires two commands and four state values')
        command_start = observation.state[-2:]
        initial_error = float(np.linalg.norm(command_start-joint_target))
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
            if baseline=='A3' and hasattr(calibration,'allows_scope'):
                context_state=history.inputs()[1] if history is not None else observation.state[None]
                if not calibration.allows_scope(context_state,actions):
                    evidence.append({'candidate':index,'skill':candidate.skill,'trusted_prefix':0,
                                     'score':None,'reason':'outside_calibration_scope'})
                    continue
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
        entry = {'candidate':index, 'skill':candidate.skill, 'trusted_prefix':prefix,
                 'score':score if prefix else None}
        if task_commands:
            if actions.ndim != 2 or actions.shape[1] != 2 or not np.isfinite(actions).all():
                raise ValueError('Task joint ranking requires finite two-joint action deltas')
            commands = command_start + np.cumsum(actions, axis=0)
            residuals = commands-joint_target
            prior = np.concatenate([(command_start-joint_target)[None], residuals[:-1]])
            # Coordinate-wise monotonic progress also forbids overshooting the goal.
            feasible = np.all((np.abs(residuals) <= np.abs(prior)+1e-8)
                              & (residuals*prior >= -1e-12), axis=-1)
            if state_lower is not None:
                feasible &= np.all((commands >= np.asarray(state_lower)[-2:]-1e-8)
                                   & (commands <= np.asarray(state_upper)[-2:]+1e-8), axis=-1)
            command_prefix = int(np.cumprod(feasible).sum())
            learned_prefix = prefix
            prefix = min(prefix, command_prefix)
            command_errors = np.linalg.norm(residuals, axis=-1)
            terminal_error = float(command_errors[prefix-1]) if prefix else None
            if prefix:
                if baseline == 'A1':
                    # The controller's full-horizon nominal is not a sliced endpoint.
                    score = float(np.linalg.norm((np.tile(commands[prefix-1],2)-target)/scale))
                else:
                    score = float(np.linalg.norm((mean[prefix-1,:2]-target[:2])/scale[:2]))
                    if bounds is not None:
                        score += float(np.linalg.norm(bounds[prefix-1,:2]/scale[:2]))
            entry.update(trusted_prefix=prefix, score=score if prefix else None,
                         learned_trusted_prefix=learned_prefix,
                         command_trajectory=commands.tolist(),
                         command_errors=command_errors.tolist(),
                         command_feasible_prefix=command_prefix,
                         command_terminal_error=terminal_error,
                         command_progress=initial_error-terminal_error if prefix else None,
                         command_progress_eligible=False)
        evidence.append(entry)
        if prefix:
            ranked.append((score,index,candidate,actions,prefix,mean,frames,bounds,diagnostics))
    if task_commands and ranked:
        # Compare only executable endpoints: an untrusted suffix cannot claim progress.
        best_error = min(evidence[row[1]]['command_terminal_error'] for row in ranked)
        eligible = []
        for row in ranked:
            entry = evidence[row[1]]
            entry['command_progress_eligible'] = entry['command_terminal_error'] <= best_error+1e-8
            if entry['command_progress_eligible']:
                eligible.append(row)
        ranked = eligible
    return ranked, evidence
