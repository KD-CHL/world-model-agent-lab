"""Descriptive projections of recorded evidence, with explicit missing values."""
from collections import Counter
import math


def sanitize(value, depth=0):
    """Redact credential/prompt fields and make imported evidence JSON-safe."""
    if depth > 32:
        return '[depth limit]'
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace('-', '_')
            sensitive = (normalized in {'api_key', 'apikey', 'token', 'access_token', 'password',
                                       'secret', 'authorization', 'credentials', 'env', 'environment_variables'}
                         or 'prompt' in normalized or normalized.endswith('_api_key'))
            result[str(key)] = '[redacted]' if sensitive else sanitize(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize(item, depth + 1) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize(events):
    tasks, latest, latencies = {}, {}, []
    errors = {'state_error': [], 'position_error_m': [], 'rmse_rad': [], 'frame_mse': []}
    counts = Counter()
    legacy_index = 0
    for row in events:
        kind, payload = row['event'], row['payload']
        counts[kind] += 1
        latest[kind] = payload
        task_id = payload.get('task_id') or 'legacy-' + str(legacy_index)
        task = tasks.setdefault(task_id, {'task_id': task_id, 'status': 'incomplete', 'cycles': None})
        if kind == 'task_started':
            task.update(status='running', goal=payload.get('goal'), max_cycles=payload.get('max_cycles'),
                        baseline=payload.get('baseline'), callable_skills=payload.get('callable_skills'))
        if kind in ('task_result', 'goal_result', 'mission_result'):
            state = payload.get('state', {})
            state = state if isinstance(state, dict) else {}
            task.update(status=payload.get('status', state.get('status', 'incomplete')),
                        cycles=payload.get('cycles', state.get('executed_cycles')),
                        elapsed_s=payload.get('elapsed_s'), replans=state.get('replans', payload.get('replans')),
                        failure_reason=state.get('failure_reason', payload.get('detail')))
            if not payload.get('task_id'):
                legacy_index += 1
        if kind == 'plan':
            evidence = payload.get('evidence', {})
            evidence = evidence if isinstance(evidence, dict) else {}
            latency = evidence.get('planning_ms')
            if latency is None and isinstance(payload.get('planning_latency_s'), (int, float)):
                latency = payload['planning_latency_s'] * 1000
            if isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0:
                latencies.append(latency)
        if kind in ('feedback', 'prediction_residual'):
            for key in errors:
                value = payload.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    errors[key].append(value)
    def statistics(values):
        values = sorted(values)
        return {'n': len(values), 'mean': sum(values) / len(values) if values else None,
                'p50': values[max(0, math.ceil(len(values) * .5) - 1)] if values else None,
                'p95': values[max(0, math.ceil(len(values) * .95) - 1)] if values else None}
    statuses = Counter(task['status'] for task in tasks.values())
    return {'tasks': list(tasks.values()), 'task_status_counts': dict(statuses),
            'event_counts': dict(counts), 'latest': latest,
            'planning_latency_ms': statistics(latencies),
            'prediction_errors': {key: statistics(values) for key, values in errors.items()},
            'scope': 'descriptive; retained event window only', 'retained_events': len(events)}
