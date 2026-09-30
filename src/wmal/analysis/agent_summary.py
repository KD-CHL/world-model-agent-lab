"""Descriptive task summaries; missing final events remain incomplete."""
from collections import Counter
import math


def summarize_agent_events(records):
    tasks, skill_statuses, latencies, residuals = {}, Counter(), [], []
    for record in records:
        event, payload = record['event'], record['payload']
        task_id = payload.get('task_id')
        if task_id is None:
            continue
        task = tasks.setdefault(task_id, {'task_id': task_id, 'status': 'incomplete',
                                         'cycles': None, 'replans': 0, 'uncertain_commands': 0})
        if event == 'task_started':
            task['mode'] = payload.get('mode')
        elif event == 'task_result':
            task.update(status=payload['status'], cycles=payload.get('cycles'))
        elif event == 'command_uncertain':
            task['uncertain_commands'] += 1
        elif event == 'replan_requested':
            task['replans'] += 1
        elif event == 'skill_result':
            skill_statuses[(payload['skill'], payload['status'])] += 1
        elif event == 'plan' and 'planning_latency_s' in payload:
            value = payload['planning_latency_s']
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                latencies.append(value)
        elif event == 'prediction_residual' and 'position_error_m' in payload:
            value = payload['position_error_m']
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                residuals.append(value)
    def percentile(values, q):
        return sorted(values)[max(0, math.ceil(len(values)*q)-1)] if values else None
    return {'schema': 'wmal.agent_summary.v1', 'tasks': list(tasks.values()),
            'task_status_counts': dict(Counter(t['status'] for t in tasks.values())),
            'skill_status_counts': [{'skill': k[0], 'status': k[1], 'count': v}
                                    for k,v in sorted(skill_statuses.items())],
            'planning_latency_s': {'n':len(latencies), 'p50':percentile(latencies,.5),
                                   'p95':percentile(latencies,.95)},
            'position_residual_m': {'n':len(residuals),
                'rmse':math.sqrt(sum(v*v for v in residuals)/len(residuals)) if residuals else None},
            'statistical_scope':'descriptive_only; train seeds and episode splits required for inference'}
