"""Descriptive statistics for experiment artifacts; no pseudo-replicated CI."""
import math
from statistics import mean


def percentile(values, fraction):
    if not values or not 0 <= fraction <= 1:
        raise ValueError('Empty values or invalid percentile')
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def control_summary(episodes, steps):
    if not episodes:
        raise ValueError('No evaluation episodes')
    result = {}
    for method in sorted({row['method'] for row in episodes}):
        group = [row for row in episodes if row['method'] == method]
        events = [row for row in steps if row['method'] == method]
        planning = [row['planning_wall_s'] for row in events]
        execution = [row['execution_wall_s'] for row in events]
        prediction_errors = [error ** 2 for row in events for error in row['prediction_error_rad'].values()]
        result[method] = {
            'episodes': len(group),
            'success_count': sum(bool(row['success']) for row in group),
            'success_rate': mean(bool(row['success']) for row in group),
            'mean_final_error_rad': mean(row['final_error_rad'] for row in group),
            'mean_control_steps': mean(row['control_steps'] for row in group),
            'physics_steps': sum(row['physics_steps'] for row in events),
            'sim_seconds': sum(row['next_sim_time_s'] - row['sim_time_s'] for row in events),
            'imagined_model_steps': sum(row['imagined_model_steps'] for row in events),
            'one_step_prediction_rmse_rad': math.sqrt(mean(prediction_errors)) if prediction_errors else None,
            'planning_latency_p50_s': percentile(planning, 0.5) if planning else None,
            'planning_latency_p95_s': percentile(planning, 0.95) if planning else None,
            'planning_latency_p99_s': percentile(planning, 0.99) if planning else None,
            'execution_latency_p50_s': percentile(execution, 0.5) if execution else None,
            'execution_latency_p95_s': percentile(execution, 0.95) if execution else None,
            'execution_latency_p99_s': percentile(execution, 0.99) if execution else None,
        }
    return result
