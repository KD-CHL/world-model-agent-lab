"""Flatten run summaries into auditable paper tables without pseudo-replicated CIs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from wmal.logging.manifest import atomic_json, build_manifest, related_path


def export_results(inputs, output):
    if not inputs:
        raise ValueError('At least one run summary is required')
    flat, seen = [], set()
    experiment_id = None
    for path in inputs:
        summary = json.loads(Path(path).read_text())
        if summary.get('run_kind') != 'mujoco_evaluation' or summary.get('schema_version') != 1:
            raise ValueError('Expected evaluation summary')
        current_experiment = summary.get('experiment_id')
        if not isinstance(current_experiment, str) or not current_experiment:
            raise ValueError('Missing experiment ID')
        if experiment_id is None:
            experiment_id = current_experiment
        elif current_experiment != experiment_id:
            raise ValueError('Cannot combine different experiment IDs')
        for method, metrics in summary['methods'].items():
            key = (summary.get('training_seed'), summary['evaluation_seed'], method,
                   summary['model_version'])
            if key in seen:
                raise ValueError('Duplicate run summary')
            seen.add(key)
            flat.append({'summary_path': str(path), 'experiment_id': current_experiment,
                         'model_version': summary['model_version'],
                         'training_seed': summary.get('training_seed'),
                         'evaluation_seed': summary['evaluation_seed'], 'method': method,
                         'test_transition_rmse_rad': summary['test_transition_rmse_rad'], **metrics})
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    by_method_seed = defaultdict(list)
    for row in flat:
        if row['training_seed'] is not None:
            by_method_seed[(row['method'], row['training_seed'])].append(row['success_rate'])
    by_method = defaultdict(list)
    for (method, _seed), rates in by_method_seed.items():
        by_method[method].append(mean(rates))
    aggregate = {'schema_version': 1, 'run_kind': 'paper_table_export',
                 'experiment_id': experiment_id, 'rows': len(flat),
                 'by_method': {method: {'training_seeds': len(rates),
                                        'mean_seed_success_rate': mean(rates),
                                        'seed_success_rates': rates}
                               for method, rates in sorted(by_method.items())},
                 'statistical_note': 'Training seeds are the replication unit. No confidence interval is emitted.'}
    atomic_json(related_path(destination, 'aggregate'), aggregate)
    atomic_json(related_path(destination, 'manifest'),
                build_manifest('paper_table_export', None,
                               {f'input_{index}': path for index, path in enumerate(inputs)},
                               {'input_run_count': len(inputs), 'output_rows': len(flat)}))
    return aggregate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(export_results(args.inputs, args.output), ensure_ascii=False))

if __name__ == "__main__":
    main()
