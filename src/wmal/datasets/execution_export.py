"""Export acknowledged execution transitions using explicit whole-episode assignments."""
import json
from pathlib import Path
import tempfile
from wmal.datasets.motion_sequences import load_motion_episodes
from wmal.logging.manifest import atomic_json, sha256_file


def export_execution_logs(logs, output, assignments, *, source):
    if source != 'mujoco_interaction':
        raise ValueError('This exporter requires an explicitly declared MuJoCo interaction source')
    if not assignments or any(value not in ('train','validation','test') for value in assignments.values()):
        raise ValueError('Provide an explicit episode-to-split map')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Dataset output already exists')
    rows, audit, versions = [], [], set()
    files = [Path(path) for path in logs]
    if not files:
        raise ValueError('No execution logs')
    for path in files:
        for line in path.read_text().splitlines():
            item = json.loads(line)
            event, payload = item['event'], item['payload']
            if event == 'executed_transition':
                if payload.get('schema')!='wmal.executed_transition.v1':
                    raise ValueError('Unsupported execution event schema')
                episode = payload['before']['episode_id']
                if episode not in assignments:
                    raise ValueError('Unassigned execution episode: '+episode)
                rows.append({'split':assignments[episode],
                    **{key:payload[key] for key in ('before','action','after')},
                    'behavior_model_version':payload['model_version'],
                    'task_id':payload.get('task_id'), 'wall_time_utc':item.get('wall_time_utc')})
                versions.add(payload['model_version'])
            elif event in ('failure','safety_stop','command_uncertain','task_result','goal_result',
                           'planning_event','session_exit'):
                audit.append(item)
    payload = {'schema':'wmal.g1.transitions.v1','data_source':source,'rows':rows,
               'provenance':{'source_declared_by_operator':True,
                    'logs':[{'path':str(path),'sha256':sha256_file(path)} for path in files],
                    'behavior_model_versions':sorted(versions),'split_assignment':assignments},
               'audit_events':audit}
    # Validate before publication, including duplicate keys, time alignment and split leakage.
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory)/'candidate.json'
        atomic_json(candidate,payload)
        splits, duration = load_motion_episodes(candidate)
    atomic_json(output,payload)
    return {'output':str(output),'transitions':len(rows),'duration_s':duration,
            'episodes':{name:len(episodes) for name,episodes in splits.items()},
            'audit_events':len(audit),'data_source':source}
