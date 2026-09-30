"""Validated episode groups and contiguous action-conditioned G1 sequences."""
import json
from pathlib import Path

from wmal.communication.g1_session_protocol import state_from_wire
from wmal.locomotion.contracts import G1VelocityAction


def load_motion_episodes(path):
    payload = json.loads(Path(path).read_text())
    if payload.get('schema') != 'wmal.g1.transitions.v1' or not payload.get('rows'):
        raise ValueError('Expected nonempty wmal.g1.transitions.v1 dataset')
    splits = {name: {} for name in ('train', 'validation', 'test')}
    assignments, duration, joints = {}, None, None
    seen = set()
    for row in payload['rows']:
        split = row['split']
        if split not in splits:
            raise ValueError('Unknown dataset split')
        before, after = state_from_wire(row['before']), state_from_wire(row['after'])
        action = G1VelocityAction(**row['action'])
        if (before.episode_id != after.episode_id or after.step_id != before.step_id+1
                or abs(after.sim_time_s-before.sim_time_s-action.duration_s) > 1e-6):
            raise ValueError('Invalid transition provenance')
        if assignments.setdefault(before.episode_id, split) != split:
            raise ValueError('Episode leakage across data splits')
        key = before.episode_id, before.step_id
        if key in seen:
            raise ValueError('Duplicate transition')
        seen.add(key)
        duration = action.duration_s if duration is None else duration
        joints = set(before.joint_positions) if joints is None else joints
        if (abs(action.duration_s-duration) > 1e-9 or set(before.joint_positions) != joints
                or set(after.joint_positions) != joints):
            raise ValueError('Mixed action duration or joint schema')
        splits[split].setdefault(before.episode_id, []).append((before, action, after))
    for episodes in splits.values():
        for transitions in episodes.values():
            transitions.sort(key=lambda row: row[0].step_id)
            for previous, following in zip(transitions, transitions[1:]):
                if following[0].step_id == previous[2].step_id and previous[2] != following[0]:
                    raise ValueError('Adjacent transition states disagree')
    return splits, duration


def contiguous_sequences(episodes, horizon):
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Sequence horizon must be positive')
    result = {}
    for episode_id, rows in episodes.items():
        sequences = []
        for start in range(len(rows)-horizon+1):
            window = rows[start:start+horizon]
            if all(left[2] == right[0] for left,right in zip(window,window[1:])):
                sequences.append(window)
        if sequences:
            result[episode_id] = sequences
    if not result:
        raise ValueError('No contiguous sequences at the requested horizon')
    return result
