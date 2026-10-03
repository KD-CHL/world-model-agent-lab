"""Immutable, strictly bounded research tasks; no generated executable code."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import numpy as np


def positive_int(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def joint_target(value):
    from wmal.envs.visual_workcell import TARGET_LOW, TARGET_HIGH
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ValueError('Target requires two named joint positions')
    if any(isinstance(x, (bool, np.bool_)) for x in value):
        raise ValueError('Boolean joint targets are invalid')
    target = np.asarray(value, dtype=float)
    if (target.shape != (2,) or not np.isfinite(target).all()
            or np.any(target < TARGET_LOW) or np.any(target > TARGET_HIGH)):
        raise ValueError('Target outside the local G1 joint envelope')
    return tuple(float(x) for x in target)


@dataclass(frozen=True)
class TaskNode:
    node_id: str
    skill: str
    target: tuple
    dependencies: tuple = ()
    hold_steps: int = 0
    max_actions: int = 24

    def __post_init__(self):
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError('Node ID is required')
        if self.skill not in ('joint_reach', 'joint_hold'):
            raise ValueError('Unknown research skill')
        object.__setattr__(self, 'target', joint_target(self.target))
        if (not isinstance(self.dependencies, (tuple, list))
                or any(not isinstance(x, str) or not x for x in self.dependencies)
                or len(set(self.dependencies)) != len(self.dependencies)):
            raise ValueError('Invalid dependencies')
        object.__setattr__(self, 'dependencies', tuple(self.dependencies))
        positive_int(self.max_actions, 'max_actions')
        positive_int(self.hold_steps, 'hold_steps', minimum=0)
        if self.skill == 'joint_hold' and not self.hold_steps:
            raise ValueError('Hold requires real executed samples')
        if self.hold_steps > self.max_actions:
            raise ValueError('Hold cannot fit its action budget')


@dataclass(frozen=True)
class TaskGraph:
    task_id: str
    nodes: tuple
    start: tuple = (.35, .87)

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError('Task ID is required')
        object.__setattr__(self, 'nodes', tuple(self.nodes))
        object.__setattr__(self, 'start', joint_target(self.start))
        if not self.nodes or len(self.nodes) > 64 or any(not isinstance(n, TaskNode) for n in self.nodes):
            raise ValueError('Task requires 1..64 valid nodes')
        by_id = {n.node_id: n for n in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError('Duplicate node IDs')
        visiting, complete = set(), set()
        def visit(name):
            if name not in by_id:
                raise ValueError('Unknown dependency')
            if name in visiting:
                raise ValueError('Task graph must be acyclic')
            if name in complete:
                return
            visiting.add(name)
            for dependency in by_id[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            complete.add(name)
        for name in by_id:
            visit(name)

    def to_dict(self):
        return {'schema': 'wmal.task_graph.v1', 'task_id': self.task_id,
                'start': list(self.start), 'nodes': [asdict(n) for n in self.nodes]}

    @property
    def digest(self):
        return sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, dict) or set(value) - {'schema','task_id','nodes','start'}
                or value.get('schema') != 'wmal.task_graph.v1' or not isinstance(value.get('nodes'), list)):
            raise ValueError('Invalid task graph schema/fields')
        nodes = []
        fields = {'node_id','skill','target','dependencies','hold_steps','max_actions'}
        for node in value['nodes']:
            if not isinstance(node, dict) or set(node) - fields:
                raise ValueError('Unknown task node fields')
            try:
                nodes.append(TaskNode(**node))
            except TypeError as exc:
                raise ValueError('Missing task node fields') from exc
        return cls(value.get('task_id'), tuple(nodes), value.get('start', (.35,.87)))
