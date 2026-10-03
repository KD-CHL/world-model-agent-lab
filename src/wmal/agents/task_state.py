"""Serializable factual state, independent of imagined model trajectories."""
from copy import deepcopy


class TaskState:
    def __init__(self, graph):
        self.status = 'ready'
        self.graph_hash = graph.digest
        self.active_node = None
        self.nodes = {n.node_id: {'status':'pending', 'actions':0, 'hold_count':0,
                                 'recoveries':0, 'evidence':None} for n in graph.nodes}
        self.completed_subgoals = []
        self.executed_cycles = 0
        self.decision_count = 0
        self.recoveries = 0
        self.recoveries_succeeded = 0
        self.rejections = 0
        self.replans = 0
        self.failure_reason = None
        self.recent_feedback = {}
        self.budget_remaining = {}
        self.belief = {}

    def snapshot(self):
        return deepcopy(vars(self))
