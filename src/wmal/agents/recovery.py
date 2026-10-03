"""Finite recovery decisions from observed evidence, not inferred root causes."""


class RecoveryPolicy:
    recoverable = {'hold_condition_lost','precondition_failed','stalled','prediction_mismatch'}

    @classmethod
    def allow(cls, reason, state, node_id, config):
        return (config.recovery and reason in cls.recoverable
                and state.recoveries<config.max_recoveries
                and state.nodes[node_id]['recoveries']<config.node_recoveries)
