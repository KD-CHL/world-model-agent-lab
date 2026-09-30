"""Typed recoverable planning failures and explicit task-level limits."""
class NavigationStalled(ValueError):
    pass


class NoFeasibleCandidate(ValueError):
    pass


RECOVERABLE = frozenset({'stalled', 'no_candidate', 'budget_exhausted', 'replan_required'})
