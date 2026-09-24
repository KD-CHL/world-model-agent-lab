import tempfile
import unittest
from pathlib import Path

from wmal.agents.task_planner import ConstrainedTaskPlanner, TaskProposal
from wmal.datasets.trajectory import Transition, read_jsonl, write_jsonl


class _Parser:
    def propose(self, instruction, joint_limits):
        return TaskProposal(instruction, {"hinge": 0.2})


class InterchangeLayerTests(unittest.TestCase):
    def test_trajectory_round_trip(self):
        row = Transition("episode-1", 0, {"hinge": 0.0}, {"hinge": 0.1}, {"hinge": 0.1})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trajectory.jsonl"
            self.assertEqual(write_jsonl(path, [row]), 1)
            self.assertEqual(read_jsonl(path), [row])

    def test_agent_proposal_is_constrained(self):
        proposal = ConstrainedTaskPlanner(_Parser(), {"hinge": (-1.0, 1.0)}).propose("reach")
        self.assertEqual(proposal.targets["hinge"], 0.2)


if __name__ == "__main__":
    unittest.main()
