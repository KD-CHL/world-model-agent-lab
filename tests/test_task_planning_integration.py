"""Task runtime must actually enable the exact-command planning contract."""
import unittest
from test_task_runtime import Predictor, Session
from wmal.agents.task_graph import TaskGraph, TaskNode
from wmal.agents.task_runtime import RuntimeConfig, TaskRuntime


class TaskPlanningIntegrationTests(unittest.TestCase):
    def test_non_grid_target_is_reached_with_one_capped_real_delta(self):
        task = TaskGraph('precise', (TaskNode('reach', 'joint_reach', [.367, .905]),))
        session = Session()
        report = TaskRuntime(task, session, Predictor(),
                             config=RuntimeConfig(baseline='A1', horizon=4)).run()
        self.assertEqual(report['state']['status'], 'succeeded')
        self.assertEqual(report['state']['executed_cycles'], 1)
        self.assertLess(abs(session.target[0] - .367), 1e-8)
        self.assertLess(abs(session.target[1] - .905), 1e-8)
