"""Reject invalid capabilities before an action can be authorized."""
import unittest
import numpy as np


class TaskGraphTests(unittest.TestCase):
    def test_graph_dependencies_and_defensive_target_copy(self):
        from wmal.agents.task_graph import TaskNode, TaskGraph
        source = [.43, .95]
        graph = TaskGraph('task', (TaskNode('a', 'joint_reach', source),
                                  TaskNode('h', 'joint_hold', source, ('a',), 3)))
        digest = graph.digest
        source[0] = .6
        self.assertEqual(graph.nodes[0].target, (.43, .95))
        self.assertEqual(graph.digest, digest)
        self.assertEqual(TaskGraph.from_dict(graph.to_dict()).digest, digest)

    def test_rejects_cycle_unknown_skill_target_and_boolean_budget(self):
        from wmal.agents.task_graph import TaskNode, TaskGraph
        for args in [('a', 'grasp', [.43, .95]), ('a', 'joint_reach', [float('nan'), .95]),
                     ('a', 'joint_reach', [.9, .95]), ('a', 'joint_hold', [.43, .95], (), 0),
                     ('a', 'joint_reach', [.43, .95], (), 0, True)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                TaskNode(*args)
        with self.assertRaises(ValueError):
            TaskGraph('t', (TaskNode('a', 'joint_reach', [.43,.95], ('b',)),
                            TaskNode('b', 'joint_reach', [.43,.95], ('a',))))
        with self.assertRaises(ValueError):
            TaskGraph.from_dict({'schema':'wmal.task_graph.v1','task_id':'x','nodes':[], 'surprise':1})

    def test_skills_reach_target_limits_and_hold_has_only_zero_actions(self):
        from wmal.agents.task_graph import TaskNode
        from wmal.skills.research_contracts import JointSkills
        from wmal.agents.predictive_skill_agent import VisualObservation
        obs = VisualObservation('e', 0, np.zeros((3,32,32)), np.array([.35,.87,.35,.87]))
        proposals = JointSkills.candidates(obs, TaskNode('a','joint_reach',[.43,.95]), 4,
                                          np.random.default_rng(0))
        self.assertEqual(len(proposals), 12)
        self.assertTrue(all(np.abs(c.actions).max() <= .06 for c in proposals))
        hold = JointSkills.candidates(obs, TaskNode('h','joint_hold',[.35,.87],hold_steps=3),3,
                                     np.random.default_rng(0))
        np.testing.assert_array_equal(hold[0].actions, np.zeros((3,2)))


if __name__ == '__main__':
    unittest.main()
