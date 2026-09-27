import unittest
from wmal.envs.indoor_scene import IndoorScene


class IndoorTests(unittest.TestCase):
    def test_route_around_furniture(self):
        scene = IndoorScene()
        self.assertFalse(scene.segment_free((0, 0), (4, 0)))
        route = scene.route((0, 0), (4, 0))
        self.assertGreater(len(route), 2)
        self.assertEqual(route[0], (0, 0))
        self.assertEqual(route[-1], (4, 0))
        self.assertTrue(all(scene.segment_free(a, b) for a, b in zip(route, route[1:])))

    def test_invalid_goal_and_thin_crossing(self):
        scene = IndoorScene()
        with self.assertRaises(ValueError):
            scene.route((0, 0), (2, 0))
        self.assertFalse(scene.segment_free((0, 0), (7, 0)))
        self.assertTrue(scene.segment_free((0, -1.2), (4, -1.2)))
