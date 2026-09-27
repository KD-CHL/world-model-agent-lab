import json
from pathlib import Path
import unittest
from dataclasses import replace

from scripts.g1_benchmark import make_scene
from wmal.locomotion.contracts import G1State, G1VelocityAction
from wmal.locomotion.learned import LearnedG1Dynamics
from wmal.training.g1_selection import rollout_metrics, select_model


class BenchmarkTests(unittest.TestCase):
    def test_all_ten_missions_have_feasible_routes(self):
        suite = json.loads(Path('configs/g1_experiments.json').read_text())
        self.assertEqual(len({e['id'] for e in suite['experiments']}), 10)
        for experiment in suite['experiments']:
            scene, point = make_scene(experiment['scene']), (0., 0.)
            scene.robot_radius += .15
            for goal in experiment['goals']:
                route = scene.route(point, tuple(goal[:2]))
                self.assertTrue(all(scene.segment_free(a,b) for a,b in zip(route, route[1:])))
                point = tuple(goal[:2])

    def test_selection_and_episode_leakage(self):
        groups = []
        for episode in range(3):
            state = G1State(str(episode),0,0.,0.,0.,.8,0.,0.,0.,0.,0.,0.,0.,.8)
            rows = []
            for i in range(15):
                action = G1VelocityAction(.1+(i%3)*.05,0.,0.,.5)
                after = replace(state, x=state.x+action.vx*.5, step_id=i+1,sim_time_s=(i+1)*.5)
                rows.append((state,action,after))
                state = after
            groups.append(rows)
        model, report = select_model(groups[0]+groups[1],groups[2])
        self.assertEqual(len(report['candidates']),8)
        self.assertLess(rollout_metrics(model,groups[2])['position_rmse_m'],.05)
        with self.assertRaises(ValueError):
            select_model(groups[0],groups[0])
        broken = groups[2][:]
        broken[1] = (replace(broken[1][0], x=9.),*broken[1][1:])
        with self.assertRaises(ValueError):
            rollout_metrics(model,broken)
