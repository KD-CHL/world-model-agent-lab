"""A real MuJoCo sample must yield a usable planning model without split leakage."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'Optional MuJoCo not installed')
class TrainingLoopTests(unittest.TestCase):
    def test_sample_train_validate_and_plan(self):
        from wmal.communication.contracts import Goal
        from wmal.models.latent_dynamics import JointDynamics
        from wmal.planners.world_planner import RolloutPlanner
        from wmal.training.collector import build_backend, collect
        from wmal.training.trainer import load_rows, train

        project = Path(__file__).parents[1]
        config = project / 'configs/robots/interface_probe.json'
        with tempfile.TemporaryDirectory() as temp:
            dataset, checkpoint = Path(temp) / 'transitions.jsonl', Path(temp) / 'model.json'
            collect(config, dataset, episodes=5, steps_per_episode=3, seed=7)
            rows = load_rows(dataset)
            episode_splits = {(r['episode_id'], r['split']) for r in rows}
            self.assertEqual(len(episode_splits), 5)
            report = train(dataset, checkpoint, members=3, seed=7)
            self.assertEqual(report['train_transitions'], 9)
            self.assertEqual(report['test_transitions_reserved'], 3)
            model = JointDynamics.load(checkpoint)
            self.assertEqual(model.version, report['model_version'])
            backend = build_backend(config)
            plan = RolloutPlanner(model, samples=8, seed=7).plan(
                backend.profile, backend.observe(), Goal('joint_goal', {'hinge': 0.3}))
            plan.validate(backend.profile, backend.observe())
            self.assertGreater(plan.commands[0].values['hinge'], 0)


if __name__ == '__main__':
    unittest.main()
