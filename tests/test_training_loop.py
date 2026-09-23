"""A real MuJoCo sample must yield a usable planning model without split leakage."""
import importlib.util
import json
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
        from wmal.logging.manifest import related_path
        from scripts.evaluate import evaluate_run
        from scripts.export_results import export_results

        project = Path(__file__).parents[1]
        config = project / 'configs/robots/interface_probe.json'
        with tempfile.TemporaryDirectory() as temp:
            dataset, checkpoint = Path(temp) / 'transitions.jsonl', Path(temp) / 'model.json'
            collect(config, dataset, episodes=5, steps_per_episode=3, seed=7)
            rows = load_rows(dataset)
            episode_splits = {(r['episode_id'], r['split']) for r in rows}
            self.assertEqual(len(episode_splits), 5)
            collection_summary = json.loads(related_path(dataset, 'summary').read_text())
            self.assertEqual(collection_summary['transitions'], 15)
            self.assertEqual(collection_summary['split_transitions']['test'], 3)
            self.assertEqual(len(rows[0]['applied_controls']), rows[0]['physics_steps'])
            self.assertTrue(rows[0]['full_physics_state_before'])
            report = train(dataset, checkpoint, members=3, seed=7)
            self.assertEqual(report['train_transitions'], 9)
            self.assertEqual(report['test_transitions_reserved'], 3)
            model = JointDynamics.load(checkpoint)
            self.assertEqual(model.version, report['model_version'])
            self.assertEqual(json.loads(related_path(checkpoint, 'train').read_text())['training_seed'], 7)
            backend = build_backend(config)
            plan = RolloutPlanner(model, samples=8, seed=7).plan(
                backend.profile, backend.observe(), Goal('joint_goal', {'hinge': 0.3}))
            plan.validate(backend.profile, backend.observe())
            self.assertGreater(plan.commands[0].values['hinge'], 0)
            evaluation_path = Path(temp) / 'evaluation.jsonl'
            summary = evaluate_run(config, dataset, checkpoint, evaluation_path, episodes=2, seed=9)
            self.assertEqual(summary['training_seed'], 7)
            self.assertEqual(summary['methods']['model_planner']['episodes'], 2)
            step_path = Path(temp) / 'evaluation.steps.jsonl'
            events = [json.loads(line) for line in step_path.read_text().splitlines()]
            self.assertTrue(events)
            self.assertTrue(all(len(event['applied_controls']) == event['physics_steps'] for event in events))
            self.assertEqual(sum(event['imagined_model_steps'] for event in events if event['method'] == 'direct'), 0)
            paper = export_results([related_path(evaluation_path, 'summary')], Path(temp) / 'paper.csv')
            self.assertEqual(paper['by_method']['model_planner']['training_seeds'], 1)


if __name__ == '__main__':
    unittest.main()
