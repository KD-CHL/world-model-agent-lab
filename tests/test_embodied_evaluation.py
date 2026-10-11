"""Independent literal metric expectations and real frozen-network evaluation."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from test_embodied_training import fixture, tiny_config
from wmal.datasets.embodied_sequences import EmbodiedDataset, validate_lineage
from wmal.models.embodied_rssm import ScalableRSSMMember
from wmal.models.embodied_world import EmbodiedWorldModel
from wmal.models.embodied_calibration import EmbodiedCalibration
from wmal.training import embodied_trainer


class EmbodiedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root/'data', events=False)
        data = EmbodiedDataset(self.manifest, 'train', horizon=2, context_steps=1)
        cfg = tiny_config(embodied_trainer).model
        from dataclasses import replace
        cfg = replace(cfg, event_dim=0)
        torch.manual_seed(19); member = ScalableRSSMMember(cfg)
        # Real network with exactly zero physical residual: predicts last measurement.
        for parameter in member.physical_head.parameters():
            parameter.data.zero_()
        normalization = data.normalization()
        # Binary-exact fixture coordinates avoid rounding at the conformal boundary.
        normalization.update(physical_mean=[0.], physical_scale=[1.])
        self.model = EmbodiedWorldModel(cfg, data.profile, normalization, [member],
            metadata={'episode_lineage': validate_lineage(data.manifest['episodes'])})
        self.path = self.root/'model.pt'; self.model.save(self.path)

    def api(self):
        try:
            return importlib.import_module('wmal.training.embodied_evaluation')
        except ModuleNotFoundError:
            self.fail('Independent embodied evaluator has not been implemented')

    # Including exact target commands inflates the coordinate count and dilutes error.
    def test_hand_computed_physical_metrics_exclude_commands(self):
        report = self.api().evaluate_embodied(self.manifest, self.path)
        self.assertEqual(report['physical']['order'], ['q'])
        np.testing.assert_allclose(report['physical']['model']['rmse_per_horizon_dim'], [[.125], [.25]], atol=1e-6)
        self.assertAlmostEqual(report['physical']['model']['rmse'], np.sqrt(.0390625), places=6)
        self.assertEqual(report['counts']['windows'], 4)
        self.assertEqual(report['counts']['original_episodes'], 1)
        self.assertEqual(report['commands']['scored_as_prediction_gain'], False)
        self.assertIsNone(report['events'])

    # Checking all four splits against ancestry rejects legitimate selection rows;
    # omitting selected test-origin aliases admits contaminated evaluation.
    def test_test_split_and_ancestry(self):
        api = self.api()
        api.evaluate_embodied(self.manifest, self.path)
        data = EmbodiedDataset(self.manifest, 'test', horizon=2)
        metadata = self.model.metadata
        metadata['episode_lineage']['train']['sha256'].append(data.rows[0]['source_sha256'])
        contaminated = EmbodiedWorldModel(self.model.config, self.model.profile, self.model.normalization,
            [self.model.members[0]], metadata=metadata)
        path = self.root/'contaminated.pt'; contaminated.save(path)
        with self.assertRaisesRegex(ValueError, 'ancestor'):
            api.evaluate_embodied(self.manifest, path)

    # Counting windows as independent episodes would exaggerate confidence.
    def test_coverage_width_and_persistence(self):
        api = self.api()
        data = EmbodiedDataset(self.manifest, 'calibration', horizon=2, context_steps=1)
        calibration = EmbodiedCalibration.fit(data, self.model, alpha=.8)
        path = self.root/'cal.json'; calibration.save(path)
        report = api.evaluate_embodied(self.manifest, self.path, calibration=path)
        np.testing.assert_allclose(report['physical']['persistence']['rmse_per_horizon_dim'], [[.125], [.25]])
        self.assertLess(report['physical']['train_linear']['rmse'], 1e-6)
        self.assertEqual(report['intervals']['joint_episode_coverage'], 1)
        np.testing.assert_allclose(report['intervals']['mean_full_width_per_horizon_dim'], [[.5], [.5]], atol=1e-6)
        self.assertEqual(report['intervals']['independent_episodes'], 1)
        self.assertGreater(report['intervals']['wilson_95'][1]-report['intervals']['wilson_95'][0], .7)
        self.assertFalse(report['intervals']['formal_sample_size_met'])
        self.assertFalse(report['sensitivity']['counterfactual_accuracy'])
        self.assertEqual(report['sensitivity']['mean_physical_delta_zero_action'], 0)

    # A successful command must save a usable report, and must not overwrite evidence.
    def test_evaluation_cli_writes_report(self):
        self.api()
        output = self.root/'report.json'
        command = [sys.executable, 'scripts/embodied_world.py', 'evaluate', '--manifest', str(self.manifest),
                   '--model', str(self.path), '--output', str(output)]
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = json.loads(output.read_text()); self.assertEqual(saved['counts']['windows'], 4)
        self.assertEqual(saved['model_version'], self.model.version)
        again = subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(json.loads(output.read_text()), saved)

    # Static pixels cannot pretend to measure motion prediction.
    def test_rgb_empty_changed_regions_report_no_evidence(self):
        report = self.api().evaluate_embodied(self.manifest, self.path)
        self.assertGreater(report['rgb']['model_mse'], 0)
        self.assertEqual(report['rgb']['persistence_mse'], 0)
        self.assertEqual(report['rgb']['changed_pixel_fraction'], 0)
        self.assertIsNone(report['rgb']['model_changed_mse'])

    # Frozen protocols forbid silently evaluating beyond trained H.
    def test_horizon_and_unknown_lineage_rejected(self):
        api = self.api()
        with self.assertRaises(ValueError):
            api.evaluate_embodied(self.manifest, self.path, horizon=3)
        unknown = EmbodiedWorldModel(self.model.config, self.model.profile, self.model.normalization,
                                    [self.model.members[0]])
        path = self.root/'unknown.pt'; unknown.save(path)
        with self.assertRaisesRegex(ValueError, 'lineage'):
            api.evaluate_embodied(self.manifest, path)

    # Invalid counterfactual actions must be counted, not abort valid real-action scoring.
    def test_out_of_scope_sensitivity_is_explicitly_unavailable(self):
        metadata = self.model.metadata
        metadata['prediction_scope'] = {'action_lower': [.2], 'action_upper': [.3]}
        model = EmbodiedWorldModel(self.model.config, self.model.profile, self.model.normalization,
                                  [self.model.members[0]], metadata=metadata)
        path = self.root/'scoped.pt'; model.save(path)
        report = self.api().evaluate_embodied(self.manifest, path)
        self.assertEqual(report['counts']['windows'], 4)
        self.assertIsNone(report['sensitivity']['mean_physical_delta_zero_action'])
        self.assertEqual(report['sensitivity']['zero_action']['out_of_scope_windows'], 4)
        self.assertEqual(report['sensitivity']['shuffled_action']['evaluated_windows'], 4)

    # An interval file may have used test origins even if model training did not.
    def test_calibration_origin_overlap_is_rejected(self):
        api = self.api()
        data = EmbodiedDataset(self.manifest, 'calibration', horizon=2, context_steps=1)
        test = EmbodiedDataset(self.manifest, 'test', horizon=2, context_steps=1)
        records = []
        for index, row in enumerate(data.rows):
            sample = data[next(i for i, entry in enumerate(data.index) if entry[0] == index)]
            identity = test.rows[0] if index == 0 else row
            records.append(dict(identity, split='calibration', profile=data.profile.to_dict(),
                prediction=self.model.predict_sample(sample), target=sample['physical'][2:]))
        cal = EmbodiedCalibration.fit(records, self.model, alpha=.8)
        path = self.root/'leaked-cal.json'; cal.save(path)
        with self.assertRaisesRegex(ValueError, 'ancestor'):
            api.evaluate_embodied(self.manifest, self.path, calibration=path)

    # Loading a tiny hand-built artifact must not advertise a formal 150M model.
    def test_false_formal_capacity_label_rejected(self):
        from dataclasses import replace
        config = replace(self.model.config, preset='base')
        member = ScalableRSSMMember(config); member.load_state_dict(self.model.members[0].state_dict())
        model = EmbodiedWorldModel(config, self.model.profile,
            self.model.normalization, [member], metadata=self.model.metadata)
        path = self.root/'false-base.pt'; model.save(path)
        with self.assertRaisesRegex(ValueError, 'parameter budget'):
            self.api().evaluate_embodied(self.manifest, path)


if __name__ == '__main__':
    unittest.main()
