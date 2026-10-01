import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wmal.monitor.catalog import RunCatalog


class RunCatalogTests(unittest.TestCase):
    def test_single_frame_archive_reports_one_frame_and_validates_alignment(self):
        np.savez(self.run / 'prediction_single.npz', predicted_rgb=np.zeros((3,8,8), dtype=np.float32),
                 episode_id=np.array(['not-a-scalar']),before_step=-1,after_step=0,decision_id='d')
        run_id = self.catalog.list_runs()[0]['run_id']
        info = self.catalog.read_prediction_info(run_id, 'prediction_single.npz')
        self.assertEqual(info['frame_count'], 1)
        self.assertFalse(info['aligned'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'runs'
        self.run = self.root / 'visual_demo'
        self.run.mkdir(parents=True)
        (self.run / 'events.jsonl').write_text('{"event":"plan"}\n', encoding='utf-8')
        (self.run / 'run_manifest.json').write_text(json.dumps({
            'run_kind': 'visual_agent', 'git_commit': 'abc123', 'seed': 4}), encoding='utf-8')
        (self.run / 'task_000.json').write_text(json.dumps({'status': 'succeeded'}), encoding='utf-8')
        rgb = np.zeros((2, 3, 3, 4), dtype=np.float32)
        rgb[0, 0, :, :] = 1.0
        np.savez_compressed(self.run / 'prediction_0_0000.npz', predicted_rgb=rgb,
                            actions=np.zeros((2, 1), dtype=np.float32))
        self.catalog = RunCatalog(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_discovers_run_and_exposes_only_safe_artifacts_and_json(self):
        runs = self.catalog.list_runs()
        self.assertEqual(len(runs), 1)
        run_id = runs[0]['run_id']
        self.assertEqual(runs[0]['run_kind'], 'visual_agent')
        artifacts = {item['relative_path']: item['kind'] for item in self.catalog.list_artifacts(run_id)}
        self.assertEqual(artifacts['events.jsonl'], 'events')
        self.assertEqual(artifacts['task_000.json'], 'json')
        self.assertEqual(artifacts['prediction_0_0000.npz'], 'prediction')
        self.assertEqual(artifacts['run_manifest.json'], 'manifest')
        self.assertEqual(self.catalog.read_json_artifact(run_id, 'task_000.json'),
                         {'status': 'succeeded'})

    def test_extracts_prediction_rgb_as_png_without_executing_npz_payloads(self):
        run_id = self.catalog.list_runs()[0]['run_id']

        image = self.catalog.read_prediction_frame(run_id, 'prediction_0_0000.npz', 0)

        self.assertTrue(image.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertGreater(len(image), 40)

    def test_prediction_metadata_reports_available_frames_and_missing_alignment(self):
        run_id = self.catalog.list_runs()[0]['run_id']
        info = self.catalog.read_prediction_info(run_id, 'prediction_0_0000.npz')
        self.assertEqual(info['frame_count'], 2)
        self.assertFalse(info['aligned'])
        self.assertEqual(info['available_images'], ['predicted_rgb'])
        self.assertEqual(info['actions'], [[0.0], [0.0]])

    def test_training_history_list_is_a_supported_json_artifact(self):
        (self.run / 'history.json').write_text('[{"epoch":1,"validation_loss":0.2}]')
        run_id = self.catalog.list_runs()[0]['run_id']
        self.assertEqual(self.catalog.read_json_artifact(run_id, 'history.json'),
                         [{'epoch': 1, 'validation_loss': .2}])

    def test_named_benchmark_logs_are_independent_event_sources(self):
        suite = self.root / 'g1_suite'
        suite.mkdir()
        (suite / 'results.json').write_text('{}')
        (suite / 'obstacle-seed0.jsonl').write_text('{"event":"goal_result"}\n')
        (suite / 'obstacle-seed1.jsonl').write_text('{"event":"plan"}\n')
        runs = self.catalog.list_runs()
        sources = [row for row in runs if row.get('events_file', '').startswith('obstacle')]
        self.assertEqual(len(sources), 2)
        self.assertEqual(len({row['run_id'] for row in sources}), 2)
        for row in sources:
            self.assertEqual(self.catalog.event_log_path(row['run_id']).name, row['events_file'])
            self.assertEqual(self.catalog.resolve_run(row['run_id']), suite)

    def test_dark_uint8_prediction_pixels_are_not_rescaled(self):
        from wmal.monitor.catalog import _encode_png
        rgb = np.ones((4, 5, 3), dtype=np.uint8)
        np.savez(self.run / 'prediction_dark.npz', predicted_rgb=rgb)
        run_id = self.catalog.list_runs()[0]['run_id']
        self.assertEqual(self.catalog.read_prediction_frame(run_id, 'prediction_dark.npz'), _encode_png(rgb))

    def test_rejects_path_traversal_symlink_escape_and_unknown_run(self):
        run_id = self.catalog.list_runs()[0]['run_id']
        outside = Path(self.temp.name) / 'outside.json'
        outside.write_text('{"secret":true}', encoding='utf-8')
        try:
            (self.run / 'linked.json').symlink_to(outside)
        except (OSError, NotImplementedError):
            pass
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.catalog.read_json_artifact(run_id, '../outside.json')
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.catalog.read_json_artifact(run_id, 'linked.json')
        inside_link = self.run / 'inside_link.json'
        inside_link.symlink_to(self.run / 'task_000.json')
        with self.assertRaises(ValueError):
            self.catalog.read_json_artifact(run_id, 'inside_link.json')
        with self.assertRaises(KeyError):
            self.catalog.resolve_run('unknown-run')

    def test_rejects_unsafe_npz_members_and_invalid_frame_indices(self):
        run_id = self.catalog.list_runs()[0]['run_id']
        with self.assertRaises(ValueError):
            self.catalog.read_prediction_frame(run_id, 'prediction_0_0000.npz', 3)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.catalog.read_prediction_frame(run_id, 'task_000.json', 0)

    def test_artifact_discovery_does_not_follow_symlink_directories(self):
        outside = Path(self.temp.name) / 'external'
        outside.mkdir()
        (outside / 'events.jsonl').write_text('{"event":"external"}\n', encoding='utf-8')
        try:
            (self.root / 'linked').symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest('Directory symlinks are not supported')
        self.assertEqual(len(self.catalog.list_runs()), 1)


if __name__ == '__main__':
    unittest.main()
