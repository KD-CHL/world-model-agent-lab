"""Hand-computed simultaneous episode bounds, independent of exact commands."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_embodied_provider import make_model, profile, inputs


class EmbodiedCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def api(self):
        try:
            self.world = importlib.import_module('wmal.models.embodied_world')
            return importlib.import_module('wmal.models.embodied_calibration')
        except ModuleNotFoundError as exc:
            self.fail(f'Task3 calibration missing: {exc.name}')

    def records(self, model):
        rows = []
        # Five independent episodes: joint scores 1,2,3,4,5. At alpha .2 rank=5.
        for episode in range(1, 6):
            for window in range(2):
                predicted = np.zeros((len(model.members), 8, 3, 2))
                target = np.zeros((3, 2)); target[window, window] = episode * [2, 4][window]
                rows.append(dict(episode_id=f'cal-{episode}', source_episode_id=f'original-{episode}',
                    sha256=f'{episode:064x}', source_sha256=f'{episode+10:064x}',
                    source_hashes=[f'{episode+20:064x}'], split='calibration',
                    profile=profile(origin={'dataset': 'independent-calibration'}).to_dict(),
                    prediction=dict(physical_states=predicted, model_version=model.version,
                        protocol_hash=model.protocol_hash(), known_commands=np.ones((3, 2))*9999),
                    target=target))
        return rows

    def clean_model(self, *, members=1):
        return make_model(self.world, members=members, metadata={'episode_lineage': {
            'schema': 'wmal.episode_lineage.v1', 'train': {'episode_ids': [], 'sha256': []},
            'selection': {'episode_ids': [], 'sha256': []}}})

    # Marginal per-H quantiles or counting overlapping windows as episodes breaks [10,20].
    def test_joint_horizon_episode_calibration(self):
        api = self.api(); model = make_model(self.world, metadata={'episode_lineage': {
            'schema': 'wmal.episode_lineage.v1', 'train': {'episode_ids': ['train'], 'sha256': ['a'*64]},
            'selection': {'episode_ids': ['val'], 'sha256': ['b'*64]}}})
        calibration = api.EmbodiedCalibration.fit(self.records(model), model)
        np.testing.assert_array_equal(calibration.bounds(horizon=3, model=model), [[10, 20]]*3)
        self.assertEqual(calibration.quantile, 5)
        self.assertEqual(calibration.episode_count, 5)
        self.assertEqual(len(calibration.episode_records), 5)
        self.assertEqual(calibration.episode_records[0]['source_hashes'], [f'{21:064x}'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cal.json'; calibration.save(path)
            loaded = api.EmbodiedCalibration.load(path)
            np.testing.assert_array_equal(loaded.bounds(horizon=2, model=model), [[10, 20]]*2)
            payload = json.loads(path.read_text()); payload['payload']['quantile'] = 0
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError): api.EmbodiedCalibration.load(path)

    # Original IDs/hashes from ancestors remain excluded even after a new export.
    def test_ancestor_calibration_leak_and_profile_mismatch(self):
        api = self.api()
        lineage = {'schema': 'wmal.episode_lineage.v1',
            'train': {'episode_ids': ['original-1'], 'sha256': ['a'*64]},
            'selection': {'episode_ids': ['val'], 'sha256': ['b'*64]}}
        model = make_model(self.world, metadata={'episode_lineage': lineage})
        with self.assertRaises(ValueError): api.EmbodiedCalibration.fit(self.records(model), model)
        lineage['train']['episode_ids'] = ['train']; lineage['train']['sha256'] = [f'{21:064x}']
        model = make_model(self.world, metadata={'episode_lineage': lineage})
        with self.assertRaises(ValueError): api.EmbodiedCalibration.fit(self.records(model), model)
        lineage['train']['sha256'] = ['a'*64]
        model = make_model(self.world, metadata={'episode_lineage': lineage})
        records = self.records(model)
        records[0]['profile']['controller']['gains'] = [9, 9]
        with self.assertRaises(ValueError): api.EmbodiedCalibration.fit(records, model)
        records = self.records(model); records[0]['profile']['physical_units'][0] = 'deg'
        with self.assertRaises(ValueError): api.EmbodiedCalibration.fit(records, model)

    # Single-member samples are not an ensemble; protocol changes must require refitting.
    def test_modes_samples_and_insufficient_episodes(self):
        api = self.api(); model = make_model(self.world, metadata={'episode_lineage': {
            'schema': 'wmal.episode_lineage.v1', 'train': {'episode_ids': [], 'sha256': []},
            'selection': {'episode_ids': [], 'sha256': []}}})
        records = self.records(model)
        calibration = api.EmbodiedCalibration.fit(records, model)
        for kw in ({'samples': 2}, {'mode': 'soft'}):
            with self.assertRaises(ValueError): calibration.validate(model, **kw)
        # n=3: ceil((3+1)*.8)=4 cannot select a finite observed order statistic.
        with self.assertRaises(ValueError): api.EmbodiedCalibration.fit(records[:6], model)
        with self.assertRaises(ValueError): api.EmbodiedCalibration.fit(records, model, bound_mode='ensemble_scaled')
        with self.assertRaises(ValueError): api.EmbodiedCalibration.fit(records, make_model(self.world))
        with self.assertRaises(ValueError): calibration.bounds(horizon=4, model=model)

    # The old selector divides raw physical bounds by training physical scales, not mean/std mixed units.
    def test_legacy_normalized_budget_and_command_zero_bounds(self):
        api = self.api(); model = make_model(self.world, metadata={'episode_lineage': {
            'schema': 'wmal.episode_lineage.v1', 'train': {'episode_ids': [], 'sha256': []},
            'selection': {'episode_ids': [], 'sha256': []}}})
        cal = api.EmbodiedCalibration.fit(self.records(model), model)
        adapter = model.as_visual_predictor(calibration=cal)
        bounds = adapter.calibration.bounds(np.zeros((2, 4)), version=model.version)
        np.testing.assert_array_equal(bounds, [[10, 20, 0, 0]]*2)
        np.testing.assert_array_equal(adapter.normalization['state_scale'], [2, 4, 5, 10])
        np.testing.assert_array_equal((bounds/adapter.normalization['state_scale']).max(-1), [5, 5])
        adapter.calibration.validate_for(model.version, adapter.semantics)
        adapter.calibration.validate_controller(model.profile.controller)

    # Calibration on the original four-way manifest may contain exposed train/selection rows;
    # only selected calibration rows must remain independent of those ancestors.
    def test_real_dataset_adapter_startup_and_exposed_unselected_splits(self):
        api = self.api()
        from wmal.datasets.embodied_sequences import (EmbodiedDataset, write_embodied_manifest, validate_lineage)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = []
            for i, split in enumerate(['train', 'validation']+['calibration']*4+['test']):
                name = f'episode-{i}.npz'
                np.savez_compressed(root/name, rgb=np.full((4, 1, 16, 16, 3), 70+i, dtype='uint8'),
                    physical=np.array([[12, -16], [13, -15], [14, -14], [15, -13]], dtype='float32'),
                    commands=np.array([[110, 220], [111, 218], [114, 222], [114, 222]], dtype='float32'),
                    actions=np.array([[1, -2], [3, 4], [0, 0]], dtype='float32'))
                rows.append(dict(path=name, episode_id=f'export-{i}', source_episode_id=f'original-{i}',
                                 source_sha256=f'{i+100:064x}', split=split))
            path = write_embodied_manifest(root/'manifest.json', rows,
                profile(origin={'dataset': 'new-independent-origin'}), {'kind': 'fixture'})
            dataset = EmbodiedDataset(path, 'calibration', horizon=2, context_steps=2)
            lineage = validate_lineage(dataset.manifest['episodes'])
            model = make_model(self.world, metadata={'episode_lineage': lineage})
            cal = api.EmbodiedCalibration.fit(dataset, model)
            self.assertEqual(cal.episode_count, 4)
            self.assertTrue(np.isfinite(cal.bounds(horizon=2, model=model)).all())
            self.assertEqual(cal.payload['calibration_origins'], [{'dataset': 'new-independent-origin'}])
            self.assertEqual(len(cal.payload['calibration_manifest_sha256']), 64)

    # Two exports of one original trajectory must contribute one maximum, not two votes.
    def test_reexports_grouped_and_test_split_rejected(self):
        api = self.api(); model = self.clean_model()
        records = self.records(model)
        reexport = dict(records[-1], episode_id='new-export', sha256='f'*64)
        cal = api.EmbodiedCalibration.fit(records+[reexport], model)
        self.assertEqual(cal.episode_count, 5)
        self.assertEqual(len(cal.episode_records), 6)
        self.assertEqual(cal.quantile, 5)
        records[0]['split'] = 'test'
        with self.assertRaises(ValueError): api.EmbodiedCalibration.fit(records, model)

    # A separate ensemble denominator must use member means, not within-member samples.
    def test_ensemble_scaled_protocol_and_legacy_spread(self):
        api = self.api(); model = self.clean_model(members=2)
        records = self.records(model)
        for record in records:
            record['prediction']['physical_states'][0, :, :, :] = [-2, -4]
            record['prediction']['physical_states'][1, :, :, :] = [2, 4]
        fixed = api.EmbodiedCalibration.fit(records, model)
        contrast = api.EmbodiedCalibration.fit(records, model, bound_mode='ensemble_scaled')
        self.assertNotEqual(fixed.payload['bound_protocol_hash'], contrast.payload['bound_protocol_hash'])
        np.testing.assert_array_equal(contrast.bounds(records[0]['prediction'], model=model), [[10, 20]]*3)
        with self.assertRaises(ValueError): contrast.bounds(horizon=2, model=model)
        adapter = model.as_visual_predictor(contrast)
        np.testing.assert_array_equal(adapter.calibration.bounds(np.array([[2, 4, 1000, 1000]]),
            version=model.version), [[10, 20, 0, 0]])

    # A declared local scope needs actual origin/action context, not only future targets.
    def test_explicit_records_scope_requires_and_checks_context(self):
        api = self.api(); model = self.clean_model(); records = self.records(model)
        scope = {'command_lower': [100, 210], 'command_upper': [115, 230], 'max_action_delta': 4}
        with self.assertRaises(ValueError): api.EmbodiedCalibration.fit(records, model, scope=scope)
        rgb, physical, commands, past, actions = inputs()
        for record in records:
            record['context'] = dict(physical=physical, commands=commands, actions=actions)
        cal = api.EmbodiedCalibration.fit(records, model, scope=scope)
        self.assertTrue(cal.allows_scope(physical, commands, actions, model=model))
        self.assertFalse(cal.allows_scope(physical, commands, actions*10, model=model))
        records[0]['context']['commands'] = np.array([[999, 999]])
        with self.assertRaises(ValueError): api.EmbodiedCalibration.fit(records, model, scope=scope)

    # Dataset-level origin ancestry remains excluded, not only the same exported row ID.
    def test_dataset_original_calibration_ancestry_rejected(self):
        api = self.api()
        from wmal.datasets.embodied_sequences import EmbodiedDataset, write_embodied_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = []
            for i, split in enumerate(['train', 'validation']+['calibration']*4+['test']):
                np.savez_compressed(root/f'{i}.npz', rgb=np.full((3, 1, 16, 16, 3), i, dtype='uint8'),
                    physical=np.zeros((3, 2), dtype='float32'), commands=np.zeros((3, 2), dtype='float32'),
                    actions=np.zeros((2, 2), dtype='float32'))
                rows.append(dict(path=f'{i}.npz', episode_id=f'new-export-{i}', source_episode_id=f'origin-{i}',
                                 source_sha256=f'{i+70:064x}', split=split))
            path = write_embodied_manifest(root/'manifest.json', rows, profile(), {'kind': 'fixture'})
            dataset = EmbodiedDataset(path, 'calibration', horizon=2, context_steps=2)
            model = make_model(self.world, metadata={'episode_lineage': {
                'schema': 'wmal.episode_lineage.v1',
                'train': {'episode_ids': ['renamed-ancestor'], 'sha256': [f'{72:064x}']},
                'selection': {'episode_ids': [], 'sha256': []}}})
            with self.assertRaises(ValueError): api.EmbodiedCalibration.fit(dataset, model)

    # Sorting metadata must preserve group membership and make equivalent inputs order-independent.
    def test_calibration_identity_independent_of_record_order(self):
        api = self.api(); model = self.clean_model(); records = self.records(model)
        records.append(dict(records[-1], episode_id='aaa-reexport', sha256='e'*64))
        first = api.EmbodiedCalibration.fit(records, model)
        second = api.EmbodiedCalibration.fit(list(reversed(records)), model)
        self.assertEqual(first.version, second.version)

    # Missing provenance and normalization mismatches fail closed with a contract error.
    def test_malformed_explicit_record_rejected(self):
        api = self.api(); model = self.clean_model(); records = self.records(model)
        del records[0]['source_sha256']
        with self.assertRaises(ValueError): api.EmbodiedCalibration.fit(records, model)


if __name__ == '__main__':
    unittest.main()
