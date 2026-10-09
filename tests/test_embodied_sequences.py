"""Hand-derived regressions for aligned observations and original-episode isolation."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wmal.logging.manifest import sha256_file


class EmbodiedSequenceTests(unittest.TestCase):
    def setUp(self):
        try:
            self.profiles = importlib.import_module('wmal.datasets.embodied_profiles')
            self.data = importlib.import_module('wmal.datasets.embodied_sequences')
        except ModuleNotFoundError as exc:
            self.fail(f'Task2 interfaces unavailable: {exc.name}')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def profile(self, **changes):
        values = dict(physical_order=['shoulder', 'elbow'],
                      command_order=['shoulder_target', 'elbow_target'],
                      action_order=['shoulder', 'elbow'], action_schema='fixture.delta.v1',
                      camera_ids=['front', 'side'], image_size=32, period_s=.2,
                      physical_units=['rad', 'rad'], command_units=['rad', 'rad'],
                      action_units=['rad', 'rad'], command_rule='joint_delta',
                      event_names=['contact'], controller={'profile': 'fixture_pd', 'gains': [4, 5]},
                      origin={'dataset': 'fixture', 'revision': 'r1', 'license': 'MIT'})
        values.update(changes)
        return self.profiles.ObservationProfile(**values)

    def arrays(self, shift=0):
        # Deliberately independent physical response; action[t] updates command[t+1].
        rgb = np.zeros((5, 2, 32, 32, 3), dtype=np.uint8)
        rgb[:, 0] = np.array([10, 20, 30, 40, 50], dtype=np.uint8)[:, None, None, None]
        rgb[:, 1] = np.array([60, 70, 80, 90, 100], dtype=np.uint8)[:, None, None, None]
        return dict(rgb=rgb, physical=np.array([[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]], dtype='float32') + shift,
                    commands=np.array([[10, 20], [11, 18], [14, 22], [9, 28], [16, 20]], dtype='float32') + shift,
                    actions=np.array([[1, -2], [3, 4], [-5, 6], [7, -8]], dtype='float32'),
                    timestamps=np.array([1., 1.2, 1.4, 1.6, 1.8]),
                    view_mask=np.array([[1, 1], [1, 0], [1, 1], [0, 1], [1, 1]], dtype=bool),
                    events=np.array([[0], [1], [0], [1]], dtype='float32'),
                    event_mask=np.array([[1], [1], [0], [1]], dtype=bool))

    def rows(self):
        rows = []
        for i, split in enumerate(('train', 'validation', 'calibration', 'test')):
            name = split + '.npz'
            np.savez_compressed(self.root / name, **self.arrays(i * 100))
            rows.append(dict(path=name, episode_id=split + '-export', split=split,
                             source_episode_id=split + '-original', source_sha256=str(i + 1) * 64))
        return rows

    def manifest(self, rows=None, profile=None):
        return self.data.write_embodied_manifest(self.root / 'manifest.json',
                    self.rows() if rows is None else rows, self.profile() if profile is None else profile,
                    {'kind': 'controlled', 'license': 'MIT', 'revision': 'r1'})

    # Catches shifting actions to the consequence instead of its preceding observation.
    def test_hand_checked_action_alignment(self):
        dataset = self.data.EmbodiedDataset(self.manifest(), 'train', horizon=2, context_steps=1)
        sample = dataset[1]
        np.testing.assert_array_equal(sample['physical'], [[0, 2], [2, 4], [4, 6], [6, 8]])
        np.testing.assert_array_equal(sample['commands'], [[10, 20], [11, 18], [14, 22], [9, 28]])
        np.testing.assert_array_equal(sample['actions'], [[1, -2], [3, 4], [-5, 6]])
        np.testing.assert_allclose(sample['timestamps'], [1, 1.2, 1.4, 1.6], atol=1e-12)
        np.testing.assert_allclose(sample['rgb'][:, :, 0, 0, 0],
                                   np.array([[10, 60], [20, 70], [30, 80], [40, 90]]) / 255)
        self.assertEqual(sample['rgb'].shape, (4, 2, 3, 32, 32))
        self.assertEqual(sample['rgb'].dtype, np.float32)
        np.testing.assert_array_equal(sample['view_mask'], [[1, 1], [1, 0], [1, 1], [0, 1]])
        np.testing.assert_array_equal(sample['events'], [[1], [0]])
        np.testing.assert_array_equal(sample['event_mask'], [[1], [0]])
        self.assertEqual(sample['context_length'], 1)
        self.assertEqual(sample['prediction_offset'], 1)
        self.assertEqual(sample['episode_id'], 'train-export')
        self.assertEqual(sample['source_episode_id'], 'train-original')
        self.assertEqual(sample['source_sha256'], '1' * 64)
        self.assertNotEqual(dataset[0]['sample_id'], sample['sample_id'])
        self.assertEqual(dataset[1]['sample_id'], sample['sample_id'])

    # Catches padding being trained as future targets or replacing real action endpoints.
    def test_padding_preserves_real_targets(self):
        dataset = self.data.EmbodiedDataset(self.manifest(), 'train', horizon=2, context_steps=3)
        sample = dataset[1]
        np.testing.assert_array_equal(sample['physical'], [[0, 2], [0, 2], [0, 2], [2, 4], [4, 6], [6, 8]])
        np.testing.assert_array_equal(sample['actions'], [[0, 0], [0, 0], [1, -2], [3, 4], [-5, 6]])
        np.testing.assert_array_equal(sample['commands'][3:], [[11, 18], [14, 22], [9, 28]])
        np.testing.assert_allclose(sample['timestamps'], [1, 1, 1, 1.2, 1.4, 1.6])
        np.testing.assert_array_equal(sample['is_first'], [1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(sample['view_mask'][:2], [[0, 0], [0, 0]])
        self.assertEqual((sample['prediction_offset'], sample['context_length']), (3, 1))
        self.assertEqual(len(sample['events']), 2)

    # Catches camera re-exports bypassing split grouping through new file hashes/IDs.
    def test_multiview_episode_cannot_leak_splits(self):
        for identity in ('source_episode_id', 'source_sha256'):
            with self.subTest(identity=identity):
                rows = self.rows()
                rows[-1][identity] = rows[0][identity]
                self.assertNotEqual(sha256_file(self.root / rows[-1]['path']), sha256_file(self.root / rows[0]['path']))
                with self.assertRaisesRegex(ValueError, 'overlap|split|leak'):
                    self.manifest(rows)

    # Catches absent measurements, shape/time mismatch, and nonfinite supervision.
    def test_missing_and_misaligned_modalities(self):
        mutations = [lambda a: a.pop('physical'), lambda a: a.update(commands=a['commands'][:-1]),
                     lambda a: a.update(actions=a['actions'][:-1]),
                     lambda a: a.update(rgb=a['rgb'][:, :1]),
                     lambda a: a.update(event_mask=np.ones((3, 1))),
                     lambda a: a.pop('event_mask'),
                     lambda a: a.update(timestamps=np.array([1, 1.2, 1.3, 1.6, 1.8])),
                     lambda a: a['physical'].__setitem__((2, 0), np.nan),
                     lambda a: a.update(view_mask=np.zeros((5, 2))),
                     lambda a: a.update(events=np.ones((4, 1)) * 2),
                     lambda a: a.update(view_mask=np.ones((5, 2)) * .5)]
        for i, mutation in enumerate(mutations):
            with self.subTest(case=i):
                rows = self.rows()
                arrays = self.arrays()
                mutation(arrays)
                np.savez_compressed(self.root / 'train.npz', **arrays)
                with self.assertRaises(ValueError):
                    self.manifest(rows)

    def legacy(self, *, wide=False, controller=True):
        from wmal.datasets.visual_sequences import write_episode, write_manifest
        joints = ['left_shoulder_pitch_joint', 'left_elbow_joint']
        rows = []
        for i, split in enumerate(('train', 'validation', 'calibration', 'test')):
            a = self.arrays(i * 100)
            states = np.concatenate([a['physical'], a['commands']], axis=1)
            actions = a['actions']
            if wide:
                states = np.ones((5, 16), dtype='float32') * i
                actions = np.ones((4, 16), dtype='float32') * i
            name = 'legacy-' + split + '.npz'
            write_episode(self.root / name, a['rgb'][:, 0], states, actions)
            rows.append(dict(path=name, episode_id=split, split=split))
        source = {'kind': 'mujoco_g1_arm_servo' if not wide else 'external', 'license': 'MIT', 'revision': 'r1'}
        if controller and not wide:
            source['controller_profile'] = {'profile': 'g1_joint_pd_v1'}
        return write_manifest(self.root / 'legacy.json', rows,
            action_schema='external.absolute.v1' if wide else 'wmal.g1.workcell.target_delta.v1',
            action_order=[f'j{i}' for i in range(16)] if wide else joints,
            state_order=[f'q{i}' for i in range(16)] if wide else joints + [j + '_target' for j in joints],
            camera='front', period_s=.2, image_size=32, source=source)

    # Catches invented qvel, overwritten legacy files, or loss of original root/hash.
    def test_legacy_four_state_explicit_mapping(self):
        manifest = self.legacy()
        before = {p: p.read_bytes() for p in self.root.iterdir()}
        adapter = self.data.from_visual_manifest(manifest, physical_indices=[0, 1],
                                                 command_indices=[2, 3], command_rule='joint_delta')
        dataset = self.data.EmbodiedDataset(adapter, 'train', horizon=2, context_steps=0)
        sample = dataset[1]
        np.testing.assert_array_equal(sample['physical'], [[2, 4], [4, 6], [6, 8]])
        np.testing.assert_array_equal(sample['commands'], [[11, 18], [14, 22], [9, 28]])
        np.testing.assert_array_equal(sample['actions'], [[3, 4], [-5, 6]])
        self.assertEqual(sample['rgb'].shape, (3, 1, 3, 32, 32))
        self.assertEqual(dataset.profile.physical_units, ('rad', 'rad'))
        self.assertEqual(dataset.profile.controller, {'profile': 'g1_joint_pd_v1'})
        self.assertEqual(dataset.profile.origin['license'], 'MIT')
        self.assertEqual(sample['source_sha256'], sha256_file(self.root / 'legacy-train.npz'))
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.iterdir()})
        with self.assertRaises(ValueError):
            self.data.EmbodiedDataset(manifest, 'train', horizon=2, context_steps=0)
        for physical, commands in [([0, 1], [1, 3]), ([0, 0], [2, 3]), ([0, 4], [2, 3]), ([True, 1], [2, 3])]:
            with self.subTest(mapping=(physical, commands)), self.assertRaises(ValueError):
                self.data.from_visual_manifest(manifest, physical_indices=physical,
                                               command_indices=commands, command_rule='joint_delta')

    # Catches normalization fitted to validation/future windows and renamed ancestry.
    def test_train_only_normalization_and_ancestor_hashes(self):
        manifest = self.manifest()
        dataset = self.data.EmbodiedDataset(manifest, 'train', horizon=2, context_steps=3)
        norm = dataset.normalization()
        np.testing.assert_allclose(norm['physical_mean'], [4, 6])
        np.testing.assert_allclose(norm['physical_scale'], [2.8284271247461903] * 2)
        np.testing.assert_allclose(norm['command_mean'], [12, 21.6])
        np.testing.assert_allclose(norm['command_scale'], [2.6076809620810595, 3.4409301068170506])
        np.testing.assert_allclose(norm['action_mean'], [1.5, 0])
        np.testing.assert_allclose(norm['action_scale'], [4.330127018922194, 5.477225575051661])
        with self.assertRaises(ValueError):
            self.data.EmbodiedDataset(manifest, 'test', horizon=2, context_steps=0).normalization()
        payload = json.loads(manifest.read_text())
        rows = payload['episodes']
        lineage = self.data.validate_lineage(rows)
        self.assertIn('train-original', lineage['train']['episode_ids'])
        self.assertIn('1' * 64, lineage['train']['sha256'])
        for key, exposed in [('source_episode_id', 'train-original'), ('source_sha256', '1' * 64)]:
            candidate = dict(rows[-1], episode_id='new-name', **{key: exposed})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'ancestor|exposure|overlap'):
                self.data.validate_lineage([candidate], lineage)
        before = json.loads(json.dumps(lineage))
        combined = self.data.validate_lineage([rows[0]], lineage)
        combined['train']['sha256'].append('f' * 64)
        self.assertEqual(lineage, before)

    # Catches externally mutated controller/origin metadata changing profile identity.
    def test_profile_serialization_is_validated_and_defensive(self):
        profile = self.profile()
        payload = profile.to_dict()
        restored = self.profiles.ObservationProfile.from_dict(json.loads(json.dumps(payload)))
        payload['controller']['gains'][0] = 999
        payload['origin']['license'] = 'changed'
        profile.controller['gains'][0] = 888
        self.assertEqual(restored.to_dict(), self.profile().to_dict())
        self.assertEqual(profile.controller['gains'], [4, 5])
        values = self.profile().to_dict()
        for change in ({'physical_units': []}, {'period_s': float('nan')}, {'command_rule': 'absolute'},
                       {'controller': {}}, {'origin': {}}, {'camera_ids': ['front', 'front']},
                       {'command_order': ['wrong', 'elbow_target']}, {'image_size': True},
                       {'extra': 1}, {'action_units': ['rad', 'm']}, {'origin': {'bad': np.array([1])}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.profiles.ObservationProfile.from_dict(values | change)

    # Catches silently inventing semantics or discarding fourteen external action dims.
    def test_arbitrary_legacy_profiles_are_explicit_offline_only(self):
        manifest = self.legacy(wide=True)
        with self.assertRaises(ValueError):
            self.data.from_visual_manifest(manifest, physical_indices=[0, 1], command_indices=[], command_rule='joint_delta')
        with self.assertRaises(ValueError):
            self.data.from_visual_manifest(manifest, physical_indices=list(range(16)), command_indices=[], command_rule='none')
        profile = self.profile(physical_order=[f'q{i}' for i in range(16)], command_order=[],
            action_order=[f'j{i}' for i in range(16)], action_schema='external.absolute.v1',
            camera_ids=['front'], event_names=[], physical_units=['rad'] * 16,
            command_units=[], action_units=['rad'] * 16, command_rule='none',
            controller={'profile': 'declared_external'}, origin={'dataset': 'external', 'license': 'MIT'})
        adapter = self.data.from_visual_manifest(manifest, physical_indices=list(range(16)),
            command_indices=[], command_rule='none', profile=profile)
        dataset = self.data.EmbodiedDataset(adapter, 'train', horizon=2, context_steps=0)
        self.assertEqual(dataset[0]['physical'].shape, (3, 16))
        self.assertEqual(dataset[0]['actions'].shape, (2, 16))
        self.assertEqual(dataset[0]['commands'].shape, (3, 0))
        self.assertEqual(dataset.profile.command_rule, 'none')

    def test_unproven_controller_and_wrong_known_command_transition_rejected(self):
        manifest = self.legacy(controller=False)
        # A collector kind does not bind which fixed controller actually ran.
        with self.assertRaises(ValueError):
            self.data.from_visual_manifest(manifest, physical_indices=[0, 1], command_indices=[2, 3], command_rule='joint_delta')
        rows = self.rows()
        a = self.arrays()
        a['commands'][2, 0] += 1
        np.savez_compressed(self.root / 'train.npz', **a)
        with self.assertRaisesRegex(ValueError, 'command'):
            self.manifest(rows)

    def test_source_groups_and_episode_sampling_keep_all_windows(self):
        rows = self.rows()
        extra = dict(rows[0], path='other-camera.npz', episode_id='second-export')
        a = self.arrays()
        a['rgb'] += 1
        np.savez_compressed(self.root / extra['path'], **a)
        rows.append(extra)
        dataset = self.data.EmbodiedDataset(self.manifest(rows), 'train', horizon=2, context_steps=0)
        self.assertEqual(dataset.episode_window_indices, {'train-original': [0, 1, 2, 3, 4, 5]})
        self.assertEqual(len({dataset[i]['sample_id'] for i in range(6)}), 6)

    def test_paths_hashes_and_lazy_cache_samples_are_safe(self):
        rows = self.rows()
        rows[0]['path'] = '../escape.npz'
        with self.assertRaises(ValueError):
            self.manifest(rows)
        manifest = self.manifest()
        dataset = self.data.EmbodiedDataset(manifest, 'train', horizon=2, context_steps=0)
        sample = dataset[0]
        sample['physical'][0, 0] = 999
        sample['events'][0, 0] = 1
        np.testing.assert_array_equal(dataset[0]['physical'][0], [0, 2])
        np.testing.assert_array_equal(dataset[0]['events'][0], [0])
        a = self.arrays()
        a['rgb'] += 1
        np.savez_compressed(self.root / 'test.npz', **a)
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.data.EmbodiedDataset(manifest, 'train', horizon=2, context_steps=0)

    def test_optional_events_and_derived_time_do_not_invent_labels(self):
        rows = self.rows()
        for row in rows:
            a = self.arrays()
            for key in ('events', 'event_mask', 'timestamps', 'view_mask'):
                a.pop(key)
            a['rgb'] += ('train', 'validation', 'calibration', 'test').index(row['split'])
            np.savez_compressed(self.root / row['path'], **a)
        dataset = self.data.EmbodiedDataset(self.manifest(rows), 'train', horizon=2, context_steps=0)
        sample = dataset[0]
        np.testing.assert_array_equal(sample['events'], [[0], [0]])
        np.testing.assert_array_equal(sample['event_mask'], [[0], [0]])
        np.testing.assert_allclose(sample['timestamps'], [0, .2, .4])
        self.assertEqual(sample['timing_source'], 'period_derived')
        self.assertEqual(sample['period_s'], .2)

    def test_invalid_windows_and_empty_training_split_are_rejected(self):
        manifest = self.manifest()
        for horizon, context in [(0, 0), (2, -1), (True, 0), (2, True), (5, 0)]:
            with self.subTest(window=(horizon, context)), self.assertRaises(ValueError):
                self.data.EmbodiedDataset(manifest, 'train', horizon=horizon, context_steps=context)
        with self.assertRaises(ValueError):
            self.data.EmbodiedDataset(manifest, 'unknown', horizon=2, context_steps=0)

    # Catches treating a processed export's hash as the original LeRobot trajectory.
    def test_legacy_import_retains_original_multifile_hashes(self):
        manifest = self.legacy(wide=True)
        payload = json.loads(manifest.read_text())
        payload['source'].update(kind='lerobot_v2.1_import',
            dataset={'repo_id': 'unitree/fixture', 'revision': 'r1', 'license': 'MIT'},
            source_episodes=[{'episode_index': i, 'parquet_sha256': str(i + 1) * 64,
                              'video_sha256': str(i + 5) * 64, 'original_split': row['split']}
                             for i, row in enumerate(payload['episodes'])])
        for i, row in enumerate(payload['episodes']):
            row['episode_id'] = f'lerobot_{i}'
        from wmal.logging.manifest import atomic_json
        atomic_json(manifest, payload)
        profile = self.profile(physical_order=[f'q{i}' for i in range(16)], command_order=[],
            action_order=[f'j{i}' for i in range(16)], action_schema='external.absolute.v1',
            camera_ids=['front'], event_names=[], physical_units=['rad'] * 16,
            command_units=[], action_units=['rad'] * 16, command_rule='none',
            controller={'profile': 'external'}, origin=payload['source'])
        adapter = self.data.from_visual_manifest(manifest, physical_indices=list(range(16)),
            command_indices=[], command_rule='none', profile=profile)
        dataset = self.data.EmbodiedDataset(adapter, 'train', horizon=2, context_steps=0)
        sample = dataset[0]
        self.assertEqual(sample['source_sha256'], '1' * 64)
        self.assertIn('5' * 64, sample['source_hashes'])
        self.assertEqual(sample['source_episode_id'], 'unitree/fixture:episode:0')
        lineage = self.data.validate_lineage(dataset.rows)
        candidate = dict(adapter['episodes'][-1], episode_id='renamed', source_episode_id='renamed-original',
                         source_hashes=['5' * 64])
        with self.assertRaisesRegex(ValueError, 'ancestor|exposure|overlap'):
            self.data.validate_lineage([candidate], lineage)
        payload['source']['source_episodes'].pop()
        atomic_json(manifest, payload)
        with self.assertRaisesRegex(ValueError, 'source|original'):
            self.data.from_visual_manifest(manifest, physical_indices=list(range(16)),
                command_indices=[], command_rule='none', profile=profile)

    # Catches declaring radians as degrees or replacing a recorded fixed controller.
    def test_explicit_g1_profile_cannot_override_known_units_or_controller(self):
        manifest = self.legacy()
        adapter = self.data.from_visual_manifest(manifest, physical_indices=[0, 1],
            command_indices=[2, 3], command_rule='joint_delta')
        for change in ({'physical_units': ['degree', 'degree'], 'command_units': ['degree', 'degree'],
                        'action_units': ['degree', 'degree']}, {'controller': {'profile': 'other_pd'}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.data.from_visual_manifest(manifest, physical_indices=[0, 1], command_indices=[2, 3],
                    command_rule='joint_delta', profile=adapter['profile'] | change)

    # Catches silent float32 overflow when constructing the model-facing sample.
    def test_finite_raw_values_must_remain_finite_in_returned_coordinates(self):
        rows = self.rows()
        a = self.arrays()
        a['physical'] = np.full((5, 2), 1e100)
        np.savez_compressed(self.root / 'train.npz', **a)
        with self.assertRaisesRegex(ValueError, 'physical|float32|finite'):
            self.manifest(rows)

    def test_lazy_episode_validation_and_bounded_cache(self):
        rows = self.rows()
        for i in range(3):
            a = self.arrays(i + 10)
            a['rgb'] += i + 1
            name = f'extra-{i}.npz'
            np.savez_compressed(self.root / name, **a)
            rows.append(dict(path=name, episode_id=f'extra-{i}', split='train',
                             source_episode_id=f'original-extra-{i}', source_sha256=str(i + 5) * 64))
        manifest = self.manifest(rows)
        dataset = self.data.EmbodiedDataset(manifest, 'train', horizon=2, context_steps=0)
        self.assertEqual(len(dataset._cache), 0)
        for index in (0, 3, 6, 9, 0):
            self.assertTrue(np.isfinite(dataset[index]['physical']).all())
            self.assertLessEqual(len(dataset._cache), 2)
        payload = json.loads(manifest.read_text())
        a = self.arrays()
        a.pop('physical')
        np.savez_compressed(self.root / 'train.npz', **a)
        payload['episodes'][0]['sha256'] = sha256_file(self.root / 'train.npz')
        from wmal.logging.manifest import atomic_json
        atomic_json(manifest, payload)
        dataset = self.data.EmbodiedDataset(manifest, 'train', horizon=2, context_steps=0)
        with self.assertRaisesRegex(ValueError, 'physical'):
            dataset[0]

    # Catches forgetting hash aliases when building the episode-level sampler.
    def test_renamed_originals_with_same_hash_share_sampling_group(self):
        rows = self.rows()
        a = self.arrays()
        a['rgb'] += 1
        np.savez_compressed(self.root / 'renamed.npz', **a)
        rows.append(dict(rows[0], path='renamed.npz', episode_id='renamed-export',
                         source_episode_id='renamed-original'))
        dataset = self.data.EmbodiedDataset(self.manifest(rows), 'train', horizon=2, context_steps=0)
        self.assertEqual(dataset.episode_window_indices, {'train-original': [0, 1, 2, 3, 4, 5]})

    def test_symlink_escape_and_missing_origin_are_rejected(self):
        rows = self.rows()
        (self.root / 'escape.npz').symlink_to('/dev/null')
        rows[0]['path'] = 'escape.npz'
        with self.assertRaises(ValueError):
            self.manifest(rows)
        rows = self.rows()
        rows[0].pop('source_sha256')
        with self.assertRaises(ValueError):
            self.manifest(rows)

    # Catches allowing inherited exposure without the original content identities.
    def test_unknown_or_incomplete_ancestor_lineage_fails_closed(self):
        rows = json.loads(self.manifest().read_text())['episodes']
        for lineage in ({}, {'schema': 'wmal.episode_lineage.v1', 'train': {'episode_ids': [], 'sha256': []}},
                        {'schema': 'wmal.episode_lineage.v1', 'train': {'episode_ids': ['ancestor'], 'sha256': []},
                         'selection': {'episode_ids': [], 'sha256': []}}):
            with self.subTest(lineage=lineage), self.assertRaises(ValueError):
                self.data.validate_lineage(rows, lineage)

    def test_origin_requires_identity_and_profile_input_is_snapshotted(self):
        values = self.profile().to_dict()
        controller, origin = values['controller'], values['origin']
        profile = self.profiles.ObservationProfile.from_dict(values)
        controller['gains'].append(9)
        origin['revision'] = 'changed'
        self.assertEqual(profile.controller['gains'], [4, 5])
        self.assertEqual(profile.origin['revision'], 'r1')
        for origin in ({'license': 'MIT'}, {'dataset': ''}, {'dataset': {'license': 'MIT'}}):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                self.profiles.ObservationProfile.from_dict(self.profile().to_dict() | {'origin': origin})

    # Catches identical sample IDs for different explicitly selected state coordinates.
    def test_sample_identity_binds_profile_and_legacy_mapping(self):
        manifest = self.legacy(wide=True)
        profile = self.profile(physical_order=['q0', 'q1'], command_order=[],
            action_order=[f'j{i}' for i in range(16)], action_schema='external.absolute.v1',
            camera_ids=['front'], event_names=[], physical_units=['rad'] * 2,
            command_units=[], action_units=['rad'] * 16, command_rule='none',
            controller={'profile': 'external'}, origin={'dataset': 'fixture', 'revision': 'r1'})
        first = self.data.from_visual_manifest(manifest, physical_indices=[0, 1], command_indices=[],
                                               command_rule='none', profile=profile)
        other = self.profiles.ObservationProfile.from_dict(profile.to_dict() | {'physical_order': ['q2', 'q3']})
        second = self.data.from_visual_manifest(manifest, physical_indices=[2, 3], command_indices=[],
                                                command_rule='none', profile=other)
        left = self.data.EmbodiedDataset(first, 'train', horizon=2, context_steps=0)
        right = self.data.EmbodiedDataset(second, 'train', horizon=2, context_steps=0)
        self.assertNotEqual(left[0]['sample_id'], right[0]['sample_id'])

    # Catches a camera-specific ID replacing a known original repo/episode ID.
    def test_legacy_import_rejects_conflicting_canonical_original_ids(self):
        manifest = self.legacy(wide=True)
        payload = json.loads(manifest.read_text())
        payload['source'].update(kind='lerobot_v2.1_import',
            dataset={'repo_id': 'same/repo', 'revision': 'r1', 'license': 'MIT'},
            source_episodes=[{'episode_index': i + 7, 'parquet_sha256': str(i + 1) * 64,
                              'video_sha256': str(i + 5) * 64}
                             for i in range(4)])
        for i, row in enumerate(payload['episodes']):
            row['episode_id'] = f'lerobot_{i + 7}'
            row['source_episode_id'] = f'same/repo:episode:{i + 7}'
        profile = self.profile(physical_order=[f'q{i}' for i in range(16)], command_order=[],
            action_order=[f'j{i}' for i in range(16)], action_schema='external.absolute.v1',
            camera_ids=['front'], event_names=[], physical_units=['rad'] * 16,
            command_units=[], action_units=['rad'] * 16, command_rule='none',
            controller={'profile': 'external'}, origin=payload['source'])
        from wmal.logging.manifest import atomic_json
        atomic_json(manifest, payload)
        adapter = self.data.from_visual_manifest(manifest, physical_indices=list(range(16)),
            command_indices=[], command_rule='none', profile=profile)
        lineage = self.data.validate_lineage([adapter['episodes'][0]])
        self.assertIn('same/repo:episode:7', lineage['train']['episode_ids'])
        # Different camera records may have different hashes; the canonical ID
        # alone must still prevent renamed test re-exports of episode 7.
        candidate = dict(adapter['episodes'][0], episode_id='renamed-side', split='test',
                         sha256='a' * 64, source_sha256='b' * 64, source_hashes=['c' * 64])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.data.validate_lineage([adapter['episodes'][0], candidate])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.data.validate_lineage([candidate], lineage)
        for camera, split, digest in [('front', 'train', 'a'), ('side', 'test', 'b')]:
            with self.subTest(camera=camera):
                exported = json.loads(json.dumps(payload))
                # The same original episode, with camera-specific ID AND hashes.
                exported['episodes'][0].update(source_episode_id=f'reexport-{camera}', source_sha256=digest * 64)
                exported['source']['source_episodes'][0].update(parquet_sha256=digest * 64,
                                                               video_sha256=('c' if camera == 'front' else 'd') * 64)
                if split == 'test':
                    exported['episodes'][0]['split'], exported['episodes'][-1]['split'] = 'test', 'train'
                atomic_json(manifest, exported)
                with self.assertRaises(ValueError):
                    self.data.from_visual_manifest(manifest, physical_indices=list(range(16)),
                        command_indices=[], command_rule='none', profile=profile)

    # Catches bypassing known radians through reordered/subset offline mappings.
    def test_g1_offline_mappings_preserve_units_for_all_selected_coordinates(self):
        manifest = self.legacy()
        base = self.data.from_visual_manifest(manifest, physical_indices=[0, 1],
            command_indices=[2, 3], command_rule='joint_delta')['profile']
        state_order = json.loads(manifest.read_text())['semantics']['state_order']
        for physical, commands, expected_physical, expected_commands in [
                ([1, 0], [], [[2, 0], [4, 2], [6, 4]], [[], [], []]),
                ([1], [2], [[2], [4], [6]], [[10], [11], [14]]),
                ([0], [3, 2], [[0], [2], [4]], [[20, 10], [18, 11], [22, 14]])]:
            profile = base | dict(physical_order=[state_order[i] for i in physical],
                command_order=[state_order[i] for i in commands], physical_units=['rad'] * len(physical),
                command_units=['rad'] * len(commands), command_rule='none')
            adapter = self.data.from_visual_manifest(manifest, physical_indices=physical,
                command_indices=commands, command_rule='none', profile=profile)
            sample = self.data.EmbodiedDataset(adapter, 'train', horizon=2, context_steps=0)[0]
            np.testing.assert_array_equal(sample['physical'], expected_physical)
            np.testing.assert_array_equal(sample['commands'], expected_commands)
            np.testing.assert_array_equal(sample['actions'], [[1, -2], [3, 4]])
            for key in ('physical_units', 'command_units', 'action_units'):
                if not profile[key]:
                    continue
                with self.subTest(mapping=(physical, commands), units=key), self.assertRaises(ValueError):
                    self.data.from_visual_manifest(manifest, physical_indices=physical,
                        command_indices=commands, command_rule='none',
                        profile=profile | {key: ['degree'] * len(profile[key])})


if __name__ == '__main__':
    unittest.main()
