"""Temporal contracts: history is real, bounded, reset-aligned and branch-pure."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from test_visual_world import make_dataset
from wmal.datasets.visual_sequences import VisualDataset
from wmal.models.visual_latent import NetworkConfig, VisualWorldModel, make_member


def model(context_steps=2):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = NetworkConfig(2, 2, image_size=32, latent_dim=16, hidden_dim=24,
                           architecture='categorical_rssm', stoch=3, classes=4,
                           context_steps=context_steps)
    semantics = dict(state_order=['x', 'y'], action_order=['x', 'y'], image_size=32,
                     camera='front', period_s=.2, action_schema='test.delta.v1', event_names=[])
    norm = dict(state_mean=[0., 0.], state_scale=[1., 1.],
                action_mean=[.12, -.2], action_scale=[.3, .2])
    return VisualWorldModel(config, semantics, norm, [make_member(config) for _ in range(2)],
                            metadata={'max_horizon': 3})


class TemporalWorldTests(unittest.TestCase):
    def test_history_changes_prediction_without_candidate_branch_side_effects(self):
        predictor = model()
        rgb = np.zeros((3, 3, 32, 32), dtype='float32')
        states = np.zeros((3, 2), dtype='float32')
        past = np.zeros((2, 2), dtype='float32')
        future = np.ones((3, 2), dtype='float32') * .1
        a = predictor.predict_context(rgb, states, past, future)
        changed = states.copy()
        changed[0] = [3., -3.]
        b = predictor.predict_context(rgb, changed, past, future)
        self.assertGreater(np.abs(a['states'] - b['states']).max(), 1e-6)
        predictor.predict_context(rgb, states, past, -future)
        repeated = predictor.predict_context(rgb, states, past, future)
        np.testing.assert_array_equal(a['states'], repeated['states'])
        self.assertEqual(a['diagnostics']['context_steps_used'], 2)
        self.assertEqual(a['diagnostics']['conditioning'], 'bounded_real_history')

    def test_temporal_model_refuses_silent_single_frame_downgrade_and_bad_history(self):
        predictor = model()
        rgb, state, future = np.zeros((3, 32, 32)), np.zeros(2), np.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, 'context'):
            predictor.predict(rgb, state, future)
        startup = predictor.predict_context(rgb[None], state[None], np.empty((0, 2)), future)
        self.assertEqual(startup['states'].shape, (2, 2, 2))
        for frames, states, past in [(np.zeros((4, 3, 32, 32)), np.zeros((4, 2)), np.zeros((3, 2))),
                                    (rgb[None], state[None], np.zeros((1, 2))),
                                    (rgb[None] + np.nan, state[None], np.empty((0, 2)))]:
            with self.assertRaises(ValueError):
                predictor.predict_context(frames, states, past, future)
        with self.assertRaises(ValueError):
            NetworkConfig(2, 2, context_steps=2)

    def test_context_is_hashed_and_legacy_zero_context_remains_loadable(self):
        from dataclasses import replace
        predictor = model(0)
        changed = VisualWorldModel(replace(predictor.config, context_steps=2), predictor.semantics,
                                   predictor.normalization, predictor.members, metadata=predictor.metadata)
        self.assertNotEqual(predictor.version, changed.version)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.pt'
            predictor.save(path)
            payload = torch.load(path, weights_only=True)
            payload['network'].pop('context_steps')
            torch.save(payload, path)
            self.assertEqual(VisualWorldModel.load(path).version, predictor.version)
            changed.save(path)
            self.assertEqual(VisualWorldModel.load(path).config.context_steps, 2)

    def test_dataset_padding_resets_without_losing_startup_target_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_dataset(Path(tmp))
            dataset = VisualDataset(manifest, 'train', horizon=3, context_steps=2)
            old = VisualDataset(manifest, 'train', horizon=3)
            self.assertEqual(len(dataset), 24)
            first, second, full = dataset[0], dataset[1], dataset[2]
            self.assertEqual(first['rgb'].shape, (6, 3, 32, 32))
            np.testing.assert_array_equal(first['is_first'], [True, True, True, False, False, False])
            np.testing.assert_array_equal(second['is_first'], [True, True, False, False, False, False])
            np.testing.assert_array_equal(full['is_first'], [True, False, False, False, False, False])
            np.testing.assert_array_equal(first['states'][2:], old[0]['states'])
            np.testing.assert_array_equal(first['actions'][2:], old[0]['actions'])
            np.testing.assert_array_equal(first['actions'][:2], np.zeros((2, 2)))
            np.testing.assert_array_equal(full['states'][:3], old[0]['states'][:3])
            self.assertEqual(dataset[6]['episode_id'], 'train_1')
            self.assertEqual(dataset[6]['context_length'], 0)

    def test_padding_and_unpadded_posterior_match_even_with_nonzero_action_mean(self):
        predictor = model()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = VisualDataset(make_dataset(Path(tmp)), 'train', horizon=3, context_steps=2)
            for index in (0, 1, 2):
                sample = dataset[index]
                actual = predictor.predict_sample(sample)
                k = sample['context_length']
                direct = predictor.predict_context(sample['rgb'][2-k:3], sample['states'][2-k:3],
                                                   sample['actions'][2-k:2], sample['actions'][2:])
                np.testing.assert_array_equal(actual['states'], direct['states'])
                norm = predictor.normalization
                member = predictor.members[0]
                args = [torch.as_tensor(v, dtype=torch.float32)[None] for v in
                        (sample['rgb'][:3], (sample['states'][:3] - norm['state_mean']) / norm['state_scale'],
                         (sample['actions'][:2] - norm['action_mean']) / norm['action_scale'],
                         (sample['actions'][2:] - norm['action_mean']) / norm['action_scale'])]
                with torch.inference_mode():
                    padded = member.imagine_context(*args, is_first=torch.tensor(sample['is_first'][:3])[None])
                np.testing.assert_allclose(padded['state'][0].numpy(), actual['states'][0], atol=1e-7)

    def test_future_targets_cannot_change_imagined_predictions(self):
        predictor = model()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = VisualDataset(make_dataset(Path(tmp)), 'test', horizon=3, context_steps=2)
            sample = dataset[2]
            before = predictor.predict_sample(sample)
            sample['rgb'][3:] = 1. - sample['rgb'][3:]
            sample['states'][3:] += 100.
            after = predictor.predict_sample(sample)
            np.testing.assert_array_equal(before['states'], after['states'])
            np.testing.assert_array_equal(before['frames'], after['frames'])

    def test_video_provider_requires_aligned_real_history_actions(self):
        predictor = model()
        entries = [dict(rgb=np.zeros((32, 32, 3), dtype='uint8'), state=[i*.1, i*.1],
                        semantics=predictor.semantics, episode_id='real', step_id=i,
                        action_from_previous=[.1, .1]) for i in range(3)]
        actions = np.ones((2, 2))*.1
        video = predictor.predict_video(entries, actions)
        expected = predictor.predict_context(np.zeros((3, 3, 32, 32)),
                                               [[0., 0.], [.1, .1], [.2, .2]], [[.1, .1], [.1, .1]], actions)
        np.testing.assert_array_equal(video, np.rint(expected['frames'].mean(0).transpose(0, 2, 3, 1)*255).astype('uint8'))
        for key, value in [('step_id', 5), ('episode_id', 'other'), ('semantics', {}),
                           ('action_from_previous', None)]:
            broken = [dict(entry) for entry in entries]
            broken[1][key] = value
            with self.assertRaises(ValueError):
                predictor.predict_video(broken, actions)


class TemporalTrainingTests(unittest.TestCase):
    def test_context_train_finetune_calibrate_and_evaluate_use_identical_protocol(self):
        from wmal.training.visual_trainer import VisualTrainingConfig, train_visual, evaluate_visual
        from wmal.models.horizon_calibration import calibrate_visual
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_dataset(root)
            config = VisualTrainingConfig(epochs=1, members=2, batch_size=16, horizon=2,
                                          architecture='categorical_rssm', latent_dim=16,
                                          hidden_dim=24, stoch=3, classes=4, context_steps=2)
            report = train_visual(manifest, root/'train', config)
            self.assertTrue(np.isfinite(report['best_validation_loss']))
            trained = VisualWorldModel.load(root/'train/best.pt')
            self.assertEqual(trained.config.context_steps, 2)
            train_visual(manifest, root/'fine', config, pretrained=root/'train/best.pt', freeze_encoder=True)
            checkpoint = root/'fine/best.pt'
            calibration = calibrate_visual(manifest, checkpoint, root/'bounds.json', horizon=2, alpha=.2)
            self.assertEqual(calibration.model_version, VisualWorldModel.load(checkpoint).version)
            result = evaluate_visual(manifest, checkpoint, horizon=2, calibration=root/'bounds.json')
            self.assertEqual(result['windows'], 14)
            self.assertEqual(result['context_steps'], 2)
            self.assertTrue(np.isfinite(result['state_rmse_by_horizon']).all())
            from dataclasses import replace
            with self.assertRaisesRegex(ValueError, 'architecture'):
                train_visual(manifest, root/'bad', replace(config, context_steps=0), pretrained=checkpoint)

    def test_context_loss_ignores_burnin_padding_as_targets(self):
        from wmal.training.visual_trainer import VisualTrainingConfig, batch_loss
        from torch.utils.data import DataLoader
        predictor = model()
        member = predictor.members[0]
        member.eval()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = VisualDataset(make_dataset(Path(tmp)), 'train', horizon=3, context_steps=2)
            batch = next(iter(DataLoader(dataset, batch_size=1)))
            config = VisualTrainingConfig(epochs=1, members=2, horizon=3, context_steps=2,
                                          architecture='categorical_rssm', latent_dim=16, hidden_dim=24)
            before, components = batch_loss(member, batch, predictor.normalization, config)
            # Padding is reset before the actual current frame; cannot supervise or condition it.
            batch['rgb'][:, :2] = 1.
            batch['states'][:, :2] = 999.
            batch['actions'][:, :2] = -999.
            after, changed = batch_loss(member, batch, predictor.normalization, config)
            torch.testing.assert_close(before, after)
            self.assertEqual(components, changed)
