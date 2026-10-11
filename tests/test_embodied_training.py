"""Task4: real updates, detached selection, exposure and optimizer-boundary replay."""
import contextlib
import copy
import importlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from wmal.datasets.embodied_profiles import ObservationProfile
from wmal.datasets.embodied_sequences import EmbodiedDataset, write_embodied_manifest
from wmal.models.embodied_config import ScalableConfig
from wmal.models.embodied_rssm import ScalableRSSMMember
from wmal.models.embodied_world import EmbodiedWorldModel


def fixture(root, *, shift=0, steps=5, events=True, image_size=16):
    """Original local data only; literal action/command alignment and four splits."""
    root.mkdir(parents=True, exist_ok=True)
    profile = ObservationProfile(physical_order=['q'], command_order=['q_target'],
        action_order=['q'], action_schema='fixture.delta.v1', camera_ids=['front'],
        image_size=image_size, period_s=.2, physical_units=['rad'], command_units=['rad'],
        action_units=['rad'], command_rule='joint_delta', event_names=['contact'] if events else [],
        controller={'profile': 'fixture_pd'}, origin={'dataset': f'fixture-{shift}'})
    rows = []
    # Three train originals; validation contains startup windows as well as later origins.
    for i, split in enumerate(['train'] * 3 + ['validation', 'calibration', 'test', 'calibration']):
        action = np.full((steps, 1), .25, dtype='float32')
        commands = 10 + shift + i + np.arange(steps + 1)[:, None] * .25
        arrays = dict(rgb=np.full((steps + 1, 1, image_size, image_size, 3), 20 + i + shift, dtype='uint8'),
            physical=shift + i + np.arange(steps + 1)[:, None] * .125,
            commands=commands, actions=action)
        if events:
            arrays.update(events=np.zeros((steps, 1), dtype='float32'),
                          event_mask=np.full((steps, 1), split == 'train', dtype=bool))
        name = f'{i}.npz'; np.savez_compressed(root / name, **arrays)
        rows.append(dict(path=name, episode_id=f'e-{shift}-{i}', split=split,
            source_episode_id=f'original-{shift}-{i}', source_sha256=f'{1000+shift*10+i:064x}'))
    return write_embodied_manifest(root / 'manifest.json', rows, profile,
        {'task': 'fixture_task', 'controller_profile': profile.controller,
         'prediction_scope': {'action_lower': [-1], 'action_upper': [1]}})


def tiny_config(api, **overrides):
    network = ScalableConfig(physical_dim=1, command_dim=1, action_dim=1, views=1,
        image_size=16, event_dim=1, context_steps=1, horizon=2, deter=8, hidden=16,
        stoch=2, classes=3, blocks=2, encoder_channels=(4, 8), fusion_width=8, decoder_width=4)
    values = dict(model=network, epochs=2, members=1, batch_size=1, accumulation=2,
        lr=.001, seed=71, sequence_length=4, horizon=2, context_steps=1,
        sample_count=1, device='cpu')
    return api.EmbodiedTrainingConfig(**(values | overrides))


def tensors_equal(test, left, right):
    test.assertEqual(left.keys(), right.keys())
    for name in left:
        if isinstance(left[name], torch.Tensor):
            test.assertTrue(torch.equal(left[name], right[name]), name)
        elif isinstance(left[name], dict):
            tensors_equal(test, left[name], right[name])
        else:
            test.assertEqual(left[name], right[name], name)


class Interrupted(Exception):
    pass


class EmbodiedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.manifest = fixture(self.root / 'data')

    def api(self):
        try:
            return importlib.import_module('wmal.training.embodied_trainer')
        except ModuleNotFoundError as exc:
            self.fail(f'Task4 real training API missing: {exc.name}')

    # A trainer that exports last/live weights, invents ensemble copies, or selects KL fails.
    def test_real_smoke_train_select_and_load(self):
        api = self.api(); cfg = tiny_config(api, members=2)
        out = self.root / 'run'
        report = api.train_embodied(self.manifest, out, cfg, show_progress=False)
        model = EmbodiedWorldModel.load(out / 'model.pt')
        self.assertEqual(len(model.members), 2)
        sample = EmbodiedDataset(self.manifest, 'validation', horizon=2, context_steps=1)[0]
        result = model.predict_sample(sample, samples=8)
        self.assertEqual(result['physical_states'].shape, (2, 8, 2, 1))
        self.assertEqual(model.event_supervised_channels, [True])
        self.assertEqual(model.metadata['training_source']['task'], 'fixture_task')
        self.assertEqual(model.metadata['prediction_scope']['action_lower'], [-1])
        self.assertEqual(report['selection_metric'], 'mean_per_horizon_dimension_normalized_rmse')
        for member in report['members']:
            values = [row['selection_metric'] for row in member['history']]
            self.assertEqual(member['selected_epoch'], values.index(min(values)) + 1)
            self.assertEqual(member['validation_unique_samples'], 4)
            self.assertGreater(member['trainable_parameters'], 0)
            self.assertEqual(member['optimizer_steps'], 10)
        model.assert_frozen()
        with self.assertRaises(ValueError):
            api.train_embodied(self.manifest, out, cfg, show_progress=False)

    # Missing rank RNG/optimizer/sampler/position restoration changes real subsequent tensors.
    def test_resume_matches_uninterrupted_step(self):
        api = self.api(); ckpt = importlib.import_module('wmal.training.embodied_checkpoint')
        cfg = tiny_config(api); complete = self.root / 'complete'; partial = self.root / 'partial'
        api.train_embodied(self.manifest, complete, cfg, show_progress=False)
        reference = torch.load(complete / 'training.pt', weights_only=True)
        expected_rng = (random.random(), np.random.rand(), torch.rand(3))
        save = ckpt.save_training_checkpoint
        def stop(path, payload):
            save(path, payload)
            raise Interrupted()
        with patch.object(api, 'save_training_checkpoint', side_effect=stop):
            with self.assertRaises(Interrupted):
                api.train_embodied(self.manifest, partial, cfg, show_progress=False)
        first = torch.load(partial / 'training.pt', weights_only=True)
        self.assertEqual(first['position']['step'], 1)
        api.train_embodied(self.manifest, partial, cfg, resume=partial / 'training.pt', show_progress=False)
        actual = torch.load(partial / 'training.pt', weights_only=True)
        tensors_equal(self, reference['model'], actual['model'])
        tensors_equal(self, reference['optimizer']['state'], actual['optimizer']['state'])
        self.assertEqual(reference['rank_states'], actual['rank_states'])
        self.assertEqual(reference['position'], actual['position'])
        self.assertEqual(expected_rng[:2], (random.random(), np.random.rand()))
        self.assertTrue(torch.equal(expected_rng[2], torch.rand(3)))
        self.assertEqual(EmbodiedWorldModel.load(complete / 'model.pt').version,
                         EmbodiedWorldModel.load(partial / 'model.pt').version)

    # RNG equality alone misses nondeterministic convolution/reduction algorithms.
    def test_deterministic_pool_preserves_forward_and_gradient(self):
        from wmal.training.embodied_loss import DeterministicAdaptivePool2d
        for size in (2, 4, 7, 8):
            value = torch.arange(size*size, dtype=torch.float64).reshape(1, 1, size, size).requires_grad_()
            original = torch.nn.AdaptiveAvgPool2d((4, 4))(value)
            replacement = DeterministicAdaptivePool2d((4, 4))(value)
            self.assertTrue(torch.equal(original, replacement))
            gradient = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
            expected = torch.autograd.grad(original, value, gradient)[0]
            actual = torch.autograd.grad(replacement, value, gradient)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires actual CUDA hardware')
    def test_cuda_t32_bf16_tensor_exact_resume(self):
        api = self.api()
        from wmal.training.embodied_checkpoint import save_training_checkpoint
        manifest = fixture(self.root / 'cuda-data', steps=32, image_size=32)
        network = ScalableConfig(physical_dim=1, command_dim=1, action_dim=1,
            views=1, image_size=32, event_dim=1, context_steps=2, horizon=4,
            deter=64, hidden=128, stoch=8, classes=8, blocks=4,
            encoder_channels=(16, 32, 64), fusion_width=64, decoder_width=32)
        cfg = tiny_config(api, model=network, epochs=1, batch_size=2,
            sequence_length=32, horizon=4, context_steps=2, sample_count=8,
            device='cuda', precision='bfloat16')
        full, partial = self.root / 'cuda-full', self.root / 'cuda-partial'
        before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark)
        api.train_embodied(manifest, full, cfg, show_progress=False)
        expected = torch.load(full / 'training.pt', weights_only=True)
        next_rng = torch.rand(4, device='cuda')
        def stop(path, payload):
            save_training_checkpoint(path, payload)
            raise Interrupted()
        with patch.object(api, 'save_training_checkpoint', side_effect=stop):
            with self.assertRaises(Interrupted):
                api.train_embodied(manifest, partial, cfg, show_progress=False)
        self.assertEqual(torch.load(partial / 'training.pt', weights_only=True)['position']['step'], 1)
        api.train_embodied(manifest, partial, cfg, resume=partial / 'training.pt', show_progress=False)
        actual = torch.load(partial / 'training.pt', weights_only=True)
        tensors_equal(self, expected['model'], actual['model'])
        tensors_equal(self, expected['optimizer']['state'], actual['optimizer']['state'])
        self.assertEqual(expected['rank_states'], actual['rank_states'])
        self.assertTrue(torch.equal(next_rng, torch.rand(4, device='cuda')))
        self.assertEqual(before, (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark))

    def test_resume_rejects_changed_data(self):
        api = self.api(); cfg = tiny_config(api, epochs=1); out = self.root / 'run'
        api.train_embodied(self.manifest, out, cfg, show_progress=False)
        other = fixture(self.root / 'other', shift=10)
        with self.assertRaisesRegex(ValueError, 'identity|data'):
            api.train_embodied(other, out, cfg, resume=out / 'training.pt', show_progress=False)
        with self.assertRaisesRegex(ValueError, 'config|identity'):
            api.train_embodied(self.manifest, out, tiny_config(api, epochs=1, lr=.002),
                              resume=out / 'training.pt', show_progress=False)

    # A completed member's model/optimizer must never seed the next independent member.
    def test_resume_at_member_boundary_including_finetune(self):
        api = self.api(); from wmal.training.embodied_checkpoint import save_training_checkpoint
        cfg = tiny_config(api, epochs=1, members=2)
        parent = self.root / 'parent'
        api.train_embodied(self.manifest, parent, cfg, show_progress=False)
        other = fixture(self.root / 'other', shift=20)
        for label, manifest, pretrained in [('fresh', self.manifest, None), ('fine', other, parent / 'model.pt')]:
            full = self.root / (label+'-full'); partial = self.root / (label+'-partial')
            api.train_embodied(manifest, full, cfg, pretrained=pretrained, show_progress=False)
            expected = EmbodiedWorldModel.load(full / 'model.pt')
            def stop(path, payload):
                save_training_checkpoint(path, payload)
                if payload['position']['member'] == 1 and payload['run']['current'] is None:
                    raise Interrupted()
            with patch.object(api, 'save_training_checkpoint', side_effect=stop):
                with self.assertRaises(Interrupted):
                    api.train_embodied(manifest, partial, cfg, pretrained=pretrained, show_progress=False)
            api.train_embodied(manifest, partial, cfg, resume=partial / 'training.pt', show_progress=False)
            actual = EmbodiedWorldModel.load(partial / 'model.pt')
            self.assertEqual(expected.version, actual.version, label)
            for left, right in zip(expected.members, actual.members):
                tensors_equal(self, left.state_dict(), right.state_dict())

    def test_completed_resume_restores_next_rng(self):
        api = self.api(); cfg = tiny_config(api, epochs=1); out = self.root / 'run'
        api.train_embodied(self.manifest, out, cfg, show_progress=False)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        random.seed(123); np.random.seed(123); torch.manual_seed(123)
        api.train_embodied(self.manifest, out, cfg, resume=out / 'training.pt', show_progress=False)
        self.assertEqual(expected[:2], (random.random(), np.random.rand()))
        self.assertTrue(torch.equal(expected[2], torch.rand(3)))

    # Changing frozen encoders/coordinates, dropping ancestor IDs, or silent shape transfer fails.
    def test_finetune_freeze_and_lineage(self):
        api = self.api(); cfg = tiny_config(api, epochs=1); parent = self.root / 'parent'
        api.train_embodied(self.manifest, parent, cfg, show_progress=False)
        inherited = EmbodiedWorldModel.load(parent / 'model.pt')
        other = fixture(self.root / 'other', shift=20); child = self.root / 'child'
        report = api.train_embodied(other, child, tiny_config(api, epochs=1, freeze_encoder=True),
                                    pretrained=parent / 'model.pt', show_progress=False)
        updated = EmbodiedWorldModel.load(child / 'model.pt')
        self.assertEqual(inherited.normalization, updated.normalization)
        for name, tensor in inherited.members[0].state_dict().items():
            if name.startswith(('encoder.', 'physical_encoder.', 'command_encoder.', 'fusion.')):
                self.assertTrue(torch.equal(tensor, updated.members[0].state_dict()[name]), name)
        self.assertIn('original-0-0', updated.metadata['episode_lineage']['train']['episode_ids'])
        self.assertIn('original-20-0', updated.metadata['episode_lineage']['train']['episode_ids'])
        self.assertGreater(report['members'][0]['frozen_parameters'], 0)
        with self.assertRaisesRegex(ValueError, 'ancestor|overlap'):
            api.train_embodied(self.manifest, self.root / 'leak', cfg,
                               pretrained=parent / 'model.pt', show_progress=False)
        mismatch = copy.copy(cfg.model)
        from dataclasses import replace
        with self.assertRaisesRegex(ValueError, 'architecture|config|matching'):
            api.train_embodied(other, self.root / 'wrong', tiny_config(api, model=replace(mismatch, hidden=24)),
                               pretrained=parent / 'model.pt', show_progress=False)

    # Future values/masks must change supervision, never the candidate prior's inputs.
    def test_dynamic_mask_is_not_rollout_input(self):
        api = self.api(); loss_api = importlib.import_module('wmal.training.embodied_loss')
        cfg = tiny_config(api); data = EmbodiedDataset(self.manifest, 'train', horizon=3)
        sample = data[0]; sample['origin'] = 1; member = ScalableRSSMMember(cfg.model)
        captured = []
        real = member.imagine_context
        def capture(*args, **kwargs):
            result = real(*args, **kwargs)
            captured.append((args, result['physical'].detach().clone()))
            return result
        with patch.object(member, 'imagine_context', side_effect=capture):
            first, _ = loss_api.embodied_loss(member, [sample], data.normalization(), cfg)
            changed = copy.deepcopy(sample); changed['rgb'][2:] = 1; changed['physical'][2:] = 100
            changed['commands'][2:] = -100; changed['event_mask'][:] = False
            changed['changed_mask'] = np.ones_like(changed['rgb'])
            second, _ = loss_api.embodied_loss(member, [changed], data.normalization(), cfg)
        self.assertTrue(torch.equal(captured[0][1], captured[1][1]))
        self.assertNotEqual(first.item(), second.item())
        # Raw command=10.25 plus actions .25,.25, then normalize; no normalized integration.
        norm = data.normalization()
        expected = torch.tensor([[[10.5], [10.75]]], dtype=torch.float32)
        expected = (expected - torch.tensor(norm['command_mean'])) / torch.tensor(norm['command_scale'])
        torch.testing.assert_close(captured[0][0][5], expected)

    # A tail divided by configured accumulation underweights the final real gradients.
    def test_tail_accumulation_correct(self):
        self.api(); dist = importlib.import_module('wmal.training.embodied_distributed')
        parameter = torch.nn.Parameter(torch.tensor(0.))
        optimizer = torch.optim.SGD([parameter], lr=1.)
        model = torch.nn.Linear(1, 1)
        metrics = dist.optimizer_update(model, optimizer, [1., 3.],
            lambda value: (parameter * value, {}))
        self.assertEqual(parameter.item(), -2.)
        self.assertEqual(metrics['microbatches'], 2)

    # Whole-frame normalization would dilute a one-pixel error by 256.
    def test_changed_region_independent_normalization(self):
        self.api(); api = importlib.import_module('wmal.training.embodied_loss')
        origin = torch.zeros(1, 1, 3, 16, 16)
        target = origin.clone(); target[..., 0, 0] = 1
        prediction = torch.zeros_like(target)
        value, fraction = api.changed_region_loss(prediction, target, origin,
                                                  torch.ones(1, 1, dtype=torch.bool), 1/255)
        self.assertEqual(value.item(), 1.)
        self.assertEqual(fraction.item(), 1/256)
        value, fraction = api.changed_region_loss(origin, origin, origin,
                                                  torch.ones(1, 1, dtype=torch.bool), 1/255)
        self.assertEqual((value.item(), fraction.item()), (0., 0.))

    # Fine-tuned head exposure remains inherited even if this child has no labels.
    def test_finetune_inherits_event_exposure(self):
        api = self.api(); parent = self.root / 'parent'; cfg = tiny_config(api, epochs=1)
        api.train_embodied(self.manifest, parent, cfg, show_progress=False)
        other = fixture(self.root / 'other', shift=40)
        payload = json.loads(other.read_text())
        from wmal.logging.manifest import atomic_json, sha256_file
        for row in payload['episodes']:
            file = other.parent / row['path']
            with np.load(file) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays['event_mask'][:] = False
            np.savez_compressed(file, **arrays); row['sha256'] = sha256_file(file)
        atomic_json(other, payload)
        api.train_embodied(other, self.root / 'child', cfg, pretrained=parent / 'model.pt', show_progress=False)
        child = EmbodiedWorldModel.load(self.root / 'child/model.pt')
        self.assertEqual(child.event_supervised_channels, [True])

    # Merely owning a head or seeing validation labels must never publish probabilities.
    def test_unseen_event_masks_fail_closed(self):
        api = self.api(); from wmal.logging.manifest import atomic_json, sha256_file
        payload = json.loads(self.manifest.read_text())
        for row in payload['episodes']:
            file = self.manifest.parent / row['path']
            with np.load(file) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays['event_mask'][:] = row['split'] != 'train'
            np.savez_compressed(file, **arrays); row['sha256'] = sha256_file(file)
        atomic_json(self.manifest, payload)
        out = self.root / 'unlabeled'
        api.train_embodied(self.manifest, out, tiny_config(api, epochs=1), show_progress=False)
        model = EmbodiedWorldModel.load(out / 'model.pt')
        self.assertEqual(model.event_supervised_channels, [False])
        sample = EmbodiedDataset(self.manifest, 'validation', horizon=2, context_steps=1)[0]
        self.assertIsNone(model.predict_sample(sample)['event_probabilities'])

    # Labels seen only after the selected epoch cannot authorize that earlier head.
    def test_selected_snapshot_owns_event_exposure(self):
        api = self.api(); from wmal.training.embodied_loss import EmbodiedLossForward
        forward, validate = EmbodiedLossForward.forward, api.validate_member
        calls = [0]; epochs = [0]
        def first_epoch_unlabeled(wrapper, batch):
            batch = copy.deepcopy(batch)
            if calls[0] < 9:
                for sample in batch:
                    sample['event_mask'][:] = False
            calls[0] += 1
            return forward(wrapper, batch)
        def earlier_selected(*args, **kwargs):
            result = validate(*args, **kwargs)
            epochs[0] += 1
            # Control only the selection branch after genuine unique validation.
            result['selection_metric'] = float(epochs[0])
            return result
        with patch.object(EmbodiedLossForward, 'forward', first_epoch_unlabeled), \
                patch.object(api, 'validate_member', earlier_selected):
            report = api.train_embodied(self.manifest, self.root / 'run', tiny_config(api), show_progress=False)
        self.assertEqual(report['members'][0]['selected_epoch'], 1)
        self.assertEqual(report['members'][0]['new_event_seen'], [True])
        self.assertEqual(EmbodiedWorldModel.load(self.root / 'run/model.pt').event_supervised_channels, [False])

    # Snapshot optimizer tensors must not alias a live optimizer on the next member/update.
    def test_optimizer_snapshot_is_detached_cpu_copy(self):
        self.api(); from wmal.training.embodied_distributed import DistributedRuntime
        model = torch.nn.Linear(1, 1); optim = torch.optim.AdamW(model.parameters(), lr=.01)
        model(torch.ones(1, 1)).sum().backward(); optim.step(); optim.zero_grad()
        runtime = DistributedRuntime('none', 'cpu')
        weights, state = runtime.snapshot(model, model, optim)
        before = {key: value['exp_avg'].clone() for key, value in state['state'].items()}
        model(torch.ones(1, 1)).sum().backward(); optim.step()
        for key in before:
            self.assertTrue(torch.equal(before[key], state['state'][key]['exp_avg']))

    def test_nonfinite_gradients_do_not_claim_optimizer_update(self):
        self.api(); from wmal.training.embodied_distributed import optimizer_update
        parameter = torch.nn.Parameter(torch.tensor(0.))
        optimizer = torch.optim.SGD([parameter], lr=1.)
        with self.assertRaisesRegex(ValueError, 'gradient|Nonfinite'):
            optimizer_update(torch.nn.Identity(), optimizer, [0], lambda _: (parameter.sqrt(), {}))
        self.assertEqual(parameter.item(), 0.)

    def test_training_reads_only_train_and_selection_arrays(self):
        api = self.api(); from wmal.datasets import embodied_sequences as data
        forbidden = {row['path'] for row in json.loads(self.manifest.read_text())['episodes']
                     if row['split'] in ('calibration', 'test')}
        original = data._read_npz
        accessed = []
        def checked(path):
            self.assertNotIn(Path(path).name, forbidden)
            accessed.append(Path(path).name)
            return original(path)
        with patch.object(data, '_read_npz', checked):
            api.train_embodied(self.manifest, self.root / 'run', tiny_config(api, epochs=1), show_progress=False)
        self.assertEqual(set(accessed), {'0.npz', '1.npz', '2.npz', '3.npz'})

    def test_default_t32_real_sequence_and_uniform_origins(self):
        api = self.api(); source = fixture(self.root / 'long', steps=32)
        cfg = tiny_config(api, sequence_length=32, epochs=1, sample_count=1)
        report = api.train_embodied(source, self.root / 'run', cfg, show_progress=False)
        self.assertEqual(report['sampling']['raw_sequences'], 6)
        self.assertEqual(report['sampling']['actions'], 31)
        exposures = report['members'][0]['exposures']
        self.assertTrue(all(row['sequence_length'] == 32 for row in exposures))
        origins = [row['origin'] for row in exposures]
        self.assertTrue(all(0 <= origin <= 29 for origin in origins))
        self.assertGreater(len(set(origins)), 1)
        self.assertTrue(any(origin != cfg.context_steps for origin in origins))

    def test_task_sampling_retains_requested_tasks_in_bootstrap(self):
        api = self.api(); from wmal.logging.manifest import atomic_json
        payload = json.loads(self.manifest.read_text())
        for index, row in enumerate(payload['episodes'][:3]):
            row['task'] = 'beta' if index == 1 else 'alpha'
        atomic_json(self.manifest, payload)
        data = EmbodiedDataset(self.manifest, 'train', horizon=3)
        sampler = api.EpisodeSampler(data, tiny_config(api, task_sampling={'alpha': 1, 'beta': 1}), 1)
        try:
            draws = sampler.draw(20)
        except ValueError as exc:
            self.fail(f'Explicit task sampling lost a requested task: {exc}')
        self.assertIn('original-0-1', [draw['original_group'] for draw in draws])
        self.assertTrue(any(draw['original_group'] != 'original-0-1' for draw in draws))

    def test_resume_rejects_version_world_size_and_unknown_parent_lineage(self):
        api = self.api(); cfg = tiny_config(api, epochs=1); out = self.root / 'run'
        api.train_embodied(self.manifest, out, cfg, show_progress=False)
        snapshot = torch.load(out / 'training.pt', weights_only=True)
        for field, value in [('world_size', 8), ('environments', [{'torch': 'other'}])]:
            altered = copy.deepcopy(snapshot); altered['identity'][field] = value
            torch.save(altered, out / 'training.pt')
            with self.assertRaisesRegex(ValueError, 'identity|version|world'):
                api.train_embodied(self.manifest, out, cfg, resume=out / 'training.pt', show_progress=False)
        parent = EmbodiedWorldModel.load(out / 'model.pt'); metadata = parent.metadata
        del metadata['episode_lineage']
        unknown = EmbodiedWorldModel(parent.config, parent.profile, parent.normalization, parent.members, metadata=metadata)
        unknown.save(self.root / 'unknown.pt')
        other = fixture(self.root / 'other', shift=60)
        with self.assertRaisesRegex(ValueError, 'Unknown|lineage'):
            api.train_embodied(other, self.root / 'child', cfg, pretrained=self.root / 'unknown.pt', show_progress=False)

    def test_training_metrics_and_actual_exposures_reported(self):
        api = self.api(); report = api.train_embodied(self.manifest, self.root / 'run',
            tiny_config(api, epochs=1), show_progress=False)
        self.assertIn('training_metrics', report['members'][0]['history'][0])
        training = report['members'][0]['history'][0]['training_metrics']
        for key in ('raw_kl', 'clipped_kl', 'kl_clipping_rate', 'changed_fraction', 'physical_rmse', 'baseline_rmse'):
            self.assertIn(key, training)
        exposure = report['members'][0]['exposures'][0]
        self.assertIn('source_episode_id', exposure)
        self.assertIn('original_group', exposure)
        self.assertEqual(report['sampling']['actions'], 3)
        self.assertLessEqual(training['kl_clipping_rate'], 1)
        self.assertGreaterEqual(training['clipped_kl'], training['raw_kl'])

    @unittest.skipUnless(torch.cuda.is_available(), 'Local CUDA hardware unavailable')
    def test_cuda_autocast_and_actual_resume(self):
        api = self.api(); from wmal.training.embodied_checkpoint import save_training_checkpoint
        for precision in ('float32', 'bfloat16', 'float16'):
            cfg = tiny_config(api, epochs=1, device='cuda', precision=precision)
            full = self.root / precision; partial = self.root / (precision+'-partial')
            api.train_embodied(self.manifest, full, cfg, show_progress=False)
            reference = torch.load(full / 'training.pt', weights_only=True)
            rng = torch.rand(3, device='cuda')
            def stop(path, payload):
                save_training_checkpoint(path, payload); raise Interrupted()
            with patch.object(api, 'save_training_checkpoint', side_effect=stop):
                with self.assertRaises(Interrupted):
                    api.train_embodied(self.manifest, partial, cfg, show_progress=False)
            api.train_embodied(self.manifest, partial, cfg, resume=partial / 'training.pt', show_progress=False)
            replay = torch.load(partial / 'training.pt', weights_only=True)
            tensors_equal(self, reference['model'], replay['model'])
            tensors_equal(self, reference['optimizer']['state'], replay['optimizer']['state'])
            self.assertEqual(reference['rank_states'], replay['rank_states'])
            self.assertTrue(torch.equal(rng, torch.rand(3, device='cuda')))
            self.assertTrue(all(value.dtype == torch.float32 for value in replay['model'].values()))

    def test_cli_real_pipeline(self):
        api = self.api(); cfg = tiny_config(api, epochs=1)
        from dataclasses import asdict
        config = self.root / 'training.json'; config.write_text(json.dumps(asdict(cfg)))
        script = Path(__file__).resolve().parents[1] / 'scripts/embodied_world.py'
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        env.pop('PYTHONPATH', None)
        def run(*args, success=True):
            result = subprocess.run([sys.executable, str(script), *args], capture_output=True,
                                    text=True, env=env, timeout=60)
            self.assertEqual(result.returncode == 0, success, result.stderr)
            return result
        inspected = json.loads(run('inspect', '--manifest', str(self.manifest)).stdout)
        self.assertEqual(inspected['splits']['train']['episodes'], 3)
        out = self.root / 'cli'
        trained = json.loads(run('train', '--manifest', str(self.manifest), '--output', str(out),
                                '--training-config', str(config), '--no-progress').stdout)
        self.assertEqual(trained['members'][0]['optimizer_steps'], 5)
        calibrated = json.loads(run('calibrate', '--manifest', str(self.manifest), '--model', str(out / 'model.pt'),
            '--output', str(self.root / 'calibration.json'), '--alpha', '.8').stdout)
        self.assertEqual(calibrated['episodes'], 2)
        profile = json.loads(run('profile', '--manifest', str(self.manifest),
            '--training-config', str(config), '--output', str(self.root / 'profile.json')).stdout)
        self.assertGreater(profile['update_seconds'], 0)
        self.assertGreater(profile['inference_seconds'], 0)
        self.assertEqual(profile['inference_samples'], 8)
        self.assertEqual(profile['optimizer_steps'], 1)
        self.assertIn('NotImplemented', run('evaluate', success=False).stderr)


if __name__ == '__main__':
    unittest.main()
