"""Behavioral regressions for action-conditioned visual prediction and trust."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def make_dataset(root):
    from wmal.datasets.visual_sequences import write_episode, write_manifest
    rng = np.random.default_rng(7)
    rows = []
    for split, count in [('train', 4), ('validation', 2), ('calibration', 5), ('test', 2)]:
        for i in range(count):
            actions = rng.uniform(-.15, .15, (8, 2)).astype('float32')
            states = np.concatenate([np.zeros((1, 2)), np.cumsum(actions, axis=0)]).astype('float32')
            rgb = np.zeros((9, 32, 32, 3), dtype='uint8')
            for t, state in enumerate(states):
                x = int(np.clip(16 + state[0] * 20, 2, 28))
                rgb[t, 10:18, x:x+3, 0] = 255
            episode = f'{split}_{i}'
            path = root / f'{episode}.npz'
            write_episode(path, rgb, states, actions, overwrite=True)
            rows.append({'episode_id': episode, 'split': split, 'path': path.name})
    return write_manifest(root/'manifest.json', rows, action_schema='test.delta.v1',
                          action_order=['x', 'y'], state_order=['x', 'y'],
                          camera='front', period_s=.2, image_size=32, source={'kind': 'controlled_test'})


class VisualDataTests(unittest.TestCase):
    def test_visual_dataset_capability_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('wmal.datasets.visual_sequences'))

    def test_windows_never_cross_episode_and_stats_ignore_test(self):
        from wmal.datasets.visual_sequences import VisualDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = VisualDataset(make_dataset(root), 'train', horizon=3)
            self.assertEqual(len(dataset), 24)
            sample = dataset[0]
            self.assertEqual(sample['rgb'].shape, (4, 3, 32, 32))
            self.assertEqual(sample['actions'].shape, (3, 2))
            norm = dataset.normalization()
            np.savez(root/'test_0.npz', rgb=np.zeros((9,32,32,3), dtype='uint8'),
                     states=np.ones((9,2))*1000, actions=np.ones((8,2))*1000)
            np.testing.assert_array_equal(dataset.normalization()['state_mean'], norm['state_mean'])

    def test_normalization_does_not_decompress_rgb_frames(self):
        from unittest.mock import patch
        from wmal.datasets.visual_sequences import VisualDataset
        with tempfile.TemporaryDirectory() as tmp:
            dataset = VisualDataset(make_dataset(Path(tmp)), 'train', horizon=3)
            original = np.lib.npyio.NpzFile.__getitem__

            def reject_rgb(archive, key):
                if key == 'rgb':
                    raise AssertionError('normalization decompressed RGB frames')
                return original(archive, key)

            with patch.object(np.lib.npyio.NpzFile, '__getitem__', reject_rgb):
                stats = dataset.normalization()
            self.assertEqual(set(stats), {'state_mean', 'state_scale', 'action_mean', 'action_scale'})

    def test_event_supervision_profile_reads_only_event_mask(self):
        from unittest.mock import patch
        from wmal.datasets.visual_sequences import VisualDataset, write_episode, write_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_dataset(root)
            payload = json.loads(manifest.read_text())
            for row in payload['episodes']:
                path = root / row['path']
                with np.load(path, allow_pickle=False) as arrays:
                    rgb, states, actions = (arrays['rgb'].copy(), arrays['states'].copy(),
                                            arrays['actions'].copy())
                events = np.zeros((8, 1), dtype=np.float32)
                event_mask = np.ones_like(events)
                write_episode(path, rgb, states, actions, events=events,
                              event_mask=event_mask, overwrite=True)
            semantics = dict(payload['semantics'])
            semantics.pop('event_names', None)
            write_manifest(manifest, payload['episodes'], source=payload['source'],
                           event_names=['seen'], **semantics)
            dataset = VisualDataset(manifest, 'train', horizon=3)
            original = np.lib.npyio.NpzFile.__getitem__

            def reject_rgb(archive, key):
                if key == 'rgb':
                    raise AssertionError('event supervision scan decompressed RGB frames')
                return original(archive, key)

            with patch.object(np.lib.npyio.NpzFile, '__getitem__', reject_rgb):
                mask = dataset.event_supervision_for_episode(0)
            np.testing.assert_array_equal(mask, [True])

    def test_event_context_is_aligned_to_prediction_origin_and_missing_reset_is_masked(self):
        from wmal.datasets.visual_sequences import VisualDataset, write_episode, write_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_dataset(root)
            payload = json.loads(manifest.read_text())
            rows = payload['episodes']
            for row in rows:
                path = root / row['path']
                with np.load(path, allow_pickle=False) as arrays:
                    rgb, states, actions = arrays['rgb'].copy(), arrays['states'].copy(), arrays['actions'].copy()
                events = np.zeros((8, 1), dtype=np.float32)
                events[0, 0] = 1.0
                write_episode(path, rgb, states, actions, events=events,
                              event_mask=np.ones_like(events), overwrite=True)
            semantics = dict(payload['semantics'])
            semantics.pop('event_names', None)
            write_manifest(manifest, rows, source=payload['source'],
                           event_names=['grasped'], **semantics)
            dataset = VisualDataset(manifest, 'validation', horizon=2)
            reset_window, after_transition = dataset[0], dataset[1]
            np.testing.assert_array_equal(reset_window['event_context'], [0.])
            np.testing.assert_array_equal(reset_window['event_context_mask'], [0.])
            np.testing.assert_array_equal(after_transition['event_context'], [1.])
            np.testing.assert_array_equal(after_transition['event_context_mask'], [1.])
            np.testing.assert_array_equal(after_transition['events'][0], [0.])

    def test_event_state_archive_rejects_future_label_misalignment(self):
        from wmal.datasets.visual_sequences import write_episode
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = np.zeros((3, 32, 32, 3), dtype=np.uint8)
            states = np.zeros((3, 1), dtype=np.float32)
            actions = np.zeros((2, 1), dtype=np.float32)
            events = np.asarray([[1.], [0.]], dtype=np.float32)
            event_states = np.zeros((3, 1), dtype=np.float32)
            with self.assertRaisesRegex(ValueError, 'transition labels'):
                write_episode(root/'misaligned.npz', rgb, states, actions,
                              events=events, event_mask=np.ones_like(events),
                              extra_arrays={'event_states': event_states,
                                            'event_state_mask': np.ones_like(event_states)})

    def test_event_persistence_metrics_use_origin_label_not_first_future_label(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import wmal.training.visual_trainer as trainer
        from wmal.datasets.visual_sequences import write_episode, write_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_dataset(root)
            payload = json.loads(manifest.read_text())
            rows = payload['episodes']
            for row in rows:
                path = root / row['path']
                with np.load(path, allow_pickle=False) as arrays:
                    rgb, states, actions = arrays['rgb'].copy(), arrays['states'].copy(), arrays['actions'].copy()
                events = np.zeros((8, 1), dtype=np.float32)
                events[0, 0] = 1.0
                event_states = np.concatenate((np.zeros((1, 1), dtype=np.float32), events), axis=0)
                write_episode(path, rgb, states, actions, events=events,
                              event_mask=np.ones_like(events),
                              extra_arrays={'event_states': event_states,
                                            'event_state_mask': np.ones_like(event_states)},
                              overwrite=True)
            semantics = dict(payload['semantics'])
            semantics.pop('event_names', None)
            manifest = write_manifest(manifest, rows, source=payload['source'],
                                      event_names=['grasped'], **semantics)
            semantics = json.loads(manifest.read_text())['semantics']

            class FixedEventModel:
                version = 'fixed-event-model'
                config = SimpleNamespace(context_steps=0)
                metadata = {'episode_lineage': {
                    'schema': 'wmal.episode_lineage.v1',
                    'train': {'episode_ids': [], 'sha256': []},
                    'selection': {'episode_ids': ['validation_0', 'validation_1'], 'sha256': []}}}

                def __init__(self):
                    self.semantics = semantics

                def predict_sample(self, sample, actions=None):
                    horizon = len(sample['actions'])
                    return {
                        'states': np.zeros((1, horizon, sample['states'].shape[-1]), dtype=np.float32),
                        'frames': np.zeros((1, horizon, 3, 32, 32), dtype=np.float32),
                        'event_probabilities': np.full((1, horizon, 1), .8, dtype=np.float32),
                        'event_probability_names': ['grasped'],
                    }

            with patch.object(trainer.VisualWorldModel, 'load', return_value=FixedEventModel()):
                report = trainer.evaluate_visual(manifest, root / 'unused.pt',
                                                  horizon=2, split='validation')
            metrics = report['event_prediction_metrics']
            self.assertAlmostEqual(metrics['persistence_brier_by_horizon'][0], 2. / 7., places=6)
            self.assertEqual(metrics['scored_windows_by_horizon'], [14, 14])
            self.assertEqual(metrics['scored_labels_by_horizon'], [14, 14])

    def test_duplicate_episode_or_content_and_wrong_alignment_are_rejected(self):
        from wmal.datasets.visual_sequences import VisualDataset, write_episode
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_dataset(root)
            payload = json.loads(manifest.read_text())
            payload['episodes'][-1]['episode_id'] = payload['episodes'][0]['episode_id']
            manifest.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                VisualDataset(manifest, 'train', horizon=3)
            manifest = make_dataset(root)
            payload = json.loads(manifest.read_text())
            payload['episodes'][-1]['sha256'] = payload['episodes'][0]['sha256']
            manifest.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                VisualDataset(manifest, 'train', horizon=3)
            with self.assertRaises(ValueError):
                write_episode(root/'bad.npz', np.zeros((3,32,32,3), dtype='uint8'),
                              np.zeros((3,2)), np.zeros((3,2)))


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional learning dependencies not installed')
class VisualNetworkTests(unittest.TestCase):
    def test_finetune_appends_named_event_channel_and_preserves_old_head_rows(self):
        import torch
        from wmal.datasets.visual_sequences import write_manifest
        from wmal.training.visual_trainer import VisualTrainingConfig, train_visual
        from wmal.models.visual_latent import VisualWorldModel
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=make_dataset(root)
            payload=json.loads(manifest.read_text())
            parent_events=['bilateral_grasp_contact','object_lifted','stable_stack']
            for row in payload['episodes']:
                path=root/row['path']
                with np.load(path) as data:
                    arrays={key:data[key] for key in data.files}
                arrays['events']=np.zeros((8,3),dtype=np.float32)
                arrays['event_mask']=np.ones((8,3),dtype=np.float32)
                np.savez(path,**arrays)
            semantics=dict(payload['semantics'])
            semantics.pop('event_names',None)
            write_manifest(manifest,payload['episodes'],**semantics,
                           source=payload['source'],event_names=parent_events)
            config=VisualTrainingConfig(epochs=1,members=2,batch_size=16,horizon=2,
                                        latent_dim=16,hidden_dim=24,learning_rate=.001,weight_decay=0.)
            train_visual(manifest,root/'parent',config)
            parent=VisualWorldModel.load(root/'parent/best.pt')
            parent_rows=[member.event_head.weight.detach().clone() for member in parent.members]

            payload=json.loads(manifest.read_text())
            next_events=parent_events+['object_dropped_after_lift']
            for row in payload['episodes']:
                path=root/row['path']
                with np.load(path) as data:
                    arrays={key:data[key] for key in data.files}
                extra=np.zeros((8,1),dtype=np.float32)
                extra_mask=np.zeros((8,1),dtype=np.float32)
                if row['split']=='train':
                    extra[:,0]=1.
                    extra_mask[:,0]=1.
                arrays['events']=np.concatenate((arrays['events'],extra),axis=1)
                arrays['event_mask']=np.concatenate((np.zeros_like(arrays['event_mask']),extra_mask),axis=1)
                np.savez(path,**arrays)
            semantics=dict(payload['semantics'])
            semantics.pop('event_names',None)
            write_manifest(manifest,payload['episodes'],**semantics,
                           source=payload['source'],event_names=next_events)

            train_visual(manifest,root/'expanded',config,pretrained=root/'parent/best.pt')
            expanded=VisualWorldModel.load(root/'expanded/best.pt')
            self.assertEqual(expanded.semantics['event_names'],next_events)
            self.assertEqual(expanded.metadata['event_supervised_channels'],[True,True,True,True])
            for old_rows,member in zip(parent_rows,expanded.members):
                torch.testing.assert_close(old_rows,member.event_head.weight[:3],rtol=0,atol=0)
            prediction=expanded.predict(np.zeros((3,32,32)),np.zeros(2),np.zeros((2,2)))
            self.assertEqual(prediction['event_probability_names'],next_events)
            self.assertEqual(prediction['event_probabilities'].shape,(2,2,4))

    def test_finetune_retains_ancestor_exclusions_across_repartitioned_manifests(self):
        from wmal.training.visual_trainer import VisualTrainingConfig, train_visual, evaluate_visual
        from wmal.models.horizon_calibration import calibrate_visual
        from wmal.models.visual_latent import VisualWorldModel
        from wmal.logging.manifest import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=make_dataset(root)
            config=VisualTrainingConfig(epochs=1,members=2,batch_size=16,horizon=2,
                                       latent_dim=16,hidden_dim=24)
            train_visual(manifest,root/'parent',config)
            payload=json.loads(manifest.read_text())
            # Former training/selection episodes become calibration/test; train on new episodes.
            for row in payload['episodes']:
                if row['episode_id']=='train_0': row['split']='calibration'
                elif row['episode_id']=='validation_0': row['split']='test'
                elif row['episode_id']=='calibration_0': row['split']='train'
                elif row['episode_id']=='test_0': row['split']='validation'
            changed=root/'repartitioned.json'
            atomic_json(changed,payload)
            train_visual(changed,root/'fine',config,pretrained=root/'parent/best.pt')
            checkpoint=root/'fine/best.pt'
            with self.assertRaisesRegex(ValueError,'overlap'):
                calibrate_visual(changed,checkpoint,root/'invalid_calibration.json',horizon=2,alpha=.2)
            with self.assertRaisesRegex(ValueError,'overlap'):
                evaluate_visual(changed,checkpoint,horizon=2)
            # A further fine-tune must still reject ancestor training as validation.
            for row in payload['episodes']:
                if row['episode_id']=='train_0': row['split']='validation'
            atomic_json(changed,payload)
            with self.assertRaisesRegex(ValueError,'overlap'):
                train_visual(changed,root/'grandchild',config,pretrained=checkpoint)
            # Renaming an episode does not erase its ancestor content identity.
            for row in payload['episodes']:
                if row['episode_id']=='train_0': row['episode_id']='renamed_ancestor_train'
            atomic_json(changed,payload)
            with self.assertRaisesRegex(ValueError,'overlap'):
                train_visual(changed,root/'renamed',config,pretrained=checkpoint)
            # Old fine-tuned checkpoints with unknown ancestry must fail closed.
            model=VisualWorldModel.load(checkpoint)
            model.metadata.pop('episode_lineage',None)
            model.save(root/'unknown_lineage.pt')
            with self.assertRaisesRegex(ValueError,'lineage'):
                evaluate_visual(manifest,root/'unknown_lineage.pt',horizon=2)

    def test_zero_residual_preserves_background_instead_of_relearning_static_world(self):
        import torch
        from wmal.models.visual_latent import NetworkConfig, VisualLatentMember
        model=VisualLatentMember(NetworkConfig(2,2,image_size=32,latent_dim=16,hidden_dim=24))
        for parameter in model.parameters():
            parameter.data.zero_()
        rgb=torch.rand(1,3,32,32)
        frames=model.imagine(rgb,torch.zeros(1,2),torch.zeros(1,2,2))['rgb']
        torch.testing.assert_close(frames[:,0],rgb)
        torch.testing.assert_close(frames[:,1],rgb)

    def test_imagination_depends_on_actions_without_future_observations(self):
        import torch
        from wmal.models.visual_latent import NetworkConfig, VisualLatentMember
        torch.manual_seed(4)
        torch.set_num_threads(1)
        model = VisualLatentMember(NetworkConfig(2, 2, image_size=32, latent_dim=16, hidden_dim=24))
        initial_rgb = torch.zeros(2,3,32,32)
        state = torch.zeros(2,2)
        actions = torch.stack([torch.zeros(3,2), torch.ones(3,2)*.1])
        prediction = model.imagine(initial_rgb, state, actions)
        self.assertEqual(tuple(prediction['rgb'].shape), (2,3,3,32,32))
        self.assertGreater(float((prediction['state'][0]-prediction['state'][1]).abs().max().detach()), 1e-5)
        self.assertGreater(float((prediction['rgb'][0]-prediction['rgb'][1]).abs().max().detach()), 1e-6)
        (prediction['rgb'].mean()+prediction['state'].mean()).backward()
        self.assertGreater(float(model.transition.weight_ih.grad.abs().sum()), 0.)

    def test_training_finetune_freeze_roundtrip_and_test_isolation(self):
        from wmal.training.visual_trainer import VisualTrainingConfig, train_visual, evaluate_visual
        from wmal.models.visual_latent import VisualWorldModel
        from wmal.datasets.visual_sequences import VisualDataset
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=make_dataset(root)
            config=VisualTrainingConfig(epochs=3, members=2, batch_size=8, horizon=2,
                                       latent_dim=16, hidden_dim=24, seed=5, learning_rate=.003)
            report=train_visual(manifest, root/'run', config)
            self.assertTrue(np.isfinite(report['best_validation_loss']))
            model=VisualWorldModel.load(root/'run/best.pt')
            sample=VisualDataset(manifest,'test',horizon=2)[0]
            result=model.predict(sample['rgb'][0],sample['states'][0],sample['actions'])
            self.assertEqual(result['states'].shape,(2,2,2))
            self.assertEqual(result['frames'].shape,(2,2,3,32,32))
            model.save(root/'copy.pt')
            restored=VisualWorldModel.load(root/'copy.pt')
            self.assertEqual(model.version,restored.version)
            np.testing.assert_array_equal(result['states'],restored.predict(
                sample['rgb'][0],sample['states'][0],sample['actions'])['states'])
            metrics=evaluate_visual(manifest,root/'run/best.pt',horizon=2)
            self.assertEqual(metrics['split'],'test')
            self.assertEqual(len(metrics['frame_mse_by_horizon']),2)
            self.assertIn('persistence_frame_mse_by_horizon',metrics)
            self.assertTrue(np.all(np.array(metrics['zero_action_frame_sensitivity_by_horizon'])>0))
            payload=json.loads(manifest.read_text())
            for row in payload['episodes']:
                if row['split']=='test':
                    with np.load(root/row['path']) as f:
                        values={key:f[key] for key in f.files}
                    values['states']+=100
                    np.savez(root/row['path'],**values)
            isolated=train_visual(manifest,root/'isolated',config)
            self.assertEqual(report['model_version'], isolated['model_version'])
            fine=train_visual(manifest,root/'fine',config,pretrained=root/'run/best.pt',freeze_encoder=True)
            tuned=VisualWorldModel.load(root/'fine/best.pt')
            for before,after in zip(model.members,tuned.members):
                for key,value in before.encoder.state_dict().items():
                    self.assertTrue(torch.equal(value,after.encoder.state_dict()[key]))


class HorizonCalibrationTests(unittest.TestCase):
    def test_episode_quantile_not_frame_quantile_and_model_binding(self):
        from wmal.models.horizon_calibration import fit_horizon_calibration
        rows=[]
        for i,error in enumerate([1.,2.,3.,4.,5.]):
            for _ in range(20 if i==0 else 1):
                rows.append({'episode_id':str(i),'split':'calibration',
                             'scores':[error,error*2]})
        calibration=fit_horizon_calibration(rows, model_version='v1', dataset_id='d',
                                             alpha=.2, state_scale=[1.,1.], std_floor=.1,
                                             semantics={'action_schema':'test.delta.v1'})
        self.assertEqual(calibration.quantiles,(5.,10.))
        np.testing.assert_allclose(calibration.bounds(np.zeros((2,2)),version='v1'),
                                   [[.5,.5],[1.,1.]])
        with self.assertRaises(ValueError):
            calibration.bounds(np.zeros((2,2)),version='v2')
        rows[0]['split']='test'
        with self.assertRaises(ValueError):
            fit_horizon_calibration(rows,model_version='v1',dataset_id='d',alpha=.2,
                                    state_scale=[1.,1.],std_floor=.1,semantics={})


if __name__=='__main__':
    unittest.main()
