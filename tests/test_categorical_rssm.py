"""Behavioral tests: action-prior/observation-posterior separation and learning."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch unavailable')
class CategoricalRSSMTests(unittest.TestCase):
    def test_training_finetune_calibration_and_test_isolation(self):
        import torch
        from test_visual_world import make_dataset
        from wmal.training.visual_trainer import VisualTrainingConfig, train_visual, evaluate_visual
        from wmal.models.visual_latent import VisualWorldModel
        from wmal.models.horizon_calibration import calibrate_visual
        config = VisualTrainingConfig(epochs=1, members=2, batch_size=16, horizon=2,
                                      latent_dim=16, hidden_dim=24, architecture='categorical_rssm',
                                      stoch=4, classes=4, free_nats=0., learning_rate=.003)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_dataset(root)
            report = train_visual(manifest, root/'run', config)
            parts = report['history'][0]['validation_components']
            for key in ('dyn_kl','rep_kl','reconstruction','prior_entropy','posterior_entropy'):
                self.assertTrue(np.isfinite(parts[key]))
            model = VisualWorldModel.load(root/'run/best.pt')
            calibration = calibrate_visual(manifest, root/'run/best.pt', root/'calibration.json',
                                           horizon=2, alpha=.2)
            self.assertEqual(calibration.model_version, model.version)
            result = evaluate_visual(manifest, root/'run/best.pt', horizon=2, calibration=root/'calibration.json')
            self.assertEqual(result['episodes'], 2)
            self.assertTrue(np.isfinite(result['state_rmse_by_horizon']).all())
            # Unexposed final test content cannot change normalization or selected weights.
            with np.load(root/'test_0.npz') as data:
                arrays = {key:data[key] for key in data.files}
            arrays['states'] += 100
            np.savez(root/'test_0.npz', **arrays)
            isolated = train_visual(manifest, root/'isolated', config)
            self.assertEqual(report['model_version'], isolated['model_version'])
            train_visual(manifest, root/'fine', config, pretrained=root/'run/best.pt', freeze_encoder=True)
            fine = VisualWorldModel.load(root/'fine/best.pt')
            for old, tuned in zip(model.members, fine.members):
                for name in ('encoder','state_encoder'):
                    for key,value in getattr(old,name).state_dict().items():
                        torch.testing.assert_close(value, getattr(tuned,name).state_dict()[key], rtol=0, atol=0)
            self.assertNotEqual(model.version, fine.version)
            with self.assertRaises(ValueError):
                calibration.validate_for(fine.version, fine.semantics)

    def member(self):
        import torch
        from wmal.models.visual_latent import NetworkConfig, make_member
        torch.set_num_threads(1)
        torch.manual_seed(3)
        return make_member(NetworkConfig(2, 2, image_size=32, latent_dim=16,
                                        hidden_dim=24, architecture='categorical_rssm',
                                        stoch=4, classes=4))

    def test_prior_actions_and_future_posterior_are_separate(self):
        import torch
        member = self.member()
        rgb = torch.zeros(1, 4, 3, 32, 32)
        states = torch.zeros(1, 4, 2)
        actions = torch.full((1, 3, 2), .2)
        before = member.imagine(rgb[:, 0], states[:, 0], actions)
        changed_rgb, changed_state = rgb.clone(), states.clone()
        changed_rgb[:, 1:] = 1
        changed_state[:, 1:] = 8
        original = member.observe(rgb, states, actions)
        altered = member.observe(changed_rgb, changed_state, actions)
        torch.testing.assert_close(original['prior_logits'][:, 1], altered['prior_logits'][:, 1])
        self.assertGreater(float((original['posterior_logits'][:, 1] -
                                  altered['posterior_logits'][:, 1]).abs().max().detach()), 1e-5)
        after = member.imagine(rgb[:, 0], states[:, 0], actions)
        torch.testing.assert_close(before['state'], after['state'])
        counter = member.imagine(rgb[:, 0], states[:, 0], torch.zeros_like(actions))
        self.assertGreater(float((before['state'] - counter['state']).abs().max().detach()), 1e-6)
        self.assertEqual(tuple(before['rgb'].shape), (1, 3, 3, 32, 32))
        self.assertEqual(tuple(before['prior_entropy'].shape), (1, 3))
        (before['state'].mean() + before['rgb'].mean()).backward()
        self.assertGreater(float(member.transition.weight_ih.grad.abs().sum()), 0)

    def test_reset_masks_state_and_incoming_action(self):
        import torch
        member = self.member()
        rgb = torch.rand(2, 3, 3, 32, 32)
        states = torch.randn(2, 3, 2)
        actions = torch.randn(2, 2, 2)
        rgb[1, 2], states[1, 2] = rgb[0, 2], states[0, 2]
        reset = torch.tensor([[True, False, True], [True, False, True]])
        output = member.observe(rgb, states, actions, is_first=reset)
        torch.testing.assert_close(output['features'][0, 2], output['features'][1, 2])
        single = member.observe(rgb[:1, 2:], states[:1, 2:], actions[:1, :0])
        torch.testing.assert_close(output['features'][:1, 2:], single['features'])

    def test_balanced_kl_gradient_paths(self):
        import torch
        from wmal.models.categorical_rssm import balanced_kl, categorical_probs, symlog, symexp
        posterior = torch.tensor([[[[2., 0., -1.]]]], requires_grad=True)
        prior = torch.tensor([[[[-1., 0., 2.]]]], requires_grad=True)
        dyn, rep = balanced_kl(posterior, prior, free_nats=0., unimix=.01)
        dyn.sum().backward(retain_graph=True)
        self.assertIsNone(posterior.grad)
        self.assertGreater(float(prior.grad.abs().sum()), 0)
        prior.grad = None
        rep.sum().backward()
        self.assertIsNone(prior.grad)
        self.assertGreater(float(posterior.grad.abs().sum()), 0)
        probabilities = categorical_probs(torch.tensor([[-1000., 1000., 0.]]), .03)
        self.assertGreaterEqual(float(probabilities.min()), .009999)
        values = torch.tensor([-100., -1., 0., 1., 100.])
        torch.testing.assert_close(symexp(symlog(values)), values)
        zero, _ = balanced_kl(prior, prior, free_nats=1., unimix=.01)
        torch.testing.assert_close(zero, torch.ones_like(zero))

    def test_old_checkpoint_and_rssm_roundtrip(self):
        import torch
        from wmal.models.visual_latent import NetworkConfig, VisualWorldModel, make_member
        semantics = {'state_order': ['x', 'y'], 'action_order': ['x', 'y'],
                     'image_size': 32, 'camera': 'front', 'period_s': .2,
                     'action_schema': 'test.delta.v1', 'event_names': []}
        normalization = {'state_mean': [0., 0.], 'state_scale': [1., 1.],
                         'action_mean': [0., 0.], 'action_scale': [1., 1.]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            legacy = NetworkConfig(2, 2, image_size=32, latent_dim=16, hidden_dim=24)
            old = VisualWorldModel(legacy, semantics, normalization, [make_member(legacy) for _ in range(2)])
            old.save(path)
            payload = torch.load(path, weights_only=True)
            # Simulate v2 before RSSM fields existed, keeping its original version hash.
            for name in ('architecture', 'stoch', 'classes', 'unimix'):
                payload['network'].pop(name, None)
            torch.save(payload, path)
            self.assertEqual(VisualWorldModel.load(path).version, old.version)
            member = self.member()
            model = VisualWorldModel(member.config, semantics, normalization,
                                     [member, self.member()], metadata={'max_horizon': 3})
            first = model.predict(np.zeros((3,32,32)), np.zeros(2), np.ones((3,2))*.1)
            model.save(path)
            loaded = VisualWorldModel.load(path)
            second = loaded.predict(np.zeros((3,32,32)), np.zeros(2), np.ones((3,2))*.1)
            self.assertEqual(model.version, loaded.version)
            np.testing.assert_array_equal(first['states'], second['states'])
            self.assertEqual(first['diagnostics']['architecture'], 'categorical_rssm')
            with self.assertRaises(ValueError):
                loaded.predict(np.zeros((3,32,32)), np.zeros(2), np.ones((4,2)))
