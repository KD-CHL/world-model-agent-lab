"""Real-network contracts for normalized multimodal, action-conditioned RSSM.

Each test names its failure mode: future leakage, carry leakage, RNG pollution,
candidate-order noise, invalid modalities, detached samples, or false budgets.
"""
from dataclasses import FrozenInstanceError, asdict, replace
import importlib
import json
from pathlib import Path
import unittest

import torch


class EmbodiedRSSMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def api(self):
        try:
            config = importlib.import_module('wmal.models.embodied_config')
            rssm = importlib.import_module('wmal.models.embodied_rssm')
        except ModuleNotFoundError as error:
            self.fail(f'Task1 scalable RSSM feature is missing: {error}')
        return config, rssm

    def member(self, **changes):
        config, rssm = self.api()
        cfg = config.network_preset('smoke', physical_dim=2, command_dim=2,
                                    action_dim=2, views=2, image_size=16,
                                    event_dim=2, context_steps=2, horizon=3)
        # Model initialization is isolated too; the sampling test snapshots after it.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            return rssm.ScalableRSSMMember(replace(cfg, **changes))

    def inputs(self, batch=2, steps=3):
        rgb = torch.linspace(0, 1, batch*steps*2*3*16*16).reshape(batch, steps, 2, 3, 16, 16)
        physical = torch.linspace(-2, 2, batch*steps*2).reshape(batch, steps, 2)
        commands = physical.flip(-1) + .5
        past = torch.full((batch, steps-1, 2), .2)
        actions = torch.tensor([[[.1, -.2], [.3, .1], [-.1, .4]]]).expand(batch, -1, -1).clone()
        future = torch.tensor([[[.7, -.4], [.8, -.2], [.9, .1]]]).expand(batch, -1, -1).clone()
        return rgb, physical, commands, past, actions, future

    def test_prior_changes_with_action_but_never_future_observation(self):
        member = self.member()
        rgb, physical, commands, past, actions, future = self.inputs()
        original = member.observe(rgb, physical, commands, past, sample=False)
        changed_rgb, changed_physical = rgb.clone(), physical.clone()
        changed_rgb[:, 1] = 1 - rgb[:, 1]
        changed_physical[:, 1] += 20
        altered = member.observe(changed_rgb, changed_physical, commands, past, sample=False)
        torch.testing.assert_close(original['prior_logits'][:, 1], altered['prior_logits'][:, 1], rtol=0, atol=0)
        self.assertGreater((original['posterior_logits'][:, 1] - altered['posterior_logits'][:, 1]).abs().max().item(), 1e-5)
        first = member.imagine_context(rgb, physical, commands, past, actions, future, sample=False)
        other = member.imagine_context(rgb, physical, commands, past, actions*0, future, sample=False)
        self.assertGreater((first['prior_logits'] - other['prior_logits']).abs().max().item(), 1e-5)
        known = member.imagine_context(rgb, physical, commands, past, actions, future+2, sample=False)
        self.assertGreater((first['prior_logits'] - known['prior_logits']).abs().max().item(), 1e-5)
        with self.assertRaises(TypeError):
            member.imagine_context(rgb, physical, commands, past, actions, future, future_rgb=rgb)
        self.assertEqual(tuple(first['physical'].shape), (2, 3, 2))
        self.assertEqual(tuple(first['rgb'].shape), (2, 3, 2, 3, 16, 16))
        self.assertEqual(tuple(first['event_logits'].shape), (2, 3, 2))
        self.assertEqual(tuple(first['features'].shape), (2, 3, member.config.deter + member.config.stoch*member.config.classes))

    def test_reset_and_startup_context(self):
        member = self.member()
        rgb, physical, commands, past, actions, future = self.inputs()
        rgb[1, -1], physical[1, -1], commands[1, -1] = rgb[0, -1], physical[0, -1], commands[0, -1]
        reset = torch.tensor([[True, False, True], [True, False, True]])
        observed = member.observe(rgb, physical, commands, past, is_first=reset, sample=False)
        single = member.observe(rgb[:1, -1:], physical[:1, -1:], commands[:1, -1:], past[:1, :0], sample=False)
        torch.testing.assert_close(observed['features'][:1, -1:], single['features'])
        torch.testing.assert_close(observed['features'][0, -1], observed['features'][1, -1])
        short = member.imagine_context(rgb[:, -1:], physical[:, -1:], commands[:, -1:], past[:, :0], actions, future, sample=True, seed=17)
        long = member.imagine_context(rgb, physical, commands, past, actions, future, is_first=reset, sample=True, seed=17)
        torch.testing.assert_close(short['physical'], long['physical'], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            member.imagine_context(rgb, physical, commands, past, actions[:, :0], future[:, :0])
        with self.assertRaises(ValueError):
            member.imagine_context(rgb, physical, commands, past, actions.repeat(1, 2, 1), future.repeat(1, 2, 1))
        with self.assertRaises(ValueError):
            member.imagine_context(rgb.repeat(1, 2, 1, 1, 1, 1), physical.repeat(1, 2, 1), commands.repeat(1, 2, 1), past.repeat(1, 3, 1)[:, :5], actions, future)

    def test_sample_seed_does_not_mutate_global_rng(self):
        member = self.member()
        args = self.inputs()
        before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
        one = member.observe(*args[:4], sample=True, seed=31)
        two = member.observe(*args[:4], sample=True, seed=31)
        other = member.observe(*args[:4], sample=True, seed=32)
        roll = member.imagine_context(*args, sample=True, seed=31)
        again = member.imagine_context(*args, sample=True, seed=31)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for old, new in zip(cuda_before, torch.cuda.get_rng_state_all() if cuda_before else []):
            self.assertTrue(torch.equal(old, new))
        torch.testing.assert_close(one['features'], two['features'], rtol=0, atol=0)
        torch.testing.assert_close(roll['physical'], again['physical'], rtol=0, atol=0)
        self.assertFalse(torch.equal(one['features'], other['features']))
        z = one['features'][..., member.config.deter:].reshape(2, 3, member.config.stoch, member.config.classes)
        torch.testing.assert_close(z.sum(-1), torch.ones(2, 3, member.config.stoch))
        self.assertTrue(torch.all((z == 0) | (z == 1)).item())

    def test_candidate_order_invariance(self):
        member = self.member()
        rgb, physical, commands, past, actions, future = self.inputs(batch=3)
        # One real prefix for all branches, three genuinely different plans.
        for value in (rgb, physical, commands, past):
            value[:] = value[:1].clone()
        actions[1] *= -1
        actions[2] *= 2
        future[1] += .2
        future[2] -= .3
        args = (rgb, physical, commands, past, actions, future)
        perm = torch.tensor([2, 0, 1])
        output = member.imagine_context(*args, sample=True, seed=5)
        reordered = member.imagine_context(*(value[perm] for value in args), sample=True, seed=5)
        # Categorical draws/carry must be identical. Flattened decoder GEMMs may
        # differ by FP32 rounding when reordered rows occupy different lanes.
        for key in ('prior_logits', 'features'):
            torch.testing.assert_close(output[key][perm], reordered[key], rtol=0, atol=0)
        for key in ('physical', 'rgb'):
            torch.testing.assert_close(output[key][perm], reordered[key], rtol=1e-5, atol=1e-6)
        for index in range(3):
            alone = member.imagine_context(*(value[index:index+1] for value in args), sample=True, seed=5)
            torch.testing.assert_close(output['physical'][index:index+1], alone['physical'], rtol=1e-5, atol=1e-6)

    def test_missing_view_mask_and_nan_rejected(self):
        member = self.member()
        args = self.inputs()
        mask = torch.ones(2, 3, 2, dtype=torch.bool)
        mask[..., 1] = False
        a = member.observe(*args[:4], view_mask=mask, sample=False)
        changed = args[0].clone()
        changed[:, :, 1] = 1 - changed[:, :, 1]
        b = member.observe(changed, *args[1:4], view_mask=mask, sample=False)
        torch.testing.assert_close(a['features'], b['features'], rtol=0, atol=0)
        for bad in (torch.zeros_like(mask), mask.float(), mask[:, :, :1]):
            with self.assertRaises(ValueError):
                member.observe(*args[:4], view_mask=bad)
        for index in range(6):
            invalid = list(args)
            invalid[index] = invalid[index].clone()
            invalid[index].flatten()[0] = float('nan')
            with self.subTest(modality=index), self.assertRaises(ValueError):
                member.imagine_context(*invalid)
        for index in range(4):
            invalid = list(args[:4])
            invalid[index] = invalid[index].clone()
            invalid[index].flatten()[0] = float('inf')
            with self.subTest(observe_modality=index), self.assertRaises(ValueError):
                member.observe(*invalid)
        with self.assertRaises(ValueError):
            member.observe(*args[:4], is_first=torch.ones(2, 3))
        with self.assertRaises(ValueError):
            member.observe(args[0]+2, *args[1:4])

    def test_camera_assignment_changes_posterior(self):
        # Pooling shared-camera tokens before fusion loses camera identity.
        member = self.member()
        rgb, physical, commands, past, _, _ = self.inputs()
        rgb[:, :, 0] = .1
        rgb[:, :, 1] = .9
        original = member.observe(rgb, physical, commands, past, sample=False)
        swapped = member.observe(rgb.flip(2), physical, commands, past, sample=False)
        torch.testing.assert_close(original['prior_logits'][:, 0], swapped['prior_logits'][:, 0], rtol=0, atol=0)
        self.assertGreater((original['posterior_logits'][:, 0] - swapped['posterior_logits'][:, 0]).abs().max().item(), 1e-5)

    def test_gradients_and_balanced_kl(self):
        from wmal.models.categorical_rssm import balanced_kl
        member = self.member()
        args = self.inputs()
        observed = member.observe(*args[:4], sample=True, seed=9)
        zloss = observed['features'][..., member.config.deter].sum()
        grads = torch.autograd.grad(zloss, tuple(member.posterior.parameters()), retain_graph=True)
        self.assertGreater(sum(grad.abs().sum().item() for grad in grads), 0)
        roll = member.imagine_context(*args, sample=True, seed=9)
        prior_grads = torch.autograd.grad(roll['features'][..., member.config.deter].sum(), tuple(member.prior.parameters()), retain_graph=True)
        self.assertGreater(sum(grad.abs().sum().item() for grad in prior_grads), 0)
        loss = sum(output['physical'].square().mean() + output['rgb'].square().mean() + output['event_logits'].square().mean() for output in (observed, roll))
        loss.backward()
        missing = [name for name, p in member.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
        self.assertEqual(missing, [])
        q = torch.tensor([[[[2., 0., -1.]]]], requires_grad=True)
        p = torch.tensor([[[[-1., 0., 2.]]]], requires_grad=True)
        dyn, rep = balanced_kl(q, p, free_nats=0., unimix=.01)
        dq, dp = torch.autograd.grad(dyn.sum(), (q, p), allow_unused=True, retain_graph=True)
        self.assertIsNone(dq)
        self.assertGreater(dp.abs().sum().item(), 0)
        dq, dp = torch.autograd.grad(rep.sum(), (q, p), allow_unused=True)
        self.assertIsNone(dp)
        self.assertGreater(dq.abs().sum().item(), 0)
        floor, _ = balanced_kl(p, p, free_nats=1., unimix=.01)
        torch.testing.assert_close(floor, torch.ones(1, 1))

    def test_residual_heads_and_full_rgb_mode(self):
        args = self.inputs()
        for mode, expected_rgb in (('residual', args[0][:, -1:, ...].expand(-1, 3, -1, -1, -1, -1)), ('full', torch.full((2, 3, 2, 3, 16, 16), .5))):
            member = self.member(rgb_mode=mode, event_dim=0)
            with torch.no_grad():
                for p in member.physical_head.parameters():
                    p.zero_()
                for p in member.decoder.parameters():
                    p.zero_()
            output = member.imagine_context(*args, sample=False)
            torch.testing.assert_close(output['physical'], args[1][:, -1:].expand(-1, 3, -1), rtol=0, atol=0)
            torch.testing.assert_close(output['rgb'], expected_rgb, rtol=0, atol=0)
            torch.testing.assert_close(output['physical_delta'], torch.zeros(2, 3, 2), rtol=0, atol=0)
            self.assertIsNone(output['event_logits'])
        member = self.member()
        with torch.no_grad():
            for p in member.physical_head.parameters():
                p.zero_()
            member.physical_head[-1].bias.fill_(.25)
        output = member.imagine_context(*args, sample=False)
        torch.testing.assert_close(output['physical'], args[1][:, -1:].expand(-1, 3, -1) + .25)

    def test_config_validation_and_no_command_profile(self):
        config, _ = self.api()
        cfg = config.network_preset('smoke')
        with self.assertRaises(FrozenInstanceError):
            cfg.deter = 0
        bad = {'physical_dim': 0, 'command_dim': -1, 'action_dim': 0, 'views': 0,
               'image_size': 20, 'event_dim': -1, 'context_steps': -1, 'horizon': 0,
               'deter': 63, 'hidden': 127, 'stoch': 0, 'classes': 1, 'blocks': 0,
               'encoder_channels': (), 'fusion_width': 0, 'decoder_width': 0,
               'unimix': float('nan'), 'norm': 'batch', 'norm_eps': 0, 'rgb_mode': 'unknown'}
        for key, value in bad.items():
            with self.subTest(field=key), self.assertRaises(ValueError):
                replace(cfg, **{key: value})
        for value in (-.1, 1.):
            with self.assertRaises(ValueError):
                replace(cfg, unimix=value)
        with self.assertRaises(ValueError):
            config.network_preset('unknown')
        args = list(self.inputs())
        args[2] = args[2][..., :0]
        args[5] = args[5][..., :0]
        member = self.member(command_dim=0)
        self.assertTrue(torch.isfinite(member.imagine_context(*args)['physical']).all().item())

    def test_preset_real_parameter_budgets(self):
        config, rssm = self.api()
        ranges = {'smoke': (1, 1_000_000), 'small': (30_000_000, 50_000_000),
                  'base': (120_000_000, 180_000_000), 'large': (240_000_000, 300_000_000)}
        for name, (lower, upper) in ranges.items():
            with self.subTest(preset=name):
                cfg = config.network_preset(name)
                rng = torch.random.get_rng_state().clone()
                report = rssm.parameter_report(cfg)
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                with torch.device('meta'):
                    member = rssm.ScalableRSSMMember(cfg)
                self.assertTrue(all(p.device.type == 'meta' for p in member.parameters()))
                unique = {id(p): p for p in member.parameters()}
                self.assertEqual(report['total'], sum(p.numel() for p in unique.values()))
                self.assertEqual(report['trainable'], report['total'])
                self.assertEqual(report['frozen'], 0)
                self.assertTrue(report['within_budget'])
                self.assertGreaterEqual(report['total'], lower)
                self.assertLessEqual(report['total'], upper)
                payload = json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'training' / f'embodied_{name}.json').read_text())
                self.assertEqual(asdict(config.ScalableConfig(**payload)), asdict(cfg))
        member = self.member()
        self.assertEqual(rssm.parameter_report(member.config)['total'], sum(p.numel() for p in member.parameters()))
