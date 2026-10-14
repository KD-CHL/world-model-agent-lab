"""Provider boundaries exercised with real small Task1 networks."""
from dataclasses import asdict, FrozenInstanceError
import importlib
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from wmal.datasets.embodied_profiles import ObservationProfile
from wmal.models.embodied_config import network_preset
from wmal.models.embodied_rssm import ScalableRSSMMember


def profile(**changes):
    values = dict(physical_order=['left_shoulder_pitch_joint', 'left_elbow_joint'],
        command_order=['left_shoulder_pitch_joint_target', 'left_elbow_joint_target'],
        action_order=['left_shoulder_pitch_joint', 'left_elbow_joint'],
        action_schema='wmal.g1.workcell.target_delta.v1', camera_ids=['workcell'],
        image_size=16, period_s=.2, physical_units=['rad']*2,
        command_units=['rad']*2, action_units=['rad']*2, command_rule='joint_delta',
        controller={'profile': 'fixture_pd', 'gains': [4, 5]}, origin={'dataset': 'train'},
        event_names=['contact', 'failure'])
    values.update(changes)
    return ObservationProfile(**values)


def make_model(api, *, members=1, metadata=None, observation_profile=None):
    p = observation_profile or profile()
    cfg = network_preset('smoke', physical_dim=len(p.physical_order),
        command_dim=len(p.command_order), action_dim=len(p.action_order),
        views=len(p.camera_ids), image_size=p.image_size, event_dim=len(p.event_names),
        context_steps=2, horizon=3)
    networks = []
    with torch.random.fork_rng(devices=[]):
        for index in range(members):
            torch.manual_seed(31 + index)
            networks.append(ScalableRSSMMember(cfg))
    norm = dict(physical_mean=[10, -20], physical_scale=[2, 4],
        command_mean=[100, 200][:cfg.command_dim], command_scale=[5, 10][:cfg.command_dim],
        action_mean=[3, -7], action_scale=[2, 4])
    return api.EmbodiedWorldModel(cfg, p, norm, networks, metadata=metadata)


def inputs():
    return (np.full((1, 1, 3, 16, 16), .3, dtype='float64'),
        np.array([[12, -16.]], dtype='float64'), np.array([[110, 220.]]),
        np.empty((0, 2)), np.array([[1, -2.], [3, 4.]]))


class EmbodiedProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def api(self):
        try:
            return importlib.import_module('wmal.models.embodied_world')
        except ModuleNotFoundError as exc:
            self.fail(f'Task3 provider missing: {exc.name}')

    # Integrating normalized actions instead of raw deltas breaks the literal command endpoints.
    def test_raw_increment_with_nonzero_normalization(self):
        model = make_model(self.api())
        result = model.predict_context(*inputs(), mode='soft', samples=1)
        np.testing.assert_array_equal(result['known_commands'], [[111, 218], [114, 222]])
        # Independently call the real normalized network to check the float32 boundary.
        tensors = [torch.tensor(x, dtype=torch.float32)[None] for x in (
            inputs()[0], [[1, 1]], [[2, 2]], np.empty((0, 2)),
            [[-1, 1.25], [0, 2.75]], [[2.2, 1.8], [2.8, 2.2]])]
        with torch.inference_mode():
            normalized = model.members[0].imagine_context(*tensors, sample=False)
        expected = normalized['physical'][0].numpy() * [2, 4] + [10, -20]
        np.testing.assert_allclose(result['physical_states'][0, 0], expected, atol=2e-6)

    # A legacy workaround that duplicates a member would violate these shapes.
    def test_single_member_and_no_fake_ensemble(self):
        model = make_model(self.api())
        result = model.predict_context(*inputs())
        self.assertEqual(result['physical_states'].shape, (1, 8, 2, 2))
        self.assertEqual(result['frames'].shape, (1, 8, 2, 1, 3, 16, 16))
        self.assertGreater(result['physical_states'].std(1).max(), 0)
        adapter = model.as_visual_predictor()
        legacy = adapter.predict_context(inputs()[0][:, 0], np.concatenate(inputs()[1:3], -1),
                                        inputs()[3], inputs()[4])
        self.assertEqual(legacy['states'].shape, (1, 2, 4))
        np.testing.assert_allclose(legacy['states'][0, :, :2], result['physical_states'].mean(1)[0])
        np.testing.assert_array_equal(legacy['states'][0, :, 2:], [[111, 218], [114, 222]])

    # All-false padding or future targets must never condition the prior.
    def test_future_targets_ignored(self):
        model = make_model(self.api())
        rgb, physical, commands, past, actions = inputs()
        sample = dict(rgb=np.repeat(rgb, 5, 0), physical=np.repeat(physical, 5, 0),
            commands=np.repeat(commands, 5, 0), actions=np.concatenate([np.zeros((2, 2)), actions]),
            view_mask=np.array([[False], [False], [True], [True], [True]]),
            prediction_offset=2, context_length=0)
        original = model.predict_sample(sample)
        for key in ('rgb', 'physical', 'commands'):
            sample[key][:2] = np.nan
            sample[key][3:] = np.nan
        sample['view_mask'][3:] = False
        changed = model.predict_sample(sample)
        np.testing.assert_array_equal(original['physical_states'], changed['physical_states'])
        np.testing.assert_array_equal(original['frames'], changed['frames'])

    # Incomplete content hashes must catch every checkpoint field and tensor mutation.
    def test_protocol_hash_and_checkpoint_roundtrip(self):
        api = self.api()
        model = make_model(api)
        sampled = model.predict_context(*inputs())
        self.assertNotEqual(sampled['protocol_hash'], model.predict_context(*inputs(), mode='soft')['protocol_hash'])
        self.assertNotEqual(sampled['protocol_hash'], model.predict_context(*inputs(), samples=2)['protocol_hash'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            model.save(path)
            loaded = api.EmbodiedWorldModel.load(path)
            self.assertEqual(model.version, loaded.version)
            np.testing.assert_array_equal(sampled['physical_states'], loaded.predict_context(*inputs())['physical_states'])
            payload = torch.load(path, weights_only=True)
            for key in ('metadata', 'normalization', 'profile', 'network', 'members'):
                with self.subTest(key=key):
                    altered = torch.load(path, weights_only=True)
                    if key == 'metadata':
                        altered[key]['note'] = 'tampered'
                    elif key == 'normalization':
                        altered[key]['action_mean'][0] += 1
                    elif key == 'profile':
                        altered[key]['origin']['dataset'] = 'tampered'
                    elif key == 'network':
                        altered[key]['unimix'] = .02
                    else:
                        tensor = next(iter(altered[key][0].values()))
                        tensor.flatten()[0] += 1
                    bad = Path(tmp) / (key + '.pt')
                    torch.save(altered, bad)
                    with self.assertRaises(ValueError):
                        api.EmbodiedWorldModel.load(bad)
            self.assertEqual(payload['schema'], 'wmal.embodied_world.v1')

    # Head presence is not recorded supervision; changing availability changes artifact identity.
    def test_event_labels_not_inferred(self):
        model = make_model(self.api())
        self.assertIsNone(model.predict_context(*inputs())['event_probabilities'])
        supervised = make_model(self.api(), metadata={'event_supervised_by_member': [[True, False]]})
        output = supervised.predict_context(*inputs())
        self.assertNotEqual(model.version, supervised.version)
        self.assertEqual(output['event_probability_names'], ['contact'])
        self.assertEqual(output['event_probabilities'].shape, (1, 8, 2, 1))
        mixed = make_model(self.api(), members=2,
                           metadata={'event_supervised_by_member': [[True, True], [True, False]]})
        self.assertEqual(mixed.predict_context(*inputs())['event_probability_names'], ['contact'])

    # Candidate traversal and ambient RNG must not change trajectories or pollute global RNG.
    def test_shared_random_numbers_and_explicit_seeds(self):
        model = make_model(self.api(), members=2)
        state = torch.random.get_rng_state().clone()
        a = model.predict_context(*inputs(), seed=17, samples=3)
        other = list(inputs())
        other[-1] = other[-1] * -1
        model.predict_context(*other, seed=17, samples=3)
        b = model.predict_context(*inputs(), seed=17, samples=3)
        np.testing.assert_array_equal(a['physical_states'], b['physical_states'])
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
        self.assertEqual(len(set(np.asarray(a['diagnostics']['trajectory_seeds']).ravel())), 6)
        self.assertGreater(np.abs(a['physical_states'] - model.predict_context(*inputs(), seed=18, samples=3)['physical_states']).max(), 0)

    # Arbitrary offline actions are supported only when no unknown command is needed.
    def test_offline_profile_and_exact_g1_adapter_gate(self):
        p = profile(command_order=[], command_units=[], command_rule='none', action_schema='torque.v1')
        model = make_model(self.api(), observation_profile=p)
        args = list(inputs()); args[2] = np.empty((1, 0))
        self.assertEqual(model.predict_context(*args)['known_commands'].shape, (2, 0))
        with self.assertRaises(ValueError):
            model.as_visual_predictor()
        unknown = make_model(self.api(), observation_profile=profile(command_rule='none'))
        with self.assertRaises(ValueError):
            unknown.predict_context(*inputs())
        reordered = make_model(self.api(), observation_profile=profile(physical_order=['left_elbow_joint', 'left_shoulder_pitch_joint']))
        with self.assertRaises(ValueError):
            reordered.as_visual_predictor()

    # Scope violations must reject, not clip or extrapolate, before imagination.
    def test_raw_scope_and_bad_context_rejected(self):
        model = make_model(self.api(), metadata={'prediction_scope': {
            'command_lower': [100, 210], 'command_upper': [113, 230], 'max_action_delta': 4}})
        with self.assertRaises(ValueError):
            model.predict_context(*inputs())
        for changed in ('mask', 'nan', 'horizon', 'seed', 'samples'):
            args = list(inputs()); kw = {}
            if changed == 'mask': kw['view_mask'] = np.zeros((1, 1), dtype=bool)
            if changed == 'nan': args[1][0, 0] = np.nan
            if changed == 'horizon': args[-1] = np.zeros((4, 2))
            if changed == 'seed': kw['seed'] = -1
            if changed == 'samples': kw['samples'] = 0
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                make_model(self.api()).predict_context(*args, **kw)

    # Stale IDs alone must not bind changed physical content, controller, reset or deadline.
    def test_response_hash_reset_and_deadline_binding(self):
        model = make_model(self.api())
        try:
            contracts = importlib.import_module('wmal.agents.prediction_contracts')
        except ModuleNotFoundError as exc:
            self.fail(f'Task3 request contract missing: {exc.name}')
        rgb, physical, commands, past, actions = inputs()
        fields = dict(request_id='req', decision_id='decision', run_id='run', task_id='task',
            node_id='node', episode_id='episode', step_id=4, context_step_ids=[4],
            observation_times=[1.2], candidate_id='candidate', skill_version='delta.v1',
            profile=model.profile, controller=model.profile.controller, model_version=model.version,
            protocol_hash=model.protocol_hash(), rgb=rgb, physical=physical, commands=commands,
            past_actions=past, actions=actions, deadline=time.monotonic()+20, timeout_s=10,
            context_steps=2, max_horizon=3, member_count=1)
        request = contracts.PredictionRequest(**fields)
        original_hash = request.context_hash
        physical[0, 0] = 42
        self.assertEqual(request.context_hash, original_hash)
        self.assertEqual(request.physical[0, 0], 12)
        with self.assertRaises((ValueError, FrozenInstanceError, AttributeError)):
            request.physical[0, 0] = 99
        with self.assertRaises(ValueError):
            request.physical.setflags(write=True)
        response = request.bind_response(model.predict_context(request.rgb, request.physical,
            request.commands, request.past_actions, request.actions))
        contracts.validate_response(request, response)
        for key, value in [('episode_id', 'reset'), ('context_hash', '0'*64),
                           ('actions_hash', '0'*64), ('deadline', fields['deadline']+1),
                           ('candidate_id', 'other'), ('protocol_hash', 'wrong'),
                           ('step_id', 4.), ('semantics', {'physical_units': ['deg', 'deg']}),
                           ('view_mask', np.zeros((1, 1), dtype=bool))]:
            bad = dict(response); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                contracts.validate_response(request, bad)
        with self.assertRaises(ValueError):
            contracts.validate_response(request, response, now=fields['deadline']+1)
        bad = dict(response); bad['physical_states'] = np.full((1, 8, 2, 2), np.nan)
        with self.assertRaises(ValueError): contracts.validate_response(request, bad)
        self.assertNotEqual(contracts.array_hash(np.array([1, 2], dtype='float32')),
                            contracts.array_hash(np.array([1, 2], dtype='float64')))
        self.assertNotEqual(contracts.array_hash(np.array([1, 2], dtype='float32')),
                            contracts.array_hash(np.array([[1, 2]], dtype='float32')))
        self.assertEqual(contracts.array_hash(np.array([1, 2], dtype='>f4')),
                         contracts.array_hash(np.array([1, 2], dtype='<f4')))

    # A stale artifact identity cannot authorize new physical weights or fake ensembles.
    def test_duplicate_weights_and_frozen_content_rejected(self):
        api = self.api(); model = make_model(api)
        clone = ScalableRSSMMember(model.config)
        clone.load_state_dict(model.members[0].state_dict())
        with self.assertRaises(ValueError):
            api.EmbodiedWorldModel(model.config, model.profile, model.normalization, [model.members[0], clone])
        with torch.no_grad():
            next(model.members[0].parameters()).flatten()[0] += 1
        with self.assertRaises(ValueError): model.predict_context(*inputs())
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            model.save(Path(tmp) / 'changed.pt')

    # Loading tensors must not advance the application random stream.
    def test_load_preserves_rng_and_metadata_snapshots(self):
        api = self.api(); meta = {'event_supervised_by_member': [[True, False]], 'note': {'x': 1}}
        model = make_model(api, metadata=meta)
        meta['note']['x'] = 2
        copy = model.metadata; copy['note']['x'] = 3
        self.assertEqual(model.metadata['note']['x'], 1)
        norm = model.normalization; norm['physical_scale'][0] = 99
        self.assertEqual(model.normalization['physical_scale'], [2, 4])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'; model.save(path)
            before = torch.random.get_rng_state().clone()
            loaded = api.EmbodiedWorldModel.load(path)
            torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
            self.assertEqual(loaded.metadata, model.metadata)

    # Finite tensors do not excuse nonfinite diagnostics or spoofed supervision.
    def test_response_rejects_spoofed_protocol_and_nonfinite_diagnostics(self):
        model = make_model(self.api())
        contracts = importlib.import_module('wmal.agents.prediction_contracts')
        rgb, physical, commands, past, actions = inputs()
        request = contracts.PredictionRequest(request_id='r', decision_id='d', run_id='run',
            task_id='t', node_id='n', episode_id='ep', step_id=0, context_step_ids=[0],
            observation_times=[0.], candidate_id='c', skill_version='v', profile=model.profile,
            controller=model.profile.controller, model_version=model.version,
            protocol_hash=model.protocol_hash(), rgb=rgb, physical=physical, commands=commands,
            past_actions=past, actions=actions, deadline=time.monotonic()+30, timeout_s=10,
            context_steps=2, max_horizon=3, member_count=1)
        response = request.bind_response(model.predict_context(*inputs()))
        request.controller['gains'][0] = 99
        self.assertEqual(request.controller['gains'], [4, 5])
        bad = dict(response); bad['diagnostics'] = {'prior_entropy': [float('inf')]}
        with self.assertRaises(ValueError): contracts.validate_response(request, bad)
        bad = dict(response); bad['protocol'] = {'mode': 'soft'}
        with self.assertRaises(ValueError): contracts.validate_response(request, bad)
        bad = dict(response); bad.update(event_supervised_channels=[True, False],
            event_probability_names=['contact'], event_probabilities=np.ones((1, 8, 2, 1))*.5)
        with self.assertRaises(ValueError): contracts.validate_response(request, bad)

    # Invalid scope declarations must fail as ValueError rather than a late comparison TypeError.
    def test_scope_declaration_fails_closed(self):
        for scope in ({'command_lower': ['a', 'b'], 'command_upper': ['c', 'd']},
                      {'max_action_delta': '4'}, {'action_lower': [0], 'action_upper': [1]}):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                make_model(self.api(), metadata={'prediction_scope': scope})

    # Hash verification must happen on stored tensors before load_state_dict can cast their dtype.
    def test_checkpoint_hashes_stored_tensor_dtype_before_cast(self):
        api = self.api(); model = make_model(api)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'; model.save(path)
            payload = torch.load(path, weights_only=True)
            key = next(iter(payload['members'][0]))
            payload['members'][0][key] = payload['members'][0][key].double()
            torch.save(payload, path)
            with self.assertRaises(ValueError): api.EmbodiedWorldModel.load(path)

    # A request must reject reset gaps, controller transfer and misaligned timestamps before inference.
    def test_request_history_controller_and_deadline_validation(self):
        model = make_model(self.api())
        contracts = importlib.import_module('wmal.agents.prediction_contracts')
        rgb, physical, commands, _, actions = inputs()
        fields = dict(request_id='r', decision_id='d', run_id='run', task_id='t', node_id='n',
            episode_id='ep', step_id=1, context_step_ids=[0, 1], observation_times=[0., .2],
            candidate_id='c', skill_version='v', profile=model.profile,
            controller=model.profile.controller, model_version=model.version, protocol_hash=model.protocol_hash(),
            rgb=np.repeat(rgb, 2, 0), physical=np.repeat(physical, 2, 0), commands=np.repeat(commands, 2, 0),
            past_actions=np.zeros((1, 2)), actions=actions, deadline=time.monotonic()+30, timeout_s=10,
            context_steps=2, max_horizon=3, member_count=1)
        contracts.PredictionRequest(**fields)
        for changes in ({'context_step_ids': [0, 2]}, {'observation_times': [0., .3]},
            {'controller': {'profile': 'transferred'}}, {'deadline': time.monotonic()-1},
            {'past_actions': np.ones((1, 2))}, {'event_supervised_channels': [1, 0]}):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                contracts.PredictionRequest(**dict(fields, **changes))

    # Real supervised channels must roundtrip through requests only when explicitly artifact-bound.
    def test_supervised_response_availability_binding(self):
        model = make_model(self.api(), metadata={'event_supervised_by_member': [[True, False]]})
        contracts = importlib.import_module('wmal.agents.prediction_contracts')
        rgb, physical, commands, past, actions = inputs()
        fields = dict(request_id='r', decision_id='d', run_id='run', task_id='t', node_id='n',
            episode_id='ep', step_id=0, context_step_ids=[0], observation_times=[0.],
            candidate_id='c', skill_version='v', profile=model.profile,
            controller=model.profile.controller, model_version=model.version, protocol_hash=model.protocol_hash(),
            rgb=rgb, physical=physical, commands=commands, past_actions=past, actions=actions,
            deadline=time.monotonic()+30, timeout_s=10, context_steps=2, max_horizon=3, member_count=1)
        prediction = model.predict_context(*inputs())
        with self.assertRaises(ValueError): contracts.PredictionRequest(**fields).bind_response(prediction)
        request = contracts.PredictionRequest(**fields, event_supervised_channels=[True, False])
        contracts.validate_response(request, request.bind_response(prediction))

    # A legacy output must select the first declared camera after averaging samples,
    # while a multi-view model still requires all real views at its input boundary.
    def test_multiview_adapter_first_camera_with_explicit_real_views(self):
        model = make_model(self.api(), observation_profile=profile(camera_ids=['front', 'side']))
        rgb, physical, commands, past, actions = inputs()
        rgb = np.repeat(rgb, 2, axis=1); rgb[:, 1] = .8
        raw = model.predict_context(rgb, physical, commands, past, actions)
        adapter = model.as_visual_predictor()
        output = adapter.predict_context(rgb, np.concatenate([physical, commands], -1), past, actions)
        np.testing.assert_allclose(output['frames'], raw['frames'].mean(1)[:, :, 0])
        self.assertEqual(adapter.semantics['camera'], 'front')
        with self.assertRaises(ValueError):
            adapter.predict_context(rgb[:, 0], np.concatenate([physical, commands], -1), past, actions)

    # Comparing full profiles wrongly treats an independent collection origin as
    # a control change; removing provenance from the hashes would lose binding.
    def test_response_accepts_independent_origin_without_erasing_identity(self):
        model = make_model(self.api())
        contracts = importlib.import_module('wmal.agents.prediction_contracts')
        independent = profile(origin={'dataset': 'independent-request', 'manifest_sha256': 'c'*64,
                                      'collection_seed': 77})
        model.validate_profile(independent)
        rgb, physical, commands, past, actions = inputs()
        fields = dict(request_id='r', decision_id='d', run_id='run', task_id='t', node_id='n',
            episode_id='ep', step_id=0, context_step_ids=[0], observation_times=[0.],
            candidate_id='c', skill_version='v', model_version=model.version,
            protocol_hash=model.protocol_hash(), rgb=rgb, physical=physical, commands=commands,
            past_actions=past, actions=actions, deadline=time.monotonic()+30, timeout_s=10,
            context_steps=2, max_horizon=3, member_count=1)
        request = contracts.PredictionRequest(**fields, profile=independent, controller=independent.controller)
        prediction = model.predict_context(*inputs())
        try:
            response = request.bind_response(prediction)
        except ValueError as exc:
            self.fail(f'Origin-only semantic compatibility was rejected: {exc}')
        contracts.validate_response(request, response)
        self.assertEqual(response['profile']['origin'], {'dataset': 'independent-request',
            'manifest_sha256': 'c'*64, 'collection_seed': 77})
        self.assertEqual(response['protocol']['profile']['origin'], {'dataset': 'train'})
        original_request = contracts.PredictionRequest(**fields, profile=model.profile,
                                                     controller=model.profile.controller)
        self.assertNotEqual(request.request_hash, original_request.request_hash)
        self.assertNotEqual(request.binding()['profile_hash'], original_request.binding()['profile_hash'])
        self.assertEqual(model.protocol_hash(), fields['protocol_hash'])
        for side in ('request', 'model'):
            bad = dict(response)
            if side == 'request':
                bad['profile'] = model.profile.to_dict()
            else:
                bad['protocol'] = dict(response['protocol'], profile=independent.to_dict())
            with self.subTest(origin_side=side), self.assertRaises(ValueError):
                contracts.validate_response(request, bad)
        for change in ({'physical_units': ['deg', 'rad']}, {'camera_ids': ['other-camera']},
                       {'controller': {'profile': 'fixture_pd', 'gains': [9, 9]}}):
            changed = profile(origin=independent.origin, **change)
            changed_request = contracts.PredictionRequest(**fields, profile=changed, controller=changed.controller)
            with self.subTest(semantic_field=list(change)), self.assertRaises(ValueError):
                changed_request.bind_response(prediction)


if __name__ == '__main__':
    unittest.main()
