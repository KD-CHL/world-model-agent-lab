"""Controller transfer must never silently reuse pre-transfer calibration."""
import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
from wmal.models.horizon_calibration import HorizonCalibration


class ControllerCalibrationTests(unittest.TestCase):
    def calibration(self):
        return HorizonCalibration('v1', 'independent-set', .2, (1.,), (1.,), .05,
                                  ('c0', 'c1', 'c2', 'c3'), {'action_schema': 'example'})

    def validate(self, calibration, profile):
        method = getattr(calibration, 'validate_controller', None)
        self.assertIsNotNone(method, 'Calibration must validate its fixed controller binding')
        method(profile)

    def test_legacy_calibration_cannot_authorize_new_controller(self):
        calibration = self.calibration()
        self.validate(calibration, {'profile': 'g1_joint_pd_v1'})
        with self.assertRaises(ValueError):
            self.validate(calibration, {'profile': 'g1_joint_pd_gravity_v2'})

    def test_new_calibration_matches_entire_controller_descriptor(self):
        calibration = self.calibration()
        self.assertTrue(hasattr(calibration, 'controller_profile'), 'Missing controller provenance')
        profile = {'profile': 'g1_joint_pd_gravity_v2', 'parameters_sha256': 'fixed-gains'}
        calibration = replace(calibration, controller_profile=profile)
        self.validate(calibration, dict(profile))
        with self.assertRaises(ValueError):
            self.validate(calibration, {**profile, 'parameters_sha256': 'different-gains'})
        with self.assertRaises(ValueError):
            self.validate(calibration, {'profile': 'g1_joint_pd_v1'})

    def test_controller_transfer_needs_explicit_opt_in_and_binds_new_calibration(self):
        import torch
        from test_visual_world import make_dataset
        from wmal.logging.manifest import atomic_json
        from wmal.models.visual_latent import NetworkConfig, VisualLatentMember, VisualWorldModel
        from wmal.models.horizon_calibration import calibrate_visual
        from wmal.datasets.visual_sequences import VisualDataset
        import json
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=make_dataset(root)
            dataset=VisualDataset(manifest,'train',horizon=2)
            config=NetworkConfig(2,2,image_size=32,latent_dim=8,hidden_dim=16)
            model=VisualWorldModel(config,dataset.semantics,dataset.normalization(),
                [VisualLatentMember(config),VisualLatentMember(config)],
                metadata={'max_horizon':2,'training_episode_ids':[],'selection_episode_ids':[]})
            model.save(root/'frozen.pt')
            payload=json.loads(manifest.read_text())
            profile={'profile':'g1_joint_pd_gravity_v2','parameters_sha256':'fixed-gains'}
            payload['source']['controller_profile']=profile
            atomic_json(manifest,payload)
            with self.assertRaisesRegex(ValueError,'transfer'):
                calibrate_visual(manifest,root/'frozen.pt',root/'denied.json',horizon=2,alpha=.2)
            self.assertFalse((root/'denied.json').exists())
            try:
                result=calibrate_visual(manifest,root/'frozen.pt',root/'bound.json',horizon=2,
                                        alpha=.2,allow_controller_transfer=True)
            except TypeError as exc:
                self.fail(f'Explicit controller transfer calibration is unavailable: {exc}')
            self.validate(result,profile)
            self.assertEqual(result.model_version,model.version)
            self.assertEqual(set(result.episode_ids),{'calibration_0','calibration_1',
                                                     'calibration_2','calibration_3','calibration_4'})
            from wmal.training.visual_trainer import evaluate_visual
            with self.assertRaisesRegex(ValueError,'transfer'):
                evaluate_visual(manifest,root/'frozen.pt',horizon=2,calibration=root/'bound.json')
            try:
                metrics=evaluate_visual(manifest,root/'frozen.pt',horizon=2,calibration=root/'bound.json',
                                        allow_controller_transfer=True)
            except TypeError as exc:
                self.fail(f'Explicit frozen-controller evaluation is unavailable: {exc}')
            self.assertEqual(metrics['controller_profile'],profile)
            self.assertTrue(metrics['controller_transfer'])
            self.assertEqual(metrics['episodes'],2)
            stale=replace(result,controller_profile=None)
            stale.save(root/'stale.json')
            with self.assertRaisesRegex(ValueError,'controller mismatch'):
                evaluate_visual(manifest,root/'frozen.pt',horizon=2,calibration=root/'stale.json',
                                allow_controller_transfer=True)

    def test_local_calibration_rejects_out_of_domain_context_or_commands(self):
        import numpy as np
        calibration=self.calibration()
        self.assertTrue(hasattr(calibration,'prediction_scope'),'Missing deployment-domain binding')
        scope={'command_lower':[.21,.73],'command_upper':[.49,1.01],
               'measured_margin':.15,'max_action_delta':.06}
        calibration=replace(calibration,prediction_scope=scope)
        self.assertTrue(calibration.allows_scope(np.array([[.35,.87,.35,.87]]),np.array([[.04,.04]])))
        self.assertFalse(calibration.allows_scope(np.array([[.35,.87,.5,.87]]),np.zeros((1,2))))
        self.assertFalse(calibration.allows_scope(np.array([[.35,.87,.48,.87]]),np.array([[.04,0.]])))
        self.assertFalse(calibration.allows_scope(np.array([[.35,.87,.35,.87]]),np.array([[.07,0.]])))
        self.assertFalse(calibration.allows_scope(np.array([[.9,.87,.35,.87]]),np.zeros((1,2))))
