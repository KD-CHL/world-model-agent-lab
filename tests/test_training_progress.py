"""Real training must remain reproducible while terminal progress stays out of JSON."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from test_visual_world import make_dataset
from test_motion_network import fixture


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional learning dependencies not installed')
class TrainingProgressTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.manifest=make_dataset(self.root)
        from wmal.training.visual_trainer import VisualTrainingConfig
        config=VisualTrainingConfig(epochs=1,members=2,batch_size=16,horizon=2,
                                    latent_dim=16,hidden_dim=24,seed=3)
        self.config=self.root/'visual_config.json'
        self.config.write_text(json.dumps(asdict(config)))

    def visual(self,name,*extra):
        from wmal.training.visual_cli import main
        stdout,stderr=io.StringIO(),io.StringIO()
        args=['train','--manifest',str(self.manifest),'--config',str(self.config),
              '--output',str(self.root/name),*extra]
        with redirect_stdout(stdout),redirect_stderr(stderr):
            try:
                main(args)
            except SystemExit as exc:
                self.fail(f'Training CLI rejected progress control: {exc}')
        return json.loads(stdout.getvalue()),stderr.getvalue()

    def test_visual_cli_reports_epochs_batches_losses_and_eta_only_on_stderr(self):
        result,stderr=self.visual('visible')
        self.assertIn('Visual epochs',stderr)
        self.assertIn('Train 1/1',stderr)
        self.assertIn('Validation 1/1',stderr)
        self.assertIn('train_loss=',stderr)
        self.assertIn('val_loss=',stderr)
        self.assertIn('ETA',stderr)
        self.assertIn('controlled_test',stderr)
        self.assertIn('train_episodes=4',stderr)
        self.assertIn('train_windows=28',stderr)
        self.assertNotIn('\x1b',stderr)  # A redirected/non-TTY stream needs no cursor-control escapes.
        self.assertRegex(stderr,r'Visual epochs:.*100%.*1/1')
        self.assertTrue(Path(result['checkpoint']).is_file())

    def test_no_progress_is_silent_and_does_not_change_visual_weights(self):
        visible,_=self.visual('visible')
        quiet,stderr=self.visual('quiet','--no-progress')
        self.assertEqual(stderr,'')
        self.assertEqual(quiet['model_version'],visible['model_version'])

    def test_legacy_string_source_remains_trainable_with_progress_on_or_off(self):
        payload=json.loads(self.manifest.read_text())
        payload['source']='legacy_simulator_export'
        self.manifest.write_text(json.dumps(payload))
        try:
            visible,stderr=self.visual('legacy_visible')
            quiet,quiet_stderr=self.visual('legacy_quiet','--no-progress')
        except AttributeError as exc:
            self.fail(f'Terminal summary broke a valid legacy source: {exc}')
        self.assertIn('legacy_simulator_export',stderr)
        self.assertEqual(quiet_stderr,'')
        self.assertEqual(visible['model_version'],quiet['model_version'])
        report=json.loads((self.root/'legacy_visible/training_report.json').read_text())
        self.assertEqual(report['metadata']['training_source'],'legacy_simulator_export')

    def test_finetune_uses_the_same_progress_without_training_another_agent(self):
        from wmal.models.visual_latent import VisualWorldModel
        base,_=self.visual('base')
        fine,stderr=self.visual('fine','--pretrained',base['checkpoint'],'--freeze-encoder')
        self.assertIn('Fine-tune epochs',stderr)
        self.assertIn('Validation 1/1',stderr)
        tuned=VisualWorldModel.load(fine['checkpoint'])
        self.assertTrue(tuned.metadata['freeze_encoder'])
        self.assertEqual(tuned.metadata['parent_model_version'],base['model_version'])

    def test_failed_batch_closes_progress_without_reporting_completed_epochs(self):
        from wmal.training.visual_cli import main
        stdout,stderr=io.StringIO(),io.StringIO()
        args=['train','--manifest',str(self.manifest),'--config',str(self.config),
              '--output',str(self.root/'failed')]
        with redirect_stdout(stdout),redirect_stderr(stderr), \
             patch('wmal.training.visual_trainer.batch_loss',side_effect=RuntimeError('injected loss failure')):
            with self.assertRaisesRegex(RuntimeError,'injected loss failure'):
                main(args)
        self.assertIn('Visual epochs',stderr.getvalue())
        self.assertNotRegex(stderr.getvalue(),r'Visual epochs:.*100%')
        self.assertEqual(stdout.getvalue(),'')
        self.assertFalse((self.root/'failed/best.pt').exists())

    def test_missing_dataset_fails_instead_of_producing_a_model(self):
        from wmal.training.visual_cli import main
        stdout,stderr=io.StringIO(),io.StringIO()
        output=self.root/'without_data'
        with redirect_stdout(stdout),redirect_stderr(stderr):
            with self.assertRaises(FileNotFoundError):
                main(['train','--manifest',str(self.root/'missing.json'),
                      '--config',str(self.config),'--output',str(output)])
        self.assertEqual(stdout.getvalue(),'')
        self.assertFalse(output.exists())

    def test_motion_progress_preserves_json_events_and_resume_equivalence(self):
        from wmal.training.motion_trainer import TrainingConfig
        script=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/train_motion_network.py'))
        dataset=self.root/'motion.json'
        fixture(dataset)
        config=self.root/'motion_config.json'
        config.write_text(json.dumps(asdict(TrainingConfig(hidden=(24,24),members=2,epochs=2,
                                                           batch_size=32,horizon=3,seed=12))))
        def run(name,*extra):
            stdout,stderr=io.StringIO(),io.StringIO()
            args=['train','--dataset',str(dataset),'--checkpoint',str(self.root/(name+'.pt')),
                  '--config',str(config),*extra]
            with redirect_stdout(stdout),redirect_stderr(stderr):
                try:
                    script['main'](args)
                except SystemExit as exc:
                    self.fail(f'Motion CLI rejected progress control: {exc}')
            events=[json.loads(line) for line in stdout.getvalue().splitlines()]
            return events,stderr.getvalue()
        continuous,stderr=run('continuous')
        self.assertIn('Motion epochs',stderr)
        self.assertIn('Train 1/2',stderr)
        self.assertIn('Validation 1/2',stderr)
        self.assertTrue(any(e.get('phase')=='training' for e in continuous))
        first,quiet=run('first','--epochs','1','--no-progress')
        self.assertEqual(quiet,'')
        resumed,stderr=run('resumed','--resume',str(self.root/'first.latest.pt'))
        self.assertIn('Train 2/2',stderr)
        self.assertNotIn('Train 1/2',stderr)
        self.assertRegex(stderr,r'Motion epochs:.*100%.*2/2')
        self.assertEqual(continuous[-1]['model_version'],resumed[-1]['model_version'])


if __name__=='__main__':
    unittest.main()
