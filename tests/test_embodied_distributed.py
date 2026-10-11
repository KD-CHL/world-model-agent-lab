"""Real gloo subprocesses: forward hooks, common bootstrap and all-rank replay."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from test_embodied_training import fixture, tiny_config, tensors_equal


class EmbodiedDistributedTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('wmal.training.embodied_distributed')
        except ModuleNotFoundError as exc:
            self.fail(f'Task4 distributed API missing: {exc.name}')

    # Mean(sqrt(mean(error^2 over samples))) per H/P, NOT sqrt(mean all) nor KL.
    def test_unique_per_horizon_dimension_selection(self):
        api = self.api()
        records = [('a', [[0., 16.], [4., 36.]]), ('b', [[0., 16.], [4., 36.]]),
                   ('a', [[0., 16.], [4., 36.]])]
        result = api.unique_validation_metrics(records, [2., 3.])
        self.assertEqual(result['unique_samples'], 2)
        self.assertEqual(result['selection_metric'], 3.)  # (0+4+2+6)/4
        self.assertEqual(result['physical_rmse'], [[0., 12.], [4., 18.]])
        self.assertEqual(api.select_epoch([3., 3., 4.]), 1)
        with self.assertRaises(ValueError):
            api.unique_validation_metrics(records + [('a', [[1., 16.], [4., 36.]])], [2., 3.])

    def test_fsdp_device_gate_and_launch_validation(self):
        api = self.api()
        with self.assertRaisesRegex(ValueError, 'CUDA|version|FSDP'):
            api.DistributedRuntime('fsdp', 'cpu')
        with self.assertRaisesRegex(ValueError, 'torchrun|WORLD_SIZE|rank'):
            api.DistributedRuntime('ddp', 'cpu')

    def test_two_rank_cpu_unique_validation_and_checkpoint(self):
        self.api(); trainer = importlib.import_module('wmal.training.embodied_trainer')
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest = fixture(root / 'data')
            cfg = tiny_config(trainer, epochs=1, distributed='ddp')
            path = root / 'config.json'; path.write_text(json.dumps(asdict(cfg)))
            # Runs this test file's worker, never child agents. Torchrun owns processes/errors.
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=2', str(Path(__file__).resolve()), '--worker',
                str(manifest), str(root / 'run'), str(path)]
            env = dict(os.environ, PYTHONPATH='tests:src', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr.count('Independent RSSM:'), 1)
            self.assertNotIn('Independent RSSM:', result.stdout)
            report = json.loads((root / 'run/report.json').read_text())
            self.assertEqual(report['world_size'], 2)
            self.assertEqual(report['members'][0]['validation_unique_samples'], 4)
            self.assertEqual(report['members'][0]['optimizer_steps'], 3)
            self.assertEqual(report['members'][0]['effective_batch_sizes'], [4, 4, 2])
            checkpoint = torch.load(root / 'run/training.pt', weights_only=True)
            self.assertEqual(len(checkpoint['rank_states']), 2)
            self.assertEqual(checkpoint['rank_states'][0]['sampler']['bootstrap'],
                             checkpoint['rank_states'][1]['sampler']['bootstrap'])
            tensors_equal(self, checkpoint['model'],
                          torch.load(root / 'run/rank1-final.pt', weights_only=True))
            # Real process restart, not unsupported in-process process-group reinitialization.
            command[-2] = str(root / 'replay')
            result = subprocess.run(command + ['--interrupt'], capture_output=True, text=True, env=env, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            boundary = torch.load(root / 'replay/training.pt', weights_only=True)
            self.assertEqual(boundary['position']['step'], 1)
            result = subprocess.run(command + ['--resume'], capture_output=True, text=True, env=env, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            replay = torch.load(root / 'replay/training.pt', weights_only=True)
            tensors_equal(self, checkpoint['model'], replay['model'])
            tensors_equal(self, checkpoint['optimizer']['state'], replay['optimizer']['state'])
            self.assertEqual(checkpoint['rank_states'], replay['rank_states'])


def worker():
    from unittest.mock import patch
    from wmal.training import embodied_trainer as trainer
    from wmal.training.embodied_checkpoint import save_training_checkpoint
    from test_embodied_training import Interrupted
    manifest, output, config = sys.argv[2:5]
    cfg = trainer.EmbodiedTrainingConfig(**json.loads(Path(config).read_text()))
    torch.set_num_threads(1)
    local_snapshots = []
    def save_local(path, payload):
        save_training_checkpoint(path, payload)
        local_snapshots.append(payload['model'])
    if '--interrupt' in sys.argv:
        def stop(path, payload):
            save_training_checkpoint(path, payload)
            raise Interrupted()
        try:
            with patch.object(trainer, 'save_training_checkpoint', side_effect=stop):
                trainer.train_embodied(manifest, output, cfg, show_progress=True)
        except Interrupted:
            pass
        return
    elif '--resume' in sys.argv:
        with patch.object(trainer, 'save_training_checkpoint', side_effect=save_local):
            trainer.train_embodied(manifest, output, cfg, resume=Path(output) / 'training.pt', show_progress=True)
    else:
        with patch.object(trainer, 'save_training_checkpoint', side_effect=save_local):
            trainer.train_embodied(manifest, output, cfg, show_progress=True)
    if int(os.environ['RANK']) == 1:
        torch.save(local_snapshots[-1], Path(output) / 'rank1-final.pt')


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        unittest.main()
