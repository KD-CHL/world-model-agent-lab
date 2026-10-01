import io
import json
import http.client
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
import contextlib

import numpy as np

from wmal.envs.visual_workcell import VisualWorkcellSession
from wmal.monitor.catalog import RunCatalog
from wmal.monitor.frame_buffer import LatestFrameBuffer
from wmal.monitor.publisher import FramePublisher
from wmal.monitor.server import MonitorServer


class MonitorIntegrationTests(unittest.TestCase):
    def test_g1_cli_monitor_records_manifest_and_completed_goal(self):
        from scripts.g1_agent_sim import main
        from wmal.locomotion.learned import LearnedG1Dynamics
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Contract-valid fixture; this goal is already reached, no model predictions are used.
            LearnedG1Dynamics(np.zeros((2, 10, 6)), .2, 'monitor-test-fixture').save(root / 'model.json')
            config = root / 'experiment.json'
            config.write_text(json.dumps({'world_model': {'factory': 'wmal.locomotion.learned:load',
                'config': {'checkpoint': str(root / 'model.json')}},
                'planner': {'samples': 4, 'horizon': 1, 'action_duration_s': .2}}))
            log = root / 'trial' / 'events.jsonl'
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                try:
                    result = main(['--config', str(config), '--log', str(log), '--goal', '0 0',
                        '--headless', '--monitor', '--monitor-port', '0', '--monitor-runs-root', str(root)])
                except SystemExit as exc:
                    result = exc.code
            self.assertEqual(result, 0, errors.getvalue())
            self.assertIn('http://127.0.0.1:', output.getvalue())
            manifest = json.loads((log.parent / 'run_manifest.json').read_text())
            self.assertEqual(manifest['model_version'], 'monitor-test-fixture')
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(rows[-1]['payload']['status'], 'succeeded')

    def test_visual_agent_records_task_lifecycle_and_aligned_feedback(self):
        run_task = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/visual_skill_agent.py'))['run_task']
        with tempfile.TemporaryDirectory() as directory, VisualWorkcellSession(image_size=32) as session:
            model = SimpleNamespace(version='fixed-A0-test', semantics=session.semantics)
            log = io.StringIO()
            result = run_task(session, model, None, [.45, .95], baseline='A0', horizon=2,
                     max_cycles=10, error_budget=1, seed=0, log=log, artifact_dir=Path(directory))
            rows = [json.loads(line) for line in log.getvalue().splitlines()]
            self.assertEqual(rows[0]['event'], 'task_started')
            self.assertEqual(rows[-1]['event'], 'task_result')
            self.assertTrue(all('wall_time_utc' in row and 'task_id' in row for row in rows))
            feedback = [row for row in rows if row['event'] == 'feedback'][-1]
            self.assertEqual(feedback['episode_id'], session.episode_id)
            self.assertEqual(feedback['step_id'], session.step_id)
            self.assertEqual(rows[-1]['state']['status'], result['state']['status'])

    def test_monitor_disconnection_leaves_simulation_controllable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run'
            path.mkdir()
            (path / 'events.jsonl').write_text('{"event":"task_started"}\n')
            catalog = RunCatalog(directory)
            run_id = catalog.list_runs()[0]['run_id']
            buffer = LatestFrameBuffer()
            publisher = FramePublisher(buffer, run_id, 'workcell_overview', width=96, height=72)
            server = MonitorServer(catalog, buffer, port=0).start()
            try:
                with VisualWorkcellSession(image_size=32, frame_publisher=publisher) as session:
                    client = http.client.HTTPConnection(*server.address)
                    client.request('GET', '/api/live/' + run_id + '/frame.png')
                    response = client.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.read().startswith(b'\x89PNG'))
                    client.close()
                    server.shutdown()
                    before = session.observe()
                    after = session.execute_actions(np.array([[.04, 0.]]),
                            episode_id=before.episode_id, step_id=before.step_id)
                    self.assertEqual(after.step_id, 1)
                    self.assertGreater(after.state[0], before.state[0])
            finally:
                if not server.stopping.is_set():
                    server.shutdown()


if __name__ == '__main__':
    unittest.main()
