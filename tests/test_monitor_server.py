import http.client
import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np

from wmal.monitor.catalog import RunCatalog
from wmal.monitor.contracts import SimFrame
from wmal.monitor.frame_buffer import LatestFrameBuffer
from wmal.monitor.server import MonitorServer


class MonitorServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / 'experiment'
        self.run.mkdir()
        self.log = self.run / 'events.jsonl'
        self.log.write_text('{"event":"task_started","payload":{"task_id":"t1","max_cycles":10}}\n', encoding='utf-8')
        self.catalog = RunCatalog(self.root)
        self.run_id = self.catalog.list_runs()[0]['run_id']
        self.buffer = LatestFrameBuffer()
        self.server = MonitorServer(self.catalog, self.buffer, port=0).start()

    def tearDown(self):
        self.server.shutdown()
        self.temp.cleanup()

    def request(self, path, method='GET', headers=None):
        connection = http.client.HTTPConnection(*self.server.address, timeout=3)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_catalog_and_incremental_event_cursor(self):
        status, _, data = self.request('/api/runs')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)['runs'][0]['run_id'], self.run_id)
        base = '/api/runs/' + self.run_id
        first = json.loads(self.request(base + '/events?after=0')[2])
        self.assertEqual(len(first['events']), 1)
        self.assertEqual(first['summary']['tasks'][0]['status'], 'running')
        with self.log.open('a') as stream:
            stream.write('{"event":"task_result","payload":{"task_id":"t1","status":"succeeded","cycles":3}}\n')
        second = json.loads(self.request(base + '/events?after=1')[2])
        self.assertEqual([row['event'] for row in second['events']], ['task_result'])
        self.assertEqual(second['summary']['tasks'][0]['status'], 'succeeded')

    def test_latest_frame_and_metadata_are_from_the_selected_run(self):
        self.buffer.publish(SimFrame(self.run_id, 'episode1', 7, 1.4,
                                    datetime.now(timezone.utc).isoformat(), 'overview',
                                    np.zeros((8, 8, 3), dtype=np.uint8)))
        status, headers, data = self.request('/api/live/' + self.run_id + '/frame.png')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'image/png')
        self.assertEqual(headers['X-Episode-ID'], 'episode1')
        self.assertEqual(float(headers.get('X-Sim-Time-S', -1)), 1.4)
        self.assertIn('X-Captured-At-UTC', headers)
        self.assertTrue(data.startswith(b'\x89PNG'))
        live = json.loads(self.request('/api/live/' + self.run_id + '/status')[2])
        self.assertEqual(live['step_id'], 7)
        self.assertEqual(live['state'], 'live')
        self.assertEqual(self.request('/api/live/unknown/frame.png')[0], 404)

    def test_sse_emits_a_real_event_snapshot(self):
        connection = http.client.HTTPConnection(*self.server.address, timeout=3)
        connection.request('GET', '/api/runs/' + self.run_id + '/stream')
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers['Content-Type'], 'text/event-stream')
        lines = []
        while True:
            line = response.readline()
            lines.append(line)
            if line == b'\n':
                break
        self.assertIn(b'task_started', b''.join(lines))
        connection.close()

    def test_root_boundary_origin_and_read_only_routes(self):
        base = '/api/runs/' + self.run_id
        self.assertEqual(self.request(base + '/artifacts/%2e%2e/outside.json')[0], 400)
        self.assertEqual(self.request('/api/runs', headers={'Origin': 'https://external.example'})[0], 403)
        self.assertEqual(self.request('/api/runs', headers={'Host': 'external.example'})[0], 403)
        self.assertEqual(self.request('/api/execute', method='POST')[0], 405)
        self.assertEqual(self.request(base + '/events?after=nan')[0], 400)

    def test_secret_fields_are_redacted_in_json_artifacts(self):
        (self.run / 'task_001.json').write_text(json.dumps({
            'api_key': 'private-value', 'state': {'status': 'succeeded'},
            'authorization': 'Bearer private-value'}))
        status, _, data = self.request('/api/runs/' + self.run_id + '/artifacts/task_001.json')
        self.assertEqual(status, 200)
        self.assertNotIn(b'private-value', data)
        self.assertEqual(json.loads(data)['state']['status'], 'succeeded')

    def test_non_loopback_binding_is_rejected(self):
        with self.assertRaises(ValueError):
            MonitorServer(self.catalog, self.buffer, host='0.0.0.0', port=0)


if __name__ == '__main__':
    unittest.main()
