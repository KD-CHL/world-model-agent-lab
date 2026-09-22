"""Exercise actual HTTP requests using a local protocol fixture, no paid API calls."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest
from wmal.communication.contracts import RobotProfile, Observation


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('wmal.agents.api_client'), 'HTTP client missing')
        from wmal.agents.api_client import ApiModelClient
        self.client_class = ApiModelClient
        self.received = []
        self.reply = {'choices': [{'message': {'content': '{"intent":"joint_goal","targets":{"joint":0.4}}'}, 'finish_reason': 'stop'}]}
        self.status = 200
        parent = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                parent.received.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(parent.status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(parent.reply).encode())
            def log_message(self, *args): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.shutdown)
        self.client = self.client_class(f'http://127.0.0.1:{self.server.server_port}/v1', 'test-only', 'fixture')
        self.profile = RobotProfile('arm', 'arm', {'joint': (-1, 1)})
        self.obs = Observation('arm', 'ep', 0, 0, {'joint': 0})

    def test_api_sends_capabilities_and_validates_goal(self):
        goal = self.client.propose_goal('move joint', self.profile, self.obs)
        self.assertEqual(goal.targets, {'joint': .4})
        path, body = self.received[0]
        self.assertEqual(path, '/v1/chat/completions')
        context = json.loads(body['messages'][1]['content'])
        self.assertEqual(context['robot']['joint_limits']['joint'], [-1, 1])

    def test_api_rejects_executable_code_and_out_of_bounds(self):
        for content in ['print(123)', '{"intent":"joint_goal","targets":{"joint":99}}']:
            self.reply['choices'][0]['message']['content'] = content
            with self.assertRaises(ValueError):
                self.client.propose_goal('move', self.profile, self.obs)

    def test_api_error_does_not_expose_response_body(self):
        self.status = 401
        self.reply = {'error': 'private-token-test-only'}
        with self.assertRaises(RuntimeError) as ctx:
            self.client.propose_goal('move', self.profile, self.obs)
        self.assertNotIn('private-token', str(ctx.exception))
        self.assertEqual(len(self.received), 1)

    def test_api_rejects_plaintext_remote_credentials(self):
        with self.assertRaises(ValueError):
            self.client_class('http://example.com/v1', 'secret', 'fixture')

    def test_truncated_response_is_not_executed(self):
        self.reply['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(ValueError):
            self.client.propose_goal('move', self.profile, self.obs)


if __name__ == '__main__': unittest.main()
