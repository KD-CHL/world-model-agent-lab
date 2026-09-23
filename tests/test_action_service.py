"""Tests the external action service boundary without GPU or ROS dependencies."""
import io
import json
import unittest

from wmal.models.action_service import ActionQuery, ActionServiceClient


class _Reply:
    def __init__(self, payload):
        self.buffer = io.BytesIO(payload)

    def __enter__(self): return self
    def __exit__(self, *_): return None
    def read(self, amount): return self.buffer.read(amount)


class _Opener:
    def __init__(self, result):
        self.result = result
        self.request = None

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return _Reply(json.dumps(self.result).encode())


class ActionServiceTests(unittest.TestCase):
    def query(self):
        image = [[[1, 2]], [[3, 4]], [[5, 6]]]
        return ActionQuery('move the arm', [[0.0, 0.2], [0.1, 0.3]],
                           [image, image], [[0.0, 0.0]] * 4, 2, 2)

    def test_request_maps_history_and_returns_bounded_chunk(self):
        opener = _Opener({'result': 'ok', 'action': [[0.2, 0.3], [0.4, 0.5]]})
        client = ActionServiceClient('http://127.0.0.1:8000', 'checkpoint-1', opener=opener)
        chunk = client.propose(self.query())
        self.assertEqual(chunk.actions[0], [0.2, 0.3])
        self.assertEqual(chunk.model_version, 'checkpoint-1')
        sent = json.loads(opener.request.data)
        self.assertEqual(sent['observation.state'][1], [0.1, 0.3])
        self.assertEqual(sent['observation.images.top'][0][0][0], [1, 2])
        self.assertEqual(opener.request.full_url, 'http://127.0.0.1:8000/predict_action')

    def test_rejects_incompatible_robot_data_before_call(self):
        opener = _Opener({'result': 'ok', 'action': [[0.2, 0.3]]})
        client = ActionServiceClient('http://localhost:8000', 'm', opener=opener)
        query = self.query()
        query.zero_actions[0][0] = 0.1
        with self.assertRaises(ValueError): client.propose(query)
        self.assertIsNone(opener.request)

    def test_rejects_nonfinite_or_wrong_width_response(self):
        for actions in ([[0.1]], [[float('nan'), 0.1]]):
            with self.subTest(actions=actions):
                client = ActionServiceClient('http://localhost:8000', 'm', opener=_Opener({'result': 'ok', 'action': actions}))
                with self.assertRaises(ValueError): client.propose(self.query())

    def test_remote_plain_http_is_rejected(self):
        with self.assertRaises(ValueError): ActionServiceClient('http://model.example:8000', 'm')


if __name__ == '__main__':
    unittest.main()
