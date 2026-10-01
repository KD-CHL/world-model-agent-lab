import json
from pathlib import Path
import tempfile
import unittest

from wmal.monitor.events import JsonlEventTail


class JsonlEventTailTests(unittest.TestCase):
    def test_same_inode_same_length_rewrite_preserving_first_row_resets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_text('{"event":"head"}\n{"event":"old"}\n')
            tail = JsonlEventTail(path)
            self.assertEqual(len(tail.read_new()), 2)
            path.write_text('{"event":"head"}\n{"event":"new"}\n')
            rows = tail.read_new()
            self.assertEqual([row['event'] for row in rows], ['head','new'])
            self.assertEqual(tail.status['source_generation'], 1)

    def test_excessively_nested_json_is_reported_and_next_event_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_text('{"event":"plan","payload":{"nested":' + '[' * 1200 + '0'
                            + ']' * 1200 + '}}\n{"event":"ok"}\n')
            tail = JsonlEventTail(path)
            try:
                rows = tail.read_new()
            except RecursionError:
                self.fail('A malformed deeply nested row escaped the read-only event reader')
            self.assertEqual([r['event'] for r in rows], ['ok'])
            self.assertEqual(tail.status['error_count'], 1)

    def test_discarding_incomplete_oversize_line_keeps_memory_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_bytes(b'x' * 100)
            tail = JsonlEventTail(path, max_line_bytes=32)
            self.assertEqual(tail.read_new(), [])
            for _ in range(3):
                with path.open('ab') as stream:
                    stream.write(b'x' * 100)
                self.assertEqual(tail.read_new(), [])
                self.assertLessEqual(len(tail._pending), 32)
            with path.open('ab') as stream:
                stream.write(b'\n{"event":"ok"}\n')
            self.assertEqual([r['event'] for r in tail.read_new()], ['ok'])

    def test_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_text('{"event":"plan","score":NaN}\n', encoding='utf-8')
            tail = JsonlEventTail(path)
            self.assertEqual(tail.read_new(), [])
            self.assertEqual(tail.status['error_count'], 1)

    def test_reads_wrapped_and_flat_events_without_losing_raw_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_text(
                json.dumps({'wall_time_utc': '2026-09-30T00:00:00Z', 'event': 'plan',
                            'payload': {'task_id': 'task-1', 'prediction': {'score': 0.2}}}) + '\n'
                + json.dumps({'event': 'feedback', 'episode_id': 'ep-1', 'step_id': 4,
                              'observed_state': [1, 2]}) + '\n', encoding='utf-8')
            tail = JsonlEventTail(path)

            rows = tail.read_new()

            self.assertEqual([row['event'] for row in rows], ['plan', 'feedback'])
            self.assertEqual(rows[0]['payload']['prediction'], {'score': 0.2})
            self.assertEqual(rows[1]['payload']['observed_state'], [1, 2])
            self.assertEqual(rows[1]['raw']['episode_id'], 'ep-1')
            self.assertEqual([row['cursor'] for row in rows], [1, 2])
            self.assertEqual(tail.read_new(), [])

    def test_holds_partial_line_until_newline_then_emits_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_bytes(b'{"event":"plan","cycle":')
            tail = JsonlEventTail(path)

            self.assertEqual(tail.read_new(), [])
            with path.open('ab') as stream:
                stream.write(b'3}\n')
            self.assertEqual(tail.read_new()[0]['payload']['cycle'], 3)
            self.assertEqual(tail.read_new(), [])

    def test_reports_malformed_rows_and_continues_with_following_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_bytes(b'{broken json}\n{"event":"task_result","status":"failed"}\n')
            tail = JsonlEventTail(path)

            rows = tail.read_new()

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['event'], 'task_result')
            self.assertEqual(rows[0]['cursor'], 1)
            self.assertEqual(tail.status['error_count'], 1)
            self.assertEqual(tail.status['last_error']['line'], 1)

    def test_truncation_and_file_replacement_restart_source_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_bytes(b'{"event":"plan","cycle":1}\n')
            tail = JsonlEventTail(path)
            self.assertEqual(tail.read_new()[0]['cursor'], 1)

            path.write_bytes(b'{"event":"feedback","step_id":2}\n')
            truncated = tail.read_new()
            self.assertEqual(truncated[0]['event'], 'feedback')
            self.assertEqual(truncated[0]['source_generation'], 1)

            replacement = Path(directory) / 'replacement.jsonl'
            replacement.write_bytes(b'{"event":"task_result","status":"succeeded"}\n')
            replacement.replace(path)
            replaced = tail.read_new()
            self.assertEqual(replaced[0]['event'], 'task_result')
            self.assertEqual(replaced[0]['source_generation'], 2)
            self.assertEqual(replaced[0]['cursor'], 3)

    def test_oversized_line_is_discarded_without_blocking_next_event(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_bytes(b'{"event":"' + b'x' * 100 + b'"}\n{"event":"ok"}\n')
            tail = JsonlEventTail(path, max_line_bytes=32)

            rows = tail.read_new()

            self.assertEqual([row['event'] for row in rows], ['ok'])
            self.assertEqual(tail.status['error_count'], 1)
            self.assertEqual(tail.status['last_error']['kind'], 'line_too_large')


if __name__ == '__main__':
    unittest.main()
