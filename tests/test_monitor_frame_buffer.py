import threading
import unittest

import numpy as np

from wmal.monitor.contracts import SimFrame
from wmal.monitor.frame_buffer import LatestFrameBuffer


def frame(run_id, episode_id, step_id):
    return SimFrame(run_id, episode_id, step_id, step_id / 10,
                    '2026-09-30T00:00:00Z', 'overview',
                    np.full((2, 3, 3), step_id % 255, dtype=np.uint8))


class LatestFrameBufferTests(unittest.TestCase):
    def test_history_is_bounded_and_counts_overwritten_frames(self):
        buffer = LatestFrameBuffer(max_history=2)
        buffer.publish(frame('run-a', 'episode-a', 0))
        buffer.publish(frame('run-a', 'episode-a', 1))
        buffer.publish(frame('run-a', 'episode-a', 2))

        self.assertEqual([item.step_id for item in buffer.history('run-a')], [1, 2])
        self.assertEqual(buffer.dropped_frames, 1)
        self.assertEqual(buffer.latest('run-a').step_id, 2)

    def test_run_and_episode_filters_never_return_an_unrelated_frame(self):
        buffer = LatestFrameBuffer(max_history=4)
        buffer.publish(frame('run-a', 'episode-a', 1))
        buffer.publish(frame('run-b', 'episode-b', 2))
        buffer.publish(frame('run-a', 'episode-c', 3))

        self.assertEqual(buffer.latest('run-a').step_id, 3)
        self.assertIsNone(buffer.latest('run-missing'))
        self.assertEqual([item.step_id for item in buffer.history('run-a', 'episode-a')], [1])
        self.assertEqual(buffer.history('run-b', 'episode-a'), [])

    def test_publish_copies_frame_and_concurrent_readers_see_valid_snapshots(self):
        buffer = LatestFrameBuffer(max_history=8)
        source = frame('run-a', 'episode-a', 1)
        buffer.publish(source)
        self.assertFalse(buffer.latest('run-a').rgb.flags.writeable)

        errors = []

        def writer():
            try:
                for step in range(2, 102):
                    buffer.publish(frame('run-a', 'episode-a', step))
            except Exception as exc:  # surfaced in the main test thread
                errors.append(exc)

        thread = threading.Thread(target=writer)
        thread.start()
        while thread.is_alive():
            current = buffer.latest('run-a')
            if current is not None and (current.rgb.shape != (2, 3, 3)
                                        or current.rgb.flags.writeable):
                errors.append(AssertionError('reader observed a malformed or writable frame'))
                break
        thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(buffer.history('run-a')), 8)
        self.assertEqual(buffer.dropped_frames, 93)

    def test_rejects_non_frame_publication(self):
        with self.assertRaises(TypeError):
            LatestFrameBuffer().publish(object())


if __name__ == '__main__':
    unittest.main()
