import unittest

import numpy as np

from wmal.monitor.contracts import RunDescriptor, SimFrame


class MonitorContractTests(unittest.TestCase):
    def test_frame_keeps_rgb_snapshot_and_marks_it_read_only(self):
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        frame = SimFrame('run-a', 'episode-a', 2, 0.4, '2026-09-30T00:00:00Z', 'overview', rgb)

        rgb[0, 0, 0] = 255
        self.assertEqual(int(frame.rgb[0, 0, 0]), 0)
        with self.assertRaises(ValueError):
            frame.rgb[0, 0, 0] = 255

    def test_frame_rejects_invalid_identity_time_and_rgb(self):
        valid = ('run-a', 'episode-a', 0, 0.0, '2026-09-30T00:00:00Z', 'overview')
        invalid = [
            (('', *valid[1:]), np.zeros((2, 2, 3), dtype=np.uint8)),
            ((valid[0], '', *valid[2:]), np.zeros((2, 2, 3), dtype=np.uint8)),
            ((*valid[:2], -1, *valid[3:]), np.zeros((2, 2, 3), dtype=np.uint8)),
            ((*valid[:3], float('nan'), *valid[4:]), np.zeros((2, 2, 3), dtype=np.uint8)),
            (valid, np.zeros((2, 2), dtype=np.uint8)),
            (valid, np.zeros((2, 2, 4), dtype=np.uint8)),
            (valid, np.zeros((2, 2, 3), dtype=np.float32)),
        ]
        for metadata, rgb in invalid:
            with self.subTest(metadata=metadata, shape=rgb.shape, dtype=rgb.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    SimFrame(*metadata, rgb)

    def test_run_descriptor_requires_an_opaque_id_and_path(self):
        descriptor = RunDescriptor('run-a', 'visual/run-a', 'visual_agent', None)
        self.assertEqual(descriptor.run_id, 'run-a')
        for run_id, path in (('', 'visual/run-a'), ('run-a', '')):
            with self.subTest(run_id=run_id, path=path):
                with self.assertRaises(ValueError):
                    RunDescriptor(run_id, path, None, None)


if __name__ == '__main__':
    unittest.main()
