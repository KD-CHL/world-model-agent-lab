import unittest
import numpy as np
from wmal.monitor.frame_buffer import LatestFrameBuffer
from wmal.monitor.publisher import FramePublisher


class FramePublisherTests(unittest.TestCase):
    def test_throttles_frames_and_keeps_latest_frame_when_source_stops(self):
        clock = [0.0]
        buffer = LatestFrameBuffer()
        publisher = FramePublisher(buffer, 'run1', 'overview', clock=lambda: clock[0])
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertTrue(publisher.publish(rgb, episode_id='e1', step_id=0, sim_time_s=.2))
        clock[0] = .1
        self.assertFalse(publisher.publish(rgb, episode_id='e1', step_id=1, sim_time_s=.4))
        clock[0] = .3
        self.assertTrue(publisher.publish(rgb, episode_id='e1', step_id=2, sim_time_s=.6))
        publisher.close()
        self.assertFalse(publisher.ready())
        self.assertEqual(buffer.latest('run1').step_id, 2)
        self.assertEqual(publisher.status()['published_frames'], 2)
        self.assertTrue(publisher.status()['closed'])

    def test_bad_telemetry_is_reported_without_raising_into_simulation(self):
        publisher = FramePublisher(LatestFrameBuffer(), 'run1', 'overview')
        self.assertFalse(publisher.publish(np.zeros((1, 1)), episode_id='e1', step_id=0, sim_time_s=0))
        self.assertEqual(publisher.status()['error_count'], 1)
        self.assertEqual(publisher.status()['last_error'], 'ValueError')

    def test_floating_g1_publishes_owner_thread_frames(self):
        from wmal.envs.g1_session import G1MuJoCoSession
        from wmal.locomotion.contracts import G1VelocityAction
        buffer = LatestFrameBuffer()
        publisher = FramePublisher(buffer, 'run-g1', 'g1_overview', width=160, height=120)
        with G1MuJoCoSession(viewer=False, realtime=False, frame_publisher=publisher) as session:
            result = session.step(G1VelocityAction(.1, 0., 0., .2))
            frame = buffer.latest('run-g1')
            self.assertIsNotNone(frame)
            self.assertEqual(frame.rgb.shape, (120, 160, 3))
            self.assertEqual(frame.episode_id, result.episode_id)
            self.assertEqual(frame.step_id, result.step_id)
            self.assertEqual(frame.sim_time_s, result.sim_time_s)
            monitored = (session.data.qpos.copy(), session.data.qvel.copy(), session.data.time)
        with G1MuJoCoSession(viewer=False, realtime=False) as plain:
            plain.step(G1VelocityAction(.1, 0., 0., .2))
            np.testing.assert_array_equal(plain.data.qpos, monitored[0])
            np.testing.assert_array_equal(plain.data.qvel, monitored[1])
            self.assertEqual(plain.data.time, monitored[2])

    def test_mujoco_monitor_does_not_change_physics_results(self):
        from wmal.envs.visual_workcell import VisualWorkcellSession
        buffer = LatestFrameBuffer()
        publisher = FramePublisher(buffer, 'run1', 'workcell_overview', max_fps=30, width=96, height=72)
        def execute(publisher=None):
            with VisualWorkcellSession(image_size=32, frame_publisher=publisher) as session:
                obs = session.observe()
                result = session.execute_actions(np.array([[.04, 0], [-.02, .03]]),
                         episode_id=obs.episode_id, step_id=obs.step_id)
                final = (session.data.qpos.copy(), session.data.qvel.copy(), session.data.time)
                frame = buffer.latest('run1') if publisher else None
                if frame:
                    self.assertEqual(frame.episode_id, result.episode_id)
                    self.assertLessEqual(frame.step_id, result.step_id)
                    self.assertEqual(frame.rgb.shape, (72, 96, 3))
                return final
        plain, monitored = execute(), execute(publisher)
        np.testing.assert_array_equal(plain[0], monitored[0])
        np.testing.assert_array_equal(plain[1], monitored[1])
        self.assertEqual(plain[2], monitored[2])


if __name__ == '__main__':
    unittest.main()
