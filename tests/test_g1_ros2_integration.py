"""Live middleware test, run after building/sourcing wmal_interfaces on Ubuntu."""
import importlib.util
from threading import Thread
import unittest
from uuid import uuid4
from dataclasses import replace

ROS_AVAILABLE = (importlib.util.find_spec('rclpy') is not None
                 and importlib.util.find_spec('wmal_interfaces') is not None)


@unittest.skipUnless(ROS_AVAILABLE, 'Requires ROS2 and generated wmal_interfaces')
class RosIntegrationTests(unittest.TestCase):
    def test_live_service_observe_step_reset(self):
        import rclpy
        from wmal.communication.g1_session_protocol import G1SessionOwner
        from wmal.communication.g1_ros2 import create_g1_session_node, G1Ros2Session
        from wmal.locomotion.contracts import G1State, G1VelocityAction
        class Session:
            def __init__(self):
                self.state = G1State(str(uuid4()),0,0.,0.,0.,.8,0.,0.,0.,0.,0.,0.,0.,.8)
            def observe(self): return self.state
            def step(self, action, duration):
                self.state = replace(self.state, step_id=self.state.step_id+1,
                                     sim_time_s=self.state.sim_time_s+duration)
            def close(self): pass
        service = '/wmal/test_' + uuid4().hex
        owner = G1SessionOwner(Session, 'test-scene')
        rclpy.init()
        node = create_g1_session_node(owner, service)
        from rclpy.executors import SingleThreadedExecutor
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        thread = Thread(target=executor.spin, daemon=True)
        thread.start()
        try:
            with G1Ros2Session('test-scene', service) as remote:
                before = remote.observe()
                after = remote.step(G1VelocityAction(.1,0.,0.,.5))
                self.assertEqual(after.step_id, before.step_id+1)
                self.assertNotEqual(remote.reset().episode_id, before.episode_id)
        finally:
            executor.shutdown(timeout_sec=2)
            thread.join(timeout=2)
            node.destroy_node()
            owner.close()
            rclpy.shutdown()
