"""Live middleware test, run after building/sourcing wmal_interfaces on Ubuntu."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Thread
import unittest
from uuid import uuid4
from dataclasses import replace

ROS_AVAILABLE = (importlib.util.find_spec('rclpy') is not None
                 and importlib.util.find_spec('wmal_interfaces') is not None)


@unittest.skipUnless(ROS_AVAILABLE, 'Requires ROS2 and generated wmal_interfaces')
class RosIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('mujoco') is not None
                         and importlib.util.find_spec('onnxruntime') is not None,
                         'Requires G1 simulation and walking dependencies')
    def test_real_simulation_service_shuts_down_cleanly_on_sigint(self):
        from wmal.communication.g1_ros2 import G1Ros2Session
        from wmal.communication.g1_session_protocol import scene_digest
        from wmal.envs.indoor_scene import IndoorScene
        service = '/wmal/test_' + uuid4().hex
        process = subprocess.Popen([sys.executable, 'scripts/serve_g1_ros2.py',
                                    '--service', service],
                                   cwd=Path(__file__).resolve().parents[1],
                                   env=dict(os.environ, MUJOCO_GL='egl'),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            with G1Ros2Session(scene_digest(IndoorScene()), service) as client:
                self.assertGreater(client.observe().pelvis_height, .48)
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stdout + stderr)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

    def test_planner_service_starts_and_rejects_invalid_payload(self):
        import rclpy
        from wmal.communication.ros2_nodes import create_planner_node
        from wmal_interfaces.srv import PlanMotion
        # No planner should be invoked for a payload missing profile/observation/goal.
        rclpy.init()
        node = None
        try:
            node = create_planner_node(object())
            client = node.create_client(PlanMotion, '/wmal/plan')
            self.assertTrue(client.wait_for_service(timeout_sec=3))
            future = client.call_async(PlanMotion.Request(json='{}'))
            rclpy.spin_until_future_complete(node, future, timeout_sec=3)
            self.assertTrue(future.done())
            self.assertFalse(future.result().ok)
            self.assertEqual(json.loads(future.result().json),
                             {'schema_version': 1, 'payload': {'error': 'PLANNING_FAILED'}})
        finally:
            if node is not None:
                node.destroy_node()
            rclpy.shutdown()

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
