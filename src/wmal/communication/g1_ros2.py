"""G1 session service and client; ROS imports deferred for non-ROS testing."""
from contextlib import AbstractContextManager
from dataclasses import asdict
import json
import math
import time
from threading import Thread
from uuid import uuid4

from wmal.communication.g1_session_protocol import SCHEMA, state_from_wire
from wmal.communication.ros2_transport import wait_future


def create_g1_session_node(owner, service_name='/wmal/g1/session'):
    from rclpy.node import Node
    from wmal_interfaces.srv import G1Session

    class SessionNode(Node):
        def __init__(self):
            super().__init__('g1_simulation_session')
            self.service = self.create_service(G1Session, service_name, self._handle_request)

        def _handle_request(self, request, response):
            try:
                if len(request.json) > 100_000:
                    raise ValueError('Oversize session request')
                response.json = json.dumps(owner.handle(json.loads(request.json)), allow_nan=False)
                response.ok = True
            except Exception as exc:
                self.get_logger().error('Session request failed: ' + type(exc).__name__)
                response.ok, response.json = False, json.dumps({'error': type(exc).__name__})
            return response

    return SessionNode()


class G1RemoteSession(AbstractContextManager):
    """Transport-independent client contract; failed mutation latches locally."""
    def __init__(self, rpc, scene_id, request_ttl_s=10.):
        if not math.isfinite(request_ttl_s) or request_ttl_s <= 0:
            raise ValueError("Invalid request lifetime")
        self.request_ttl_s = request_ttl_s
        self.rpc, self.scene_id = rpc, scene_id
        self.closed, self.faulted = False, False
        self._state = None

    @property
    def is_running(self):
        return not self.closed and not self.faulted

    def _request(self, operation, **fields):
        if self.closed or (self.faulted and operation == 'step'):
            raise RuntimeError('Remote session is closed or faulted')
        request_id = str(uuid4())
        if operation != 'observe':
            fields['expires_at_unix_s'] = time.time() + self.request_ttl_s
        try:
            response = self.rpc({'schema': SCHEMA, 'request_id': request_id,
                                 'scene_id': self.scene_id, 'operation': operation, **fields})
            if (response.get('schema') != SCHEMA or response.get('request_id') != request_id
                    or response.get('scene_id') != self.scene_id or response.get('status') != 'ok'):
                raise RuntimeError('Invalid or uncertain remote receipt')
            state = state_from_wire(response['state'])
            if type(response.get('faulted')) is not bool:
                raise ValueError('Missing or invalid remote fault state')
            self.faulted = self.faulted or response['faulted']
            self._state = state
            return state
        except Exception:
            if operation != 'observe':
                self.faulted = True
            raise

    def observe(self):
        return self._request('observe')

    def step(self, action, duration_s=None):
        if duration_s is not None and duration_s != action.duration_s:
            raise ValueError('Action duration mismatch')
        if self._state is None:
            raise ValueError('Observe before planning a remote action')
        before = self._state
        after = self._request('step', episode_id=before.episode_id, step_id=before.step_id,
                              sim_time_s=before.sim_time_s, action=asdict(action))
        if (after.episode_id != before.episode_id or after.step_id != before.step_id+1
                or abs(after.sim_time_s-before.sim_time_s-action.duration_s) > 1e-6):
            self.faulted = True
            raise ValueError('Remote execution time or step mismatch')
        return after

    def reset(self):
        before = self.observe()
        after = self._request('reset', episode_id=before.episode_id, step_id=before.step_id,
                              sim_time_s=before.sim_time_s)
        if after.episode_id == before.episode_id:
            self.faulted = True
            raise ValueError('Reset did not change episode')
        self.faulted = False
        return after

    def set_goal_marker(self, goal):
        # Markers are visualization only; no implicit remote mutation.
        pass

    def close(self):
        self.closed = True

    def __exit__(self, *args):
        self.close()


class G1Ros2Session(G1RemoteSession):
    def __init__(self, scene_id, service_name='/wmal/g1/session', timeout_s=10.):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Invalid ROS timeout')
        import rclpy
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from wmal_interfaces.srv import G1Session
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node('g1_agent_session_' + uuid4().hex[:8], context=self.context)
        self.client = self.node.create_client(G1Session, service_name)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        self.timeout_s, self.request_type = timeout_s, G1Session.Request
        super().__init__(self._rpc, scene_id, request_ttl_s=timeout_s)

    def _rpc(self, payload):
        if not self.client.wait_for_service(timeout_sec=self.timeout_s):
            raise TimeoutError('G1 session service unavailable')
        request = self.request_type()
        request.json = json.dumps(payload, allow_nan=False)
        future = self.client.call_async(request)
        try:
            response = wait_future(future, self.timeout_s)
        except TimeoutError:
            self.client.remove_pending_request(future)
            raise
        if not response.ok or len(response.json) > 1_000_000:
            raise RuntimeError('G1 session rejected request')
        return json.loads(response.json)

    def close(self):
        if self.closed:
            return
        super().close()
        self.executor.shutdown(timeout_sec=2)
        self.thread.join(timeout=2)
        self.node.destroy_node()
        self.context.shutdown()
