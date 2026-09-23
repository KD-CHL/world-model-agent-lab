"""ROS 2 client with a dedicated executor; never spin a node twice."""
from dataclasses import asdict
from threading import Condition, Event, Thread
from time import monotonic
from wmal.communication.contracts import Observation, Plan, ExecutionResult, encode, decode


def wait_future(future, timeout_s):
    ready = Event()
    future.add_done_callback(lambda _: ready.set())
    if not ready.wait(timeout_s):
        raise TimeoutError('ROS response deadline exceeded')
    error = future.exception()
    if error:
        raise RuntimeError('ROS request failed') from None
    return future.result()


class Ros2Channel:
    """Owns its rclpy Context. Public methods must run outside executor callbacks."""
    def __init__(self, profile, state_max_age_s=2.0, planning_service='/wmal/plan'):
        import rclpy
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.action import ActionClient
        from wmal_interfaces.msg import RobotState
        from wmal_interfaces.srv import PlanMotion
        from wmal_interfaces.srv import ResetSimulation
        from wmal_interfaces.action import ExecuteMotion
        self._rclpy = rclpy
        self.profile = profile
        self._context = Context()
        rclpy.init(context=self._context)
        self.node = Node('agent_' + profile.robot_id, context=self._context)
        self._executor = MultiThreadedExecutor(num_threads=2, context=self._context)
        self._executor.add_node(self.node)
        self._condition = Condition()
        self._latest = None
        self._received_at = 0.0
        self._after = None
        self._closed = False
        self._active = None
        self.max_age = state_max_age_s
        namespace = '/wmal/' + profile.robot_id
        self._sub = self.node.create_subscription(RobotState, namespace + '/state', self._state_callback, 1)
        self._planner = self.node.create_client(PlanMotion, planning_service)
        self._reset_client = self.node.create_client(ResetSimulation, namespace + '/reset')
        self._motion = ActionClient(self.node, ExecuteMotion, namespace + '/execute_motion')
        self._request_type, self._goal_type = PlanMotion.Request, ExecuteMotion.Goal
        self._reset_type = ResetSimulation.Request
        self._thread = Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _state_callback(self, message):
        try:
            observation = decode(message.json, Observation)
            observation.validate(self.profile)
        except (ValueError, TypeError, KeyError):
            return
        with self._condition:
            if self._latest and observation.episode_id != self._latest.episode_id:
                return
            if self._latest and observation.episode_id == self._latest.episode_id and observation.step_id < self._latest.step_id:
                return
            self._latest, self._received_at = observation, monotonic()
            self._condition.notify_all()

    def observe(self, timeout_s=5):
        deadline = monotonic() + timeout_s
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError('ROS channel closed')
                obs = self._latest
                fresh_step = self._after is None or (obs and (obs.episode_id != self._after[0] or obs.step_id > self._after[1]))
                if obs and fresh_step and monotonic() - self._received_at <= self.max_age:
                    return obs
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError('No fresh robot state')
                self._condition.wait(remaining)

    def plan(self, profile, observation, goal, timeout_s=30):
        if not self._planner.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError('World planner service unavailable')
        request = self._request_type()
        request.json = encode({'profile': asdict(profile), 'observation': asdict(observation), 'goal': asdict(goal)})
        future = self._planner.call_async(request)
        try:
            response = wait_future(future, timeout_s)
        except TimeoutError:
            self._planner.remove_pending_request(future)
            raise
        if not response.ok:
            raise RuntimeError('World planner rejected request')
        plan = decode(response.json, Plan)
        plan.validate(profile, observation)
        return plan

    def reset(self, timeout_s=10):
        if not self._reset_client.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError('Robot reset service unavailable')
        response = wait_future(self._reset_client.call_async(self._reset_type()), timeout_s)
        if not response.ok:
            raise RuntimeError('Robot reset rejected')
        observation = decode(response.json, Observation)
        observation.validate(self.profile)
        self._after = None
        with self._condition:
            self._latest, self._received_at = observation, monotonic()
            self._condition.notify_all()
        return observation

    def execute(self, command, timeout_s=30):
        command.validate(self.profile)
        if not self._motion.wait_for_server(timeout_sec=timeout_s):
            raise TimeoutError('Robot action server unavailable')
        goal = self._goal_type()
        goal.json = encode(command)
        send = self._motion.send_goal_async(goal)
        try:
            handle = wait_future(send, timeout_s)
        except TimeoutError:
            # A late acceptance must not leave an unowned action running.
            def cancel_late(future):
                if future.exception() is None:
                    late = future.result()
                    if late.accepted:
                        late.cancel_goal_async()
            send.add_done_callback(cancel_late)
            raise
        if not handle.accepted:
            return ExecutionResult(command.command_id, 'rejected', 'Robot rejected motion')
        self._active = handle
        try:
            reply = wait_future(handle.get_result_async(), timeout_s)
            result = decode(reply.result.json, ExecutionResult)
            if result.command_id != command.command_id:
                raise ValueError('Mismatched action result')
            from action_msgs.msg import GoalStatus
            if result.status == 'succeeded' and reply.status != GoalStatus.STATUS_SUCCEEDED:
                raise ValueError('Action status mismatch')
            self._after = (command.episode_id, command.expected_step)
            return result
        except TimeoutError:
            try:
                wait_future(handle.cancel_goal_async(), min(timeout_s, 2.0))
            except (TimeoutError, RuntimeError):
                pass
            return ExecutionResult(command.command_id, 'timeout', 'Cancellation requested; do not retry motion')
        finally:
            self._active = None

    def close(self):
        if self._closed:
            return
        if self._active:
            self._active.cancel_goal_async()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(timeout_sec=2)
        self._thread.join(timeout=2)
        self._motion.destroy()
        self.node.destroy_node()
        self._context.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
