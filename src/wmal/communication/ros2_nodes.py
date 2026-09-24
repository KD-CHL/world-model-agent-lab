"""ROS node factories. ROS imports are local so core tests run without ROS."""
from dataclasses import asdict
from threading import RLock
from time import monotonic, sleep
from wmal.communication.contracts import RobotProfile, Observation, Goal, MotionCommand, ExecutionResult, encode, decode
from wmal.communication.episode_guard import EpisodeGuard


def create_planner_node(planner):
    from rclpy.node import Node
    from wmal_interfaces.srv import PlanMotion

    class PlannerNode(Node):
        def __init__(self):
            super().__init__('world_model_planner')
            self.service = self.create_service(PlanMotion, '/wmal/plan', self.handle)

        def handle(self, request, response):
            try:
                data = decode(request.json, dict)
                profile = RobotProfile(**data['profile'])
                observation = Observation(**data['observation'])
                goal = Goal(**data['goal'])
                plan = planner.plan(profile, observation, goal)
                plan.validate(profile, observation)
                response.json, response.ok = encode(plan), True
            except Exception as exc:
                self.get_logger().error('Planning failed: ' + type(exc).__name__)
                response.json, response.ok = encode({'error': 'PLANNING_FAILED'}), False
            return response

    return PlannerNode()


def create_robot_node(backend, tolerance=0.03, stable_steps=5, camera=None):
    from rclpy.node import Node
    from rclpy.action import ActionServer, GoalResponse, CancelResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from wmal_interfaces.msg import RobotState
    if camera is not None:
        from wmal_interfaces.msg import RobotSensorFrame
    from wmal_interfaces.srv import ResetSimulation
    from wmal_interfaces.action import ExecuteMotion

    class RobotNode(Node):
        def __init__(self):
            super().__init__('robot_' + backend.profile.robot_id)
            self.lock = RLock()
            self.guard = EpisodeGuard()
            self.busy = False
            self.active = False
            self.fault = False
            namespace = '/wmal/' + backend.profile.robot_id
            self.publisher = self.create_publisher(RobotState, namespace + '/state', 1)
            self.camera = camera
            self.sensor_publisher = None
            if camera is not None:
                if (not isinstance(camera, dict) or not camera.get('name')
                        or type(camera.get('width')) is not int or type(camera.get('height')) is not int
                        or not 1 <= camera['width'] <= 2048 or not 1 <= camera['height'] <= 2048):
                    raise ValueError('Camera requires name and bounded integer width/height')
                camera_id = backend.mj.mj_name2id(backend.model, backend.mj.mjtObj.mjOBJ_CAMERA, camera['name'])
                if camera_id < 0:
                    raise ValueError('Configured camera is missing from MuJoCo asset')
                self.sensor_publisher = self.create_publisher(
                    RobotSensorFrame, camera.get('topic', namespace + '/sensor_frame'), 1)
            self.reset_service = self.create_service(ResetSimulation, namespace + '/reset', self.reset)
            self.group = ReentrantCallbackGroup()
            self.action = ActionServer(self, ExecuteMotion, namespace + '/execute_motion',
                                       execute_callback=self.execute, goal_callback=self.accept,
                                       cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=self.group)
            self.timer = self.create_timer(float(backend.model.opt.timestep), self.tick)
            self.state_timer = self.create_timer(0.05, self.publish_state)

        def publish_state(self):
            with self.lock:
                message = RobotState()
                observation = backend.observe()
                message.json = encode(observation)
                self.publisher.publish(message)
                if self.sensor_publisher is not None:
                    rgb = backend.render_rgb(camera=self.camera['name'], width=self.camera['width'],
                                             height=self.camera['height'])
                    frame = RobotSensorFrame()
                    frame.observation_json = message.json
                    frame.camera = self.camera['name']
                    frame.image.header.stamp = self.get_clock().now().to_msg()
                    frame.image.header.frame_id = self.camera['name']
                    frame.image.height, frame.image.width = self.camera['height'], self.camera['width']
                    frame.image.encoding, frame.image.is_bigendian = 'rgb8', 0
                    frame.image.step = self.camera['width'] * 3
                    frame.image.data = rgb.tobytes()
                    self.sensor_publisher.publish(frame)

        def reset(self, request, response):
            del request
            with self.lock:
                if self.busy:
                    response.ok, response.json = False, encode({'error': 'ROBOT_BUSY'})
                    return response
                try:
                    observation = backend.reset()
                    self.fault = False
                    self.guard = EpisodeGuard()
                    response.ok, response.json = True, encode(observation)
                except Exception as exc:
                    self.fault = True
                    self.get_logger().error('Reset failed: ' + type(exc).__name__)
                    response.ok, response.json = False, encode({'error': 'RESET_FAILED'})
            return response

        def accept(self, request):
            with self.lock:
                if self.busy or self.fault:
                    return GoalResponse.REJECT
                try:
                    command = decode(request.json, MotionCommand)
                    backend.interface.validate_command(command)
                    observation = backend.observe()
                    if command.mode == 'joint_positions' and max(abs(command.values[name] - observation.joints[name]) / command.duration_s for name in command.values) > backend.profile.max_joint_velocity_rad_s:
                        return GoalResponse.REJECT
                    self.guard.accept(command, observation)
                except (ValueError, TypeError, KeyError):
                    return GoalResponse.REJECT
                self.busy = True
                return GoalResponse.ACCEPT

        def tick(self):
            with self.lock:
                if not self.active:
                    return
                try:
                    backend.step()
                except Exception as exc:
                    self.fault, self.active = True, False
                    self.get_logger().error('Simulation failed: ' + type(exc).__name__)

        def execute(self, handle):
            command = None
            status, detail = 'failed', 'Execution did not start'
            try:
                command = decode(handle.request.json, MotionCommand)
                with self.lock:
                    backend.begin(command)
                    started_sim = backend.observe().sim_time_s
                    self.active = True
                wall_deadline = monotonic() + command.duration_s * 5 + 2
                stable = 0
                while monotonic() < wall_deadline:
                    with self.lock:
                        obs = backend.observe()
                        if handle.is_cancel_requested:
                            status, detail = 'canceled', 'Cancellation received'
                            break
                        if self.fault:
                            status, detail = 'failed', 'Simulation fault'
                            break
                        if command.mode == 'joint_positions':
                            error = max(abs(obs.joints[k] - v) for k, v in command.values.items())
                            # Count distinct simulation steps, not polling iterations.
                            if obs.step_id != getattr(self, '_last_checked_step', -1):
                                stable = stable + 1 if error <= tolerance else 0
                                self._last_checked_step = obs.step_id
                            if stable >= stable_steps:
                                status, detail = 'succeeded', 'Observed joint target reached'
                                break
                        elif obs.sim_time_s - started_sim >= command.duration_s:
                            status, detail = 'succeeded', 'Velocity command duration completed; task success not assessed'
                            break
                        if obs.sim_time_s - started_sim >= command.duration_s:
                            status, detail = 'failed', 'Joint target not reached within command duration'
                            break
                    feedback = ExecuteMotion.Feedback()
                    feedback.sim_time_s, feedback.step_id, feedback.status = obs.sim_time_s, obs.step_id, 'executing'
                    handle.publish_feedback(feedback)
                    sleep(0.01)
                else:
                    status, detail = 'timeout', 'Simulation action wall deadline exceeded'
            except Exception as exc:
                self.get_logger().error('Execution failed: ' + type(exc).__name__)
                status, detail = 'failed', 'Execution exception'
            finally:
                with self.lock:
                    self.active = False
                    try:
                        backend.stop()
                    except Exception:
                        self.fault = True
                    self.busy = False
                    self.publish_state()
            if status == 'succeeded':
                handle.succeed()
            elif status == 'canceled':
                handle.canceled()
            else:
                handle.abort()
            result = ExecuteMotion.Result()
            result.json = encode(ExecutionResult(command.command_id if command else 'invalid', status, detail))
            return result

        def destroy_node(self):
            with self.lock:
                self.active = False
                backend.stop()
            self.action.destroy()
            return super().destroy_node()

    return RobotNode()
