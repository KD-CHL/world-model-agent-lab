"""CLI for API-agent, ROS robot bridge and world-model planning service."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from wmal.communication.contracts import RobotProfile


def read_config(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Configuration must be an object')
    return value


def run_agent():
    parser = argparse.ArgumentParser(description='API Agent → world-model planner → ROS 2 robot')
    parser.add_argument('--config', required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--max-cycles', type=int, default=10)
    parser.add_argument('--log', default='runs/agent/events.jsonl')
    args = parser.parse_args()
    from wmal.agents.api_client import ApiModelClient
    from wmal.agents.runner import AgentRunner
    from wmal.communication.ros2_transport import Ros2Channel
    from wmal.logging.events import EventLog
    try:
        profile = RobotProfile(**read_config(args.config)['profile'])
        llm = ApiModelClient.from_env()
        with Ros2Channel(profile) as channel:
            result = AgentRunner(llm, channel, profile, max_cycles=args.max_cycles, log=EventLog(args.log)).run(args.instruction)
        print(json.dumps(asdict(result), ensure_ascii=False))
        return 0 if result.status == 'succeeded' else 1
    except (ImportError, ValueError, KeyError, OSError) as exc:
        parser.exit(2, 'Startup failed: ' + type(exc).__name__ + '; check configuration and ROS setup.\n')


def serve():
    parser = argparse.ArgumentParser(description='ROS 2 world planning or MuJoCo robot service')
    parser.add_argument('role', choices=['robot', 'planner'])
    parser.add_argument('--config', help='Robot config, required for robot role')
    parser.add_argument('--plugin', help='Planner factory module:function, required for planner role')
    args = parser.parse_args()
    if args.role == 'robot' and not args.config:
        parser.error('--config required')
    if args.role == 'planner' and not args.plugin:
        parser.error('--plugin required; a trained model is not bundled')
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from wmal.communication.plugins import load_factory
    from wmal.communication.ros2_nodes import create_robot_node, create_planner_node
    rclpy.init()
    node = None
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        if args.role == 'planner':
            node = create_planner_node(load_factory(args.plugin))
        else:
            from wmal.envs.mujoco_backend import MujocoBackend
            config_path = Path(args.config).resolve()
            config = read_config(config_path)
            profile = RobotProfile(**config['profile'])
            model = (config_path.parent / config['model_path']).resolve()
            locomotion = load_factory(config['locomotion_plugin']) if config.get('locomotion_plugin') else None
            backend = MujocoBackend(str(model), profile, config['actuator_map'], locomotion=locomotion)
            node = create_robot_node(backend)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(timeout_sec=5)
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
