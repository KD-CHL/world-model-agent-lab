"""Persistent interactive G1 MuJoCo Agent with an explicit world-model plugin."""
import argparse
import json
import math
from pathlib import Path

from wmal.locomotion.agent import G1Agent
from wmal.locomotion.contracts import G1_POLICY_PERIOD_S, G1Goal
from wmal.locomotion.planner import G1RolloutPlanner
from wmal.locomotion.world_model import load_world_model
from wmal.logging.events import EventLog


def validate_experiment_config(config):
    if not isinstance(config, dict):
        raise ValueError('Experiment configuration must be an object')
    world_model = config.get('world_model')
    planner = config.get('planner')
    simulator = config.get('simulator', {})
    agent = config.get('agent', {})
    if not isinstance(world_model, dict) or not isinstance(world_model.get('factory'), str):
        raise ValueError('world_model.factory must explicitly name a module:function adapter')
    if not isinstance(world_model.get('config', {}), dict):
        raise ValueError('world_model.config must be an object')
    if not isinstance(planner, dict):
        raise ValueError('planner configuration is required')
    if not isinstance(simulator, dict) or not isinstance(agent, dict):
        raise ValueError('simulator and agent configuration must be objects')
    samples = planner.get('samples', 48)
    horizon = planner.get('horizon', 3)
    duration = planner.get('action_duration_s', 0.5)
    if type(samples) is not int or samples < 4 or type(horizon) is not int or horizon < 1:
        raise ValueError('planner.samples must be >=4 and planner.horizon must be positive')
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not 0.05 <= duration <= 1):
        raise ValueError('planner.action_duration_s must be in [0.05, 1]')
    frames = round(duration / G1_POLICY_PERIOD_S)
    if frames < 3 or abs(frames * G1_POLICY_PERIOD_S - duration) > 1e-9:
        raise ValueError('planner.action_duration_s must be a G1 policy-period multiple')
    for name, default, upper in (('max_linear_velocity', 0.45, 0.5),
                                 ('max_yaw_rate', 0.7, 1.0)):
        value = planner.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= upper:
            raise ValueError(f'planner.{name} must be in (0, {upper}]')
    uncertainty_weight = planner.get('uncertainty_weight', 0.02)
    if (isinstance(uncertainty_weight, bool) or not isinstance(uncertainty_weight, (int, float))
            or not math.isfinite(uncertainty_weight) or uncertainty_weight < 0):
        raise ValueError('planner.uncertainty_weight must be finite and nonnegative')
    if type(planner.get('seed', 0)) is not int:
        raise ValueError('planner.seed must be an integer')
    for section, key in ((simulator, 'viewer'), (simulator, 'realtime')):
        if key in section and type(section[key]) is not bool:
            raise ValueError(f'simulator.{key} must be boolean')
    if 'max_cycles_per_goal' in agent:
        cycles = agent['max_cycles_per_goal']
        if type(cycles) is not int or cycles < 1:
            raise ValueError('agent.max_cycles_per_goal must be positive')
    for name, default in (('max_tilt_rad', 0.65), ('min_pelvis_height_m', 0.48)):
        value = agent.get(name, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f'agent.{name} must be finite and positive')
    if agent.get('max_tilt_rad', 0.65) >= 1.2:
        raise ValueError('agent.max_tilt_rad must be below 1.2 radians')
    expected_version = world_model.get('expected_version')
    if expected_version is not None and (not isinstance(expected_version, str) or not expected_version):
        raise ValueError('world_model.expected_version must be a nonempty string')
    return config


def parse_goal(text):
    if not isinstance(text, str):
        raise ValueError('Enter x y [yaw_degrees], or quit')
    if text.strip().lower() in ('q', 'quit', 'exit'):
        return None
    parts = text.split()
    if len(parts) not in (2, 3):
        raise ValueError('Enter world-frame x y [yaw_degrees], or quit')
    try:
        values = [float(value) for value in parts]
    except ValueError as exc:
        raise ValueError('Goal coordinates must be numbers') from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Goal coordinates must be finite')
    yaw = math.radians(values[2]) if len(values) == 3 else None
    return G1Goal(values[0], values[1], yaw)


def interactive_loop(agent, session, *, input_fn=input, output_fn=print,
                     max_cycles=100, log_path='runs/g1_agent/events.jsonl'):
    log = log_path if callable(log_path) else EventLog(log_path)
    output_fn('G1 world-model Agent ready. Enter world-frame x y [yaw_deg], or quit.')
    while session.is_running:
        try:
            raw = input_fn('goal> ')
        except (EOFError, KeyboardInterrupt):
            break
        try:
            goal = parse_goal(raw)
        except ValueError as exc:
            output_fn(f'Invalid goal: {exc}')
            continue
        if goal is None:
            break
        log('goal', {'x': goal.x, 'y': goal.y, 'yaw_rad': goal.yaw})
        result = agent.run_goal(goal, session, max_cycles=max_cycles)
        summary = {'status': result.status, 'cycles': result.cycles,
                   'detail': result.detail,
                   'final_state': result.final_state.__dict__ if result.final_state else None}
        log('goal_result', summary)
        output_fn(f"{result.status}: {result.detail} ({result.cycles} replans)")
        if result.status in ('safety_stop', 'stopped', 'failed'):
            break
    if not session.is_running:
        log('session_exit', {'reason': 'viewer_closed'})
        output_fn('MuJoCo viewer is closed; ending the experiment session.')
    output_fn('G1 simulation session ended.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Experiment JSON with world-model factory')
    parser.add_argument('--log', help='Override JSONL experiment log path')
    args = parser.parse_args(argv)
    try:
        config = validate_experiment_config(json.loads(Path(args.config).read_text(encoding='utf-8')))
        world_model_config = config['world_model']
        model = load_world_model(world_model_config['factory'], world_model_config.get('config', {}))
        expected_version = world_model_config.get('expected_version')
        if expected_version and model.version != expected_version:
            raise ValueError('Loaded world-model version does not match expected_version')
        settings = config['planner']
        planner = G1RolloutPlanner(
            model, samples=settings.get('samples', 48), horizon=settings.get('horizon', 3),
            seed=settings.get('seed', 0), action_duration_s=settings.get('action_duration_s', 0.5),
            max_linear_velocity=settings.get('max_linear_velocity', 0.45),
            max_yaw_rate=settings.get('max_yaw_rate', 0.7),
            uncertainty_weight=settings.get('uncertainty_weight', 0.02))
        agent_config = config.get('agent', {})
        agent = G1Agent(planner, max_tilt_rad=agent_config.get('max_tilt_rad', 0.65),
                        min_pelvis_height_m=agent_config.get('min_pelvis_height_m', 0.48),
                        log=EventLog(args.log or config.get('log_path', 'runs/g1_agent/events.jsonl')))
        from wmal.envs.g1_session import G1MuJoCoSession
        simulator = config.get('simulator', {})
        with G1MuJoCoSession(viewer=simulator.get('viewer', True),
                             realtime=simulator.get('realtime', True)) as session:
            interactive_loop(agent, session,
                             max_cycles=agent_config.get('max_cycles_per_goal', 100),
                             log_path=agent.log)
        return 0
    except (ImportError, OSError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'G1 Agent startup/session failed: {type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
