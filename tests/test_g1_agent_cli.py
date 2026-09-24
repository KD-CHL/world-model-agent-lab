import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path


class G1AgentCliTests(unittest.TestCase):
    def test_goal_parser_accepts_world_xy_and_optional_heading(self):
        from scripts.g1_agent_sim import parse_goal

        goal = parse_goal('1.5 -0.25 90')
        self.assertEqual((goal.x, goal.y), (1.5, -0.25))
        self.assertAlmostEqual(goal.yaw, 1.57079632679)
        self.assertIsNone(parse_goal('quit'))

    def test_config_requires_explicit_model_factory_and_planner_limits(self):
        from scripts.g1_agent_sim import validate_experiment_config

        with self.assertRaisesRegex(ValueError, 'factory'):
            validate_experiment_config({'world_model': {}, 'planner': {}})
        config = validate_experiment_config({
            'world_model': {'factory': 'my_model:load', 'config': {'checkpoint': '/tmp/model.pt'}},
            'planner': {'samples': 12, 'horizon': 2, 'action_duration_s': 0.5},
            'simulator': {'realtime': False}})
        self.assertEqual(config['world_model']['factory'], 'my_model:load')
        with self.assertRaisesRegex(ValueError, 'max_linear_velocity'):
            validate_experiment_config({
                'world_model': {'factory': 'my_model:load'},
                'planner': {'max_linear_velocity': 2.0}})

    def test_missing_world_model_plugin_fails_before_opening_simulator(self):
        from scripts.g1_agent_sim import main

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / 'experiment.json'
            config_path.write_text(json.dumps({
                'world_model': {'factory': 'module_that_does_not_exist:load'},
                'planner': {'samples': 4, 'horizon': 1}}))
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                main(['--config', str(config_path)])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('ModuleNotFoundError', error.getvalue())

    def test_interactive_loop_keeps_session_for_multiple_goals_and_logs_results(self):
        from scripts.g1_agent_sim import interactive_loop

        class Session:
            is_running = True

        class Result:
            status = 'succeeded'
            cycles = 1
            detail = 'observed'
            final_state = None

        class Agent:
            def __init__(self):
                self.sessions = []
                self.goals = []
            def run_goal(self, goal, session, max_cycles):
                self.sessions.append(session)
                self.goals.append(goal)
                return Result()

        agent, session = Agent(), Session()
        responses = iter(['1 0', '2 0 45', 'quit'])
        output = []
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / 'agent.jsonl'
            interactive_loop(agent, session, input_fn=lambda _: next(responses),
                             output_fn=output.append, max_cycles=3,
                             log_path=log_path)
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
        self.assertEqual(len(agent.goals), 2)
        self.assertTrue(all(candidate is session for candidate in agent.sessions))
        self.assertEqual([record['event'] for record in records],
                         ['goal', 'goal_result', 'goal', 'goal_result'])


if __name__ == '__main__':
    unittest.main()
