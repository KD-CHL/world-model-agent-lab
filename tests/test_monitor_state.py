import unittest
from wmal.monitor.state import summarize, sanitize


class MonitorStateTests(unittest.TestCase):
    def test_session_telemetry_does_not_invent_a_new_task(self):
        rows = [{'event': 'task_result', 'payload': {'status': 'succeeded'}},
                {'event': 'session_exit', 'payload': {}}]
        self.assertEqual(len(summarize(rows)['tasks']), 1)

    def test_missing_final_event_remains_incomplete_and_preserves_units(self):
        rows = [
            {'event': 'plan', 'payload': {'task_id': 'one', 'planning_latency_s': .025}},
            {'event': 'prediction_residual', 'payload': {'task_id': 'one', 'position_error_m': .05}},
        ]
        summary = summarize(rows)
        self.assertEqual(summary['tasks'][0]['status'], 'incomplete')
        self.assertEqual(summary['planning_latency_ms']['mean'], 25)
        self.assertEqual(summary['prediction_errors']['position_error_m']['mean'], .05)
        self.assertIsNone(summary['prediction_errors']['rmse_rad']['mean'])

    def test_prompt_redaction_preserves_agent_evidence(self):
        result = sanitize({'task_goal': [1, 2], 'prediction': {'uncertainty_kind': 'ensemble_spread'},
                           'nested': {'api_key': 'secret', 'system_prompt': 'private'}})
        self.assertEqual(result['task_goal'], [1, 2])
        self.assertEqual(result['prediction']['uncertainty_kind'], 'ensemble_spread')
        self.assertNotIn('secret', str(result))
        self.assertNotIn('private', str(result))
