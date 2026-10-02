"""Actual history cannot be replaced by predicted frames or skipped-step feedback."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_temporal_world import model
from wmal.agents.predictive_skill_agent import AgentTask, PredictiveSkillAgent, SkillCandidate, VisualObservation


def observation(step, state=(0., 0.), episode='real'):
    return VisualObservation(episode, step, np.zeros((3, 32, 32)), np.array(state))


class AgentTemporalTests(unittest.TestCase):
    def agent(self):
        return PredictiveSkillAgent(AgentTask('task', np.array([10., 10.]), ('move',), max_cycles=20),
                                     model(), baseline='A2', action_lower=[-.2, -.2], action_upper=[.2, .2])

    def candidate(self, length=2):
        return SkillCandidate('move', np.ones((length, 2)) * .1)

    def test_real_feedback_builds_bounded_belief_and_candidates_do_not_advance_it(self):
        agent = self.agent()
        before = observation(0)
        decision = agent.plan(before, [self.candidate(), self.candidate()])
        self.assertEqual(agent.state.belief['context_steps_used'], 0)
        trace = [observation(1, (.1, .1)), observation(2, (.2, .2))]
        agent.record_feedback(decision, trace[-1], observations=trace)
        self.assertEqual(agent.state.belief['context_steps_used'], 2)
        self.assertEqual(agent.state.belief['history_step_ids'], [0, 1, 2])
        self.assertTrue(agent.state.belief['context_valid'])
        decision = agent.plan(trace[-1], [self.candidate(1)])
        rgb = np.stack([before.rgb, trace[0].rgb, trace[1].rgb])
        states = np.array([[0., 0.], [.1, .1], [.2, .2]])
        expected = agent.predictor.predict_context(rgb, states, np.ones((2, 2)) * .1, np.ones((1, 2)) * .1)
        np.testing.assert_allclose(decision.predicted_states, expected['states'].mean(0))
        agent.record_feedback(decision, observation(3, (.3, .3)))
        self.assertEqual(agent.state.belief['history_step_ids'], [1, 2, 3])
        self.assertEqual(agent.state.executed_cycles, 3)

    def test_partial_failure_marks_context_unusable_and_never_retries(self):
        agent = self.agent()
        decision = agent.plan(observation(0), [self.candidate()])
        agent.record_execution_failure(decision, observation(1, (.1, .1)), 'controller_failure')
        self.assertFalse(agent.state.belief['context_valid'])
        self.assertEqual(agent.state.belief['history_step_ids'], [0])
        self.assertEqual(agent.state.executed_cycles, 1)
        with self.assertRaises(ValueError):
            agent.plan(observation(1), [self.candidate()])

    def test_intermediate_mismatch_triggers_reobserve_even_when_final_prediction_matches(self):
        from wmal.models.horizon_calibration import fit_horizon_calibration
        predictor = model()
        calibration = fit_horizon_calibration(
            [dict(episode_id=f'cal_{i}', split='calibration', scores=[.1, .1]) for i in range(5)],
            model_version=predictor.version, dataset_id='independent', alpha=.2,
            state_scale=[1., 1.], std_floor=.05, semantics=predictor.semantics)
        agent = PredictiveSkillAgent(AgentTask('task', np.array([10., 10.]), ('move',)), predictor,
                                     calibration, baseline='A3', error_budget=10.,
                                     action_lower=[-.2, -.2], action_upper=[.2, .2])
        decision = agent.plan(observation(0), [self.candidate()])
        trace = [observation(1, decision.predicted_states[0]+1.),
                 observation(2, decision.predicted_states[1])]
        feedback = agent.record_feedback(decision, trace[-1], observations=trace)
        self.assertEqual(feedback['outside_calibration_by_step'], [True, False])
        self.assertTrue(feedback['outside_calibration'])
        self.assertEqual(feedback['first_mismatch_step_id'], 1)
        next_decision = agent.plan(trace[-1], [self.candidate(1)])
        self.assertEqual(next_decision.status, 'reobserve')
        self.assertEqual(next_decision.evidence['reason'], 'prediction_mismatch_reobserve')

    def test_trace_sink_failure_latches_mujoco_at_last_executed_step(self):
        from wmal.envs.visual_workcell import VisualWorkcellSession
        with VisualWorkcellSession('stack_block', image_size=32) as session:
            before = session.observe()
            def reject_receipt(value):
                raise RuntimeError('trace recorder unavailable')
            with self.assertRaises(RuntimeError):
                session.execute_actions(np.array([[.01, .01], [.01, .01]]),
                    episode_id=before.episode_id, step_id=before.step_id, on_observation=reject_receipt)
            self.assertEqual(session.step_id, 1)
            self.assertEqual(session.fault_reason, 'execution_failed')
            stopped = session.data.time
            session.idle()
            self.assertEqual(session.data.time, stopped)

    def test_missing_malformed_trace_is_rejected_atomically_and_valid_receipt_can_follow(self):
        for bad in (None, [observation(2)], [observation(1), observation(1)],
                    [observation(1, episode='other'), observation(2)],
                    [observation(1), observation(2, (9., 9.))]):
            with self.subTest(trace_steps=None if bad is None else [v.step_id for v in bad]):
                agent = self.agent()
                decision = agent.plan(observation(0), [self.candidate()])
                final = observation(2, (.2, .2))
                with self.assertRaises(ValueError):
                    agent.record_feedback(decision, final, observations=bad)
                self.assertEqual(agent.state.executed_cycles, 0)
                self.assertEqual(agent.state.belief['history_step_ids'], [0])
                agent.record_feedback(decision, final, observations=[observation(1, (.1, .1)), final])
                self.assertEqual(agent.state.executed_cycles, 2)

    def test_trace_snapshots_cannot_be_changed_by_public_observation_headers(self):
        agent = self.agent()
        first = observation(0)
        decision = agent.plan(first, [self.candidate(1)])
        first.state.shape = (1, 2)
        final = observation(1, (.1, .1))
        agent.record_feedback(decision, final)
        final.rgb.shape = (3*32*32,)
        decision = agent.plan(observation(1, (.1, .1)), [self.candidate(1)])
        self.assertEqual(decision.prefix_length, 1)

    def test_real_mujoco_execution_reports_every_real_step(self):
        from wmal.envs.visual_workcell import VisualWorkcellSession
        with tempfile.TemporaryDirectory() as tmp:
            with VisualWorkcellSession('stack_block', image_size=32) as session:
                before = session.observe()
                trace = []
                final = session.execute_actions(np.array([[.01, .01], [.01, -.01]]),
                            episode_id=before.episode_id, step_id=before.step_id, on_observation=trace.append)
                self.assertEqual([v.step_id for v in trace], [before.step_id+1, before.step_id+2])
                self.assertEqual(final.step_id, trace[-1].step_id)
                np.testing.assert_array_equal(final.state, trace[-1].state)
                self.assertGreater(np.abs(trace[0].state-before.state).max(), .001)
