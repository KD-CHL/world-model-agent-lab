"""Exact task commands constrain, but never replace, learned consequences."""
import unittest
from dataclasses import replace
import numpy as np

from wmal.agents.predictive_skill_agent import SkillCandidate, VisualObservation
from wmal.agents.selection_policy import select_candidates
from wmal.agents.task_graph import TaskNode
from wmal.envs.visual_workcell import TARGET_LOW, TARGET_HIGH
from wmal.models.horizon_calibration import HorizonCalibration
from wmal.skills.research_contracts import JointSkills


class Consequences:
    """Controlled ensemble forecasts at the predictor's public boundary."""
    version = 'joint-planning-fixture'
    semantics = {'action_order': ['s', 'e'], 'state_order': ['q1', 'q2', 'c1', 'c2']}

    def __init__(self, forecasts):
        self.forecasts = iter(forecasts)

    def predict(self, rgb, state, actions):
        values = np.asarray(next(self.forecasts), dtype=float)
        if values.shape != (len(actions), 4):
            raise AssertionError('Fixture forecast does not match the requested horizon')
        return {'model_version': self.version, 'states': np.stack([values, values]),
                'frames': np.broadcast_to(rgb, (2, len(actions), *rgb.shape)).copy()}


def calibration(quantiles=(1., 1., 1., 1.)):
    return HorizonCalibration(Consequences.version, 'held-out', .2, quantiles,
                              (1., 1., 1., 1.), .01, ('cal1', 'cal2'), Consequences.semantics)


def observation(command=(.43, .95)):
    return VisualObservation('episode', 0, np.zeros((3, 32, 32)),
                             np.array([.38, .99, *command]))


def choose(obs, candidates, forecasts, *, goal=(.43, .95), baseline='A3',
           quantiles=(1., 1., 1., 1.), remaining=4, task=True):
    kwargs = {'joint_target': np.array(goal)} if task else {}
    return select_candidates(Consequences(forecasts), obs, candidates, np.tile(goal, 2),
                             baseline=baseline, calibration=calibration(quantiles),
                             error_budget=.2, scale=np.ones(4), remaining=remaining,
                             state_lower=np.concatenate([TARGET_LOW-.15, TARGET_LOW]),
                             state_upper=np.concatenate([TARGET_HIGH+.15, TARGET_HIGH]), **kwargs)


class JointCandidateTests(unittest.TestCase):
    # Repeating a signed fixed step instead of capping the remaining error breaks these.
    def test_near_goal_arrives_precisely_then_settles(self):
        candidates = JointSkills.candidates(observation((.425, .96)),
            TaskNode('near', 'joint_reach', [.43, .95]), 4, np.random.default_rng(4), goal_arrival=True)
        np.testing.assert_allclose(candidates[0].actions,
                                   [[.005, -.01], [0., 0.], [0., 0.], [0., 0.]], atol=1e-12)
        np.testing.assert_allclose(candidates[0].nominal_terminal, [.43, .95, .43, .95])

    def test_four_step_arrival_does_not_overshoot(self):
        candidates = JointSkills.candidates(observation((.35, .87)),
            TaskNode('reach', 'joint_reach', [.43, .95]), 4, np.random.default_rng(4), goal_arrival=True)
        np.testing.assert_allclose(candidates[0].actions,
                                   [[.04, .04], [.04, .04], [0., 0.], [0., 0.]], atol=1e-12)
        np.testing.assert_allclose(candidates[1].actions, np.full((4, 2), .02), atol=1e-12)

    def test_arrival_can_reach_an_envelope_boundary_exactly(self):
        candidates = JointSkills.candidates(observation((.06, 1.14)),
            TaskNode('edge', 'joint_reach', [.05, 1.15]), 4, np.random.default_rng(4), goal_arrival=True)
        np.testing.assert_allclose(candidates[0].actions,
                                   [[-.01, .01], [0., 0.], [0., 0.], [0., 0.]], atol=1e-12)

    def test_default_controller_retains_legacy_proposals(self):
        candidates = JointSkills.candidates(observation((.35, .87)),
            TaskNode('legacy', 'joint_reach', [.43, .95]), 4, np.random.default_rng(4))
        np.testing.assert_allclose(candidates[0].actions, np.full((4, 2), .04))
        self.assertEqual(len(candidates), 12)


class TaskJointRankingTests(unittest.TestCase):
    def test_outside_calibration_scope_rejects_before_model_query(self):
        class NoQuery(Consequences):
            def predict(self,*args):
                raise AssertionError('Out-of-domain prediction is not calibrated')
        bound=replace(calibration(),prediction_scope={'command_lower':[.21,.73],
            'command_upper':[.49,1.01],'measured_margin':.15,'max_action_delta':.06})
        ranked,evidence=select_candidates(NoQuery([]),observation((.48,.95)),
            [SkillCandidate('joint_reach',[[.04,0.]])],np.tile([.43,.95],2),
            baseline='A3',calibration=bound,error_budget=1.,scale=np.ones(4),remaining=1,
            state_lower=None,state_upper=None,joint_target=np.array([.43,.95]))
        self.assertEqual(ranked,[])
        self.assertEqual(evidence[0]['reason'],'outside_calibration_scope')

    # Trusting the model's command head would choose motion away from an attained command.
    def test_command_at_goal_remains_stable_with_misleading_model_head(self):
        candidates = [SkillCandidate('joint_reach', [[.02, .02]]),
                      SkillCandidate('joint_reach', [[0., 0.]])]
        hold_forecast = [[.44, .96, .47, .99]]
        ranked, evidence = choose(observation(), candidates,
                                 [[[.43, .95, .43, .95]], hold_forecast])
        self.assertEqual(min(ranked, key=lambda r: (r[0], r[1]))[1], 1)
        self.assertFalse(evidence[0]['command_progress_eligible'])
        self.assertEqual(evidence[0]['learned_trusted_prefix'], 1)
        self.assertTrue(evidence[1]['command_progress_eligible'])
        np.testing.assert_allclose(evidence[1]['command_trajectory'], [[.43, .95]])
        np.testing.assert_allclose(ranked[0][5], hold_forecast)
        np.testing.assert_allclose(ranked[0][7], np.full((1, 4), .01))
        np.testing.assert_array_equal(ranked[0][6], np.zeros((1, 3, 32, 32)))

    # Favourable measured predictions must not make a stalled command beat available progress.
    def test_progress_eligibility_uses_exact_commands(self):
        candidates = [SkillCandidate('joint_reach', [[0., 0.]]),
                      SkillCandidate('joint_reach', [[.04, .04]])]
        ranked, evidence = choose(observation((.35, .87)), candidates,
            [[[.43, .95, .43, .95]], [[.38, .99, .35, .87]]])
        self.assertEqual([row[1] for row in ranked], [1])
        self.assertFalse(evidence[0]['command_progress_eligible'])
        self.assertTrue(evidence[1]['command_progress_eligible'])
        np.testing.assert_allclose(evidence[1]['command_trajectory'], [[.39, .91]])
        self.assertGreater(evidence[1]['command_progress'], 0.)

    # Removing learned consequence scoring breaks selection between equal command arrivals.
    def test_learned_measured_outcomes_rank_equal_command_arrivals(self):
        candidates = [SkillCandidate('joint_reach', [[.04, .04], [.04, .04]]),
                      SkillCandidate('joint_reach', [[.03, .03], [.05, .05]])]
        for good in (0, 1):
            with self.subTest(good=good):
                forecasts = [np.tile([.49, 1.01, .43, .95], (2, 1)),
                             np.tile([.49, 1.01, .43, .95], (2, 1))]
                forecasts[good] = np.tile([.43, .95, .53, 1.05], (2, 1))
                ranked, evidence = choose(observation((.35, .87)), candidates, forecasts)
                self.assertEqual(min(ranked, key=lambda r: (r[0], r[1]))[1], good)
                self.assertTrue(all(row.get('command_progress_eligible', False) for row in evidence))

    # Exact commands do not permit any action when learned physical predictions are untrusted.
    def test_exact_arrival_cannot_bypass_learned_trust(self):
        candidate = SkillCandidate('joint_reach', [[.005, -.01]])
        ranked, evidence = choose(observation((.425, .96)), [candidate],
                                   [[[.43, .95, .43, .95]]], quantiles=(30.,))
        self.assertEqual(ranked, [])
        self.assertEqual(evidence[0].get('learned_trusted_prefix'), 0)
        self.assertFalse(evidence[0].get('command_progress_eligible', True))

    def test_untrusted_second_step_cannot_supply_command_progress(self):
        candidates = [SkillCandidate('joint_reach', [[.02, .02], [.06, .06]]),
                      SkillCandidate('joint_reach', [[.04, .04], [.04, .04]])]
        forecasts = [np.tile([.43, .95, .43, .95], (2, 1))] * 2
        ranked, evidence = choose(observation((.35, .87)), candidates, forecasts,
                                   quantiles=(1., 30.))
        self.assertEqual([row[1] for row in ranked], [1])
        self.assertEqual(ranked[0][4], 1)
        self.assertFalse(evidence[0]['command_progress_eligible'])
        np.testing.assert_allclose(evidence[1]['command_terminal_error'], np.sqrt(.0032))

    def test_command_feasibility_stops_before_overshoot(self):
        candidate = SkillCandidate('joint_reach', np.full((4, 2), .04))
        ranked, evidence = choose(observation((.35, .87)), [candidate],
                                   [np.tile([.43, .95, .43, .95], (4, 1))])
        self.assertEqual(ranked[0][4], 2)
        self.assertEqual(evidence[0]['learned_trusted_prefix'], 4)
        self.assertEqual(evidence[0]['command_feasible_prefix'], 2)

    def test_a1_uses_budget_sliced_nominal_endpoint(self):
        candidates = [SkillCandidate('joint_reach', [[.02, .02], [.06, .06]], np.tile([.43, .95], 2)),
                      SkillCandidate('joint_reach', [[.04, .04], [.04, .04]], np.tile([.43, .95], 2))]
        ranked, _ = choose(observation((.35, .87)), candidates, [], baseline='A1', remaining=1)
        self.assertEqual([row[1] for row in ranked], [1])
        self.assertAlmostEqual(ranked[0][0], .08)
        self.assertIsNone(ranked[0][5])

    # The opt-in must not change legacy scoring or evidence for other selector callers.
    def test_default_legacy_ranking_still_uses_model_command_head(self):
        candidates = [SkillCandidate('joint_reach', [[.02, .02]]),
                      SkillCandidate('joint_reach', [[0., 0.]])]
        ranked, evidence = choose(observation(), candidates,
            [[[.43, .95, .43, .95]], [[.43, .95, .47, .99]]], task=False)
        self.assertEqual(min(ranked, key=lambda r: (r[0], r[1]))[1], 0)
        self.assertNotIn('command_progress_eligible', evidence[0])


if __name__ == '__main__':
    unittest.main()
