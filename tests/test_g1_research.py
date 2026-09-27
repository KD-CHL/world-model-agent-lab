from dataclasses import replace
import tempfile
import unittest

from wmal.locomotion.contracts import G1State, G1VelocityAction, G1Goal
from wmal.locomotion.learned import LearnedG1Dynamics, load
from wmal.locomotion.feedback import ResidualFeedback
from wmal.locomotion.planner import G1RolloutPlanner


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.state = G1State('episode', 0, 0., 0., 0., .8, 0., 0., 0., 0., 0., 0., 0., .8)
        self.rows = []
        for i in range(60):
            action = G1VelocityAction((i % 10 - 4) * .04, (i % 3 - 1) * .05, 0., .5)
            after = replace(self.state, step_id=1, sim_time_s=.5,
                            x=action.vx * .4, y=action.vy * .4)
            self.rows.append((self.state, action, after))

    def test_learned_prediction_roundtrip_and_planning(self):
        model = LearnedG1Dynamics.fit(self.rows)
        action = G1VelocityAction(.2, 0., 0., .5)
        prediction = model.predict(self.state, action, .5)
        self.assertAlmostEqual(prediction.state.x, .08, delta=.01)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory + '/model.json')
            restored = load({'checkpoint': directory + '/model.json'})
            self.assertEqual(restored.version, model.version)
            self.assertEqual(restored.predict(self.state, action, .5), prediction)
        plan = G1RolloutPlanner(model).plan(self.state, G1Goal(1., 0.))
        self.assertGreater(plan.action.vx, 0.)
        with self.assertRaises(ValueError):
            model.predict(self.state, action, .2)

    def test_feedback_changes_candidate_bounds_and_rejects_stale(self):
        model = LearnedG1Dynamics.fit(self.rows)
        planner = G1RolloutPlanner(model)
        feedback = ResidualFeedback(alpha=1.)
        predicted = self.rows[0][2]
        feedback.update(predicted, replace(predicted, x=predicted.x + 1), planner)
        self.assertEqual(planner.feedback_scale, .25)
        sequences = planner._sequences(self.state, G1Goal(1., 0.))
        self.assertLessEqual(abs(sequences[:, :, 0]).max(), .45 * .25)
        with self.assertRaises(ValueError):
            feedback.update(predicted, self.state, planner)
        feedback.update(predicted, predicted, planner)
        self.assertEqual(planner.feedback_scale, 1.)

    def test_adaptation_and_mixed_duration(self):
        prior = LearnedG1Dynamics.fit(self.rows)
        adapted = LearnedG1Dynamics.fit(self.rows, prior=prior)
        self.assertNotEqual(prior.version, adapted.version)
        rows = self.rows + [(self.state, G1VelocityAction(0., 0., 0., .2), self.rows[0][2])]
        with self.assertRaises(ValueError):
            LearnedG1Dynamics.fit(rows)
