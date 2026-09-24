import sys
import types
import unittest


def make_model(config):
    return Model()


class Model:
    version = 'fixture-world-model-v1'
    from wmal.locomotion.contracts import G1_ACTION_SCHEMA, G1_STATE_SCHEMA
    state_schema = G1_STATE_SCHEMA
    action_schema = G1_ACTION_SCHEMA

    def predict(self, state, action, duration_s):
        from dataclasses import replace
        return replace(state, step_id=state.step_id + 1,
                       sim_time_s=state.sim_time_s + duration_s,
                       x=state.x + action.vx * duration_s)


class InvalidModel:
    version = 'invalid-v1'


class G1ContractsTests(unittest.TestCase):
    def test_state_rejects_nonfinite_pose(self):
        from wmal.locomotion.contracts import G1State

        with self.assertRaisesRegex(ValueError, 'finite'):
            G1State('episode-1', 0, 0.0, float('nan'), 0.0, 0.78, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78)

    def test_velocity_action_rejects_unsupported_speed(self):
        from wmal.locomotion.contracts import G1VelocityAction

        with self.assertRaisesRegex(ValueError, 'limit'):
            G1VelocityAction(1.1, 0.0, 0.0, 0.5)
        with self.assertRaisesRegex(ValueError, 'limit'):
            G1VelocityAction(0.4, 0.4, 0.0, 0.5)

    def test_factory_loads_and_prediction_preserves_version_and_state_schema(self):
        from wmal.locomotion.contracts import G1State, G1VelocityAction
        from wmal.locomotion.world_model import load_world_model

        module = types.ModuleType('wmal_test_world_model_factory')
        module.make_model = make_model
        sys.modules[module.__name__] = module
        try:
            model = load_world_model(module.__name__ + ':make_model', {'checkpoint': 'fixture.pt'})
        finally:
            del sys.modules[module.__name__]
        state = G1State('episode-1', 2, 0.04, 1.0, 0.0, 0.78, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78)
        prediction = model.predict(state, G1VelocityAction(0.2, 0.0, 0.0, 0.5), 0.5)
        self.assertEqual(model.version, 'fixture-world-model-v1')
        self.assertAlmostEqual(prediction.state.x, 1.1)
        self.assertEqual(prediction.state.episode_id, state.episode_id)
        with self.assertRaisesRegex(ValueError, 'duration'):
            model.predict(state, G1VelocityAction(0.2, 0.0, 0.0, 0.5), 0.2)

    def test_factory_rejects_model_without_predict_method(self):
        from wmal.locomotion.world_model import WorldModelAdapter

        with self.assertRaisesRegex(ValueError, 'predict'):
            WorldModelAdapter(InvalidModel())

    def test_factory_rejects_semantically_incompatible_action_schema(self):
        from wmal.locomotion.world_model import WorldModelAdapter

        class WrongSchema:
            version = 'wrong-schema'
            state_schema = Model.state_schema
            action_schema = 'joint_positions.v1'
            def predict(self, *_):
                raise AssertionError('incompatible model must be rejected at startup')

        with self.assertRaisesRegex(ValueError, 'action_schema'):
            WorldModelAdapter(WrongSchema())

    def test_external_model_errors_are_sanitized_before_cli_or_logs_see_them(self):
        from wmal.locomotion.contracts import G1State, G1VelocityAction
        from wmal.locomotion.world_model import WorldModelAdapter

        class BrokenModel(Model):
            def predict(self, *_):
                raise RuntimeError('provider response contains api-token-secret')

        state = G1State('ep', 0, 0.0, 0, 0, 0.78, 0, 0, 0, 0, 0, 0, 0, 0.78)
        with self.assertRaisesRegex(RuntimeError, 'inference failed') as error:
            WorldModelAdapter(BrokenModel()).predict(
                state, G1VelocityAction(0, 0, 0, 0.1))
        self.assertNotIn('api-token-secret', str(error.exception))

    def test_prediction_rejects_wrong_episode_and_nonfinite_values(self):
        from wmal.locomotion.contracts import G1Prediction, G1State, G1VelocityAction
        from wmal.locomotion.world_model import WorldModelAdapter

        state = G1State('episode-1', 2, 0.04, 1.0, 0.0, 0.78, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78)
        stale = G1State('episode-2', 3, 0.06, 1.0, 0.0, 0.78, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78)
        class StalePredictor:
            version = 'stale-v1'
            state_schema = Model.state_schema
            action_schema = Model.action_schema
            def predict(self, *_):
                return stale

        with self.assertRaisesRegex(ValueError, 'episode'):
            WorldModelAdapter(StalePredictor()).predict(
                state, G1VelocityAction(0.1, 0.0, 0.0, 0.5))
        with self.assertRaisesRegex(ValueError, 'finite'):
            G1Prediction(state, {'x': float('nan')})
        with self.assertRaisesRegex(ValueError, 'finite'):
            G1State('episode-1', 2, 0.04, float('inf'), 0.0, 0.78, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78)

    def test_prediction_rejects_changed_joint_state_schema(self):
        from dataclasses import replace
        from wmal.locomotion.contracts import G1State, G1VelocityAction
        from wmal.locomotion.world_model import WorldModelAdapter

        class DroppedJointModel(Model):
            def predict(self, state, action, duration_s):
                return replace(state, step_id=state.step_id + 1,
                               sim_time_s=state.sim_time_s + duration_s,
                               joint_positions={})

        state = G1State('ep', 1, 0.02, 0, 0, 0.78, 0, 0, 0, 0, 0, 0, 0, 0.78,
                        {'left_hip': 0.0})
        with self.assertRaisesRegex(ValueError, 'schema'):
            WorldModelAdapter(DroppedJointModel()).predict(
                state, G1VelocityAction(0, 0, 0, 0.1))


class PredictivePlannerTests(unittest.TestCase):
    def test_planner_selects_action_using_model_predicted_goal_progress(self):
        from wmal.locomotion.contracts import G1Goal, G1State
        from wmal.locomotion.planner import G1RolloutPlanner

        state = G1State('ep', 0, 0.0, 0, 0, 0.78, 0, 0, 0, 0, 0, 0, 0, 0.78)
        planner = G1RolloutPlanner(Model(), samples=80, horizon=3, seed=5,
                                   action_duration_s=0.5)
        plan = planner.plan(state, G1Goal(1.0, 0.0))
        self.assertGreater(plan.action.vx, 0.0)
        self.assertLessEqual(plan.action.duration_s, 0.5)
        self.assertEqual(plan.predicted_state.step_id, state.step_id + 1)
        self.assertEqual(plan.model_version, 'fixture-world-model-v1')

    def test_agent_reobserves_and_allows_another_goal_on_same_session(self):
        from dataclasses import replace
        from wmal.locomotion.agent import G1Agent
        from wmal.locomotion.contracts import G1Goal, G1State
        from wmal.locomotion.planner import G1RolloutPlanner

        class Session:
            is_running = True
            def __init__(self):
                self.state = G1State('ep', 0, 0.0, 0, 0, 0.78, 0,
                                     0, 0, 0, 0, 0, 0, 0.78)
                self.observations = 0
                self.actions = []
            def observe(self):
                self.observations += 1
                return self.state
            def step(self, action, duration_s):
                self.actions.append(action)
                self.state = replace(self.state, step_id=self.state.step_id + 1,
                                     sim_time_s=self.state.sim_time_s + duration_s,
                                     x=self.state.x + action.vx * duration_s)
                return self.state

        session = Session()
        planner = G1RolloutPlanner(Model(), samples=80, horizon=3, seed=5,
                                   action_duration_s=0.5)
        agent = G1Agent(planner)
        first = agent.run_goal(G1Goal(0.18, 0), session, max_cycles=8)
        self.assertEqual(first.status, 'succeeded')
        self.assertGreater(len(session.actions), 0)
        self.assertEqual(session.observations, len(session.actions) + 1)
        prior_actions = len(session.actions)
        second = agent.run_goal(G1Goal(session.state.x, 0), session, max_cycles=2)
        self.assertEqual(second.status, 'succeeded')
        self.assertEqual(len(session.actions), prior_actions)
        self.assertTrue(session.is_running)

    def test_agent_stops_before_planning_when_posture_is_unsafe(self):
        from wmal.locomotion.agent import G1Agent
        from wmal.locomotion.contracts import G1Goal, G1State

        class Session:
            is_running = True
            def __init__(self):
                self.actions = 0
            def observe(self):
                return G1State('ep', 0, 0.0, 0, 0, 0.78, 0,
                               0, 0, 0, 0, 0.8, 0, 0.78)
            def step(self, *_):
                self.actions += 1

        class Planner:
            def plan(self, *_):
                raise AssertionError('unsafe state must not be planned')

        session = Session()
        result = G1Agent(Planner()).run_goal(G1Goal(1, 0), session)
        self.assertEqual(result.status, 'safety_stop')
        self.assertEqual(session.actions, 0)

    def test_planner_does_not_select_a_sequence_predicted_to_fall(self):
        from dataclasses import replace
        from wmal.locomotion.contracts import G1Goal, G1State
        from wmal.locomotion.planner import G1RolloutPlanner

        class FallingModel(Model):
            def predict(self, state, action, duration_s):
                state = super().predict(state, action, duration_s)
                return replace(state, roll=0.8)

        state = G1State('ep', 0, 0.0, 0, 0, 0.78, 0, 0, 0, 0, 0, 0, 0, 0.78)
        planner = G1RolloutPlanner(FallingModel(), samples=4, horizon=1,
                                   action_duration_s=0.5)
        with self.assertRaisesRegex(ValueError, 'safe'):
            planner.plan(state, G1Goal(1, 0))


if __name__ == '__main__':
    unittest.main()
