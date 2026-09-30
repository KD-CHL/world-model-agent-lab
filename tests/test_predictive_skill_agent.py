"""Task progress must come from feedback; trust controls committed prefixes."""
import unittest
import importlib.util
from pathlib import Path
import runpy
import numpy as np


class PredictiveAgentTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'),'Optional learning dependencies not installed')
    def test_a0_consumes_a_finite_nominal_plan_instead_of_replaying_initial_deltas(self):
        script=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/visual_skill_agent.py'))
        actions=script['fixed_joint_plan'](np.array([.35,.87]),np.array([.45,.95]),30)
        np.testing.assert_allclose(actions.sum(0),[.1,.08],atol=1e-8)
        np.testing.assert_allclose(actions[0],[.04,.04],atol=1e-8)
        np.testing.assert_allclose(actions[3:],np.zeros((27,2)),atol=1e-8)

    def setup_components(self, baseline='A3', threshold=.2):
        from wmal.agents.predictive_skill_agent import PredictiveSkillAgent, AgentTask, SkillCandidate
        from wmal.models.horizon_calibration import HorizonCalibration
        class LinearPredictor:
            version='known_linear_v1'
            semantics={'action_schema':'test.delta.v1','action_order':['x'],
                       'state_order':['x'],'camera':'front','period_s':.2,'image_size':32}
            def predict(self,rgb,state,actions):
                mean=state+np.cumsum(actions,axis=0)
                return {'states':np.stack([mean-.005,mean+.005]),
                        'frames':np.zeros((2,len(actions),3,32,32)),
                        'event_probabilities':None,'model_version':self.version}
        calibration=HorizonCalibration('known_linear_v1','d',.2,(1.,5.,10.),(1.,),.02,
                                       tuple(str(i) for i in range(5)),LinearPredictor.semantics)
        task=AgentTask('reach', np.array([.3]), ('move',), tolerance=.03, max_cycles=6)
        candidate=SkillCandidate('move', np.array([[.1],[.1],[.1]]))
        agent=PredictiveSkillAgent(task,LinearPredictor(),calibration,baseline=baseline,
                                  error_budget=threshold,action_lower=[-.2],action_upper=[.2])
        return agent,candidate

    def test_trust_shortens_prefix_and_predicted_success_does_not_complete_task(self):
        from wmal.agents.predictive_skill_agent import VisualObservation
        agent,candidate=self.setup_components(threshold=.11)
        before=VisualObservation('episode',0,np.zeros((3,32,32)),np.array([0.]))
        decision=agent.plan(before,[candidate])
        self.assertEqual(decision.prefix_length,2)
        self.assertEqual(agent.state.completed_subgoals,[])
        after=VisualObservation('episode',2,np.zeros((3,32,32)),np.array([.2]))
        agent.record_feedback(decision,after)
        self.assertEqual(agent.state.status,'running')
        self.assertAlmostEqual(agent.state.recent_feedback['state_error'],0.)
        next_decision=agent.plan(after,[candidate])
        agent.record_feedback(next_decision,VisualObservation('episode',2+next_decision.prefix_length,
                              np.zeros((3,32,32)),np.array([.3])))
        self.assertEqual(agent.state.status,'succeeded')
        self.assertEqual(agent.state.completed_subgoals,['reach'])

    def test_no_trusted_action_stale_feedback_and_unknown_skill_fail_closed(self):
        from wmal.agents.predictive_skill_agent import VisualObservation, SkillCandidate
        agent,candidate=self.setup_components(threshold=.001)
        before=VisualObservation('episode',0,np.zeros((3,32,32)),np.array([0.]))
        decision=agent.plan(before,[candidate])
        self.assertEqual(decision.prefix_length,0)
        self.assertEqual(decision.status,'reobserve')
        with self.assertRaises(ValueError):
            agent.plan(before,[SkillCandidate('unregistered',candidate.actions)])
        agent,candidate=self.setup_components()
        decision=agent.plan(before,[candidate])
        with self.assertRaises(ValueError):
            agent.record_feedback(decision,VisualObservation('other',1,before.rgb,before.state))
        with self.assertRaises(ValueError):
            agent.record_feedback(decision,before)
        with self.assertRaises(ValueError):
            agent.plan(before,[SkillCandidate('move',np.array([[.5],[.5],[.5]]))])

    def test_a2_fixed_commit_and_a0_does_not_call_predictor(self):
        from wmal.agents.predictive_skill_agent import VisualObservation
        before=VisualObservation('episode',0,np.zeros((3,32,32)),np.array([0.]))
        agent,candidate=self.setup_components('A2',threshold=.001)
        self.assertEqual(agent.plan(before,[candidate]).prefix_length,3)

    def test_partial_failure_accounts_only_actual_steps_and_never_replays(self):
        from wmal.agents.predictive_skill_agent import VisualObservation
        agent,candidate=self.setup_components()
        before=VisualObservation('episode',0,np.zeros((3,32,32)),np.array([0.]))
        decision=agent.plan(before,[candidate])
        actual=VisualObservation('episode',1,before.rgb,np.array([.1]))
        agent.record_execution_failure(decision,actual,'controller_failed')
        self.assertEqual(agent.state.executed_cycles,1)
        self.assertEqual(agent.state.status,'execution_failed')
        with self.assertRaises(ValueError):
            agent.plan(actual,[candidate])
        agent,candidate=self.setup_components('A0')
        class Forbidden:
            version=agent.predictor.version
            semantics=agent.predictor.semantics
            def predict(self,*args): raise AssertionError('A0 must not query world model')
        agent.predictor=Forbidden()
        self.assertEqual(agent.plan(before,[candidate]).prefix_length,3)


if __name__=='__main__':
    unittest.main()
