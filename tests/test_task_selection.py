"""Shared selection retains calibrated-prefix semantics without an executor."""
import unittest
import numpy as np
from wmal.agents.predictive_skill_agent import SkillCandidate, VisualObservation


class Predictor:
    version = 'v1'
    semantics = {'action_order':['x'], 'state_order':['x']}
    def predict(self, rgb, state, actions):
        values = state + np.cumsum(actions,axis=0)
        return {'model_version':self.version, 'states':np.stack([values,values]),
                'frames':np.broadcast_to(rgb,(2,len(actions),*rgb.shape)).copy()}


class Calibration:
    def validate_for(self, version, semantics):
        if version != 'v1':
            raise ValueError('wrong version')
    def bounds(self, std, *, version):
        return np.array([[.01],[.4],[.01]])[:len(std)]


class SelectionTests(unittest.TestCase):
    def choose(self, baseline, predictor=None):
        from wmal.agents.selection_policy import select_candidates
        obs = VisualObservation('e',0,np.zeros((3,32,32)),np.array([0.]))
        candidate = SkillCandidate('move',np.array([[.1],[.1],[.1]]),np.array([.3]))
        return select_candidates(predictor or Predictor(),obs,[candidate],np.array([.3]),
                  baseline=baseline, calibration=Calibration(),error_budget=.2,scale=np.ones(1),
                  remaining=3,state_lower=None,state_upper=None)

    def test_a1_does_not_query_learned_predictor(self):
        class Forbidden(Predictor):
            def predict(self,*args):
                raise AssertionError('A1 queried learned dynamics')
        ranked, evidence = self.choose('A1',Forbidden())
        self.assertEqual(ranked[0][4],3)
        self.assertEqual(evidence[0]['score'],0.)

    def test_a2_full_prefix_a3_cannot_jump_untrusted_step(self):
        self.assertEqual(self.choose('A2')[0][0][4],3)
        self.assertEqual(self.choose('A3')[0][0][4],1)

    def test_wrong_response_version_rejected(self):
        class Stale(Predictor):
            def predict(self,*args):
                result = super().predict(*args)
                result['model_version']='old'
                return result
        with self.assertRaises(ValueError):
            self.choose('A2',Stale())
