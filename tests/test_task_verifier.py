"""A held position is a sequence of real control samples, not repeated reads."""
import unittest
import numpy as np
from wmal.agents.predictive_skill_agent import VisualObservation


def observation(step, q=(.43,.95), command=(.43,.95), episode='e'):
    return VisualObservation(episode, step, np.zeros((3,32,32)), np.array([*q,*command]))


class JointVerifierTests(unittest.TestCase):
    def test_hold_only_advances_on_real_contiguous_execution(self):
        from wmal.skills.task_verifier import JointVerifier
        verifier = JointVerifier((.43,.95), required_steps=3)
        self.assertFalse(verifier.update(observation(0), executed=False))
        self.assertFalse(verifier.update(observation(0), executed=False))
        self.assertFalse(verifier.update(observation(1), executed=True))
        self.assertFalse(verifier.update(observation(2), executed=True))
        self.assertTrue(verifier.update(observation(3), executed=True))
        self.assertEqual(verifier.count, 3)
        with self.assertRaises(ValueError):
            verifier.update(observation(3), executed=True)

    def test_mismatch_resets_hold_and_cross_episode_is_rejected(self):
        from wmal.skills.task_verifier import JointVerifier
        verifier = JointVerifier((.43,.95),3)
        verifier.update(observation(0))
        verifier.update(observation(1),executed=True)
        self.assertFalse(verifier.update(observation(2,q=(.5,.95)),executed=True))
        self.assertEqual(verifier.count, 0)
        with self.assertRaises(ValueError):
            verifier.update(observation(3,episode='other'),executed=True)

    def test_reach_requires_actual_and_command_targets(self):
        from wmal.skills.task_verifier import JointVerifier
        verifier = JointVerifier((.43,.95))
        self.assertFalse(verifier.update(observation(0,command=(.35,.87))))
        self.assertFalse(verifier.update(observation(0,q=(.35,.87))))
        self.assertTrue(verifier.update(observation(0)))
