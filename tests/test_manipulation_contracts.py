import threading
import unittest

from wmal.envs.manipulation_session import LEFT_ARM_JOINTS, ManipulationSession
from wmal.skills.manipulation_contracts import (
    ManipulationSkillSpec, SkillRun, StackBlockSkills, WaypointProfile,
)


def profile(points=None):
    defaults = {
        'clearance': dict(zip(LEFT_ARM_JOINTS, (.35,.18,0.,.87,0.,0.,0.))),
        'cross_edge': dict(zip(LEFT_ARM_JOINTS, (.36,.18,0.,.87,0.,0.,0.))),
        'pregrasp': dict(zip(LEFT_ARM_JOINTS, (.37,.18,0.,.87,0.,0.,0.))),
        'grasp': dict(zip(LEFT_ARM_JOINTS, (.38,.18,0.,.87,0.,0.,0.))),
        'lift_1': dict(zip(LEFT_ARM_JOINTS, (.385,.18,0.,.87,0.,0.,0.))),
        'lift_2': dict(zip(LEFT_ARM_JOINTS, (.39,.18,0.,.87,0.,0.,0.))),
        'lift_3': dict(zip(LEFT_ARM_JOINTS, (.395,.18,0.,.87,0.,0.,0.))),
        'lift_4': dict(zip(LEFT_ARM_JOINTS, (.40,.18,0.,.87,0.,0.,0.))),
        'lift_5': dict(zip(LEFT_ARM_JOINTS, (.405,.18,0.,.87,0.,0.,0.))),
        'lift': dict(zip(LEFT_ARM_JOINTS, (.39,.18,0.,.87,0.,0.,0.))),
        'transport': dict(zip(LEFT_ARM_JOINTS, (.40,.18,0.,.87,0.,0.,0.))),
        'place': dict(zip(LEFT_ARM_JOINTS, (.41,.18,0.,.87,0.,0.,0.))),
    }
    if points:
        defaults.update(points)
    return WaypointProfile(
        profile_id='test-profile', joint_order=LEFT_ARM_JOINTS,
        points=defaults)


class ManipulationContractTests(unittest.TestCase):
    def setUp(self):
        self.session = ManipulationSession(render=False, control_period_s=.02)
        self.addCleanup(self.session.close)
        self.observation, _ = self.session.reset(0, 'nominal')
        self.profile = profile()
        self.skills = StackBlockSkills(self.profile)

    def test_waypoint_profile_rejects_duplicate_names_nonfinite_and_bad_dimensions(self):
        with self.assertRaises(ValueError):
            WaypointProfile('dup', ('left_elbow_joint', 'left_elbow_joint'),
                            {'grasp': (.1, .2)})
        with self.assertRaises(ValueError):
            WaypointProfile('nan', ('left_elbow_joint',), {'grasp': (float('nan'),)})
        with self.assertRaises(ValueError):
            WaypointProfile('shape', ('left_elbow_joint',), {'grasp': (.1, .2)})

    def test_waypoint_profile_checks_model_limits_and_requires_full_skill_set(self):
        bad = profile({'grasp': dict.fromkeys(LEFT_ARM_JOINTS, 99.0)})
        with self.assertRaises(ValueError):
            bad.validate_for(self.session)
        with self.assertRaises(ValueError):
            WaypointProfile('missing', LEFT_ARM_JOINTS, {'grasp': dict.fromkeys(LEFT_ARM_JOINTS, .1)})
        self.profile.validate_for(self.session)

    def test_unverified_retreat_is_not_exposed_as_a_skill(self):
        with self.assertRaises(ValueError):
            self.skills.command('retreat')
        self.assertFalse(any(name.startswith('retreat') for name in self.profile.points))

    def test_skill_spec_requires_finite_budgets_and_callable_contracts(self):
        with self.assertRaises(ValueError):
            ManipulationSkillSpec('bad', '{}', None, None, None, 0, 1., (), ())
        with self.assertRaises(ValueError):
            ManipulationSkillSpec('bad', '{}', lambda *_: True, lambda *_: None,
                                  lambda *_: True, 1, float('nan'), (), ())

    def test_only_next_skill_is_proposed_and_completion_requires_independent_evidence(self):
        candidates = self.skills.candidates(self.observation)
        self.assertEqual([item.name for item in candidates], ['move_to_pregrasp'])
        command = candidates[0]
        fake_success = SkillRun('move_to_pregrasp', 'executed', (), 1, (), True, None)
        self.skills.record_success(command, fake_success)
        self.assertEqual([item.name for item in self.skills.candidates(self.observation)], ['grasp_block'])
        with self.assertRaises(ValueError):
            self.skills.record_success(self.skills.candidates(self.observation)[0], fake_success)

    def test_out_of_order_or_failed_precondition_is_rejected_before_any_step(self):
        command = self.skills.command('grasp_block')
        with self.assertRaises(ValueError):
            self.skills.execute(self.session, command, budget=50)
        self.assertEqual(self.session.sim_time_s, 0.0)

    def test_cancel_and_timeout_are_checked_before_dispatch(self):
        command = self.skills.candidates(self.observation)[0]
        canceled = threading.Event()
        canceled.set()
        result = self.skills.execute(self.session, command, budget=120, cancel_event=canceled)
        self.assertEqual(result.status, 'cancelled')
        self.assertEqual(self.session.sim_time_s, 0.0)
        with self.assertRaises(TimeoutError):
            self.skills.execute(self.session, command, budget=120, timeout_s=.0001)
        self.assertEqual(self.session.sim_time_s, 0.0)

    def test_skill_dispatch_uses_bounded_session_receipts_and_unique_action_ids(self):
        # The zero-motion profile verifies dispatch contracts independently of grasp success.
        command = self.skills.candidates(self.observation)[0]
        run = self.skills.execute(self.session, command, budget=600)
        self.assertIn(run.status, ('executed', 'success_unverified'))
        self.assertGreater(run.control_steps, 0)
        self.assertTrue(all(receipt.executed_steps <= 128 for receipt in run.receipts))
        self.assertEqual(len({receipt.action_id for receipt in run.receipts}), len(run.receipts))
        self.assertEqual(run.final_observation.step_id, self.session.truth_for_verifier().step_id)


if __name__ == '__main__':
    unittest.main()
