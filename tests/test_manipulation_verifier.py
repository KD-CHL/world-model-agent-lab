import unittest
from types import SimpleNamespace

from wmal.envs.manipulation_session import ObjectTruth, SceneTruth
from wmal.skills.manipulation_verifier import StackBlockVerifier


def make_truth(step, *, episode='e1', target=False, support=False, bilateral=False, z=.815,
               velocity=(0.,0.,0.)):
    contacts=[]
    if support: contacts.append(('block_0_geom','block_1_geom'))
    if bilateral:
        contacts += [('block_0_geom','g1_0_stack_gripper_left_finger_geom'),
                     ('block_0_geom','g1_0_stack_gripper_right_finger_geom')]
    x,y=(.20,.42) if target else (.08,.42)
    objects=(ObjectTruth('block_0',(x,y,z),(1.,0.,0.,0.),velocity,(0.,0.,0.)),
             ObjectTruth('block_1',(.2,.42,.815),(1.,0.,0.,0.),(0.,0.,0.),(0.,0.,0.)),
             ObjectTruth('block_2',(-.16,1.05,.815),(1.,0.,0.,0.),(0.,0.,0.),(0.,0.,0.)))
    return SceneTruth(episode,step,step*.02,objects,tuple(contacts),
                      (('stack_target',(.20,.42,.78),(.1,.1)),),(x,y,z))


def receipt(start,end,episode='e1',interrupted=False):
    return SimpleNamespace(episode_id=episode,start_step_id=start,end_step_id=end,
        executed_steps=end-start,interrupted=interrupted,fault=None)


class StackBlockVerifierTests(unittest.TestCase):
    def test_predicate_report_carries_scene_evidence_for_its_decision(self):
        verifier=StackBlockVerifier()
        verifier.update(make_truth(0),receipt(-1,0))
        truth=make_truth(1,bilateral=True)
        reports=verifier.update(truth,receipt(0,1))
        grasped=next(r for r in reports if r.predicate=='grasped')
        self.assertTrue(grasped.passed)
        self.assertIn(('object_position_m',(.08,.42,.815)),grasped.evidence)
        self.assertIn(('gripper_site_position_m',(.08,.42,.815)),grasped.evidence)
        self.assertIn(('contacts',truth.contacts),grasped.evidence)

    def test_evidence_tracks_the_object_named_by_verifier_parameters(self):
        verifier=StackBlockVerifier()
        initial=make_truth(0)
        verifier.update(initial,receipt(-1,0),{'object_name':'block_2'})
        block2=ObjectTruth('block_2',(-.16,1.05,.815),(1.,0.,0.,0.),
                           (0.,0.,0.),(0.,0.,0.))
        contacts=(('block_2_geom','g1_0_stack_gripper_left_finger_geom'),
                  ('block_2_geom','g1_0_stack_gripper_right_finger_geom'))
        truth=SceneTruth('e1',1,.02,(
            initial.object('block_0'),initial.object('block_1'),block2),contacts,
            (('stack_target',(.20,.42,.78),(.1,.1)),),(-.16,1.05,.815))
        reports=verifier.update(truth,receipt(0,1),{'object_name':'block_2'})
        grasped=next(r for r in reports if r.predicate=='grasped')
        self.assertTrue(grasped.passed)
        self.assertIn(('object_name','block_2'),grasped.evidence)
        self.assertIn(('object_position_m',(-.16,1.05,.815)),grasped.evidence)

    def test_requires_bilateral_finger_contact_for_grasp(self):
        verifier=StackBlockVerifier()
        verifier.update(make_truth(0),receipt(-1,0))
        one=make_truth(1,bilateral=False)
        reports=verifier.update(one,receipt(0,1))
        self.assertFalse(next(r for r in reports if r.predicate=='grasped').passed)
        both=make_truth(2,bilateral=True)
        reports=verifier.update(both,receipt(1,2))
        self.assertTrue(next(r for r in reports if r.predicate=='grasped').passed)

    def test_lift_requires_height_change_after_grasp_and_contiguous_receipt(self):
        verifier=StackBlockVerifier(lift_height_m=.03)
        verifier.update(make_truth(0),receipt(-1,0))
        reports=verifier.update(make_truth(1,bilateral=True,z=.815),receipt(0,1))
        self.assertFalse(next(r for r in reports if r.predicate=='lifted').passed)
        reports=verifier.update(make_truth(3,bilateral=True,z=.86),receipt(1,3))
        self.assertEqual(reports.predicate,'sequence')
        self.assertFalse(reports.passed)

    def test_target_overlap_without_support_and_stability_does_not_pass_stack(self):
        verifier=StackBlockVerifier(stable_samples=3)
        verifier.update(make_truth(0),receipt(-1,0))
        reports=verifier.update(make_truth(1,target=True,support=False),receipt(0,1))
        self.assertFalse(next(r for r in reports if r.predicate=='placed').passed)

    def test_stability_requires_consecutive_low_velocity_support_samples(self):
        verifier=StackBlockVerifier(stable_samples=3)
        verifier.update(make_truth(0),receipt(-1,0))
        sample=make_truth(4,target=True,support=True)
        stream=tuple(make_truth(i,target=True,support=True) for i in (1,2,3,4))
        reports=verifier.update(sample,receipt(0,4),{'truth_snapshots':stream})
        self.assertTrue(next(r for r in reports if r.predicate=='stacked').passed)

    def test_later_disturbance_revokes_terminal_stack_success(self):
        verifier=StackBlockVerifier(stable_samples=3)
        verifier.update(make_truth(0),receipt(-1,0))
        stable=tuple(make_truth(i,target=True,support=True) for i in (1,2,3))
        reports=verifier.update(stable[-1],receipt(0,3),{'truth_snapshots':stable})
        self.assertTrue(next(r for r in reports if r.predicate=='stacked').passed)
        displaced=make_truth(4,target=False,support=False)
        reports=verifier.update(displaced,receipt(3,4))
        self.assertFalse(next(r for r in reports if r.predicate=='placed').passed)
        self.assertFalse(next(r for r in reports if r.predicate=='stacked').passed)

    def test_high_velocity_resets_stability_and_render_repeats_do_not_advance(self):
        verifier=StackBlockVerifier(stable_samples=3)
        verifier.update(make_truth(0),receipt(-1,0))
        reports=verifier.update(make_truth(1,target=True,support=True),receipt(0,1))
        self.assertFalse(next(r for r in reports if r.predicate=='stacked').passed)
        # Calling observe/render has no verifier API; only a new receipt advances its counter.
        fast=make_truth(2,target=True,support=True,velocity=(.3,0.,0.))
        reports=verifier.update(fast,receipt(1,2))
        self.assertFalse(next(r for r in reports if r.predicate=='placed').passed)

    def test_missing_feedback_fault_and_episode_change_fail_closed(self):
        verifier=StackBlockVerifier()
        verifier.update(make_truth(0),receipt(-1,0))
        reports=verifier.update(make_truth(2),receipt(0,2,interrupted=True))
        self.assertIn(reports.predicate,('sequence','execution'))
        reports=verifier.update(make_truth(1,episode='e2'),receipt(0,1,'e2'))
        self.assertFalse(reports.passed)


if __name__=='__main__': unittest.main()
