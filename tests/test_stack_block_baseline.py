import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stack_block_baseline as baseline


CONFIG = Path(baseline.ROOT / 'configs/tasks/stack_block_baseline.json')


class StackBlockBaselineTests(unittest.TestCase):
    def test_failure_cause_prefers_verifier_detail_over_generic_skill_failure(self):
        stages=[{'skill':'grasp_block','failure':'independent_predicate_not_passed','predicates':[
            {'predicate':'grasped','passed':False,'failure':'bilateral_finger_contact_missing'},
            {'predicate':'lifted','passed':False,'failure':'object_not_raised_after_grasp'},
            {'predicate':'placed','passed':False,'failure':'stable_support_contact_not_observed'},
            {'predicate':'stacked','passed':False,'failure':'stable_stack_not_observed'}]}]
        self.assertEqual(baseline.resolve_failure_cause(None,stages),
                         'bilateral_finger_contact_missing')
        self.assertEqual(baseline.resolve_failure_cause('object_dropped_after_lift',stages),
                         'object_dropped_after_lift')

    def test_default_config_disables_model_and_has_unique_deterministic_seeds(self):
        config=baseline.load_config(CONFIG)
        self.assertEqual(config['world_model'],'disabled')
        self.assertEqual(config['seeds'],[0])
        with tempfile.TemporaryDirectory() as tmp:
            bad=Path(tmp)/'bad.json'
            data=json.loads(CONFIG.read_text()); data['world_model']='A3'; bad.write_text(json.dumps(data))
            with self.assertRaises(ValueError): baseline.load_config(bad)

    def test_new_run_directory_is_required_and_existing_output_is_preserved(self):
        config=baseline.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'existing'; output.mkdir(); marker=output/'keep'; marker.write_text('x')
            with self.assertRaises(FileExistsError): baseline.run_batch(config,output)
            self.assertEqual(marker.read_text(),'x')

    def test_manifest_event_and_result_shapes_keep_failed_episodes_in_denominator(self):
        config=baseline.load_config(CONFIG)
        fake={'seed':0,'episode_id':'ep','status':'failure','success':False,
              'failure_cause':'one_sided_grasp','control_steps':15,'sim_time_s':.3,
              'stage_events':[],'initial_state_profile':'nominal','model_signature':'hash'}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            with patch.object(baseline,'run_episode',return_value=fake):
                result=baseline.run_batch(config,out,CONFIG)
            manifest=json.loads((out/'run_manifest.json').read_text())
            saved=json.loads((out/'results.json').read_text())
            self.assertFalse(manifest['world_model_loaded'])
            self.assertEqual(result['sample_count'],1)
            self.assertEqual(result['failure_count'],1)
            self.assertEqual(saved['success_rate'],0.)
            self.assertTrue(saved['failure_denominator_includes_interrupted_episodes'])
            self.assertTrue((out/'events.jsonl').exists())

    def test_duplicate_seeds_are_rejected_before_run(self):
        config=json.loads(CONFIG.read_text()); config['seeds']=[1,1]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'dup.json'; path.write_text(json.dumps(config))
            with self.assertRaises(ValueError): baseline.load_config(path)

    def test_jsonl_action_trace_serializes_gripper_command_receipt(self):
        from wmal.envs.manipulation_session import ActionCommand
        from wmal.skills.manipulation_contracts import SkillRun

        config=baseline.load_config(CONFIG)

        def execute_one_gripper_action(skills, session, skill_command, budget, *,
                                       cancel_event=None, timeout_s=None, verifier=None):
            observation=session.observe()
            joint_index=observation.joint_names.index('left_shoulder_pitch_joint')
            action=ActionCommand(
                episode_id=observation.episode_id,
                expected_step_id=observation.step_id,
                action_id='traceable-gripper-action',
                joint_targets={'left_shoulder_pitch_joint':
                               observation.joint_position[joint_index]},
                duration_steps=1, gripper_mode='close')
            receipt=session.execute(action)
            return SkillRun(skill_command.name, 'failed', (receipt,),
                            receipt.executed_steps, (), False,
                            receipt.observations[-1], 'stop-after-one-step')

        with tempfile.TemporaryDirectory() as tmp:
            run_dir=Path(tmp)/'run'; run_dir.mkdir()
            with patch.object(baseline.StackBlockSkills, 'execute', execute_one_gripper_action):
                baseline.run_episode(config, 0, run_dir)
            event=json.loads((run_dir/'events.jsonl').read_text(encoding='utf-8'))
            trace=event['action_trace'][0]
            self.assertIn('gripper_command_receipt', trace)
            self.assertEqual(trace['gripper_command_receipt']['command_id'],
                             'traceable-gripper-action')
            self.assertTrue(trace['gripper_command_receipt']['accepted'])
            self.assertTrue(trace['gripper_command_receipt']['executed'])

    def test_interrupted_feedback_still_writes_confirmed_execution_to_jsonl(self):
        from wmal.envs.manipulation_session import ManipulationSession

        config = baseline.load_config(CONFIG)
        original_observe = ManipulationSession.observe
        calls = {'count': 0}

        def fail_after_first_control_dispatch(session):
            calls['count'] += 1
            if calls['count'] >= 5:
                raise RuntimeError('injected persistent camera failure')
            return original_observe(session)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run'
            run_dir.mkdir()
            try:
                with patch.object(ManipulationSession, 'observe', fail_after_first_control_dispatch):
                    baseline.run_episode(config, 0, run_dir, on_step=lambda *_: None)
            except Exception as error:
                self.fail(f'interrupted execution evidence must be written: {error}')
            lines = (run_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(lines), 1)
            event = json.loads(lines[0])
            self.assertEqual(event['status'], 'failed')
            self.assertEqual(event['failure'], 'post_action_observation_failed')
            self.assertEqual(event['end_step_id'], 1)
            self.assertFalse(event['observation_complete'])
            self.assertEqual(event['observation_final']['step_id'], 0)
            trace = event['action_trace'][0]
            self.assertEqual(trace['control_steps'], 1)
            self.assertEqual(trace['end_step_id'], 1)
            self.assertEqual(trace['final_observation'], None)
            self.assertIsNotNone(trace['final_truth'])

    def test_visual_collector_stores_reset_and_post_action_event_states(self):
        from collect_stack_block_visual import EpisodeBuffer
        from wmal.envs.manipulation_session import ActionCommand, LEFT_ARM_JOINTS, ManipulationSession

        session = ManipulationSession(render=True, image_size=32, control_period_s=.02)
        self.addCleanup(session.close)
        observation, _ = session.reset(5, 'nominal')
        buffer = EpisodeBuffer()
        buffer.reset(observation, session.truth_for_verifier())
        receipt = session.execute(ActionCommand(
            episode_id=observation.episode_id, expected_step_id=observation.step_id,
            action_id='event-state-alignment',
            joint_targets={name: observation.joint_position[observation.joint_names.index(name)]
                           for name in LEFT_ARM_JOINTS},
            duration_steps=1))
        buffer.step(receipt.observations[0], receipt,
                    session.truth_trace_for_verifier(receipt)[0])
        _, _, _, events, event_mask, extra = buffer.arrays()
        self.assertEqual(extra['event_states'].shape, (2, 4))
        self.assertEqual(extra['event_state_mask'].shape, (2, 4))
        np.testing.assert_array_equal(extra['event_states'][0], np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(extra['event_states'][1:], events)
        np.testing.assert_array_equal(extra['event_state_mask'][1:], event_mask)


if __name__=='__main__': unittest.main()
