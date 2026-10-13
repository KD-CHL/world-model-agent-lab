import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stack_block_baseline as baseline
import collect_stack_block_visual as visual_collector


CONFIG = Path(baseline.ROOT / 'configs/tasks/stack_block_baseline.json')


class StackBlockBaselineTests(unittest.TestCase):
    def test_target_block_release_requires_open_gripper_and_no_target_finger_contact(self):
        gripper = SimpleNamespace(
            fault=None, opening_m=.12, target_opening_m=.12, at_target=True,
            contact_left=False, contact_right=False)
        observation = SimpleNamespace(gripper=gripper)
        truth = SimpleNamespace(contacts=())

        self.assertTrue(baseline._target_block_released(
            observation, truth, max_opening_m=.12))
        # Gripper contact flags cover every workcell object, including the
        # supporting block. Only target-to-finger contact invalidates release.
        support_contact = SimpleNamespace(
            fault=None, opening_m=.12, target_opening_m=.12, at_target=True,
            contact_left=True, contact_right=False)
        self.assertTrue(baseline._target_block_released(
            SimpleNamespace(gripper=support_contact), SimpleNamespace(contacts=(
                ('block_1_geom', 'g1_0_stack_gripper_left_finger_geom'),)),
            max_opening_m=.12))
        self.assertFalse(baseline._target_block_released(
            SimpleNamespace(gripper=SimpleNamespace(
                fault=None, opening_m=.069, target_opening_m=.034,
                at_target=False, contact_left=True, contact_right=True)),
            truth, max_opening_m=.12))
        self.assertFalse(baseline._target_block_released(
            observation, SimpleNamespace(contacts=(
                ('block_0_geom', 'g1_0_stack_gripper_left_finger_geom'),)),
            max_opening_m=.12))
        faulted = SimpleNamespace(**{**gripper.__dict__, 'fault': 'gripper_fault'})
        self.assertFalse(baseline._target_block_released(
            SimpleNamespace(gripper=faulted), truth, max_opening_m=.12))

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

    def test_run_batch_rejects_invalid_contact_loss_hold_before_creating_output(self):
        config = baseline.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with self.assertRaisesRegex(ValueError, 'contact-loss hold'):
                baseline.run_batch(
                    config, output, placement_contact_loss_hold_steps=-1)
            self.assertFalse(output.exists())

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
            self.assertEqual(saved['schema'], 'wmal.stack_block_results.v2')
            self.assertIn('target_block_released', saved['task_success_definition'])
            self.assertIn('target_block_released', manifest['task_success_definition'])
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

    def test_rgbxy_recovery_requires_a_frozen_localizer_and_room_for_one_attempt(self):
        config = baseline.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run'
            run_dir.mkdir()
            with self.assertRaisesRegex(ValueError, 'requires a frozen RGB-XY localizer'):
                baseline.run_episode(config, 0, run_dir,
                                     recovery_policy='rgbxy_regrasp_once')

    def test_pre_release_alignment_requires_a_frozen_localizer(self):
        from wmal.skills.manipulation_recovery import StackBlockRGBXYPreReleaseAlignment

        config = baseline.load_config(
            baseline.ROOT / 'configs/tasks/stack_block_rgbxy_recovery_dev.json')
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run'
            run_dir.mkdir()
            with self.assertRaisesRegex(ValueError, 'requires a frozen RGB-XY localizer'):
                baseline.run_episode(
                    config, 9200, run_dir,
                    placement_policy=StackBlockRGBXYPreReleaseAlignment.POLICY_ID)

    def test_visual_collector_requires_localizer_for_rgbxy_placement_collection(self):
        from wmal.skills.manipulation_recovery import StackBlockRGBXYPreReleaseAlignment

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'dataset'
            with self.assertRaisesRegex(ValueError, 'requires a frozen RGB-XY localizer'):
                visual_collector.collect(
                    CONFIG, output, seed_start=9400, episode_count=4,
                    profile='disturbed', image_size=128,
                    placement_policy=StackBlockRGBXYPreReleaseAlignment.POLICY_ID)
            self.assertFalse(output.exists())

    def test_visual_collector_cli_forwards_placement_artifact_arguments(self):
        from wmal.skills.manipulation_recovery import StackBlockRGBXYPreReleaseAlignment

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'dataset'
            with patch.object(visual_collector, 'collect', return_value=output) as collect:
                visual_collector.main([
                    '--config', str(CONFIG), '--output-dir', str(output),
                    '--seed-start', '9400', '--episodes', '4', '--profile', 'disturbed',
                    '--image-size', '128', '--placement-policy',
                    StackBlockRGBXYPreReleaseAlignment.POLICY_ID,
                    '--placement-checkpoint', 'checkpoint.pt',
                    '--placement-calibration', 'calibration.json',
                    '--placement-device', 'cpu', '--placement-max-control-steps', '512',
                    '--placement-contact-loss-hold-steps', '96'])

        kwargs = collect.call_args.kwargs
        self.assertEqual(kwargs['placement_policy'],
                         StackBlockRGBXYPreReleaseAlignment.POLICY_ID)
        self.assertEqual(kwargs['placement_checkpoint'], 'checkpoint.pt')
        self.assertEqual(kwargs['placement_calibration'], 'calibration.json')
        self.assertEqual(kwargs['placement_device'], 'cpu')
        self.assertEqual(kwargs['placement_max_control_steps'], 512)
        self.assertEqual(kwargs['placement_contact_loss_hold_steps'], 96)

    def test_visual_collector_splits_episodes_deterministically_without_seed_order_bias(self):
        labels = visual_collector._episode_split_labels(200, 9700)

        self.assertEqual({name: labels.count(name) for name in
                          ('train', 'validation', 'calibration', 'test')},
                         {'train': 120, 'validation': 20,
                          'calibration': 20, 'test': 40})
        self.assertEqual(len(labels), 200)
        self.assertEqual(labels, visual_collector._episode_split_labels(200, 9700))
        self.assertNotEqual(labels[:120], ['train'] * 120)

    def test_post_release_frame_sampler_ignores_opening_motion(self):
        from types import SimpleNamespace

        opening = SimpleNamespace(
            gripper_mode='open',
            observations=tuple(SimpleNamespace(step_id=step) for step in range(1281, 1409)))
        settling = SimpleNamespace(
            gripper_mode=None,
            observations=tuple(SimpleNamespace(step_id=step) for step in range(1409, 1537)))

        sampled = baseline._post_release_observations((opening, settling))

        self.assertEqual(tuple(row.step_id for row in sampled), (1472, 1504, 1536))

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
