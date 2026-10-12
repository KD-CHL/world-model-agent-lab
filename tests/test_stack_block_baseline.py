import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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


if __name__=='__main__': unittest.main()
