"""Invalid task inputs are rejected before model loading or simulator startup."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT=Path(__file__).resolve().parents[1]/'scripts/g1_task_agent.py'


class TaskAgentCLITests(unittest.TestCase):
    def test_unrelated_replanning_is_not_attributed_to_pulse(self):
        import runpy
        script=runpy.run_path(str(SCRIPT))
        pulse=type('Pulse',(),{'fired':True,'record':{'before_step':5}})()
        report={'state':{'recoveries':2,'nodes':{'hold_A':{'recoveries':0}}}}
        result=script['disturbance_summary'](pulse,report,'shoulder-pulse')
        self.assertTrue(result['injected'])
        self.assertFalse(result['recovery_triggered'])
        self.assertTrue(result['any_recovery'])
    def test_help_is_runnable_and_explains_task_config(self):
        result=subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--config',result.stdout)
        self.assertIn('--recovery',result.stdout)

    def test_illegal_config_is_rejected_before_nonexistent_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json'
            config.write_text(json.dumps({'task':{'schema':'wmal.task_graph.v1','task_id':'x','nodes':[]}}))
            result=subprocess.run([sys.executable,str(SCRIPT),'--checkpoint','missing.pt',
                 '--config',str(config),'--output',str(Path(tmp)/'result')],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('error:',result.stderr)
            self.assertNotIn('FileNotFoundError',result.stderr)
            self.assertFalse((Path(tmp)/'result').exists())

    def test_a0_r1_and_missing_a3_calibration_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            for args in (['--baseline','A0','--recovery'],['--baseline','A3']):
                result=subprocess.run([sys.executable,str(SCRIPT),'--checkpoint','missing.pt',
                     '--config','configs/tasks/g1_joint_sequence.json','--output',str(Path(tmp)/'out'),*args],
                     capture_output=True,text=True)
                self.assertEqual(result.returncode,2,result.stderr)
                self.assertIn('error:',result.stderr)
                self.assertFalse((Path(tmp)/'out').exists())
