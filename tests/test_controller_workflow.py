"""Run public collection and calibration commands; provenance survives the workflow."""
import importlib.util
import json
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[1]


class ControllerWorkflowTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('mujoco'),'MuJoCo unavailable')
    def test_local_collection_records_scope_and_never_commands_outside_it(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'local'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/visual_world.py'),'collect',
                '--output',str(output),'--episodes','20','--steps','4','--image-size','32',
                '--controller','gravity_compensated','--target-low','.21','.73',
                '--target-high','.49','1.01'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['source']['prediction_scope']['command_lower'],[.21,.73])
            for row in manifest['episodes']:
                with np.load(output/row['path']) as episode:
                    self.assertTrue(np.all(episode['states'][:,2:] >= np.array([.21,.73])-1e-7))
                    self.assertTrue(np.all(episode['states'][:,2:] <= np.array([.49,1.01])+1e-7))

    def test_task_cli_exposes_explicit_controller_selection(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/g1_task_agent.py'),'--help'],
                               capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--controller',result.stdout)

    @unittest.skipUnless(importlib.util.find_spec('mujoco'),'MuJoCo unavailable')
    def test_collection_records_controller_and_separates_episode_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'dataset'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/visual_world.py'),'collect',
                '--output',str(output),'--episodes','20','--steps','3','--image-size','32',
                '--controller','gravity_compensated'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['source']['controller_profile']['profile'],'g1_joint_pd_gravity_v2')
            self.assertTrue(all(e['episode_id'].startswith('workcell_gc2_') for e in manifest['episodes']))
            self.assertEqual({e['split'] for e in manifest['episodes']},
                             {'train','validation','calibration','test'})
