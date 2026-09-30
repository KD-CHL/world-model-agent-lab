"""Real MuJoCo RGB/control alignment; run with MUJOCO_GL=egl headlessly."""
import os
import importlib.util
import io
import runpy
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('mujoco'),'Optional simulation dependencies not installed')
class VisualWorkcellTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'),'Optional learning dependencies not installed')
    def test_viewer_eof_keeps_rendering_without_repeated_input_prompts(self):
        from wmal.envs.visual_workcell import VisualWorkcellSession
        from wmal.models.visual_latent import VisualWorldModel
        script=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/visual_skill_agent.py'))
        class Viewer:
            checks=0
            def is_running(self):
                self.checks+=1
                return self.checks<=5
        with VisualWorkcellSession(image_size=32) as session:
            semantics=session.semantics
        model=SimpleNamespace(version='a0_no_predictor',semantics=semantics,
                config=SimpleNamespace(image_size=32),
                metadata={'training_source':{'task':'stack_block'},'max_horizon':4})
        viewer=Viewer()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryFile(mode='w+') as stdin:
            stdout=io.StringIO()
            argv=['visual_skill_agent.py','--checkpoint','unused.pt','--baseline','A0',
                  '--output',str(Path(tmp)/'run'),'--viewer']
            with patch.object(VisualWorldModel,'load',return_value=model), \
                 patch.object(VisualWorkcellSession,'open_viewer',return_value=viewer), \
                 patch('sys.argv',argv),patch('sys.stdin',stdin),redirect_stdout(stdout):
                script['main']()
            self.assertNotIn('请输入',stdout.getvalue())
            self.assertEqual(viewer.checks,6)

    def test_partial_physics_failure_cancels_target_and_latches_idle_until_reset(self):
        from wmal.envs.visual_workcell import VisualWorkcellSession
        with VisualWorkcellSession('stack_block',image_size=32) as session:
            before=session.observe()
            real_step=session.mj.mj_step
            calls=0
            def fail_second(model,data,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:
                    raise RuntimeError('injected physics failure')
                return real_step(model,data,**kwargs)
            with patch.object(session.mj,'mj_step',side_effect=fail_second):
                with self.assertRaises(RuntimeError):
                    session.execute_actions(np.array([[.04,0.],[.04,0.]]),
                            episode_id=before.episode_id,step_id=before.step_id)
            self.assertEqual(session.step_id,1)
            # Cancellation replaces the unfinished .43 target with measured joint positions.
            np.testing.assert_allclose(session.data.ctrl[session.aids],session.data.qpos[session.qadr])
            stopped=session.data.qpos.copy()
            stopped_time=session.data.time
            for _ in range(10):
                session.idle()
            np.testing.assert_array_equal(session.data.qpos,stopped)
            self.assertEqual(session.data.time,stopped_time)
            with self.assertRaisesRegex(RuntimeError,'reset'):
                session.execute_actions(np.zeros((1,2)),episode_id=session.episode_id,step_id=1)
            recovered=session.reset()
            self.assertNotEqual(recovered.episode_id,before.episode_id)
            after=session.execute_actions(np.array([[.02,0.]]),
                    episode_id=recovered.episode_id,step_id=0)
            self.assertEqual(after.step_id,1)
            self.assertGreater(float(after.state[0]-recovered.state[0]),.001)

    def test_safe_joint_delta_moves_rendered_g1_and_rejects_stale_or_out_of_bounds(self):
        from wmal.envs.visual_workcell import VisualWorkcellSession
        with VisualWorkcellSession('stack_block',image_size=32) as session:
            before=session.observe()
            after=session.execute_actions(np.array([[.04,0.]]),
                    episode_id=before.episode_id,step_id=before.step_id)
            self.assertEqual(after.step_id,1)
            self.assertGreater(float(after.state[0]-before.state[0]),.002)
            self.assertGreater(float(np.abs(after.rgb-before.rgb).sum()),.01)
            with self.assertRaises(ValueError):
                session.execute_actions(np.zeros((1,2)),episode_id=before.episode_id,step_id=0)
            with self.assertRaises(ValueError):
                session.execute_actions(np.ones((1,2)),episode_id=after.episode_id,step_id=1)
            self.assertEqual(session.observe().step_id,1)

    def test_collection_is_four_way_disjoint_and_opens_real_camera_windows(self):
        from wmal.training.visual_collector import collect_visual_workcell
        from wmal.datasets.visual_sequences import VisualDataset
        with tempfile.TemporaryDirectory() as tmp:
            path=collect_visual_workcell(Path(tmp)/'data',episodes=20,steps=3,image_size=32)
            dataset=VisualDataset(path,'train',horizon=2)
            self.assertEqual(dataset[0]['rgb'].shape,(3,3,32,32))
            self.assertGreater(float(dataset[0]['rgb'].std()),.01)
            self.assertEqual(set(r['split'] for r in dataset.manifest['episodes']),
                             {'train','validation','calibration','test'})


if __name__=='__main__':
    unittest.main()
