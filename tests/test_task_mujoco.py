"""Actual G1 physics validates hold time and the explicit experimental pulse."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import importlib.util
import runpy
from pathlib import Path
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('mujoco'),'MuJoCo unavailable')
class TaskMuJoCoTests(unittest.TestCase):
    def test_actual_reach_and_hold_have_contiguous_time_and_measured_evidence(self):
        from wmal.envs.visual_workcell import VisualWorkcellSession
        from wmal.agents.task_graph import TaskGraph,TaskNode
        from wmal.agents.task_runtime import RuntimeConfig,TaskRuntime
        class NominalInterface:
            version='no-learned-query'
            normalization={'state_scale':[1,1,1,1]}
            metadata={'max_horizon':4}
            config=type('Config',(),{'context_steps':0})()
            def predict(self,*args):
                raise AssertionError('A1 must not query world model')
        task=TaskGraph('physical_A',(
            TaskNode('reach_A','joint_reach',[.43,.95]),
            TaskNode('hold_A','joint_hold',[.43,.95],('reach_A',),3)))
        with VisualWorkcellSession() as session:
            predictor=NominalInterface()
            predictor.semantics=session.semantics
            started=float(session.data.time)
            result=TaskRuntime(task,session,predictor,config=RuntimeConfig(baseline='A1',horizon=1)).run()
            self.assertEqual(result['state']['status'],'succeeded')
            self.assertEqual(result['state']['nodes']['hold_A']['hold_count'],3)
            self.assertAlmostEqual(session.data.time-started,result['state']['executed_cycles']*.2,places=7)
            evidence=result['state']['nodes']['hold_A']['evidence']
            self.assertLessEqual(np.linalg.norm(np.asarray(evidence['observed_state'])[:2]-[.43,.95]),.035)

    def test_pulse_restores_force_and_faulted_session_cannot_move(self):
        from wmal.envs.visual_workcell import VisualWorkcellSession
        script=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/g1_task_agent.py'))
        with VisualWorkcellSession() as session:
            pulse=script['ShoulderPulse'](session)
            pulse.node_id='hold_A'
            original=session.data.qfrc_applied.copy()
            obs=pulse.observe()
            obs=pulse.execute_actions(np.zeros((1,2)),episode_id=obs.episode_id,step_id=obs.step_id)
            self.assertFalse(pulse.fired)
            obs=pulse.execute_actions(np.zeros((1,2)),episode_id=obs.episode_id,step_id=obs.step_id)
            self.assertTrue(pulse.fired)
            np.testing.assert_array_equal(session.data.qfrc_applied,original)
            pulse.abort('test-stop')
            step=session.step_id
            with self.assertRaises(RuntimeError):
                pulse.execute_actions(np.zeros((1,2)),episode_id=obs.episode_id,step_id=obs.step_id)
            self.assertEqual(session.step_id,step)
