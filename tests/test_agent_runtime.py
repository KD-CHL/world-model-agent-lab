"""Behavior tests: validation must prevent incorrect robot side effects."""
import copy
import importlib
import unittest


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('wmal.communication.contracts')
        self.assertTrue(hasattr(self.c, 'RobotProfile'), 'Robot contracts not implemented')
        self.profile = self.c.RobotProfile('arm', 'arm', {'joint': (-1.0, 1.0)})
        self.obs = self.c.Observation('arm', 'ep1', 3, 0.06, {'joint': 0.0})

    def test_nan_and_unknown_joints_are_rejected(self):
        for targets in ({'joint': float('nan')}, {'missing': 0.1}, {'joint': 2.0}):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.c.Goal('joint_goal', targets).validate(self.profile)

    def test_plan_cannot_target_other_robot_or_episode(self):
        cmd = self.c.MotionCommand('c1', 'other', 'ep1', 3, 'joint_positions', {'joint': 0.2}, 0.2)
        plan = self.c.Plan('p1', 'model1', [cmd])
        with self.assertRaises(ValueError):
            plan.validate(self.profile, self.obs)

    def test_only_first_command_can_bind_current_observation(self):
        cmd = self.c.MotionCommand('c1', 'arm', 'ep1', 2, 'joint_positions', {'joint': 0.2}, 0.2)
        with self.assertRaises(ValueError):
            self.c.Plan('p1', 'm1', [cmd]).validate(self.profile, self.obs)

    def test_legged_velocity_requires_controller_capability(self):
        interfaces = importlib.import_module('wmal.robots.interfaces')
        for kind, cls in [('go2', interfaces.Go2Interface), ('g1', interfaces.G1Interface)]:
            p = self.c.RobotProfile(kind, kind, {'joint': (-1, 1)})
            command = self.c.MotionCommand('c', kind, 'ep', 0, 'base_velocity', {'vx': .1, 'vy': 0, 'yaw_rate': 0}, .2)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                cls(p).validate_command(command)

    def test_episode_guard_rejects_replay(self):
        guard = importlib.import_module('wmal.communication.episode_guard').EpisodeGuard()
        cmd = self.c.MotionCommand('c1', 'arm', 'ep1', 3, 'joint_positions', {'joint': .1}, .2)
        guard.accept(cmd, self.obs)
        with self.assertRaises(ValueError):
            guard.accept(cmd, self.obs)
        fresh = copy.copy(cmd)
        fresh.command_id = 'c2'
        fresh.episode_id = 'old'
        with self.assertRaises(ValueError):
            guard.accept(fresh, self.obs)

    def test_world_model_changes_planned_action(self):
        planner_module = importlib.import_module('wmal.planners.world_planner')
        class Model:
            version = 'test-model'
            def predict(self, joints, targets, duration_s):
                return dict(targets)
        planner = planner_module.RolloutPlanner(Model(), samples=10, seed=0)
        goal = self.c.Goal('joint_goal', {'joint': .5})
        plan = planner.plan(self.profile, self.obs, goal)
        plan.validate(self.profile, self.obs)
        self.assertGreater(plan.commands[0].values['joint'], 0)
        self.assertEqual(plan.model_version, 'test-model')

    def test_runner_uses_observed_success_not_action_claim(self):
        runner_module = importlib.import_module('wmal.agents.runner')
        c = self.c
        class Client:
            def propose_goal(self, instruction, profile, observation):
                return c.Goal('joint_goal', {'joint': .5})
        class Channel:
            def observe(self, timeout_s=5):
                return c.Observation('arm', 'ep1', 3, .06, {'joint': 0.0})
            def plan(self, profile, observation, goal, timeout_s=5):
                return c.Plan('p', 'test', [c.MotionCommand('c1', 'arm', 'ep1', 3, 'joint_positions', {'joint': .5}, .2)])
            def execute(self, command, timeout_s=5):
                return c.ExecutionResult('c1', 'succeeded', 'controller claim')
        result = runner_module.AgentRunner(Client(), Channel(), self.profile, max_cycles=1).run('move')
        self.assertNotEqual(result.status, 'succeeded')

    def test_runner_rejects_mismatched_execution_receipt(self):
        runner_module = importlib.import_module('wmal.agents.runner')
        c = self.c
        class Client:
            def propose_goal(self, *args): return c.Goal('joint_goal', {'joint': .5})
        class Channel:
            def observe(self, timeout_s=5): return self_obs
            def plan(self, *args, **kw): return c.Plan('p', 'm', [c.MotionCommand('c', 'arm', 'ep1', 3, 'joint_positions', {'joint': .5}, .2)])
            def execute(self, *args, **kw): return c.ExecutionResult('wrong', 'succeeded', '')
        self_obs = self.obs
        result = runner_module.AgentRunner(Client(), Channel(), self.profile).run('move')
        self.assertEqual(result.status, 'failed')


if __name__ == '__main__':
    unittest.main()
