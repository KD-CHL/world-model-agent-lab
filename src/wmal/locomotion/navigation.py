"""Global known-map routing around a local learned-model planner."""
import math
import copy
from wmal.locomotion.contracts import G1Goal


class NavigationPlanner:
    def __init__(self, local, scene, log=None, prediction_margin=.15):
        if not math.isfinite(prediction_margin) or prediction_margin < 0:
            raise ValueError('Invalid prediction margin')
        self.local, self.scene = local, copy.copy(scene)
        self.scene.robot_radius += prediction_margin
        self.log = log or (lambda event, payload: None)
        self.local.scene = self.scene
        self.last_position = None
        self.stalled = 0
        self.episode = None
        self.goal_key = None
        self.last_yaw = None

    @property
    def feedback_scale(self):
        return self.local.feedback_scale

    @feedback_scale.setter
    def feedback_scale(self, value):
        self.local.feedback_scale = value

    def plan(self, state, goal):
        position = (state.x, state.y)
        goal_key = (goal.x, goal.y, goal.yaw)
        if self.episode != state.episode_id or self.goal_key != goal_key:
            self.last_position, self.stalled, self.episode = None, 0, state.episode_id
            self.goal_key = goal_key
        if self.last_position is not None:
            yaw_change = abs((state.yaw-self.last_yaw+math.pi) % (2*math.pi)-math.pi)
            self.stalled = self.stalled + 1 if math.dist(position, self.last_position) < .015 and yaw_change < .015 else 0
        self.last_position = position
        self.last_yaw = state.yaw
        if self.stalled >= 12:
            raise ValueError('Navigation stalled for twelve cycles')
        route = self.scene.route(position, (goal.x, goal.y))
        waypoint = route[1]
        for candidate in route[1:]:
            if math.dist(position, candidate) <= .7 and self.scene.segment_free(position, candidate):
                waypoint = candidate
        # Preserve heading constraints only at the actual final target.
        local_goal = goal if waypoint == (goal.x, goal.y) else G1Goal(*waypoint)
        self.log('navigation', {'route': route, 'waypoint': waypoint, 'stalled_cycles': self.stalled})
        return self.local.plan(state, local_goal)
