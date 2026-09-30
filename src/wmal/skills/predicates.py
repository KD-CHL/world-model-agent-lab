"""Observed G1 navigation predicates; units: metres and radians."""
import math


def navigation_ready(state, goal):
    return (state.pelvis_height >= .48 and abs(state.roll) <= .65
            and abs(state.pitch) <= .65)


def navigation_reached(state, goal):
    return (math.hypot(state.x-goal.x, state.y-goal.y) <= goal.position_tolerance_m
            and (goal.yaw is None or abs((goal.yaw-state.yaw+math.pi) % (2*math.pi)-math.pi)
                 <= goal.yaw_tolerance_rad))
