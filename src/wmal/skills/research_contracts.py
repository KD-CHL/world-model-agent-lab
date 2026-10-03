"""Two executable skills with fixed controller/action semantics, not grasping."""
import numpy as np
from wmal.agents.predictive_skill_agent import SkillCandidate
from wmal.envs.visual_workcell import TARGET_LOW, TARGET_HIGH, MAX_DELTA


class JointSkills:
    names = ('joint_reach', 'joint_hold')

    @staticmethod
    def ready(observation, node):
        state = observation.state
        if state.shape != (4,) or not np.isfinite(state).all():
            return False
        if np.any(state[2:] < TARGET_LOW) or np.any(state[2:] > TARGET_HIGH):
            return False
        return node.skill != 'joint_hold' or np.linalg.norm(state[2:] - node.target) <= .015

    @staticmethod
    def candidates(observation, node, horizon, rng):
        if type(horizon) is not int or not 1 <= horizon <= 4:
            raise ValueError('Research prediction horizon must be 1..4')
        if not JointSkills.ready(observation, node):
            return []
        current = observation.state[2:]
        if node.skill == 'joint_hold':
            return [SkillCandidate(node.skill, np.zeros((horizon,2)), np.tile(current,2))]
        direction = np.sign(np.asarray(node.target) - current)
        proposals = [np.tile(direction*.04*gain,(horizon,1)) for gain in (1.,.5,-.5,0.)]
        proposals.extend(rng.uniform(-.04,.04,(horizon,2)) for _ in range(8))
        result = []
        for proposal in proposals:
            target = current.copy()
            actions = []
            for delta in proposal:
                updated = np.clip(target+delta, TARGET_LOW+.005, TARGET_HIGH-.005)
                actions.append(updated-target)
                target = updated
            values = np.asarray(actions)
            if np.any(np.abs(values) > MAX_DELTA+1e-8):
                raise ValueError('Controller proposal exceeded delta contract')
            result.append(SkillCandidate(node.skill,values,np.tile(target,2)))
        return result
