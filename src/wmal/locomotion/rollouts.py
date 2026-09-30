"""Validated, member-preserving candidate trajectories in physical units."""
from dataclasses import dataclass, replace
import math
import numpy as np
from wmal.locomotion.contracts import G1State, G1VelocityAction, G1Prediction

ROLLOUT_FIELDS = ('x','y','yaw','vx','vy','yaw_rate','roll','pitch','pelvis_height')


@dataclass(frozen=True)
class EnsembleRollout:
    model_version: str
    initial: G1State
    candidates: tuple
    members: np.ndarray  # [member, candidate, time, field]; world pose/velocity, metres/radians/seconds

    def __post_init__(self):
        if not isinstance(self.model_version,str) or not self.model_version.strip() or not isinstance(self.initial,G1State):
            raise ValueError('Invalid rollout provenance')
        if not self.candidates or not self.candidates[0]:
            raise ValueError('Empty candidate rollout')
        horizon = len(self.candidates[0])
        if any(len(row)!=horizon or any(not isinstance(a,G1VelocityAction) for a in row) for row in self.candidates):
            raise ValueError('Invalid rollout action batch')
        values = np.array(self.members,dtype=np.float64,copy=True)
        if (values.ndim!=4 or values.shape[0]<2
                or values.shape[1:]!=(len(self.candidates),horizon,len(ROLLOUT_FIELDS))
                or not np.isfinite(values).all()):
            raise ValueError('Invalid member trajectory tensor')
        values.setflags(write=False)
        object.__setattr__(self,'members',values)

    def means(self):
        values = self.members.mean(axis=0)
        values[...,2] = np.arctan2(np.sin(self.members[...,2]).mean(axis=0),
                                 np.cos(self.members[...,2]).mean(axis=0))
        return values

    def prediction(self, candidate, step):
        mean = self.means()[candidate,step]
        duration = sum(a.duration_s for a in self.candidates[candidate][:step+1])
        state = replace(self.initial,step_id=self.initial.step_id+step+1,
            sim_time_s=self.initial.sim_time_s+duration,
            **{name:float(mean[i]) for i,name in enumerate(ROLLOUT_FIELDS) if name!='pelvis_height'},
            pelvis_height=max(0.,float(mean[8])),z=max(0.,float(mean[8])))
        spread = self.members[:,candidate,step,:2].var(axis=0).sum()
        return G1Prediction(state,{'position_variance':float(spread)},uncertainty_kind='ensemble_spread')

    @property
    def imagination_steps(self):
        return int(np.prod(self.members.shape[:3]))
