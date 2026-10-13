"""Bounded fixed-waypoint skills for the first stack-block manipulation task."""
from dataclasses import dataclass
from math import isfinite
from threading import Event
from time import monotonic
from uuid import uuid4

from wmal.envs.manipulation_session import ActionCommand, LEFT_ARM_JOINTS


SKILL_ORDER = ('move_to_pregrasp', 'grasp_block', 'lift_block',
               'move_to_stack_target', 'place_block')
WAYPOINT_ORDER = ('clearance', 'cross_edge', 'pregrasp', 'grasp',
                  'lift_1', 'lift_2', 'lift_3', 'lift_4', 'lift_5', 'lift',
                  'transport', 'place')


@dataclass(frozen=True)
class ManipulationSkillSpec:
    name: str
    parameter_schema: str
    precondition: object
    execute: object
    success_verifier: object
    max_control_steps: int
    timeout_s: float
    allowed_joint_names: tuple[str, ...]
    gripper_modes: tuple[str, ...]

    def __post_init__(self):
        if not self.name or not self.parameter_schema or not all(callable(x) for x in
                (self.precondition, self.execute, self.success_verifier)):
            raise ValueError('Invalid manipulation skill contract')
        if type(self.max_control_steps) is not int or self.max_control_steps < 1:
            raise ValueError('Skill must have a positive finite control-step budget')
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)) or not isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError('Skill timeout must be finite and positive')
        if not set(self.allowed_joint_names) <= set(LEFT_ARM_JOINTS):
            raise ValueError('Skill joint allowlist exceeds left-arm capability')
        if not set(self.gripper_modes) <= {'open', 'close', 'hold'}:
            raise ValueError('Unknown gripper mode')


@dataclass(frozen=True)
class WaypointProfile:
    profile_id: str
    joint_order: tuple[str, ...]
    points: dict
    max_delta_per_control_step: float = .06

    def __post_init__(self):
        if not self.profile_id or len(set(self.joint_order)) != len(self.joint_order) or not self.joint_order:
            raise ValueError('Waypoint profile needs a unique ordered joint list')
        if not set(self.joint_order) <= set(LEFT_ARM_JOINTS):
            raise ValueError('Waypoint profile includes joints outside the left-arm allowlist')
        if not isinstance(self.points, dict) or not set(self.points) <= set(WAYPOINT_ORDER):
            raise ValueError('Waypoint profile must define every named task waypoint exactly once')
        if isinstance(self.max_delta_per_control_step, bool) or not isfinite(self.max_delta_per_control_step) or self.max_delta_per_control_step <= 0:
            raise ValueError('Invalid waypoint slew limit')
        normalized = {}
        raw_points = dict(self.points)
        if set(raw_points) != set(WAYPOINT_ORDER):
            raise ValueError('Waypoint profile must define every named task waypoint exactly once')
        for name, values in raw_points.items():
            pairs = dict(values) if isinstance(values, dict) else dict(zip(self.joint_order, values)) if len(values) == len(self.joint_order) else {}
            if set(pairs) != set(self.joint_order):
                raise ValueError(f'Waypoint {name} does not match the joint order')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in pairs.values()):
                raise ValueError(f'Waypoint {name} contains a nonfinite joint target')
            normalized[name] = {n: float(pairs[n]) for n in self.joint_order}
        object.__setattr__(self, 'points', normalized)

    def validate_for(self, session):
        if self.max_delta_per_control_step > session._max_joint_delta + 1e-12:
            raise ValueError('Profile slew limit exceeds session limit')
        for point in self.points.values():
            for name, value in point.items():
                lo, hi = session._joint_limits[name]
                if not lo <= value <= hi:
                    raise ValueError(f'Waypoint joint target outside model limits: {name}')
        return True


@dataclass(frozen=True)
class SkillCommand:
    name: str
    episode_id: str
    expected_step_id: int
    command_id: str
    parameters: tuple = ()


@dataclass(frozen=True)
class SkillRun:
    name: str
    status: str
    receipts: tuple
    control_steps: int
    evidence: tuple
    verified: bool
    final_observation: object
    failure: str | None = None
    details: object = None


@dataclass
class StackBlockTaskState:
    completed: set
    episode_id: str | None = None
    last_step_id: int = 0

    def reset(self, episode_id):
        self.episode_id, self.last_step_id = episode_id, 0
        self.completed.clear()


_WAYPOINTS = {
    # Physics-settled joint-servo profile; the synthetic jaw mount compensates
    # wrist rotation at grasp so the opening axis remains horizontal.
    'clearance': (.72283,.75497,-.13441,-.50329,-.56547,-1.16171,-.84556),
    'cross_edge': (.29916,1.19243,.47193,-.90669,-1.00414,-1.34530,-.39834),
    'pregrasp': (.45974,1.06151,.52751,-.57503,-1.26954,-1.56221,-.30969),
    'grasp': (.61009,.85098,.22687,-.17431,-1.46201,-1.43878,-.20428),
    # Incremental, physics-settled lift targets keep the grasp pose nearly
    # constant while the gripper raises the block in short bounded segments.
    'lift_1': (.69088,.92219,.28451,-.26477,-1.18743,-1.42650,.12126),
    'lift_2': (.74324,.92842,.41058,-.41851,-.94214,-1.29506,.38769),
    'lift_3': (.76779,.93945,.49392,-.53785,-.84990,-1.18463,.52587),
    'lift_4': (.79062,.96834,.58173,-.65927,-.79332,-1.07020,.64985),
    'lift_5': (.81057,1.01550,.68097,-.77905,-.76728,-.94996,.76837),
    'lift': (.81920,1.07288,.77250,-.86294,-.78045,-.84762,.86396),
    'transport': (-.03416,1.39001,.80030,-.85453,-.85378,-.30902,1.08291),
    # 6D site pose prevents a tilted wrist from sweeping a finger into support.
    'place': (.14480,.97427,.49163,-.46590,-.71339,-.95920,.63467),
}


def default_waypoint_profile():
    return WaypointProfile('g1-left-stack-v1', LEFT_ARM_JOINTS, _WAYPOINTS)


class StackBlockSkills:
    """Proposes one ordered skill at a time; all motion goes through the session."""
    def __init__(self, profile=None):
        self.profile = profile or default_waypoint_profile()
        self.task_state = StackBlockTaskState(set())
        self._issued = set()
        self._consumed = set()

    def candidates(self, observation, task_state=None, profile=None):
        state = task_state or self.task_state
        if state.episode_id != observation.episode_id:
            state.reset(observation.episode_id)
        if state.last_step_id > observation.step_id:
            return ()
        next_skill = next((name for name in SKILL_ORDER if name not in state.completed), None)
        if next_skill is None:
            return ()
        command = SkillCommand(next_skill, observation.episode_id, observation.step_id, str(uuid4()))
        self._issued.add(command.command_id)
        return (command,)

    def command(self, name):
        if name not in SKILL_ORDER:
            raise ValueError('Unknown stack-block skill')
        return SkillCommand(name, self.task_state.episode_id or '', self.task_state.last_step_id,
                            str(uuid4()))

    def record_success(self, command, run):
        if (not isinstance(command, SkillCommand) or command.command_id not in self._issued
                or command.command_id in self._consumed
                or not isinstance(run, SkillRun) or run.name != command.name
                or not run.verified or run.status not in ('succeeded', 'executed')):
            raise ValueError('Skill completion requires independent verifier evidence')
        self._consumed.add(command.command_id)
        self.task_state.completed.add(command.name)
        self.task_state.last_step_id = (run.final_observation.step_id
                                        if run.final_observation is not None
                                        else command.expected_step_id)

    def execute(self, session, command, budget, *, cancel_event=None, timeout_s=None,
                verifier=None, pre_release_alignment=None, pre_release_anchor=None):
        if not isinstance(command, SkillCommand) or command.name not in SKILL_ORDER:
            raise ValueError('Unknown skill command')
        if type(budget) is not int or budget < 1:
            raise ValueError('Skill budget must be a positive integer')
        obs = session.observe()
        if (command.episode_id != obs.episode_id or command.expected_step_id != obs.step_id
                or command.command_id in self._consumed):
            raise ValueError('Stale or duplicate skill command')
        legal = self.candidates(obs)
        if not legal or legal[0].name != command.name:
            raise ValueError('Skill is out of order or its precondition is not met')
        self._issued.add(command.command_id)
        if cancel_event is not None and cancel_event.is_set():
            return SkillRun(command.name, 'cancelled', (), 0, (), False, obs, 'cancelled_before_dispatch')
        timeout = timeout_s if timeout_s is not None else 60.
        if timeout <= 0:
            raise ValueError('timeout_s must be positive')
        if timeout_s is not None and timeout_s < .001:
            raise TimeoutError('Skill deadline elapsed before dispatch')
        self._consumed.add(command.command_id)
        start = monotonic()
        self.profile.validate_for(session)
        if command.name != 'place_block' and (
                pre_release_alignment is not None or pre_release_anchor is not None):
            raise ValueError('Pre-release alignment can only be used for place_block')
        if ((pre_release_alignment is None) != (pre_release_anchor is None)):
            raise ValueError('Pre-release alignment requires both a controller and grasp anchor')
        receipts, used = [], 0
        alignment_details = None
        aligned_release = False
        if pre_release_alignment is not None:
            release_reserve = 256
            alignment_budget = min(
                int(getattr(pre_release_alignment, 'max_control_steps', 0)),
                budget - release_reserve)
            if alignment_budget < 1:
                return SkillRun(command.name, 'budget_exhausted', (), 0, (), False,
                                session.observe(), 'control_step_budget_exhausted')
            alignment = pre_release_alignment.execute(
                session, pre_release_anchor,
                budget_control_steps=alignment_budget, cancel_event=cancel_event)
            receipts.extend(alignment.receipts)
            used += alignment.control_steps
            alignment_details = alignment.to_dict()
            if alignment.status != 'ready_to_release':
                alignment_evidence = tuple(
                    verifier(alignment.receipts, command.name)
                    if verifier is not None and alignment.receipts else ())
                status = ('cancelled' if alignment.status == 'cancelled' else
                          'budget_exhausted' if alignment.reason == 'control_step_budget_insufficient'
                          else 'failed')
                return SkillRun(command.name, status, tuple(receipts), used,
                                alignment_evidence,
                                False, alignment.final_observation,
                                alignment.reason, {'placement_alignment': alignment_details})
            aligned_release = True
        phases = {
            'move_to_pregrasp': (('clearance',None),('cross_edge',None),('pregrasp',None)),
            'grasp_block': (('grasp',None),('grasp','close')),
            'lift_block': tuple((name,None) for name in
                                ('lift_1','lift_2','lift_3','lift_4','lift_5','lift')),
            'move_to_stack_target': (('transport',None),),
            # Reach the support while holding, release at the final pose, then
            # leave a settle window for independent stability checks.
            'place_block': ((('place','open'),('place',None)) if aligned_release else
                            (('place',None),('place','open'),('place',None))),
        }[command.name]
        try:
            for waypoint, grip in phases:
                current = session.observe()
                target = self.profile.points[waypoint]
                if aligned_release:
                    # The Cartesian controller already put the tool at the
                    # computed pose. Hold its current actuator targets while
                    # opening and settling instead of returning to the fixed
                    # joint waypoint.
                    target = session.actuator_targets_for(LEFT_ARM_JOINTS)
                current_map = dict(zip(current.joint_names, current.joint_position))
                # Calibrated waypoints were settled for a full 128 control
                # cycles; reproduce that fixed duration to match their pose.
                steps = (512 if grip == 'close' else
                         64 if waypoint.startswith(('lift_',))
                         or waypoint == 'lift' else 128)
                while used < budget and steps:
                    if cancel_event is not None and cancel_event.is_set():
                        return SkillRun(command.name,'cancelled',tuple(receipts),used,(),False,session.observe(),'cancelled_mid_skill')
                    if monotonic()-start > timeout:
                        return SkillRun(command.name,'timeout',tuple(receipts),used,(),False,session.observe(),'timeout_mid_skill')
                    chunk=min(128,steps,budget-used)
                    action=ActionCommand(current.episode_id,current.step_id,
                        f'{command.command_id}:{waypoint}:{len(receipts)}',target,chunk,grip,
                        session.gripper_spec.max_opening_m if grip == 'open' else .074,.04,
                        session.gripper_spec.max_force_n)
                    receipt=session.execute(action); receipts.append(receipt); used+=receipt.executed_steps
                    if receipt.interrupted or receipt.executed_steps != chunk:
                        # A failed feedback read is latched by the session. Do not
                        # immediately repeat the same read and lose the receipt;
                        # return the last observation that was actually confirmed.
                        return SkillRun(command.name,'failed',tuple(receipts),used,(),False,
                                        session.last_observation,receipt.fault or 'partial_execution')
                    steps-=receipt.executed_steps; current=session.observe()
                    # Closing is deliberately incremental: stop immediately at
                    # measured bilateral fingertip contact, with four bounded
                    # 128-control-step chunks as the maximum squeeze budget.
                    if grip == 'close' and current.gripper.contact_left and current.gripper.contact_right:
                        steps = 0
                if steps:
                    return SkillRun(command.name,'budget_exhausted',tuple(receipts),used,(),False,session.observe(),'control_step_budget_exhausted')
            final=session.observe()
            evidence=tuple(verifier(receipts,command.name) if verifier else ())
            verified=bool(evidence) and all(getattr(item,'passed',False) for item in evidence)
            # A pure transit only needs measured endpoint convergence.
            if command.name in ('move_to_pregrasp','move_to_stack_target'):
                expected_position = {
                    'move_to_pregrasp': (.08,.42,.96),
                    'move_to_stack_target': (.20,.42,1.00),
                }[command.name]
                position_error = sum((final.tool_position_m[i]-expected_position[i])**2
                                     for i in range(3))**.5
                verified = position_error < .03
                evidence = (('tool_position_error_m', position_error),)
            transit = command.name in ('move_to_pregrasp','move_to_stack_target')
            status=('executed' if verified and transit else
                    'succeeded' if verified else 'success_unverified')
            run=SkillRun(command.name,status,tuple(receipts),used,evidence,verified,final,
                         None if verified else 'independent_predicate_not_passed',
                         ({'placement_alignment': alignment_details}
                          if alignment_details is not None else None))
            if verified:
                self.task_state.completed.add(command.name); self.task_state.last_step_id=final.step_id
            return run
        finally:
            pass
