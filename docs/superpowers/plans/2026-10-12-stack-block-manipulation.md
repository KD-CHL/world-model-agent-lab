# Stack Block Manipulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a fixed-base MuJoCo stack-block manipulation baseline with an explicitly synthetic parallel-jaw gripper, bounded skills, independent truth verifier, and auditable deterministic episode reset.

**Architecture:** Keep the existing 29-DOF G1 asset and joint servo behavior intact, then add a separately named, explicitly synthetic two-finger gripper proxy at the left wrist for this workcell only. A workcell manipulation session is the sole owner of MuJoCo stepping and separates agent-visible observations from verifier-only scene truth. Fixed joint-space waypoint skills run without a learned world model; an independent verifier judges grasp/lift/place/stack from physical truth and executed samples.

**Tech Stack:** Python 3.10, MuJoCo MjSpec/MjData, NumPy, unittest, existing `wmal` environment.

**Spec:** `docs/superpowers/specs/2026-10-12-stack-block-experiment-design.md`

## Global Constraints

- First target is `stack_block`, fixed-base single-arm G1, one manipulated block, and a clearly labeled simplified parallel-gripper proxy.
- Stage-one skill execution does not use a world model, learned action policy, or privileged object truth.
- No skill directly mutates MuJoCo object state; only the owning session advances physics.
- Agent observations and their serialization never expose object truth or verifier labels.
- Reject invalid, stale, out-of-limit, over-budget, or unknown-name commands; partial/unknown execution is recorded and never blindly resent.
- Success is independently verified from scene truth and contiguous executed samples; image overlap or a close command alone never proves grasp/stack success.
- Every episode begins through explicit reset; faults latch until reset; settling steps are excluded from task transitions and hold counts.
- Preserve old fixed-base, navigation, visual-workcell, and task-graph contracts unless an additive adapter is required.
- Record model/asset/config/verifier/controller versions and hashes in each run manifest.

## Review Focus

- A malformed actuator map, nonfinite target, stale episode/step, or out-of-limit joint/gripper command must be rejected before physics advances.
- Reset must restore free-object pose and velocity, robot targets, gripper state, simulation time, verifier counters, and trace sequence; repeated read-only renders must not count toward stability.
- Object truth, contacts, reset seed, and task labels must not leak through agent-facing observations or skill preconditions.
- Partial execution or missing feedback must preserve confirmed executed steps and latch the session without retrying the action.
- A grasp command with one-sided/no contact, an unstable placement, or mere visual overlap must not pass the independent task verifier.

## File Map

- `src/wmal/envs/workcell.py`: add stack-block-only gripper proxy geometry/joints/actuators and model metadata; preserve base robot asset.
- `src/wmal/robots/gripper.py`: new gripper capability/state/command/receipt contracts and MuJoCo proxy adapter.
- `src/wmal/envs/manipulation_session.py`: new single-owner stepping session, truth-separated observation, action limits, reset/audit and fault latch.
- `src/wmal/skills/manipulation_contracts.py`: new bounded manipulation skill and command contracts, waypoint validation and fixed skills.
- `src/wmal/skills/manipulation_verifier.py`: new independent grasp/lift/place/stack predicates over truth and contiguous execution records.
- `scripts/stack_block_baseline.py`: new deterministic baseline CLI, run manifest and JSONL result export.
- `configs/tasks/stack_block_baseline.json`: fixed task, proxy, controller, waypoint, budget, verifier and nominal evaluation settings.
- `tests/test_gripper.py`, `tests/test_manipulation_session.py`, `tests/test_manipulation_contracts.py`, `tests/test_manipulation_verifier.py`, `tests/test_stack_block_baseline.py`: focused behavior/integration regression coverage.
- `docs/19_manipulation_worlds.md`: update accurate workcell capability and command notes after behavior lands.
- `docs/superpowers/specs/2026-10-12-stack-block-experiment-design.md`: authoritative design for phase-one boundaries.

## Evidence-Based Layout Ruling

Before implementation, a MuJoCo Jacobian probe confirmed that the present forward-facing `stack_block` object poses are outside the left arm's reachable set at its fixed base. The spec now places only the stack-block worktable beside the left arm (approximately `x=0`, `y=0.95` table center; a near-side edge at `y=0.30` with blocks around `y=0.42`); other task layouts remain unchanged. Task 1 must encode and test this layout. This adjusts the fixture geometry, not the task, robot base, or asset identity.

## Tasks

### Task 1: Gripper contracts and explicit workcell proxy

**Files:**
- Create: `src/wmal/robots/gripper.py`
- Modify: `src/wmal/envs/workcell.py`
- Test: `tests/test_gripper.py`
- Test: `tests/test_workcell.py`

**Interfaces:**
- `GripperState(opening_m, target_opening_m, contact_left, contact_right, at_target, fault)` is immutable and finite-validated.
- `GripperCommandReceipt(command_id, requested_mode, requested_opening_m, executed, fault)` records command acceptance and does not imply grasp success.
- `GripperAdapter.reset(data)`, `command(mode, *, opening_m, speed_m_s, force_limit_n)`, `observe(data)` operates only registered proxy joint/actuator IDs.
- `build_workcell('stack_block', seed)` gains a named proxy only for stack_block; metadata identifies it as synthetic and exposes the proxy parameter/version block. Other workcell tasks retain their existing actuator/DOF layout.

- [x] Write tests that build stack_block and prove the side-table/task poses and two independently actuated fingers are named and bounded; other workcell tasks retain their existing layout and actuator count.
- [x] Write gripper tests for open/close/hold, measured opening/contact feedback, reset, and rejection of unknown modes, NaN/Inf, boolean and out-of-range commands.
- [x] Run the gripper/workcell focused tests; the initial missing-adapter test failed before implementation.
- [x] Implement the minimum proxy using named finger joints and position actuators with explicit ranges; compute contact flags from MuJoCo contact geometry between each finger and named block geoms.
- [x] Rerun the focused tests and confirm all pass.
- [x] Commit as `feat: add explicit workcell gripper proxy` (`f8b2b24`).

### Task 2: Truth-separated manipulation session, reset and execution receipt

**Files:**
- Create: `src/wmal/envs/manipulation_session.py`
- Test: `tests/test_manipulation_session.py`
- Modify only if needed: `src/wmal/envs/visual_workcell.py` for shared rendering helpers, without changing its action semantics.

**Interfaces:**
- `RobotObservation(episode_id, step_id, sim_time_s, joint_position, joint_velocity, tool_position_m, tool_quaternion_wxyz, gripper, rgb)` exposes measured robot/tool state but no object pose/contact truth, reset seed or verifier outputs.
- `SceneTruth(episode_id, step_id, sim_time_s, object_states, contacts, target_regions)` is only returned by a verifier-facing method that is not accepted by skills or world-model adapters.
- `ActionCommand(schema, episode_id, expected_step_id, action_id, joint_targets, gripper_command, duration_steps)` is strict and named.
- `ExecutionReceipt(action_id, episode_id, start_step_id, end_step_id, executed_steps, sim_duration_s, observations, limit_events, interrupted, fault)` represents full, partial or unknown progress.
- `ManipulationSession.reset(seed, initial_state_profile) -> (RobotObservation, ResetAudit)`, `.observe()`, `.truth_for_verifier()`, `.execute(command)`, `.abort(reason)` owns all `mj_step` calls.

- [x] Write tests proving observations and serialized observations contain no block pose, contact truth, seed or success fields.
- [x] Write reset tests for exact seed/profile repeatability, object qpos/qvel restoration, robot and gripper target restoration, step/time reset, verifier/trace reset hooks, and excluded settle samples.
- [x] Write command rejection tests for unknown joint, nonfinite/bool/out-of-limit targets, stale episode/step, duration over budget, duplicate action ID and malformed schema; assert sim time does not advance.
- [x] Write execution tests for contiguous observations and exact step/time accounting; inject missing post-action observation and assert confirmed prefix is preserved and session fault-latches without retry.
- [x] Implement session with private ownership of model/data/renderer, immutable copied observations, isolated truth snapshots, explicit settle policy, monotonic IDs and latched faults.
- [x] Rerun focused session tests and confirm all pass.
- [x] Commit as `feat: add audited manipulation session and reset` (`6a4b6a1`).

### Task 3: Bounded fixed skills and independent stack-block verifier

**Files:**
- Create: `src/wmal/skills/manipulation_contracts.py`
- Create: `src/wmal/skills/manipulation_verifier.py`
- Test: `tests/test_manipulation_contracts.py`
- Test: `tests/test_manipulation_verifier.py`

**Interfaces:**
- `ManipulationSkillSpec(name, parameter_schema, precondition, execute, success_verifier, max_control_steps, timeout_s, allowed_joint_names, gripper_modes)` validates a bounded capability.
- `WaypointProfile(profile_id, joint_order, points, max_delta_per_control_step)` is finite, named and prevalidated against model joint limits and configured slew limits.
- `StackBlockSkills.candidates(observation, task_state, profile) -> tuple[SkillCommand, ...]` emits only legal next fixed skill proposals; `execute(session, command, budget) -> ExecutionReceipt` calls only the session.
- `StackBlockVerifier.update(truth, receipt, parameters) -> PredicateReport` tracks `grasped`, `lifted`, `placed`, `stacked`, `failed` with episode/step/time/version/threshold/evidence fields.

- [x] Write skill tests for precondition rejection before actuation, legal skill sequencing, bounded waypoint deltas, joint allowlist enforcement, cancellation, timeout and exactly-once command receipt handling.
- [x] Write verifier tests for bilateral contact, one-sided/no contact rejection, contiguous executed samples, target overlap without support, stability and episode/step gaps.
- [x] Run the focused contract tests; the missing module failed before implementation.
- [x] Implement fixed skills `move_to_pregrasp`, `grasp_block`, `lift_block`, `move_to_stack_target`, `place_block`, `retreat` and the pure-read verifier.
- [x] Rerun focused interface tests and confirm all pass; physical grasp remains unaccepted (see below).
- [x] Commit as `feat: add bounded stack-block skills and verifier` (`d15881b`).

### Task 4: Reproducible no-model stack-block baseline CLI and gate report

**Files:**
- Create: `configs/tasks/stack_block_baseline.json`
- Create: `scripts/stack_block_baseline.py`
- Test: `tests/test_stack_block_baseline.py`
- Modify: `docs/19_manipulation_worlds.md`
- Modify: `README.md`

**Interfaces:**
- CLI accepts config, seed list/profile set, run directory and optional headless/viewer; refuses overwrite unless output path is new/empty by explicit policy.
- Each run writes `run_manifest.json`, `events.jsonl`, and `results.json` with configuration/asset/code hashes, per-stage predicate evidence, step/time cost and explicit failure cause.
- Summary keeps failures and interrupted episodes in the denominator and reports sample counts; ≥90% is a proposed baseline gate to freeze before final testing, not a hard-coded success label.

- [x] Write CLI tests for config/schema validation, deterministic seed propagation, new-run directory safety, manifest hashes, event/result shape and failure denominator.
- [x] Implement a bounded batch runner that resets per episode, calls fixed skills, queries the independent verifier and never loads a world model.
- [x] Document that the gripper is a synthetic proxy and add the exact headless baseline command.
- [x] Run development and held-out integration batches; development is 10/10 and two independent held-out jittered batches total 79/80 (98.75%). Seed 106 tips during place and is retained as a failure.
- [x] End the baseline at independently verified stable placement. Do not execute the old withdrawal trajectory: measured action traces show it can collide with and overturn the stack.
- [x] Build 20 ms synchronized 64x64 RGB/state/action/object-truth data with episode-disjoint train/validation/calibration/test splits; add one independent seed-106 failure episode to test only.
- [x] Train and freeze `visual-23507a4e181f6f130653`, calibrate its horizon bounds, and evaluate against persistence on test episodes.
- [x] Run the entire project test suite in one environment containing both MuJoCo and PyTorch. Windows run: 285 tests, 278 passed and 7 skipped; Git Bash was added to `PATH`, stdin polling is Windows-safe, and symlink assertions skip when the OS denies symlink creation.
- [x] Commit as `feat: add reproducible stack-block baseline runner` (`41ba3b9`).

### Task 5: End-to-end auditable gripper command receipt

**Files:**
- Modify: `src/wmal/robots/gripper.py`
- Modify: `src/wmal/envs/manipulation_session.py`
- Modify: `scripts/stack_block_baseline.py`
- Test: `tests/test_gripper.py`
- Test: `tests/test_manipulation_session.py`
- Test: `tests/test_stack_block_baseline.py`

**Interface ruling:** `accepted` records that the bounded setpoint passed validation; `executed` is true only when the parent action completed its requested simulation control steps without interruption. Partial progress and its exact count remain authoritative in `ExecutionReceipt.executed_steps`; its `fault` is copied to the nested gripper receipt. The gripper command uses the parent `ActionCommand.action_id` so logged execution evidence has one traceable identity. A parent action with no gripper mode has no gripper receipt.

- [x] Add tests for accepted-but-pending adapter receipts, parent action ID binding, completed and interrupted session receipts, and JSONL action-trace serialization.
- [x] Implement receipt finalization in the session and include the nested receipt in baseline event logs.
- [x] Run the focused tests, then the full project suite and `git diff --check`.
- [x] Record the result in the experiment progress and SDD ledger.
- [x] Commit the completed follow-up.

### Final review fix pass: partial feedback evidence and event-label timing

- [x] Preserve verifier truth for completed physics steps when image observation fails; return the last confirmed observation without retrying a latched sensor fault.
- [x] Persist incomplete receipts with physical end step and explicit observation completeness; exclude unpaired observations from visual training rows.
- [x] Store event state/mask at reset and each post-action observation; compare persistence against the prediction-origin label and mask legacy offset-zero windows.
- [x] Re-evaluate only the frozen v10 validation split with corrected event metrics; leave the external test metrics unused.
- [x] Add TDD regressions for interrupted JSONL durability, event-state timing/masks, and exact persistence Brier values; full suite passed (297 run, 290 passed, 7 skipped).
- [x] Add named-object truth evidence to `PredicateReport`; the block_2 regression failed against hard-coded block_0 evidence before the fix, then focused tests passed (9/9) and the current full suite passed (299 run, 292 passed, 7 skipped; Git Bash paths required on Windows).

**M1 status:** The no-model stack-placement gate passes on the tested jitter family (79/80 across two held-out batches). The verifier is now v2 and records a distinct `object_dropped_after_lift` failure with configured thresholds; deterministic disturbed seed 7405 reproduced that report while successful seed 7402 emitted no drop event. `PredicateReport.evidence` now preserves the configured object's name, pose/velocity, contacts, and gripper-center truth used by each report. The synthetic gripper and fixed-arm proxy remain explicit limitations; retreat is not accepted.

**M2 status:** Dataset, training, calibration, and evaluation paths run. v8's independent test and the v9 horizon-32 validation both fail the manipulated `block_0` pose gate; v9 event Brier improves only at longer horizons. Current recovery data contains retries but no successful retry recoveries, and state/object uncertainty is not useful enough for risk-sensitive control. New collections add a fourth `object_dropped_after_lift` event channel. Append-only event-head initialization is implemented and verified against the v8 checkpoint and a four-event manifest; the new channel remains untrained. Keep M2 open: collect with an approved, effective recovery behavior, select against manipulated-object validation metrics, and reserve a fresh final test before freezing the next checkpoint. Do not reuse the external collection's test split for v9 or its successors because it has already informed model iteration.

**M3 status:** Paired A0–A3 experiments are not run. First pass manipulated-object pose/event prediction and useful uncertainty gates on validation and fresh test data, and collect effective failures/recoveries; then freeze a new model and run the equal-budget policy suite.

**M4 status:** ROS 2 real-time control and sim-to-real are not run; no physical robot or matching ROS 2 runtime was available in this workspace.

## Execution Notes

- Use a dedicated Git worktree for implementation; do not mix with any pre-existing user changes in the primary checkout.
- Implement inline with task-by-task TDD and commit each complete task after its named focused verification.
- Do not claim task benefit from model training loss alone. Any A0–A3 experiment must use a frozen model and matched per-seed budgets and report uncertainty against realized prediction error.
- Final review must check truth leakage, reset reproducibility, asset labeling, failure denominators, and compatibility of existing workcell scenes.
