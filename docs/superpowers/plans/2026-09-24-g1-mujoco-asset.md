# G1 MuJoCo Asset Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Unitree G1 29-DOF model loadable and controllable in the project's MuJoCo backend as a fixed-base manipulation test rig.

**Architecture:** Vendor only the pinned upstream MJCF's referenced meshes and license. Create a documented fixed-base derivative with a floor and named RGB camera; configure every limited hinge and torque motor in the existing adapter; expose deterministic headless and interactive simulation modes.

**Tech Stack:** Python 3.10, MuJoCo Python API, unittest, JSON configuration, MJCF.

**Spec:** `docs/superpowers/specs/2026-09-23-g1-unifolm-research-track-design.md` (simulation scope and prerequisites only).

## Global Constraints

- Fixed-base manipulation test only; no balance, locomotion, or real-robot claims.
- Preserve Unitree source attribution and BSD-3-Clause license.
- Use the existing `MujocoBackend` and `RobotProfile` contracts; don't add a new controller framework.
- Keep WMA dataset action mapping marked unverified; 29-DOF actuation alone does not establish schema equivalence.

## Review Focus

- Broken mesh references or missing license: integration test loads the checked-in MJCF and provenance notice identifies the upstream revision.
- Partial actuator coverage or wrong joint bounds: test asserts 29 mapped limited joints and compares every bound/motor against compiled model.
- Unsupported base velocity: config exposes only `joint_positions`; test rejects/does not advertise locomotion.
- No usable image output: camera render test checks named-camera RGB shape.
- Unbounded command execution: CLI validates command targets through `RobotProfile` and uses bounded duration/physics steps.

---

### Task 1: G1 model and test rig

**Files:**
- Create: `robots/assets/unitree_g1/g1_29dof_fixed_base.xml` and referenced `meshes/*.STL`
- Create: `robots/assets/unitree_g1/LICENSE`
- Create: `robots/assets/unitree_g1/README.md`
- Test: `tests/test_g1_mujoco.py`

**Interfaces:** Compiled model exposes 29 limited hinge joints, 29 direct unit-gear torque motors, a `pelvis` body, floor, and camera `pack_camera`.

- [x] Test XML load, counts, camera frame, and one bounded shoulder target.
- [x] Confirm incomplete configuration fails before implementation.
- [x] Import the upstream model and only its referenced meshes; remove only `floating_base_joint` in the derivative and document the difference.
- [x] Add floor and fixed named camera to the derivative, preserving XML mesh lookup.
- [x] Run the focused test and check attribution/license.

### Task 2: Complete G1 profile and actuator map

**Files:**
- Create: `configs/robots/g1_fixed_base.json`
- Test: `tests/test_g1_mujoco.py`

**Interfaces:** JSON contains asset path, 29 exact joint bounds, all actuator mappings, per-joint PD gains, fixed-base metadata, and RGB camera settings.

- [x] Assert mapping keys equal compiled actuated joints and each configured bound matches the MJCF range.
- [x] Populate config from compiled model joint ranges and actuator names; do not enable `base_velocity`.
- [x] Run focused config/backend test.

### Task 3: Simulation CLI

**Files:**
- Modify: `scripts/simulate.py`
- Test: `tests/test_g1_mujoco.py`

**Interfaces:** `--config`, bounded `--steps`, optional `--joint NAME=RAD`, optional `--viewer`; default is deterministic headless hold, viewer selects `pack_camera`.

- [x] Test CLI headless run and invalid joint target rejection.
- [x] Verify test fails against placeholder CLI.
- [x] Implement asset-relative paths, step loop, safe target validation, optional MuJoCo viewer, and concise final state summary.
- [x] Run CLI tests; the optional interactive GUI was not exercised in this headless test run.

### Task 4: Docs and full verification

**Files:**
- Modify: `README.md`, `robots/assets/README.md`, `docs/09_agent_ros2_design.md`, `docs/11_planning_world_model_pipeline.md`, `docs/15_unifolm_training_adaptation.md`, `THIRD_PARTY_NOTICES.md`

- [x] Document setup/commands and exact fixed-base/no-walking limitation; distinguish simulator readiness from dataset/schema readiness.
- [x] Run focused MuJoCo tests, full unittest suite, JSON check, and `git diff --check`.
