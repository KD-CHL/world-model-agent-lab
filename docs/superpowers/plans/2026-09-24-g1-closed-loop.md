# G1 World-Model Closed-Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a persistent G1 MuJoCo experiment loop that calls an explicitly configured action-conditioned world model and replans through an Agent without training a model.

**Architecture:** Isolate floating-base contracts, model adapter, planner, Agent, and persistent simulator session under `wmal.locomotion`; retain existing joint-only APIs and ONNX policy. The CLI owns session lifetime across repeated goals, while each task is receding-horizon and executes only the selected first velocity chunk.

**Tech Stack:** Python 3.10+, NumPy, MuJoCo, ONNX Runtime, unittest; no new training dependency.

**Spec:** `docs/superpowers/specs/2026-09-24-g1-closed-loop-design.md`

## Global Constraints

- No world-model training in this implementation.
- Never present the low-level Unitree locomotion ONNX or MuJoCo as a learned world model.
- Require an explicit versioned action-conditioned predictor factory; do not silently fall back.
- Execute one bounded action chunk per prediction and replan only from a fresh observation.
- Keep G1 task completion separate from simulator-session lifetime.
- Preserve the existing joint-only/ROS APIs and all pre-existing user changes.

## Review Focus

- Missing/incompatible model factory must fail before simulator actions; test startup adapter validation.
- Model returns wrong keys/nonfinite values; test prediction rejection.
- Stale observation/episode; test planner/Agent refuses execution.
- Task succeeds or exhausts budget; test session remains reusable for a second goal.
- Fall, excessive tilt, or viewer close; test safe termination and no further command.

---

### Task 1: Floating-base contracts and world-model adapter

**Files:**
- Create: `src/wmal/locomotion/contracts.py`
- Create: `src/wmal/locomotion/world_model.py`
- Create: `src/wmal/locomotion/__init__.py`
- Create: `tests/test_g1_locomotion_runtime.py`

**Interfaces:** `G1State`, `G1Goal`, `G1VelocityAction`, `G1Prediction`; `WorldModel.predict(state, action, duration_s)`; `load_world_model("module:function", config) -> WorldModel`.

- [ ] Write tests for finite/schema validation, valid predictor factory, missing predictor method, wrong prediction schema, and nonfinite outputs.
- [x] Write tests for finite/schema validation, valid predictor factory, missing predictor method, wrong prediction schema, and nonfinite outputs.
- [x] Run the focused tests and observe expected failures.
- [x] Implement validated data contracts and explicit factory adapter with model version provenance.
- [x] Re-run the focused tests and verify all pass.

### Task 2: Predictive receding-horizon planner and Agent

**Files:**
- Create: `src/wmal/locomotion/planner.py`
- Create: `src/wmal/locomotion/agent.py`
- Modify: `tests/test_g1_locomotion_runtime.py`

**Interfaces:** `G1RolloutPlanner.plan(state, goal) -> G1Plan`; `G1Agent.run_goal(goal, session, max_cycles) -> G1TaskResult`. The session protocol provides `observe()`, `step(action, duration_s)`, and `is_running`.

- [x] Test candidate selection based on model-predicted state, command bounds, and first-chunk-only plan contract.
- [x] Test Agent observes after each executed chunk, replans using fresh state, stops at goal/budget, and can run another goal on the same session.
- [x] Run focused tests and observe expected failures.
- [x] Implement bounded deterministic action sampling, goal/stability/effort/uncertainty cost, provenance, and task/session lifetime separation.
- [x] Re-run focused tests and verify passes.

### Task 3: Persistent MuJoCo G1 session

**Files:**
- Create: `src/wmal/envs/g1_session.py`
- Modify: `src/wmal/envs/g1_locomotion.py`
- Modify: `tests/test_g1_mujoco.py`

**Interfaces:** `G1MuJoCoSession(viewer=True, realtime=True)` context manager, `observe() -> G1State`, `step(action, duration_s) -> G1State`, `close()`. The session reuses the published low-level policy and floating-base model.

- [x] Add a test proving session API steps bounded velocity chunks, remains alive across two goals, and returns finite state; skip only when declared runtime dependencies/assets are unavailable.
- [x] Verify the test fails because the session class/API is absent.
- [x] Implement session lifecycle with one viewer across multiple calls, real-time pacing, command validation, posture/fall checks, and idempotent close.
- [x] Run focused MuJoCo test and existing G1 model tests.

### Task 4: Interactive CLI, config, event log, and user guide

**Files:**
- Create: `scripts/g1_agent_sim.py`
- Create: `configs/experiments/g1_closed_loop.example.json`
- Create: `docs/17_g1_world_model_agent_loop.md`
- Modify: `README.md`
- Create: `tests/test_g1_agent_cli.py`

**Interfaces:** CLI takes explicit experiment config; config names world-model `factory`, model config/version expectations, planner bounds/horizon, simulator settings, and log output. Interactive inputs are bounded world-frame goals (`x y [yaw_deg]`) and `quit`.

- [x] Test config rejects absent model factory and malformed settings; test goal parser and JSONL provenance.
- [x] Run focused tests and verify missing implementation failures.
- [x] Implement session-owning interactive loop, repeat goals without closing viewer, and structured per-cycle logs.
- [x] Document how to connect a local/remote pretrained model factory and clearly distinguish test predictors from valid experimental models.
- [x] Run focused tests; full suite and scaffold/diff checks remain in Task 5.

**Ruling:** Keep the runnable entry as `scripts/g1_agent_sim.py` rather than registering a project console entry point, because the `scripts/` directory is outside the package discovery root (`src/`) and exposing it through setuptools would require unrelated packaging changes.

### Task 5: Acceptance audit

**Files:** all files above.

- [x] Verify the real MuJoCo session can remain alive for multiple step calls and is closed only on explicit exit/safety fault.
- [x] Verify configured world model is called for candidate rollouts and its version appears in plans/logs.
- [x] Verify no-training mode performs no checkpoint creation or training invocation.
- [x] Run full test suite and inspect repository status/diff to ensure unrelated user changes remain untouched.
