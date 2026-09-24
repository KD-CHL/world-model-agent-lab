# G1 World-Model Agent Closed-Loop Design

**Status:** Approved by the user’s ongoing objective to design the architecture and then implement it without model training.

## Goal

Provide a persistent, testable G1 MuJoCo experiment loop in which an Agent accepts repeated goals, a configured world model predicts action-conditioned future base states, a planner chooses a bounded velocity command, the existing Unitree low-level policy converts that command into joint targets, and MuJoCo returns fresh state for replanning. No world-model training is included. A compatible model must be explicitly supplied at runtime; the project must not label the Unitree ONNX locomotion policy or MuJoCo itself as a learned world model.

## Architecture

Add an isolated `wmal.locomotion` package so current joint-only `Observation`, `Goal`, and `RolloutPlanner` contracts remain backward compatible. The package owns validated G1 floating-base state, navigation goal, velocity action, prediction, world-model adapter, receding-horizon planner, and Agent loop. The simulator session owns MuJoCo stepping and the Unitree policy, and exposes observe/step/close operations without deciding goals or planning.

The world-model protocol is action-conditioned: `predict(state, action, duration_s)` returns a versioned predicted state and optional uncertainty. A factory loaded from an explicit `module:function` configuration adapts local checkpoints or external inference clients. Startup rejects missing methods, empty versions, nonfinite outputs, mismatched state schemas, or incompatible configuration. There is no silent fallback model.

The first planner uses bounded deterministic candidate search over `(vx, vy, yaw_rate)` action chunks. It evaluates predicted terminal goal error, heading error, roll/pitch stability, velocity effort, and available uncertainty; only the first action is executed. After each chunk the Agent reads a new simulator observation, checks safety/progress, and replans. The interactive CLI keeps the viewer/session alive across goals until the user exits or a safety stop occurs.

## State and safety

G1 state contains world-frame base position, yaw, body linear/angular velocity, roll/pitch, pelvis height, and a compact joint/contact summary. Goals use world-frame planar position and optional yaw. Model outputs must preserve the state schema and episode provenance. Commands are bounded by policy-supported linear/yaw speeds and chunk duration. The simulator rejects nonfinite commands, invalid durations, stale steps, falls, excessive tilt, and closed viewers; failures stop the current loop and report a safe failure rather than claiming success.

The initial implementation is a research/simulation interface, not a real-robot safety certification. The model checkpoint’s semantic compatibility remains the experimenter’s responsibility and is recorded in configuration and logs.

## Runtime and artifacts

- `G1MuJoCoSession`: owns model/data/viewer/low-level policy; resets once per session; supports repeated bounded `step` calls.
- `WorldModelAdapter`: explicit callable factory and strict protocol validation, independent of model framework or transport.
- `G1RolloutPlanner`: predictive candidate evaluation and first-action selection.
- `G1Agent`: accepts a goal, replans from fresh observations until success, interruption, safety stop, or per-goal cycle budget; session lifetime is outside the task lifetime.
- `scripts/g1_agent_sim.py`: loads a JSON experiment config, starts the persistent viewer, accepts repeated navigation goals, and writes JSONL records containing model version, goal, observation, predictions, command, execution, and residual.
- Example configuration names a user-supplied model factory and checkpoint; deterministic fake predictors exist only in tests and are not presented as trained models.

## Failure behavior

Malformed model configuration and incompatible prediction fail during startup or before execution. Model-call failure, stale observation, nonfinite prediction, safety threshold violation, or viewer closure prevents the next command. An unsuccessful goal does not close the viewer automatically, but safety-critical simulator faults terminate the session. Logs never serialize credentials or raw HTTP error bodies.

## Validation and acceptance

Unit tests cover state/action validation, adapter compatibility and failures, planner use of model predictions, first-action-only execution, fresh-observation replanning, goal/session lifetime separation, and safety termination. An integration test with a deterministic fake model verifies multiple goals share one session without training or network access. Where MuJoCo and the published policy are installed, a bounded headless smoke test verifies session observe/step/close with finite state and posture guards. Documentation must provide the model-factory contract, configuration, command, interaction examples, and explicitly state that useful predictive planning requires a compatible pretrained or externally served model.

## Out of scope

Training/fine-tuning a world model, implementing a general natural-language LLM goal parser, visual navigation/perception, ROS 2 locomotion transport, deployment to physical G1, and asserting that any arbitrary external model is semantically correct.
