# G1 UnifoLM Research Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible G1 LeRobot dataset validation, WMA decision/joint fine-tuning launch support, synchronized image/state action inference contracts, and visual-model evaluation hooks while retaining the current numeric planner baseline.

**Architecture:** Keep Unitree WMA code, weights, and training environment external. Project-owned tools validate/provenance-track LeRobot V2.1 data, generate a WMA training config and launch command, and provide explicit action-vector mapping plus synchronized image/state history for the existing HTTP inference client. ROS camera support is optional/configured; numeric `RolloutPlanner` remains independent. WMA modes match upstream semantics: decision-only or joint decision+simulation training.

**Tech Stack:** Python 3.10+, existing NumPy/MuJoCo/ROS 2 interfaces, optional PyYAML and Hugging Face Hub tooling, Unitree WMA upstream checkout, LeRobot V2.1.

**Spec:** `docs/superpowers/specs/2026-09-23-g1-unifolm-research-track-design.md`

## Global Constraints

- The first robot/task is Unitree G1 pack-camera; G1/Go2 locomotion, bimanual operation, and real-robot deployment are out of scope.
- Keep WMA source, checkpoints, and dataset media outside this MIT-licensed repository.
- Preserve raw LeRobot episodes read-only; split train/validation/test by whole episode and keep provenance.
- Do not assume dataset actions are joint positions. Require explicit dimensions, ordering, units, normalization, and control period.
- Send only the first action from each WMA chunk and require a fresh synchronized observation before the next inference.
- A missing G1 MuJoCo asset or action map must disable closed-loop simulation rather than use placeholder mappings.
- WMA `decision_making_only=False` is joint decision+simulation training, not a documented simulation-only mode.
- Full WMA training is hardware-dependent; launch configuration must expose GPU/process count and immutable output paths.

## Review Focus

- Malformed or partially downloaded LeRobot metadata must fail with actionable validation errors before conversion.
- Episode split assignment must be deterministic, disjoint, exhaustive, and nonempty for train/validation/test.
- Dataset dimensions and state/action key order must never be inferred from dict ordering or silently truncated/padded.
- RGB encodings, strides, stale timestamps, reset episodes, and missing frames must not produce an invalid model request.
- Action unnormalization and mapping must reject non-finite, wrong-width, out-of-limit, over-speed, stale, or over-duration commands before ROS execution.

---

### Task 1: LeRobot Dataset Manifest and Provenance Validation

**Files:**
- Create: `src/wmal/datasets/lerobot.py`
- Modify: `src/wmal/datasets/provenance.py`
- Modify: `src/wmal/datasets/splits.py`
- Create: `scripts/prepare_wma_dataset.py`
- Modify: `pyproject.toml`
- Test: not planned under current execution instruction.

**Interfaces:**
- Produces `inspect_lerobot_v21(root, *, dataset_id, revision, license_id, seed) -> dict`; validates LeRobot v2.1 metadata, video/parquet episode files, camera/state/action shapes and keys, assigns stable episode-level splits, and returns a JSON-serializable manifest.
- Produces `write_manifest(manifest, path) -> Path` with atomic output.
- Script supports metadata inspection and optional `huggingface_hub.snapshot_download` when explicitly requested with a pinned revision and output path.

- [x] Implement strict metadata and file-layout validation with deterministic split helper.
- [x] Emit manifest fields needed for reproducibility and later WMA conversion.
- [x] Keep Hugging Face download dependencies optional and never download implicitly.

### Task 2: WMA Fine-Tuning Config Builder and Launcher

**Files:**
- Create: `src/wmal/training/unifolm_wma.py`
- Create: `scripts/train_unifolm_wma.py`
- Modify: `pyproject.toml`
- Modify: `THIRD_PARTY_NOTICES.md`
- **Interfaces:** consumes Task 1 manifest; produces validated WMA experiment configuration and subprocess launch.

**Interfaces:**
- `build_wma_config(template_path, *, checkpoint, prepared_data, dataset_key, mode, output_path) -> Path`, where `mode` is `decision` or `joint`.
- `build_wma_command(upstream_root, config_path, *, run_name, output_dir, processes, master_port=12366) -> list[str]`.
- `launch_wma_training(..., dry_run=False) -> dict` records command, input hashes, mode and run paths; no base checkpoint overwrite.

- [x] Support optional PyYAML; validate required WMA config paths before writing.
- [x] Set Base checkpoint, prepared data path, data mixture, and upstream-compatible training mode.
- [x] Generate single-process and torch distributed launch commands without editing upstream files.
- [x] Write a run manifest beside logs/checkpoints and reject output collisions unless explicitly resumed by upstream.

### Task 3: RGB Frame Conversion, History, and Time Synchronization

**Files:**
- Create: `src/wmal/communication/observation_history.py`
- Modify: `src/wmal/communication/ros2_nodes.py`
- Modify: `src/wmal/communication/ros2_transport.py`
- Modify: `src/wmal/agents/cli.py`
- Modify: `configs/robots/g1.template.json`
- Test: not planned under current execution instruction.

**Interfaces:**
- `CameraFrame(episode_id, step_id, sim_time_s, camera, encoding, width, height, step_bytes, data)`.
- `ObservationHistory(capacity, max_skew_s).append_pair(obs, camera=..., rgb=...)`, `.matched_history(...)`, `.clear()`.
- ROS robot bridge publishes configured MuJoCo RGB camera frames; ROS channel subscribes to the configured image topic and exposes only matched state/frame histories.

- [x] Add dependency-light RGB8/BGR8 row-stride conversion and shape validation.
- [x] Buffer by episode/step, reject old or cross-episode frames, and clear buffers on reset.
- [x] Add optional camera configuration to robot service; preserve robot-only mode when no camera is configured.
- [x] Add client accessors with timeouts that fail closed if a configured camera has no fresh frame.

### Task 4: Explicit G1 Action Mapping and WMA Policy Runner

**Files:**
- Create: `src/wmal/robots/action_mapping.py`
- Create: `src/wmal/agents/action_runner.py`
- Modify: `src/wmal/models/action_service.py`
- Modify: `src/wmal/communication/ros2_transport.py`
- Modify: `src/wmal/agents/cli.py`
- Modify: `configs/robots/g1.template.json`
- Test: not planned under current execution instruction.

**Interfaces:**
- `ActionMapping.from_config(data, profile)` supports declared action order, robot joint order, units, source normalization bounds, target limits, and duration.
- `ActionPolicyRunner.run(instruction, *, max_cycles, timeout_s)` builds synchronized `ActionQuery`, calls `ActionServiceClient`, maps only the first action, validates a `MotionCommand`, executes, and obtains a fresh observation before looping.
- Policy mode is an explicit CLI choice and does not change the existing goal/planner Agent path.

- [x] Implement invertible min-max action unnormalization and explicit index mapping.
- [x] Reject duplicate/missing names, incompatible widths, unsupported action semantics, non-finite outputs, and limit/speed violations.
- [x] Execute only chunk step zero, bind command to current episode/step, log model version and residual metadata, and never retry a timed-out action.
- [x] Reject G1 placeholder profiles lacking actual joint and actuator mappings.

### Task 5: Separate Visual Prediction Contract and Offline Evaluation

**Files:**
- Create: `src/wmal/models/video_prediction.py`
- Create: `src/wmal/evaluation/video_prediction.py`
- Create: `scripts/evaluate_video_predictions.py`
- Modify: `src/wmal/evaluation/README.md`
- **Interfaces:** visual predictor plugin exposes `version` and `predict_video(observation_history, action_sequence) -> predicted RGB frame sequence`; evaluator compares only to aligned held-out real frames and reports image metrics/provenance separately.

- [x] Define protocol and strict frame/time/shape checks without inventing an undocumented WMA service endpoint.
- [x] Load only explicit `module:factory` plugin paths from CLI configuration.
- [x] Provide offline evaluator for frame MAE/PSNR and optional feature metric plugin; label model-generated frames as predictions.
- [x] Keep video outputs out of `PredictionReport` numeric state fields.

### Task 6: Research CLI, Configuration, and Documentation Integration

**Files:**
- Modify: `README.md`
- Modify: `docs/15_unifolm_training_adaptation.md`
- Modify: `docs/09_agent_ros2_design.md`
- Modify: `docs/11_planning_world_model_pipeline.md`
- Modify: `docs/13_paper_data_pipeline.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- **Interfaces:** user-facing commands mirror Tasks 1–5 and clearly state actual upstream mode semantics, dependencies, dataset/license boundaries, and current G1 asset prerequisites.

- [x] Document one pinned G1 dataset acquisition/manifest/conversion path.
- [x] Document decision-only and joint WMA training commands, output layout, and GPU/process options.
- [x] Document WMA policy inference launch and required G1 mapping/camera fields.
- [x] Document visual offline evaluation and numeric baseline comparison boundaries.

### Task 7: Integration Review and Completion

**Files:** all files listed above.

- [x] Review all interfaces against the approved G1 WMA specification (static review only).
- [x] Confirm no vendor source, checkpoint, or dataset binary entered the repository.
- [x] Report unverified hardware/runtime prerequisites; local GPU was confirmed present, while G1 asset and full training remain unverified.
