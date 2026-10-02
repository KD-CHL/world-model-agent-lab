# Temporal World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align bounded real-history RSSM training, inference, calibration and Agent execution.

**Architecture:** Recompute posterior from a fixed real observation/action context, then roll out only priors. An Agent history ledger accepts contiguous true execution traces. Context protocol is part of model identity.

**Tech Stack:** Existing wmal Python 3.10, PyTorch, NumPy, unittest, MuJoCo; no new environment.

**Spec:** `docs/superpowers/specs/2026-10-02-temporal-world-model-design.md`

## Global Constraints

- Research mainline: prediction trust assists planning; fixed low-level controller.
- All commands use existing `wmal`; do not install Dreamer/JAX dependencies.
- Work in this project directory on a local `codex/` branch; do not push this new task.
- Preserve old K=0 model hashes/calibrations and existing demonstration defaults.
- Context default K=0; new opt-in research configuration K=2, H=4.

## Review Focus

- Startup padding with nonzero action normalization must reset incoming actions.
- Future target changes cannot affect open-loop imagination inputs.
- Candidate branches cannot append or mutate real history.
- Missing/interleaved execution observations must be rejected atomically.
- Changing context or fine-tuning control semantics invalidates incompatible artifacts.

### Task 1: Temporal model and dataset contract

**Files:** Modify `src/wmal/models/{visual_latent,categorical_rssm}.py`,
`src/wmal/datasets/visual_sequences.py`; create `tests/test_temporal_world.py`.

**Interfaces:** Produce `NetworkConfig.context_steps`, `VisualDataset(..., context_steps=0)`,
`CategoricalRSSMMember.imagine_context(rgb, states, past_actions, actions, is_first=None, sample=False)`,
`VisualWorldModel.predict_context(rgb, state, past_actions, actions)` and
`predict_sample(sample, actions=None)` for offline consumers.

- [x] Write tests: same current frame/different real history changes prediction; repeated branches identical;
  malformed/long history rejected; old K=0 hash preserved; dataset short prefixes reset/pad without crossing episodes.
- [x] Run `/home/chl/miniconda3/envs/wmal/bin/python -m unittest discover -s tests -p test_temporal_world.py -v`; expect missing contract failures.
- [x] Implement explicit context, validation and dataset alignment; no future observations passed to imagine.
- [x] Repeat targeted tests and whole suite; expect all pass before task completion.

### Task 2: Deployment-matched learning, calibration and evaluation

**Files:** Modify `src/wmal/training/{rssm_loss,visual_trainer}.py`,
`src/wmal/models/horizon_calibration.py`; add `configs/training/rssm_temporal_world.json`.

**Interfaces:** Consume Task 1. Produce `VisualTrainingConfig.context_steps=0` and artifacts
whose context configuration is preserved by save/load and matched on fine-tune.

- [x] Write tests for real one-epoch training, fine-tune, independent calibration/evaluation,
  start padding parity and target-only future supervision; expect context config/loss failures.
- [x] Implement context-aware loss and wire datasets/offline prediction to same protocol.
- [x] Run targeted + full regressions; run real local dataset smoke in a new ignored output directory.

### Task 3: Agent temporal belief and execution trace

**Files:** Create `src/wmal/agents/observation_history.py`; modify
`src/wmal/agents/predictive_skill_agent.py`, `src/wmal/envs/visual_workcell.py`,
`scripts/visual_skill_agent.py`; create `tests/test_agent_temporal_history.py`.

**Interfaces:** Consume Task 1 `predict_context`; produce bounded immutable real-history ledger;
`record_feedback(..., observations=None)` accepting full action-aligned trace;
`execute_actions(..., on_observation=None)` reporting each executed real step.

- [x] Write tests for branch isolation, bounded retention, rejected missing/duplicate/cross-episode trace,
  atomic failure and genuine step-by-step MuJoCo feedback; expect missing ledger/trace failures.
- [x] Implement private ledger and trace collection, preserving old K=0 callers.
- [x] Run targeted + complete regressions, inspect real MuJoCo temporal Agent smoke output.

### Task 4: Engineering documentation and independent verification

**Files:** Add `docs/27_temporal_world_model_engineering.md`; update `README.md`.

- [x] Document data/control/context contracts, module boundaries, ablation protocol,
  exact wmal commands, source mapping and honest smoke results.
- [x] Run full ROS-overlay suite and compile checks, record exact count/results.
- [ ] Request fresh whole-change code review; fix important findings with failing regression first.
  Review was dispatched but service usage limit prevented any verdict; author self-review only.
- [x] Commit verified implementation locally; report limitations and local branch status.

Verification: 206 tests passed without skips, 16 new behavioral tests; real train/fine/calibrate/evaluate
and bounded MuJoCo A2/A3 experiments completed. Evidence and review boundary:
`docs/architecture/temporal-world-verification.md`.
