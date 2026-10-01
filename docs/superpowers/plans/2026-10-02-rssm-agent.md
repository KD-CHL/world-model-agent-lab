# RSSM 与技能 Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 可选随机状态世界模型与可审计多子目标 Agent，保持研究变量隔离。

**Architecture:** 独立 RSSM member 通过既有视觉模型加载/训练/校准接口接入。
Agent 只维护任务和观测证据，不训练控制器；旧确定性网络默认不变。

**Tech Stack:** 本项目 wmal conda，Python/PyTorch/NumPy/unittest/MuJoCo。

**Spec:** `docs/superpowers/specs/2026-10-02-rssm-agent-design.md`

## Global Constraints

- 环境只使用 wmal，不安装 Dreamer 的 JAX/其他项目环境。
- 当前目录自主执行，不进行本轮远程推送。
- 保留 A0–A3、旧 checkpoint 哈希与四路 episode 隔离。
- 不训练语言模型、VLA、记忆或 G1 控制器；不伪造奖励/终止标签。
- 部署/校准从当前真实观测初始化，未来目标帧仅进入训练监督。

## Review Focus

- 老检查点新增字段后仍可加载并保持相同模型版本。
- 未来 RGB 进入后验训练不能污染规划开环预测。
- 验证期类别概率代理可重复，不将 entropy 当成功概率。
- 拒绝动作不能无预算无限重观测；部分失败不能重发。
- 微调不能将祖先暴露回合用于校准/测试。

### Task 1: 随机状态网络与兼容加载

**Files:** Create `src/wmal/models/categorical_rssm.py`, `tests/test_categorical_rssm.py`; Modify `src/wmal/models/visual_latent.py`.

**Interfaces:** `CategoricalRSSMMember(config)` produces `observe(rgb[B,T,3,S,S],states[B,T,D],actions[B,T-1,A],is_first=None,sample=False)` posterior/prior/features/reconstruction and `imagine(rgb[B,3,S,S],state[B,D],actions[B,H,A])` rgb/state/diagnostics. `make_member(config)` chooses implementation; `NetworkConfig` owns architecture/stoch/classes/unimix. Existing `VisualWorldModel.predict/save/load` remain compatible.

- [x] Write tests: `test_prior_actions_and_future_posterior_are_separate`, `test_reset_masks_state_and_incoming_action`, `test_balanced_kl_gradient_paths`, `test_old_checkpoint_and_rssm_roundtrip`.
- [x] Run `PYTHONPATH=src:tests /home/chl/miniconda3/envs/wmal/bin/python -m unittest test_categorical_rssm -v`; Expected: missing capability FAIL.
- [x] Implement categorical probabilities, straight-through sample, independent prior/posterior, reset, symlog state head, RGB decoder and loss helpers; preserve old config hash by omitting new fields for deterministic.
- [x] Run same tests; Expected: PASS; commit network implementation.

### Task 2: 训练/微调/评估完整接入

**Files:** Modify `src/wmal/training/visual_trainer.py`; Create `configs/training/rssm_world.json`; expand `tests/test_categorical_rssm.py`.

**Interfaces:** `VisualTrainingConfig` accepts architecture, stoch, classes, unimix, dyn_weight, rep_weight, free_nats, reconstruction_weight; `batch_loss` chooses RSSM loss or legacy unchanged loss. Existing CLI/config/calibration/evaluate remain shared.

- [x] Write `test_training_finetune_calibration_and_test_isolation`, including checkpoint unchanged after test-file mutation and frozen RGB/state encoders.
- [x] Run focused test; Expected: config arguments not accepted FAIL.
- [x] Add recurrent posterior reconstruction/KL and prior-only open-loop losses; validation loss components include KL/entropy, deterministic validation; preserve lineage and semantics.
- [x] Run network + legacy tests; Expected: PASS; commit training integration.

### Task 3: 可审计调度与实际多子目标入口

**Files:** Modify `src/wmal/agents/predictive_skill_agent.py`, `scripts/visual_skill_agent.py`; Create `tests/test_agent_rssm_integration.py`.

**Interfaces:** `PredictiveSkillAgent.active_stage`, `max_reobservations=3`, defensive array snapshots, identity-bound pending receipts, new state fields/diagnostics. `run_task(...,waypoints=())` uses active_stage; CLI `--waypoints shoulder elbow ...` sequential targets, final --goal remains last.

- [x] Write tests: finite reobserve budget, real-feedback-only multi-stage advancement, source arrays cannot mutate receipt, forged receipt rejected, RSSM diagnostics retained, run_task stage-target integration.
- [x] Run focused tests; Expected: missing budget/stage capability FAIL.
- [x] Implement state/evidence changes and waypoint CLI with validation; A0 multi-stage uses one-step fixed sequence to avoid silently skipping intermediate goals.
- [x] Run Agent + legacy tests; Expected: PASS; commit integration.

### Task 4: 真实数据验收与用户文档

**Files:** Create `docs/25_rssm_agent.md`, `docs/architecture/rssm-agent-verification.md`; Modify `README.md`.

**Interfaces:** Existing CLI trains/calibrates/evaluates RSSM and launches persistent viewer/monitor; run outputs remain git-ignored.

- [x] Run full unittest suite with `MUJOCO_GL=egl PYTHONPATH=src`; Expected: no failures, optional unavailable ROS test explicitly skipped.
- [x] Use existing local MuJoCo visual manifest to train short RSSM run and freeze-encoder fine-tune; calibrate and evaluate disjoint sets. Expected: finite reports and distinct bound versions; record all commands/results.
- [x] Launch bounded actual MuJoCo waypoint Agent run; Expected: genuine prediction/execution evidence, report success or refusal honestly, no guaranteed task success from smoke weights.
- [x] Write source citations, training/finetune/calibration/launch commands, model boundary and experiment roadmap; commit docs after verification.
- [x] Independent review entire change, regression-test important findings, rerun suite and record outcome.
