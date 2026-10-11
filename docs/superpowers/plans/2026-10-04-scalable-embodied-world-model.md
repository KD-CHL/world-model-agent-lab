# Scalable Embodied World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 实现可规模化多模态RSSM、独立训练与校准、版本化推理，以及现有G1真实仿真闭环。

**Architecture:** 新模型族与旧VisualWorldModel并存，已知command和学习型physical state分开。每个成员独立训练，posterior真实观测更新后进行action-only prior预测。新provider通过显式adapter进入现有Agent，复用执行回执和监控，不改变旧模型身份。

**Tech Stack:** Python / PyTorch / NumPy / MuJoCo / unittest；仅wmal环境。

**Spec:** `docs/superpowers/specs/2026-10-04-scalable-embodied-world-model-design.md`

## Global Constraints

- 主模型目标约150M参数，三个独立成员约450M，另设约300M的容量对照。
- WM-S 30–50M；WM-B 120–180M；WM-L 240–300M；smoke配置不冒称这些容量的结果。
- 世界模型预测候选动作后果；固定技能或低层控制器执行；语言模型、VLA、记忆、低层控制器与世界模型不同时训练。
- 环境统一为wmal；代码留在当前目录/功能分支，旧模型和校准不改变。
- 不运行未授权的长时间服务器训练、真机动作、merge或push。
- 未来真实观测不输入candidate prior；真实反馈是唯一任务完成依据。
- 数据四split按episode/源内容隔离，微调继承暴露清单；没有标签不输出事件概率。
- 每成员默认8条有seed的categorical轨迹，candidate顺序不改变随机数；协议变化重新校准。
- 每卡正式探测目标allocated/reserved均≤报告总显存80%；记录真实资源，不声称八卡已测。
- 使用apply_patch编辑；先失败测试再实现；每任务完整回归并本地提交。

## Review Focus

1. 相机全部缺失、NaN、错误单位/周期、短startup历史：拒绝或按显式profile/mask处理（Tasks1–3）。
2. 不同candidate遍历顺序、非零action normalization：采样和known-command正确，不污染其他分支（Tasks1/3）。
3. 派生视角跨split、祖先暴露、变更数据后resume：拒绝泄漏或不精确恢复（Tasks2/4）。
4. rank补齐validation与尾梯度累积批次：唯一窗口计数与实际有效批次正确（Task4）。
5. 卡住的推理、late/reset响应、部分执行缺反馈：在派发前拒绝，不盲目重试（Task5）。

## Execution and evidence

用户明确要求计划完成后直接实现、不再等待确认；这覆盖本次规格/计划审批等待。
Tasks1–4逐任务子代理实现与独立审查；Task5子代理额度阻塞后按记录切换主代理inline执行，最终整分支审查另行核验。不并发派发实现者。
基础环境：`/home/chl/miniconda3/envs/wmal/bin/python`。
完整测试：激活wmal、`source scripts/setup_ros2.bash`后`python -m unittest discover -s tests -v`，`MUJOCO_GL=egl`。
每任务测试需保存RED/GREEN证据和全部失败/skip；审查报告不等于实测。
八卡与物体交互资源尚未提供，分别交付可运行配置/接口并标记相应验收未运行。

---

### Task 1: Scalable multimodal RSSM model family

**Files:** Create `src/wmal/models/embodied_config.py`, `embodied_rssm.py`, `embodied_blocks.py`; Create `configs/training/embodied_smoke.json`, `embodied_small.json`, `embodied_base.json`, `embodied_large.json`; Test `tests/test_embodied_rssm.py`.

**Interfaces:**
- `ScalableConfig`: frozen dataclass, physical_dim/command_dim/action_dim/views/image_size/event_dim, context_steps/horizon, deter/hidden/stoch/classes/blocks, encoder_channels/fusion_width/decoder_width, unimix. Validate dimensions, ranges, normalization and block divisibility.
- `network_preset(name, **semantic_dimensions) -> ScalableConfig`, names smoke/small/base/large. Model budget includes all heads and encoder. Count using meta device, report exact preset dimensions; tune meaningful widths to meet ranges, not padding parameters.
- `ScalableRSSMMember(config)` owns normalized-coordinate network only. `observe(rgb, physical, commands, actions, *, view_mask=None, is_first=None, sample=True, seed=0) -> dict`; B,T,V,3,S,S RGB, B,T,P physical, B,T,C commands, B,T-1,A actions.
- `imagine_context(rgb, physical, commands, past_actions, actions, future_commands, *, view_mask=None, is_first=None, sample=True, seed=0) -> dict`; future_commands B,H,C precomputed by provider in correct coordinates. Output `physical` B,H,P residual-from-initial, `rgb` B,H,V,3,S,S, event_logits optional, logits/features/entropy for training. Future observations never accepted.
- `parameter_report(config) -> dict` counts unique total/trainable and budget status on meta tensors without allocating large weights.

- [x] Write failing tests: `test_prior_changes_with_action_but_never_future_observation`, `test_reset_and_startup_context`, `test_sample_seed_does_not_mutate_global_rng`, `test_candidate_order_invariance`, `test_missing_view_mask_and_nan_rejected`, `test_gradients_and_balanced_kl`, `test_preset_real_parameter_budgets`.
- [x] Run `python -m unittest discover -s tests -p test_embodied_rssm.py -v`; expected missing model feature failure, record RED.
- [x] Implement multi-view shared CNN + per-state MLP + fusion, action embedding, grouped recurrent core, categorical posterior/prior with explicit per-call sampling, physics residual and full/residual RGB modes. Reuse symlog and KL primitives, not upstream code.
- [x] Run focused tests plus full suite; all pass, no unreported failure. Create precise preset counts/configs; old classes untouched.
- [x] Commit `feat: add scalable multimodal embodied RSSM family`; report dimensions/counts, red/green and global regression.

### Task 2: Dataset profiles, aligned modality windows and provenance

**Files:** Create `src/wmal/datasets/embodied_sequences.py`, `embodied_profiles.py`; Test `tests/test_embodied_sequences.py`. Old visual_sequences only consumed, not rewritten.

**Interfaces:**
- `ObservationProfile` dataclass declares physical/command/action order, action_schema, camera IDs, size, period, optional event names, controller and known-command rule `joint_delta` or `none`; `to_dict()`/validated `from_dict()`.
- `EmbodiedDataset(manifest, split, *, horizon, context_steps)`: accepts explicit new schema `wmal.embodied_episodes.v1` or legacy adapter with profile. `__getitem__` returns rgb T,V,3,S,S float, physical T,P, commands T,C, actions T-1,A, view_mask/is_first, events/event_mask, context_length, prediction_offset, sample_id, source_episode_id/hash and episode_id.
- `from_visual_manifest(manifest, *, physical_indices, command_indices, command_rule)`: explicit legacy view/profile, no rewriting raw files. Existing G1 has indices [0,1]/[2,3]; other datasets require declared mapping, no automatic 16D→2D conversion.
- `normalization()` fits training only for physical/command/action means/scales; `episode_window_indices` enables episode bootstrap and uniform episode sampling.
- `write_embodied_manifest(path, episodes, profile, source)` and `validate_lineage(...)`: source IDs/hash cannot appear in multiple splits, paths stay inside root, modalities align; ancestor IDs/hashes stay excluded.

- [x] Write RED tests `test_hand_checked_action_alignment`, `test_padding_preserves_real_targets`, `test_multiview_episode_cannot_leak_splits`, `test_missing_and_misaligned_modalities`, `test_legacy_four_state_explicit_mapping`, `test_train_only_normalization_and_ancestor_hashes`.
- [x] Run focused unittest, expected unavailable interfaces; record RED.
- [x] Implement bounded lazy NPZ windows, explicit legacy adaptation and dataset validation; no future masks/images passed by inference adapter. Preserve timing and origin license metadata.
- [x] Focused + full suite; verify literal hand-derived physical/commands/action/time windows and hash rejection.
- [x] Commit `feat: add aligned embodied data profiles and provenance`.

### Task 3: Prediction provider, checkpoint identity and calibrated protocol

**Files:** Create `src/wmal/models/embodied_world.py`, `embodied_calibration.py`; Create `src/wmal/agents/prediction_contracts.py`; Test `tests/test_embodied_provider.py`, `test_embodied_calibration.py`.

**Interfaces:**
- `EmbodiedWorldModel(config, profile, normalization, members, *, metadata=None, device='cpu')`: allow one true member. `save(path)`, `load(path, device='cpu')`, `version`, `semantics`.
- `predict_context(rgb, physical, commands, past_actions, actions, *, seed=0, samples=8, mode='sample', view_mask=None) -> dict`: raw coordinates in, M,S,H,P physical_states, H,C known_commands, M,S,H,V,3,S,S frames, identity/diagnostics out. Normalize only internally; exact commands computed raw before normalization; validate command scope before prediction. `mode='soft'` independent identity/protocol.
- `predict_sample(sample, **kwargs)` strips startup padding and never reads future RGB/state.
- `PredictionRequest` immutable snapshots of profile/controller/episode/context IDs, content hashes, candidate/actions, model/protocol, client deadline; `validate_response(request,response)` validates all binding and finite outputs. Hash canonical arrays with dtype/shape, not just ID.
- `EmbodiedCalibration.fit(records, model, *, alpha=.2, samples=8, mode='sample', scope=None)` uses episode-level simultaneous physical residual maximum over windows/dims/horizons, training-fixed scales, no commands. save/load/validate/bounds with model/protocol/controller/profile/scope binding. Calibration metadata records exact episode IDs/origin hashes; one member refuses ensemble-scaled mode.
- `as_visual_predictor(calibration=None, samples=8)` legacy Agent adapter merges physical and exact command output, uses member mean over samples, first declared camera; diagnostics preserve raw sample uncertainty. Calibration adapter supplies physical bounds and zeros for known commands while correctly translating old normalized error budgets.

- [x] RED tests: `test_raw_increment_with_nonzero_normalization`, `test_single_member_and_no_fake_ensemble`, `test_future_targets_ignored`, `test_protocol_hash_and_checkpoint_roundtrip`, `test_event_labels_not_inferred`, `test_response_hash_reset_and_deadline_binding`, `test_joint_horizon_episode_calibration`, `test_ancestor_calibration_leak_and_profile_mismatch`.
- [x] Run focused unittest (both files); expected missing interfaces; record RED.
- [x] Implement trusted weights_only checkpoint export, content version including semantics/sampling, bounded validated request/response, independent calibration. Never silently load old calibration as new.
- [x] Focused + full suite; old model checksums/shape behavior preserved; adapter only supports declared G1-compatible mappings, others offline-only.
- [x] Commit `feat: add versioned embodied predictions and calibrated bounds`.

### Task 4: Independent training, distributed execution and reproducible resume

**Files:** Create `src/wmal/training/embodied_loss.py`, `embodied_trainer.py`, `embodied_distributed.py`, `embodied_checkpoint.py`, `embodied_cli.py`; Create `scripts/embodied_world.py`; Modify `pyproject.toml` only new CLI; Test `tests/test_embodied_training.py`, `test_embodied_distributed.py`.

**Interfaces:**
- `EmbodiedTrainingConfig`: model preset/config, epochs/members/batch_size/accumulation/lr/seed/device/precision/sequence_length, horizon/context, distributed none/ddp/fsdp, loss weights, mode/sample count, freeze_encoder, task sampling. Server probes T32,K8,H8,micro2,accum4,8 ranks; smoke small data dimensions explicit.
- `embodied_loss(member,batch,normalization,config) -> (loss, metrics)`: posterior reconstruction plus multi-step prior state/RGB/independently normalized changed-region losses, balanced KL and supervised event masks. Sampling uses training-config protocol; padded/burn-in excluded. Metrics physical RMSE and baseline in raw units, raw/clipped KL and clipping rate.
- `train_embodied(manifest,output,config,*,pretrained=None,resume=None,show_progress=True) -> report`: independent members in sequence, episode bootstrap then uniform episode/start sampling. Validation unique sample IDs; choose physical normalized H-average RMSE, earlier tie; no cal/test exposure.
- Checkpoint saves optimizer/scheduler/scaler/rank RNG/sampler/epoch/step/config/data/profile identity at optimizer-step boundary. Exact resume validates same world size and versions; pretrained resets optimizer with explicit exposure inheritance and compatible shapes only.
- Distributed setup honors torchrun LOCAL_RANK/RANK/WORLD_SIZE; CPU gloo for tests, CUDA NCCL for server. Validation de-duplicates IDs; last accumulation group divided by actual group length. DDP gradient synchronization correct; supported FSDP version-gated with explicit error, no silent fallback. Progress emitted rank0 only; failure propagates.
- CLI `wmal-embodied-world {inspect,train,calibrate,evaluate,profile}`; consumes new or explicitly mapped legacy data, output must be new except explicit same-run resume. `profile` performs a real update/inference probe and reports actual peaks, throughput and budget; no server performance claims from dry-run.

- [x] RED tests `test_real_smoke_train_select_and_load`, `test_resume_matches_uninterrupted_step`, `test_resume_rejects_changed_data`, `test_finetune_freeze_and_lineage`, `test_dynamic_mask_is_not_rollout_input`, `test_tail_accumulation_correct`, `test_two_rank_cpu_unique_validation_and_checkpoint`, `test_cli_real_pipeline`.
- [x] Run focused unittest; record RED before implementations.
- [x] Implement loss/training/export/resume/CLI with existing progress and atomic JSON helpers; do not vendor Dreamer trainer or change legacy commands. Record real parameter counts and trainable split.
- [x] Focused + full suite, tiny real CPU/GPU training, CPU two-rank gloo; server CUDA8/FSDP performance remains unverified without hardware. Audit resume test against actual tensors and next RNG samples, not file presence.
- [x] Commit `feat: train and resume scalable embodied world models`.

### Task 5: G1 Agent integration, bounded inference service and execution fencing

**Files:** Create `src/wmal/agents/embodied_provider.py`, `src/wmal/models/embodied_service.py`; Modify `scripts/g1_task_agent.py` and relevant new CLI loading; Test `tests/test_embodied_agent_integration.py`, `test_embodied_service.py`. Existing task success/skills/controller untouched.

**Interfaces:**
- `load_embodied_agent_provider(checkpoint,calibration,*,device='cpu',samples=8) -> predictor,calibration_adapter`, explicit --model-family visual/embodied in G1 CLI, legacy default remains visual.
- Local bounded provider runs prediction in a cancellable owned worker process; timeout terminates that worker and locks further calls until explicit reset, does not leave a background mutator or auto-retry.
- HTTP service `/predict` uses versioned request/response, authentication for non-loopback binding, body/response limits and no execution endpoint. Remote provider uses real transport timeout and client monotonic deadline, checks content binding before exposing result. Credentials supplied env only, never logs.
- Revalidate physical/history/profile/episode/version before dispatch, reject stale/late/reset/duplicate responses. Reuse fixed TaskAgent feedback/accounting and fault lock; no skipped intermediate observation.
- Preserve viewer/monitor continued session and quit lifecycle; waiting-time physics changes invalidate old predictions. Timeout doesn't execute an implicit fallback action.

- [x] RED tests `test_true_mujoco_adapter_trace`, `test_late_after_reset_never_dispatches`, `test_stuck_worker_terminated_and_locked`, `test_real_loopback_transport_binding_and_size_limit`, `test_no_auth_nonloopback_refused`, `test_repeated_task_and_quit_cleanup`.
- [x] Run focused unittest; record RED before adding family switch/service.
- [x] Implement explicit adapters and narrow inference-only service, forward monitor diagnostics including model/protocol/scope/raw errors. Keep old detector, gravity profile, budget and task graph unchanged.
- [x] Focused + full suite; actual MuJoCo steps, real local HTTP and worker timeout. No true-robot commands or public server launch.
- [x] Commit `feat: integrate embodied predictions with bounded G1 planning`.

### Task 6: Reproducible evaluation, startup docs and final acceptance

**Files:** Create `src/wmal/training/embodied_evaluation.py`; Update README and create `docs/30_scalable_embodied_world.md`, `docs/architecture/embodied-world-verification.md`; Test `tests/test_embodied_evaluation.py`.

**Interfaces:**
- `evaluate_embodied(manifest,checkpoint,*,calibration=None,horizon=None,device='cpu') -> report`: untouched test and ancestor exclusions, raw physical per-dim/per-H RMSE, persistence/simple train-fit baseline, RGB/changed-region MSE, sensitivity explicitly separate from accuracy, interval coverage/width and episode/sample counts.
- Output labels identify smoke vs capacity model, current joint-only task, protocol/controller transfer/scope and actual hardware. Physical known-command scores never included as learned prediction gains.
- Document inspect/collect/profile/train/resume/finetune/calibrate/evaluate/G1 persistent viewer commands using onlywmal; eight-card commands use torchrun actual supported configuration. No placeholder checkpoint marketed ready-to-run.
- Acceptance evidence includes current regression counts, parameter meta counts, real local smoke pipeline and genuine MuJoCo protocol integration. Server tests or research gains not run explicitly marked unverified; negative outcomes retained.

- [x] RED `test_hand_computed_physical_metrics_exclude_commands`, `test_test_split_and_ancestry`, `test_coverage_width_and_persistence`, `test_evaluation_cli_writes_report`.
- [x] Run focused tests; record RED, implement evaluator and wire CLI.
- [x] Run full suite with ROS overlay, compileall and git diff --check; actual local GPU smoke collect/train/calibrate/evaluate and Agent smoke, inspect artifacts and frozen legacy identities. No long large-model training.
- [x] Write accurate documentation/evidence and update plan checkboxes. Commit `docs: verify scalable world model training and G1 simulation`.
- [ ] Whole-change independent review; repair important findings with regression tests, rerun full suite; retain feature branch, do not merge/push without new request.
