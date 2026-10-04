# Task2 implementation and verification report

Date: 2026-10-05 (Asia/Shanghai).
Workspace: /home/chl/GitHub/CodeSpace/world-model-agent-lab-main.
Starting revision: 8c338b0 (Task1 normalized network).
Requested commit: `feat: add aligned embodied data profiles and provenance`.

## Result and ownership

Implemented Task2 in exactly these new files:

- `src/wmal/datasets/embodied_profiles.py`
- `src/wmal/datasets/embodied_sequences.py`
- `tests/test_embodied_sequences.py`
- `.superpowers/sdd/2026-10-04-scalable-embodied-world-model/task-2-report.md`

The task brief was read before implementation. The complete TDD skill and its
`writing-good-tests.md` reference were read before writing tests. Production
implementation began only after the initial observed RED run. The
verification-before-completion skill was read and applied before completion.
No legacy files, Task1 models, configurations, package exports, scripts, or
other tasks were edited. No dependencies installed, no independent training
job launched, no subagents spawned, no push/merge performed. The explicitly
requested full regression suite ran its existing training/simulation tests.

## Interfaces and concrete semantics

`ObservationProfile` is a frozen dataclass with explicit physical/command/action
orders and per-coordinate units, action schema, camera IDs, image size,
positive control period, optional event names, command rule, fixed controller,
and origin identity. Orders/units become immutable tuples. Controller/origin
are stored as canonical JSON snapshots and each property returns a detached
dictionary. Constructor inputs, `to_dict()`, `from_dict()`, manifests, descriptor
inputs, dataset rows, and returned lineage do not alias caller-owned metadata.
Nonfinite/non-JSON values and missing/unknown profile fields are rejected.

`joint_delta` requires target names corresponding exactly to action names
(`<joint>_target`) and identical command/action units. Logged next commands
must equal current commands plus actual actions in raw coordinates, with
absolute tolerance 1e-6 and no relative tolerance. `none` is an explicitly
offline arbitrary action profile; it supplies no inferred command rollout.

New manifest schema: `wmal.embodied_episodes.v1`.
Read-only in-memory adapter schema: `wmal.embodied_visual_adapter.v1`.
Cumulative exposure schema: `wmal.episode_lineage.v1`.

Stored NPZ arrays:

- uint8 RGB `[T,V,S,S,3]`
- physical `[T,P]`, commands `[T,C]`, actions `[T-1,A]`
- optional recorded timestamps `[T]`, observed-camera masks `[T,V]`
- optional paired events/event masks `[T-1,E]`

Sample arrays: RGB float32 `[K+H+1,V,3,S,S]` in [0,1],
physical/commands float32 `[K+H+1,P/C]`, actions float32 `[K+H,A]`,
view/reset masks, and future-only events/event masks `[H,E]`.
Samples also carry context length, prediction offset K, offset, period,
timestamps/timing source, stable sample ID, exported episode ID, original
source_episode_id, source_sha256, and additional source_hashes.

Every real observation needs at least one observed camera. Missing events are
zero-valued placeholders with zero supervision masks, even if event names are
declared; they are not inferred labels. Missing timestamps are explicitly
marked `period_derived`, using the period and optional episode start_time_s.
Recorded time is preserved and validated against the control period.

Startup padding repeats the first observation/command/time, pads actions with
zeros, marks padded views unobserved, and sets reset flags through the first
real observation. It does not replace any true prediction action or target.
All valid H-step starting points are indexed, including startup windows;
no targets cross an episode or are padded at its end.

`EmbodiedDataset` checks all four split identities, relative paths, symlink
containment, steps, and all file hashes at construction. Selected arrays load
and validate lazily. The LRU retains at most two decompressed episodes and
rechecks file hashes when loading. Returned arrays cannot mutate cached labels
or coordinates. `episode_window_indices` merges original IDs and original
hash aliases (including renamed re-exports) into episode bootstrap groups.

`normalization()` is train-only. Float64 streaming/merged population moments
fit real coordinates once per exported training episode; overlap windows,
burn-in, and startup padding do not add weight. It returns physical_mean/scale,
command_mean/scale, action_mean/scale with a 0.01 scale floor. Model-facing
numeric values must remain finite after float32 conversion.

`write_embodied_manifest(path, episodes, profile, source)` validates every NPZ
before atomically writing. Original source identities must be explicitly
supplied; new compressed file hashes never replace original hashes. Sources,
licenses, revisions, camera identity, timing, and controller metadata remain
serializable and are retained.

`validate_lineage(episodes, lineage=None, *, context='dataset')` checks both
exported and original IDs/hashes across train/validation/calibration/test.
Different camera exports may share the same split. It includes all original
and exported identities in inherited train/selection exposure. Ancestor
exposure may be reused for training but must be excluded from validation,
calibration, and test. Unknown/incomplete ancestry and ID-only exposure fail
closed. The returned cumulative JSON value is detached from its inputs.

Legacy adaptation requires explicit physical/command indices. Exact G1
4-state/2-action semantics map [0,1] to two measured shoulder/elbow positions
and [2,3] to two known targets, all in radians. No qvel is invented.
No action dimensions are discarded, projected, padded, or automatically
converted; the 16D external fixture remains 16D even when physical coordinates
are explicitly selected. Arbitrary legacy robots require a caller-supplied
profile with units/controller/origin and `command_rule='none'`.
Unsupported inferred legacy command semantics are rejected.
The adapter binds the original manifest root/hash, original semantics,
profile, explicit mapping, and episode rows; tampered descriptors are checked
against the original manifest on construction.

For existing LeRobot conversion manifests, original parquet/video hashes are
read from `source.source_episodes`, matched by the converter's documented
`lerobot_<episode_index>` identity. Both original hashes remain exclusions;
the primary source hash is the parquet hash and additional hashes include
parquet and video. Original episode IDs are namespaced as
`<dataset repo_id>:episode:<episode_index>`. Missing/ambiguous original
records are rejected. No source files are rewritten.

## Hand-derived checks

The fixture independently declares physical observations
[0,2], [2,4], [4,6], [6,8], [8,10];
commands [10,20], [11,18], [14,22], [9,28], [16,20];
actions [1,-2], [3,4], [-5,6], [7,-8];
timestamps 1.0, 1.2, 1.4, 1.6, 1.8.
Camera intensities are respectively 10/20/30/40/50 and 60/70/80/90/100.

At start 1, K=1, H=2 the sample contains the first four observations and the
first three actions, with real prediction targets [4,6], [6,8] and actions
[3,4], [-5,6]. At K=3, two padded observations precede the exact same true
prediction targets. Event targets [1], [0] retain masks [1], [0].
Assertions use literal independently derived values, not production helpers.

Expected train means/scales:

- physical mean [4,6], scale [2.8284271247461903,2.8284271247461903]
- command mean [12,21.6], scale [2.6076809620810595,3.4409301068170506]
- action mean [1.5,0], scale [4.330127018922194,5.477225575051661]

## RED/GREEN commands and observed outputs

All focused iterations used this exact command from the workspace:

```bash
PYTHONPATH=src /home/chl/miniconda3/envs/wmal/bin/python -m unittest discover -s tests -p test_embodied_sequences.py -v
```

1. Initial RED, before either production file existed: exit 1,
   `Ran 13 tests in 0.002s; FAILED (failures=13)`.
   Every failure was `AssertionError: Task2 interfaces unavailable:
   wmal.datasets.embodied_profiles`, caused by the absent module, not a fixture
   typo. This included all six required named tests:
   `test_hand_checked_action_alignment`,
   `test_padding_preserves_real_targets`,
   `test_multiview_episode_cannot_leak_splits`,
   `test_missing_and_misaligned_modalities`,
   `test_legacy_four_state_explicit_mapping`,
   `test_train_only_normalization_and_ancestor_hashes`.

2. Initial GREEN after implementing those contracts: exit 0,
   `Ran 13 tests in 0.130s; OK`.

3. Self-review RED for original multi-file import ancestry, immutable known
   G1 units/controller, float32 overflow, and incomplete ancestry: exit 1,
   `Ran 20 tests in 0.183s; FAILED (failures=5)`.
   Observed failures:
   - `test_explicit_g1_profile_cannot_override_known_units_or_controller`:
     two subtests, `ValueError not raised` for degree units and a different PD.
   - `test_finite_raw_values_must_remain_finite_in_returned_coordinates`:
     `ValueError not raised` for finite 1e100 physical values.
   - `test_legacy_import_retains_original_multifile_hashes`:
     returned the processed NPZ hash instead of literal original `'1'*64`.
   - `test_unknown_or_incomplete_ancestor_lineage_fails_closed`:
     `ValueError not raised` for inherited IDs with absent content hashes.

4. Intermediate run after those fixes: exit 1,
   `Ran 20 tests in 0.157s; FAILED (errors=2)`.
   `test_arbitrary_legacy_profiles_are_explicit_offline_only` and
   `test_legacy_import_retains_original_multifile_hashes` errored because the
   shared external 16D fixture accidentally recorded a G1 PD controller while
   explicitly declaring an external controller. Corrected the fixture so G1
   metadata is only supplied for G1. No production validation was weakened.

5. GREEN after the fixture correction: exit 0,
   `Ran 20 tests in 0.166s; OK`.

6. Additional self-review RED: exit 1,
   `Ran 22 tests in 0.174s; FAILED (failures=5)`.
   - `test_origin_requires_identity_and_profile_input_is_snapshotted`:
     three subtests, `ValueError not raised` for license-only, empty dataset,
     and a dataset object with no repository identity.
   - `test_sample_identity_binds_profile_and_legacy_mapping`:
     two distinct coordinate mappings returned the same sample ID.
   - `test_unproven_controller_and_wrong_known_command_transition_rejected`:
     `ValueError not raised` because an absent historical controller was
     being guessed from the collector kind. Removed that inference.

7. GREEN after origin/sample/controller corrections: exit 0,
   `Ran 22 tests in 0.173s; OK`.

8. Final focused GREEN after refactoring lineage accumulation to linear
   set accumulation (sort only at the end): exit 0,
   `Ran 22 tests in 0.194s; OK`. No failures/errors/skips.

## Full regression: run exactly once before commit

Command:

```bash
export CONDA_DEFAULT_ENV=wmal CONDA_PREFIX=/home/chl/miniconda3/envs/wmal MUJOCO_GL=egl
export PATH=/home/chl/miniconda3/envs/wmal/bin:$PATH
source scripts/setup_ros2.bash && /home/chl/miniconda3/envs/wmal/bin/python -m unittest discover -s tests -v
```

Result: exit 0; `Ran 319 tests in 23.111s; OK`.
319 passed, 0 failures, 0 errors, 0 skips. ROS setup succeeded, including
project Python 3.10 bindings and generated interface support.
The suite exercised the 22 Task2 tests, Task1 normalized-network regressions,
legacy model/trainer/calibration tests, real MuJoCo and ROS interfaces, and
existing monitor/agent tests. No test failure or skip is omitted.

The output contains one expected ROS negative-test diagnostic:
`[world_model_planner]: Planning failed: ValueError` within
`test_planner_service_starts_and_rejects_invalid_payload`; that test passed.
Existing training-loop tests also emitted normal JSON collection/evaluation
events. Neither diagnostic is a unittest failure.

Additional checks: the three owned Python files compiled successfully using
`/home/chl/miniconda3/envs/wmal/bin/python -m py_compile ...`.
`git diff --check` passed before staging. The first staged check identified a
trailing blank line in this report; it was removed using apply_patch, and the
staged check was repeated before commit. Staged ownership contains exactly the
four new files listed above. Read-only inspection confirmed existing
processed G1 manifests lack recorded controller descriptors. Tracked legacy
dataset/model/script files have no diff.

## Self-review and concerns for Tasks3/4

Self-review checked realistic mutations: shifted actions, lost padding target,
train-only moments accidentally weighted by windows, missing or invented
labels, masked camera mismatch, broken timestamp alignment, compressed-hash
substitution, renamed original episodes, incomplete ancestor hashes, mutable
profile metadata, altered controller/unit semantics, and sample-ID collisions.
Focused tests caught and corrected the provenance/units/overflow/identity issues
above. Shape/path/hash/cache checks use real NPZ files, including symlink escape
and mutations, rather than mocks.

Remaining integration boundaries (not claimed implemented):

- Tasks3/4 own the inference provider and trainer. They must call
  `validate_lineage` with inherited exposure before selection/calibration/test;
  this dataset module cannot inspect unavailable checkpoints.
- For inference, slice ONLY real history: with K=prediction_offset and
  L=context_length, retain observations/view masks [K-L:K+1] and already
  executed actions [K-L:K]. Candidate actions start at K. Future RGB,
  physical/commands, labels and future masks are supervision, not inference
  inputs. Task1 requires a real camera per input observation; padded zero-view
  history must be stripped before its network API.
- Existing processed G1 manifests without a controller descriptor require an
  explicitly declared profile; the adapter refuses to guess a historical PD.
  A declared explicit profile still has to match schema/orders/mapping/units.
- Arbitrary `none` profiles are offline only. A future provider must reject
  deterministic online command inference for them. No executable conversion
  from 16D real-robot commands to the 2D G1 action interface is supplied.
- Origin hashes are provenance declarations, not cryptographic proof of
  external raw files that are unavailable here. Exported files are verified;
  known LeRobot multi-file originals are retained from recorded hashes.
- Compressed NPZ requires a whole episode to be decompressed. Retained cache
  count is bounded by two episodes, not an absolute byte quota; the window
  index remains proportional to the number of valid starts. Normalization fits
  each export once, so intentional duplicate camera exports can weight moments;
  bootstrap groups nevertheless merge the original trajectory aliases.
- Missing timestamps preserve only declared period/episode-local time, not a
  fabricated absolute measurement. Missing event labels remain unsupervised.
- No new provider/trainer, checkpoint, performance claim, server training, or
  live robot capability is delivered by Task2.

## Complete full-suite transcript

```text
test_rejects_incompatible_robot_data_before_call (test_action_service.ActionServiceTests) ... ok
test_rejects_nonfinite_or_wrong_width_response (test_action_service.ActionServiceTests) ... ok
test_remote_plain_http_is_rejected (test_action_service.ActionServiceTests) ... ok
test_request_maps_history_and_returns_bounded_chunk (test_action_service.ActionServiceTests) ... ok
test_calibrated_alarms_are_consecutive_and_explicit (test_agent_architecture.ArchitectureTests) ... ok
test_calibration_cli_roundtrip (test_agent_architecture.ArchitectureTests) ... ok
test_calibration_is_episode_based_and_rejects_leakage (test_agent_architecture.ArchitectureTests) ... ok
test_coordinator_reinvokes_language_after_typed_failure (test_agent_architecture.ArchitectureTests) ... ok
test_dataset_calibration_reads_only_validation_and_rejects_leakage (test_agent_architecture.ArchitectureTests) ... ok
test_joint_adapter_uses_shared_events_and_observed_success (test_agent_architecture.ArchitectureTests) ... ok
test_language_recovery_context_and_endpoint_validation (test_agent_architecture.ArchitectureTests) ... ok
test_observed_goal_success_takes_priority_over_residual_alarm (test_agent_architecture.ArchitectureTests) ... ok
test_recovery_cannot_change_endpoint_or_retry_uncertain_failure (test_agent_architecture.ArchitectureTests) ... ok
test_recovery_preserves_original_mission_and_budget (test_agent_architecture.ArchitectureTests) ... ok
test_skill_rejects_false_success_and_failed_precondition (test_agent_architecture.ArchitectureTests) ... ok
test_summary_retains_incomplete_and_failed_tasks (test_agent_architecture.ArchitectureTests) ... ok
test_total_budget_overrides_per_goal_budget (test_agent_architecture.ArchitectureTests) ... ok
test_visual_adapter_checks_success_after_last_allowed_action (test_agent_architecture.ArchitectureTests) ... ok
test_zero_progress_cannot_cause_unbounded_recovery (test_agent_architecture.ArchitectureTests) ... ok
test_duplicate_does_not_execute_twice_and_reset_discards_old (test_agent_architecture.ProtocolTests) ... ok
test_expired_command_never_executes (test_agent_architecture.ProtocolTests) ... ok
test_observe_cannot_clear_local_timeout_latch (test_agent_architecture.ProtocolTests) ... ok
test_old_reset_receipt_is_rejected_after_a_later_reset (test_agent_architecture.ProtocolTests) ... ok
test_partial_execution_fault_is_cached (test_agent_architecture.ProtocolTests) ... ok
test_roundtrip_and_reset (test_agent_architecture.ProtocolTests) ... ok
test_scene_mismatch_and_stale_step (test_agent_architecture.ProtocolTests) ... ok
test_timeout_latches_client_and_no_retry (test_agent_architecture.ProtocolTests) ... ok
test_array_snapshots_and_forged_execution_receipt (test_agent_rssm_integration.AgentRSSMIntegrationTests) ... ok
test_feedback_only_multistage_and_diagnostics_audit (test_agent_rssm_integration.AgentRSSMIntegrationTests) ... ok
test_receipt_array_metadata_mutation_cannot_change_authoritative_feedback (test_agent_rssm_integration.AgentRSSMIntegrationTests) ... ok
test_reobserve_budget_is_internal_and_cannot_loop_forever (test_agent_rssm_integration.AgentRSSMIntegrationTests) ... ok
test_run_task_generates_actions_toward_active_waypoint (test_agent_rssm_integration.AgentRSSMIntegrationTests) ... ok
test_successful_receipt_resets_consecutive_reobserve_budget (test_agent_rssm_integration.AgentRSSMIntegrationTests) ... ok
test_episode_guard_rejects_replay (test_agent_runtime.RuntimeTests) ... ok
test_legged_velocity_requires_controller_capability (test_agent_runtime.RuntimeTests) ... ok
test_nan_and_unknown_joints_are_rejected (test_agent_runtime.RuntimeTests) ... ok
test_only_first_command_can_bind_current_observation (test_agent_runtime.RuntimeTests) ... ok
test_plan_cannot_target_other_robot_or_episode (test_agent_runtime.RuntimeTests) ... ok
test_plan_rejects_joint_target_that_exceeds_configured_speed (test_agent_runtime.RuntimeTests) ... ok
test_runner_rejects_mismatched_execution_receipt (test_agent_runtime.RuntimeTests) ... ok
test_runner_uses_observed_success_not_action_claim (test_agent_runtime.RuntimeTests) ... ok
test_world_model_changes_planned_action (test_agent_runtime.RuntimeTests) ... ok
test_intermediate_mismatch_triggers_reobserve_even_when_final_prediction_matches (test_agent_temporal_history.AgentTemporalTests) ... ok
test_missing_malformed_trace_is_rejected_atomically_and_valid_receipt_can_follow (test_agent_temporal_history.AgentTemporalTests) ... ok
test_partial_failure_marks_context_unusable_and_never_retries (test_agent_temporal_history.AgentTemporalTests) ... ok
test_real_feedback_builds_bounded_belief_and_candidates_do_not_advance_it (test_agent_temporal_history.AgentTemporalTests) ... ok
test_real_mujoco_execution_reports_every_real_step (test_agent_temporal_history.AgentTemporalTests) ... ok
test_trace_sink_failure_latches_mujoco_at_last_executed_step (test_agent_temporal_history.AgentTemporalTests) ... ok
test_trace_snapshots_cannot_be_changed_by_public_observation_headers (test_agent_temporal_history.AgentTemporalTests) ... ok
test_api_error_does_not_expose_response_body (test_api_client.ApiTests) ... ok
test_api_rejects_executable_code_and_out_of_bounds (test_api_client.ApiTests) ... ok
test_api_rejects_plaintext_remote_credentials (test_api_client.ApiTests) ... ok
test_api_sends_capabilities_and_validates_goal (test_api_client.ApiTests) ... ok
test_truncated_response_is_not_executed (test_api_client.ApiTests) ... ok
test_unexpected_response_shape_is_rejected_without_execution (test_api_client.ApiTests) ... ok
test_balanced_kl_gradient_paths (test_categorical_rssm.CategoricalRSSMTests) ... ok
test_missing_event_labels_are_not_inference_probabilities (test_categorical_rssm.CategoricalRSSMTests) ... ok
test_old_checkpoint_and_rssm_roundtrip (test_categorical_rssm.CategoricalRSSMTests) ... ok
test_prior_actions_and_future_posterior_are_separate (test_categorical_rssm.CategoricalRSSMTests) ... ok
test_reset_masks_state_and_incoming_action (test_categorical_rssm.CategoricalRSSMTests) ... ok
test_training_finetune_calibration_and_test_isolation (test_categorical_rssm.CategoricalRSSMTests) ... ok
test_controller_transfer_needs_explicit_opt_in_and_binds_new_calibration (test_controller_calibration.ControllerCalibrationTests) ... ok
test_legacy_calibration_cannot_authorize_new_controller (test_controller_calibration.ControllerCalibrationTests) ... ok
test_local_calibration_rejects_out_of_domain_context_or_commands (test_controller_calibration.ControllerCalibrationTests) ... ok
test_new_calibration_matches_entire_controller_descriptor (test_controller_calibration.ControllerCalibrationTests) ... ok
test_collection_records_controller_and_separates_episode_namespace (test_controller_workflow.ControllerWorkflowTests) ... ok
test_local_collection_records_scope_and_never_commands_outside_it (test_controller_workflow.ControllerWorkflowTests) ... ok
test_task_cli_exposes_explicit_controller_selection (test_controller_workflow.ControllerWorkflowTests) ... ok
test_camera_assignment_changes_posterior (test_embodied_rssm.EmbodiedRSSMTests) ... ok
test_candidate_order_invariance (test_embodied_rssm.EmbodiedRSSMTests) ... ok
test_config_validation_and_no_command_profile (test_embodied_rssm.EmbodiedRSSMTests) ... ok
test_gradients_and_balanced_kl (test_embodied_rssm.EmbodiedRSSMTests) ... ok
test_missing_view_mask_and_nan_rejected (test_embodied_rssm.EmbodiedRSSMTests) ... ok
test_preset_real_parameter_budgets (test_embodied_rssm.EmbodiedRSSMTests) ... ok
test_prior_changes_with_action_but_never_future_observation (test_embodied_rssm.EmbodiedRSSMTests) ... ok
test_reset_and_startup_context (test_embodied_rssm.EmbodiedRSSMTests) ... ok
test_residual_heads_and_full_rgb_mode (test_embodied_rssm.EmbodiedRSSMTests) ... ok
test_sample_seed_does_not_mutate_global_rng (test_embodied_rssm.EmbodiedRSSMTests) ... ok
test_arbitrary_legacy_profiles_are_explicit_offline_only (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_explicit_g1_profile_cannot_override_known_units_or_controller (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_finite_raw_values_must_remain_finite_in_returned_coordinates (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_hand_checked_action_alignment (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_invalid_windows_and_empty_training_split_are_rejected (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_lazy_episode_validation_and_bounded_cache (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_legacy_four_state_explicit_mapping (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_legacy_import_retains_original_multifile_hashes (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_missing_and_misaligned_modalities (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_multiview_episode_cannot_leak_splits (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_optional_events_and_derived_time_do_not_invent_labels (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_origin_requires_identity_and_profile_input_is_snapshotted (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_padding_preserves_real_targets (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_paths_hashes_and_lazy_cache_samples_are_safe (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_profile_serialization_is_validated_and_defensive (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_renamed_originals_with_same_hash_share_sampling_group (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_sample_identity_binds_profile_and_legacy_mapping (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_source_groups_and_episode_sampling_keep_all_windows (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_symlink_escape_and_missing_origin_are_rejected (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_train_only_normalization_and_ancestor_hashes (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_unknown_or_incomplete_ancestor_lineage_fails_closed (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_unproven_controller_and_wrong_known_command_transition_rejected (test_embodied_sequences.EmbodiedSequenceTests) ... ok
test_config_requires_explicit_model_factory_and_planner_limits (test_g1_agent_cli.G1AgentCliTests) ... ok
test_goal_parser_accepts_world_xy_and_optional_heading (test_g1_agent_cli.G1AgentCliTests) ... ok
test_interactive_loop_keeps_session_for_multiple_goals_and_logs_results (test_g1_agent_cli.G1AgentCliTests) ... ok
test_missing_world_model_plugin_fails_before_opening_simulator (test_g1_agent_cli.G1AgentCliTests) ... ok
test_all_ten_missions_have_feasible_routes (test_g1_benchmark.BenchmarkTests) ... ok
test_selection_and_episode_leakage (test_g1_benchmark.BenchmarkTests) ... ok
test_http_retry_and_mujoco_end_to_end (test_g1_coordinator.CoordinatorTests) ... ok
test_schema_and_unsupported (test_g1_coordinator.CoordinatorTests) ... ok
test_single_use_and_uncertain_execution (test_g1_coordinator.CoordinatorTests) ... ok
test_floating_model_has_free_root_and_29_policy_actuators (test_g1_locomotion.G1LocomotionTests) ... ok
test_published_velocity_policy_completes_ten_measured_footsteps (test_g1_locomotion.G1LocomotionTests) ... ok
test_walk_cli_exposes_mujoco_viewer_option (test_g1_locomotion.G1LocomotionTests) ... ok
test_walk_command_prints_machine_readable_success_metrics (test_g1_locomotion.G1LocomotionTests) ... ok
test_external_model_errors_are_sanitized_before_cli_or_logs_see_them (test_g1_locomotion_runtime.G1ContractsTests) ... ok
test_factory_loads_and_prediction_preserves_version_and_state_schema (test_g1_locomotion_runtime.G1ContractsTests) ... ok
test_factory_rejects_model_without_predict_method (test_g1_locomotion_runtime.G1ContractsTests) ... ok
test_factory_rejects_semantically_incompatible_action_schema (test_g1_locomotion_runtime.G1ContractsTests) ... ok
test_prediction_rejects_changed_joint_state_schema (test_g1_locomotion_runtime.G1ContractsTests) ... ok
test_prediction_rejects_wrong_episode_and_nonfinite_values (test_g1_locomotion_runtime.G1ContractsTests) ... ok
test_state_rejects_nonfinite_pose (test_g1_locomotion_runtime.G1ContractsTests) ... ok
test_velocity_action_rejects_unsupported_speed (test_g1_locomotion_runtime.G1ContractsTests) ... ok
test_agent_reobserves_and_allows_another_goal_on_same_session (test_g1_locomotion_runtime.PredictivePlannerTests) ... ok
test_agent_stops_before_planning_when_posture_is_unsafe (test_g1_locomotion_runtime.PredictivePlannerTests) ... ok
test_planner_does_not_select_a_sequence_predicted_to_fall (test_g1_locomotion_runtime.PredictivePlannerTests) ... ok
test_planner_selects_action_using_model_predicted_goal_progress (test_g1_locomotion_runtime.PredictivePlannerTests) ... ok
test_config_loads_all_29_limited_joints_and_renders_pack_camera (test_g1_mujoco.G1MujocoTests) ... ok
test_keyboard_joint_input_drives_physics_with_rate_and_limit_bounds (test_g1_mujoco.G1MujocoTests) ... ok
test_persistent_g1_session_steps_and_remains_open_until_explicit_close (test_g1_mujoco.G1MujocoTests) ... ok
test_simulation_cli_runs_and_rejects_unknown_joint (test_g1_mujoco.G1MujocoTests) ... ok
test_adaptation_and_mixed_duration (test_g1_research.ResearchTests) ... ok
test_feedback_changes_candidate_bounds_and_rejects_stale (test_g1_research.ResearchTests) ... ok
test_learned_prediction_roundtrip_and_planning (test_g1_research.ResearchTests) ... ok
test_live_service_observe_step_reset (test_g1_ros2_integration.RosIntegrationTests) ... ok
test_planner_service_starts_and_rejects_invalid_payload (test_g1_ros2_integration.RosIntegrationTests) ... [ERROR] [1791129847.911402275] [world_model_planner]: Planning failed: ValueError
ok
test_real_simulation_service_shuts_down_cleanly_on_sigint (test_g1_ros2_integration.RosIntegrationTests) ... ok
test_invalid_goal_and_thin_crossing (test_indoor_scene.IndoorTests) ... ok
test_route_around_furniture (test_indoor_scene.IndoorTests) ... ok
test_agent_proposal_is_constrained (test_interchange_layers.InterchangeLayerTests) ... ok
test_trajectory_round_trip (test_interchange_layers.InterchangeLayerTests) ... ok
test_channel_first_camera_shape_is_recorded_as_width_height (test_lerobot_manifest.LeRobotManifestTests) ... ok
test_conversion_finalizer_selects_camera_and_restores_skipped_video (test_lerobot_manifest.LeRobotManifestTests) ... ok
test_artifact_discovery_does_not_follow_symlink_directories (test_monitor_catalog.RunCatalogTests) ... ok
test_dark_uint8_prediction_pixels_are_not_rescaled (test_monitor_catalog.RunCatalogTests) ... ok
test_discovers_run_and_exposes_only_safe_artifacts_and_json (test_monitor_catalog.RunCatalogTests) ... ok
test_extracts_prediction_rgb_as_png_without_executing_npz_payloads (test_monitor_catalog.RunCatalogTests) ... ok
test_named_benchmark_logs_are_independent_event_sources (test_monitor_catalog.RunCatalogTests) ... ok
test_prediction_metadata_reports_available_frames_and_missing_alignment (test_monitor_catalog.RunCatalogTests) ... ok
test_rejects_path_traversal_symlink_escape_and_unknown_run (test_monitor_catalog.RunCatalogTests) ... ok
test_rejects_unsafe_npz_members_and_invalid_frame_indices (test_monitor_catalog.RunCatalogTests) ... ok
test_single_frame_archive_reports_one_frame_and_validates_alignment (test_monitor_catalog.RunCatalogTests) ... ok
test_training_history_list_is_a_supported_json_artifact (test_monitor_catalog.RunCatalogTests) ... ok
test_frame_keeps_rgb_snapshot_and_marks_it_read_only (test_monitor_contracts.MonitorContractTests) ... ok
test_frame_rejects_invalid_identity_time_and_rgb (test_monitor_contracts.MonitorContractTests) ... ok
test_run_descriptor_requires_an_opaque_id_and_path (test_monitor_contracts.MonitorContractTests) ... ok
test_discarding_incomplete_oversize_line_keeps_memory_bounded (test_monitor_events.JsonlEventTailTests) ... ok
test_excessively_nested_json_is_reported_and_next_event_is_read (test_monitor_events.JsonlEventTailTests) ... ok
test_holds_partial_line_until_newline_then_emits_exactly_once (test_monitor_events.JsonlEventTailTests) ... ok
test_nonfinite_json_is_rejected (test_monitor_events.JsonlEventTailTests) ... ok
test_oversized_line_is_discarded_without_blocking_next_event (test_monitor_events.JsonlEventTailTests) ... ok
test_reads_wrapped_and_flat_events_without_losing_raw_fields (test_monitor_events.JsonlEventTailTests) ... ok
test_reports_malformed_rows_and_continues_with_following_events (test_monitor_events.JsonlEventTailTests) ... ok
test_same_inode_same_length_rewrite_preserving_first_row_resets (test_monitor_events.JsonlEventTailTests) ... ok
test_truncation_and_file_replacement_restart_source_generation (test_monitor_events.JsonlEventTailTests) ... ok
test_history_is_bounded_and_counts_overwritten_frames (test_monitor_frame_buffer.LatestFrameBufferTests) ... ok
test_publish_copies_frame_and_concurrent_readers_see_valid_snapshots (test_monitor_frame_buffer.LatestFrameBufferTests) ... ok
test_rejects_non_frame_publication (test_monitor_frame_buffer.LatestFrameBufferTests) ... ok
test_run_and_episode_filters_never_return_an_unrelated_frame (test_monitor_frame_buffer.LatestFrameBufferTests) ... ok
test_g1_cli_monitor_records_manifest_and_completed_goal (test_monitor_integration.MonitorIntegrationTests) ... ok
test_monitor_disconnection_leaves_simulation_controllable (test_monitor_integration.MonitorIntegrationTests) ... ok
test_visual_agent_records_task_lifecycle_and_aligned_feedback (test_monitor_integration.MonitorIntegrationTests) ... ok
test_bad_telemetry_is_reported_without_raising_into_simulation (test_monitor_publisher.FramePublisherTests) ... ok
test_floating_g1_publishes_owner_thread_frames (test_monitor_publisher.FramePublisherTests) ... ok
test_mujoco_monitor_does_not_change_physics_results (test_monitor_publisher.FramePublisherTests) ... ok
test_throttles_frames_and_keeps_latest_frame_when_source_stops (test_monitor_publisher.FramePublisherTests) ... ok
test_catalog_and_incremental_event_cursor (test_monitor_server.MonitorServerTests) ... ok
test_each_reader_receives_rotation_reset (test_monitor_server.MonitorServerTests) ... ok
test_latest_frame_and_metadata_are_from_the_selected_run (test_monitor_server.MonitorServerTests) ... ok
test_non_loopback_binding_is_rejected (test_monitor_server.MonitorServerTests) ... ok
test_root_boundary_origin_and_read_only_routes (test_monitor_server.MonitorServerTests) ... ok
test_secret_fields_are_redacted_in_json_artifacts (test_monitor_server.MonitorServerTests) ... ok
test_sse_emits_a_real_event_snapshot (test_monitor_server.MonitorServerTests) ... ok
test_stopped_source_does_not_leave_an_unfinished_task_running (test_monitor_server.MonitorServerTests) ... ok
test_bad_field_types_do_not_poison_later_valid_events (test_monitor_state.MonitorStateTests) ... ok
test_explicit_session_stop_marks_unfinished_task_incomplete (test_monitor_state.MonitorStateTests) ... ok
test_missing_final_event_remains_incomplete_and_preserves_units (test_monitor_state.MonitorStateTests) ... ok
test_prompt_redaction_preserves_agent_evidence (test_monitor_state.MonitorStateTests) ... ok
test_session_telemetry_does_not_invent_a_new_task (test_monitor_state.MonitorStateTests) ... ok
test_browser_entry_serves_all_local_assets_with_expected_content_types (test_monitor_ui.MonitorUITests) ... ok
test_network_learns_roundtrips_and_plans (test_motion_network.NetworkTests) ... ok
test_resume_matches_continuous_cpu_training (test_motion_network.NetworkTests) ... ok
test_rollout_loss_backpropagates_through_sequence (test_motion_network.NetworkTests) ... ok
test_test_targets_do_not_affect_training_or_normalization (test_motion_network.NetworkTests) ... ok
test_leakage_duplicates_and_discontinuous_states_are_rejected (test_motion_network.SequenceTests) ... ok
test_sequences_do_not_cross_missing_transitions (test_motion_network.SequenceTests) ... ok
test_joint_motion_uses_physics_and_cancel_holds_current_pose (test_mujoco_backend.MujocoTests) ... ok
test_episode_assignment_and_partial_execution_audit (test_planning_network.ExecutionExportTests) ... ok
test_individual_member_constraint_is_screened_and_wrong_rollout_episode_is_rejected (test_planning_network.ExecutionExportTests) ... ok
test_language_recovery_receives_bounded_prediction_evidence (test_planning_network.ExecutionExportTests) ... ok
test_mean_path_is_checked_even_when_member_paths_clear_an_obstacle (test_planning_network.ExecutionExportTests) ... ok
test_latent_resume_and_test_isolation (test_planning_network.PlanningNetworkTests) ... ok
test_latent_training_learns_restores_and_publishes_batch_planning_evidence (test_planning_network.PlanningNetworkTests) ... ok
test_rollout_retains_nonlinear_member_paths_instead_of_merging_every_step (test_planning_network.PlanningNetworkTests) ... ok
test_velocity_targets_reconstruct_terminal_state_when_average_velocity_differs (test_planning_network.PlanningNetworkTests) ... ok
test_a0_consumes_a_finite_nominal_plan_instead_of_replaying_initial_deltas (test_predictive_skill_agent.PredictiveAgentTests) ... ok
test_a2_fixed_commit_and_a0_does_not_call_predictor (test_predictive_skill_agent.PredictiveAgentTests) ... ok
test_no_trusted_action_stale_feedback_and_unknown_skill_fail_closed (test_predictive_skill_agent.PredictiveAgentTests) ... ok
test_partial_failure_accounts_only_actual_steps_and_never_replays (test_predictive_skill_agent.PredictiveAgentTests) ... ok
test_trust_shortens_prefix_and_predicted_success_does_not_complete_task (test_predictive_skill_agent.PredictiveAgentTests) ... ok
test_corrupt_cache_is_preserved_and_replaced_by_verified_download (test_ros2_build_helpers.RosBuildHelperTests) ... ok
test_interrupted_download_never_becomes_a_final_cache_entry (test_ros2_build_helpers.RosBuildHelperTests) ... ok
test_job_limit_reaches_make_and_cmake_and_rejects_zero (test_ros2_build_helpers.RosBuildHelperTests) ... ok
test_rejects_non_project_environment_without_changing_ros_domain (test_ros2_setup.RosSetupTests) ... ok
test_setup_loads_matching_bindings_and_generated_session_interface (test_ros2_setup.RosSetupTests) ... ok
test_a0_r1_and_missing_a3_calibration_rejected_without_output (test_task_agent_cli.TaskAgentCLITests) ... ok
test_help_is_runnable_and_explains_task_config (test_task_agent_cli.TaskAgentCLITests) ... ok
test_illegal_config_is_rejected_before_nonexistent_checkpoint (test_task_agent_cli.TaskAgentCLITests) ... ok
test_unrelated_replanning_is_not_attributed_to_pulse (test_task_agent_cli.TaskAgentCLITests) ... ok
test_a3_rejects_wrong_controller_calibration_before_execution (test_task_convergence.ControllerBindingTests) ... ok
test_bound_controller_success_records_frozen_profile (test_task_convergence.ControllerBindingTests) ... ok
test_controller_change_after_authorization_stops_before_dispatch (test_task_convergence.ControllerBindingTests) ... ok
test_nonlegacy_a3_requires_controller_bound_calibration (test_task_convergence.ControllerBindingTests) ... ok
test_command_in_tolerance_gets_four_real_stable_cycles_before_stall (test_task_convergence.TaskConvergenceTests) ... ok
test_command_progress_does_not_stall_unchanged_measured_joints (test_task_convergence.TaskConvergenceTests) ... ok
test_five_node_dag_waits_for_servo_and_counts_only_executed_holds (test_task_convergence.TaskConvergenceTests) ... ok
test_frozen_measured_pose_fails_after_bounded_settling_and_recovery (test_task_convergence.TaskConvergenceTests) ... ok
test_genuine_command_and_measured_no_progress_still_fails_finitely (test_task_convergence.TaskConvergenceTests) ... ok
test_hold_failure_preserves_completed_node_and_bounded_repair (test_task_convergence.TaskConvergenceTests) ... ok
test_graph_dependencies_and_defensive_target_copy (test_task_graph.TaskGraphTests) ... ok
test_rejects_cycle_unknown_skill_target_and_boolean_budget (test_task_graph.TaskGraphTests) ... ok
test_skills_reach_target_limits_and_hold_has_only_zero_actions (test_task_graph.TaskGraphTests) ... ok
test_arrival_can_reach_an_envelope_boundary_exactly (test_task_joint_planning.JointCandidateTests) ... ok
test_default_controller_retains_legacy_proposals (test_task_joint_planning.JointCandidateTests) ... ok
test_four_step_arrival_does_not_overshoot (test_task_joint_planning.JointCandidateTests) ... ok
test_near_goal_arrives_precisely_then_settles (test_task_joint_planning.JointCandidateTests) ... ok
test_a1_uses_budget_sliced_nominal_endpoint (test_task_joint_planning.TaskJointRankingTests) ... ok
test_command_at_goal_remains_stable_with_misleading_model_head (test_task_joint_planning.TaskJointRankingTests) ... ok
test_command_feasibility_stops_before_overshoot (test_task_joint_planning.TaskJointRankingTests) ... ok
test_default_legacy_ranking_still_uses_model_command_head (test_task_joint_planning.TaskJointRankingTests) ... ok
test_exact_arrival_cannot_bypass_learned_trust (test_task_joint_planning.TaskJointRankingTests) ... ok
test_learned_measured_outcomes_rank_equal_command_arrivals (test_task_joint_planning.TaskJointRankingTests) ... ok
test_outside_calibration_scope_rejects_before_model_query (test_task_joint_planning.TaskJointRankingTests) ... ok
test_progress_eligibility_uses_exact_commands (test_task_joint_planning.TaskJointRankingTests) ... ok
test_untrusted_second_step_cannot_supply_command_progress (test_task_joint_planning.TaskJointRankingTests) ... ok
test_current_task_evidence_excludes_other_explicit_and_legacy_tasks (test_task_monitor.TaskMonitorTests) ... ok
test_fault_latched_task_is_preserved_as_terminal (test_task_monitor.TaskMonitorTests) ... ok
test_task_nodes_and_budget_projected_without_cross_task_inheritance (test_task_monitor.TaskMonitorTests) ... ok
test_actual_reach_and_hold_have_contiguous_time_and_measured_evidence (test_task_mujoco.TaskMuJoCoTests) ... ok
test_pulse_restores_force_and_faulted_session_cannot_move (test_task_mujoco.TaskMuJoCoTests) ... ok
test_rgb_reception_failure_preserves_real_physical_cycle_and_locks (test_task_mujoco.TaskMuJoCoTests) ... ok
test_non_grid_target_is_reached_with_one_capped_real_delta (test_task_planning_integration.TaskPlanningIntegrationTests) ... ok
test_a0_compiles_dependency_order_not_input_array_order (test_task_runtime.TaskRuntimeTests) ... ok
test_a0_frozen_program_no_model_queries (test_task_runtime.TaskRuntimeTests) ... ok
test_a3_prediction_mismatch_stops_before_second_authorized_action (test_task_runtime.TaskRuntimeTests) ... ok
test_budget_and_unknown_receipt_are_terminal_without_retry (test_task_runtime.TaskRuntimeTests) ... ok
test_callback_exception_keeps_actual_progress_and_latches_session (test_task_runtime.TaskRuntimeTests) ... ok
test_canceled_before_first_action (test_task_runtime.TaskRuntimeTests) ... ok
test_config_rejects_a0_recovery_boolean_budget_and_nan (test_task_runtime.TaskRuntimeTests) ... ok
test_controller_exception_after_side_effect_still_records_actual_step (test_task_runtime.TaskRuntimeTests) ... ok
test_cooperative_stop_records_actual_prefix_and_cannot_replay (test_task_runtime.TaskRuntimeTests) ... ok
test_fault_keeps_confirmed_prefix_rgb_state_and_residual_evidence (test_task_runtime.TaskRuntimeTests) ... ok
test_hold_failure_r0_stops_r1_repairs_only_blocked_node (test_task_runtime.TaskRuntimeTests) ... ok
test_k2_real_history_survives_task_node_transition (test_task_runtime.TaskRuntimeTests) ... ok
test_lost_sensor_after_receipt_counts_physics_without_fake_trace (test_task_runtime.TaskRuntimeTests) ... ok
test_new_task_never_overwrites_prior_prediction_artifacts (test_task_runtime.TaskRuntimeTests) ... ok
test_no_trusted_prefix_is_bounded_and_does_not_fake_new_evidence (test_task_runtime.TaskRuntimeTests) ... ok
test_one_time_authorization_and_header_tampering (test_task_runtime.TaskRuntimeTests) ... ok
test_pre_dispatch_observation_failure_reports_prior_steps_and_locks (test_task_runtime.TaskRuntimeTests) ... ok
test_real_five_node_scheduler_with_executed_hold_evidence (test_task_runtime.TaskRuntimeTests) ... ok
test_replanning_alone_is_not_counted_as_verified_recovery (test_task_runtime.TaskRuntimeTests) ... ok
test_reset_servo_transient_is_not_misclassified_as_wrong_command_pose (test_task_runtime.TaskRuntimeTests) ... ok
test_stale_authorization_is_consumed_and_does_not_execute (test_task_runtime.TaskRuntimeTests) ... ok
test_stalled_execution_recovery_is_finite (test_task_runtime.TaskRuntimeTests) ... ok
test_version_change_after_authorization_blocks_first_dispatch (test_task_runtime.TaskRuntimeTests) ... ok
test_version_change_during_prediction_does_not_execute (test_task_runtime.TaskRuntimeTests) ... ok
test_wrong_initial_command_pose_is_rejected_without_actions (test_task_runtime.TaskRuntimeTests) ... ok
test_a1_does_not_query_learned_predictor (test_task_selection.SelectionTests) ... ok
test_a2_full_prefix_a3_cannot_jump_untrusted_step (test_task_selection.SelectionTests) ... ok
test_predictor_cannot_mutate_candidate_or_observation_inputs (test_task_selection.SelectionTests) ... ok
test_wrong_response_version_rejected (test_task_selection.SelectionTests) ... ok
test_hold_only_advances_on_real_contiguous_execution (test_task_verifier.JointVerifierTests) ... ok
test_mismatch_resets_hold_and_cross_episode_is_rejected (test_task_verifier.JointVerifierTests) ... ok
test_reach_requires_actual_and_command_targets (test_task_verifier.JointVerifierTests) ... ok
test_context_loss_ignores_burnin_padding_as_targets (test_temporal_world.TemporalTrainingTests) ... ok
test_context_train_finetune_calibrate_and_evaluate_use_identical_protocol (test_temporal_world.TemporalTrainingTests) ... ok
test_context_is_hashed_and_legacy_zero_context_remains_loadable (test_temporal_world.TemporalWorldTests) ... ok
test_dataset_padding_resets_without_losing_startup_target_windows (test_temporal_world.TemporalWorldTests) ... ok
test_future_targets_cannot_change_imagined_predictions (test_temporal_world.TemporalWorldTests) ... ok
test_history_changes_prediction_without_candidate_branch_side_effects (test_temporal_world.TemporalWorldTests) ... ok
test_padding_and_unpadded_posterior_match_even_with_nonzero_action_mean (test_temporal_world.TemporalWorldTests) ... ok
test_temporal_model_refuses_silent_single_frame_downgrade_and_bad_history (test_temporal_world.TemporalWorldTests) ... ok
test_video_provider_requires_aligned_real_history_actions (test_temporal_world.TemporalWorldTests) ... ok
test_sample_train_validate_and_plan (test_training_loop.TrainingLoopTests) ... {"phase": "collection", "episode": 0, "split": "train", "transitions_written": 3, "output": "/tmp/tmpxldt25tl/transitions.jsonl"}
{"phase": "collection", "episode": 1, "split": "train", "transitions_written": 6, "output": "/tmp/tmpxldt25tl/transitions.jsonl"}
{"phase": "collection", "episode": 2, "split": "train", "transitions_written": 9, "output": "/tmp/tmpxldt25tl/transitions.jsonl"}
{"phase": "collection", "episode": 3, "split": "validation", "transitions_written": 12, "output": "/tmp/tmpxldt25tl/transitions.jsonl"}
{"phase": "collection", "episode": 4, "split": "test", "transitions_written": 15, "output": "/tmp/tmpxldt25tl/transitions.jsonl"}
{"phase": "evaluation", "episode": 0, "method": "direct", "success": true, "control_steps": 1, "final_error_rad": 2.2248085318476996e-10}
{"phase": "evaluation", "episode": 0, "method": "model_planner", "success": true, "control_steps": 4, "final_error_rad": 0.005199366324773261}
{"phase": "evaluation", "episode": 1, "method": "direct", "success": true, "control_steps": 1, "final_error_rad": 7.619263553415578e-10}
{"phase": "evaluation", "episode": 1, "method": "model_planner", "success": true, "control_steps": 3, "final_error_rad": 0.024175208965242112}
ok
test_failed_batch_closes_progress_without_reporting_completed_epochs (test_training_progress.TrainingProgressTests) ... ok
test_finetune_uses_the_same_progress_without_training_another_agent (test_training_progress.TrainingProgressTests) ... ok
test_legacy_string_source_remains_trainable_with_progress_on_or_off (test_training_progress.TrainingProgressTests) ... ok
test_missing_dataset_fails_instead_of_producing_a_model (test_training_progress.TrainingProgressTests) ... ok
test_motion_progress_preserves_json_events_and_resume_equivalence (test_training_progress.TrainingProgressTests) ... ok
test_no_progress_is_silent_and_does_not_change_visual_weights (test_training_progress.TrainingProgressTests) ... ok
test_visual_cli_reports_epochs_batches_losses_and_eta_only_on_stderr (test_training_progress.TrainingProgressTests) ... ok
test_v21_alignment_provenance_and_original_test_are_preserved (test_visual_lerobot.VisualLeRobotTests) ... ok
test_collection_is_four_way_disjoint_and_opens_real_camera_windows (test_visual_workcell.VisualWorkcellTests) ... ok
test_partial_physics_failure_cancels_target_and_latches_idle_until_reset (test_visual_workcell.VisualWorkcellTests) ... ok
test_safe_joint_delta_moves_rendered_g1_and_rejects_stale_or_out_of_bounds (test_visual_workcell.VisualWorkcellTests) ... ok
test_viewer_eof_keeps_rendering_without_repeated_input_prompts (test_visual_workcell.VisualWorkcellTests) ... ok
test_episode_quantile_not_frame_quantile_and_model_binding (test_visual_world.HorizonCalibrationTests) ... ok
test_duplicate_episode_or_content_and_wrong_alignment_are_rejected (test_visual_world.VisualDataTests) ... ok
test_visual_dataset_capability_exists (test_visual_world.VisualDataTests) ... ok
test_windows_never_cross_episode_and_stats_ignore_test (test_visual_world.VisualDataTests) ... ok
test_finetune_retains_ancestor_exclusions_across_repartitioned_manifests (test_visual_world.VisualNetworkTests) ... ok
test_imagination_depends_on_actions_without_future_observations (test_visual_world.VisualNetworkTests) ... ok
test_training_finetune_freeze_roundtrip_and_test_isolation (test_visual_world.VisualNetworkTests) ... ok
test_zero_residual_preserves_background_instead_of_relearning_static_world (test_visual_world.VisualNetworkTests) ... ok
test_all_worlds_compile_and_settle (test_workcell.WorkcellTests) ... ok
test_seed_and_unknown_task (test_workcell.WorkcellTests) ... ok
test_actual_five_node_task_keeps_all_measured_hold_predicates (test_workcell_controller.WorkcellControllerTests) ... ok
test_b_target_tracks_with_gravity_and_unchanged_command (test_workcell_controller.WorkcellControllerTests) ... ok
test_feedforward_and_pd_share_the_actuator_torque_limit (test_workcell_controller.WorkcellControllerTests) ... ok
test_legacy_default_preserves_known_tracking_bias (test_workcell_controller.WorkcellControllerTests) ... ok
test_profile_binds_physical_parameters_and_reset_clears_stale_feedforward (test_workcell_controller.WorkcellControllerTests) ... ok

----------------------------------------------------------------------
Ran 319 tests in 23.111s

OK
```
