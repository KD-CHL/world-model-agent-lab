# M2.1 Stack-block recovery implementation ledger

Base: `eac61bb0d6234f5aee5b2f779049dc8d23630ba4`
Branch: `codex/stack-block-m2-recovery`
Plan source: GitHub issue #1 and `docs/superpowers/specs/2026-10-12-stack-block-experiment-design.md`.

## Evidence before implementation

- Existing repeated-place recovery failed on the disturbed pilot; it did not correct the measured placement offset.
- Earlier affine/homography localization probes are rejected for control; their documented validation detection/error was inadequate.
- The frozen RGB-XY model's independent stationary-tabletop report has 99.2% frame detection and 3.55 mm mean planar error, with a 14.72 mm episode-max conformal radius at nominal 80% coverage. This is not evidence for moving-object localization.
- Existing post-lift failure labels came from verifier-only truth. Recovery triggers must instead use the measured gripper contact history; truth remains analysis-only.

Ruling: choose RGB-XY re-localization plus one bounded regrasp after measured bilateral-contact loss. The fixed-known-pose option is not valid for the disturbed/drop domain because the manipulated block may move and reset truth/seed are outside the agent observation. Use only the frozen RGB-XY model after a repeated-image stationarity gate; fail closed on missing calibration, excessive radius, invalid workspace, actuator fault, or nonconvergence. This ruling costs a potential failed recovery if the learned camera mapping does not cover settled drop poses; the acceptance replay will test that before data collection.

## Tasks

- [ ] Verify the frozen localizer on previously inspected development drop trajectories without using those samples as final-test evidence.
- [ ] Add a tested, actuator-aware Cartesian servo that uses robot observations and bounded MuJoCo steps; do not reuse the removed open-loop IK helper.
- [ ] Add a contact-loss-only recovery trigger and one-attempt regrasp skill using the calibrated RGB-XY estimate; keep verifier truth out of action selection.
- [ ] Integrate recovery with the baseline runner and synchronized collector; preserve every failed/declined/interrupted episode and action receipt.
- [ ] Run deterministic recovery and unrecoverable development replays, report trigger precision, recovery rate, false triggers, costs, and failure causes.
- [ ] Freeze the policy only if replay gates pass; otherwise record the limiting evidence and continue improving within M2.1.

## Evidence revision — release failure, not transport contact loss

The initial trigger hypothesis above is superseded by the event-aligned drop replay. In both disturbed development drops (seeds 9200 and 9203), bilateral gripper contact persists through `move_to_stack_target`; it ends only after the commanded `open` phase begins in `place_block` (steps 1283 and 1285). The block then settles on the table at approximately (0.275, 0.419) and (0.277, 0.421) m, while the configured stack target is (0.20, 0.42) m. The recovery pilot seed 7209 settles at (0.273, 0.419) m. Therefore contact loss is expected at release and is not a valid recovery trigger. The reproducible failure is a horizontal placement miss of about 7–8 cm.

The frozen RGB-XY model was run in the declared `wmal` environment on final stationary frames from both development drops. It estimated (0.272–0.273, 0.418) m and (0.274–0.275, 0.420) m, respectively, with per-frame planar errors of 2.6–3.2 mm. Its 14.725 mm calibrated radius leaves the x-axis confidence interval just outside the 55 mm target tolerance on these examples. This supports a candidate **post-release, observation-only, uncertainty-aware miss trigger**, not a general recovery claim. A miss is actionable only after three separated stationary visible frames agree, the entire calibrated confidence interval lies outside the target region, the gripper is open and fault-free, and the scene/model profile is allow-listed. Ambiguous or unobservable outcomes fail closed.

The next implementation therefore combines (1) an RGB-XY miss assessment using only `RobotObservation`, the fixed task target from the task contract, and the frozen checkpoint/calibration; (2) an actuator-aware measured-pose servo for one bounded regrasp/place attempt; and (3) replay evaluation that records all declines and failures. Verifier truth remains offline scoring only. Do not freeze the recovery policy until this gate is exercised across development drop seeds with no false triggers on successful stacks or failed grasps.

## Revised tasks

- [ ] Add tests for an uncertainty-aware three-frame post-release miss trigger, false-trigger declines, invalid calibration, hidden-truth exclusion, and bounded attempts.
- [ ] Implement the observed-only miss assessor and wire it to the frozen RGB-XY artifact; verify the exact development drops are detected and successful stacks are not.
- [ ] Implement an actuator-aware closed-loop Cartesian servo with measured end-effector feedback and hard joint/workspace/step limits; test the moved-object grasp and target correction in MuJoCo.
- [ ] Integrate one regrasp/place recovery attempt into the episode runner/collector; log the model/calibration hashes, all observation steps, receipts, trigger/decline reason, cost, and offline-only verifier result.
- [ ] Replay a development matrix covering drop-after-release, already-stacked, failed-grasp, missing detection, and controller fault; freeze only if the stated gates pass.

## 2026-10-13 execution update — actuator bias and replay gate

The declared `wmal` environment was used for all new Python, test, and MuJoCo work in this update.

- Added the RGB-XY post-release assessor, bounded one-shot recovery, and runner/collector logging. The assessor fired on the seed-9200 and seed-106 off-target release episodes. The full runner also checks successful task outcomes; the recovery policy was not used to alter its frozen checkpoint.
- Added the measured-pose Cartesian servo and session Jacobian, then fixed its first MuJoCo failure. The cause was a nonzero difference between static joint position and actuator hold target. Cartesian joint increments now preserve the current actuator target offset. A reconstructed seed-9200 snapshot passed a 50 mm vertical-motion replay in 112 control steps (5.93 mm final position error, 0.008 rad orientation error).
- Added tests for finite left-arm actuator target access and for the servo preserving a nonzero target bias. Recovery suite: 14/14 pass; baseline runner suite: 10/10 pass; changed Python files compile.
- Fresh no-model M1 baseline: nominal seed 9200, 1/1; jittered development seeds 0-9, 10/10; jittered holdout seeds 100-119, 19/20 (95%). Seed 106 is the sole holdout failure (`object_dropped_after_lift`), meeting the spec's 90% pilot threshold for this family while identifying a residual release failure.
- The current M2 policy did not pass its end-to-end gate. Seed 9200 and seed 106 both triggered correctly, cleared above the drop, and failed descent with 45.3 mm and 42.6 mm pose errors. In the seed-9200 diagnostic replay the left elbow-yaw collision body hit the table. Separate diagnostic regrasp attempts found that bilateral contact can involve the support block or fail to carry the dropped block upward. These diagnostic trials are not counted as recovery successes.

Do not mark the policy frozen and do not treat attempted-recovery logs as successful recovery data. Continue only after a safe reachable grasp/placement path is demonstrated and grasp lift is confirmed through robot-visible evidence. Keep the seed-9200/seed-106 trigger results as development evidence. A full trigger false-positive/fault matrix and fresh recovery-rich dataset remain outstanding.

### Follow-up diagnostic — lateral push is not a fallback yet

A development-only push from the seed-9200 drop pose shifted the manipulated block x from 0.275 m to 0.255 m and shifted the free support block x from 0.193 m to 0.175 m. Relative block separation remained about 0.080 m and stacking still failed. The support is not fixed in this workcell. Also treat seed 106 as consumed for M2 recovery development; reserve a new paired seed list for final A0-A3 testing.

### Follow-up probe — pre-release alignment remains exploratory

The release-envelope matrix is an in-memory prototype using the frozen RGB-XY reset estimate and measured grasp pose to servo directly toward a computed stack target before opening and settling. Seeds 9200, 9201, and 9203 passed the offline stack verifier; seed 9202 failed during grasp. The earlier 9200-9211 same-family baseline records identify 9200 and 9203 as post-lift drops and 9202 as a grasp-contact failure. This is a promising development-only slice, not a frozen policy or a budget-matched comparison: the prototype used extra servo steps and verifier truth was used only offline. Seed 9203 lost bilateral finger contact during descent, so success in that episode does not provide robot-visible proof of support transfer. Do not count these episodes as M2 recovery successes or world-model gains. Preserve them for candidate-path development. All commands in this update used the declared `wmal` environment.

### Follow-up implementation — separate pre-release alignment policy

Implemented `rgbxy_pre_release_align_dev_v1` as a development-only baseline-runner policy. It keeps the fixed-waypoint baseline selectable, estimates an object-to-tool offset from the frozen reset RGB prediction and measured bilateral grasp pose, runs bounded Cartesian feedback, and only opens inside the logged release envelope. It does not use world-model output or verifier truth for action selection.

Paired on disturbed seeds 9200-9203 with the same configuration, 900-step skill cap, and 4096-step episode cap: fixed waypoint 1/4; alignment 3/4. Seeds 9200 and 9203 changed from post-lift drops to verifier-confirmed stacks; seed 9201 stayed successful; seed 9202 retained its grasp-contact failure. Actual episode steps were 1536/1536/1024/1536 versus 1696/1696/1024/1696, so the two rescued trials cost 160 extra steps each. This is a four-seed development result with equal maximum budgets, not equal realized action cost or a general task-benefit result.

The release envelope can accept a servo iteration-limit residual (XY <=20 mm, vertical gap 0-60 mm, orientation <=0.15 rad). Seed 9203 passed despite losing right-finger contact before release. Do not treat that as support-transfer verification or hardware safety evidence. Add an observation-based release-acceptance gate and use fresh seeds before freezing or starting A0-A3. Artifacts are under `runs/stack_block_pre_release_align_paired_baseline_20261013_v1` and `runs/stack_block_rgbxy_pre_release_align_dev_20261013_v1`. Targeted tests in `wmal`: 18 recovery/alignment, 8 skill-contract, 11 runner; Python compilation and `git diff --check` passed.

## 2026-10-13 — contact-guard paired evidence

The new `rgbxy_pre_release_align_contact_guard_dev_v2` policy was evaluated against fixed waypoint on the same disturbed seeds 9300–9329. Outcomes were baseline 21/30, candidate 22/30, with 5 baseline-only wins, 6 candidate-only wins, 16 shared wins and 3 shared failures. Mean control steps were 1484.8 vs 1560.9 (+76.1 for the candidate); both runs disabled the world model. Treat this as a development result only, not a task-benefit claim or an equal-realized-effort comparison.

A stationary hold after the five contact-loss aborts produced a mean 1.93 mm maximum RGB-XY prediction spread but 41.60 mm mean planar truth error; the block moved at most 1.07 mm on average. A low-spread estimate is not a calibrated support observation in the elevated/occluded domain. Offline verifier truth showed stable support in these five cases, but it was not available to policy code.

Ruling: retain fail-closed bilateral-contact loss handling; the tabletop calibration radius cannot authorize release in the stacking domain. This may decline a release when an object is already supported; the cost if wrong is a missed success. Any support-transfer decision needs a separately trained/calibrated episode-disjoint observation model or a different robot-visible cue.

### Follow-up — strict release gate and independent paired evaluation

The end-of-episode score is now `stable_stack AND target_block_released`: the gripper must be fault-free, fully open and at target, with no `block_0`-to-finger contact. The bilateral gripper flags report contact with any workcell object; a support-block touch alone does not invalidate release. Truth is used here only by the offline scorer. The synchronized result/manifest schema records this definition as `wmal.stack_block_results.v2`.

The first fresh v3 validation on seeds 9500–9529 scored fixed waypoint 17/30 and strict alignment 1/30 under the initial overly restrictive release score. Instrumentation found that the servo's 24-iteration default stopped at 288 steps despite a 512-step allowance. Candidate v4 uses 42 iterations (504 possible steps). On development seeds 9400–9429, with the corrected object-specific release score, fixed waypoint scored 16/30 and v4 scored 22/30 (12 both, 10 candidate-only, 4 baseline-only, 4 neither). These seeds are development-only.

Untouched validation on seeds 9600–9659 gave fixed waypoint 42/60 and v4 39/60. Paired outcomes: 32 both succeed, 10 baseline-only, 7 candidate-only, 11 both fail. Two-sided exact McNemar p=0.629. Mean episode control steps were 1450.7 versus 1535.1 (+84.4 candidate). Candidate failures: 10 missing bilateral grasp contacts, 10 contact-loss declines, 1 target-release failure. Both arms used no world model. Keep the fixed-waypoint policy as the current reference; v4 is not frozen and does not pass a task-benefit gate. M2.1 remains open pending a robot-observable support-transfer signal and a candidate that improves fresh paired task outcomes. No A0–A3, ROS 2 closed-loop, or sim-to-real claim follows from these data.
## 2026-10-13 - CUDA visual-world-model development run

Used the 50-episode synchronized hold-data manifest with episode splits 30/5/5/10 (train/validation/calibration/test); hold episodes were 5/1/1/0 across those splits. Training in the existing `wmal` environment used RTX 5060 Ti / CUDA 12.8, 3 members, batch size 512, 5 epochs. The report confirms `device=cuda`; best validation loss was 0.2964 (from 0.5122 at epoch 1). Checkpoint: `runs/stack_block_contact_loss_wm_dev_20261013_v1/best.pt`, SHA-256 `9f2c00b4e66a37d588ac7713c3ca211c571659240d0cf8d9c4891d9f4adad1d3`.

Calibration used alpha 0.2 and five calibration episodes; test used 10 disjoint episodes, 530 windows at stride 32. State RMSE beat persistence by 7.2-19.3%, while event Brier was worse at all horizons. Aggregate normalized object RMSE beat persistence by 18.1-29.2%, but object episode-simultaneous interval coverage was just 10-20%; there were no support-hold examples in test. Event/object ensemble-spread error Spearman was 0.841/0.478. Treat these as exploratory prediction metrics, not evidence that uncertainty is safe for recovery or that the model improves the task.

The Windows evaluator repeatedly crashed inside `numpy.corrcoef` during rank-correlation aggregation. Replaced that call with an equivalent scalar Pearson correlation over average ranks using `math.fsum`; the full CUDA test evaluation then completed. Report: `runs/stack_block_contact_loss_wm_dev_20261013_v1/test_evaluation.json`. Rollouts still disable the world model. A larger hold-rich calibration/test dataset and paired same-budget closed-loop A0-A3 evaluation remain open; this checkpoint is not frozen.

### Follow-up — larger contact-transfer dataset for M2/M3

Added deterministic shuffled episode assignment to the collector, preserving 60/10/10/20 train/validation/calibration/test counts while avoiding seed-order separation. Test coverage verifies count, determinism, and interleaving. The new disturbed dataset covers seeds 9700–9899 and has 120/20/20/40 episodes, 187,788/29,532/31,772/63,012 transitions, 128×128 RGB, 70 state channels, 14 action channels, and 39 object-truth labels at 20 ms intervals. Manifest hash: `ce73f1837f1e7dfd68a761f196514764bdfb0e9f381fb1e3b50e0982b6e69abf`.

By split, successful episodes are 82/10/14/28, stable-stack episodes 99/15/17/35, and post-contact-loss hold episodes 18/5/3/7. The untouched test set contains 7 held-out contact-loss holds. Candidate v5 GPU training began in the existing `wmal` environment; `tqdm` is absent there, so the trainer's no-progress mode wrote epoch metrics/checkpoints to `history.json` and `best.pt`.

### Follow-up — compact CUDA candidates and final test gate

Completed v5/v6 compact runs with three members, six epochs, batch 128, horizon four, and stride 32. V5 (seed 2725) best validation loss was 1.00942; v6 (seed 2726, spatial object encoder) was 1.04301. The earlier stride-eight v5 exploration stopped after two of twelve epochs and was not used for selection. On 20 validation episodes/1851 stride-16 windows, v6's manipulated-block position error was slightly lower than v5's but both were far worse than persistence, and both lost to persistence on event Brier. The seed and encoder changed together, so this is not a controlled architecture ablation.

V6 was calibrated at alpha 0.2 on 20 separate calibration episodes, then evaluated once on 40 sealed test episodes/3952 stride-16 windows using CUDA. Its `block_0` position RMSE was 15.47–15.67 mm at horizons 1/4 versus persistence 0.56–1.86 mm. Event Brier was 0.04011/0.03007 versus 0.00550/0.00430. Robot-state RMSE did beat persistence, but episode-simultaneous object coverage across horizons was only 10%, 12.5%, 12.5%, and 55%; state coverage was 80%, 65%, 95%, and 100% with very wide scale-normalized intervals. The 80% nominal uncertainty gate and manipulated-object/event gates fail. Model `visual-526b50bc33cf93628371` is an exploratory negative result, not frozen for control.

The dataset manifest hash is `ce73f1837f1e7dfd68a761f196514764bdfb0e9f381fb1e3b50e0982b6e69abf`; v6 checkpoint hash is `3faf3c75fff1981988f146a9b0de360dfe3b203472274f472c9829da61279c67`, and its calibrated test report hash is `4f31947f3d4d37b9ad20f9ce750941927d92ea2496ee3d2b99d851dbb721baa0`. Full candidate/config/calibrator hashes, metric arrays, and artifact paths are recorded in `docs/20_stack_block_experiment_progress.md`. This 40-episode test split is consumed; use a fresh one for any future model iteration. M2.1 remains open after the strict-release policy lost 39/60 versus 42/60 on fresh paired seeds. Do not launch A2/A3 or make a world-model benefit claim. M3/M4 remain open pending an effective observation-based recovery policy, a credible predictor with calibrated error, new paired testing, ROS 2 closed-loop, and sim-to-real evidence.

### Follow-up — current-score M1 baseline gate and code checks

Replayed the fixed-waypoint baseline on 40 fresh jittered seeds 30000–30039 under the v2 `stable_stack AND target_block_released` score. All 40 succeeded with both predicates true, zero world-model loads, and 1536 control steps each. This re-establishes the 90% stage-one pilot floor for this initial-state family under the current score; it does not override the 42/60 disturbed-family baseline. The run is `runs/stack_block_m1_v2_fresh_jitter_30000_30039_20261013_v1`; manifest/results hashes are `23db31a137f265fd014a1ddacdd2269f619ad55210dc4ff98b8d7c6734cf9558` and `5df4a536899c58b68a982617d21b5570eb568fe7529b10538ddf61f6347262c7`.

Verification in `wmal`: 53 unittest files, 329 tests, 322 passes, 7 skips, zero failed files; `git diff --check` passed. The monolithic Windows test process terminated on a duplicate OpenMP runtime after direct `python.exe` first exposed a missing `wmal` DLL search path and GBK decoding of a UTF-8 config. Per-file isolation with that same environment's `Library\bin` on `PATH` and `PYTHONUTF8=1` passed without an unsafe duplicate-runtime override. Logs are under `runs/stack_block_m2_recovery_unittest_by_file_20261013`.
