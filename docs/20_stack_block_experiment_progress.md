# Stack-block world-model experiment progress

Last updated: 2026-10-13

This report records the first executable path from the gripper proxy through skills, synchronized visual data, training, calibration, and held-out prediction evaluation. It is a small simulation study, not evidence of real-robot transfer.

## Stage 1: no-world-model manipulation baseline

The task is stacking `block_0` on fixed `block_1` using the synthetic parallel-jaw proxy and the fixed `g1-left-stack-v1` waypoint skills. The verifier uses scene truth and only accepts a stable, upright support contact. After a stable place is verified, the current baseline stops; it does not execute the old withdrawal profile because that motion was observed to collide with the stack and can tip it.

| Split / run | Seeds | Result |
| --- | ---: | ---: |
| Development | 0–9 | 10/10 |
| Independent run | 100–119 | 19/20 |
| Additional independent run | 1000–1059 | 60/60 |
| Combined held-out episodes | 100–119 and 1000–1059 | 79/80 (98.75%; Wilson 95% CI 93.25%–99.78%) |

Seed 106 is the observed failure: the manipulated block tips during placement and the independent verifier rejects the terminal state. These episodes use one narrow jittered initial-state family; the result does not establish robustness to larger object/fixture perturbations, friction changes, or a Dex1 hand. The ≥90% proposed gate is met for this bounded proxy task, while retreat remains unaccepted.

## Stage 2: synchronized data and frozen model

Dataset: `runs/stack_block_visual_jittered_40_v2/manifest.json`, 41 episodes total. The 40 generated episodes are split by episode (24 train, 6 validation, 6 calibration, 4 test); one separately replayed seed-106 failure is added to test. Each episode aligns 64×64 RGB, 70-dimensional measured robot state, 14-dimensional commanded action, event labels, object truth, simulation timestamps, and step IDs at a 20 ms control period. The original v2 model did not consume `object_truth`. The derived `manifest_v3.json` explicitly names all 39 object-state dimensions; the truth is used only as a future prediction target and held-out label, never as an observation input.

Training used [`stack_block_visual_v2.json`](../configs/training/stack_block_visual_v2.json), CUDA, 3 ensemble members, 5 epochs, and the existing episode-disjoint trainer. Model version: `visual-23507a4e181f6f130653`.

| Artifact | SHA-256 |
| --- | --- |
| Dataset manifest | `bd2882e1648aa4d90173cd598abce2ec84888eb4e1e8ffea6f759fea882c81fb` |
| `best.pt` | `e38889f40a77e356c89e1636f11482b1d4dc22786d556c890c26b3748c9b9707` |
| Calibration | `480bcfb3022f5c6650d5d43f568fbdfa1a1d086e5a5fa1dfff0cfc6cb16a8eb6` |
| Test report | `99b4327629eb5c731d016ae63a27616e3df52cfb59e2dc50cd5730c6fff2febc` |

## Stage 2 evaluation: prediction and uncertainty

Test uses 5 independent episodes (four successes and the seed-106 failure) and 240 sampled windows, one every 32 control steps. The model predicts 4-step futures. A persistence baseline repeats the current image/state/event.

| Metric | Model | Persistence |
| --- | --- | --- |
| RGB MSE, horizon 1 | 0.0000390 | 0.0000287 |
| RGB MSE, horizon 4 | 0.0001534 | 0.0001431 |
| Changed-region RGB MSE, horizon 1 | 0.008246 | 0.008248 |
| Changed-region RGB MSE, horizon 4 | 0.012318 | 0.012319 |
| Robot-state RMSE, horizon 1 | 0.0847 | 0.0924 |
| Robot-state RMSE, horizon 4 | 0.1560 | 0.2040 |
| Event Brier score, horizon 1 | 0.0315 | 0.0000 |
| Event Brier score, horizon 4 | 0.0256 | 0.0028 |

The model improves robot-state prediction but does not beat the simple persistence predictor on pixels or event Brier score. Its calibrated state intervals cover all sampled test points, but mean half-width grows from 5.45 to 8.61 state standard deviations, so the intervals are too broad to guide useful risk-sensitive control. Ensemble event spread and absolute event error have Spearman correlation 0.70 on this small test set; that signal needs more independent failures before use as a safety gate. High event accuracy is not meaningful by itself because most sampled frames are negative or unchanged.

**Decision:** do not claim a world-model task benefit or present A0–A3 as a successful ablation yet. Improve the object-state/event model and collect distinct failures plus recovery behavior, then freeze the next model before paired A0–A3 runs. The current 64×64 camera is centered on the stack area; larger-resolution runs can be considered if small-object prediction remains limiting.

## Stage 2 extension: named object-state prediction target

`configs/training/stack_block_visual_v3.json` adds a supervised 39-dimensional object-state head. Training completed on CUDA with 3 ensemble members and 5 epochs; the selected checkpoint is model version `visual-b6d5dac956aa00a89d64`. Validation object-state loss fell from 0.332 to 0.215 normalized MSE. Test evaluation used the same five held-out episodes and 240 windows (stride 32), with no model selection or calibration on test data.

| Object-state metric | Model | Persistence |
| --- | ---: | ---: |
| Normalized RMSE, horizon 1 | 1.479 | 0.802 |
| Normalized RMSE, horizon 4 | 1.472 | 0.898 |
| `block_0` position RMSE, horizon 1 | 0.456 | 0.021 |
| `block_0` position RMSE, horizon 4 | 0.460 | 0.081 |

Object ensemble spread versus normalized absolute error has Spearman correlation 0.497 across these windows. The object head has not demonstrated useful prediction: it loses to persistence overall and badly misses the manipulated block's position. This experiment keeps the leakage boundary explicit, but it does not pass the predictive-quality gate. The small, mostly successful, narrowly jittered dataset and just five held-out episodes are insufficient for claims about recovery or robustness. Do not start A0–A3 based on this model.

Reproduction (from the repository root, with the `tf-agent` environment used for PyTorch/CUDA):

```powershell
$env:PYTHONPATH = (Get-Location).Path + '\\src'
python -c "from wmal.training.visual_cli import main; main(['train','--manifest','runs/stack_block_visual_jittered_40_v2/manifest_v3.json','--output','runs/stack_block_world_model_v3','--config','configs/training/stack_block_visual_v3.json','--no-progress'])"
python -c "from wmal.training.visual_cli import main; main(['evaluate','--manifest','runs/stack_block_visual_jittered_40_v2/manifest_v3.json','--checkpoint','runs/stack_block_world_model_v3/best.pt','--output','runs/stack_block_world_model_v3/test_report.json','--horizon','4','--window-stride','32','--device','cuda'])"
```

`runs/` is gitignored and contains local data/checkpoints. Recorded SHA-256 values: manifest `1ff31ed6e4c0825d978dd5dd95025623d4bc2b339101b53c940c81a42e605889`; checkpoint `52170194288d43f82180884b2c04a92c062976fec3b426a3bef812cb66ffbcd7`; test report `7b5a0cc683b0fbfb03fbf47cbb01a2c202e23f1d14bd486f3753dd41dc2acb86`; training report `cb6c516d44560a2234cdd4460a886866a1e79526ad96ea8854efff9990871ca8`.

## Stages 3 and 4

- A0–A3 paired, equal-budget multi-initial-state policy experiments: not run. The prediction baseline has not shown a useful image/event advantage, and withdrawal is not accepted.
- ROS 2 closed-loop and sim-to-real: not run. Current evidence is MuJoCo-only and uses a synthetic gripper proxy.

## Stage 2 extension: RGB block localization probe

The zoom-camera dataset uses a fixed 64×64 RGB view (`workcell_overview`, 32° vertical FOV). To test whether task-relevant position can be recovered directly from pixels, `wmal.training.visual_cli fit-stack-localizer` fits a red-component-centroid-to-world-XY affine map from the **first frame of train episodes only**. The serialized artifact records the manifest hash, camera identity, train episode IDs, and per-episode hashes. `evaluate-stack-localizer` defaults to the disjoint validation split and refuses an evaluation/training overlap.

On the six validation episodes of `stack_block_visual_zoomed_disturbed_60_v1`, position RMSE was 1.74 mm in x and 3.29 mm in y; mean planar error was 3.45 mm and worst episode error was 5.09 mm. This supports using the localizer as a controlled, color-specific observation baseline for this fixed camera and scene. It is not a general object detector, does not estimate height/orientation/occlusion, and has not been used to choose a model or inspect final-test data. The localizer and model are not yet integrated into an A0–A3 manipulation runner.

The `tf-agent` environment was used for this probe because `numpy.linalg.lstsq` terminates the process in the separate `wmal` environment on this host. The reproduction commands are:

```powershell
$env:PYTHONPATH = (Get-Location).Path + '\\src'
python -m wmal.training.visual_cli fit-stack-localizer --manifest runs/stack_block_visual_zoomed_disturbed_60_v1/manifest.json --output runs/stack_block_rgb_localizer_v1.json
python -m wmal.training.visual_cli evaluate-stack-localizer --manifest runs/stack_block_visual_zoomed_disturbed_60_v1/manifest.json --localizer runs/stack_block_rgb_localizer_v1.json --output runs/stack_block_rgb_localizer_validation_v1.json --split validation
```

## Recovery-policy collection pilot

The collector now accepts `--recovery-policy repeat_place_once`. When `place_block` returns `success_unverified`, it records the failure evidence and permits exactly one further bounded `place_block` skill in the same episode. Each event labels the recovery trigger separately from the recovery action; the episode row records attempt count and whether the retry actually recovered task success. Other failures are kept as failures and are not silently retried.

Pilot: `stack_block_recovery_pilot_12`, seeds 7200–7211, disturbed profile, 64×64 RGB. Result: 9/12 task successes; one of the three failures triggered the placement retry, and the retry did not recover it. Verifier-only final truth showed the block about 7.3 cm from the support center; repeating the same waypoint could not correct that offset. This is a logged negative recovery attempt, not an accepted recovery behavior. The pilot validates trajectory/event labeling and exposed the need for a visual re-localization plus an adjusted recovery action. It is development evidence only and must not be mixed into final-test or used to claim recovery efficacy.

### Trajectory localization follow-up

The first localizer only fit each train episode's initial frame; on the failed placement image for seed 7209 it missed y by 3.6 cm, so it cannot be used there. A second fit uses one visible RGB/object-label sample every 32 control steps from train episodes and robust Huber reweighting. On validation trajectories (225 sampled frames across 6 held-out episodes), x/y RMSE was 8.2/21.8 mm, maximum planar error 42.0 mm, and the red target was not detected in 37 frames (14.1%). On seed 7209's final placement image it estimated within 5.2 mm, but that single development example does not cure the broader error or detector gaps. **Do not dispatch vision-guided regrasp yet.** The next recovery controller must either carry calibrated localization uncertainty and safely decline ambiguous detections, or improve sensing/localization first. The test split was not read for this follow-up.

A table-height/rest-only fit raised detection to 133/139 validation candidates (95.7%), and x RMSE was 6.5 mm; y RMSE remained 24.0 mm with a 48.7 mm maximum planar error. It localized the failed seed-7209 block within 2.4 mm after it settled, but missed by 39 mm while the block was being transported. A short MuJoCo kinematic-IK prototype reached its requested pose in a copied `MjData`, yet executing its joint targets through the real position-actuator session settled 17–37 mm away or drifted farther after repeated replanning. The pure kinematic helper was removed rather than treated as a working controller. Recovery still needs actuator-aware feedback or a safe calibrated waypoint family; the current visual estimate is only suitable for supervised analysis.

#### Ground-plane homography probe (negative)

The affine localizer was extended with a normalized projective homography, deterministic four-point RANSAC, Huber refitting, and a calibrated table-work-area rejection bound. The fit used only stationary/table-height frames from the 36 training episodes (stride 8); model parameters and the frame detector were not selected on test. On the six validation episodes, 362/551 eligible frames were accepted (65.7%). Conditional position RMSE was 18.2 mm in x and 28.0 mm in y, mean planar error 31.4 mm, and maximum 49.5 mm. Rejected out-of-area projections are counted as detection declines, not silently discarded successes. This is worse than the affine tabletop-only fit's 95.7% detection rate and lower conditional error, and does not support autonomous recovery. Keep the homography implementation as an inspectable research probe, but do not use it to issue robot actions; the calibration camera/data currently do not justify the projective model under the observed red-mask failures.

The robot-state conformal calibration was also inspected before considering A3. Its simultaneous per-horizon normalized-error quantiles are 73.6, 87.7, 87.7, and 87.7. These bounds are too wide to constitute a useful execution gate. A3 is therefore not eligible for a task-benefit run with the current calibration, regardless of the good aggregate world-model state RMSE.

Reproduce the homography probe from the repository root with the `tf-agent` Python environment:

```powershell
$env:PYTHONPATH = (Get-Location).Path + '\\src'
python -m wmal.training.visual_cli fit-stack-localizer --manifest runs/stack_block_visual_zoomed_disturbed_60_v1/manifest.json --output runs/stack_block_rgb_localizer_homography_v2.json --frame-stride 8 --tabletop-only
python -m wmal.training.visual_cli evaluate-stack-localizer --manifest runs/stack_block_visual_zoomed_disturbed_60_v1/manifest.json --localizer runs/stack_block_rgb_localizer_homography_v2.json --output runs/stack_block_rgb_localizer_homography_validation_v2.json --split validation --frame-stride 8 --tabletop-only
```

Probe artifact SHA-256: localizer `eb0af9809dab95df9a3452817966fa906e46604c79f3acc828aed268160b813c`; validation report `c8fae448a961862122c9cc3d70e942505632dbb280583d0ac82545745baef2d2`. `runs/` remains gitignored; the code, command, dataset manifest hash, and reported metrics are retained here.

## Stage 2 extension: 128 px recovery-rich trajectories and v7/v8

To address the missing recovery episodes and small red-object pixels, two new 80-episode collections used a 128×128 `workcell_overview` camera, disturbed reset family, seeds 7300–7379 and 7400–7479, and the logged `repeat_place_once` policy. The first collection is for model development (48 train / 8 validation / 8 calibration / 16 test); it had 26 task failures, 16 retry actions, and 0 retry recoveries. After tuning v8 on this collection's train/validation split, a second 80-episode collection was gathered with the same split sizes for external calibration/test. Its overall count was 28 failures, 16 retry actions, and 0 retry recoveries. The new external test split itself had 6 failures, 2 retry actions, and 0 retry recoveries (10/16 task success). Repeating the same place waypoint is confirmed to be ineffective across both collections; failure and retry labels are retained, but the data do not demonstrate a useful recovery policy.

The `VisualTrainingConfig.window_stride` addition bounds redundant training windows while preserving every episode in each bootstrap and always retaining each episode's last full-horizon window. v7 used the spatial encoder at 128 px, batch 128, stride 16, and 8 fine-tuning epochs; an initial dense epoch checkpoint from the same training split was used only as initialization. Its original development test showed aggregate object RMSE gains but continued to miss `block_0` pose and was used to motivate v8, so it is not treated as a final result. v8 fine-tuned from v7 on the same 48 train and 8 validation episodes with `object_weight=3` and `object_observation_weight=8`; the validation-selected checkpoint is `visual-8a89709ce81b690f0a8a`.

v8 was calibrated using only the eight calibration episodes from the second collection and evaluated once on its 16 untouched test episodes (720 windows sampled at stride 32). The external manifest hash is `9e1675b8f6dfabcfbf02a1a44cf70bdc6bddf3519f7617e475f8a3affcd0d351`.

| Metric | v8 | Persistence | Interpretation |
| --- | ---: | ---: | --- |
| Aggregate normalized object RMSE, h1 / h4 | 0.514 / 0.422 | 0.752 / 0.872 | Aggregate improves |
| `block_0` position RMSE, h1 / h4 | 14.9 / 14.6 mm | 0.52 / 2.19 mm | Critical manipulated pose is worse |
| `block_0` orientation geodesic RMSE, h1 / h4 | 16.5° / 16.3° | 0.59° / 1.49° | Critical manipulated orientation is worse |
| Current-image `block_0` position RMSE | 22.5 mm | — | The perception head does not localize accurately enough |
| Event Brier, h1 / h4 | 0.0287 / 0.0240 | 0 / 0.0037 | Sparse-event predictor loses to persistence |
| Event spread vs. absolute-error Spearman | 0.935 | — | Ranking signal only; not a calibrated probability |

The episode-level `block_0:position_m` conformal interval covered 87.5% of test episodes at each horizon, but mean half-width was 7.20–8.11 normalized units; its calibration quantiles were 52.9–54.0. The eight-episode calibration sample is also small. Robot-state calibration quantile was 58.0. These intervals are too broad to guide useful A3 prefixes. **The improved aggregate object score is not sufficient for A2/A3:** the action target's physical position/orientation and event score fail their predictive-quality gates. No world-model-controlled stack episode or A0–A3 paired manipulation experiment is claimed.

The 128 px projective RGB localizer was also fit from the external collection's 48 train episodes and checked on its eight validation episodes only. It accepted 360/388 stationary/table-height frames (92.8%), but x/y RMSE was 65.2/23.1 mm, mean planar error 44.7 mm, and maximum 193.9 mm. A camera-ray/known-table-plane probe using the MJCF camera extrinsics, 32° FOV, and nominal block-center height `.818 m` accepted 35/36 eligible validation frames across six episodes; RMSE was 27.0/36.2 mm, mean error 43.4 mm, and worst error 74.2 mm. These methods do not meet even the fixed skill's grasp-position tolerance and remain offline analysis only. The remaining localization error is consistent with the red silhouette centroid not representing the 3D block center under the oblique camera and robot occlusions; this explanation is a hypothesis, not yet isolated experimentally.

External projective-localizer artifact SHA-256 `760ece3b70f23ad2fd79e862816b997d63ce1ca5b25dc7da3312458b5d38eb79`; validation report `ed477e2ce59b009720da145c4fdf4c2048cfe7da0dbadde4e7d78534d6fd0093`.

Reproduction for the external evaluation (from the repository root, using `tf-agent`):

```powershell
$env:PYTHONPATH = (Get-Location).Path + '\\src'
python scripts/collect_stack_block_visual.py --config configs/tasks/stack_block_disturbed_dev.json --output-dir runs/stack_block_visual_external_80_v1 --seed-start 7400 --episodes 80 --profile disturbed --image-size 128 --recovery-policy repeat_place_once
python -m wmal.training.visual_cli calibrate --manifest runs/stack_block_visual_external_80_v1/manifest.json --checkpoint runs/stack_block_world_model_v8_localization/best.pt --output runs/stack_block_world_model_v8_localization/external_state_calibration.json --horizon 4 --alpha 0.2 --window-stride 32 --device cuda
python -m wmal.training.visual_cli calibrate-object --manifest runs/stack_block_visual_external_80_v1/manifest.json --checkpoint runs/stack_block_world_model_v8_localization/best.pt --output runs/stack_block_world_model_v8_localization/external_object_calibration.json --horizon 4 --alpha 0.2 --window-stride 32 --device cuda
python -m wmal.training.visual_cli evaluate --manifest runs/stack_block_visual_external_80_v1/manifest.json --checkpoint runs/stack_block_world_model_v8_localization/best.pt --output runs/stack_block_world_model_v8_localization/external_test_report.json --horizon 4 --window-stride 32 --device cuda --calibration runs/stack_block_world_model_v8_localization/external_state_calibration.json --object-calibration runs/stack_block_world_model_v8_localization/external_object_calibration.json
```

Artifact hashes: development manifest `f871d1f7c49abb10fb82bab49bcb2279f4ddcce745952eecfda1762e0a984cd0`; external manifest `9e1675b8f6dfabcfbf02a1a44cf70bdc6bddf3519f7617e475f8a3affcd0d351`; v8 checkpoint `e0099df7708e4669efb0f3a7693a54aff25f2c233d3b40afb2be1d5bf5321f2`; external test report `56f4441aed9109deb5afc971c338c629d0645b1eeda1776e90a8f6f5dadf5e81`; state calibration `33342f83f254866848d4ec2ed9680ebd9979c40777ded0c589bcb309fd040990`; object calibration `33ca49cdec60ba3ba5837da5a34b0bb9eb8d944ea5becec7d1b382b81dd54629`. `runs/` artifacts are gitignored; the config, commands, dataset hashes, split, checkpoint identity, and final metrics are recorded here.

## Stage 2 extension: frozen RGB-to-XY tabletop localizer

To isolate whether a perception model can recover the manipulated block's planar position from pixels, `wmal.perception.rgb_xy_regressor` adds a supervised RGB-only convolutional regressor. Object truth is used only to select stationary, reset-table-height frames and provide XY labels; inference receives only RGB. The predictor rejects frames where the red block mask is absent or outside the configured pixel-area range. It does not estimate height, orientation, motion, or object identity under occlusion.

The model was trained on the train split (48 episodes, 1,511 selected frames) of the already external `stack_block_visual_external_80_v1` collection, and the epoch was selected by the disjoint validation split (8 episodes, 252 accepted frames). Checkpoint `rgbxy-2411-24-9e1675b8f6df` selected epoch 24 with a 3.31 mm episode-mean planar error; validation detection was 252/256 sampled stationary candidates (98.4%). These figures are model-selection results, not independent evidence.

After freezing the checkpoint, an independent 80-episode collection used seeds 7500–7579, the same disturbed task and camera domain, and split sizes 48 train / 8 validation / 8 calibration / 16 test. The localizer never used that collection's train/validation labels. A domain signature binds camera/object semantics and collection code identity; evaluation rejects train/selection/calibration episode overlap. Calibration used eight calibration episodes and the episode maximum over at most 32 stationary tabletop frames sampled at stride 16. The split-conformal 80% radius is 14.72 mm (the maximum of those eight episode scores, rank 8/8).

| Independent result | Value |
| --- | ---: |
| Calibration detection | 254/256 frames (99.2%) |
| Calibration mean planar error | 2.50 mm |
| Calibrated episode-max radius | 14.72 mm at nominal 80% coverage |
| Test episodes / accepted frames | 16 / 508 of 512 candidates |
| Test detection | 99.2% |
| Test x/y RMSE | 2.95 / 3.65 mm |
| Test mean / p95 / maximum planar error | 3.55 / 8.91 / 18.15 mm |
| Test samples within 10 mm | 95.7% |
| Test episode-max coverage at 14.72 mm | 15/16 visible episodes (93.8%) |

The calibrated radius is an episode-level statistical bound for this fixed camera and stationary tabletop sampling protocol, not a per-frame guarantee or an image-conditioned uncertainty estimate. The localization result is suitable as an offline/perception baseline and motivates a future settled-block re-observation experiment. It is not integrated into manipulation control: the data gate excludes moving and raised blocks, and this head does not predict action-conditioned outcomes. The world-model block pose/event gates still fail, so no A0–A3 task-benefit claim follows.

Reproduce from the repository root with the `tf-agent` environment:

```powershell
$env:PYTHONPATH = (Get-Location).Path + '\\src'
python -m wmal.training.visual_cli fit-rgb-xy-localizer --manifest runs/stack_block_visual_external_80_v1/manifest.json --output runs/stack_block_rgbxy_localizer_dev_v2 --epochs 24 --batch-size 128 --frame-stride 16 --max-samples-per-episode 32 --learning-rate 0.0003 --seed 2411 --device cuda
python scripts/collect_stack_block_visual.py --config configs/tasks/stack_block_disturbed_dev.json --output-dir runs/stack_block_rgbxy_localizer_external_80_v1 --seed-start 7500 --episodes 80 --profile disturbed --image-size 128 --recovery-policy repeat_place_once
python -m wmal.training.visual_cli calibrate-rgb-xy-localizer --manifest runs/stack_block_rgbxy_localizer_external_80_v1/manifest.json --checkpoint runs/stack_block_rgbxy_localizer_frozen_v1/best.pt --output runs/stack_block_rgbxy_localizer_frozen_v1/external_calibration.json --alpha 0.2 --frame-stride 16 --max-samples-per-episode 32 --batch-size 128 --device cuda
python -m wmal.training.visual_cli evaluate-rgb-xy-localizer --manifest runs/stack_block_rgbxy_localizer_external_80_v1/manifest.json --checkpoint runs/stack_block_rgbxy_localizer_frozen_v1/best.pt --output runs/stack_block_rgbxy_localizer_frozen_v1/external_test_report.json --split test --frame-stride 16 --max-samples-per-episode 32 --batch-size 128 --device cuda --calibration runs/stack_block_rgbxy_localizer_frozen_v1/external_calibration.json
```

Artifact SHA-256: training manifest `9e1675b8f6dfabcfbf02a1a44cf70bdc6bddf3519f7617e475f8a3affcd0d351`; independent manifest `e3979302b7b4596cf91d92054d50fc0d8f0ccfabba39defd92c52eee3489a43b`; frozen checkpoint `9a77ef7ea1de4f64dcb2f006ee37ac225973f2cbfdc5375f2235db86a0e2f798`; calibration `23a6e448d5f9baaedfd36e0d6891a13c67e3f557d0a9856d14e6dd1091476ed6`; test report `3b465a91a9babf7f9d20fecfc5c5387f512e7853a26c46c9e36a332bbcc14493`.

## Stage 2 extension: zoomed disturbed model selection and final test

Dataset `stack_block_visual_zoomed_disturbed_60_v1` has 60 episodes split by episode into 36 train, 6 validation, 6 calibration, and 12 final test. It contains synchronized 64×64 RGB, 70 robot-state channels, 14 action channels, 39 named object-state labels, task events, step IDs, and 20 ms simulation timestamps. The policy that generated these episodes had world-model access disabled; the current collection does not contain recovery trajectories, so that part of the S2 data requirement remains open.

Two candidates were fit only against train and validation: v5's standard-resolution residual object head and v6's spatially preserving encoder. v5 was selected because its best validation global loss was 0.422 (versus 0.558 for v6), future object loss 0.162 (versus 0.207), and current-image object loss 0.229 (versus 0.327). Selected checkpoint: `visual-474d09e1b47245a4bc5f`; checkpoint SHA-256 `fba18c29879f1ead0a08b01aab8c201ed7148a56e4f7cbaF925ad116873d50dd`.

After freezing v5, an episode-level group-wise conformal calibrator was fit on the six calibration episodes (alpha 0.2, stride 32). The calibrated model was then evaluated once on 12 unseen test episodes, with 516 sampled windows. Aggregate normalized object-state RMSE was 0.560/0.408 at horizons 1/4 against persistence 0.795/0.870, and ensemble spread/error Spearman correlation was 0.563. However, the manipulated block's position was a critical exception: physical RMSE was about 9.9 mm at all four horizons, versus persistence 0.54 mm at horizon 1 and 2.23 mm at horizon 4. Its quaternion geodesic error was about 7.7° versus persistence 0.58°–1.52°. The aggregate score therefore does not establish useful prediction of the action target.

Calibration was not effective enough to gate control: test episode-wide simultaneous coverage across all 39 object-state dimensions was 8.3%, 16.7%, 8.3%, and 33.3% by horizon, below the nominal 80%; mean calibrated half-width was 3.72–4.44 normalized units. The block_0 position group reached only 66.7% episode coverage. Keep the calibration result as a negative finding; do not use this checkpoint for A2/A3 task-benefit claims. The fresh final test split is now consumed for this frozen v5 evaluation and must not be reused for tuning.

Recorded artifact hashes: manifest `a577cd2bc3280ee7af4b82199ebab6672fbb0a4fc67f7d8ca1cc1c4227649ace`; checkpoint `fba18c29879f1ead0a08b01aab8c201ed7148a56e4f7cbaF925ad116873d50dd`; object calibration `c249b396af4c8d78ab5e2bd9193f86b89336eba30d7807956c47b498647e063d`; test report `1f0c2029c08dabf834351b29319168512cb0aed4a95fbe8ad6e44e6e52adc33e`.

## Stage 2 extension: horizon-32 v9 candidate (validation-only negative result)

v9 fine-tuned the frozen v8 checkpoint on the train split (48 episodes) of `stack_block_visual_external_80_v1`; the eight validation episodes were used for checkpoint selection. The 16-episode test split was not read for this v9 run. This collection's test split had already been evaluated for v8, and those findings motivated further object-prediction work, so it is no longer a blind final test for a successor candidate. Do not report it as an independent v9 test.

The candidate used `configs/training/stack_block_visual_v9_h32.json`, a 32-control-step horizon at the 20 ms control period (0.64 s), and selected epoch 6 by global validation loss 4.8814. The validation evaluation used 8 episodes, 364 windows, and stride 32. Model version: `visual-251091eed33c95b031c1`.

| Validation metric | v9 model | Persistence |
| --- | ---: | ---: |
| `block_0` position RMSE, h1 / h4 / h8 / h16 / h32 | 14.79 / 14.32 / 14.66 / 14.10 / 14.55 mm | 0.53 / 2.16 / 6.34 / 10.77 / 11.97 mm |
| `block_0` quaternion geodesic RMSE, h1 / h4 / h8 / h16 / h32 | 11.58 / 10.85 / 10.79 / 11.07 / 11.14° | 0.53 / 1.44 / 2.27 / 4.89 / 4.85° |
| Event Brier, h1 / h4 / h8 / h16 / h32 | 0.02656 / 0.02026 / 0.02396 / 0.01830 / 0.01805 | 0 / 0.00733 / 0.02473 / 0.02564 / 0.03022 |

The candidate loses badly to persistence on the manipulated block's position and orientation at every listed horizon. Event prediction is worse at short horizons and better at longer horizons. Event/object ensemble spread versus absolute error has Spearman correlation 0.907 / 0.649, respectively; these are rank associations on eight episodes, not calibrated uncertainty. Current-image `block_0` position RMSE is 19.59 mm. The validation gate therefore fails, and this checkpoint is not eligible for A2/A3 control claims. No v9 calibration or final-test evaluation was run.

The next M2 attempt needs a new collection with a useful recovery policy and an explicit manipulated-object validation objective. Keep the 16 test episodes from this manifest closed for v9 and successor comparisons; obtain a new untouched test collection after model selection. M3 remains deferred until the manipulated-object prediction and uncertainty gates pass.

Reproduction from the repository root with the `tf-agent` environment:

```powershell
$env:PYTHONPATH = (Get-Location).Path + '\src'
python -m wmal.training.visual_cli train --manifest runs/stack_block_visual_external_80_v1/manifest.json --output runs/stack_block_world_model_v9_h32 --config configs/training/stack_block_visual_v9_h32.json --pretrained runs/stack_block_world_model_v8_localization/best.pt --no-progress
python -m wmal.training.visual_cli evaluate --manifest runs/stack_block_visual_external_80_v1/manifest.json --checkpoint runs/stack_block_world_model_v9_h32/best.pt --output runs/stack_block_world_model_v9_h32/validation_h32_report.json --horizon 32 --window-stride 32 --device cuda --split validation
```

Artifact hashes: config `93f55b828c4824dd23214a9df064c8717c623d19f4a6530df74db838b4839238`; dataset manifest `9e1675b8f6dfabcfbf02a1a44cf70bdc6bddf3519f7617e475f8a3affcd0d351`; v9 checkpoint `7600e526f696b7ddc4bee5ed89ccb5cff0ae16cc3a7f08624cfe75e56a0135d6`; training report `23a370df285a19a24b6047639d8fb4e3bb61605b36f62402bf7e53d0e2b4499e`; validation report `edbd303e782b8bfe2077b4042e4f4148ae209b188f494dd6aa07e548f338bc35`.
## Stage 1 verifier v2 and dropped-object event

The approved contract requires an independently classified failure when a block falls after a verified lift. Verifier v1 only surfaced failed placement as `stable_support_contact_not_observed`. On the recorded disturbed seed 7405, the block remained bilaterally held through transport, then fell from the intended stack location to the table during release; the old result contained no distinct drop event. The implementation now tracks two consecutive executed samples at or below reset height + 20 mm after lift, with no bilateral contact or stable stack, and emits `failed=true, failure=object_dropped_after_lift`. A fall more than 50 mm below reset height is classified as `object_fell_below_table`.

The synchronized collector adds `object_dropped_after_lift` as event channel four. It is a current dropped-state label that stays active while the block is down and clears when bilateral contact is reacquired. It is constructed only from scene truth in the offline labeler; `RobotObservation` and agent serialization are unchanged. The thresholds are present in all stack-block task configs and in verifier reports. Verifier identity is `stack-block-verifier.v2`; failure evidence includes reset height, relative height, and consecutive drop-sample count.

| Deterministic simulation replay | Task outcome | Drop label | Verifier report |
| --- | --- | --- | --- |
| Seed 7402, disturbed | success | none | none |
| Seed 7405, disturbed | failure | step 1291 through 1536 | step 1291: `object_dropped_after_lift` |

Both replays used the synchronized 128 px collector callback; each produced 1,536 aligned transitions and a four-channel event array. The successful episode was not marked as dropped. The failure episode now reports `failure_cause=object_dropped_after_lift` rather than the generic skill-verification error. These are deterministic development replays on already-used seeds, not new held-out data or a recovery result.

Existing v8/v9 checkpoints and manifests retain their three-event schema. They have not been recalibrated or evaluated with the new event. A future four-event model must either start from compatible weights or explicitly expand the event head while preserving shared event names; the 3-channel checkpoint is not a drop predictor. The external 80-episode test split remains closed, and no new model training or test evaluation was done in this change.

## 2026-10-13: fresh-seed drop-event QA batch

Collected 12 new disturbed-profile episodes with the M1 controller and no recovery policy, seeds 9200–9211. The task outcomes were 9/12 successful and 3/12 failed. Two post-lift drops (seeds 9200 and 9203) were emitted as `object_dropped_after_lift` through the episode end; one separate grasp-contact failure (seed 9202) was not mislabeled as a drop. All 12 episodes had four event channels, contiguous transition/step alignment, matching simulation-time lengths, and finite labels; the alignment report contains no errors.

This is a small collector/verifier QA batch, not a re-estimate of the jittered M1 success gate, not recovery data, and not a world-model evaluation. Its generated validation/calibration/test labels were inspected for QA, so none of its episodes can serve as a blind final model test. Do not use this batch for model selection or calibration either. The M2 recovery-data requirement and the untouched final test collection remain open.

Artifacts under the ignored `runs/` directory: manifest SHA-256 `81c66feb5872a6bdac9504867c3135c98cf1c2b4dc67c759c8269085e8519492`; alignment report SHA-256 `a8142a260fa4f133756e0e99e1188b06a9201c2cada895dbe4732eabd4d8c9f79`.

The visual trainer now permits an append-only named event schema when every non-event semantic and network setting matches the parent checkpoint. It copies all shared member weights, preserves each old event-head row by name order, initializes the appended row, and extends per-member supervision evidence with `false` until labels are observed. Reordered/removed event channels and other architecture or semantic changes remain rejected. A regression test fine-tunes a three-event parent on a four-event manifest, proves the shared rows remain exact when only the new channel is labeled, and confirms four-channel prediction output. A read-only probe against `visual-8a89709ce81b690f0a8a` and the 12-episode manifest also passed; member masks expanded from 3×3 to 3×4. The fourth event head has not been trained or evaluated, so this enables a future M2 run but establishes no prediction quality.

The targeted append-event and existing temporal fine-tuning tests passed. The full suite ran 284 cases: 272 passed, 6 skipped, and 6 errored for this Windows environment (symlink privilege, missing `bash`, and pipe-backed stdin unsupported by `select`). There were no remaining assertion failures after preserving the existing architecture-mismatch error contract. Run artifacts include event alignment report SHA-256 `a8142a260fa4f133756e0e99e1188b06a9201c2cada895dbe4732eabd4d8c9f79` and read-only event-head probe SHA-256 `e26634ee83d4d63479f9d09aed22ec6a0b55f2f792dd4f23af1406b6afc663de`.

## 2026-10-13: failure-cause detail and Windows verification

The baseline result resolver now prefers a latched verifier cause, then a non-generic skill error, then the failed predicate corresponding to the skill (for example `grasp_block` → `grasped`) before falling back to generic skill status. A deterministic seed-9202 replay now reports `bilateral_finger_contact_missing` at `grasp_block`, step 1024. The earlier 12-episode QA manifest remains immutable and retains its original generic cause for that episode; subsequent collections use the more specific result.

Windows compatibility was repaired in the interactive visual runner (console/file/pipe stdin polling), same-size JSONL rewrite detection (content verification no longer depends on timestamp granularity), symlink tests (skip only the unavailable symlink assertion), and ROS 2 shell-helper tests (Git Bash path and POSIX path conversion). After the context-32 changes, the full suite exited successfully: 289 tests, 282 passed, 7 skipped, zero failures/errors. It ran in the `tf-agent` environment with MuJoCo and PyTorch, `PYTHONPATH=src`, and Git for Windows `bin` and `usr/bin` on `PATH`.

## 2026-10-13: context-32 candidate and validation-only result

To address the v9 failure, v10 uses a categorical RSSM with 32 context steps. The first three epochs were interrupted after a slow run; this continuation fine-tuned their `last.pt` for five additional epochs, with a newly initialized optimizer (so this is not an uninterrupted eight-epoch optimizer run). The best checkpoint is `visual-ca7f510a6b83125feb18`, selected at epoch 5 with validation loss 5.9110. Training used 48 train episodes; checkpoint selection and validation evaluation used the disjoint 8-episode validation split (364 windows, stride 32) of `stack_block_visual_external_80_v1`. The 16-episode test split was not decoded or scored. The evaluator verifies manifest content hashes for all archives, but materializes arrays only for the requested split.

The candidate still fails the manipulated-object gate. `block_0` position RMSE at horizons 1/4/8/16/32 was 18.47/18.22/18.32/19.35/19.53 mm, versus persistence at 0.53/2.16/6.34/10.77/11.97 mm. Quaternion geodesic RMSE was 18.23/18.32/18.29/18.61/18.63 degrees, versus persistence at 0.53/1.44/2.27/4.89/4.85 degrees. Overall normalized object-state RMSE beat persistence, but that aggregate is dominated by the stationary blocks and does not rescue the `block_0` result. The previously recorded event persistence Brier values were invalid: the evaluator used the first future label as if it were known at the prediction origin. A validation-only re-evaluation of the same frozen checkpoint uses the last known origin event label and masks eight offset-zero windows whose legacy archives lack a reset label. For bilateral grasp contact, object lifted, and stable stack, model Brier at horizons 1/4/8/16/32 is 0.0682/0.0690/0.0714/0.0728/0.0737, versus corrected persistence at 0.0009/0.0066/0.0243/0.0253/0.0300; 356 of 364 windows (1,068 labels) are scored at each horizon. Event ensemble-spread/error Spearman is 0.671. The fourth dropped-object event is absent from this three-event dataset. The corrected result still supports a validation-only negative finding, not usable control confidence or task benefit. M2 remains open and M3 remains gated.

During this run, `CategoricalRSSMMember.observe` was changed to batch-encode the aligned real observation sequence before the recurrent update loop. Normalization and bootstrap event-supervision scans now read only the small NPZ arrays they need instead of decompressing RGB. Evaluation compares event predictions with future-horizon labels and persistence with the last event state known at the prediction origin. New collections store `event_states` and masks aligned to every observation, including reset; legacy archives without reset labels omit offset-zero event scoring. Interrupted manipulation receipts now retain truth for every completed control step; the baseline logs that partial execution and pairs only observed frames with matching truth when building training rows. Regression tests cover these timing and fault paths. Fine-tuning took 1,034.9 seconds for five epochs on the recorded CUDA run; data-path changes remove avoidable work but do not make recurrent multi-step training cheap.

Reproduce the continuation and validation evaluation from the repository root with the `tf-agent` environment, while retaining the three-epoch parent artifact:

```powershell
$env:PYTHONPATH = (Get-Location).Path + '\src'
python -m wmal.training.visual_cli train --manifest runs/stack_block_visual_external_80_v1/manifest.json --output runs/stack_block_world_model_v10_context32_finetune5 --config configs/training/stack_block_visual_v10_context32_finetune5.json --pretrained runs/stack_block_world_model_v10_context32/last.pt --no-progress
python -m wmal.training.visual_cli evaluate --manifest runs/stack_block_visual_external_80_v1/manifest.json --checkpoint runs/stack_block_world_model_v10_context32_finetune5/best.pt --output runs/stack_block_world_model_v10_context32_finetune5/validation_h32_report.json --horizon 32 --window-stride 32 --device cuda --split validation
python -m wmal.training.visual_cli evaluate --manifest runs/stack_block_visual_external_80_v1/manifest.json --checkpoint runs/stack_block_world_model_v10_context32_finetune5/best.pt --output runs/stack_block_world_model_v10_context32_finetune5/validation_h32_event_origin_report.json --horizon 32 --window-stride 32 --device cuda --split validation
```

Artifact SHA-256: config `797b61b6b50ff3386b7fe03ecea19ea259ca6403995155b2ee29be8a2a6f4765`; manifest `9e1675b8f6dfabcfbf02a1a44cf70bdc6bddf3519f7617e475f8a3affcd0d351`; three-epoch parent checkpoint `edf53c22673e4b9a4f1de44b1e3e693c2e90646573e56d636d57f2540148deae`; selected checkpoint `6f94d789ac0ef957a5c2908ed15180c7a40efda7be07d62d0a4fded79e1f0d26`; training report `dd85ecdaa295401b2e61c2dc2f114db56c8c94009a238246224e7ccde8ddead0`; validation report `39702cf7b8055acc4d430c33520763b061968354cebad567516222c77d32e940`.

## 2026-10-13: end-to-end gripper command receipts

A phase-one interface audit found that the adapter returned only `accepted`, while `ManipulationSession.execute` discarded that result and baseline JSONL omitted it. The adapter receipt now distinguishes accepted from fully executed, includes a fault, and can bind to the parent action ID. The session finalizes it after the requested control steps: `executed=true` only for a complete uninterrupted action; partial progress remains exact in the parent `ExecutionReceipt.executed_steps`, and its fault is copied into the gripper receipt. Actions without a gripper command carry no nested receipt. Baseline `action_trace` serializes the receipt alongside the action.

Regression tests first failed on the missing receipt fields and trace entry, then passed for accepted-but-pending commands, parent action ID binding, full and interrupted execution, JSONL serialization, and rejection of an empty command ID without mutating the gripper target. Targeted tests passed: gripper 6/6, manipulation session 8/8, baseline 6/6. Full suite: 292 tests, 285 passed, 7 skipped, zero failures/errors; `git diff --check` passed. This closes an execution-evidence gap only; it does not change the synthetic gripper model or the 79/80 no-model task result.

## 2026-10-13: final-review corrections

The final review found that a post-action camera fault could lose a partial receipt: the session had advanced physics without storing final truth, the skill immediately repeated the failed observation, and the baseline required equal observation/truth lengths before writing JSONL. The session now snapshots verifier truth after each completed physics step, exposes the last confirmed observation, and returns interrupted skills without retrying the failed sensor read. Baseline JSONL reports the physical end step and whether the observation is complete; visual training rows are emitted only for observation/truth pairs with matching step IDs. A persistent camera-failure regression confirms the interrupted receipt and final physical truth survive in JSONL.

The event persistence baseline now uses the last known event state at the prediction origin. New collector archives store reset and post-action event states plus masks; legacy archives without a reset label use prior-transition labels and omit offset-zero windows from both model and persistence scores. The old v10 persistence numbers are superseded by `runs/stack_block_world_model_v10_context32_finetune5/validation_h32_event_origin_report.json`, SHA-256 `ddf0aed253bf6357f0569d34c7a96d2e4d741a6e61b9796d06166c0f480d0a93`. This is a validation-only recomputation with the frozen v10 checkpoint; it does not use test metrics. The corrected event gate remains negative. No model was trained.

TDD regressions failed on the missing partial truth snapshot, retry after sensor failure, absent prediction-origin event context, and leaked h1 persistence label; all pass after the fixes. Full suite: 297 tests, 290 passed, 7 skipped, zero failures/errors; `git diff --check` passed.

## 2026-10-13: M1 baseline rerun and M2 recovery-controller gate

All experiments in this entry used the repository's `wmal` environment. A fresh no-world-model baseline batch passed 10/10 jittered development seeds and 19/20 fixed jittered holdout seeds (95%); nominal seed 9200 also passed. This clears the predeclared M1 pilot floor of 90% for this initial-state family. The sole holdout failure was seed 106, `object_dropped_after_lift`. Run manifests, events, and results are under the active worktree's ignored `runs/stack_block_jittered_dev_stage1_gate_20261013_v1` and `runs/stack_block_jittered_holdout_stage1_gate_20261013_v1` directories. These are small pilot batches, not a claim of broad robustness or real-hand performance.

The M2 implementation adds a calibrated three-frame RGB-XY placement-miss assessor, one bounded regrasp attempt, and a feedback Cartesian servo. A servo replay exposed that the measured joint angle differs from the MuJoCo position-actuator target at static equilibrium. The servo had been replacing the actuator target with the measured angle, which removed the holding bias and sent the arm off course. `ManipulationSession.actuator_targets_for()` now exposes allow-listed left-arm targets, and Cartesian updates add joint deltas to the active targets. A reconstructed seed-9200 placement snapshot then reached a 50 mm vertical test target within 5.93 mm after 112 control steps. Targeted tests passed: recovery 14/14 and baseline integration 10/10.

The end-to-end M2 gate remains **failed**. On disturbed seed 9200 and jittered holdout seed 106, the RGB assessor correctly reported `confidently_outside_target`, but the recovery servo stopped during descent with 45.3 mm and 42.6 mm pose errors, respectively. An offline MuJoCo contact inspection on seed 9200 identified the left elbow-yaw collision body contacting the table. Alternative wrist orientations can lower the tool, but bilateral gripper contact alone is not a valid grasp signal: diagnostic closes either contacted the adjacent support block or allowed the dropped block to slip back to the table during lift. The recovery runner stops when descent fails and records the failure; it does not proceed to an unverified close or claim recovery.

Keep the frozen RGB-XY checkpoint available for observation and trigger evaluation, but do not freeze the current recovery policy or collect its attempts as successful recovery trajectories. The drop locations are at or beyond the current robot's safe low-height workspace, and the tabletop localizer has not been calibrated for in-hand height or lift verification. M2 still needs a reachable, collision-free recovery grasp and an observation-based grasp/lift acceptance gate before paired A0-A3 experiments can support a task-benefit claim. The recovery run artifacts are under `runs/stack_block_rgbxy_recovery_holdbias_dev_20261013_v1` and `runs/stack_block_rgbxy_recovery_seed106_dev_20261013_v1` in the active worktree.

A diagnostic lateral push on seed 9200 moved the dropped block from x=0.275 m to x=0.255 m, but also moved the free support block from x=0.193 m to x=0.175 m; their relative separation stayed about 0.080 m, so the stack predicate still failed. The support is a free object rather than a fixed fixture. Do not treat pushing as a recovery fallback unless the support motion is controlled. Seed 106 has now been used for recovery development and must not be counted as a blind recovery-policy test seed; use a fresh paired seed list after the policy is repaired.

### Development-only pre-release alignment probe

An in-memory prototype used the frozen RGB-XY reset estimate and measured grasp pose to compute a tool target, then servoed directly toward the stack target before opening and settling. The release-envelope matrix covered disturbed seeds 9200-9203: the independent verifier reported a stable stack for 9200, 9201, and 9203; seed 9202 failed at grasp before placement. In the earlier same-family baseline batch, 9200 and 9203 were post-lift drops and 9202 was a grasp-contact failure, so this probe appears to rescue the two release failures on this four-seed development slice. It is not a frozen or budget-matched comparison: the implementation existed only as a monkeypatch, the prototype used extra servo steps, and the final verifier uses simulator truth offline.

The 9203 trace also lost bilateral finger contact during the final descent. Its later successful stack result does not establish a robot-visible support-transfer signal. A persistent policy must fail closed on uncertain grasp/support state or add an observation-based acceptance gate before release. Keep these logs as candidate-path development evidence only; do not count them as M2 recovery successes, a world-model benefit, or an A0-A3 result. Artifacts are under `runs/stack_block_pre_release_release_envelope_matrix_20261013_v1` in the active worktree. This probe and the records in this entry used only the repository's `wmal` environment.

### 2026-10-13 follow-up — integrated pre-release alignment paired replay

The candidate is now implemented as the separate `rgbxy_pre_release_align_dev_v1` policy in the baseline runner. It uses the frozen RGB-XY estimate only at reset, the measured bilateral grasp pose to form an object-to-tool offset, and a bounded measured-pose servo before the existing release and settle phases. The existing fixed-waypoint path remains available as the paired baseline. The policy is simulation-development-only and does not load a world model.

On the same disturbed seeds 9200-9203 with the same 900-control-step skill cap and 4096-step episode cap, fixed waypoint scored 1/4 and pre-release alignment scored 3/4. Seeds 9200 and 9203 changed from `object_dropped_after_lift` to verifier-confirmed stacks; seed 9201 stayed successful and seed 9202 retained `bilateral_finger_contact_missing`. The baseline used 1536, 1536, 1024, and 1536 steps; the candidate used 1696, 1696, 1024, and 1696. This is equal maximum budget, not equal realized control effort: the two rescued episodes each cost 160 extra control steps. The paired slice is too small to establish general task benefit.

The release envelope intentionally accepts a final pose with XY residuals up to 20 mm, vertical gap from 0 to 60 mm, and orientation error up to 0.15 rad when the servo stops at its iteration limit. Seed 9203 was accepted with one gripper contact absent. Therefore this is a useful simulation hypothesis but not a verified support-transfer condition and not a hardware-ready release rule. Add an observation-based acceptance gate, then evaluate it on fresh seeds before freezing or using it for A0-A3. Paired artifacts are under `runs/stack_block_pre_release_align_paired_baseline_20261013_v1` and `runs/stack_block_rgbxy_pre_release_align_dev_20261013_v1`; the earlier in-memory probe remains under `runs/stack_block_pre_release_release_envelope_matrix_20261013_v1`.

Validation in `wmal`: recovery/alignment 18/18, skill contracts 8/8, and baseline integration 11/11; changed Python files compile. `git diff --check` passed. Seed 9200-9203 are development seeds and are not reserved for final evaluation.

### 2026-10-13 follow-up — contact-guard paired evaluation

The `rgbxy_pre_release_align_contact_guard_dev_v2` candidate and fixed-waypoint baseline were run on the same 30 disturbed seeds (9300–9329), with the same frozen RGB-XY artifact and 900-step skill cap. The baseline stacked 21/30; the candidate stacked 22/30. Five seeds succeeded only under baseline and six only under the candidate (16 both succeed, 3 both fail). Mean episode control steps were 1484.8 for baseline and 1560.9 for the candidate, a +76.1-step candidate cost. This is a small development comparison with equal caps but unequal realized costs; the +1 success is not evidence of reliable task benefit. Both manifests state `world_model_loaded=false`.

The candidate's contact guard stopped alignment on five episodes before release. Their offline verifier traces show the block was already supported and stable, but the policy had only unilateral or no finger contact at the stop. A bounded stationary-hold probe on those five episodes found RGB-XY predictions with a mean maximum within-window spread of 1.93 mm while mean planar error was 41.60 mm; actual block motion during the hold averaged 1.07 mm maximum displacement. The stable image estimate is substantially biased in the elevated/occluded stacking domain. The frozen localizer's `stationary_tabletop_episode_max_error` radius does not apply here.

Ruling: keep contact loss fail-closed and do not authorize release from the current RGB-XY estimate or a stable-image check alone. The observed estimator bias makes the tabletop calibration an invalid support certificate; the cost if this ruling is wrong is a missed release when the block is already safely supported. A support-transfer gate needs its own episode-disjoint data and calibration, or a distinct robot-observable signal. The hold probe used verifier truth only for offline diagnosis, never for action selection.
### 2026-10-13 follow-up - CUDA-trained visual world model development run

The synchronized 50-episode contact-loss/hold manifest (`stack_block_contact_loss_hold_dataset_dev_20261013_v2`, manifest SHA-256 `b8550e9b5b5f004fa19c73fcaf728c20a2f8b6524a07b01477731c0526dbf74f`) was used for an exploratory visual-world-model run. Its episode split is 30 train, 5 validation, 5 calibration, and 10 test. Seven episodes contain a stationary hold: five train, one validation, one calibration, and none in test. Training used only `C:\Users\Admin\miniconda3\envs\wmal\python.exe`, NVIDIA RTX 5060 Ti / CUDA 12.8, `stack_block_visual_v2.json`, three ensemble members, batch size 512, and five epochs. The report records `device=cuda`; train/validation losses decreased from 0.6107/0.5122 to 0.3275/0.2964. Best model `visual-128fe1b59c62147d2795` is at `runs/stack_block_contact_loss_wm_dev_20261013_v1/best.pt`, SHA-256 `9f2c00b4e66a37d588ac7713c3ca211c571659240d0cf8d9c4891d9f4adad1d3`.

State and named-object intervals were calibrated with alpha 0.2 on the five calibration episodes, then evaluated on the separate 10-episode test split (530 windows, stride 32). Reported state RMSE improved over persistence by 7.2%, 10.9%, 15.6%, and 19.3% across horizons 1-4. Episode-simultaneous state interval coverage was 1.0/0.9/0.9/0.9, but mean scaled half-width grew from 2.94 to 9.86. Aggregate normalized object-state RMSE improved over persistence by 18.1%, 29.2%, 24.8%, and 26.1%; object interval coverage was only 0.1/0.2/0.1/0.2. Event Brier was worse than persistence at every horizon (model 0.0236/0.0175/0.0152/0.0161; persistence 0.0071/0.0085/0.0080/0.0108), despite event ensemble-spread/error Spearman 0.841. Object spread/error Spearman was 0.478. The five-episode calibration set and absence of hold episodes in test make the nominal 80% coverage evidence weak and leave the support-transfer domain unvalidated.

The first test-evaluator attempts hit a repeatable Windows `python.exe` native crash in `numpy.corrcoef` while computing rank correlations. `_rank_correlation` now computes the same Pearson correlation of average-rank vectors with `math.fsum`; the full 530-window CUDA evaluation then completed and is saved at `runs/stack_block_contact_loss_wm_dev_20261013_v1/test_evaluation.json`. These are prediction metrics only: all manipulation rollouts still have `world_model_loaded=false`, so this run establishes no closed-loop task benefit, recovery safety, or A0-A3 result. Do not freeze this model. The next gate is a substantially larger episode-disjoint hold/support-transfer dataset, especially calibration and test episodes, followed by a paired same-budget closed-loop ablation.

## 2026-10-13 follow-up — strict release scoring and fresh paired validation

The stack task's terminal score now requires the independent stable-stack predicate plus a healthy gripper that has reached its fully open target and has no physical contact between `block_0` and either finger. Gripper contact flags are global to all workcell blocks, so incidental contact with the supporting `block_1` does not mean that the target block remains grasped. This is an offline scoring check; the placement policy still has no verifier-truth access. The output schema is `wmal.stack_block_results.v2` and both manifest and result record the task-success definition.

On development seeds 9400–9429, strict candidate v4 and fixed waypoint scored 22/30 and 16/30. The paired outcomes were 12 shared successes, 10 candidate-only successes, 4 baseline-only successes, and 4 shared failures. The v4 servo uses up to 42 iterations of 12 control steps (504 total), filling the configured 512-step alignment allowance; the previous default stopped after 24 iterations despite unused budget. These development seeds were used for tuning and are not independent validation.

Fresh paired validation used disturbed seeds 9600–9659, identical episode/skill caps, the same frozen RGB-XY checkpoint and calibration for the candidate, and `world_model_loaded=false` in both arms. Fixed waypoint succeeded on 42/60 (70.0%); strict v4 succeeded on 39/60 (65.0%). Pair counts were 32 shared successes, 10 baseline-only, 7 candidate-only, and 11 shared failures. The two-sided exact McNemar p-value for the 17 discordant pairs is 0.629. Mean episode control steps were 1450.7 for baseline and 1535.1 for the candidate (+84.4). Candidate failures were 10 missing bilateral grasp contacts, 10 bilateral contact-loss declines, and 1 target-block release failure.

Ruling: do not freeze v4 or advance it to A2/A3. The corrected score shows the policy can complete a physical release, but on the untouched seeds it did not improve success and used more control steps. The tabletop RGB-XY model does not resolve the support-transfer/contact-loss problem. The 60-seed manifests and paired results are under `runs/stack_block_strict_release_eval_9600_9659_baseline_v2` and `runs/stack_block_strict_release_eval_9600_9659_candidate_v4`; config is `configs/tasks/stack_block_rgbxy_pre_release_strict_release_eval_9600_9659.json`. This is still simulation-only evidence, not a world-model ablation, ROS 2 integration, or sim-to-real evaluation.

## 2026-10-13 follow-up — contact-transfer world-model dataset

Collected a new 200-episode RGB/state/action/object-truth set with strict v4, the frozen RGB-XY checkpoint, and a bounded 96-step stationary observation after alignment contact loss. The collector now assigns the 60/10/10/20 episode splits through a deterministic shuffle seeded by the collection seed, avoiding contiguous seed-range splits. The manifest records `split_seed=9700`, per-episode hashes, action/image semantics, the localizer artifacts, and the collector/controller code hashes. Manifest SHA-256: `ce73f1837f1e7dfd68a761f196514764bdfb0e9f381fb1e3b50e0982b6e69abf`.

The data uses 128×128 overview RGB, 70 robot-state channels, 14 action channels, 39 named object-truth channels, and 20 ms aligned transitions. Split sizes and successful task episodes were train 120/82, validation 20/10, calibration 20/14, and test 40/28. Stable-stack episodes were 99/15/17/35; contact-loss stationary-hold episodes were 18/5/3/7 respectively, totaling 3,168 hold steps. The final test therefore contains both 35 stable-stack trajectories and 7 explicit contact-loss holds; it remains sealed from model selection and calibration.

The dataset is under `runs/stack_block_wm_contact_transfer_200_dev_20261013_v1` (about 3.87 GB). All episodes and manifest arrays passed synchronization/dimension checks. World-model fit is running on the same `wmal` CUDA environment; no checkpoint is frozen until validation selection, episode-disjoint calibration, and one final test pass the object, event, and uncertainty gates.

## 2026-10-13 follow-up — GPU model selection, calibration, and sealed test

Training on the existing `wmal` CUDA environment completed for two compact candidates, each with three ensemble members, six epochs, batch size 128, horizon four, and a 32-step training-window stride. The v5 run used seed 2725 and reached best validation loss 1.00942; v6 used seed 2726 with `object_spatial_encoder=true` and reached 1.04301. Because both the seed and encoder differ, these are candidate runs, not an isolated encoder ablation. An earlier 12-epoch, stride-eight v5 exploration was stopped after two epochs due to run time and was excluded from selection. Its partial files remain under `runs/stack_block_wm_contact_transfer_v5_20261013_v1`.

On the 20-episode validation split (1851 windows, stride 16), v6 had slightly lower `block_0` position RMSE at each horizon than v5 (15.78–15.94 mm versus 16.32–16.98 mm), but persistence was only 0.55–1.78 mm. Both models also had worse event Brier scores than persistence at all four horizons. V6 was therefore selected solely for a single exploratory final test of the calibration and evaluator path; neither checkpoint passed the task-critical validation gate or was frozen for control.

V6 state and named-object intervals were calibrated separately on the 20 calibration episodes with alpha 0.2, horizon four, and stride 16. State global normalized quantiles were 52.30, 58.21, 194.06, and 194.06; object global normalized quantiles were 169.54, 191.48, 191.48, and 191.48. These large values already indicate broad bounds. The untouched 40-episode test split was then evaluated exactly once with both calibrators on CUDA: 3952 windows, stride 16, checkpoint `visual-526b50bc33cf93628371`.

| Test metric | Horizon 1 | Horizon 4 |
| --- | ---: | ---: |
| `block_0` position RMSE, model / persistence | 15.47 / 0.56 mm | 15.67 / 1.86 mm |
| Event Brier, model / persistence | 0.04011 / 0.00550 | 0.03007 / 0.00430 |
| Robot-state RMSE, model / persistence | 0.05274 / 0.06468 | 0.09559 / 0.13717 |
| Episode-simultaneous object coverage | 10% | 55% |
| Episode-simultaneous state coverage | 80% | 100% |
| Mean calibrated state half-width, scale-normalized | 2.80 | 21.76 |

The full four-horizon object coverages were 10%, 12.5%, 12.5%, and 55%, below the nominal 80% target. State coverage was 80%, 65%, 95%, and 100%, with wide scale-normalized half-widths of 2.80, 3.81, 16.24, and 21.76. Target-position group coverage was 70%, 80%, 87.5%, and 97.5%, but its mean scale-normalized half-width grew from 2.49 to 4.04. Object/event ensemble-spread versus realized-error Spearman correlations were 0.505/0.690; rank association does not repair the poor prediction or interval coverage. The aggregate object-state metric improves over persistence at longer horizons, but it does not represent the manipulated block's position accuracy. Full frame MSE was also slightly worse than persistence at every horizon.

Artifact SHA-256 values for reproducibility:

| Artifact | SHA-256 |
| --- | --- |
| v5 compact config | `c6cd41b4e06f8fdef57af8f75e104e457135c29c7cd91f75cff64afd140ec204` |
| v5 compact `best.pt` | `8c0e0b44e2023d59533bbaf59825c6b3c89519b8962085c9ccb9ed626ac1e3dd` |
| v6 compact config | `43c61f71853eb2d32868bcdf01aff9d226f2febbaf593da37d53c01779e828b3` |
| v6 compact `best.pt` | `3faf3c75fff1981988f146a9b0de360dfe3b203472274f472c9829da61279c67` |
| v6 state calibrator | `fa6b08843f09262b188a7ec1f205fe34dd08774b05d0f5b7dd544f0d5dffac19` |
| v6 object calibrator | `f7f8c6a108cadfde51335aba44fab7b927202df72749ae80c572150bddd55f54` |
| v6 calibrated test report | `4f31947f3d4d37b9ad20f9ce750941927d92ea2496ee3d2b99d851dbb721baa0` |

The compact artifacts are under `runs/stack_block_wm_contact_transfer_v5_20261013_compact_v1` and `runs/stack_block_wm_contact_transfer_v6_20261013_compact_v1`; the latter contains `state_calibration_stride16.json`, `object_calibration_stride16.json`, and `test_stride16_calibrated.json`. The test split is now consumed and must not guide another candidate's selection. Ruling: do not freeze v5/v6, authorize A2/A3, or claim world-model task benefit. M2 still needs a robot-observable support-transfer/recovery decision and a better manipulated-object/event predictor; a future candidate requires fresh paired policy validation and a new independent test split. ROS 2 closed-loop and sim-to-real remain untested.

### Stage-one v2 score regression and repository verification

The original 79/80 stage-one pilot used the earlier terminal score. With `wmal.stack_block_results.v2`, a fresh fixed-waypoint, no-model batch on jittered seeds 30000–30039 passed 40/40: every episode had both a stable-stack predicate and confirmed target-block release. Mean cost was 1536 control steps per episode; the run loaded no world model. This clears the proposed 90% pilot floor for this initial-state family under the current score. It does not change the separate disturbed-family result of 42/60 fixed-waypoint successes. The fresh run is under `runs/stack_block_m1_v2_fresh_jitter_30000_30039_20261013_v1`; its `run_manifest.json` SHA-256 is `23db31a137f265fd014a1ddacdd2269f619ad55210dc4ff98b8d7c6734cf9558` and `results.json` SHA-256 is `5df4a536899c58b68a982617d21b5570eb568fe7529b10538ddf61f6347262c7`.

Repository verification used only the existing `wmal` interpreter. All 53 unittest files passed when isolated by file: 329 tests run, 322 passed, 7 skipped. Logs are in `runs/stack_block_m2_recovery_unittest_by_file_20261013`. The one-process full suite could not complete because this Windows environment loaded conflicting `libomp.dll` and `libiomp5md.dll` runtimes; launching `wmal\python.exe` without its `Library\bin` on `PATH` also crashed on a minimal NumPy matrix multiplication. The per-file run prepended that same environment's DLL directory and enabled UTF-8 for configuration files. No alternate Python environment or unsafe OpenMP duplicate-runtime override was used. `git diff --check` passed.
## 2026-10-13 实验产物发布

本轮 `runs/stack_block*` 的 682 个原始文件共 6,088,924,653 字节，按文件边界分成五个 ZIP 附件，保留原有相对路径。逐文件及 ZIP 的大小和 SHA-256 见 [`data/manifests/stack_block_experiments_20261013.json`](../data/manifests/stack_block_experiments_20261013.json)。分包命令为 `python scripts/package_stack_block_experiments.py --output <输出目录>`，使用项目既有 `wmal` 环境。下载五个附件后分别解压到仓库根目录即可还原 `runs/`；以清单校验每个附件和文件。附件包含开发探针、失败和恢复轨迹、训练权重、校准及测试报告。它们是开发证据，不是已冻结或已通过控制验收的世界模型。
