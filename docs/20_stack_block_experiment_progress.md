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

Windows compatibility was repaired in the interactive visual runner (console/file/pipe stdin polling), same-size JSONL rewrite detection (content verification no longer depends on timestamp granularity), symlink tests (skip only the unavailable symlink assertion), and ROS 2 shell-helper tests (Git Bash path and POSIX path conversion). The latest full suite exited successfully: 285 tests, 278 passed, 7 skipped, zero failures/errors. It ran in the `tf-agent` environment with MuJoCo and PyTorch, `PYTHONPATH=src`, and Git Bash on `PATH`.
