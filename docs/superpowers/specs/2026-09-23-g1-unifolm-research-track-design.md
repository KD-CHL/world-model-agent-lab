# G1 UnifoLM Research Track Design

**Status:** Design approved in conversation; awaiting review of this written specification.

## Goal

Extend World Model Agent Lab into a reproducible G1 manipulation research setup using Unitree's UnifoLM-WMA training approach and published G1 pack-camera data, while retaining the existing state-dynamics planner as a comparable baseline. First-stage evaluation is simulation-only. G1 and Go2 locomotion, bimanual extensions, and real-robot deployment are out of scope.

## Research tracks

1. **State-dynamics baseline:** the existing `JointDynamics`/`NeuralJointDynamics` and `RolloutPlanner` remain a separately measured low-dimensional state prediction and receding-horizon planning track.
2. **WMA decision-making:** initialize from the published WMA-0 Base checkpoint, fine-tune on the selected G1 pack-camera task data, request action chunks through the WMA inference service, validate and execute one action at a time, and re-infer on fresh observations.
3. **WMA simulation:** fine-tune the video prediction mode separately. Evaluate future video predictions as visual predictions; do not report generated frames as simulator ground truth or place them in numeric `PredictionReport.predicted_state` fields.
4. **VLA comparison:** not part of the first implementation. Preserve the conceptual distinction that VLA emits actions and is not a drop-in state-transition model.

## Data and training

- Use the published Unitree G1 pack-camera dataset in its LeRobot V2.1 source form. Preserve the raw dataset as read-only and store prepared data, checkpoints, and large media outside the Git checkout.
- Run Unitree's own WMA converter and trainer from an isolated upstream checkout/environment. Do not vendor WMA code or checkpoints into this MIT-licensed repository.
- Add project-owned dataset manifests and validation around source revision, dataset terms, episode IDs/splits, camera stream, state/action keys and ordering, dimensions, units, normalization, frame/action timing, converter version, and file checksums.
- Split train/validation/test at episode level and derive normalization statistics from train only. Initially use one dataset and one fixed task mixture weight to make failures diagnosable.
- Fine-tune WMA-0 Base into separate decision-making and simulation experiment outputs. Keep the base checkpoint immutable. Record upstream commit, checkpoint hash, dataset revision/license, conversion and training configuration, random seed, hardware, and evaluation summaries.
- Treat the upstream 8-GPU shell configuration as a reference, not a local guarantee. Training launch must accept explicit upstream root, checkpoint, prepared data path, output path, GPU/process count, and mode; fail early for missing paths or incompatible dimensions.

## Runtime architecture

### Observation acquisition

Add a G1-specific observation adapter that pairs RGB frames with robot state in the same episode and step/time window. It must enforce the configured camera name, image dimensions/color order, state ordering and units, monotonic timestamps, and the required history length. Reset, missing/stale frames, mismatched episodes, or incompatible shapes clear/reject the history rather than reusing stale data.

The current G1 config is a placeholder. Before closed-loop evaluation, supply a real G1 MuJoCo model, actuator map, camera, joint limits, action order, and controller configuration whose semantics match the dataset. No implicit truncation, padding, unit conversion, or action interpretation is allowed.

### WMA action execution

Add a separate policy execution path that uses `ActionServiceClient`'s bounded HTTP request/response behavior and the WMA `/predict_action` contract. Map the returned action vector using explicit per-robot configuration, including source/target dimension, ordering, units, normalization statistics, action meaning, and control period. Reject non-finite values, wrong widths, stale observations, unknown joints, and commands outside position/velocity/duration limits.

Execute only the first action step from the predicted chunk. Obtain a fresh synchronized observation before another inference. Record model version/hash and request/episode/step IDs. Preserve the existing `AgentRunner` and ROS planning service path for state-model experiments; route methods explicitly through configuration so one model cannot masquerade as another.

### Simulation-model evaluation

Keep WMA video inference behind a separate visual prediction protocol that accepts synchronized observations and candidate actions and returns predicted frames/features. The first implementation may support offline prediction evaluation before any planner use. Candidate ranking/planning from generated video requires a separately specified task scorer and validation; it is not assumed by this design.

## Evaluation

Compare WMA decision-making and the numeric baseline on the same G1 MuJoCo asset, task instructions, held-out episodes, initial conditions, and success detector. Report task success, action/state error where defined, control steps, inference/planning wall time, actual physics interactions, model calls, failure/termination causes, and checkpoint/data provenance. Report WMA visual prediction metrics separately from action success. Do not use model-reported success claims as ground truth.

## Failure behavior

- Startup rejects missing model/data/checkpoint paths, absent camera or actuator mappings, profile/schema mismatches, and unsupported license metadata.
- Inference timeout, stale/missing sensor data, invalid action chunk, or mapping failure prevents execution and requests a safe hold/stop through the robot adapter; never automatically replay a timed-out physical action.
- Every new action is bound to current robot ID, episode ID, and observation step and revalidated on the execution side.
- External training failures retain logs and do not overwrite the base checkpoint or successful prior run.

## Licensing and dependency isolation

WMA repository code declares CC BY-NC-SA 4.0. Its code, pretrained weights, derived checkpoints, and each dataset's terms must be checked separately; no commercial-use or redistribution permission is inferred. The upstream training/inference environment remains separate from `wmal` to isolate its CUDA/PyTorch/video dependencies. Dataset/model licenses and their attribution are captured in manifests and third-party notices.

## Implementation phases

1. Dataset manifest/schema validation and G1 action/camera configuration contracts; document exact acquisition and external conversion workflow.
2. Reproducible WMA experiment launcher/config generation that targets a separately installed upstream checkout and records provenance.
3. G1 camera/state synchronization, action-chunk mapping, safe one-step ROS execution, and WMA service configuration.
4. G1 MuJoCo asset integration and same-task baseline comparison; reject closed-loop mode until asset/action schema is verified.
5. Separate offline WMA simulation-mode evaluation and, only after metric validation, decide whether to develop a visual candidate scorer.

Each phase requires focused tests for its contracts and an integration check appropriate to its available dependencies. Full WMA fine-tuning requires compatible GPU hardware and is not a unit-test prerequisite.

## Acceptance criteria

- A pinned LeRobot source dataset can be validated and its episode-level provenance/splits recorded without modifying raw data.
- A user can launch WMA fine-tuning from WMA-0 Base in an external environment with explicit inputs/outputs and a saved config/provenance record; decision-making and simulation outputs are distinct.
- The project can call the WMA action service with correctly assembled G1 image/state history and reject malformed, stale, dimensionally incompatible, or unsafe responses before execution.
- A valid action chunk is mapped and only its first action step is sent; another action requires a fresh observation.
- G1 WMA and state-model baseline runs use the same configured simulator/task/success criteria and emit comparable provenance-rich reports.
- No WMA source, weights, or dataset binaries are copied into this repository.

## Known prerequisites and risks

- The repository currently has no complete G1 MuJoCo asset, camera publisher, or verified dataset-to-simulator action map. These are phase-4 prerequisites for meaningful closed-loop G1 results.
- The current host's NVIDIA query reported an NVML driver/library mismatch; training hardware availability must be rechecked on the execution host.
- WMA's sample training command targets eight GPUs; practical batch size and step count must be selected from actual hardware measurements.
- G1 dataset action semantics may not match the available simulation controller. Conversion must preserve metadata and explicitly map semantics instead of assuming joint positions.
- No model or dataset is downloaded as part of this design/specification stage.
