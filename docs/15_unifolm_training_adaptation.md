# 基于 UnifoLM 的机器人世界模型训练与微调路线

调研日期：2026-09-23。本文把上游训练流程作为外部研究工作流，说明如何获取数据、组织训练、保存模型以及接回本项目。上游模型代码和权重不复制进本仓库。

## 研究目标和模型分工

本项目当前的 `JointDynamics` / `NeuralJointDynamics` 学习低维关节状态转移，服务于 `RolloutPlanner` 的候选动作 rollout。UnifoLM-WMA 的核心世界模型预测动作条件下的未来视频，并提供 decision-making action head；UnifoLM-VLA 根据图像、指令和机器人状态预测动作 chunk。三者输入输出和训练目标不同，不能互换 checkpoint。

建议实验分三条对照线：

| 轨道 | 初始化/数据 | 训练目标 | 在本项目中的角色 |
| --- | --- | --- | --- |
| 状态动力学基线 | 本地 MuJoCo 或映射后的关节轨迹 | 预测下一关节状态 | `RolloutPlanner` 的数值状态模型 |
| UnifoLM-WMA | Open-X 微调的 WMA-0 Base，再用目标任务数据后训练 | 动作条件未来视频；decision-making 和/或 simulation mode | 视觉预测/策略服务，通过独立适配器提供推理 |
| UnifoLM-VLA | UnifoLM-VLM-Base 与目标任务 LeRobot/RLDS 数据 | 指令条件动作 chunk | Agent 的动作策略对照，不当作状态转移模型 |

首个端到端研究建议选用与开源数据/权重相匹配的 Unitree G1 桌面操作任务（如 pack camera），固定主相机、机器人状态顺序和动作顺序。项目中的 G1 配置目前仍是空资产模板，必须先补齐 MuJoCo 模型、关节限位、执行器映射与相机，才可做本地仿真闭环；否则先在上游提供的部署/评估路径评测模型，并仅将其作为离线模型结果。

## 上游训练方法摘要

### UnifoLM-WMA

上游 README 将流程分为：

1. 在 Open-X 数据上对视频生成世界模型进行微调，得到 WMA-0 Base。
2. 用下游任务数据后训练 decision-making mode。
3. 用下游任务数据联合后训练 decision-making 与 simulation；若只做决策策略实验，可选择 decision-only。上游公开配置没有证据支持“simulation-only”模式，本项目不提供此选项。

训练配置中包含视频片段、主视角图像、语言指令、机器人 state/action；示例 `agent_state_dim` 和 `agent_action_dim` 为 16，最大自由度假设为 16，超过时需同步修改配置。示例训练配置使用 batch size 8、混合精度、序列长度 16；官方 shell 样例按 8 GPU 启动，属于大模型训练配置，不能据此假定单卡可直接复现。

WMA 的数据准备脚本以 Hugging Face LeRobot V2.1 为输入，将视频、每 episode 的 `observation.state`/`action` HDF5、统计量和 CSV 元信息整理为上游 `WMAData` 可读取的结构。上游训练器按 `dataset_and_weights` 混合数据集；各数据权重需合计为 1。建议初期只选一个目标任务数据集，避免混合比例掩盖数据/动作映射问题。

参考入口：[WMA README](https://github.com/unitreerobotics/unifolm-world-model-action)、[训练配置](https://github.com/unitreerobotics/unifolm-world-model-action/blob/main/configs/train/config.yaml)、[数据转换器](https://github.com/unitreerobotics/unifolm-world-model-action/blob/main/prepare_data/prepare_training_data.py)。

### UnifoLM-VLA

上游说明要求 LeRobot 数据先转为 HDF5，再构建 RLDS 并注册数据集配置、动作/本体状态维数、chunk 长度和归一化方式。训练脚本使用 Accelerate/DeepSpeed，并在样例中按 8 个进程启动。它适合拿来回答“视觉语言条件策略能否改善任务动作生成”，而不是用于当前 `model.predict(state, target, duration)` 接口。

参考入口：[VLA README](https://github.com/unitreerobotics/unifolm-vla)、[示例训练脚本](https://github.com/unitreerobotics/unifolm-vla/blob/main/scripts/run_scripts/run_unifolm_vla_train.sh)、[机器人维数与归一化配置](https://github.com/unitreerobotics/unifolm-vla/blob/main/src/unifolm_vla/rlds_dataloader/constants.py)。

## 数据获取与目录约定

1. 从 [UnifoLM-WMA 数据集说明](https://github.com/unitreerobotics/unifolm-world-model-action#dataset) 或 [Unitree Hugging Face datasets](https://huggingface.co/unitreerobotics/datasets) 选择任务数据；记录 Hugging Face dataset repo、revision、下载时间和该 dataset 自身的 license。
2. 大型数据、视频和权重存放在仓库外的持久数据盘，避免提交到 Git。建议：

```text
/data/robot-world-model/
  raw/lerobot/<dataset-repo>/       # 下载的只读原始数据
  prepared/wma/<dataset-name>/      # WMA 上游转换结果
  prepared/vla/<dataset-name>/      # VLA 的 HDF5/RLDS 转换结果
  manifests/<dataset-name>.json     # revision、license、机器人、相机、维数、转换版本、划分
  checkpoints/wma/<experiment>/    # 上游训练 checkpoint
  checkpoints/vla/<experiment>/
```

3. 保留原始 episode ID，并按 episode/task/场景划分 train、validation、test。禁止把同一 episode 的相邻帧分到不同 split。统计量只从 train 计算；保存 action/state 的单位、顺序、归一化范围和控制周期。
4. WMA 当前转换器说明针对 LeRobot V2.1 的目录布局，具体下载版本应先核对 `meta/info.json`、`meta/tasks.jsonl`、episode parquet 与视频流是否匹配其预期。VLA 的 LeRobot 转换链不同，不复用 WMA 输出目录。

本项目提供 `scripts/prepare_wma_dataset.py`，默认只校验本地数据；只有显式加 `--download` 才下载。示例使用 WMA README 对应的 G1 pack-camera 数据集，需先确认数据集页面许可及访问条件：

```bash
python -m pip install -e '.[robot-data]'
PYTHONPATH=src python scripts/prepare_wma_dataset.py \
  --root /data/robot-world-model/raw/lerobot/g1_pack_camera \
  --dataset-id unitreerobotics/G1_Dex1_MountCameraRedGripper_Dataset \
  --revision v2.1 --license-id '<核实后的数据集许可标识>' \
  --download --seed 4 --hash-episode-files
```

脚本验证固定的 LeRobot V2.1 `meta/`、parquet 与 `videos/` 布局，并在原始 episode 粒度生成确定性 train/validation/test manifest。完整哈希会读取全部视频和 parquet，数据很大时可先省略 `--hash-episode-files`。通过许可核验后可显式调用上游转换器：

```bash
PYTHONPATH=src python scripts/prepare_wma_dataset.py \
  --root /data/robot-world-model/raw/lerobot/g1_pack_camera \
  --dataset-id unitreerobotics/G1_Dex1_MountCameraRedGripper_Dataset \
  --revision v2.1 --license-id '<核实后的数据集许可标识>' \
  --convert --wma-root /opt/unifolm-world-model-action \
  --prepared-root /data/robot-world-model/prepared/wma \
  --manifest /data/robot-world-model/manifests/g1_pack_camera.json
```

此转换命令在 WMA 上游仓库中运行，不属于当前项目的 `scripts/collect.py`。转换后先检查 episode 数量、视频解码、state/action shape、帧数对齐及数据集统计量，再开始昂贵的训练。

## WMA 上游训练命令

WMA 使用独立环境，避免与本项目 `wmal` 环境的依赖互相覆盖。以下安装步骤来自上游 README；CUDA/PyTorch 仍需按训练机驱动匹配：

```bash
conda create -n unifolm-wma python=3.10.18 -y
conda activate unifolm-wma
conda install pinocchio=3.2.0 ffmpeg=7.1.1 -c conda-forge -y
git clone --recurse-submodules https://github.com/unitreerobotics/unifolm-world-model-action.git
cd unifolm-world-model-action
pip install -e .
pip install -e external/dlimp
```

从 [WMA Hugging Face 模型页](https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Base) 获取 Base checkpoint。项目脚本生成独立配置，不修改上游 checkout。`--dataset-key` 必须是当前 WMA checkout 已注册的 loader 名称；默认模板中的其余训练参数仍由研究者按 GPU 显存、数据维数和验证方案审阅：

```bash
PYTHONPATH=src python scripts/train_unifolm_wma.py \
  --upstream-root /opt/unifolm-world-model-action \
  --checkpoint /data/robot-world-model/checkpoints/wma/WMA-0-Base \
  --prepared-data /data/robot-world-model/prepared/wma/g1_pack_camera \
  --dataset-key unitree_g1_pack_camera --mode decision \
  --config-output /data/robot-world-model/configs/g1_pack_decision.yaml \
  --run-name g1-pack-decision-s0 \
  --run-output /data/robot-world-model/runs/g1-pack-decision-s0 \
  --processes 1 --dry-run
```

先人工检查生成配置和 dry-run 命令，再去掉 `--dry-run` 开始训练。`--processes` 是每节点 GPU/进程数，可用 `--cuda-visible-devices 0,1` 明确选择设备；训练配置、基础权重哈希、命令与日志记录在独立 run 目录。decision-only 与 joint decision+simulation 用不同 `--mode`、配置和输出目录。不要未经显存评估就照搬上游 8-GPU 示例。

若需要完全手动控制，也可在上游 `scripts/train.sh` 设置实验名 `name` 和 `save_root`，然后执行：

```bash
bash scripts/train.sh
```

上游示例默认 8 张 GPU，配置示例最多 300000 steps，并每 1000 steps 保存 checkpoint。训练耗时和显存取决于视频尺寸、batch size 与 GPU；开始完整训练前，先按上游要求做小规模启动验证。run 目录保存日志和权重，模型产物应放在 Git 仓库外。

VLA 策略对照使用其独立环境和 LeRobot→HDF5→RLDS 转换链。上游说明推荐 CUDA 12.4，示例训练脚本默认 8 个 processes；在 `run_unifolm_vla_train.sh` 设置 `base_vlm`、`oxe_data_root`、`data_mix`、输出目录和训练步数，再按实际 GPU 数调整 Accelerate 配置。VLA checkpoint 经其 inference/deployment server 调用，不由本项目 `load_dynamics` 读取。

## 微调阶段、checkpoint 和评估

WMA 微调以其官方模型结构和配置运行：

1. 将 WMA-0 Base checkpoint 放入外部持久目录，在 `configs/train/config.yaml` 设置 `model.pretrained_checkpoint`。
2. 设置训练数据路径、数据集名称和采样权重；确认图像尺寸、state/action 维度、视频长度、主视角和归一化方式与数据一致。
3. decision-making 与 simulation mode 分开做实验。先只解冻/训练配置中设计为可训练的模块，固定随机种子、训练步数、GPU 数和验证间隔。
4. checkpoint、日志和配置快照写到独立 `save_root/experiment_name`。保留上游 checkpoint 与下游微调 checkpoint，不覆盖基础模型。
5. 在 held-out episode 上分别评估未来视频预测、动作误差、任务成功率与推理时延。模型生成视频只作为预测，不算仿真真值；action head 的成功率需在真实环境或高保真仿真测量。

VLA 使用其上游 RLDS loader 和训练配置完成 SFT/微调，另存 run/checkpoint；用相同数据划分和相同机器人任务与本项目 Agent/规划器比较。

## 接回本项目的接口

当前仓库保留 numeric planner，并增加相互独立的策略与视觉预测连接方向：

- 数值状态模型由 `load_dynamics(checkpoint)` 加载，提供 `predict(state, targets, duration_s)`，供 `RolloutPlanner` 在线 rollout。现有 JSON/`.pt` checkpoint 只属于本项目 `JointDynamics`/`NeuralJointDynamics` 格式，不能读取 WMA/VLA checkpoint。
- `ActionServiceClient` 连接 `/predict_action` 风格的独立 HTTP 服务；`run_agent.py --mode wma-policy` 发送同步相机历史/状态，只映射并执行 chunk 的第一步，然后要求新的同步观测。必须提供 `camera`、`wma_policy.state_order`、action-to-joint 映射、单位、归一化与控制周期。可配置 `success_plugin`（`module:factory` 返回 `(instruction, observation) -> bool` 判据）；无成功检测器时只会在预算耗尽后退出，不会伪报成功。该服务必须由用户在上游或自己的环境中启动，本项目没有捏造未公开的 WMA 服务端。
- `VideoPredictionProvider` 是独立离线视觉预测协议；通过显式插件对 held-out 序列评估 MAE/PSNR，输出与数字状态模型分开的 JSON/JSONL。不得把未来帧、策略 action chunk 或未经校准的模型分数塞进 `PredictionReport.predicted_state`。

G1 模板的模型路径、关节限位、actuator map 和相机目前仍为空；它只是接口模板，不能用于闭环。需先准备匹配的 G1 MuJoCo 资产、相机名、数据到模型动作 schema 的明确映射和成功判据。训练后还需由上游/自建策略服务提供兼容 `/predict_action` 的 HTTPS 或本机 HTTP endpoint。推荐顺序：核对数据许可和 manifest → 上游 dry-run/小规模训练 → held-out 离线评估 → 验证服务协议和动作映射 → 准备匹配 G1 资产后做仿真闭环 → 最后开展对照实验。WMA/VLA 训练依赖与当前 `wmal` 环境分开，避免 CUDA/PyTorch/FlashAttention 依赖冲突。

## 许可证与可复现性

WMA GitHub 仓库当前声明 **CC BY-NC-SA 4.0**。遵守署名、非商业和相同方式共享等条款；不要将其源码、权重或其衍生 checkpoint 混入本项目 MIT 许可产物并宣称可任意商用。代码、预训练权重和每个 Hugging Face 数据集的许可证须分别核查。

检索时 VLA 仓库根目录未发现 `LICENSE` 文件；在确认仓库代码、模型卡和各 dataset 的明确许可前，只记录为研究参考，不复制源码或权重，也不对外再分发。许可证缺失不代表自动获得授权。

训练记录最少包含：上游 repo commit、模型 checkpoint revision/hash、数据集 repo/revision/license、转换器 commit、数据 manifest、episode 划分、state/action schema、相机配置、归一化统计、训练配置、随机种子、GPU/显存、训练步数、验证指标和推理服务版本。
