# 基于 UnifoLM 的机器人世界模型训练与微调路线

> 2026-09-28 研究方向更新：以 [预测可信度研究方案](21_research_plan_prediction_reliability.md) 和 [当前数据清单](22_world_model_dataset_catalog.md) 为准。本页保留上游工具参考；decision/joint 训练不等于 prediction-only，现有 split 清单也不代表上游训练已执行隔离。项目原始数据现存于仓库内 `data/raw/lerobot/`，受 Git 排除规则保护。

调研日期：2026-09-23。本文把上游训练流程作为外部研究工作流，说明如何获取数据、组织训练、保存模型以及接回本项目。上游模型代码和权重不复制进本仓库。

## 研究目标和模型分工

本项目当前的 `JointDynamics` / `NeuralJointDynamics` 学习低维关节状态转移，服务于 `RolloutPlanner` 的候选动作 rollout。UnifoLM-WMA 的核心世界模型预测动作条件下的未来视频，并提供 decision-making action head；UnifoLM-VLA 根据图像、指令和机器人状态预测动作 chunk。三者输入输出和训练目标不同，不能互换 checkpoint。

建议实验分三条对照线：

| 轨道 | 初始化/数据 | 训练目标 | 在本项目中的角色 |
| --- | --- | --- | --- |
| 状态动力学基线 | 本地 MuJoCo 或映射后的关节轨迹 | 预测下一关节状态 | `RolloutPlanner` 的数值状态模型 |
| UnifoLM-WMA | Open-X 微调的 WMA-0 Base，再用目标任务数据后训练 | 动作条件未来视频；decision-making 和/或 simulation mode | 视觉预测/策略服务，通过独立适配器提供推理 |
| UnifoLM-VLA | UnifoLM-VLM-Base 与目标任务 LeRobot/RLDS 数据 | 指令条件动作 chunk | Agent 的动作策略对照，不当作状态转移模型 |

首个端到端研究建议选用与开源数据/权重相匹配的 Unitree G1 桌面操作任务（如 pack camera），固定主相机、机器人状态顺序和动作顺序。项目已接入 Unitree G1 29-DOF MuJoCo 模型的固定底座衍生版、关节限位、PD 电机映射与 `pack_camera` 渲染相机，可做本地关节物理仿真及 RGB 渲染；此世界坐标相机尚未与数据集外参标定。该模型的手部是无驱动的 rubber-hand 网格，pelvis 固定；数据集与仿真关节映射尚未确认，故仿真资产就绪不等于 WMA 任务闭环就绪。

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

1. 首选 [Unitree G1 Dex1 MountCameraRedGripper 数据集](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_MountCameraRedGripper_Dataset)：上游 WMA 将其列为 G1 数据集；Hugging Face 页面标注 Apache-2.0、LeRobot v2.1、201 episodes / 172,649 frames、约 5.83 GB，state/action 均为 16 维。数据有多个视角；WMA 只支持主视角，当前示例选用 `observation.images.cam_left_high`。数据 commit 固定为 `3f94cbcbbb63dde2440c4a3bebd2f827e41d52df`（上游 `v2.1` tag 指向该数据版本）。WMA 官方训练配置中的 dataset key 是 `unitree_g1_pack_camera`，因此转换产物目录名、CSV 名和训练配置 key 必须严格一致。
2. 大型数据、视频和权重存放在仓库外的持久数据盘，避免提交到 Git。建议：

```text
/data/robot-world-model/
  raw/lerobot/unitree_g1_pack_camera/ # 下载的只读原始数据，目录名供上游 dataset key 使用
  prepared/wma/                       # 转换器会创建 unitree_g1_pack_camera/ 子目录
  prepared/vla/<dataset-name>/      # VLA 的 HDF5/RLDS 转换结果
  manifests/<dataset-name>.json     # revision、license、机器人、相机、维数、转换版本、划分
  checkpoints/wma/<experiment>/    # 上游训练 checkpoint
  checkpoints/vla/<experiment>/
```

3. 保留原始 episode ID，并按 episode/task/场景划分 train、validation、test。禁止把同一 episode 的相邻帧分到不同 split。统计量只从 train 计算；保存 action/state 的单位、顺序、归一化范围和控制周期。
4. WMA 当前转换器说明针对 LeRobot V2.1 的目录布局，具体下载版本应先核对 `meta/info.json`、`meta/tasks.jsonl`、episode parquet 与视频流是否匹配其预期。VLA 的 LeRobot 转换链不同，不复用 WMA 输出目录。

本项目提供 `scripts/prepare_wma_dataset.py`，默认只校验本地数据；只有显式加 `--download` 才下载。数据页面的完整 commit SHA 已在命令中固定：

```bash
python -m pip install -e '.[robot-data]'
hf auth login
PYTHONPATH=src python scripts/prepare_wma_dataset.py \
  --root /data/robot-world-model/raw/lerobot/unitree_g1_pack_camera \
  --dataset-id unitreerobotics/G1_Dex1_MountCameraRedGripper_Dataset \
  --revision 3f94cbcbbb63dde2440c4a3bebd2f827e41d52df --license-id apache-2.0 \
  --camera-key observation.images.cam_left_high \
  --download --seed 4
```

`observation.images.cam_left_high` 已在数据集 `meta/info.json` 中列出；若改用其他版本，仍需重新核实相机 key。脚本会拒绝 `main` 和短 SHA，验证 LeRobot V2.1 `meta/`、parquet 与视频布局，并按 episode 生成确定性 train/validation/test manifest。若希望对所有 episode 数据文件计算完整 SHA256，再加 `--hash-episode-files`（会顺序读取数 GB 数据）。

转换器会处理完整数据集；当前本地 manifest 的 train/validation/test 划分不会自动传给 WMA 上游训练器。因此需在论文实验中明确 WMA 的评估划分策略：用转换产物中的 episode metadata 实施 episode 级切分，或先生成独立 train-only 源数据集再转换；不能把当前脚本写出的 split 当成已应用的训练隔离。

官方 `prepare_training_data.py` 会把所有视角写进 CSV，且只对 AV1 输入生成目标视频、跳过其他编码。本项目封装在转换后会自动将 CSV 限定为指定相机，并把上游遗漏的源视频补到目标目录；仍需对转换结果做一次视频可解码性检查。封装不会修改上游仓库。

验证 manifest 无误后，可显式调用官方转换器：

```bash
PYTHONPATH=src python scripts/prepare_wma_dataset.py \
  --root /data/robot-world-model/raw/lerobot/unitree_g1_pack_camera \
  --dataset-id unitreerobotics/G1_Dex1_MountCameraRedGripper_Dataset \
  --revision 3f94cbcbbb63dde2440c4a3bebd2f827e41d52df --license-id apache-2.0 \
  --camera-key observation.images.cam_left_high \
  --convert --wma-root /opt/unifolm-world-model-action \
  --prepared-root /data/robot-world-model/prepared/wma \
  --manifest /data/robot-world-model/manifests/unitree_g1_pack_camera.json
```

此转换命令在 WMA 上游仓库中运行，不属于当前项目的 `scripts/collect.py`。转换后先检查 episode 数量、视频解码、state/action shape、帧数对齐及数据集统计量，再开始昂贵的训练。

## WMA 上游训练命令

所有项目命令统一使用 `wmal`，不创建或切换上游项目的虚拟环境。WMA 作为可选外部源码依赖接入；它的完整依赖组合尚未在统一环境中验收，因此不能直接照搬上游安装步骤覆盖已验证的 PyTorch/CUDA。先审核依赖解析结果，若存在版本冲突则先修复适配，不用另一个环境绕开。

```bash
conda activate wmal
python -m pip install -e '.[robot-data]'
# 已取得上游源码后，仅做安装计划检查，不自动更改依赖。
python -m pip install --dry-run -e /opt/unifolm-world-model-action
python -m pip install --dry-run -e /opt/unifolm-world-model-action/external/dlimp
```

从 [WMA Hugging Face 模型页](https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Base) 登录并接受访问条件后下载 `unifolm_wma_base.ckpt`。模型页的当前 commit 固定为 `b7bb75323e8c614fb9b0d25905d07a0fc11e1793`。官方 WMA 配置和训练记录使用的 G1 loader key 为 `g1_pack_camera`；使用当前官方 WMA checkout 前，仍要确认该 key 在其 config/loader 中存在，且转换后的 CSV 只引用选定的主视角。项目脚本生成独立配置，不修改上游 checkout：

```bash
PYTHONPATH=src python scripts/train_unifolm_wma.py \
  --upstream-root /opt/unifolm-world-model-action \
  --checkpoint /data/robot-world-model/checkpoints/wma/WMA-0-Base/unifolm_wma_base.ckpt \
  --prepared-data /data/robot-world-model/prepared/wma \
  --dataset-key unitree_g1_pack_camera --mode decision \
  --config-output /data/robot-world-model/configs/g1_pack_decision.yaml \
  --run-name g1-pack-decision-s0 \
  --run-output /data/robot-world-model/runs/g1-pack-decision-s0 \
  --processes 1 --dry-run
```

先人工检查生成配置和 dry-run 命令，再去掉 `--dry-run` 开始训练。`--processes` 是每节点 GPU/进程数，可用 `--cuda-visible-devices 0,1` 明确选择设备；训练配置、基础权重哈希、命令与日志记录在独立 run 目录。decision-only 与 joint decision+simulation 用不同 `--mode`、配置和输出目录。不要未经显存评估就照搬上游 8-GPU 示例。

```bash
hf download unitreerobotics/UnifoLM-WMA-0-Base unifolm_wma_base.ckpt \
  --repo-type model --revision b7bb75323e8c614fb9b0d25905d07a0fc11e1793 \
  --local-dir /data/robot-world-model/checkpoints/wma/WMA-0-Base
```

随后将上一条训练命令的 `--checkpoint` 参数改为 `/data/robot-world-model/checkpoints/wma/WMA-0-Base/unifolm_wma_base.ckpt`。模型为 gated repo，须先在网页接受访问条件并通过 `hf auth login` 登录。注意：Hub 的 license 元数据标为 Apache-2.0，但模型卡正文指向 CC BY-NC-SA 4.0；这属于许可信息冲突，使用前应以权重随附许可证/权利人说明为准，未澄清前按限制更严格的 CC BY-NC-SA 4.0 处理，不要据此宣称可商业使用。

若需要完全手动控制，也可在上游 `scripts/train.sh` 设置实验名 `name` 和 `save_root`，然后执行：

```bash
bash scripts/train.sh
```

上游示例默认 8 张 GPU，配置示例最多 300000 steps，并每 1000 steps 保存 checkpoint。训练耗时和显存取决于视频尺寸、batch size 与 GPU；开始完整训练前，先按上游要求做小规模启动验证。run 目录保存日志和权重，模型产物应放在 Git 仓库外。

VLA 策略对照也只能接入本项目 `wmal`，其 LeRobot→HDF5→RLDS 转换链和完整依赖尚未在该环境验证。上游 CUDA/PyTorch 版本要求不能直接替换当前 GPU 构建；应先完成兼容性审核。示例训练脚本默认 8 个 processes；在 `run_unifolm_vla_train.sh` 设置 `base_vlm`、`oxe_data_root`、`data_mix`、输出目录和训练步数，再按实际 GPU 数调整 Accelerate 配置。VLA checkpoint 经其 inference/deployment server 调用，不由本项目 `load_dynamics` 读取。

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
- `ActionServiceClient` 连接 `/predict_action` 风格的独立 HTTP 服务；`run_agent.py --mode wma-policy` 发送同步相机历史/状态，只映射并执行 chunk 的第一步，然后要求新的同步观测。必须提供 `camera`、`wma_policy.state_order`、action-to-joint 映射、单位、归一化与控制周期。可配置 `success_plugin`（`module:factory` 返回 `(instruction, observation) -> bool` 判据）；无成功检测器时只会在预算耗尽后退出，不会伪报成功。服务端也应在本项目 `wmal` 中完成依赖兼容性适配后启动；目前未提供或验证完整 WMA 服务端，不应把客户端协议示例当作可用服务。
- `VideoPredictionProvider` 是独立离线视觉预测协议；通过显式插件对 held-out 序列评估 MAE/PSNR，输出与数字状态模型分开的 JSON/JSONL。不得把未来帧、策略 action chunk 或未经校准的模型分数塞进 `PredictionReport.predicted_state`。

G1 固定底座模型路径、29 个关节限位、全部 actuator map 和 `pack_camera` 已在 `configs/robots/g1_fixed_base.json` 配置，可运行命令行 MuJoCo 仿真。仍需确认数据集 state/action 顺序、单位、归一化与模型控制语义，添加有驱动的 gripper（若研究任务需要抓取）和成功判据。训练后还需由上游/自建策略服务提供兼容 `/predict_action` 的 HTTPS 或本机 HTTP endpoint。推荐顺序：核对数据许可和 manifest → 统一 `wmal` 内依赖兼容性审核 → 上游 dry-run/小规模训练 → held-out 离线评估 → 验证服务协议和动作映射 → 在固定底座仿真验证可映射动作 → 最后开展对照实验。出现 CUDA/PyTorch/FlashAttention 依赖冲突时，应修复可选上游适配并重新验证，不切换到其他项目环境。

## 许可证与可复现性

WMA GitHub 仓库声明 **CC BY-NC-SA 4.0**；Base 模型页的 Hub license 元数据标为 Apache-2.0，但卡片正文又写 CC BY-NC-SA 4.0，存在冲突。权重使用前应以随附许可证/权利人说明澄清，在此之前按较严格的 CC BY-NC-SA 4.0 限制处理，不作商业用途，也不将源码、权重或衍生 checkpoint 混入本项目 MIT 产物并宣称可任意商用。数据集页另标 Apache-2.0；源码、权重和数据分别核查许可。

检索时 VLA 仓库根目录未发现 `LICENSE` 文件；在确认仓库代码、模型卡和各 dataset 的明确许可前，只记录为研究参考，不复制源码或权重，也不对外再分发。许可证缺失不代表自动获得授权。

训练记录最少包含：上游 repo commit、模型 checkpoint revision/hash、数据集 repo/revision/license、转换器 commit、数据 manifest、episode 划分、state/action schema、相机配置、归一化统计、训练配置、随机种子、GPU/显存、训练步数、验证指标和推理服务版本。
