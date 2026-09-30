# 视觉世界模型：数据、训练、微调和 Agent 完整命令

## 已验证环境

所有命令统一使用本项目的 `wmal` 环境。PyTorch CUDA 12.8、MuJoCo、ONNX Runtime、PyArrow 和 PyAV 均安装在这个环境，不依赖其他项目环境。本机 RTX 5060 Ti 为 8GB，需要包含 sm_120 的 PyTorch 构建；项目固定 `torch==2.11.0+cu128`。

在仓库根目录执行：

```bash
cd /home/chl/GitHub/CodeSpace/world-model-agent-lab-main
conda activate wmal
export PYTHONPATH="$PWD/src"
python -c "import torch,mujoco; print(torch.__version__, torch.cuda.is_available(), mujoco.__version__)"
```

首次安装使用仓库的 `environment.yml`。已有 `wmal` 环境补齐依赖：

```bash
conda activate wmal
export PYTHONPATH="$PWD/src"
python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[simulation,walking,visual-world,visual-data,robot-data]'
python -m pip check
```

普通训练/仿真启动时，将 PYTHONPATH 设置为本项目 `src`，不要追加 shell 自动加入的 ROS Python 路径或其他项目环境。系统 ROS 的 Python 3.12 包与 `wmal` 的 Python 3.10 ABI 不兼容；ROS 接入仍需单独验证。原始数据和 `runs/` 已在 gitignore 排除，本轮生成数据放 `data/processed/`，权重、校准、日志放 `runs/`，未提交大型文件。

## 已有模型到底用什么训练

不是空数据训练。窗口当前加载的 `visual_g1_release_finetuned_20260930` 使用
`data/processed/visual_g1_20260930/manifest.json`：采集器在 MuJoCo 中发出受限关节动作，再记录动作前后的 RGB 和状态。
40 条轨迹每条 16 次动作，共 640 次转移；训练/验证/校准/测试为 24/4/8/4 条。
训练样本是 `(当前图像、当前状态、候选动作段) → (实际未来图像、实际未来状态)`，目标来自仿真执行，不是模型自己的预测。
基模是本项目 CNN/GRU 小网络随机初始化训练 30 轮，随后同一份仿真数据做冻结编码器的 2 轮实验。
这不是 UniFoLM 预训练模型的开源数据微调，不能据此宣称已经学会堆叠；场景元数据中的 G1_Stack_Block 链接只是建模参考，不是该模型实际读取的下载数据。

独立的 `runs/lerobot_visual_20260930/` 则使用已下载的 G1 MountCamera 数据，经 AV1/parquet 对齐导入：
`data/processed/lerobot_visual_av1_20260930/manifest.json`，20 条 ×32 帧，620 次转移，曾训练 3 轮。
它有 16 维操作动作，只完成离线训练链测试，未接到两关节仿真控制器。
训练报告 `training_report.json` 的 `metadata.training_source`、episode ID、内容哈希和父模型版本可核查来源。
原始视频、NPZ 和权重都不在 Git 中；缺少 manifest/数据文件会明确失败，不会凭空生成训练样本或可用权重。

## 训练进度条

视觉训练和 `--pretrained` 微调共享进度条，导航状态网络 `train_motion_network.py train` 也使用相同显示：

- 开始时打印实际数据文件、来源类型、训练/验证 episode 与窗口数量、设备。
- `Visual epochs` / `Fine-tune epochs` / `Motion epochs` 显示完成轮数、百分比、耗时和 ETA。
- `Train i/N` / `Validation i/N` 显示实际批次数、集成成员和当前累计平均损失；每轮摘要完整显示 train_loss、val_loss、best，窄终端也不会丢失这些指标。
- 一个 epoch 只有在训练、验证和该轮权重/历史保存完成后才增加；中断或失败会关闭显示，不把未完成轮次报成完成。
- 进度写 stderr；最终 JSON/原有 JSONL 事件仍写 stdout，磁盘指标不变。训练器 Python API 默认安静，CLI 默认开启。

批量运行加 `--no-progress` 可关闭所有训练提示，不改变模型结果。ETA 是已完成训练速度的估计，开始时为 `?`；首次数据哈希校验/加载在正式进度条之前，并有单独状态提示。
进度依赖 `tqdm` 已加入本项目 `learning` / `visual-world` extra，不需要其他环境。该改动不接管上游 UniFoLM 自己的训练循环，也未给旧的其他训练入口添加进度条。

本机已导入的数据可直接启动（输出目录必须新建/为空）：

```bash
python scripts/visual_world.py train \
  --manifest data/processed/lerobot_visual_av1_20260930/manifest.json \
  --output runs/lerobot_with_progress_run1 --device cuda --epochs 30
# 同样的命令加 --no-progress 可关闭提示。
```

## 路线 A：真实 MuJoCo 图像学习与可执行闭环

下列目录应当不存在或为空，入口会拒绝覆盖已有实验。示例是新的 run 名，不覆盖本轮记录。

```bash
export MUJOCO_GL=egl
python scripts/visual_world.py collect \
  --output data/processed/visual_g1_run1 \
  --episodes 40 --steps 16 --image-size 64 --seed 30

python scripts/visual_world.py train \
  --manifest data/processed/visual_g1_run1/manifest.json \
  --output runs/visual_g1_run1 --device cuda --epochs 30

python scripts/visual_world.py calibrate \
  --manifest data/processed/visual_g1_run1/manifest.json \
  --checkpoint runs/visual_g1_run1/best.pt \
  --output runs/visual_g1_run1/calibration.json --horizon 4 --alpha .2

python scripts/visual_world.py evaluate \
  --manifest data/processed/visual_g1_run1/manifest.json \
  --checkpoint runs/visual_g1_run1/best.pt \
  --calibration runs/visual_g1_run1/calibration.json \
  --output runs/visual_g1_run1/test_metrics.json --horizon 4

python scripts/visual_skill_agent.py \
  --checkpoint runs/visual_g1_run1/best.pt \
  --calibration runs/visual_g1_run1/calibration.json \
  --output runs/visual_agent_run1 --baseline A3 --goal .45 .95 --max-cycles 30
```

40 episode 分为 24 train、4 validation、8 calibration、4 test。alpha=.2 是小规模 smoke 的区间参数，不是 80% 动作成功保证。alpha=.1 需至少 9 个校准 episode，建议正式实验远多于该最小值。每步控制周期 .2 秒，输入两关节目标增量≤.06rad；状态包括两关节实测位置和两个伺服目标。

### 本机已经生成的检查点：打开并保持交互窗口

发布前已在 `wmal` 重新生成带累计数据记录的 `runs/visual_g1_release_finetuned_20260930/best.pt` 与 `calibration.json`，使用：

```bash
conda activate wmal
cd /home/chl/GitHub/CodeSpace/world-model-agent-lab-main
export PYTHONPATH="$PWD/src"
unset MUJOCO_GL
python scripts/visual_skill_agent.py \
  --checkpoint runs/visual_g1_release_finetuned_20260930/best.pt \
  --calibration runs/visual_g1_release_finetuned_20260930/calibration.json \
  --output runs/visual_viewer_run1 --baseline A3 --goal .45 .95 \
  --error-budget .24 --viewer
```

任务结束后窗口保持打开，控制器保持最后目标。在终端输入 `0.4 0.9` 启动新目标，输入 `quit` 或关闭窗口退出。执行异常会取消未完成目标、冻结物理积分并锁定会话，输入 `reset` 明确重置场景后才允许新目标；不会自动恢复或重发。终端 EOF 不会反复刷提示或自动关闭窗口。注意这不是向前行走或抓取积木命令。交互窗口路径提供实现，但本轮实际验证是 EGL 无窗口执行，不宣称 GUI 已人工验收。

调节 `--error-budget .35` 可以研究更保守的前缀，但不可为了看起来完成任务而用最终测试结果反复调阈值；预算应在开发数据确定后冻结。A0/A1/A2 用 `--baseline` 切换，每种方法使用新的 output 目录。A3 拒绝时不会回退为未验证动作。

### 微调已有本项目网络

```bash
python scripts/visual_world.py train \
  --manifest data/processed/visual_g1_run1/manifest.json \
  --pretrained runs/visual_g1_run1/best.pt --freeze-encoder \
  --output runs/visual_g1_finetune_run1 --device cuda --epochs 10
```

保持动作顺序/状态顺序/相机/周期/分辨率/网络宽度与预训练检查点一致；更换这些字段会拒绝。新检查点需要重新校准，不能复用旧 model_version 的校准文件。此命令是已训练紧凑网络的微调，不接受 UniFoLM-WMA ckpt。

检查点累计保存祖先训练/选型 episode 的 ID 和内容哈希：祖先训练 episode 不能改作后续验证数据，所有训练/选型 episode 不能改作校准或测试，即使改名也会检查内容。旧微调权重若缺少祖先数据记录，不允许离线校准、评估或继续微调，需从有记录的基模重新生成；不会默认为“未见过的数据”。

## 路线 B：已下载的开源操作数据用于离线训练

本机原数据在 `data/raw/lerobot/unitree_g1_pack_camera/`，追踪清单为 `data/manifests/unitree_g1_pack_camera.json`。其实际任务是 mount camera，不应由文件夹名称推断为 G1_Stack_Block。

数据导入也使用 `wmal` 中的 PyArrow 和 PyAV/libdav1d。AV1 视频通过 PyAV 解码，并核对逐帧 PTS 与 parquet 时间戳，不静默截断错位帧，不需要另一个解码环境。

```bash
conda activate wmal
export PYTHONPATH="$PWD/src"
python scripts/visual_world.py import-lerobot \
  --source-manifest data/manifests/unitree_g1_pack_camera.json \
  --output data/processed/lerobot_visual_run1 \
  --action-schema unitree.g1.dex1.joint_target.v1 \
  --max-episodes 20 --max-frames 32 --image-size 64

python scripts/visual_world.py train \
  --manifest data/processed/lerobot_visual_run1/manifest.json \
  --output runs/lerobot_visual_run1 --device cuda --epochs 30

python scripts/visual_world.py evaluate \
  --manifest data/processed/lerobot_visual_run1/manifest.json \
  --checkpoint runs/lerobot_visual_run1/best.pt \
  --output runs/lerobot_visual_run1/test_metrics.json --horizon 4
```

导入、训练、评估均在同一个 `wmal` 环境中进行。action-schema 是对数据控制语义的显式声明，维数相同不能证明控制语义相同，接真机前仍需核对。原 train/test episode 保留；原 validation 排序后二分为选型/校准。20 episode subset 为 12/2/4/2，calibration 4 条不够 alpha=.1，需要扩大数据，不能借用 test 补数量。V3 打包视频暂不支持。

去掉 `--max-episodes` 使用全部 episode；`--max-frames` 控制每条导入前缀，正式评估应覆盖更丰富阶段，而不是只用开头静止片段。20×32 是数据/显存冒烟，不能作为模型泛化实验。16D Dex1 模型不能用于 `visual_skill_agent.py` 的 2D 固定底座增量接口，兼容检查会拒绝。

若目标是**开源大模型微调**，继续采用 [UniFoLM 专用路线](15_unifolm_training_adaptation.md)，本轮没有下载/训练 UniFoLM 10GB 以上权重，也未验证其在本机 8GB 显存可运行。紧凑网络用于独立方法验证与工程调试，不能作为大模型微调已完成的替代说法。

## 输出与复现

```text
data/processed/<dataset>/
  manifest.json               来源、动作语义、episode split、文件哈希
  episode_*.npz               RGB(T+1)、state(T+1)、action(T)
runs/<training>/
  best.pt / last.pt           权重/语义/归一化/来源/模型版本；不是 optimizer 续训状态
  history.json                train/validation 及分项损失
  training_report.json        参数量、训练/选型 episode、父模型、冻结规则
  calibration.json            独立校准 episode 与逐步误差界
  test_metrics.json           预测/persistence、反事实动作敏感性、episode 覆盖
runs/<agent>/
  events.jsonl                候选、预测收益、执行前缀、真实反馈、失败原因、时延
  prediction_*.npz            初始 RGB、预测 RGB/状态、动作、终端真实 RGB/状态
  task_*.json                 实际成功状态、步数、重规划次数、目标误差
```

代码逻辑与来源见 [网络与 Agent 设计](architecture/visual-world-agent.md)。验证命令：

```bash
conda activate wmal
MUJOCO_GL=egl PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m unittest discover -s tests -p test_visual_lerobot.py -v
```

最终证据与不足见 [本轮验收](architecture/visual-world-verification.md)。
