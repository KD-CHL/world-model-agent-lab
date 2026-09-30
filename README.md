# World Model Agent Lab

Ubuntu + MuJoCo 世界模型机器人与高层 Agent 研究项目。

## G1 世界模型规划研究入口

新增可训练的 **RGB 动作条件视觉世界模型与可信度辅助技能 Agent**：支持真实 MuJoCo 图像采集、LeRobot AV1/parquet 对齐导入、CNN/GRU 多步集成训练、冻结编码器微调、独立 episode 校准、A0–A3 和保持打开的交互窗口。见 [架构与论文假设](docs/architecture/visual-world-agent.md)、[完整命令与输出位置](docs/23_visual_world_training_and_agent.md)、[实际验收与局限](docs/architecture/visual-world-verification.md)。当前闭环验证是 G1 两关节到达目标，不宣称操作任务/UniFoLM 大模型微调已完成。

研究主线：[预测可信度辅助机器人Agent规划与两条训练路线](docs/21_research_plan_prediction_reliability.md)；[数据集核查清单](docs/22_world_model_dataset_catalog.md)。固定技能和任务状态机，当前先验证状态规划网络的训练闭环；视觉动作条件模型保留为独立支线，两条路线均需独立消融。

规划世界模型训练网络：[神经动力学集成的结构、采样、训练、续训和规划接入](docs/architecture/motion-network.md)。入口 `scripts/train_motion_network.py`，支持 `collect`、`train`、`evaluate`；G1 接入配置为 `configs/g1_neural.json`。

Agent 架构优化：共享技能生命周期、G1 有预算的任务重规划、验证集残差阈值、G1 ROS2 会话与统一日志汇总，见 [设计与启动说明](docs/architecture/agent-runtime.md)。

自然语言 API 规划入口：`python scripts/g1_llm_agent.py`。环境变量配置、模型边界和通信故障处理见 [大模型与 G1 协调指南](docs/20_llm_coordination.md)。

十二种桌面操作环境见 [G1 操作任务虚拟世界](docs/19_manipulation_worlds.md)。运行 `python scripts/workcell_sim.py --task stack_block` 打开积木工作台；支持双机器人和柔性毛巾场景。

新增 [十项动作规划实验与模型训练优化](docs/18_ten_experiments.md)：批量运行任务、多个规划种子、阶段结果统计，以及验证集多步预测选型。

室内绕障实验：`python scripts/g1_agent_sim.py --config configs/g1_indoor.json`，输入目标 `4 0`。
场景、agent 分层规划和无窗口评估见 [室内规划指南](docs/17_indoor_navigation.md)。

新增可训练的底座动力学集成、预测残差反馈和完整采集/训练/评估命令。
详见 [研究架构与完整运行指南](docs/16_g1_research_architecture.md)。

```bash
conda activate wmal
export PYTHONPATH="$PWD/src"
python scripts/g1_research.py collect --episodes 12 --steps 30
python scripts/g1_research.py train
python scripts/g1_research.py evaluate --feedback --steps 100
python scripts/g1_agent_sim.py --config configs/g1_research.json
```

交互提示中输入 `1 0` 设置世界坐标目标；任务结束后窗口继续接受新目标，输入 `quit` 退出。
训练模型是轻量行走状态预测基线；UniFoLM 视频模型需要额外适配，不能用操作数据的动作维度直接替代底座速度指令。

**当前状态：** 原有 API Agent、ROS 2/MuJoCo 状态规划基线继续保留；G1 固定底座关节测试与浮动底座行走分开配置。行走测试使用 Unitree 官方 `unitree_rl_mjlab` 的同源 G1 碰撞模型和已发布速度策略 ONNX，在本地 MuJoCo 验证脚步触地、前进位移、机身高度和倾斜；该验证不代表真机安全或 sim-to-real 验证。UnifoLM 数据集动作映射、外部 WMA 推理服务和完整训练仍需单独验证；未捆绑上游 WMA 代码或数据。

## 阅读顺序

1. [文献证据](docs/01_literature.md)
2. [研究问题与路线](docs/02_research_and_route.md)
3. [架构与时序](docs/03_architecture.md)
4. [接口和调度](docs/04_contracts_and_runtime.md)
5. [训练/评估伪代码](docs/05_training_pseudocode.md)
6. [实验与数据](docs/06_experiments_and_data.md)
7. [实施计划与预算](docs/07_delivery_plan.md)
8. [完整目录树](docs/08_directory_tree.md)
9. [Agent、ROS 2 设计与启动指南](docs/09_agent_ros2_design.md)
10. [世界模型规划的数据闭环与运行入口](docs/11_planning_world_model_pipeline.md)
11. [可选视觉动作服务调用设计](docs/10_world_model_call_design.md)
12. [视觉动作模型训练与第二数据源可行性](docs/12_secondary_data_feasibility.md)
13. [论文实验数据清单与仿真输出](docs/13_paper_data_pipeline.md)
14. [规划层世界模型开源项目筛选](docs/14_open_source_planning_references.md)
15. [基于 UnifoLM 的机器人世界模型训练与微调路线](docs/15_unifolm_training_adaptation.md)
16. [G1 本地浮动底座步行测试](docs/16_g1_local_walking.md)
17. [G1/UnifoLM 研究功能实施规格](docs/superpowers/specs/2026-09-23-g1-unifolm-research-track-design.md)

运行主线包含关节状态规划与 G1 导航规划。G1 目前支持 MuJoCo 采集 → 多步潜在动力学集成训练 → 候选轨迹批量预测 → Agent 规划及恢复 → 直接或 ROS 2 会话执行 → 观测残差和训练数据导出。G1 训练与直接 MuJoCo 闭环已做本地冒烟验证；ROS 2 实际中间件及其他机器人训练需分别验证。

世界模型训练升级见 [网络、源码参考与闭环接口](docs/architecture/planning-network.md) 和 [本轮验收报告](docs/architecture/planning-network-verification.md)。训练入口为 `scripts/train_motion_network.py`，新配置为 `configs/training/planning_network.json`；旧物理网络和检查点继续可用。参考机制采用独立功能命名，证据表给出固定 GitHub 源码版本。

## 当前可以执行

## 当前模型的数据来源

模型不是在无数据情况下生成的。当前 MuJoCo 窗口加载的视觉模型使用本机采集的仿真数据：
`data/processed/visual_g1_20260930/manifest.json`，40 条轨迹、640 次动作转移，包含 RGB、两关节目标增量和执行后的实测状态。
24/4/8/4 条分别用于训练/验证/校准/测试；基模随机初始化训练 30 轮，再用同一仿真数据冻结编码器训练 2 轮。
这次“微调”是工程实验，不是用新增开源数据对 UniFoLM 预训练权重调优，也不代表已学会抓取或堆叠。

另一条独立离线路线使用已下载的 `G1_Dex1_MountCameraRedGripper_Dataset`：原始数据在
`data/raw/lerobot/unitree_g1_pack_camera/`，已转换子集在 `data/processed/lerobot_visual_av1_20260930/`。
该子集为 20 条轨迹、每条 32 帧（620 次转移），曾单独训练 3 轮；其 16 维动作模型不能直接替换窗口中的 2 维增量模型。
数据和权重由 gitignore 排除，所以远程仓库只看得到来源清单，下载代码不会自动获得这些大文件。
完整数据来源、采集/导入及训练命令见 [视觉训练指南](docs/23_visual_world_training_and_agent.md)。

`scripts/visual_world.py train`（含 `--pretrained` 微调）和 `scripts/train_motion_network.py train` 默认显示轮次、训练/验证批次、集成成员、损失、耗时及 ETA。增加 `--no-progress` 可关闭提示；进度写 stderr，不改变 stdout 的 JSON、训练算法或 checkpoint 格式。所有命令只使用 `wmal`。

## Conda 环境安装与启动

项目提供了 [environment.yml](environment.yml)，用于创建独立的 `wmal` 环境。
采集、数据导入、训练、微调、推理和 MuJoCo 仿真统一使用本项目的 `wmal` 环境，
不借用其他项目的 Conda 环境，也不把其他环境的 site-packages 加入 PYTHONPATH。
环境配置包含 Python 3.10、MuJoCo、NumPy、ONNX Runtime、PyTorch CUDA 12.8、
PyArrow、PyAV、Pillow、YAML/Hugging Face 工具和本项目的可编辑安装。
当前 GPU 配置固定 `torch==2.11.0+cu128`，适配本机 RTX 5060 Ti；其他硬件需单独调整构建。

首次安装：

```bash
conda env create -f environment.yml
conda activate wmal
export PYTHONPATH="$PWD/src"
```

已有环境补齐训练和数据依赖：

```bash
conda activate wmal
export PYTHONPATH="$PWD/src"
python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[simulation,walking,visual-world,visual-data,robot-data]'
```

如果环境已经存在，可同步依赖：

```bash
conda env update -n wmal -f environment.yml
conda activate wmal
export PYTHONPATH="$PWD/src"
```

验证 Python、GPU、MuJoCo 与数据工具：

```bash
conda activate wmal
export PYTHONPATH="$PWD/src"
python -c "import sys,torch,mujoco,pyarrow,av,wmal; print(sys.executable); print('Torch',torch.__version__,'CUDA',torch.cuda.is_available()); print('MuJoCo',mujoco.__version__,'PyArrow',pyarrow.__version__,'PyAV',av.__version__)"
python -m pip check
```

运行完整测试：

```bash
conda activate wmal
export PYTHONPATH="$PWD/src"
PYTHONPATH=src python -m unittest discover -s tests -v
```

运行 MuJoCo 探针、采样和训练：

```bash
conda activate wmal
PYTHONPATH=src python scripts/simulate.py --config configs/robots/interface_probe.json
PYTHONPATH=src python scripts/simulate.py --config configs/robots/g1_fixed_base.json --steps 1000
PYTHONPATH=src python scripts/simulate.py --config configs/robots/g1_fixed_base.json --viewer
PYTHONPATH=src python scripts/walk_g1.py --steps 10
PYTHONPATH=src python scripts/collect.py --config configs/robots/interface_probe.json
PYTHONPATH=src python scripts/train.py
```

交互窗口中，按 `N` / `P` 选择下一个/上一个关节，按 `[` / `]` 将当前关节目标减少/增加 `0.1 rad`，按 `0` 将目标设为当前关节位置。目标会按配置的关节速度上限平滑执行；默认相机可用鼠标旋转，若需锁定相机可加 `--camera pack_camera`。

ROS 2 中间件仍需在 Ubuntu 24.04 上单独安装 Jazzy。其系统 Python 3.12 绑定不能直接用于本项目 Python 3.10 环境；需先完成匹配绑定的构建与通信验证，不能仅靠 `source /opt/ros/jazzy/setup.bash` 混入系统包。直接 MuJoCo 训练和仿真不需要 ROS。

Agent 请求 HTTP API，将受限目标交给 ROS 2 世界模型规划服务，再以 ROS 2 action 执行第一步并读取新观测。安装及三终端启动命令见 [启动指南](docs/09_agent_ros2_design.md)。

仿真采样、训练和评估命令见[规划主线](docs/11_planning_world_model_pipeline.md)。另有独立的视觉动作服务客户端，用于后续策略对照；图像采集和机器人动作映射尚待具体资产接入，见[动作服务设计](docs/10_world_model_call_design.md)。

G1 数据准备、WMA decision/joint 微调、策略服务调用和离线视频预测评估入口见[UnifoLM 训练与接入指南](docs/15_unifolm_training_adaptation.md)。固定底座关节测试仍使用 [固定底座配置](configs/robots/g1_fixed_base.json)。浮动底座行走使用 Unitree 预训练策略，运行 `PYTHONPATH=src python scripts/walk_g1.py --steps 10`；命令只在完成 10 次交替足部触地且通过姿态/高度保护后报告成功，并输出前进位移等指标。详见 [G1 本地步行指南](docs/16_g1_local_walking.md)。这与 UnifoLM 数据集动作映射和 WMA 策略微调是独立路径。原始数据、微调权重和训练输出建议保存在仓库外的数据盘。

需要保持 MuJoCo 窗口运行、由世界模型预测和 Agent 连续重规划时，使用 [G1 世界模型闭环指南](docs/17_g1_world_model_agent_loop.md) 与 `PYTHONPATH=src python scripts/g1_agent_sim.py --config configs/experiments/g1_closed_loop.example.json`。该实验入口要求显式配置兼容的 G1 浮动底座状态预测器；Unitree ONNX 仅作为低层行走控制器。完成单个目标不会关闭会话，可继续输入后续目标；本入口不负责训练世界模型。

运行 MuJoCo 探针、G1 资产加载/相机渲染和本地协议测试：`MUJOCO_GL=egl PYTHONPATH=src python -m unittest discover -s tests -v`。ROS 2 端到端构建与通信需在 Ubuntu 安装 ROS 后验证。纯语法/JSON检查使用 `python3 scripts/check_scaffold.py`。

## 目录职责

`configs/` 保存实验参数；`src/wmal/` 按层组织实现；`robots/assets/` 管理资产入口；`scripts/` 提供采集、训练、仿真和汇总命令；`examples/` 存接口示例；`data/`、`runs/` 保存数据和结果；`tests/` 覆盖训练、规划与通信行为。

`requirements-ubuntu.txt` 说明依赖边界；MuJoCo 可选依赖在 `pyproject.toml`。原始论文仍在本地文献库，未复制PDF。

## 命名与实现原则

文件、目录、类、配置标识采用功能命名。方法借鉴见[开源规划模块映射](docs/11_planning_world_model_pipeline.md)，上游代码许可证见[第三方研究项目说明](THIRD_PARTY_NOTICES.md)。当前实现不代表已提出或验证原创算法；改名不能替代论文复现说明、许可归属或实验验证。
