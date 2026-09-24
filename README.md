# World Model Agent Lab

Ubuntu + MuJoCo 世界模型机器人与高层 Agent 研究项目。

**当前状态：** 原有 API Agent、ROS 2/MuJoCo 状态规划基线继续保留；新增 G1/UnifoLM 研究接口，包括 LeRobot V2.1 数据清单、WMA 微调配置/启动、同步图像状态输入、动作 chunk 安全映射和独立视觉预测评估。此代码没有捆绑上游源码、权重或数据；G1 MuJoCo 资产和动作 schema 尚待配置，外部 WMA 推理服务与完整训练尚未运行验证。

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
16. [G1/UnifoLM 研究功能实施规格](docs/superpowers/specs/2026-09-23-g1-unifolm-research-track-design.md)

运行主线：MuJoCo 采样训练动作条件状态模型 → API Agent 解析受限关节目标 → ROS 2 规划服务滚动评估候选动作 → ROS 2 action 驱动 MuJoCo → Agent 读取新状态并重新规划。当前训练模型仅在单关节探针上验证。

## 当前可以执行

## Conda 环境安装与启动

项目提供了 [environment.yml](environment.yml)，用于创建独立的 `wmal` 环境。
该环境包含 Python 3.10、MuJoCo、NumPy 和本项目的可编辑安装。PyTorch 为可选训练依赖，
按机器的 CPU/GPU 环境单独安装。

首次安装：

```bash
conda env create -f environment.yml
conda activate wmal
```

如需训练神经世界模型：

```bash
conda activate wmal
python -m pip install '.[learning]'
```

如果环境已经存在，可同步依赖：

```bash
conda env update -f environment.yml --prune
conda activate wmal
```

验证 Python、MuJoCo 和项目导入：

```bash
conda activate wmal
python -c "import sys, mujoco, wmal; print(sys.executable); print('MuJoCo', mujoco.__version__)"
```

运行完整测试：

```bash
conda activate wmal
PYTHONPATH=src python -m unittest discover -s tests -v
```

运行 MuJoCo 探针、采样和训练：

```bash
conda activate wmal
PYTHONPATH=src python scripts/simulate.py --config configs/robots/interface_probe.json
PYTHONPATH=src python scripts/collect.py --config configs/robots/interface_probe.json
PYTHONPATH=src python scripts/train.py
```

ROS 2 仍需在 Ubuntu 24.04 上单独安装 Jazzy；安装后先执行
`source /opt/ros/jazzy/setup.bash`，再使用本 Conda 环境启动 ROS 2 节点。

Agent 请求 HTTP API，将受限目标交给 ROS 2 世界模型规划服务，再以 ROS 2 action 执行第一步并读取新观测。安装及三终端启动命令见 [启动指南](docs/09_agent_ros2_design.md)。

仿真采样、训练和评估命令见[规划主线](docs/11_planning_world_model_pipeline.md)。另有独立的视觉动作服务客户端，用于后续策略对照；图像采集和机器人动作映射尚待具体资产接入，见[动作服务设计](docs/10_world_model_call_design.md)。

G1 数据准备、WMA decision/joint 微调、策略服务调用和离线视频预测评估入口见[UnifoLM 训练与接入指南](docs/15_unifolm_training_adaptation.md)。原始数据、权重、训练输出建议保存在仓库外的数据盘；先用 `--dry-run` 审阅训练配置。`configs/robots/g1.template.json` 有意保留空资产和空动作映射，未补齐时不能启动闭环。

运行 MuJoCo 探针及本地协议测试：`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`。ROS 2 端到端构建与通信需在 Ubuntu 安装 ROS 后验证。纯语法/JSON检查使用 `python3 scripts/check_scaffold.py`。

## 目录职责

`configs/` 保存设计参数；`src/wmal/` 按层划分源码位置；`robots/assets/` 预留资产；`scripts/` 预留命令；`examples/` 存接口示例；`data/`、`runs/` 保存未来数据和结果；`tests/` 预留实现后的行为测试。

`requirements-ubuntu.txt` 说明依赖边界；MuJoCo 可选依赖在 `pyproject.toml`。原始论文仍在本地文献库，未复制PDF。

## 命名与实现原则

文件、目录、类、配置标识采用功能命名。方法借鉴见[开源规划模块映射](docs/11_planning_world_model_pipeline.md)，上游代码许可证见[第三方研究项目说明](THIRD_PARTY_NOTICES.md)。当前实现不代表已提出或验证原创算法；改名不能替代论文复现说明、许可归属或实验验证。
