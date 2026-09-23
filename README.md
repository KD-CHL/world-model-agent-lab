# World Model Agent Lab

Ubuntu + MuJoCo 世界模型机器人与高层 Agent 研究项目。

**当前状态：API Agent、ROS 2 适配、MuJoCo 关节控制及低维动力学模型的采样、训练、规划和探针评估已实现。研究机械臂资产、Go2/G1 步态/平衡控制器与正式任务世界模型仍待接入。**

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

运行主线：MuJoCo 采样训练动作条件状态模型 → API Agent 解析受限关节目标 → ROS 2 规划服务滚动评估候选动作 → ROS 2 action 驱动 MuJoCo → Agent 读取新状态并重新规划。当前训练模型仅在单关节探针上验证。

## 当前可以执行

Agent 请求 HTTP API，将受限目标交给 ROS 2 世界模型规划服务，再以 ROS 2 action 执行第一步并读取新观测。安装及三终端启动命令见 [启动指南](docs/09_agent_ros2_design.md)。

仿真采样、训练和评估命令见[规划主线](docs/11_planning_world_model_pipeline.md)。另有独立的视觉动作服务客户端，用于后续策略对照；图像采集和机器人动作映射尚待具体资产接入，见[动作服务设计](docs/10_world_model_call_design.md)。

运行 MuJoCo 探针及本地协议测试：`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`。ROS 2 端到端构建与通信需在 Ubuntu 安装 ROS 后验证。纯语法/JSON检查使用 `python3 scripts/check_scaffold.py`。

## 目录职责

`configs/` 保存设计参数；`src/wmal/` 按层划分源码位置；`robots/assets/` 预留资产；`scripts/` 预留命令；`examples/` 存接口示例；`data/`、`runs/` 保存未来数据和结果；`tests/` 预留实现后的行为测试。

`requirements-ubuntu.txt` 说明依赖边界；MuJoCo 可选依赖在 `pyproject.toml`。原始论文仍在本地文献库，未复制PDF。

## 命名与实现原则

文件、目录、类、配置标识采用功能命名。后续代码独立编写；参考方法在文献证据中保留出处，使用第三方实现时保留许可和归属。当前为原创编写的设计骨架，不代表已提出或验证原创算法；改名不能替代方法创新与实验验证。
