# World Model Agent Lab

Ubuntu + MuJoCo 世界模型机器人与高层 Agent 研究项目。

**当前状态：架构设计与文件骨架。未实现训练、Agent推理或MuJoCo仿真；未验证论文假设。**

## 阅读顺序

1. [文献证据](docs/01_literature.md)
2. [研究问题与路线](docs/02_research_and_route.md)
3. [架构与时序](docs/03_architecture.md)
4. [接口和调度](docs/04_contracts_and_runtime.md)
5. [训练/评估伪代码](docs/05_training_pseudocode.md)
6. [实验与数据](docs/06_experiments_and_data.md)
7. [实施计划与预算](docs/07_delivery_plan.md)
8. [完整目录树](docs/08_directory_tree.md)

主线提案：动作条件潜在动力学 + 技能结果预测 + 可靠性门控的高层Agent。模块按职责独立组织，不以文献算法命名或预设为其适配器。机器人、算力和软件版本未锁定。

## 当前可以执行

在项目根目录运行 `python3 scripts/check_scaffold.py`，仅验证文件、Python语法和JSON。
`python3 scripts/train.py --help` 可查看预留入口；实际运行明确以非零状态返回 NOT IMPLEMENTED，不生成虚假训练结果。

## 目录职责

`configs/` 保存设计参数；`src/wmal/` 按层划分源码位置；`robots/assets/` 预留资产；`scripts/` 预留命令；`examples/` 存接口示例；`data/`、`runs/` 保存未来数据和结果；`tests/` 预留实现后的行为测试。

`requirements-ubuntu.txt` 仅说明依赖锁定流程，不代表完整安装环境。无需安装MuJoCo或GPU依赖即可检查本骨架。原始论文仍在本地文献库，未复制PDF。

## 命名与实现原则

文件、目录、类、配置标识采用功能命名。后续代码独立编写；参考方法在文献证据中保留出处，使用第三方实现时保留许可和归属。当前为原创编写的设计骨架，不代表已提出或验证原创算法；改名不能替代方法创新与实验验证。
