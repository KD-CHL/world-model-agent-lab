# 论文实验数据清单与仿真输出规范

目标是为“世界模型规划如何改善 Agent 与机器人在长任务中的决策和恢复”提供**可复核的数据证据**。期刊分区没有统一的合格数据量门槛；实验必须与具体研究问题、对照和统计单位一致。当前单关节探针只用于验证记录链路，不能直接支撑长时序机器人操作论文。

## 数据集与原始记录

| 研究问题 | 仿真时采集的字段 | 汇总指标与统计单位 | 当前状态 |
|---|---|---|---|
| 世界模型是否预测可控结果 | `observation`、`commanded_action`、`applied_controls`、`predicted_next`、`observed_next`、`model_version`、`step_id` | 保留回合上的单步/多步误差、候选排序与结果相关性；按训练种子汇总 | 单步已采；多步与候选排序待实现 |
| 规划是否改善任务完成 | `task_target`、技能/子任务标签、终态检测、完成原因、场景 ID | 整任务与技能成功率、完成仿真时间/控制步；同场景配对 | 当前仅关节目标探针；操作任务待实现 |
| 失败后能否恢复 | 扰动类型、大小、注入时刻、检测时刻、重规划事件、恢复终态 | 恢复率、恢复步数/时间、误触发/漏触发；独立扰动回合 | 待实现扰动与检测器 |
| 可信度是否有用 | 预测输出类型、集成成员、校准集标签、触发阈值、后验误差 | 对适用类型报告覆盖率/校准、失效识别、误报与漏报 | 模型可输出成员；未校准，不报告失败概率 |
| 系统代价与实时性 | `physics_steps`、`imagined_model_steps`、`planning_wall_s`、`execution_wall_s`、API/ROS 请求时间与超时 | 仿真交互数、想象步数、延迟 p50/p95/p99、截止期违约率、LLM 调用量、训练资源 | 探针已有前四类；ROS/LLM/训练资源需目标机器接入 |
| 可复现与失效归因 | 数据来源、`episode_id`、配置/资产/数据/检查点哈希、随机种子、代码提交、失败类别、完整仿真状态 | 多训练种子原始值与置信区间；所有失败/中断单列 | 探针已有基本来源与哈希；跨种子和失败分类待实验 |

第二数据源必须另记来源、原始数据集/模型检查点和父轨迹；外部示范、模型生成视频、模型引导的 MuJoCo 交互分别统计。详见[第二数据源可行性](12_secondary_data_feasibility.md)。训练/验证/测试以 episode 与场景组隔离。测试集不用于阈值、检查点或任务设计调优。训练种子是模型方法的独立重复；一个种子的多个评估回合不能冒充多个训练重复。

## 现有代码会输出什么

在项目根目录执行：

```bash
PYTHONPATH=src python scripts/collect.py --config configs/robots/interface_probe.json --output runs/probe/transitions.jsonl --episodes 10 --steps-per-episode 8 --seed 4
PYTHONPATH=src python scripts/train.py --dataset runs/probe/transitions.jsonl --checkpoint runs/probe/model.json --seed 4
PYTHONPATH=src python scripts/evaluate.py --config configs/robots/interface_probe.json --dataset runs/probe/transitions.jsonl --checkpoint runs/probe/model.json --output runs/probe/evaluation.jsonl --episodes 10 --seed 20
PYTHONPATH=src python scripts/export_results.py --inputs runs/probe/evaluation.summary.json --output runs/probe/paper_metrics.csv
```

采样每回合、评估每方法每回合在终端输出一条进度 JSON；JSONL 在控制过程中逐行写盘，不需等仿真结束。完成后自动生成：

| 文件 | 内容 |
|---|---|
| `transitions.jsonl` | 每控制动作的前后关节状态、目标、完整物理状态、每物理步施加的控制、episode/step/仿真时钟、数据划分 |
| `transitions.manifest.json` / `transitions.summary.json` | 配置与资产哈希、代码版本、随机种子、MuJoCo 版本；采样回合、划分、物理步数、仿真与墙钟时长 |
| `model.json` / `model.train.json` | 学习参数；训练数据哈希、训练种子、验证集误差、检查点哈希 |
| `evaluation.steps.jsonl` | 每控制步的候选选择结果、实际命令、一步预测与后验、模型调用/想象步数、延迟与物理步数 |
| `evaluation.jsonl` | 每回合目标、最终状态、成功、终止原因、控制步数与误差 |
| `evaluation.manifest.json` / `evaluation.summary.json` | 输入哈希、模型/环境版本；按方法汇总的成功率、误差、规划延迟与预算；明确单种子统计边界 |
| `paper_metrics.csv` / `paper_metrics.aggregate.json` | 多运行扁平表及按训练种子先聚合的均值；不自动生成伪精确置信区间 |

状态快照使用 MuJoCo 的 `mjSTATE_FULLPHYSICS`，另存执行控制序列。官方[仿真状态文档](https://mujoco.readthedocs.io/en/latest/programming/simulation.html)说明物理状态、时间、插件状态及用户控制输入的区别；若使用 sleeping、外部插件或噪声源，还需保存相应内部/随机状态并做重放一致性测试。当前日志不保证复杂机器人场景的位级精确回放。原始视频可作为观察材料，不能替代状态和动作轨迹。

## 正式论文实验的验收顺序

1. 冻结一个明确任务及机器人资产，制定环境终态判据、训练/验证/测试场景组和主要指标；预注册对照与预算。
2. 用相同机器人、观测、动作接口和 MuJoCo 交互预算比较固定控制、无高层预测反馈、世界模型规划和完整 Agent 机制。第二数据源实验另做等预算对照，并单独报告预训练/演示量。
3. 完成多训练种子与每种子的多评估回合，保留所有失败和中断；按种子/场景做统计，不把控制步当独立样本。置信区间和检验方法在看测试结果前确定。
4. 运行扰动、未见对象或组合任务，报告恢复、误判与代价。导出原始表、图表脚本、配置、版本与数据可用性说明。

当前代码的 `direct` 与 `model_planner` 在单关节探针上可能都达到上限成功率；这只能证明记录和控制路径连通，**不能作为规划优势证据**。任务、场景和资源确定后，才可设计有辨别力的 SCI 论文实验。
