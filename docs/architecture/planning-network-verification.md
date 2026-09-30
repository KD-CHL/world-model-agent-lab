# 规划网络升级验收记录

日期：2026-09-30。本轮通过 GitHub 插件读取固定版本的训练/推理源码，独立编写功能模块，来源与机制映射见 [设计说明](planning-network.md)。没有提交或推送本轮修改，未覆盖此前未提交工作。

## 自动检查

- 历史全套：当时 99 项，98 通过、1 跳过、0 失败。当前统一在 `conda activate wmal` 后使用 `PYTHONPATH=src python -m unittest discover -s tests -v`，最新复验结果以视觉网络验收文档为准。
- 唯一跳过为 ROS 2 实际中间件测试，本机没有 rclpy；不能据此宣称 Ubuntu ROS 2 端到端已通过。
- 8 项新测试覆盖位移与末端速度分离、潜在网络学习/保存/加载、恢复训练、测试数据隔离、成员独立轨迹、逐成员与平均路径约束、批量返回回合一致性、语言恢复证据及数据导出。
- 平均路径障碍问题经历了先失败后修复的回归验收：成员路径在障碍两侧不代表其平均路径无障碍，现同时检查二者。
- Python/JSON/接口 XML 检查通过，`git diff --check` 通过。

完整日志保存在本机 `/tmp/wmal-planning-final-tests.log`；它不是可携带的发布工件。重新运行上述命令可生成新的验证记录。

## MuJoCo 数据与训练

使用已有真实 MuJoCo 仿真交互数据（不是模型生成数据、不是实体机器人数据）：

- `runs/motion_network_smoke/transitions.json`，128 条，8 个回合；4 train、2 validation、2 test；动作保持时间 0.5 s。
- 数据 SHA-256：`6c2e0ca7264513c564c6cd174ca5ac68abe1a4e9044e491d6e663b3f617a4eb3`。
- 新配置 `configs/training/planning_network_smoke.json`：3 成员，hidden=[32,32]，三步训练，40 epochs，seed=7，CPU 单线程。
- v2 模型版本 `motion-8c5eb24aebbe6661`；总参数 28,365；训练循环约 0.65 s，未计解释器启动/加载和采集时间。
- 验证复合损失从首轮 1.19894 降至最佳 0.74377，最佳 epoch=39。各架构损失项不同，不能直接用复合损失数值比较优劣。
- 原 v1 模型版本 `motion-b06413e7d1cefe9c`；总参数 4,818，40 epochs，seed=7，同一数据划分；训练循环约 0.31 s。参数量与计算时间不相同，当前并非等计算比较。
- 训练配置在测试评估前固定，未根据本轮测试表现调参；检查点只由 validation 选择。

产物：`runs/planning_network_smoke/model.pt`、`model.latest.pt`、`model.pt.report.json`。原网络报告与权重继续保留。新报告包含数据、训练源码、网络源码的 SHA，以及实际平台与 PyTorch/MuJoCo 版本。

## 冻结模型测试

用同一成员独立滚动评估路径评估两个保留测试回合：

| 指标 | v1 | v2 |
|---|---:|---:|
| 单步位置 RMSE / m | 0.04011 | 0.03220 |
| 两步位置 RMSE / m | 0.07498 | 0.05403 |
| 三步位置 RMSE / m | 0.09758 | 0.06912 |
| 单步末端线速度 RMSE / m/s | 0.15713 | 0.10361 |
| 三步末端线速度 RMSE / m/s | 0.13960 | 0.10730 |
| 单步 yaw RMSE / rad | 0.05598 | 0.04511 |
| 三步 yaw RMSE / rad | 0.08964 | 0.10906 |

位置和末端速度在这组测试上降低，三步 yaw 误差增加。该混合结果必须保留，不据此宣称全面提升。仅一个训练种子、两个测试回合，窗口互相重叠，不是论文统计重复；未建立性能增益或创新性的结论。

评估报告：`runs/planning_network_smoke/evaluation.json`、`baseline_evaluation.json`。与此前 v1 报告比较时注意：本轮两种网络均用成员独立滚动，早期 v1 报告使用逐步均值反馈，预测语义有差异。

## 校准、规划推理与执行

验证集两个回合拟合的一步位置残差报警阈值为 0.06552 m，产物 `calibration.json`。阈值含模型版本与持续时间绑定。这个样本量只验证接口，不足以证明误报率、覆盖率或分布偏移下的可靠性；它不将 ensemble spread 转换为失败概率。

同一 v2 模型、48 候选、3 步、3 成员、CPU 单线程、20 次交替顺序计时：

| 推理方式 | p50 / ms | p95 / ms | 逻辑模型步调用 | 想象成员步数 |
|---|---:|---:|---:|---:|
| 每候选独立调用 | 22.059 | 23.736 | 432 | 432 |
| 批量候选调用 | 0.893 | 1.083 | 9 | 432 |

两种方式都保持成员轨迹，输出已用 atol=2e-6/rtol=1e-5 检查等价。这是推理组织的耗时对比，不包含场景筛选、物理积分、ROS 2 或外部 API。原始计时保存在 `latency.json`；不能把这些结果解释为系统实时性承诺。

真实 G1 MuJoCo 闭环再次运行：

```bash
conda activate wmal
PYTHONPATH=src python scripts/g1_agent_sim.py --config runs/planning_network_smoke/agent_config.json --headless --goal '0.6 0' --log runs/planning_network_smoke/closed_loop_verified.jsonl
```

在已知室内场景中 5 次控制段后，独立环境观测判为成功；最终 `(x,y)=(0.52145,0.00329)` m，目标 `(0.6,0)` m、容差 0.15 m。记录了 5 条 plan、5 条实测 transition、5 条残差。整段规划耗时约 3.28–22.00 ms，首段包含首次推理开销；这是单任务烟测，不是成功率、恢复能力或低延迟统计结论。

与早先 v1 的 4 段成功结果没有做配对多次统计，不宣称任务完成步数更优。本次没有调用真实外部 LLM；语言恢复接口以离线测试验证，机器人执行走直接 MuJoCo 会话，没有实际 ROS 2 中间件。

执行日志经 `export-execution` 完整导出为 `verified_execution_data.json`：5 条 transition、1 个 train 回合、1 条终态审计事件。它演示数据回流，不足以单独启动训练；必须另收 validation/test 回合。日志和该导出都没有完整积分/控制器状态，不能视为精确回放文件。

所有 `runs/` 数据和权重均被 Git 忽略，留在本机；架构、配置、训练/导出入口和测试属于可评审源码。实际运行平台为 macOS/CPU，Ubuntu/GPU、ROS 2 及机械臂/Go2 训练未在本轮验证。
