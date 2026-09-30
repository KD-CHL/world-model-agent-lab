# 十项 G1 动作规划实验与训练优化

## 实验设计

所有任务均由 mission agent 分配目标、导航层生成路点、世界模型展开候选动作、局部规划器选择第一段动作，交由已训练低层行走策略在 MuJoCo 执行。任务目标不是预录的动作轨迹。

| ID | 实验 | 场景与目标 | 考察能力 |
|---|---|---|---|
| 01_forward | 直行到点 | 空场地，(1.5, 0) | 基础位移与停靠 |
| 02_reverse | 后退到点 | 空场地，(-0.6, 0) | 身后目标规划；不强制动作始终后退 |
| 03_lateral | 侧向目标 | 空场地，(0, 1) | 侧向目标与机体坐标转换 |
| 04_diagonal | 斜向到点 | 空场地，(1.5, 1) | 平移与转向协调 |
| 05_heading | 朝向停靠 | (1, 0)，最终朝向 π/2 | 位置与姿态联合目标 |
| 06_obstacle | 障碍绕行 | 室内，(4, 0) | 全局绕障、局部预测 |
| 07_corridor | 狭窄通道 | 双侧障碍间 1.5 米通道 | 通行余量与预测误差 |
| 08_patrol | 三点巡航 | (1,0)→(1,1)→(0,1) | 多目标连续执行 |
| 09_return | 绕障往返 | (4,0)→(0,0) | 同会话往返与误差累积 |
| 10_delivery | 多站点配送路线 | (0.5,-1.5)→(3.2,-0.9)→(4,0) | 多阶段障碍路径 |

“配送”表示访问站点，当前不包含物体搬运、抓取或负载动力学。全部是静态已知地图，位置容差 0.15 米，朝向容差 0.15 弧度。实验列表与每阶段预算在 `configs/g1_experiments.json`。

## Agent 架构

`MissionAgent → NavigationPlanner → G1RolloutPlanner → WorldModelAdapter → G1MuJoCoSession`

- MissionAgent 管理顺序目标、阶段结果及失败停止；多个阶段共享物理会话，保留真实执行误差。
- NavigationPlanner 使用 A* 与可视路点，每周期重规划；规划足迹比执行边界额外留出 0.15 米，降低预测偏差触碰边界的机会。
- G1RolloutPlanner 包含随机、停止、转向、前进以及机体坐标平移候选，支持身后与侧向目标。
- 受阻检测同时检查位移与转角，避免将原地转向误判为停滞。
- 记录每阶段结果、预测误差和规划耗时；失败保留在统计分母中，批量运行继续完成其他实验。

## 训练方法

新增 `--optimize`：

1. 保持 episode 级训练、验证、测试隔离。
2. 对训练 episode 做 bootstrap，以轨迹为单位采样。
3. 比较特征标准化开/关与四档 ridge 正则化，共八个候选。标准化统计只来自训练数据。
4. 用验证集连续三步闭环模型展开选型，目标为位置 RMSE 加 0.2 倍偏航 RMSE；0.2 是预设米/弧度换算权重。
5. 只对选中模型报告测试集一步和三步误差；测试集不参与参数选择。

模型仍是轻量 ridge 集成；多步指标用于选型，并非神经网络多步反向传播训练。标准化不保证改善，选型允许保留未标准化模型。`--prior` 支持以已有权重为正则化中心进行适配。

## 命令

```bash
cd /home/chl/GitHub/CodeSpace/world-model-agent-lab-main
conda activate wmal
export PYTHONPATH="$PWD/src"

# 使用已采集的数据训练优化版本，保留原模型
python scripts/g1_research.py train --optimize --checkpoint runs/g1_research/optimized.json

# 十项实验，每项三个规划随机种子
python scripts/g1_benchmark.py --checkpoint runs/g1_research/optimized.json \
  --output runs/g1_suite_new --seeds 0 1 2

# 可视化单项实验
python scripts/g1_benchmark.py --checkpoint runs/g1_research/optimized.json \
  --output runs/g1_suite_view --experiments 06_obstacle --viewer

# 关闭残差反馈的消融对照
python scripts/g1_benchmark.py --checkpoint runs/g1_research/optimized.json \
  --output runs/g1_suite_no_feedback --seeds 0 1 2 --no-feedback
```

输出目录必须不存在，以防覆盖历史实验。每个任务、种子有独立 JSONL 日志，汇总为 `results.json`，包括成功率、各阶段误差、平均/p95 规划耗时、预测 RMSE、模型与实验配置哈希。失败时程序完成其余任务后返回 1。

三个种子只改变候选采样，当前不随机化物理场景或初始状态。论文需进一步增加布局、摩擦、初始状态变化与无模型对照；这十项任务及少量训练数据用于功能验收，不能替代泛化实验。
