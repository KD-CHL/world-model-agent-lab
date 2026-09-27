# G1 世界模型辅助规划研究架构

## 研究目标与贡献

主贡献候选：以动作条件世界模型预测候选速度序列，通过滚动规划选择动作，每次只执行第一段，并依据新的观测重规划。
支撑贡献候选：机器人状态与动作的数据适配、预测残差驱动的规划速度约束。
这些是可运行研究方法，创新性与性能提升仍需论文对照实验论证。

## 数据流与实现

```text
MuJoCo + 已训练低层行走策略 → 状态/速度指令/下一状态 → episode 分割
                                                     ↓
                                              训练/适配模型
                                                     ↓
目标 → G1Agent → G1RolloutPlanner → WorldModelAdapter → 候选未来状态
           ↑           ↓ 最优序列第一段
           └──观测── G1MuJoCoSession
                └──预测残差── ResidualFeedback → 下一轮候选速度范围
```

模块责任：

- `locomotion/learned.py`：bootstrap ridge 集成；输入底座状态与机体速度指令，预测局部位移率、偏航率与姿态。集成分歧提供不确定性代理量，尚非校准概率。
- `locomotion/planner.py`：采样动作序列，逐步模型展开；过滤预测低高度或大倾角，综合终点误差、控制代价和不确定性，执行首段。
- `locomotion/feedback.py`：对位置预测残差做指数平滑，误差超过阈值时收缩下一轮速度范围，误差下降后恢复。它不在线更新模型权重。
- `locomotion/agent.py`：任务终止、状态时序检查、预测与反馈日志。会话生命周期由调用者管理。
- `scripts/g1_research.py`：真实仿真采集、离线训练、已有检查点适配、单目标评估。
- `scripts/g1_agent_sim.py`：常驻交互窗口，可连续输入多个目标。

实现顺序：复用状态/动作协议 → 可训练模型与持久化 → 接入反馈 → 采集训练评估命令 → 合成数据单测与真实仿真验收。

## 完整命令

在项目根目录执行：

```bash
conda activate wmal
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# 采集；每个 episode 重新初始化机器人
python scripts/g1_research.py collect --episodes 12 --steps 30 --seed 0

# 训练并输出验证/测试指标
python scripts/g1_research.py train --seed 0

# 在同一目标上分别运行无反馈和有反馈版本
python scripts/g1_research.py evaluate --steps 100 --seed 0
python scripts/g1_research.py evaluate --steps 100 --seed 0 --feedback

# 打开常驻交互窗口；目标完成后可继续输入新目标
python scripts/g1_agent_sim.py --config configs/g1_research.json
```

窗口提示 `goal>` 时输入 `1 0`（世界坐标目标），或 `1 0 90`（再约束朝向角度）。输入 `quit` 退出。等待输入时仿真暂停，窗口保持打开。

采集数据：`data/g1_locomotion/transitions.json`；训练权重：`runs/g1_research/model.json`；报告：同名 `.report.json`；评估与交互日志位于 `runs/g1_research/`。数据与模型均受现有 Git 排除规则保护。

适配另一个环境采集的数据：

```bash
python scripts/g1_research.py train \
  --dataset data/g1_locomotion/new_transitions.json \
  --prior runs/g1_research/model.json \
  --checkpoint runs/g1_research/adapted.json
```

这是以已有集成平均权重为正则化中心的参数适配。训练仅用 train episodes；validation 与 test 只做指标计算。每个数据集至少三个独立 episode，训练至少十条转移；动作持续时间必须一致。测试集不能用于选超参数。

## 开源模型与数据接入边界

已有 LeRobot/UniFoLM 转换及训练入口保留，参见 `15_unifolm_training_adaptation.md`。下载的操作数据中的 16 维状态/动作不等于浮动底座状态与机体速度指令，不能直接训练本行走模型。需要带底座位姿、执行速度命令、时间戳的转移记录，或另外建立并验证语义映射。

大型模型可实现既有 `predict(state, action, duration_s)` 插件协议，并通过配置替换 `world_model.factory`。视频预测模型还需要状态读出模块才能用于本规划器；此版本没有声称已将 UniFoLM 视频模型接入行走规划。

轻量模型不预测关节轨迹、接触或图像；这些字段保留输入值，规划只使用预测底座量。训练覆盖范围外的预测不可靠。预测过滤与仿真姿态停止不是安全保证。

## 论文实验与验收

当前自动化验收覆盖模型保存加载、已知动力学拟合、规划动作选择、反馈收缩恢复与过期反馈拒绝；真实 MuJoCo 冒烟验证采集→训练→目标评估。

正式实验应固定初始状态、目标集合、低层策略和计算预算，跨多个种子比较：直接目标跟踪、世界模型规划、规划加反馈；再比较数据规模与适配前后模型。当前命令提供后两种方法的单目标运行，完整多场景基准及无模型基线仍待扩展。
报告成功率、终点误差、预测误差、规划延迟和姿态停止次数，并保留失败试验；不可用单次成功代替统计结论。
