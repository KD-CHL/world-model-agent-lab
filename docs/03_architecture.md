# 架构与时序

```mermaid
flowchart TD
 T[TaskSpec] --> A[高层 Agent / 固定规划器]
 A --> C[技能候选与前置条件检查]
 C --> P[独立技能结果模型与校准器]
 P --> G[可靠性门控与重规划]
 G --> S[技能执行器]
 S --> M[潜在动力学控制 动作规划]
 M --> E[动作适配与约束]
 E --> U[MuJoCo 单一 owner]
 U --> O[观测与独立任务检测]
 O --> A
 O --> G
 U --> D[轨迹与事件日志]
 D --> L[回放与离线训练]
 L --> V[验证与版本发布]
 V --> P
 V --> M
 D --> R[冻结评估与论文分析]
```

| 模块 | 输入→输出 | 职责与边界 |
|---|---|---|
| agents | TaskSpec/Feedback→技能候选 | 低频规划；不访问物理积分器 |
| models | 状态/动作或技能→潜在/结果预测 | 区分低层控制模型与技能结果模型 |
| planners | 候选/PredictionReport→选择或重规划 | 门控不等于安全证明 |
| skills | SkillSpec→ActionCommand | 超时、终止及恢复次数管理 |
| envs/robots | ActionCommand→Observation | 唯一状态修改入口；坐标与控制模式适配 |
| communication | 消息→确认/错误 | 版本、时序和 episode 隔离 |
| datasets/replay/training | 仿真轨迹→检查点 | 数据来源及训练集边界 |
| evaluation/logging/analysis | 冻结快照→指标及图表 | 独立检测器与可追溯证据 |

```mermaid
sequenceDiagram
 participant A as Agent
 participant P as 技能预测
 participant S as 技能控制
 participant E as 仿真owner
 participant L as 日志/检测
 A->>P: 固定候选集合+最新观测
 P-->>A: 结果预测/可信度/有效期
 A->>S: 选定技能+版本
 loop 控制周期
 S->>E: 带step和deadline的动作
 E->>L: 执行动作与后验观测
 L-->>S: 检测/残差/约束状态
 end
 L-->>A: 成功、失败或事件
 A->>A: 更新任务状态；必要时重规划
```

```mermaid
sequenceDiagram
 participant E as 采样器
 participant B as 数据/回放
 participant T as 训练器
 participant V as 验证器
 E->>B: 完整回合及技能片段
 B->>T: 仅train分组序列
 T->>V: 临时检查点
 V-->>T: 验证指标与模型哈希
 T->>E: 下个回合原子切换已发布版本
```

完整例子：任务要求将被遮挡物体放入容器。Agent 提出直接抓取或先移开遮挡物；前置条件检查与结果预测共同筛选。若直接抓取预测不可信，则选择移开遮挡物；执行器用对应技能嵌入驱动 MPC，MuJoCo 回传结果。物体受扰动移动时，实际残差触发重新观测和候选评估。只有独立的“物体在容器内且保持稳定”检测器确认才完成。此例是设计行为，不是实验结果。

删除高层预测反馈后，低层 潜在动力学控制 仍可保留；丢失的是执行前比较候选结果及基于预测偏差触发重规划的能力。用候选排序准确率、无效尝试数、恢复成功率和整任务成功率量化，而非只看语言计划。
