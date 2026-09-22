# 提示词优化的文献依据与阅读边界

日期：2026-09-22。来源为本地 PDF；未进行外部发表信息、最新代码或新颖性检索。

## 实际盘点

当前路径共有 145 篇 PDF。历史库位置仅用于定位思路，数量与内容以本次盘点为准。

- Agent_and_Task_Planning: 29
- Control_and_Reliability: 24
- Perception_and_State_Representation: 10
- Robot_Learning_and_Simulation: 14
- World_Model_and_Dynamics: 63
- survey: 5

## 阅读范围

对筛选论文提取可读全文，再定向阅读摘要和相关方法/实验页面；5 篇综述均查看首页与摘要，部分查阅分类与控制章节。下表仅列实际用于优化提示词的证据。未逐篇精读全部 145 篇，未进行 PDF 版面或图表数值全面审计，未执行论文代码。页码为 PDF 文件的物理页码，从 1 开始。

## 方法到提示词的映射

| 文献 | PDF 页码/章节 | 本次阅读得到的方法依据 | 转化为提示词的约束及边界 |
|---|---|---|---|
| [TD_MPC2_Scalable_Robust_World_Models_for_Continuous_Control](</Users/chl/文献库/World_Model_and_Dynamics/TD_MPC2_Scalable_Robust_World_Models_for_Continuous_Control.pdf>) | 3–5，§3 | 控制相关潜在转移、奖励/价值学习和短视野 MPC；Q ensemble 用于价值学习。 | 主线候选；不得把 Q 集成当成动力学不确定性，也不假设自带语言技能接口。 |
| [Mastering_Diverse_Control_Tasks_through_World_Models](</Users/chl/文献库/World_Model_and_Dynamics/Mastering_Diverse_Control_Tasks_through_World_Models.pdf>) | 3，Critic learning / Actor learning | 从想象轨迹学习 actor–critic，环境交互由 actor 选动作，无需前瞻搜索。 | 训练与执行分开；不能把 Dreamer 描述为逐步在线 MPC。 |
| [DayDreamer_World_Models_for_Physical_Robot_Learning](</Users/chl/文献库/World_Model_and_Dynamics/DayDreamer_World_Models_for_Physical_Robot_Learning.pdf>) | 3，§2 | 学习与采样解耦，learner 与 actor 并行。 | 借鉴调度；论文的真实机器人实验不是本项目 MuJoCo 实验证据。 |
| [DINO_WM_World_Models_on_Pre_trained_Visual_Features_Enable_Zero_shot_Planning](</Users/chl/文献库/World_Model_and_Dynamics/DINO_WM_World_Models_on_Pre_trained_Visual_Features_Enable_Zero_shot_Planning.pdf>) | 3–5，§3 | 冻结 DINOv2 patch 表征；离线动作条件预测；目标图像与 CEM-MPC。 | 备选视觉主线；目标图像获取与语言映射需单独设计。 |
| [Deep_Reinforcement_Learning_in_a_Handful_of_Trials_using_Probabilistic_Dynamics_Models](</Users/chl/文献库/World_Model_and_Dynamics/Deep_Reinforcement_Learning_in_a_Handful_of_Trials_using_Probabilistic_Dynamics_Models.pdf>) | 3–4，§3–4 | PETS 区分随机与认知不确定性，使用概率动力学集成。 | 不确定性估计须对应模型结构；迁移至潜在空间需另行验证。 |
| [Do_As_I_Can_Not_As_I_Say_Grounding_Language_in_Robotic_Affordances](</Users/chl/文献库/Agent_and_Task_Planning/Do_As_I_Can_Not_As_I_Say_Grounding_Language_in_Robotic_Affordances.pdf>) | 4–5，§3–4 | 语言相关性与技能可执行性结合；区分规划与执行成功率。 | 高层候选筛选及分层评价；价值函数不天然是通用校准成功概率。 |
| [Inner_Monologue_Embodied_Reasoning_through_Planning_with_Language_Models](</Users/chl/文献库/Agent_and_Task_Planning/Inner_Monologue_Embodied_Reasoning_through_Planning_with_Language_Models.pdf>) | 3–4，§3 | 将成功检测与场景反馈送入规划过程。 | 闭环 Agent 基线；反馈来源和真值访问必须披露。 |
| [ProgPrompt_Generating_Situated_Robot_Task_Plans_using_Large_Language_Models](</Users/chl/文献库/Agent_and_Task_Planning/ProgPrompt_Generating_Situated_Robot_Task_Plans_using_Large_Language_Models.pdf>) | 3–4，§III–IV | 以可用动作与对象约束规划；执行中检查条件。实机桌面实验未实现同样的 assert 闭环。 | 技能契约与前置条件；不能把仿真闭环结论无条件泛化到实机。 |
| [PIVOT_R_Primitive_Driven_Waypoint_Aware_World_Model_for_Robotic_Manipulation](</Users/chl/文献库/World_Model_and_Dynamics/PIVOT_R_Primitive_Driven_Waypoint_Aware_World_Model_for_Robotic_Manipulation.pdf>) | 3–4，§3及图2 | 原语、任务相关路标与分层异步执行。 | 层间时间尺度与接口启发；不代表可无改动接入 TD-MPC2。 |
| [Closed_Loop_Visuomotor_Control_with_Generative_Expectation_for_Robotic_Manipulation](</Users/chl/文献库/World_Model_and_Dynamics/Closed_Loop_Visuomotor_Control_with_Generative_Expectation_for_Robotic_Manipulation.pdf>) | 3、6，§3及算法1 | 视觉子目标、可度量表征误差、子目标切换及重规划。 | 监测与恢复机制参考；本次不采纳其性能数值作为预期增益。 |
| [When_to_Trust_Your_Model_Model_Based_Policy_Optimization](</Users/chl/文献库/World_Model_and_Dynamics/When_to_Trust_Your_Model_Model_Based_Policy_Optimization.pdf>) | 4、8，§4.2及§6.2 | 从环境数据分叉的短模型 rollout；比较模型使用方式与偏差。 | 模型生成数据与真实仿真交互分开记录；不是通用最优 rollout 长度证明。 |
| [Meta_World_A_Benchmark_and_Evaluation_for_Multi_Task_and_Meta_Reinforcement_Learning](</Users/chl/文献库/Robot_Learning_and_Simulation/Meta_World_A_Benchmark_and_Evaluation_for_Multi_Task_and_Meta_Reinforcement_Learning.pdf>) | 2，图1及介绍 | 共享操作结构的多任务/元学习任务集。 | 技能评估参考；连续长时序场景是新增设计，不能冒称标准协议。 |
| [MuJoCo_A_Physics_Engine_for_Model_Based_Control](</Users/chl/文献库/Robot_Learning_and_Simulation/MuJoCo_A_Physics_Engine_for_Model_Based_Control.pdf>) | 1，摘要及§I | 面向模型控制的物理仿真、接触动力学与仿真模型。 | 作为交互环境；历史论文不用于证明当前软件版本兼容性或运行速度。 |
| [s44163-026-02122-1](</Users/chl/文献库/survey/s44163-026-02122-1.pdf>) | 3、21–23，§3.6–4.1 | 状态、时间、不确定性、结构、模态与决策耦合六个维度。 | 架构设计坐标；控制效果、校准与计算预算需要共同评价。 |
| [SmartBot - 2026 - Wang - World Models for Robotic Manipulation  A Survey](</Users/chl/文献库/survey/SmartBot - 2026 - Wang - World Models for Robotic Manipulation  A Survey.pdf>) | 1、3，§1–2 | 区分预测表征、预测与动作的联系和学习生命周期中的角色。 | 不能仅凭预测视觉质量证明机器人收益；注意方法适用成本。 |
| [2609.16074](</Users/chl/文献库/survey/2609.16074.pdf>) | 1、7，架构分类 | 世界预测与动作生成的耦合；端到端与双系统接口。 | 动作对齐必须成为接口问题；仅作综述分类依据，未核验其中所有新方法。 |

## 本项目建议与论文结论的区分

- 单机械臂、先状态后视觉、一条世界模型主线、结构化消息契约、模型版本管理与数据泄漏防护，是针对本项目的工程与实验设计建议，不冒充某篇论文的原始方法。
- “预测可信度驱动的候选筛选与事件重规划”仅是候选研究假设。尚未证明新颖性、收益或优于普通执行反馈。
- TD-MPC2、PETS、SayCan 等具有不同训练目标和接口，不能直接拼接后宣称已完成方法融合。特别需要验证低层动作预测如何支持高层技能评估。
- 所有论文中出现的性能数字均未被用作本项目的预期效果或验收承诺。
- 部分 PDF 提取出现字体数值解析警告；本轮依据可读方法文本，不依赖有歧义的公式字形和表格数值。


## 迁移证据矩阵补充
| 方法 | 适用假设 | 迁移改动 | 验证实验 |
|---|---|---|---|
| TD-MPC2 | 状态、奖励、任务标识可用 | 离散技能映射到任务嵌入；场景内技能切换 | 原算法单技能基线与共享场景技能成功率 |
| Dreamer/DayDreamer | 序列观测与奖励可用 | Ubuntu/MuJoCo 环境适配 | 等交互预算学习曲线；初期不实现 |
| DINO-WM | 离线动作轨迹和目标图像可用 | 相机、目标获取和动作尺度适配 | 目标图像规划与未见布局；后续扩展 |
| PETS | 可学习的状态转移分布 | 仅借鉴 bootstrap 概率预测思想，应用到技能结果模型 | ensemble 对单模型、校准前后对比 |
| SayCan | 技能集合与可执行性估计可用 | 用经过验证的技能结果预测作筛选；不复制原始概率乘积 | 相同候选下排序与执行收益 |
| Inner Monologue | 执行检测器能提供反馈 | 结构化 ExecutionFeedback | 开环与执行反馈基线 |
| ProgPrompt | 有限技能 API 和对象集合 | 白名单及机器可验证前置条件 | 非法技能率、条件检查消融 |
| PIVOT-R | 层间中间表示与不同执行频率 | 借鉴异步调度，不复制路标模型 | 等预算固定周期与事件触发 |

以上是迁移设计，不是已实现或已验证的组合方法。
