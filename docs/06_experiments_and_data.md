# 实验、数据字典与图表

## 场景与划分

共享桌面资产只加载一次，各技能在同一场景连续执行。任务依赖：clear_obstacle → pick_target → place_target；终态需目标物进入容器且连续若干控制步保持，不以奖励阈值替代。稳定步数、容差、超时在验证集预设。训练/校准/验证/测试采用互斥 scene_group 与 episode_id；未见组合测试不得使用测试组合调参。Meta-World 只作单技能参照，自建任务另命名与发布协议。

状态版向所有方法公开同样的关节及物体状态。质量/摩擦、注入时刻等仅评测器可见。视觉扩展移除 objects_pose_world，并独立说明成功检测器的特权信息；不能比较状态方法与视觉方法后归因于算法。

## 对照

| ID | 高层 | 低层 | 预测/触发 | 解释 |
|---|---|---|---|---|
| B0 | 固定任务图 | 冻结潜在动力学控制技能 | 无高层预测 | 有限已知任务参考，不称通用oracle |
| B1 | LLM | 相同技能 | 仅执行反馈 | Agent闭环基线 |
| B2 | 同一LLM与候选 | 相同技能 | 结果预测+固定周期 | 基本预测收益 |
| B3 | 同一LLM与候选 | 相同技能 | 结果预测+原始分数阈值 | 校准收益对照 |
| Ours | 同一LLM与候选 | 相同技能 | 校准门控+事件触发 | 核心候选方法 |
| L0 | 固定同一高层 | 等预算SAC技能（拟定） | 无学习世界模型 | 低层算法贡献，单独报告 |

增加 Ours 去校准/去过滤/改固定周期消融，采用同候选离线排序回放减少语言随机性，再做完整闭环评价。高层实验冻结同一控制器，L0不能和高层消融混为一组结论。

第一轮只选任务长度1/3和质量或摩擦偏移一种扰动；之后扩展未见组合和延迟。使用互斥训练与测试参数范围，具体数值先由任务可行性预实验确定并锁定。主实验建议5训练种子×每种子50测试回合，属于计划规模而非充分统计功效保证。调参预算对所有方法披露。

## 假设—证据映射

| 问题 | 数据字段 | 指标/统计单位 | 图表 |
|---|---|---|---|
| H1完成率提升 | task_success, train_seed, scene_group, variant | 每训练种子的成功率；跨种子95%区间 | 配对成功率图 |
| 预测有效性 | candidate_id, predicted_outcome, observed_outcome | 位姿误差/排序相关性，按回合聚合 | 预测误差与任务结果 |
| 校准与检测 | score_semantics, success_label, trigger, later_failure | Brier/可靠性图；事件精确率召回率及误报漏报 | 校准图和触发权衡 |
| 恢复能力 | disturbance_id, recovery_start/end, recovered | 恢复率与恢复仿真时间；失败单列 | 恢复曲线 |
| 资源约束 | sim_steps, imagined_steps, wall_ns, calls | 独立计数；p50/p95/p99与deadline违约率 | 完成率—延迟曲线 |
| 失效归因 | failure_category, detector_version | 规划/不可达/模型/控制/通信/检测 | 失败类别图 |

成功率主要统计单位是独立训练种子；回合嵌套种子，采用分层bootstrap或种子级配对分析，不能将250回合冒充250训练重复。5种子区间仍可能不稳定，报告原始种子结果。运行崩溃单列并预定是否重跑，不能静默删除。检查点只按验证集主要指标选择，相同时选较早检查点。

## 存储规范

- run_manifest.json：run_id、variant、git_commit或源码哈希、配置哈希、模型/校准/控制器/检测器版本、硬件软件、全部种子、数据版本。
- transitions：episode_id、step_id、sim_time_s、capture_time、observation、commanded_action、executed_action、hold_steps、reward、next_observation、terminated、truncated、reason、source。数组用分片NPZ，索引用JSONL；大规模后再考虑Zarr。
- 第二数据源另记 `source_kind`、`source_dataset`、`source_episode_id`、`parent_trajectory_id`、`generator_checkpoint`、`license_record` 与 `executed_in_mujoco`；模型动作在 MuJoCo 执行后仍计为仿真交互，模型生成视频不能当作环境真值。准入与可行性见[视觉动作模型支线](12_secondary_data_feasibility.md)。
- skill_segments.jsonl：候选、技能参数、初态/终态、执行跨度、成功/超时/censor、控制器版本、phase。
- events.jsonl：请求/回复、结构化Agent输入输出、候选分数、预测跨度、触发原因、原始观测引用、时延和错误码。
- evaluation_privileged.jsonl：隐藏扰动、场景真值、评测标签，单独目录且Agent接口不读取。
- figures/及tables/：仅由分析入口从原始数据生成，保存命令和输入manifest。

观测/动作每控制步记录，预测每候选请求记录，事件每次触发记录；视频按渲染周期采样。记录频率不可用论文图表抽样频率代替。
