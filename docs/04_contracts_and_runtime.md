# 接口契约与 Ubuntu/MuJoCo 调度

Agent 与 ROS 2 接口使用 schema_version=1 的 JSON 负载。Observation 包含 robot_id、episode_id、step_id、sim_time_s、命名关节位置。MotionCommand 包含 command_id、机器人和观测标识、模式、有限目标/速度值与持续时长。HTTP/ROS 等待使用本机单调时间计时；ResetSimulation 生成新 episode。机器人节点拒绝旧 episode、旧 step 与重复命令。

| 类型 | 必填字段与语义 |
|---|---|
| TaskSpec | task_id、instruction、goal_predicates、constraints、max_control_steps、allowed_skill_ids；谓词引用已注册检测器 |
| SkillSpec | skill_id、parameter_schema、preconditions、termination_predicates、success_predicates、timeout_steps、recovery_skill_ids、max_retries、controller_version |
| Observation | robot_q_rad、robot_dq_rad_s、gripper_width_m、objects_pose_world、sensor_validity、capture_step、observation_profile；隐藏扰动参数不得进入本消息 |
| PredictionReport | model_version、observation_step、horizon_steps、predicted_state、objective_cost、不确定性类型与数据 |
| MotionCommand | command_id、robot_id、episode_id、expected_step、mode、named values、duration_s；模式和范围按 profile 校验 |
| ExecutionResult | command_id、succeeded/failed/canceled/rejected/timeout、简短状态 |

约定：右手世界坐标系，长度米、角度弧度、时间秒、力牛顿、力矩牛米；四元数统一 wxyz 并标明朝向。初期统一末端增量位姿+夹爪命令，由机器人适配器转换为关节目标；具体 IK/控制器实现前必须验证，不直接把位姿写入 actuator ctrl。传感器缺失用 null 和有效标志，不能补零冒充观测。

成功概率只有 score_semantics=calibrated_probability 且 calibration_version 匹配模型/控制器时可使用。方差和 ensemble 分歧使用各预测变量自身单位或平方单位，不能与概率相加。

错误枚举：INVALID_SCHEMA、UNKNOWN_SKILL、PRECONDITION_FAILED、STALE_OBSERVATION、EPISODE_MISMATCH、STEP_MISMATCH、DUPLICATE_REQUEST、MODEL_VERSION_MISMATCH、DEADLINE_EXCEEDED、SKILL_TIMEOUT、CONSTRAINT_VIOLATION、SIMULATION_ERROR。重复请求正常返回缓存回执并标记 duplicate，不再次推进仿真。

## 调度提案（未实测）

MuJoCo timer 按模型 timestep 积分，状态单独发布。Agent 在每个 ROS action 完成后等新 step 并重新调用 planner。探针 XML 的 timestep 为 0.002 s；Ubuntu 实时性未验证，项目不作周期截止承诺。

当前架构使用 ROS 2：单机器人仿真 owner 提供 state topic、execute_motion action 和 reset service；独立 planner node 提供 `/wmal/plan`；Agent 调用 API 并作为 ROS client。机器人场景与模型 owner 是唯一物理 step/reset 来源。ROS 2 尚未在当前 macOS 环境构建验证。

加速离线评估允许等待策略而暂停仿真时钟，须报告墙钟开销；实时调度实验不能隐去等待。过期命令拒绝，最多保持上个受限目标一个控制周期，然后进入已验证的保持/停止策略。保持策略本身需场景验证，不等于机械安全保证。

物理 owner 唯一调用 step/reset。模型写临时目录→校验哈希与接口→发布 manifest→在回合边界原子换快照。冻结评估全程禁止热更新。异步训练需预留推理资源，测量 GPU 争用及 deadline 违约率。

## 可复现回放

记录模型/XML及资产哈希、MuJoCo版本、积分器和接触求解配置、初始完整仿真状态（qpos/qvel/act/time、mocap、userdata及相关插件/求解器状态）、环境和采样 RNG 状态、外力/扰动事件、实际执行 ctrl 及保持步数、重置与模型切换。具体状态API在实现时核验官方文档。先验证同版本同机轨迹误差容限，再讨论跨机；视频只是可视化证据。
