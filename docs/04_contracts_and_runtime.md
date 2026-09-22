# 接口契约与 Ubuntu/MuJoCo 调度

全部消息使用版本化 JSON 可序列化对象。必填公共信封：schema_version、request_id(UUID)、episode_id、step_id(控制步非积分步)、sim_time_s、wall_time_utc、model_version。进程内超时使用单调时钟，不能用 UTC 时钟判断 elapsed。重置产生新 episode_id；相同 request_id 重试返回缓存结果，不重复执行动作。

| 类型 | 必填字段与语义 |
|---|---|
| TaskSpec | task_id、instruction、goal_predicates、constraints、max_control_steps、allowed_skill_ids；谓词引用已注册检测器 |
| SkillSpec | skill_id、parameter_schema、preconditions、termination_predicates、success_predicates、timeout_steps、recovery_skill_ids、max_retries、controller_version |
| Observation | robot_q_rad、robot_dq_rad_s、gripper_width_m、objects_pose_world、sensor_validity、capture_step、observation_profile；隐藏扰动参数不得进入本消息 |
| PredictionReport | candidate_id、skill_id、input_step、valid_until_step、prediction_horizon_steps、outcome_mean、outcome_variance、ensemble_disagreement、success_score、score_semantics、calibration_version、model_version、controller_version、status |
| ActionCommand | control_mode、values、frame、units、hold_physics_steps、expected_step、deadline_monotonic_ns、controller_version；模式和维度由机器人配置校验 |
| ExecutionFeedback | candidate_id、skill_id、start/end_step、success、terminated、truncated、termination_reason、observed_predicates、prediction_residual、error_code、detector_version |

约定：右手世界坐标系，长度米、角度弧度、时间秒、力牛顿、力矩牛米；四元数统一 wxyz 并标明朝向。初期统一末端增量位姿+夹爪命令，由机器人适配器转换为关节目标；具体 IK/控制器实现前必须验证，不直接把位姿写入 actuator ctrl。传感器缺失用 null 和有效标志，不能补零冒充观测。

成功概率只有 score_semantics=calibrated_probability 且 calibration_version 匹配模型/控制器时可使用。方差和 ensemble 分歧使用各预测变量自身单位或平方单位，不能与概率相加。

错误枚举：INVALID_SCHEMA、UNKNOWN_SKILL、PRECONDITION_FAILED、STALE_OBSERVATION、EPISODE_MISMATCH、STEP_MISMATCH、DUPLICATE_REQUEST、MODEL_VERSION_MISMATCH、DEADLINE_EXCEEDED、SKILL_TIMEOUT、CONSTRAINT_VIOLATION、SIMULATION_ERROR。重复请求正常返回缓存回执并标记 duplicate，不再次推进仿真。

## 调度提案（未实测）

起始参数：physics_dt=0.002 s，action_repeat=10，控制周期0.020 s；状态每控制步采样，图像/渲染另设周期，初期无图像控制。Agent 仅在技能边界/事件调用。参数必须在目标 Ubuntu 硬件做时延与稳定性实验后调整。

最小方案：单进程直接调用，显式逻辑步进，训练与评估分时运行。异步扩展：每个环境一个 owner 进程；训练器独立进程；结构化控制消息经有界队列传递，大图像后续才用共享内存。ROS 2 仅在实机/外部生态集成时引入，网络中间件仅用于跨机任务。队列易调试但有复制开销；共享内存减少复制但需生命周期和同步；ROS 2 增加配置与部署成本。

加速离线评估允许等待策略而暂停仿真时钟，须报告墙钟开销；实时调度实验不能隐去等待。过期命令拒绝，最多保持上个受限目标一个控制周期，然后进入已验证的保持/停止策略。保持策略本身需场景验证，不等于机械安全保证。

物理 owner 唯一调用 step/reset。模型写临时目录→校验哈希与接口→发布 manifest→在回合边界原子换快照。冻结评估全程禁止热更新。异步训练需预留推理资源，测量 GPU 争用及 deadline 违约率。

## 可复现回放

记录模型/XML及资产哈希、MuJoCo版本、积分器和接触求解配置、初始完整仿真状态（qpos/qvel/act/time、mocap、userdata及相关插件/求解器状态）、环境和采样 RNG 状态、外力/扰动事件、实际执行 ctrl 及保持步数、重置与模型切换。具体状态API在实现时核验官方文档。先验证同版本同机轨迹误差容限，再讨论跨机；视频只是可视化证据。
