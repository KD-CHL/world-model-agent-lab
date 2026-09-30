# 技能运行边界

- `registry.py`：SkillSpec、命名注册和可用能力列表。
- `executor.py`：TaskSpec、ExecutionFeedback、前置条件、周期预算和独立成功复核。
- `predicates.py`：G1 导航的姿态前置条件与位置/朝向成功检查。

关节和视觉策略适配器见 `agents/skill_adapters.py`；G1 任务恢复见 `locomotion/mission.py`。
完整设计、ROS2 启动、校准和验收见 [Agent 架构优化](../../../docs/architecture/agent-runtime.md)。
