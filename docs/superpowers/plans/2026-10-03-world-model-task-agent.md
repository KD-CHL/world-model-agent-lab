# G1 World-Model Task Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 G1 五节点任务图、独立真实验证、可信度短段执行与有限恢复。

**Architecture:** 在现有世界模型与固定控制器外增加任务运行时。提取原 A0–A3
选择逻辑供新旧路径复用，保持旧回执规则不变，新路径使用独立的部分执行账本。
监控只读取任务、预测和恢复事件，不拥有动作权限。

**Tech Stack:** Python 3.10、NumPy、现有 PyTorch/MuJoCo、unittest、原生 JavaScript。

**Spec:** `docs/superpowers/specs/2026-10-03-world-model-task-agent-design.md`

## Global Constraints

- 仅使用本项目 `wmal` 环境，在当前目录的 `codex/task-graph-agent` 开发，不新建工作树。
- 不训练、更换模型，不改变固定控制器、旧命令、检查点与校准格式。
- 两关节目标增量每步≤.06 rad，预测 H≤4 且不超过训练长度；真实历史 K 与模型一致。
- 总动作80、每节点24、决策120、总恢复4、单节点恢复2、连续拒绝3。
- 到达实测 L2≤.035 rad，命令 L2≤.015 rad；保持连续3次实际控制采样。
- A0只允许R0；恢复开关与A0–A3独立；未知执行结果必须锁止。
- 不推送、不合并、不接真机、不调用外部API；所有实验保存到新的 runs 子目录。

## Review Focus

- 非有限参数、布尔预算或未知配置字段：加载时拒绝，不能开始执行（任务1/4）。
- 旧观测/版本变化/候选数组被篡改：授权拒绝且没有第二次动作（任务2/3）。
- 保持计数跨节点、跨episode或由重复读数增长：拒绝或清零（任务1/3）。
- 取消、部分执行及观测接收失败：真实步数保留，未知进度锁止，不重试（任务3）。
- 多任务日志混合、末尾预算不足、扰动未触发：监控隔离与报告分母正确（任务4/5）。

## File Map

- `agents/task_graph.py`：不可变图与严格JSON输入。
- `agents/task_state.py`：节点/任务运行状态、有限预算。
- `skills/research_contracts.py`：reach/hold候选与能力检查。
- `skills/task_verifier.py`：真实到达、连续保持。
- `agents/selection_policy.py`：共享候选评估。
- `agents/task_execution.py`：一次性授权、逐动作执行与回执。
- `agents/recovery.py`：有限、显式恢复规则。
- `agents/task_runtime.py`：单线程闭环与事件/图像证据。
- `scripts/g1_task_agent.py`、`configs/tasks/g1_joint_sequence.json`：启动入口与冻结任务。
- `monitor/state.py`、`static/monitor.js`：旧日志兼容的任务图投影。
- `docs/28_task_graph_agent.md`、`docs/architecture/task-agent-verification.md`：命令与验收证据。

### Task 1: Task graph, capabilities and observed predicates

**Files:** Create task_graph.py、task_state.py、research_contracts.py、task_verifier.py；
Test `tests/test_task_graph.py`、`tests/test_task_verifier.py`。

**Interfaces:** `TaskNode(node_id, skill, target, dependencies=(), hold_steps=0, max_actions=24)`；
`TaskGraph(task_id, nodes, start=(.35,.87))`、`TaskGraph.from_dict(dict)`、`to_dict()`、`digest`；
`TaskState(graph)`、`snapshot()`；`JointVerifier(target, required_steps=0).update(observation, executed=False)`；
`JointSkills.candidates(observation, node, horizon, rng)` 返回现有 SkillCandidate 列表。

- [ ] 写失败测试：拒绝环/未知技能/越界/NaN/布尔预算；图哈希不受输入字典修改影响；
  hold静态读数不计时、3个连续执行采样才成功、失配清零、重复/跨episode拒绝。
- [ ] 运行 `/home/chl/miniconda3/envs/wmal/bin/python -m unittest discover -s tests -p 'test_task_*.py' -v`；Expected: 新模块缺失，测试不能通过。
- [ ] 实现上述接口；候选沿用旧脚本的方向/随机提案与限幅，hold仅零动作。
- [ ] 重跑相同命令；Expected: 新图和判据测试全部PASS。
- [ ] 提交 `feat: add observed task graphs and joint skill contracts`。

### Task 2: Shared world-model selection

**Files:** Create `agents/selection_policy.py`；Modify `agents/predictive_skill_agent.py`；
Test `tests/test_task_selection.py`，回归旧predictive/temporal/RSSM测试。

**Interfaces:** `select_candidates(predictor, observation, candidates, target, *, baseline,
calibration, error_budget, scale, remaining, state_lower, state_upper, history=None)` 返回
`(ranked, evidence)`；ranked沿用旧元组字段，保留候选隔离、版本/shape检查与旧评分。

- [ ] 写失败测试：A1不调用模型、A2完整长度、A3不跨越首个不可信步、错误模型版本拒绝。
- [ ] 运行 `python -m unittest discover -s tests -p 'test_task_selection.py' -v`；Expected: 新选择模块缺失。
- [ ] 原样提取旧选择算法；旧Agent调用同一函数，不修改旧完整反馈回执契约。
- [ ] 新测试与旧 `test_predictive_skill_agent.py`、`test_agent_temporal_history.py`、
  `test_agent_rssm_integration.py`、`test_temporal_world.py` 全PASS。
- [ ] 提交 `refactor: share calibrated candidate selection without changing legacy receipts`。

### Task 3: Bounded runtime, execution ledger and recovery

**Files:** Create task_execution.py、recovery.py、task_runtime.py；Test `tests/test_task_runtime.py`。

**Interfaces:** `RuntimeConfig(baseline='A3', recovery=False, horizon=4, error_budget=1., seed=0,
max_actions=80, max_decisions=120, max_recoveries=4, node_recoveries=2, max_rejections=3,
planning_timeout_s=5., execution_timeout_s=2.)`；
`TaskRuntime(graph, session, predictor, calibration=None, *, config, emit=None, artifact_dir=None)`；
`run(cancel=None)` 返回序列化报告；`ExecutionManager(session).authorize(observation, actions, binding, deadline)`；
`execute(token, stop=None, on_step=None)` 返回真实trace/实际计数/状态。恢复策略不重试未知执行。

- [ ] 写失败测试：真实五节点完成、保持失效R0停止/R1仅修复当前节点、总预算有限；
  A0/R1拒绝、保持拒绝预算、取消、部分正常停止、异常/错误step锁止、篡改授权拒绝；
  K2跨节点保留真实历史、模型变更不执行、预测超界逐步停止、无进展有限恢复。
- [ ] 运行 `python -m unittest discover -s tests -p 'test_task_runtime.py' -v`；Expected: runtime缺失。
- [ ] 实现单线程驱动与新部分回执协议；复用任务1/2；每步独立观测验证，实际进度先入账；
  记录零动作保持、恢复、预测RGB工件；不把异常视作可恢复预测失配。
- [ ] 全部新task测试PASS，并跑完整ROS-overlay回归，Expected: 无失败。
- [ ] 提交 `feat: run bounded world-model task graphs with auditable recovery`。

### Task 4: Runnable entry point, immutable config and monitor projection

**Files:** Create g1_task_agent.py、g1_joint_sequence.json、docs/28_task_graph_agent.md；
Modify monitor/state.py、static/monitor.js、README.md；Test `tests/test_task_agent_cli.py`、
`tests/test_task_monitor.py`、`tests/monitor_ui.test.cjs`。

**Interfaces:** script `main(argv=None)` 需要 --checkpoint、--config、--output；
可选 --calibration、--baseline、--recovery、--viewer、--monitor、--realtime、--disturbance；
配置schema `wmal.task_graph.v1`；沿用原task_started/plan/feedback/task_result事件。

- [ ] 写失败测试：CLI help与非法config/A0R1/缺校准在物理启动前拒绝；旧monitor日志不变，
  任务节点、保持计数、恢复和预算能显示且不跨task继承；Node真实DOM投影验证。
- [ ] 运行 CLI/monitor新测试和 `node --test tests/monitor_ui.test.cjs`；Expected: 新入口/投影缺失。
- [ ] 实现入口与显式reset、manifest输入哈希、持续viewer/monitor；test-only torque包装器
  施加+.5N·m一个周期且finally清零，日志不向Agent泄漏标签；显示真实节点与恢复状态。
- [ ] 重跑测试全PASS；保存用户命令，只使用 `conda activate wmal`。
- [ ] 提交 `feat: expose G1 task experiments and recovery monitoring`。

### Task 5: Real simulation acceptance and final review

**Files:** Create `tests/test_task_mujoco.py`、`docs/architecture/task-agent-verification.md`；
修改计划进度与验收文档，不修改冻结模型或开发阈值以制造成功。

**Interfaces:** TaskRuntime与CLI使用前面接口；实际报告含baseline/R开关、节点成功、
真实步数、恢复触发/成功、拒绝、预测残差、完整工件与终态。

- [ ] 写失败测试：真实MuJoCo任务保持推进sim time、扰动清零、fault后无后续动作；
  启动实际配置并读取真实task_result，不断言没有依据的A3性能优势。
- [ ] 运行新集成测试；Expected: 新物理验收尚未成立，定位具体缺口后实现必要边界。
- [ ] 使用现有冻结检查点分别执行干净场景A1/A2/A3开发smoke和一次+.5N·m受控扰动；
  每次新目录，失败和未触发均保留。学习型五节点未通过则报告，不改判据。
- [ ] 运行ROS-overlay完整 `python -m unittest discover -s tests -v`、Node测试、
  `git diff --check` 与编译检查；Expected: 全部自动化测试通过，无虚构验收。
- [ ] 独立整体代码审查，修复重要问题并补红绿测试；记录不可用审查或遗留项。
- [ ] 提交 `test: record task-agent simulation acceptance and limitations`，保留开发分支，不推送。

## Execution Decision

用户明确要求“完成实现计划后开始编写代码实现”。因此保存、自查计划后直接
在本会话逐任务执行（native），无需再询问同一启动意愿；最后安排一次独立审查。
文档审批与启动要求视为同一最新指令授权，未扩展到远程推送或真机操作。
