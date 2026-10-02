# 时序世界模型工程验收（2026-10-02）

实现规格与机制边界见 `docs/superpowers/specs/2026-10-02-temporal-world-model-design.md`；
用户命令与科研指标见 `docs/27_temporal_world_model_engineering.md`。

## 实测证据

- 使用本项目 `wmal` 环境，未安装上游项目环境或 JAX。
- 新增 16 项行为测试：历史影响、候选隔离、padding/reset、未来目标不可输入推演、
  模型版本/旧模型兼容、训练/冻结微调/独立校准/评估、历史 trace 原子校验、
  部分失败锁止、逐步超界重观测、真实 MuJoCo 每步观测、反馈接收器失败锁止。
- 最终完整 ROS-overlay `unittest`：206 项通过，无跳过；19.839 秒。
- `git diff --check` 与 Python 编译检查通过。
- 旧演示检查点仍可加载，版本 `visual-fd46b9bdceb6f45ce944`，context_steps=0。

真实数据来自本机既有 MuJoCo 仿真 episode，而不是人工捏造模型输出。
40 个 episode、640 次动作转移，train/validation/calibration/test = 24/4/8/4。
训练配置 `configs/training/rssm_temporal_world.json`（K=2、H=4），本轮验收临时覆盖轮数。

| 阶段 | 本机工件（被 Git 排除） | 结果 |
| --- | --- | --- |
| 12 轮 CUDA 训练 | `runs/rssm_temporal_smoke_20261002/` | val loss 1.2928958795，18.67 s |
| 2 轮冻结编码器继续学习 | `runs/rssm_temporal_finetuned_smoke_20261002/` | val loss 1.2651932881，3.17 s |
| 校准 | 上述微调目录 `calibration.json` | 8 独立 episode，alpha=.2，绑定微调模型版本 |
| 测试 | 上述微调目录 `evaluation.json` | 4 episode / 52 windows，结果不全优于保持基线 |
| A2 闭环 | `runs/rssm_temporal_agent_a2_trace_20261002/` | 8 实际动作，预算耗尽；2 个完整四步真实 trace 工件 |
| A3 闭环 | `runs/rssm_temporal_agent_a3_20261002/` | 3 实际动作到达关节目标，误差范数 .03372 rad；仅一目标一种子 |

微调模型版本 `visual-6218b8dd9f7ddb953a8f`。A3 的三个候选计划分别使用 0、1、2 步真实
历史；三个实际步骤均未超出校准界。不能由此推断视觉质量良好、操作任务完成或 A3 普遍优于 A2。
没有替换现有演示默认模型，没有进行真机测试。

## 审查边界

独立代码审查代理启动后因服务额度限制失败，未返回报告。本轮仅完成作者自查、
自动化测试及仿真实测，不能宣称独立审查通过。共享分支合并/推送之前仍建议独立审查。
代码保留在本地 `codex/temporal-rssm`，本轮没有自动推送或合并 main。
