# G1 任务图 Agent：启动与实验边界

实现方案见 [设计规格](superpowers/specs/2026-10-03-world-model-task-agent-design.md)，
开发步骤见 [实现计划](superpowers/plans/2026-10-03-world-model-task-agent.md)。

运行时组织 reach_A → hold_A → reach_B → hold_B → return_start，保留固定低层
控制器与世界模型。两维动作是肩/肘命令目标增量，不是抓取或全身行走。
保持必须连续执行三个真实零增量控制周期；反复读取画面不算保持时间。

## 环境与无窗口运行

所有 Python 命令仅使用本项目环境。下列本机模型路径指向已有冻结检查点；
其他机器须先按世界模型训练指南获取兼容模型及独立校准，不随 Git 分发权重。

```bash
cd /home/chl/GitHub/CodeSpace/world-model-agent-lab-main
conda activate wmal
export PYTHONPATH="$PWD/src"
export MUJOCO_GL=egl

python scripts/g1_task_agent.py \
  --config configs/tasks/g1_joint_sequence.json \
  --checkpoint runs/rssm_temporal_finetuned_smoke_20261002/best.pt \
  --calibration runs/rssm_temporal_finetuned_smoke_20261002/calibration.json \
  --baseline A3 --recovery \
  --output "runs/task_agent_a3_$(date +%Y%m%d_%H%M%S)"
```

此命令运行真实学习型预测，并不保证当前小数据模型能完成五节点。预测不可信、
动作停滞或预算耗尽会保存失败/需要检查状态，不能把终端退出码当作任务成功。
任务结果以 task_000.json 的 state.status 与真实完成证据为准。

## 持续 MuJoCo 窗口与客户端

```bash
conda activate wmal
export PYTHONPATH="$PWD/src"
unset MUJOCO_GL

python scripts/g1_task_agent.py \
  --config configs/tasks/g1_joint_sequence.json \
  --checkpoint runs/rssm_temporal_finetuned_smoke_20261002/best.pt \
  --calibration runs/rssm_temporal_finetuned_smoke_20261002/calibration.json \
  --baseline A3 --recovery --viewer --monitor --realtime \
  --monitor-port 8766 \
  --output "runs/task_agent_live_$(date +%Y%m%d_%H%M%S)"
```

打开终端打印的客户端链接。任务结束后窗口保持打开；输入 `run` 显式重置
并启动新的任务/episode，输入 `quit` 退出。故障锁止不可通过重新发同一动作解除。
窗口空闲物理步不回填已结束任务；下一任务必须 reset，避免伪造时序历史。
客户端只读，显示节点、保持采样、动作预算、恢复次数、预测与真实反馈。

## 对照与扰动

同一配置支持 --baseline A0/A1/A2/A3。A0不调用模型且只允许 --no-recovery；
A1按名义结果修正，--horizon 1 可用于控制器排查；A2/A3比较时应保持相同跨度、
候选种子与预算。A3必须指定匹配模型版本的校准。恢复通过 --recovery 独立开关。

增加 `--disturbance shoulder-pulse`：在 hold_A 首次保持动作之后，对肩关节施加
.5 N·m一个周期的外力矩，然后恢复原外力。只用于模拟实验，无真假标签提供
给 Agent。扰动可能未使条件失效；必须区分“外力已注入”与“恢复被触发”。
不要调大扰动或放宽成功阈值来制造方法优势。条件变化后的覆盖率需另外评估。
`recoveries_succeeded` 只计入实测确认的条件重建，重新发起规划本身不算恢复成功。
扰动结果中的 recovery_triggered 指 hold_A 的恢复尝试，后续 B 点的重规划另计，
不能归因于肩关节脉冲。

## 输出与解释

每次实验使用新/空目录，包含：

- run_manifest.json：配置、图、代码实现、检查点、校准及控制器溯源。
- events.jsonl：节点、候选、计划、逐步条件、实际执行、部分停止与恢复。
- task_*.json：终态、节点证据、实际步数账本、预测残差及扰动记录。
- prediction_*.npz：实际执行前缀与逐步预测/真实 RGB 和状态。

到达与保持要求实测误差≤.035 rad、命令误差≤.015 rad，均为两关节 L2范数。
启动预检只验证命令起点与实测实验包络，不宣称机器人已经稳定。既有伺服存在
重力/瞬态跟踪偏差，可能令某些固定目标与严格保持条件无法同时满足。
完整验收和未通过项见 [验收记录](architecture/task-agent-verification.md)。
