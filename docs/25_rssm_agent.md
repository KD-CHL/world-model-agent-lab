# RSSM 世界模型与多子目标 Agent

这条路线在现有视觉网络上加入**确定性记忆 + 离散随机潜变量 + 动作先验 + 观测后验**。
旧 CNN/GRU 集成仍为默认对照，不是把原网络改名为 Dreamer。
设计、固定版本源码与研究假设见 [完整规格](superpowers/specs/2026-10-02-rssm-agent-design.md)，
真实验收见 [报告](architecture/rssm-agent-verification.md)。

## 1. 系统分层

```text
目标/有序子目标（未来可由冻结 LLM 提议，需校验）
             ↓
任务调度器：active_stage / 已完成子目标 / 预算 / 真实观测 belief
             ↓
固定技能生成候选 [H,A] → 动作/状态语义和限位检查
             ↓
RSSM 集成：观测后验初始化 → 候选动作先验 rollout → 未来 RGB/状态
             ↓
独立校准误差界 → 排序 / 连续可信短前缀 / 拒绝执行
             ↓
固定 G1 伺服控制器 → 真实 MuJoCo 观测 → 预测残差与子目标核验
             └──────────────→ 调度器更新 / 重观测 / 下一子目标
```

Agent 不直接积分机器人状态，不用预测图像宣布成功。当前实际技能是固定底座
G1 两关节 `arm_delta`；waypoint 验证多步组织，不等于已实现抓取、堆叠、擦桌。
浮动底座行走路径不变，不能把这个模型的关节增量当底座速度。

最小状态包含既有六项以及 `active_subgoal`、`decision_count`、`reobservations`、
`belief`。belief 明确是**实测状态与预测证据账本**，不是长期隐变量记忆。
无可信候选连续重观测三次后 `needs_review`，完整执行反馈清零连续计数；
部分失败进入 `execution_failed`，不自动重发。任务总动作预算不因子目标切换重置。
观测与决策数组不可写，回执绑定当前 pending 决策；回合、步号、模型版本仍校验。

## 2. 网络与梯度

`src/wmal/models/categorical_rssm.py` 独立实现，PyTorch，无 JAX 依赖：

```text
token_t = concat(CNN(rgb_t), MLP(symlog(normalize(state_t))))
h_t     = GRU([z_(t-1), normalize(action_(t-1))], h_(t-1))
p_t     = prior(h_t)                         # 不读取未来观测
q_t     = posterior(h_t, token_t)            # 仅真实观测更新
z_t     = categorical(q_t)                  # 训练 straight-through
feature = concat(h_t,z_t)
heads   = RGB decoder + symlog 状态 head + 可选 masked 事件 head
```

训练总损失：

```text
L = frame_weight * prior-only H步 RGB 误差
  + state_weight * prior-only H步 symlog状态误差
  + reconstruction_weight * 后验 RGB/状态重构
  + dyn_weight * max(KL(stopgrad(q)||p), free_nats)
  + rep_weight * max(KL(q||stopgrad(p)), free_nats)
  + event_weight * 有实际监督 mask 的事件 BCE
```

两个 KL 项同值但梯度方向不同；free nats 在随机变量/类别求和后使用。
unimix 避免类别分布退化到零概率。窗口完整且不跨 episode；显式 reset
同时清除历史 h,z 和前一动作。未来目标图像只参与后验训练/监督，不进入规划。

验证、规划、校准统一使用类别概率的确定性软状态代理，而非每次随机抽样。
它**不是非线性 stochastic rollout 的精确期望**。集成成员保持自己的轨迹；
entropy 只作为诊断，ensemble spread 只是离散程度，A3 使用的可信度仍来自
独立 calibration 误差界。它们都不是碰撞/成功概率或真机安全证明。

本轮在线每次从当前真实观测初始化，候选段内 recurrent rollout。网络支持
观测序列 API，但跨控制周期携带 latent memory 尚不启用；将来启用必须同时
修改训练上下文、离线校准与在线观测协议，单独消融，不能只改在线 carry。

Dreamer 还训练 reward/continuation 和想象 actor–critic。本项目当前数据没有
这些真实标签，故不伪造 reward、不把文件最后一帧当环境终止，也不声称复现
完整 DreamerV3。这是监督动作后果世界模型，不是已经训练好的强化学习策略。

## 3. 环境与数据

所有命令在项目根目录，仅使用本项目环境：

```bash
cd /home/chl/GitHub/CodeSpace/world-model-agent-lab-main
source /home/chl/miniconda3/etc/profile.d/conda.sh
conda activate wmal
export PYTHONPATH=src
export MUJOCO_GL=egl
```

若需要补充依赖，只在 `wmal` 内执行 `python -m pip install -e '.[visual-world,simulation]'`。
不要创建 Dreamer/Every-Embodied 的环境，不更换现有 CUDA 构建。

已有本机数据：`data/processed/visual_g1_20260930/manifest.json`。重新采集：

```bash
python scripts/visual_world.py collect --output data/processed/rssm_g1_v1 \
  --task stack_block --episodes 40 --steps 16 --image-size 64 --seed 0
```

训练集只用 train episode 归一化，validation 选型，calibration 校准，test 冻结后评估。
本机 40 episode 的 24/4/8/4 划分只是工程验收规模，不够证明泛化。
开源 LeRobot 导入见 [原训练指南](23_visual_world_training_and_agent.md)。
16 维操作数据必须明确动作/状态/相机顺序；可训练独立模型，但不可直接驱动
当前 2 维 G1 伺服。视频权重迁移需另设 backbone adapter 与兼容性验证。

## 4. 训练、微调、校准、评估

训练新模型（CUDA 可用时用 `--device cuda`，CPU 改为 `cpu`）：

```bash
python scripts/visual_world.py train \
  --manifest data/processed/visual_g1_20260930/manifest.json \
  --config configs/training/rssm_world.json --output runs/rssm_g1_v1 --device cuda
```

输出：`best.pt`、`last.pt`、`history.json`、`training_report.json`。默认显示 epoch、
成员/批次、损失、耗时/ETA；用 `--no-progress` 可关闭。报告含 KL、重构和 entropy。
运行目录须新建/为空，防止覆盖证据。`--pretrained` 是微调，不是 optimizer/RNG 续训。

```bash
python scripts/visual_world.py train \
  --manifest data/processed/visual_g1_20260930/manifest.json \
  --config configs/training/rssm_world.json --output runs/rssm_g1_finetuned \
  --pretrained runs/rssm_g1_v1/best.pt --freeze-encoder --epochs 10 --device cuda

python scripts/visual_world.py calibrate \
  --manifest data/processed/visual_g1_20260930/manifest.json \
  --checkpoint runs/rssm_g1_finetuned/best.pt \
  --output runs/rssm_g1_finetuned/calibration.json --horizon 4 --alpha 0.2 --device cuda

python scripts/visual_world.py evaluate \
  --manifest data/processed/visual_g1_20260930/manifest.json \
  --checkpoint runs/rssm_g1_finetuned/best.pt \
  --calibration runs/rssm_g1_finetuned/calibration.json \
  --output runs/rssm_g1_finetuned/evaluation.json --horizon 4 --device cuda
```

冻结同时覆盖 RGB 与实测状态编码器，归一化坐标不变。这里用同一仿真数据只是
微调管线演示；后续科研应换成兼容语义的新训练数据，完整保留祖先暴露 lineage。
旧确定性 checkpoint 或官方 Dreamer 权重不能当 RSSM pretrained；需架构/语义匹配。
换权重后版本变化，必须重新校准；旧校准自动拒绝新版本。

## 5. 真正调用与持续仿真

```python
from wmal.models.visual_latent import VisualWorldModel
model = VisualWorldModel.load('runs/rssm_g1_finetuned/best.pt', device='cpu')
result = model.predict(rgb_chw_float01, measured_state, candidate_actions_h_by_a)
# states [member,H,D], frames [member,H,3,S,S], diagnostics；不会执行机器人
```

使用上述新训练权重启动 Agent；预测质量不够时可能拒绝，并不保证目标成功：

```bash
python scripts/visual_skill_agent.py \
  --checkpoint runs/rssm_g1_finetuned/best.pt \
  --calibration runs/rssm_g1_finetuned/calibration.json \
  --baseline A3 --waypoints 0.40 0.90 0.45 0.95 --goal 0.35 0.85 \
  --horizon 4 --max-cycles 40 --error-budget 0.5 \
  --output runs/rssm_g1_mission --device cpu --monitor --monitor-port 8766 --realtime
```

打开 `http://127.0.0.1:8766`；任务结束会话继续运行，终端输入新目标或 `quit`。
如需原生 MuJoCo 窗口，加 `--viewer`；本机显示不可用时保持浏览器实时画面即可。
故障锁定后需 `reset` 显式重置。下一次终端目标不重复最初 waypoint。
输出 `events.jsonl`、`task_000.json`、预测/实测对齐 NPZ；客户端只读。
A0 多子目标使用 `--horizon 1` 与预先固定整段计划，其余基线共享当前子目标输入。

本轮实际短训练路径及运行结果以验收报告为准，不能把上面正式实验目录与
smoke 权重混用，也不能因一次近距离目标通过就宣布多场景能力。

## 6. 论文与后续网络路线

推荐题目方向：**基于世界模型预测可信度的机器人智能体分层规划与执行修正**。
可研究的主机制是任务相关误差预算统一决定候选排序、执行长度和重观测/恢复，
而不是“RSSM + Agent”组合本身。支撑贡献为动作后果网络与可审计闭环评测。

优先数据微调路线：先验证兼容 RGB/本体状态/动作的编码器预训练，再冻结或少量
微调编码器、训练 RSSM/head；不能无验证把 DINO/UniFoLM 权重装进此 CNN。
从零路线仅训练紧凑世界模型，不从零训练完整 Agent/controller。

固定数据划分/控制器/候选预算，比较：

- 旧确定性网络 vs RSSM；等参数量或等训练时间，多个 seed。
- 去 KL、去 posterior 重构、去开环多步监督；检查 representation collapse 与长视野误差。
- A0/A1/A2/A3，未校准 spread，固定执行前缀，无真实残差反馈。
- 单子目标 vs 顺序 waypoint；扰动与失败恢复；固定其余模块后才引入冻结 LLM。

报告 persistence/变化区域视觉误差、多步状态误差、动作敏感性、校准覆盖/宽度、
子目标/全任务成功、拒绝率、恢复代价、规划时延。提前固定检索/历史规则，
最终测试不用于选网络或调 error_budget。物理抓取能力仍需真实接触资产、
夹爪控制和独立成功检测；不是添加场景名字即可成立。
