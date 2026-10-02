# 时序世界模型：研究架构、工程契约与运行指南

本轮不是增加一个“大模型名字”，而是补齐之前 RSSM 在线只读单帧的缺口。
模型现在能够从真实观测与动作历史建立 posterior，再对候选技能进行 prior-only
推演。Agent 只接纳实际执行反馈；预测不能完成任务或写入真实历史。

## 1. 参考机制与不做什么

| 来源 | 借鉴内容 | 本项目实现 | 保留边界 |
| --- | --- | --- | --- |
| [DreamerV3 RSSM 源码](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/rssm.py) | posterior/prior 分离，episode reset，离散 latent，分离 KL 梯度 | 独立 PyTorch categorical RSSM、固定历史条件、开放环监督 | 不复制 block GRU，不使用官方权重，不训练 actor–critic |
| [every-embodied](https://github.com/datawhalechina/every-embodied/tree/ccf7013050b9342b2927bd2a4e2b89784f04524f) 的 EVA-Client 工程导航 | 观测、动作块、控制器、日志分层；实际动作与反馈对齐 | 有界历史账本、逐步真实反馈、预测/执行差异记录 | 教程作为工程参考，不冒称已接入其中所有框架或复现实验结果 |

固定源码提交保证可以追溯。研究机制映射不等于方法新颖性证明。
仍采用模块化实验：世界模型独立训练；技能控制器、LLM、记忆更新规则固定。
当前动作空间是 G1 固定底座左肩 pitch / 左肘的两维目标增量；不是全身行走、抓取或堆叠。

## 2. 模块职责与时间轴

```text
数据 episode：真实 RGB / 实测状态 / 已执行动作
     ↓ 同一 episode 内切窗、固定 K=2 历史、独立数据划分
世界模型：CNN+state encoder → posterior(h,z)
     ↓ 候选技能 actions，未来不输入真实帧
prior rollout → 未来帧 / 状态 / ensemble spread
     ↓ 独立 episode 校准误差界，不用熵冒充可信度
Agent：子目标 → 候选技能 → 选择可信短动作段
     ↓ 绑定 episode / step / model_version 的执行回执
固定控制器 → MuJoCo → 每个动作端点真实观测
     ↓ 对齐每一步预测与实际结果，超界触发重新观测
有界真实历史更新 → 下一轮 posterior / 规划
```

| 文件 | 单一职责 |
| --- | --- |
| `datasets/visual_sequences.py` | episode 完整性、划分、归一化、带历史的序列窗口 |
| `models/categorical_rssm.py` | 真实序列 posterior 更新与动作条件 prior 推演 |
| `models/visual_latent.py` | 模型内容身份、检查点、具名语义、在线/离线输入适配 |
| `training/rssm_loss.py` | posterior 表征监督与部署条件匹配的开放环预测监督 |
| `models/horizon_calibration.py` | 独立 episode 校准与模型版本绑定 |
| `agents/observation_history.py` | 固定范围的真实历史与完整执行 trace 校验 |
| `agents/predictive_skill_agent.py` | A0–A3、子目标、动作段选择、真实反馈驱动状态机 |
| `envs/visual_workcell.py` | 固定控制器、MuJoCo 步进、逐动作真实观测、故障锁止 |
| `scripts/visual_skill_agent.py` | 实验装配、事件日志、预测工件与持续观测窗口 |

核心接口：

```python
prediction = model.predict_context(
    rgb=real_rgb,          # [T, 3, S, S] float [0,1]
    state=real_states,     # [T,D] 原始具名坐标
    past_actions=executed_actions,  # [T-1,A] 真正已执行动作
    actions=candidate_actions,     # [H,A] 尚未执行的候选
)
agent.record_feedback(decision, final_observation, observations=real_execution_trace)
```

T 在 1 到 K+1 之间；启动 T=1 是显式短历史。候选 H 不超过训练时域。
K>0 禁止单帧 `predict()` 静默降级。视频 provider 的历史条目除 RGB/state/semantics，
还必须包含 episode_id、连续 step_id 和 action_from_previous；只取最后 K+1 个条目。
Agent 每轮重算 posterior，不保存无限长 latent 或可变检索记忆。新任务建立新历史；
持续窗口内新任务的初始状态可能偏离校准分布，不能声称拥有闭环覆盖保证。

执行两步就必须收到两步真实观测；最终观测不能代替缺失中间帧。
漏步、重复步、跨 episode、结束点不一致在任何状态写入之前拒绝。
控制器或 trace 接收器异常会锁止仿真，必须显式 reset。失败历史标为不可用，禁止自动重试。

## 3. 网络与损失

真实历史 (o,a) → encoder → GRU h → posterior q(z|h,o)。
候选动作 a → GRU h → prior p(z|h) → RGB/state decoder。
离散 z 使用 unimix；训练表征路径使用 straight-through categorical sample，
部署与开放环监督使用可重复的概率代理。代理不是随机轨迹的精确期望。

总损失 = 加权开放环 RGB 误差 + symlog 状态误差 + posterior 重构
+ dynamics KL + representation KL + 有真实标签时的 masked event loss。
KL 分开停止梯度并使用 free nats。历史是 burn-in，不是监督 target；
启动 padding 与第一个真实观测都有 reset，归一化后的填充动作不会带入初始状态。
未来真实帧只计算监督损失，不能输入候选 rollout。
没有 reward、terminal 或 event 标签时，不伪造这些任务输出。

训练、校准、测试都使用相同 K 与启动 reset 规则。K 进入模型内容哈希，
改变 K 后必须重新校准；旧 K=0 deterministic/RSSM 检查点身份保持兼容。
微调要求网络配置和动作/状态/相机/控制周期语义一致。`--pretrained` 是继续学习权重，
不是恢复优化器、随机数状态的精确断点续训。

## 4. 唯一环境与完整命令

以下均在项目根目录运行；每次使用新的输出目录，避免覆盖实验。
不需要 DreamerV3 自己的环境。现有 MuJoCo 数据路径可复用，也可重新采集。

```bash
conda activate wmal
python -m pip install -e '.[visual-world]'
export MUJOCO_GL=egl

# 新采集时执行；已有同名目录请换一个新名字，不覆盖旧数据。
wmal-visual-world collect --output data/processed/g1_temporal_new \
  --task stack_block --episodes 40 --steps 16 --image-size 64 --seed 0

# 下面使用本机已有数据；换数据时保持所有命令中的 manifest 一致。
wmal-visual-world train \
  --manifest data/processed/visual_g1_20260930/manifest.json \
  --config configs/training/rssm_temporal_world.json \
  --output runs/rssm_temporal_train --device cuda

wmal-visual-world train \
  --manifest data/processed/visual_g1_20260930/manifest.json \
  --config configs/training/rssm_temporal_world.json \
  --pretrained runs/rssm_temporal_train/best.pt --freeze-encoder \
  --output runs/rssm_temporal_finetune --device cuda --epochs 10

wmal-visual-world calibrate \
  --manifest data/processed/visual_g1_20260930/manifest.json \
  --checkpoint runs/rssm_temporal_finetune/best.pt \
  --output runs/rssm_temporal_finetune/calibration.json --horizon 4 --alpha .2

wmal-visual-world evaluate \
  --manifest data/processed/visual_g1_20260930/manifest.json \
  --checkpoint runs/rssm_temporal_finetune/best.pt \
  --calibration runs/rssm_temporal_finetune/calibration.json \
  --output runs/rssm_temporal_finetune/evaluation.json --horizon 4
```

开源 LeRobot 路线也可使用此网络，换成已转换的 manifest。必须保留该数据的真实动作维度、
频率与相机定义。16 维真机操作数据不能假装成这里的 2 维关节增量；未做映射和独立验证
之前只进行离线训练/评测，不能直接调用本地 G1 控制器。

已有本轮 smoke 工件的无窗口命令：

```bash
conda activate wmal
MUJOCO_GL=egl python scripts/visual_skill_agent.py \
  --checkpoint runs/rssm_temporal_finetuned_smoke_20261002/best.pt \
  --calibration runs/rssm_temporal_finetuned_smoke_20261002/calibration.json \
  --baseline A3 --horizon 4 --max-cycles 8 --error-budget .5 \
  --goal .4 .9 --output runs/rssm_temporal_a3_new
```

该模型可能拒绝执行；拒绝不是任务成功，也不要为了“看到动”放宽可信度阈值。
A2 可移除 `--calibration` 并设 `--baseline A2`，观察未经可信度过滤的研究对照。
桌面持续窗口可先 `unset MUJOCO_GL`，加 `--viewer --realtime`；目标结束后窗口继续接受指令。
只读客户端可加 `--monitor --monitor-port 8766`，避免占用已有 8765 实验。
预测工件保留逐步 observed_rgb_sequence、observed_state_sequence、observed_step_ids；
反馈事件包含逐步状态/RGB 误差、超界列表及首次超界步号。

## 5. 本轮真实验收与局限

本机已有 40 条 MuJoCo episode，24/4/8/4 独立训练/验证/校准/测试，640 次转移。
K=2、H=4；训练 12 轮 GPU 18.67 秒，冻结编码器微调 2 轮 3.17 秒。
基模 `visual-5bafec56845d80b05454`；微调 `visual-6218b8dd9f7ddb953a8f`。
这是本项目模型从随机初始化训练后的继续学习，不是 Dreamer 或 UniFoLM 预训练权重微调。

4 个独立测试 episode / 52 windows 的 state RMSE（rad）为
0.0342 / 0.0389 / 0.0441 / 0.0481；状态保持基线为
0.0310 / 0.0471 / 0.0608 / 0.0715。第一步未胜过基线。
RGB MSE 约 0.0092，而保持画面基线约 0.00015–0.00043；动态区域同样未整体胜过基线。
测试 episode 同时覆盖率逐时域为 1.0 / 0.75 / 1.0 / 0.75，仅四条轨迹不能证明可靠性。

本轮 A2 闭环执行了 8 个真实动作并建立 [6,7,8] 的末端历史，预算耗尽，目标未完成。
A3 在 error-budget=0.5 下执行 3 步，达到关节目标（终点状态误差范数 0.03372 rad，
任务容差 0.035 rad）。这仅是一种目标、单一种子的一次到达实验，不证明 A3 优于 A2，
也不代表堆叠积木成功；它没有操作物体。现有演示默认模型未替换。
数据/权重/实验输出仍被 Git 排除，克隆仓库不会自动获得本机工件。

## 6. 可发表问题与控制变量

主假设：固定真实历史条件与校准协议时，逐步预测误差反馈能否使可信度机制更早
发现失败，并减少长动作段造成的不可恢复偏差？这是一项待检验假设，不是现成论文结论。
支撑贡献是时序条件一致的可解释世界模型接口与逐动作证据链，不以代码量充当创新。

- K=0 vs K=2：同数据划分、模型容量、优化器、训练轮数、种子与计算平台，单独测历史贡献。
- A0 固定计划；A1 反馈/名义结果；A2 世界模型评估；A3 校准短段规划。
- A3 最终反馈检测 vs 逐步反馈检测：同预测模型、同校准、同技能集，只改变残差检查范围。
- 固定候选技能与历史范围；语言计划器后加，固定其权重/提示与重规划预算。
- 记录任务成功率、失败恢复、拒绝率、执行段长度、逐时域误差/覆盖、规划耗时与编码开销。

episode 校准依赖可交换性；策略选择、环境变化与闭环反馈会产生分布偏移。
误差界不是碰撞安全证书、任务成功概率或真机安全保证。扩展物体交互前需要真实事件标签、
相机/动作映射和接触反馈，不能从两关节位移实验外推抓取能力。

验证入口：`python -m unittest discover -s tests -v`。
需要同时验证本机 ROS overlay 时，在激活 wmal 后先 `source scripts/setup_ros2.bash`；
不要再覆盖 PYTHONPATH，否则会丢失 overlay。新增时序测试包含真实 MuJoCo 执行与故障锁止。
本轮完整 ROS-overlay 回归：206 项通过，无跳过；旧演示模型版本仍为
`visual-fd46b9bdceb6f45ce944`。新增训练/执行诊断工件不写入 Git。
独立代码审查已尝试，但审查代理因服务额度限制未产生审查报告；目前只有作者自查与
自动化/仿真实测证据，不能写成独立审查通过。合并共享分支前建议补做一次独立审查。
