# 动作条件视觉世界模型与可信度辅助技能 Agent

## 研究定位与边界

本轮把可训练的视觉后果预测加入现有工程，而非继续把底座状态回归器称作视频世界模型。新增网络从真实 RGB、结构化状态、候选动作段学习未来 RGB 和任务状态；高层 Agent、技能生成规则和 MuJoCo 伺服器均不训练。现有行走状态网络、LLM Agent、ROS2 会话不被替换。

当前验证载体是固定底座 G1 左肩/左肘到达目标，**不是抓取、积木堆叠或完整人形操作策略**。十二种 workcell 可提供场景，但更换训练场景需要独立训练/校准，不能默认跨场景可信。训练 RGB 是实际渲染的机器人关节运动，不是模拟未来帧作为在线规划 oracle。

## 开源学习与独立实现

2026-09-30 核查，采用以下代码机制作为设计依据，不复制第三方实现，不混用权重格式。

| 开源项目/固定源码 | 学习机制 | 本项目实现与区别 |
| --- | --- | --- |
| [UniFoLM-WMA 配置](https://github.com/unitreerobotics/unifolm-world-model-action/blob/3e198de68de55f93f24b3ad623dd499390aaee45/configs/train/config.yaml)、[训练模型](https://github.com/unitreerobotics/unifolm-world-model-action/blob/3e198de68de55f93f24b3ad623dd499390aaee45/src/unifolm_wma/models/ddpms.py) | 图像潜在表示、动作条件、冻结与可训练模块分工 | 保留输入控制语义；本轮是紧凑 CNN/GRU 集成，不是视频扩散复现；上游权重需要独立外部适配器 |
| [Dreamer RSSM](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/rssm.py) | 观测更新与无真实观测的想象分离 | 每次由实际 RGB/状态初始化隐变量，然后仅以候选动作推进；未实现离散随机 RSSM、KL 或 imagined actor-critic |
| [TD-MPC2 世界模型](https://github.com/nicklashansen/tdmpc2/blob/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py) | 潜在动作条件转移与规划接口 | 多步潜在一致性与候选动作后果预测；不训练 Q 函数或策略，不宣称复现 TD-MPC2 |

上游 WMA 当前训练模板默认 `decision_making_only: True`，不能因使用 WMA 数据就认定正在训练预测模式。本项目现有 WMA launcher 与本轮紧凑视觉网络是两条独立训练路径。

## 模块与信息流

```text
真实 RGB + 实测关节 + 当前伺服目标
               │
               ▼
任务状态机 → 固定技能生成器 → 结构化候选动作 [H,A]
                              │
                              ▼
                    独立视觉世界模型集成
                    CNN 编码 → GRU 动作条件转移
                    ├─ 未来 RGB 残差解码
                    ├─ 未来任务状态
                    └─ 可选事件头（需真实标签/显式 mask）
                              │
                              ▼
              独立校准集 → 冻结的逐步误差界
                              │
                              ▼
               Agent 候选排序 / 可信执行前缀
                              │
                              ▼
               独立控制器校验整段限位 → 执行
                              │
                              ▼
                 真实反馈 → 预测/执行对齐日志
```

实现模块：

- `datasets/visual_sequences.py`：四路 episode 划分、内容去重、哈希校验、跨 episode 泄漏检测、有限缓存、训练集流式归一化。
- `datasets/visual_lerobot.py`：V2.1 parquet 与 AV1 视频按时间戳对齐；原始 train/test 不改动；原 validation 按 episode 分成选型集与校准集。
- `models/visual_latent.py`：独立成员 CNN、GRU、潜在转移、未来帧残差解码、状态增量头和可选事件头；严格命名动作/相机/采样周期契约。
- `training/visual_trainer.py`：episode bootstrap、开环多步损失、梯度裁剪、最佳验证集检查点、显式冻结编码器微调、独立测试和 persistence 对照。
- `models/horizon_calibration.py`：逐步的 episode-max 归一化残差校准；模型版本、动作顺序、相机、控制周期绑定。
- `agents/predictive_skill_agent.py`：目标、已完成子目标、可调用技能、候选技能、反馈、失败原因、执行/重规划预算；A0–A3；过期与重复反馈拒绝；部分执行失败只计真实步数。
- `envs/visual_workcell.py`：真实 MuJoCo 图像/状态/动作接口；显式两关节增量控制；整段限位检查；不会将 Dex1 16D 操作数据直接映射为本机 G1 指令。
- `scripts/visual_skill_agent.py`：有预算的闭环、JSONL 证据、预测图像 NPZ，完成任务后保持 viewer 并接受下一目标。

## 训练网络

令当前 RGB 为 `I0`，状态为 `s0`，候选动作序列为 `a[0:H]`。每个集成成员独立参数：

```text
z0 = CNN(I0)
h0 = initial(z0, normalized(s0))
h(k+1) = GRU([z(k), normalized(s(k)), normalized(a(k))], h(k))
z(k+1) = z(k) + 0.1 * latent(h(k+1))
s(k+1) = s(k) + state_head(h(k+1), z(k+1))
I(k+1) = clip(I0 + residual_decoder(h(k+1), z(k+1)), 0, 1)
```

图像残差始终相对**真实初始帧**，不输入未来目标帧；残差范围足以表示完整像素变化。初始零残差保留背景，避免在静态房间里用有限样本反复学习背景重建。该设计仍可能退化为复制当前帧，必须报告 persistence、变化区域误差和动作反事实敏感性，不因图像“看起来合理”宣布预测有效。

训练损失：加权未来像素 MSE + 归一化多步状态 MSE + stop-gradient 未来编码的一致性损失 + 有观测 mask 的事件 BCE。变化区域仅作为训练目标加权，在线 rollout 不读取这些区域。数据未提供事件监督时 event_dim=0，不能从未训练的事件头输出任务成功/碰撞概率。

默认每个成员独立 bootstrap 整条训练 episode；所有成员独立保留 autoregressive 状态，不能把均值重新当作每个成员下一步输入。训练程序选择最佳检查点仅用 validation；测试集和 calibration 不参与梯度、归一化或自动模型选型。开发 smoke 指标被反复查看，不可充当论文最终保留测试集。

微调 `--pretrained ... --freeze-encoder` 保持已有检查点归一化和编码器坐标不变，只训练其余模块；这是本项目网络的微调，**不是 UniFoLM 大模型微调**。不支持改变动作语义后直接继承检查点；也不把微调当作 optimizer/RNG 续训。

检查点保存累计 episode lineage：历代训练和选型 episode 的 ID 与内容哈希不会在微调时被当前 manifest 替换。校准/测试排除所有已暴露 episode；后续验证集还需排除祖先训练集，改名相同内容也会拒绝。旧基模只有 ID 时保留其 ID 排除，但不能补造未知的内容哈希；旧微调权重缺少完整祖先记录时，离线校准/评估及继续微调会拒绝，需从有记录的基模重新生成。

仿真执行异常会取消尚未完成的伺服目标并锁定会话；空闲窗口仅显示冻结状态，不能继续物理积分或接受下一动作。终端输入 `reset` 明确重置场景后才能开始新目标；不自动恢复或重发失败段。输入流 EOF 只停止终端轮询，窗口仍保留。

## 可信度与 Agent 方法

对每个预测步 k，将标准化误差按同 episode 的所有窗口和状态维取最大：

```text
score_episode(k) = max_window,dimension |actual - ensemble_mean| /
                  max(ensemble_std, std_floor * training_state_scale)
rank = ceil((N_calibration + 1) * (1 - alpha))
q(k) = episode_scores(k) 的 rank 阶统计量
bound(k,dimension) = q(k) * max(std(k,dimension), std_floor * scale(dimension))
```

样本量不足以取得有限 rank 时直接拒绝；q 用非递减上包络，避免长步误差界系数反而缩小。A3 使用 `max_dimension(bound/scale) <= error_budget` 与状态约束联合决定最大连续可信前缀，并按前缀终端目标距离及区间宽度排序候选。它校准的是任务状态误差，不是“动作成功概率”。

| 基线 | 决策信息 | 实际执行 |
| --- | --- | --- |
| A0 | 一次生成有限 nominal 计划，按进度消耗，不查询世界模型 | 固定长度；反馈仅记录和确认成功；越界会停止 |
| A1 | 当前反馈 + 控制器显式 nominal terminal | 固定长度；不查询世界模型 |
| A2 | 学习模型候选后果 | 固定长度，无校准前缀 |
| A3 | A2 + 独立校准界 + 实际残差反馈 | 自适应短前缀；残差越界先重新观测；无可信前缀拒绝执行 |

预测到达目标不代表成功，只有真实反馈能推进 `TaskStage` 或完成任务。Agent/世界模型/机器人交换 episode、step、decision_id、model_version；机器人独立检查整个 committed prefix。异常执行采用停止而非重发，防止重复执行。

统计边界：区间在同分布、可交换 episode 条件下解释为每个固定预测步的误差覆盖；**不提供跨所有预测步同时的覆盖保证，不提供自适应选择候选后的条件覆盖，也不提供碰撞或真机安全证明**。更换场景、相机或控制策略会改变分布。当前 smoke 校准 episode 少，覆盖率必须据实展示，不能把 1-alpha 当作实际在线成功率。

## 可以形成论文的问题，而非现成创新结论

推荐研究假设：**任务状态误差预算能否统一控制固定技能 Agent 的候选选择、执行长度和失败后重新观测？** 支撑贡献为动作条件视觉/状态多步学习和可重复的预测—执行证据链。

不确定性 MPC、共形区间与扩散规划已有相关研究，例如 [NeurIPS 2023 共形扩散动力学规划](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe318a2b6c699808019a456b706cd845-Abstract-Conference.html) 和 [动态环境共形安全规划](https://arxiv.org/abs/2210.10254)。因此“给世界模型加不确定性”本身不能作为首创。需进一步证明任务相关误差空间、技能长度机制和闭环失败响应相对这些方法的差异。

推荐严格消融：同一冻结模型、同一 controller、同一候选预算，比较 A0–A3；A2 加未校准 ensemble spread；A3 固定前缀；A3 无实际残差反馈；状态-only 与 RGB+状态；随机初始化与预训练微调。以不少于 5 个训练 seed、配对任务初态、独立最终测试集报告任务成功率、失败恢复、超预算率、预测误差/区间宽度/覆盖、实际执行长度、拒绝率、规划时延。不要同时训练 LLM、VLA、记忆、人形控制器与世界模型。

当前到达实验只证明工程链可执行；要走向主论文任务，仍需带真实任务状态/标签的可执行多步骤技能，如 reach/grasp/place，并验证夹爪控制资产和接触成功检测。不能用 workcell 的视觉标记自动消失代替物理任务成功。

完整命令与本轮证据见 [视觉世界模型运行指南](../23_visual_world_training_and_agent.md)。
