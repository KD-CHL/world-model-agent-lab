# 时序世界模型与研究 Agent：设计决策

## 目标与边界

研究主线保持为“预测可信度辅助机器人 Agent 规划”。世界模型学习动作后果，
Agent 组织子目标和候选技能，固定 MuJoCo 控制器执行；不联合训练 LLM、VLA、
行走策略或记忆模块。此次补齐训练与部署之间的时序条件一致性，而非扩张模型数量。
所有命令使用现有 `wmal` 环境。开发在当前项目目录，不另建环境或搬迁工作树。

## 源码依据

- DreamerV3 固定提交 `e3f02248693a79dc8b0ebd62c93683888ddaccfe` 的
  [RSSM](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/rssm.py)：
  真实观测更新 posterior，imagine 只使用 prior，episode reset 清空递归状态，
  dynamics / representation KL 分开停止梯度。独立 PyTorch 实现，不复制 JAX 框架，
  不声称复现完整 DreamerV3 actor–critic。
- [every-embodied](https://github.com/datawhalechina/every-embodied/tree/ccf7013050b9342b2927bd2a4e2b89784f04524f)
  的 EVA-Client 工程导航启发观测、动作块、执行与评测分离，以及动作和反馈时间轴对齐。
  它是教程资源，不把教程里的第三方模型效果作为本项目的实验结论。

## 选择

三种方案：继续单帧初始化；长期缓存递归 latent；重算固定真实历史。
采用第三种：最多 `context_steps=2` 个过去动作及三个真实观测，确定性 posterior
概率代理重建 belief。比长期 latent 缓存更容易审计、重放、固定消融条件；代价是重复编码。
保留 `context_steps=0` 和已有 deterministic 模型的旧行为及版本哈希。

## 契约

`VisualWorldModel.predict_context(rgb[T,3,S,S], states[T,D],
past_actions[T-1,A], future_actions[H,A])`：`1 <= T <= K+1`，`1 <= H <= trained_horizon`。
输入仅是真实历史，未来输入只有动作；输出是独立 ensemble 的未来帧、状态、诊断。
候选评估无副作用。K>0 模型的单帧入口明确拒绝静默降级；启动时可显式传入 T=1。
K=0 保持旧接口。K 进入模型内容哈希，旧校准不能绑定不同条件协议。

训练 windows 保持每个目标 offset：左侧附加最多 K 个同 episode 的真实 transitions。
不足的历史左填充首观测和零动作，`is_first` 在每个填充与首个真实观测处 reset。
padding 和 burn-in 不参与 reconstruction / KL 监督。开放环预测从真实 context 的
末端 posterior 开始，不能读取未来 targets；标签只用于计算损失。前缀动作必须按
归一化后的“零 incoming action”处理，不让非零 action mean 改变 episode 初始状态。
校准和测试读取同一 K，去掉填充，覆盖启动短历史与完整历史；按 episode 聚合。

Agent 保存有界真实历史账本：episode / step 必须连续，重复候选评估不追加历史。
执行多个动作时必须返回逐步真实观测；漏步、跨 episode、伪造结束点均拒绝，不把
最终观测复制成中间反馈。验证失败不更新 Agent 状态，仍保留待处理执行回执。
最终观测更新任务状态，只有实际反馈判定子目标完成。执行失败仍锁止，不自动重试。

## 工程验收

- 模型与数据契约、Agent 账本分离到聚焦的小模块，配置和文档声明输入语义。
- 测试覆盖短历史、padding reset、未来泄漏、候选分支隔离、缺失 trace、校准版本、旧模型兼容。
- 新增真实数据训练、微调、校准、独立测试 smoke；不把 smoke 当性能证明。
- 完整现有测试回归、独立代码审查。新增时序配置不替换现有可运行演示的默认模型。
- 不新增无标签 reward / continuation / event 输出；不把熵或 ensemble spread 叫可靠性。
- 本次不推送、不进行真机动作。训练有效性仍需同数据划分的 K=0/K=2 与 A0–A3 消融。
