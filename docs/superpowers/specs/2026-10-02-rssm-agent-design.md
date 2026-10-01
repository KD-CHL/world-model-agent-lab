# 随机状态世界模型与可审计技能 Agent

## 意图与范围

目标是把动作后果预测升级成可训练的随机潜在动力学，并让固定技能 Agent
组织多子目标、选择可信短段、接收真实反馈。主创新仍是预测可信度辅助规划，
不是声称 RSSM、KL、集成或状态机本身原创。保留 A0–A3 和旧确定性网络，
不同时训练语言模型、VLA、记忆、G1 控制器与世界模型。

用户已要求在当前目录自主完成阶段，无需逐阶段同意。因此在当前 main 工作，
先保存规格/计划后直接实施；不据此推断本轮远程推送授权。

## 源码依据与方案选择

2026-10-02 读取固定版本：

| 源码 | 学习依据 | 采用/不采用 |
| --- | --- | --- |
| [Dreamer RSSM](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/rssm.py#L56) | observe/imagine 分离、离散随机状态、reset、unimix、双 stop-gradient KL、free nats | 独立 PyTorch 紧凑实现；不是 JAX 权重转换或完整 Dreamer 复现 |
| [Dreamer Agent loss](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py#L155) | 重构、奖励、continuation 与想象策略的分工 | 当前只训练观测/动力学及有 mask 的事件；无奖励/终止标签就不制造监督，也不训练 actor–critic |
| [Every-Embodied MuJoCo 示例](https://github.com/datawhalechina/every-embodied/blob/ccf7013050b9342b2927bd2a4e2b89784f04524f/examples/01_hello_every_embodied_mujoco.py) | IK、轨迹、技能阶段与交互运行分离；示例通过运动学 attach 搬方块 | 学习工程分层，不复制其机器人或脚本抓取成功判据；本项目不以 attach/瞬移构造成功 |
| [Every-Embodied 世界模型课程](https://github.com/datawhalechina/every-embodied/tree/ccf7013050b9342b2927bd2a4e2b89784f04524f/17-%E5%85%B7%E8%BA%AB%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B) | 动作条件预测与策略生成分工、数据对齐、演示与闭环验证区别 | 用作学习索引，不当成可直接加载的统一骨干或权重 |

Dreamer 所读仓库 MIT；Every-Embodied CC BY 4.0。代码独立编写，引用保留；
没有复制上游实现文件、安装上游环境或下载基础模型权重。

三条路线：推荐紧凑 RSSM + 固定技能 + 独立校准，便于变量隔离；完整 Dreamer
想象 actor–critic 会改变执行策略，留作未来独立实验；视频扩散微调可作为重型
预训练支线，但不能自动从视频得到关节安全约束。没有唯一“世界模型规范”，
这里落实动作条件、观测/想象隔离、随机状态、真实多步监督与可检验误差。

## 网络与训练

新增 `architecture=categorical_rssm`，旧默认 `deterministic` 不变。
RGB 编码 + symlog 标准化实测状态编码得到 token；确定性记忆 h 和离散 z
构成潜在状态。动作先推进 h，再生成 p(z_next|h_next)；真实下一观测只进入
q(z_next|h_next,token_next)。训练用 straight-through categorical sample，
验证/规划采用类别概率软状态的确定性代理，避免候选排序与校准被随机抽样噪声改变。
这个代理不是非线性随机 rollout 的精确数学期望，必须标注。

观测序列支路监督 RGB/状态重构和 balanced KL；开环支路仅从真实初始观测
及候选动作生成未来 RGB/状态，监督 H 步误差。像素动态区域仍按真实目标加权，
目标图像绝不送入开环支路。KL 分别 detach posterior/prior，sum 类别和随机变量，
free nats 默认 1，dyn 权重 1，rep 权重 .1；unimix .01。状态头在 symlog
坐标学习并用 symexp 还原，图像直接重构，不宣称物理一致性已保证。

原窗口只包含完整同 episode 序列，不 padding；observe 的显式 is_first 可
清除历史和入射动作。网络提供 recurrent observe API；本轮部署和校准一致地
从单个当前真实观测初始化，再在候选段内保留 h,z。跨真实周期的长期 latent
记忆不启用，以免改变校准条件；这是明确的后续消融，不把日志记忆冒充 latent belief。

沿用 train/validation/calibration/test 四路 episode 划分、内容哈希、祖先暴露
记录、episode bootstrap、梯度裁剪、仅 validation 选型、进度条、best/last 和
微调冻结。旧检查点版本哈希必须保持一致；RSSM 不加载确定性/上游权重。
事件缺标签不输出概率；奖励/continuation 需未来 schema 提供真实标签/mask 后单独加入。

## Agent 与通信

现有最小状态继续保留，新增 active_subgoal、决策次数、重观测预算和
belief summary（真实 episode/step/state/model、预测残差、区间越界与网络诊断）。
`active_stage` 是技能生成器唯一当前目标；命令支持显式顺序 waypoint 任务。
子目标完成只依赖真实反馈，预测成功不能推动状态机。

`max_reobservations=3` 为连续拒绝/重观测上限，达到上限进入 needs_review；
实际完整执行反馈清零该计数。动作预算耗尽保持 budget_exhausted。
观察/候选/任务目标/决策数组防御性复制，决策回执必须对应当前 pending
对象，不能凭复制 decision_id 篡改预测、prefix 或动作。部分执行异常停止，不重发。

RSSM 诊断（prior/posterior entropy）与 ensemble spread 分开记录，只有绑定
模型版本的校准界控制 A3。熵不是可信成功概率，碰撞/接触判据仍属于真实环境。
模型更新改变版本必须重新校准；只读客户端自动显示新增 agent_state/event 信息。

## 验收与研究方案

测试 posterior/prior 区分、reset 隔离、KL 梯度方向、概率下界、未来帧不泄漏、
动作敏感、checkpoint/旧版本兼容、实际数据训练微调与四路隔离、连续重观测
终止、多阶段真实完成、回执不可篡改。真实 MuJoCo 数据短训练只做工程验收，
不得描述为已学会抓取或证明论文增益。

论文：同数据/固定控制器对比确定性 GRU 与 RSSM、去 KL、去开环监督；
同冻结模型比较 A0–A3、未校准 spread、固定前缀与无残差反馈。记录多步状态/
视觉误差、变化区域/persistence、区间覆盖/宽度、动作敏感性、子目标成功、
恢复/拒绝、规划时延，至少多种训练 seed 和配对任务；验证集选配置，最终测试冻结。
