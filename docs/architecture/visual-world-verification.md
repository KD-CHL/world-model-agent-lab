# 2026-09-30：视觉网络与技能 Agent 本地验收

本记录区分工程链验收与论文有效性。约 1,800 行新增 Python 实现与测试，另有配置/架构/命令文档；没有用占位类或复制上游源码增加规模。数据与权重只保留本地；本轮源码发布范围包含统一环境配置、实现、测试与文档，不包含大型产物。

## 实际执行

- EGL MuJoCo 采集 40 episode ×16 transition，64×64 真 RGB；24/4/8/4 train/validation/calibration/test。
- 3 个独立 CNN/GRU 成员，各 293,639 参数。GPU 训练 30 epoch：`runs/visual_g1_residual_20260930/`，验证损失 0.00635。
- 从上述权重冻结编码器微调 2 epoch：`runs/visual_g1_finetuned_20260930/`，每成员 235,255 可训练参数；逐张量检查编码器不变。微调不是性能提升结论，其验证损失 0.00683，并未优于父模型。
- 初次验收 model_version：`visual-ea417a5c52df7ccf6460`，逐模型版本重新校准 alpha=.2，8 条独立校准 episode。最新发布前重新生成版本见下方复验记录。
- 原始 LeRobot V2.1/AV1 数据导入 20 episode ×32 frame，四路 12/2/4/2，保留原 test；真实时间戳与视频 PTS 对齐。
- 真实 16D 操作数据 GPU 训练 3 epoch：`runs/lerobot_visual_20260930/`，验证损失 0.03047；仅是离线训练链冒烟，不接 2D MuJoCo controller。

所有路径位于仓库的 `data/processed/`、`runs/`，已用 `git check-ignore` 验证不会提交。早期 v1 全帧重建权重保留作为开发记录；当前 loader 为 v2 残差图像架构，会拒绝旧 schema，不要加载 `runs/visual_g1_20260930/best.pt`。命令指南指向的是最终 v2 微调检查点。

## 预测结果，不选择性汇报

最终紧凑网络的 MuJoCo smoke 测试集 4 episode /52 window：

| 指标 | 第 1 预测步 | 第 4 预测步 |
| --- | --- | --- |
| 学习状态 RMSE，四个角度分量联合 | 0.00684 rad | 0.00914 rad |
| 保持当前状态 RMSE | 0.03105 rad | 0.07147 rad |
| 学习未来 RGB MSE，[0,1]像素 | 0.000156 | 0.000422 |
| 保持当前 RGB MSE | 0.000155 | 0.000435 |
| 每个固定预测步的跨窗口/维度 episode 覆盖 | 0.75 | 1.00 |

第 2 步 episode 覆盖仅 0.50，不能把 alpha=.2 解释为已经达到 80% 在线可信度。第 3/4 步宽区间覆盖 1.00，也不代表有用性优于窄区间，需要联合报告区间宽度和拒绝率。

图像预测只在部分步数小幅优于 persistence，第 1 步反而略差；静态背景会稀释动作相关误差。`test_metrics.json` 额外输出变化区域 MSE、变化区域 persistence、零动作反事实图像/状态敏感性；这些是检测指标，不等价于图像因果贡献或任务成功率。预览 `prediction_preview.png` 上排真实未来、下排预测，也主要反映背景一致性，不能单靠视觉印象证明预测能力。

真实 LeRobot 3-epoch smoke 第 4 步状态 RMSE 0.05674，而 persistence 为 0.04966；RGB MSE 0.004112，而 persistence 为 0.004136。**该数据上的短训模型没有展示稳定优势。** 初始帧前缀也不是完整任务阶段，下一步应增加阶段覆盖、长序列与开源预训练后果预测适配。

上述 smoke 测试在开发中被反复查看，并用于改进图像结构。因此这些数字是开发诊断，不是未触碰的论文最终测试证据。正式实验须重新冻结方案，并保留独立 final-test 数据。

## 实际闭环

同初态/同目标 `.45 .95` 的本地诊断，结果以真实反馈确认：

| 方法 | 结果 | 实际控制步 | 实际目标误差 | 日志目录 |
| --- | --- | --- | --- | --- |
| A0：一次预计算有限计划 | succeeded | 4 | 0.03387 rad | `runs/visual_final_A0_finite_20260930` |
| A1：nominal 反馈修正 | succeeded | 16 | 0.03231 rad | `runs/visual_final_A1_20260930` |
| A2：模型后果排序，固定 4 步段 | succeeded | 20 | 0.03364 rad | `runs/visual_final_A2_20260930` |
| A3：error_budget=1 | succeeded | 20 | 0.03364 rad | `runs/visual_final_A3_20260930` |
| A3：error_budget=.24 机制检查 | succeeded | 4（2+2） | 0.03433 rad | `runs/visual_final_A3_prefix_20260930` |

默认预算下 A2/A3 产生相同执行段，不能声称 A3 增益已被证明。预算=.24 是单个开发目标上的机制检查，不是通过最终测试调出的“最佳参数”。A0 用不依赖反馈修正的有限轨迹消费计划，避免循环重复初始增量制造弱基线；旧 `visual_final_A0_20260930` 是这一开发问题的诊断记录，不能纳入最终方法比较。

可运行的有预算循环、持久 viewer 实现、下一目标输入、过期拒绝、真实反馈对齐、部分失败停止均在代码中；本轮实际运行 EGL headless，**未人工验证桌面 viewer**。也未测试真实机器人、Dex1 抓取或 ROS2 的视觉通路。

## 自动验证

原始验收记录为 114 项、112 通过、2 跳过；该跨环境记录不再作为当前启动依据。统一项目 `wmal` 后重新执行完整 `unittest discover -s tests -v`：114 项，113 通过、1 跳过，无失败。数据转换测试已改用项目依赖 PyAV 生成测试视频，不再要求额外的 OpenCV/另一环境。

代码审查发现并修复祖先训练数据排除丢失和执行失败后空闲窗口继续积分两处问题；另修复 stdin EOF 刷提示。三个新增回归均先确认旧代码失败，再验证修复；最终全套为 **117 项、116 通过、1 跳过**。真实 MuJoCo 故障注入确认取消目标后 `qpos` 和仿真时间均不变，明确 reset 后方可恢复；祖先 episode 改名但内容相同也不能用于后续验证。

唯一跳过项为 `test_live_service_observe_step_reset`：本机未验证与 `wmal` Python ABI 兼容的 ROS2/generated interfaces。

统一环境复验：

- 解释器：`/home/chl/miniconda3/envs/wmal/bin/python`，Python 3.10；运行包路径均位于该环境。
- PyTorch 2.11.0+cu128 / CUDA 12.8，包含 sm_120，在 RTX 5060 Ti 上实际执行 tensor 运算。
- MuJoCo 3.12.0、ONNX Runtime 1.23.2、PyArrow 25.0.1、PyAV 17.1.0。
- GPU 训练 3-epoch smoke 输出 `runs/wmal_env_visual_smoke_20260930/`，验证损失 0.07959；只验证环境链，不替代已有模型性能报告。
- 真实 20-episode AV1 导入输出 `data/processed/wmal_lerobot_validation_20260930/`，同一环境完成解码/对齐。
- 已有检查点在该环境完成 G1 A3 闭环（预算 .24）：实际 2+2 步到达，目标误差 0.03433 rad，记录 `runs/wmal_env_agent_smoke_20260930/`。
- 普通运行的 PYTHONPATH 仅保留项目 `src`，排除 shell 自动加入的系统 ROS Python 3.12 路径；`pip check` 通过。未更改系统 ROS 或其他项目环境。

最终清理路径复验使用 `PYTHONPATH=src`：完整测试仍为 114 项、113 通过、1 跳过；CUDA 训练 3 epoch 输出 `runs/wmal_clean_gpu_validation_20260930/`，验证损失 0.07959；A3 实际闭环输出 `runs/wmal_clean_agent_validation_20260930/`，仍为 2+2 步到达、误差 0.03433 rad。以上产物均不进入 Git。

修复后发布命令使用重新生成的 `runs/visual_g1_release_base_20260930/`（30 epoch，验证损失 0.00634668）与 `runs/visual_g1_release_finetuned_20260930/`（冻结编码器 2 epoch，验证损失 0.00682488）。累计 episode ID/哈希和逐张量编码器冻结均实际核对；最新 model_version 为 `visual-fd46b9bdceb6f45ce944`，已独立重做校准与评估，未复用旧版本的校准文件。预测指标与上方开发 smoke 在所示精度接近，仍不构成论文最终测试证据。该版本 A3 .24 预算闭环实际成功，2+2 步、误差 0.03433 rad，日志 `runs/visual_g1_release_agent_20260930/`。旧产物保留供追踪，不覆盖已有实验。

新增回归覆盖数据对齐/跨 episode 泄漏/内容重复、训练集归一化、测试标签不影响训练、动作影响未来状态与 RGB、无未来观测的多步反传、静态背景残差、检查点回读、冻结微调、episode 校准与版本绑定、A0 不查询模型/不循环重播、真实反馈而非预测确认成功、可信前缀缩短、过期/未知技能/越界拒绝、部分失败不重发，以及真实 MuJoCo 控制和相机变化。

`scripts/check_scaffold.py` 与 `git diff --check` 通过；Python/JSON/XML 检查不能替代 ROS2 或 GUI 验证。

## 数据来源说明与训练进度复验

当前窗口权重的训练源为 `mujoco_g1_arm_servo`，读取 40 条仿真轨迹、640 次转移；基模 30 epoch，冻结编码器实验 2 epoch 使用同一份仿真数据。开源数据并非缺失：本机 MountCamera 原始数据 201 个 parquet episode，约 5.5GB；独立离线模型读取的导入子集为 20 条 ×32 帧、620 次转移。这两种模型分别为 2D 增量接口和 16D 操作接口，不能互换。训练报告记录实际来源，不能用场景里的参考数据集 URL 代替它。

进度条实现后的完整 `wmal` 测试：**124 项、123 通过、1 ROS2 通信测试跳过**。新增 7 项验证 stderr 进度/纯 JSON stdout、关闭显示的模型版本不变、微调使用相同进度、批次失败不伪报 epoch 完成、缺少数据不产出权重、旧字符串来源清单继续可训练，以及导航网络 resume 与连续训练一致。代码审查指出的来源类型兼容问题已用实际训练先复现再修复，不收紧原有数据 schema。非 TTY 输出不含 cursor-up ANSI，窄终端保留完整每轮损失摘要。

真实开源子集 CUDA 进度冒烟：`runs/lerobot_progress_smoke_20260930/` 训练 2 epoch，验证损失 0.033017；从该权重冻结编码器训练 1 epoch 输出 `runs/lerobot_progress_finetune_20260930/`，验证损失 0.0292946。逐张量确认编码器未变、父模型/数据记录保留。这些短训仅验证进度与训练链，不作为算法增益证据，也不连接到两关节控制器。旧模型/数据未覆盖，新产物仍被 Git 排除。

`pip check`、脚手架检查、`git diff --check` 通过；本轮只在 `wmal` 更新项目安装元数据，未替换 PyTorch/CUDA 或增加其他环境。

## 论文前仍需补齐

1. 独立 final-test 与多个训练 seed；不能用本记录单目标判断算法优势。
2. 更丰富且物理可执行的多步骤技能、真实任务状态/接触成功检测；当前是关节到达载体。
3. 状态-only/RGB+状态、预训练/随机初始化、spread/校准、固定/自适应前缀、无/有反馈的配对消融。
4. 覆盖率不足的分析：场景、动作分布、episode 数与归一化残差是否适用；不把误差界当作安全证书。
5. 单独实现并验证 UniFoLM simulation-mode 后果预测适配，确认动作语义和显存可行性；没有下载大权重或宣称已有大模型微调成果。

完整设计：[视觉网络与 Agent 架构](visual-world-agent.md)；完整命令：[训练与闭环入口](../23_visual_world_training_and_agent.md)。
