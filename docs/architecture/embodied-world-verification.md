# 可规模化具身世界模型验收记录

日期2026-10-11；环境仅`wmal`，Torch2.11.0+cu128、CUDA12.8、本机RTX5060Ti，总显存8,073,248,768 bytes。
这是工程/小样本验证，不是世界模型规划收益或8×3090训练能力认证。
源码学习机制与固定版本见[规格](../superpowers/specs/2026-10-04-scalable-embodied-world-model-design.md)，完整命令见[使用指南](../30_scalable_embodied_world.md)。

## 实现范围

- 新模型族和旧Visual/RSSM并存；多视角共享编码、状态/动作融合、分组categorical RSSM、物理残差/RGB/仅真实监督事件分支。
- 原始物理状态与确定性指令分离；真实K历史后验与动作-only先验，不读未来观测。采样局部Generator、候选公共随机数、M与S分开。
- 四split来源hash/原始ID/派生别名隔离；物理/指令/动作单位、相机、周期、控制器显式profile；只读旧数据适配。
- 独立成员bootstrap训练、train-only归一化、validation物理指标选型、冻结编码器微调及祖先暴露继承；独立calibration与最后test。
- 优化器边界续训，模型/Adam/RNG/采样器/剩余成员状态严格恢复；完整forward-owned DDP/FSDP包装、尾累积实际批数、唯一验证窗口。
- 版本化不可变request/response、本地可终止worker、仅预测HTTP、deadline/重置/重复响应隔离；派发前观测内容和仿真时钟校验，实际逐动作反馈。
- 独立测试物理RMSE、train-fit线性/保持基线、动态区域RGB、动作敏感性、原始episode联合覆盖与Wilson区间；新报告不覆盖已有证据。

## 参数与实测资源

静态meta计数：默认两物理/两指令/两动作、单相机128px、无事件。

| 配置 | 每成员唯一参数 | 状态 |
| --- | ---: | --- |
| WM-S | 40,174,005 | 容量范围计数通过，未正式训练 |
| WM-B | 151,501,701 | 容量范围计数通过，未正式训练 |
| WM-L | 298,675,541 | 容量范围计数通过，未正式训练 |
| B三独立成员 | 454,505,103 | 顺序训练设计，未正式训练 |
| 本机smoke | 403,733 | 32px，K2/H4，实际训练单成员 |

正式标签有静态参数预算门槛；小网络不能通过profile或evaluate冒充WM-B。
本机最终源代码完整更新资源探测：T32、microbatch2、accumulation4、BF16、M3/S8；更新3.669618s、2.180064序列/s；含provider验证/hash的推理0.358697s。
allocated峰值90,960,384、reserved92,274,688 bytes，约总显存1.127%/1.143%。
profile中三个成员是独立参数对象，但只有第一成员做一次更新，其余随机探测；不是三个已训练模型。
八卡NCCL吞吐、多卡FSDP、150M/300M正式模型训练、真机和公网TLS**均未实测**，不能由上述小配置线性推算。

## 数据与训练冻结顺序

新采集`data/local/embodied_gc2_smoke_20261011/manifest.json`：seed1171、40原始episodes、每条32个实际MuJoCo转移，共1280转移；train/val/cal/test=24/4/8/4。
SHA256 `cdbbdf76d7dea97494a835cc3849fa043db311ff6c4581f9e5df3575bf8d4f9f`。
gravity_v2控制器hash `2157635392f1feb42df248076aeed8a5c6621cc6513c47c2a9d1c46fd7af415d`；周期.2s；指令肩[.21,.49]/肘[.73,1.01]，实测margin.15、maxdelta.06。
固定底座两关节实验，不是抓取、堆叠或全身任务。

实际训练`runs/task4_local_gpu_smoke_recovered_20261011/`：M1/T32/H4/K2/S8，1epoch/6更新/48训练序列曝光；116唯一validation窗，normalized选型指标.722408902。
推理权重`model.pt`版本 `embodied-26182f249a2b3e62eebfdeab3cffb6f0fa03fe04ce066916cf6d85dd51854c50`，文件SHA256 `ac296b15dde948b110fe95150fd2eed6313a3ecc23a9f1f8ee1fcf6571251c07`。
先冻结这份选型权重，随后8cal episode拟合，最后4test episode的116窗评分；未根据test修改模型、权重、控制器或Agent预算。
第一次评估因“打乱动作后中间指令越出模型范围”失败；增加适用范围计数后重跑同一冻结权重，不裁剪、不更换真实测试轨迹。这是评估器修复，不是test驱动训练选型。
校准`runs/task6_embodied_calibration_20261011.json`，测试报告`runs/task6_embodied_evaluation_20261011.json`，均留在本机被Git排除。

## 独立测试结果：保留负结果

| 指标 | 新单成员smoke | 保持基线 | train-fit线性动力学 |
| --- | ---: | ---: | ---: |
| 物理关节总RMSE(rad) | .056630 | .054900 | .009267 |
| H1物理RMSE(rad) | .039015 | .033256 | .009490 |
| H4物理RMSE(rad) | .067042 | .067351 | .009134 |
| 全帧RGB MSE([0,1]) | .00344813 | .0000793726 | 不生成RGB |
| 动态区域RGB MSE | .00905570 | .00699406 | 不生成RGB |

简单线性基线仅拟合本manifest的768个train转移，未使用validation/cal/test；全帧动态像素约1.1325%。
已知目标指令不计入学习预测收益；事件标签没有证据，报告事件为null。
新模型整体未优于保持基线，更未优于线性动力学；一epoch只证明训练/导出/加载/评估链路可运行。

固定尺度校准alpha.2，quantile2.628655538；半宽肩.202497/肘.178768rad。
test原始episode联合覆盖3/4=.75，Wilson95%区间[.300642,.954413]；窗口覆盖114/116=.982759不能当116个独立试验。
cal8/test4不足正式样本要求，不能认证80%覆盖、安全概率或规划增益。
零动作对照116窗，平均物理预测变化.00381784rad；打乱动作109可用/7越界，变化.00260704rad。
这些只证明预测响应动作改变，没有真实反事实分支，不能解释为因果准确率。

## 机器人闭环与兼容性

实际新模型A3：`runs/task6_embodied_agent_20261011/`，原五节点任务/预算保持不变。
结果`needs_review / no_trusted_prefix`，完成0子目标、派发0动作。固定尺度quantile2.628655538超过归一化预算1.0，所有前缀被拒绝；没有放宽预算或隐式fallback。
这是预期的安全拒绝行为，也是当前模型不能完成该任务的负结果，不能称为仿真成功。
测试中的中性残差模型只用于证明真实MuJoCo/协议/执行回执衔接，不是学习能力结果。

旧冻结时序RSSM在Task5接入改动后重新实际运行：`runs/task5_legacy_frozen_regression_20261011/task_000.json`，五节点成功/17实际动作。
旧权重 `runs/rssm_temporal_finetuned_smoke_20261002/best.pt` SHA仍为 `ceb52e163de9e2b46dd253fc87a3f2f4d1cffcb81d224e580e4554dc37df5b02`，未改权重、成功条件或预算；该结果不证明新模型优于旧模型。
实际worker卡死/重置和真实loopback HTTP测试通过；实际MuJoCo推理超时测试派发0动作且fault_latched。
任务结束后保留窗口与run/quit资源清理由代码/测试覆盖；本次没有人工观察桌面viewer，所以不声称窗口视觉验收通过。
用户已有8765监控未重启、未占用；验收只使用自有子进程、无窗口MuJoCo和临时端口，不发送真机动作。

## 恢复、回归与审查

真实CUDA BF16三次独立进程的连续/中断/恢复：100个模型tensors、Adam、各rank RNG、采样器/cursor和下次随机draw精确一致。
此前native adaptive-pooling backward确定性失败保留为历史负结果；训练loss包装采用参数无关的固定顺序反向，推理结构/权重格式不变，没有warn-only降级。
双rank CPU/gloo三次真实torchrun进程验证恢复、唯一验证窗及尾批4/4/2；不能替代多卡CUDA实测。
新评估器6项初始RED → 后续9项GREEN；另有打乱动作越界和虚假容量标签的RED→GREEN。
完整回归使用wmal+ROS overlay+EGL：Task5完整396项通过，Task6首次402项通过，最终405项/73.657s通过，无跳过。compileall(src/scripts/tests)与git diff --check退出0；整分支审查在交付前更新。
已有ROS负向测试会打印预期错误日志，不代表测试失败；这项日志整洁性留待后续。
Tasks1–4各有独立任务审查；Task5子代理额度不足后改主代理直接实现，最终独立审查若仍不可用会明确披露，不能把自审写成独立批准。

2026-10-12最终补验：独立整分支审查已读取实现并检查跨层边界，但在形成最终结论前额度中断；原审查代理恢复仍不可用，已关闭。最终为主代理自审，不是独立批准。
一次修复波次关闭四项已知Important问题：保留原预测deadline至授权/每动作派发并复核实测仿真时间；HTTP请求行/headers/空闲与慢滴连接采用绝对时限；所有已预测但被拒绝的候选保留原始预测、界限与model/protocol/scope证据；本地会话2048请求后明确要求reset而不淘汰旧ID。
新增七项回归覆盖上述边界：原派发测试确实在过期/时钟漂移后执行1动作、旧长会话未拒绝、旧HTTP空闲连接阻塞、旧拒绝证据缺失、旧剩余前缀执行2动作，修复后均按断言拒绝/停止。
服务器慢滴的额外测试以去掉timer启动的内存mutation复现阻塞（最初mutation的super闭包错误修正后才算有效证据），没有改源文件或把错误RED当作成立。
最终23项服务/Agent针对性测试通过；完整wmal+ROS overlay+EGL **412项/77.687s通过，无跳过**；compileall、diffcheck退出0。
最终真实MuJoCo复验：`runs/final_legacy_regression_20261012/`仍五节点成功/17动作；`runs/final_embodied_regression_20261012/`仍no_trusted_prefix/0动作，但被拒候选现在保留实际预测与界限，且绑定模型/协议hash。两份冻结权重SHA与上文一致。
一次自审没有独立审查的强度，后续合并/论文实验仍应追加独立审查；本次按用户明确授权仅推送功能分支，不自动合并main。

下一步应先扩大train数据和训练轮次，冻结变体后以新的独立cal/test比较；再做固定其余条件的A0–A3/R0–R1、容量、动态区域损失和样本/成员机制消融。不得继续利用本次已经打开的test来挑选模型。
