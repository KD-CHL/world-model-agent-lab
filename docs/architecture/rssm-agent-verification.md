# RSSM/Agent 本地验收（2026-10-02）

## 环境与代码验证

项目目录 `/home/chl/GitHub/CodeSpace/world-model-agent-lab-main`，所有 Python
命令使用 `/home/chl/miniconda3/envs/wmal/bin/python`，没有新建其他项目环境。
全量 `MUJOCO_GL=egl PYTHONPATH=src python -m unittest discover -s tests -v`：
181 项，180 通过，1 项 ROS 2 依赖缺失跳过。新测试先出现缺失能力/行为失败，
实现后通过；覆盖 KL 梯度、reset、后验/先验隔离、旧版本兼容、微调冻结、
测试隔离、回执不可篡改、预算和多子目标调度。后续审查修复以文末记录为准。

读取实际旧文件 `runs/visual_g1_release_finetuned_20260930/best.pt`，加载后的
版本仍为 `visual-fd46b9bdceb6f45ce944`，没有失效已有模型/校准绑定。

## 实际数据训练与微调

数据 `data/processed/visual_g1_20260930/manifest.json`，SHA256
`ec8369bff3e882bdb1746579567940d39652fddb89e5f5b8bcc04e1a5ae91f30`。
40 episode，640 transitions；24 train/4 validation/8 calibration/4 test。
训练 H=4，train 312 窗口，validation 52 窗口，3 独立 episode-bootstrap 成员，
每成员 341,767 参数。数据为真实 MuJoCo 渲染与伺服交互，不是手工写未来状态。

```bash
conda activate wmal
export PYTHONPATH=src
python scripts/visual_world.py train \
  --manifest data/processed/visual_g1_20260930/manifest.json \
  --config configs/training/rssm_world.json --output runs/rssm_g1_smoke_20261002 \
  --epochs 12 --device cpu
python scripts/visual_world.py train \
  --manifest data/processed/visual_g1_20260930/manifest.json \
  --config configs/training/rssm_world.json --output runs/rssm_g1_finetuned_smoke_20261002 \
  --epochs 2 --pretrained runs/rssm_g1_smoke_20261002/best.pt --freeze-encoder --device cpu
```

基模 `visual-abddb35a286feaaab49a`，12 轮耗时 77.57s，最佳验证 loss 1.76047。
微调 `visual-bc975493de3546bc9c95`，2 轮耗时 10.50s，最佳验证 loss 1.68683。
实际比较两个 checkpoint 的所有成员 RGB/state encoder 张量，完全相等；
其余权重更新使版本变化。loss 含 free-nats 常数，与旧网络总 loss 不可直接比。
这是同一仿真数据上的微调管线验证，不是新开源数据适应或 Dreamer 权重微调。

## 独立校准与测试

```bash
python scripts/visual_world.py calibrate \
  --manifest data/processed/visual_g1_20260930/manifest.json \
  --checkpoint runs/rssm_g1_finetuned_smoke_20261002/best.pt \
  --output runs/rssm_g1_finetuned_smoke_20261002/calibration.json \
  --horizon 4 --alpha .2 --device cpu
python scripts/visual_world.py evaluate \
  --manifest data/processed/visual_g1_20260930/manifest.json \
  --checkpoint runs/rssm_g1_finetuned_smoke_20261002/best.pt \
  --calibration runs/rssm_g1_finetuned_smoke_20261002/calibration.json \
  --output runs/rssm_g1_finetuned_smoke_20261002/evaluation.json --horizon 4 --device cpu
```

校准仅用 8 独立 calibration episode；q 各步均 12.56616，std floor .05。
4 个 test episode，52 窗口：

| 指标 | 1 步 | 2 步 | 3 步 | 4 步 |
| --- | --- | --- | --- | --- |
| RSSM 状态 RMSE (rad) | .04760 | .04596 | .04956 | .05241 |
| persistence 状态 RMSE | .03105 | .04709 | .06084 | .07147 |
| RSSM 全帧 MSE | .00954 | .00933 | .00928 | .00926 |
| persistence 全帧 MSE | .000155 | .000267 | .000363 | .000435 |
| episode-max 区间覆盖 | .50 | 1.00 | 1.00 | 1.00 |

零动作反事实的状态/图像变化均非零，说明有动作敏感性，但反事实没有真实目标，
不能当准确性指标。当前状态短期并未全面优于 persistence，图像及变化区域误差
也比 persistence 大：完整背景重建尚未充分学好。四个测试回合的覆盖非常粗糙，
不支持校准可靠性的统计结论。最终论文测试需另留从未反复查看的锁定数据；
此处明确是开发 smoke 数据，不能作为最终论文保留测试。

## 实际 MuJoCo 调度

同一短训练权重与 seed，waypoint `[.40,.90]` → final `[.35,.85]`，预算16：

```bash
MUJOCO_GL=egl python scripts/visual_skill_agent.py \
  --checkpoint runs/rssm_g1_finetuned_smoke_20261002/best.pt \
  --calibration runs/rssm_g1_finetuned_smoke_20261002/calibration.json \
  --baseline A3 --waypoints .40 .90 --goal .35 .85 --horizon 4 --max-cycles 16 \
  --error-budget .5 --output runs/rssm_g1_mission_smoke_20261002 --device cpu
```

A3：3 次拒绝后 needs_review，0 步执行，原因 no_trusted_prefix。
其归一化最小误差界 q*.05≈.628，超过 .5 预算，因此拒绝是可解释的，不是窗口故障。
未降低阈值来伪造任务成功。

另跑 A2：同命令去 calibration，baseline A2，输出 `runs/rssm_g1_a2_smoke_20261002`。
4 次规划、实际16步，budget_exhausted，未完成 waypoint；产生预测/实测 NPZ、
entropy、状态误差等完整证据。末段状态误差 RMSE .06501 rad，frame MSE .00930。
A1 以 horizon1 输出 `runs/rssm_g1_a1_smoke_20261002`：实际16步、budget_exhausted，
目标伺服与实测角存在差距，没有满足 .035 的四维完成判据，亦不宣布成功。

这证明训练→推理→候选排序/拒绝→真实执行反馈协议可运行，以及多子目标成功
判据没有被预测或脚本目标代替；**不证明新 RSSM 已完成真实多子目标任务，
不证明优于旧模型，不证明抓取/接触或真机安全**。受控积分环境单元测试完整
通过两阶段调度，实际 G1 效果仍需更多数据、训练与 controller 包络验证。

## 后续实验判断

正式研究优先改进预训练视觉编码、静态背景/动态区域建模、隐藏状态上下文、
控制器实际偏差覆盖；改动训练上下文时必须匹配重做校准。保留当前失败结果，
冻结 LLM/技能/控制器后逐项比较，不通过放宽真实成功阈值制造增益。
本轮本地代码已提交，数据/权重/run 输出保持 git 排除；未执行远程推送。

## 独立审查与修复

独立审查确认观测/想象隔离、实际旧权重兼容、推理重复性、预算与 lineage，
发现两项 Important，均先用回归测试复现，再修复：

1. NumPy 数值只读并不禁止 dtype/shape 改写：现保存私有权威回执，拒绝公开
   arrays 的 metadata/content 改写，反馈只使用私有预测/误差界。
2. event_dim 声明不等于训练监督：现记录每个 bootstrap 成员的有效训练标签
   通道，屏蔽缺监督通道，全无则 None；微调继承祖先记录，并返回通道名字。

未回避审查明确排除的边界：性能优势和真实 waypoint 成功均不成立；跨周期
latent memory、完整 Dreamer 策略、物理抓取和真机安全未实施，不作为已完成能力；
原 monitor 不作无关修改；依用户指令原地开发，未新建工作树。

修复后全量重新验证：183 项，182 通过，1 项 ROS 2 跳过；无失败。
定向新网络/Agent/旧网络测试25项通过。旧实际 checkpoint 版本仍保持一致。
额外复跑真实 MuJoCo A2（max-cycles4）至
`runs/rssm_g1_receipt_smoke_20261002`：新私有回执链实际执行4步并接收对齐
反馈，正常 budget_exhausted；仍未完成 waypoint，不扩大成功声明。
