# 冻结 RSSM 五节点闭环修复与复现

当前功能限固定底座 G1 左肩/左肘的到达、保持与返回，不代表积木抓取或全身控制。
RSSM 检查点权重未修改；低层控制器新增显式版本，控制器与校准在一次任务中冻结。
旧控制器 `legacy` 仍为默认值，旧训练命令和 Agent 保持兼容。

## 本机直接启动

下列检查点及新校准已在本机生成；权重、原始数据与实验工件由 Git 排除。
成功判据仍是实测两关节 L2≤.035 rad、指令 L2≤.015 rad、保持连续3个实际控制周期。

```bash
cd /home/chl/GitHub/CodeSpace/world-model-agent-lab-main
conda activate wmal
export PYTHONPATH="$PWD/src"
unset MUJOCO_GL

python scripts/g1_task_agent.py \
  --config configs/tasks/g1_joint_sequence.json \
  --checkpoint runs/rssm_temporal_finetuned_smoke_20261002/best.pt \
  --calibration runs/rssm_gc2_local_fixedscale_calibration_20261004.json \
  --controller gravity_compensated \
  --baseline A3 --recovery --horizon 4 \
  --viewer --monitor --monitor-port 8766 --realtime \
  --output "runs/task_agent_gc2_live_$(date +%Y%m%d_%H%M%S)"
```

任务结束后窗口/客户端仍保持打开；终端输入 `run` 显式重置并开始新 episode，
输入 `quit` 退出。无桌面时移除 `--viewer --realtime`，设置 `export MUJOCO_GL=egl`；
保留 `--monitor` 可持续查看客户端。端口占用时使用其他空闲端口。
终端返回0仅表示报告生成，成功必须检查 task_*.json 的 state.status。

## 从新数据重新校准（不训练权重）

局部命令范围预先取五节点目标包络并各扩展 .06 rad：肩 [.21,.49]、肘 [.73,1.01]。
数据源记录此范围，校准绑定同一范围；真实历史或任何完整候选命令越界时，A3拒绝使用。
这是一项局部任务实验配置，不代表模型已支持原完整关节范围。

```bash
conda activate wmal
export PYTHONPATH="$PWD/src"
export MUJOCO_GL=egl

python scripts/visual_world.py collect \
  --output data/local/g1_gc2_local_calibration_repro \
  --task stack_block --episodes 40 --steps 16 --image-size 64 --seed 42 \
  --controller gravity_compensated \
  --target-low .21 .73 --target-high .49 1.01

python scripts/visual_world.py calibrate \
  --manifest data/local/g1_gc2_local_calibration_repro/manifest.json \
  --checkpoint runs/rssm_temporal_finetuned_smoke_20261002/best.pt \
  --output runs/rssm_gc2_local_calibration_repro.json \
  --horizon 4 --alpha .2 --std-floor 1 --allow-controller-transfer

python scripts/g1_task_agent.py \
  --config configs/tasks/g1_joint_sequence.json \
  --checkpoint runs/rssm_temporal_finetuned_smoke_20261002/best.pt \
  --calibration runs/rssm_gc2_local_calibration_repro.json \
  --controller gravity_compensated --baseline A3 --recovery --horizon 4 \
  --output "runs/task_agent_gc2_repro_$(date +%Y%m%d_%H%M%S)"
```

采集目录必须是新/空目录。40条 episode 按24/4/8/4划分；此流程只使用8条 calibration
拟合误差界，新增 train/validation 数据未用于训练任何模块。
`--allow-controller-transfer` 只允许显式离线迁移校准，不绕过模型版本、控制器参数、
祖先训练/选型 episode 内容哈希、独立 test 和局部范围检查。

`--std-floor 1` 是固定状态尺度残差校准的配置：在本次集成标准差低于状态尺度时，
分母为训练时固定的 state_scale，误差界主要由独立 episode 的实际残差决定，
不再把小模型的集成分歧当成可靠的逐状态误差缩放系数。
它改变了校准方法，不是调大 error_budget；原 .05 缩放校准的失败记录保留。
这仍不是任务成功概率或碰撞安全保证。较长跨度的预测覆盖率仍需改进。

## 独立预测检查

固定校准后另外采集 seed=43 数据，只使用其 test split评估，不再次拟合校准：

```bash
python scripts/visual_world.py collect \
  --output data/local/g1_gc2_local_holdout_repro \
  --episodes 40 --steps 16 --image-size 64 --seed 43 \
  --controller gravity_compensated \
  --target-low .21 .73 --target-high .49 1.01

python scripts/visual_world.py evaluate \
  --manifest data/local/g1_gc2_local_holdout_repro/manifest.json \
  --checkpoint runs/rssm_temporal_finetuned_smoke_20261002/best.pt \
  --calibration runs/rssm_gc2_local_calibration_repro.json \
  --output runs/rssm_gc2_local_holdout_repro.json \
  --horizon 4 --allow-controller-transfer
```

对照时 A0–A3 应使用相同控制器、任务、候选协议、H、预算和种子；A0必须禁用恢复。
本次功能补验不能将旧控制器失败与新控制器成功作为 A3 贡献，也不声称已验证扰动恢复。
完整记录见 [修复补验](architecture/rssm-five-node-fix-verification.md)。
