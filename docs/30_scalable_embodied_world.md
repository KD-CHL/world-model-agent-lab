# 可规模化世界模型、独立评估与持续 Agent 会话

## 研究边界与模块

主问题是：当后果预测有误差时，Agent能否用冻结、独立校准的误差界选择执行长度并及时重新规划。
世界模型只预测；固定技能/控制器只执行；任务调度器依据真实反馈判断子目标。语言层、VLA、长期记忆和控制器不与本次世界模型同时训练。
`stack_block` 是场景名：当前在线研究接口仍是固定底座 G1 的左肩 pitch/左肘两关节，到达、保持、返回，不是积木抓取、全身行走或接触任务。

```text
真实 RGB / 实测关节 / 当前指令 / 连续历史
    → 编码与融合 → categorical RSSM 观测后验
    → 每个候选技能的动作条件先验展开
    → 未知物理后果、RGB、真实有监督事件、原始样本/成员离散度
    → 独立校准 + 执行约束 + 任务收益 → 短执行前缀
    → 固定控制器 → MuJoCo 逐动作真实反馈 → 对齐残差、更新任务、重新规划
```

接口实现：`models/embodied_rssm.py`、`models/embodied_world.py`、`training/embodied_loss.py`、
`training/embodied_trainer.py`、`agents/embodied_provider.py`、`models/embodied_service.py`。
来源固定版本及借鉴边界见 [设计规格](superpowers/specs/2026-10-04-scalable-embodied-world-model-design.md#2-来源及借鉴边界)。
every-embodied提供方法/数据/仿真导航；DreamerV3启发后验/先验分离、离散潜变量、balanced KL和分组动力学。
这里是独立PyTorch实现，不是官方Dreamer权重加载器，不训练actor–critic，不复制上游环境。

网络有共享多视角CNN、具名状态/动作编码、同一时刻模态融合、分组循环动力学、categorical prior/posterior、物理残差头和RGB解码。
训练包含后验重构、动作-only先验多步预测、全帧/动态区域损失、FP32 balanced KL/free nats、仅mask有效的事件损失。
只用最近K步真实历史；startup补齐不被当成实测视角；未来真图/状态不进入预测先验。
指令按原始动作增量积分，不作为学习误差收益，也不裁剪越界候选。不存在标签的事件不输出“成功概率”。

## 环境与数据

以下命令在项目根目录执行。仅用`wmal`；已安装依赖不需要重复安装。

```bash
conda activate wmal
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# 无窗口采集/评估用 EGL；桌面 viewer 前另行 unset MUJOCO_GL
export MUJOCO_GL=egl
```

数据放`data/local/`，模型、训练状态、校准和评估放`runs/`；这些目录已被Git排除。
原生格式`wmal.embodied_episodes.v1`：NPZ保存对齐RGB、physical、commands、actions、视角有效mask、时间和可选事件mask；manifest声明机器人、坐标顺序/单位、周期、相机、控制器及来源身份。
train/validation/calibration/test按原始episode拆分，原始ID、内容hash和派生别名都不得跨split；微调继承祖先训练与选型暴露记录。
最多缓存两个解压episode，原始文件只读。旧四维G1状态需要明确分成实测[0,1]、指令[2,3]；不会编造qvel。
开源LeRobot数据沿用[数据获取/导入指南](23_visual_world_training_and_agent.md)；机器人、单位、动作语义或控制器不同，必须独立profile/转换/训练评估，不能把高维操作数据强行映射成这两个关节。

## 本机小配置完整流程

所有输出必须是新路径。示例名字若已存在请换新后缀，不覆盖既有实验。
采集新数据、32个实际动作保证T32训练窗；固定控制器与局部适用范围。

```bash
python scripts/visual_world.py collect \
  --output data/local/embodied_gc2_repro \
  --task stack_block --episodes 40 --steps 32 --image-size 32 --seed 1171 \
  --controller gravity_compensated --target-low .21 .73 --target-high .49 1.01

python scripts/embodied_world.py inspect \
  --manifest data/local/embodied_gc2_repro/manifest.json \
  --legacy-physical-indices 0,1 --legacy-command-indices 2,3 --legacy-command-rule joint_delta

python scripts/embodied_world.py profile \
  --manifest data/local/embodied_gc2_repro/manifest.json \
  --legacy-physical-indices 0,1 --legacy-command-indices 2,3 --legacy-command-rule joint_delta \
  --network-config configs/training/embodied_smoke.json \
  --device cuda --precision bfloat16 --members 3 --batch-size 2 --accumulation 4 \
  --output runs/embodied_resource_repro.json

python scripts/embodied_world.py train \
  --manifest data/local/embodied_gc2_repro/manifest.json \
  --legacy-physical-indices 0,1 --legacy-command-indices 2,3 --legacy-command-rule joint_delta \
  --network-config configs/training/embodied_smoke.json \
  --device cuda --precision bfloat16 --epochs 1 --members 1 --batch-size 2 --accumulation 4 \
  --output runs/embodied_smoke_repro

python scripts/embodied_world.py calibrate \
  --manifest data/local/embodied_gc2_repro/manifest.json \
  --legacy-physical-indices 0,1 --legacy-command-indices 2,3 --legacy-command-rule joint_delta \
  --model runs/embodied_smoke_repro/model.pt --device cuda --samples 8 --mode sample --alpha .2 \
  --output runs/embodied_calibration_repro.json

python scripts/embodied_world.py evaluate \
  --manifest data/local/embodied_gc2_repro/manifest.json \
  --legacy-physical-indices 0,1 --legacy-command-indices 2,3 --legacy-command-rule joint_delta \
  --model runs/embodied_smoke_repro/model.pt --calibration runs/embodied_calibration_repro.json \
  --device cuda --output runs/embodied_test_repro.json
```

原生manifest不加legacy三个参数。`inspect`检查元数据和文件hash，不读取episode数组。
`--network-config`是网络配置；`--training-config`是`EmbodiedTrainingConfig`训练协议JSON，不能互换。
进度条默认开启，`--no-progress`仅关闭显示。训练状态为`training.pt`，冻结推理权重为`model.pt`，另有`report.json`、逐epoch指标。
成员独立初始化、原始episode bootstrap、顺序训练；单成员的8次随机轨迹不是8成员ensemble。
validation只按物理预测指标选epoch；冻结选型后calibration拟合区间，最后test评估，不能看到test后再调参。

## 精确续训与微调

续训必须同一输出目录、同一配置/代码身份/环境/数据/设备拓扑；只在优化器更新边界恢复。
它恢复优化器、scheduler、scaler、各rank RNG、采样器及剩余成员状态，不是只加载权重。
已完成训练的同配置续训不会自动追加epoch，修改epoch应新开微调实验。

```bash
python scripts/embodied_world.py train \
  --manifest data/local/embodied_gc2_repro/manifest.json \
  --legacy-physical-indices 0,1 --legacy-command-indices 2,3 --legacy-command-rule joint_delta \
  --network-config configs/training/embodied_smoke.json \
  --device cuda --precision bfloat16 --epochs 1 --members 1 --batch-size 2 --accumulation 4 \
  --output runs/embodied_smoke_repro --resume runs/embodied_smoke_repro/training.pt
```

微调必须先独立采集新四split数据，例如将上述collect的输出改为`data/local/embodied_gc2_finetune`、seed改为1172。
使用相同profile/网络与成员数量；保留父模型归一化和祖先暴露；重置优化器。旧CNN/GRU、旧RSSM、UniFoLM或Dreamer官方权重不兼容本模型族，不能直接传入。
冻结编码器可在训练协议JSON设`"freeze_encoder": true`，其tensor保持不变；其余模型依然训练。

```bash
python scripts/embodied_world.py train \
  --manifest data/local/embodied_gc2_finetune/manifest.json \
  --legacy-physical-indices 0,1 --legacy-command-indices 2,3 --legacy-command-rule joint_delta \
  --network-config configs/training/embodied_smoke.json \
  --device cuda --precision bfloat16 --epochs 1 --members 1 --batch-size 2 --accumulation 4 \
  --pretrained runs/embodied_smoke_repro/model.pt --output runs/embodied_finetune_repro
```

## 八卡容量与资源门槛

单成员默认单相机128px、K8/H8、T32/31动作：WM-S40,174,005；WM-B151,501,701；WM-L298,675,541。
三个独立B成员454,505,103，总量不代表一份共享模型。改变视角/事件维度后需重新计数。
目前只有本机8GB RTX5060Ti的smoke实测和双rank CPU/gloo测试，**没有8×3090实测**。
正式先执行完整更新+M/S8推理资源探测，allocated/reserved均不得超过各卡启动时显存的80%；超限报错，不静默缩小网络或批次。
profile其余成员是独立随机探测对象，不是已训练的研究模型。

以下命令仅供已有八卡服务器的`wmal`环境使用。先准备128px、同一profile的新原生manifest `data/local/embodied_server/manifest.json`（不能使用上述32px数据），未在本机执行。

```bash
torchrun --standalone --nproc-per-node=8 scripts/embodied_world.py profile \
  --manifest data/local/embodied_server/manifest.json \
  --network-config configs/training/embodied_base.json \
  --device cuda --distributed ddp --precision bfloat16 \
  --members 3 --batch-size 2 --accumulation 4 --output runs/embodied_base_gpu8_probe.json

torchrun --standalone --nproc-per-node=8 scripts/embodied_world.py train \
  --manifest data/local/embodied_server/manifest.json \
  --network-config configs/training/embodied_base.json \
  --device cuda --distributed ddp --precision bfloat16 \
  --members 3 --batch-size 2 --accumulation 4 --output runs/embodied_base_gpu8
```

有效训练批次2×4×8=64序列，三个成员顺序训练。FSDP只允许CUDA+已明确支持的Torch2.11.0，使用`--distributed fsdp`替代ddp；不支持时明确报错，多卡FSDP仍未实测。
网络base模板是G1两关节profile；其他机器人必须显式适配坐标/单位/相机/动作，不可仅改manifest路径。

## 持续仿真与推理服务

以下新模型可以验证闭环但不保证完成任务：本次一epoch的smoke A3因区间超过预算拒绝全部候选，保留负结果。
不要通过放宽误差预算把失败“修成成功”。若需观察已通过的五节点动作，使用[旧冻结模型复现命令](29_rssm_five_node_fix.md)，它不是新模型能力结果。

```bash
# 桌面窗口；结束任务后仍保留会话。输入 run 重新开始，quit 退出。
unset MUJOCO_GL
python scripts/g1_task_agent.py \
  --model-family embodied --checkpoint runs/embodied_smoke_repro/model.pt \
  --calibration runs/embodied_calibration_repro.json --config configs/tasks/g1_joint_sequence.json \
  --controller gravity_compensated --baseline A3 --device cuda \
  --output runs/embodied_agent_viewer_repro --viewer --realtime --monitor --monitor-port 8766
```

监控只读，模型没有执行权限。request绑定run/task/node/episode/step、真实历史ID和实测时间、candidate/skill、动作hash、controller/model/protocol。
选中后另生成execution grant并链接prediction decision；dispatch前再次检查真实观测与仿真时间。每个执行端点都返回真实反馈，不能跳过中间观测。
超时会终止自有推理worker并锁定，必须明确重置；不会后台继续规划、盲目重试或自动派发fallback。旧visual模型保持默认加载路径。

可选独立进程服务（新终端仍用`wmal`，不占用户已有8765监控端口）：

```bash
conda activate wmal
export PYTHONPATH="$PWD/src"
python -m wmal.models.embodied_service \
  --checkpoint runs/embodied_smoke_repro/model.pt --device cuda --host 127.0.0.1 --port 8767
```

Agent窗口命令添加`--inference-url http://127.0.0.1:8767/predict`，本地仍加载相同冻结模型与校准用于响应验证。
服务仅有`/predict`，没有执行接口；非loopback绑定必须设置环境变量`WMAL_INFERENCE_TOKEN`且客户端使用经验证的HTTPS，TLS需由前置代理终止；服务本身不提供公网TLS。
客户端只接受字面IP/localhost以避免无界DNS，跨主机deadline以剩余预算转换，原始客户端绑定必须回传。传输限制和过期/重复/reset响应都在暴露预测前检查。
这不是公网生产部署指南；未测试真机、公网TLS或八卡推理吞吐。

## 评估解释

report仅对test真实动作统计原始物理单位的每H/每维RMSE，另列保持基线、只用train拟合的固定线性delta递归基线、RGB全帧和动态区域MSE。
动态区域按真实未来图与起点图的差异mask独立归一化；没有变化则报告null，不虚称动态预测有效。
零动作/打乱动作使用相同观测及随机种子，统计预测变化，不是反事实准确率；越出模型范围的对照单列数量，绝不裁剪。
校准原始episode不得出现在test；原始episode内所有窗口/物理坐标联合覆盖，Wilson区间以episode计数。
正式样本量标记要求cal/test各至少100条独立轨迹，但这也不替代分布、覆盖、风险与成本的完整统计检验。
完整实测数值、负结果与尚未验证的能力见[验收记录](architecture/embodied-world-verification.md)。
