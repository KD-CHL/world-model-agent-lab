# G1 操作任务虚拟工作环境

这是依据用户列出的十二项 Unitree 数据集任务设计的程序化虚拟工作环境。沿用现有室内墙体、货架、储物柜，新建带桌腿的工作台及各任务道具。几何尺寸、相机、材质与质量为仿真假设，没有从真实数据集完成标定或场景重建。

## 十二个场景

| --task | 场景内容 | 物理建模边界 |
|---|---|---|
| stack_block | 三色积木、堆叠目标区 | 自由刚体方块 |
| bag_insert | 开口袋形容器、待装包裹 | 袋子使用刚性容器代理，不模拟布袋变形 |
| erase_board | 立式白板、笔迹、板擦 | 板擦为自由刚体；笔迹是视觉标记 |
| clean_table | 桌面杂物、收集箱 | 自由刚体与开放容器 |
| pack_pencilbox | 笔、开放笔盒 | 刚体近似 |
| pour_medicine | 可移动开口瓶、药粒、接收杯 | 刚体颗粒，无液体 |
| pack_pingpong | 乒乓球、装球盒 | 球体刚体 |
| prepare_fruit | 三个水果、果盘 | 刚体水果；无切割 |
| organize_tools | 工具、三个分类槽 | 组合刚体工具 |
| fold_towel | 毛巾、折叠参考区 | 9×9 MuJoCo flex 网格，材料参数未标定 |
| wipe_table | 海绵、污渍区域 | 海绵为刚体；污渍不随接触自动消失 |
| dualrobot_clean_table | 两台 G1、共享桌面、杂物和收集箱 | 两台独立固定底座机器人；无协同策略 |

机器人保留现有 G1 29 个关节，用位置执行器保持初始姿态，底座固定以便开发操作任务。第二台机器人位于桌子对面。`stack_block` 单独将工作台移至左臂附近，并添加两指平行夹爪仿真代理；它不是 Unitree Dex1 资产。代理使用命名关节、位置执行器和刚性接触几何，原手掌碰撞体在该场景停用以避免与代理重复接触。其余任务仍保持 29/58 个位置执行器。

首版操作会话已提供显式 reset、含末端位姿的 agent 可见机器人观测、独立 scene truth、受限动作回执和固定 waypoint 技能接口。夹爪闭合控制已改为按限速推进目标开度；抓取闭合以观测到的双侧接触提前结束，抬升拆为多个有界段，放置包含释放后的稳定等待。基线在放置稳定谓词成立后即结束，不执行仍未通过验收的撤离 waypoint；试验观察到旧撤离轨迹会碰撞并推倒方块，因此不把它计入成功路径。

**无世界模型技能基线**：jittered 开发 seeds 0–9 为 10/10；两个独立 seed 批次（100–119、1000–1059）合计 79/80（98.75%；Wilson 95% 区间约 93.3%–99.8%）。seed 106 在放置时倾倒，验证器拒绝通过；该结果仅覆盖当前窄幅 jitter 初态。基线已达到初始 90% 门槛，但尚未验收撤离技能，也没有验证实际 Dex1 手部或 ROS 2 闭环。

**视觉数据和训练路径已跑通**：旧采集集每个控制步包含 64×64 RGB、70 维机器人观测、14 维动作、3 类事件标签和 39 维物体真值；episode 按 24/6/6/5 隔离为 train / validation / calibration / test，测试集包含一个独立失败 episode。现行 collector 增加第 4 类 `object_dropped_after_lift` 事件，表示已稳定抬升后方块回落至 reset 高度附近；该真值标签只供训练和评估，不进入 agent 观测。既有 v2/v3/v8/v9 模型仍绑定旧的 3 事件语义，不能直接视为已训练了掉落预测器。v2 仅预测图像、机器人状态和事件；v3 将物体真值加入具名 future-target 监督，仍不把真值作为输入。v3 模型版本为 `visual-b6d5dac956aa00a89d64`。

held-out 评估使用每 32 个控制步一个窗口、共 240 窗口 / 5 个 episode。四步视觉 MSE 为 `3.90e-5` 至 `1.53e-4`，均高于 persistence 的 `2.87e-5` 至 `1.43e-4`；变化区域误差与 persistence 接近且略差。机器人状态 RMSE 为 `0.085` 至 `0.156`，优于 persistence 的 `0.092` 至 `0.204`。校准覆盖率为 100%，但平均半宽从 5.45 增至 8.61 个状态标准差，范围过宽，暂不能作为有用的决策置信度。事件准确率受正负样本不平衡影响，Brier 分数没有超过 persistence；事件 ensemble spread 与绝对误差的 Spearman 相关约 0.70，但独立测试 episode 仅 5 个，不能作稳健结论。**因此当前世界模型还未证明能改善控制决策，暂不进入声称有收益的 A0–A3 结果。**

v3 物体状态 held-out 结果见 [阶段进展报告](20_stack_block_experiment_progress.md)：39 维归一化 RMSE 四步约 `1.46`，高于 persistence 的约 `0.80–0.91`；被操作方块 `block_0` 的位置预测显著差于 persistence，姿态也未学好。物体状态 ensemble spread 与绝对误差 Spearman 约 `0.50`，只表示小测试集上有排序信号，不是校准概率。故对象状态预测同样未过门槛，不能据此执行 A0–A3。

下一步应先扩大失败与恢复轨迹、改善物体姿态预测/可信度并扩展独立测试，再执行固定预算的配对消融。ROS 2 真机和 sim-to-real 仍未实施。

## 启动与切换

```bash
cd /home/chl/GitHub/CodeSpace/world-model-agent-lab-main
conda activate wmal
export PYTHONPATH="$PWD/src"

# 积木工作台，窗口持续运行
python scripts/workcell_sim.py --task stack_block

# 毛巾柔性物体
python scripts/workcell_sim.py --task fold_towel

# 两台 G1 共享桌面
python scripts/workcell_sim.py --task dualrobot_clean_table
```

用 `--task` 选择表内任一场景；关闭窗口后可启动另一个。按 R 重置机器人和全部物体；鼠标使用 MuJoCo 原生视角交互。机器人不会自动执行任务。

批量物理检查、导出 RGB 相机图片：

```bash
python -m pip install Pillow
MUJOCO_GL=egl python scripts/workcell_sim.py --all --steps 1000 --render
```

每个场景运行 2 秒物理时间，产物位于 `runs/workcells/<task>/`：

- `scene.json`：任务、随机种子、对象名称与位姿、目标区域、模型自由度、仿真警告及建模限制。
- `workcell_overview.png`、`workcell_top.png`、`workcell_front.png`：斜视、俯视、机器人前方视角。相机是合成位置，不等同于数据集相机标定。

`--seed` 可复现道具小幅位置变化；同输出目录会更新检查报告与图片。产物受 `runs/` Git 排除规则保护。

## 后续 agent / 视觉世界模型接口

环境入口为 `build_workcell(task, seed)`，返回场景元数据、MuJoCo model 和 data。可通过命名相机采集 RGB，通过 `scene.objects` 查询物体位姿；物体位姿属于仿真真值，应只用于监督或评估，不能混作视觉模型预测。

一般工作台包含单机器人 29 或双机器人 58 个位置执行器；`stack_block` 另有 2 个代理夹爪位置执行器。可运行无世界模型技能试验并导出 `run_manifest.json`、`events.jsonl` 与 `results.json`：

```bash
python scripts/stack_block_baseline.py --output-dir runs/stack_block_baseline_<unique-name>
```

输出路径必须是新目录。未完成和失败 episode 保留在成功率分母。运行器明确不加载世界模型；只有当技能基线通过预先冻结的可靠性门槛后，才应开展操作轨迹训练与 A0–A3 对照。现有视觉训练工具仍针对 joint-servo 数据，不应将其结果误称为抓取/堆叠模型验证。

## 来源

任务语义参考用户提供的 Unitree 数据集列表；已核查链接中的 [Stack Block](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Stack_Block)、[Bag Insert](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Bag_Insert)、[Fold Towel](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Fold_Towel) 及其他可访问页面。若链接发生重定向，应以 Hugging Face 当前仓库标识为准。机器人网格复用本项目已有 Unitree 资产与其许可证；新增道具是程序化基本几何。
