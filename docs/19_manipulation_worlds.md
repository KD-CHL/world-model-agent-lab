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

机器人保留现有 G1 29 个关节，用位置执行器保持初始姿态，底座固定以便开发操作任务。第二台机器人位于桌子对面。现有模型不包含可控 Dex1 夹爪，所以这个环境不能宣称已经完成抓取、叠放或数据集动作重放。

## 启动与切换

```bash
cd /home/chl/GitHub/CodeSpace/world-model-agent-lab-main
conda activate wmal
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

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

模型包含单机器人 29 或双机器人 58 个位置执行器。后续控制器应根据 joint/actuator 名称映射动作，验证关节限位，并单独补充 Dex1 抓取资产。视觉世界模型需要以这些场景采集的同步图像与真实动作转移训练或微调，目前环境构建没有加载视觉世界模型，也没有实现任务成功奖励。

## 来源

任务语义参考用户提供的 Unitree 数据集列表；已核查链接中的 [Stack Block](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Stack_Block)、[Bag Insert](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Bag_Insert)、[Fold Towel](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Fold_Towel) 及其他可访问页面。若链接发生重定向，应以 Hugging Face 当前仓库标识为准。机器人网格复用本项目已有 Unitree 资产与其许可证；新增道具是程序化基本几何。
