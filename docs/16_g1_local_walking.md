# G1 浮动底座本地步行测试

本测试将两条路径分开：`configs/robots/g1_fixed_base.json` 继续用于关节级调试；步行使用 Unitree 官方 `unitree_rl_mjlab` 的 G1 29-DOF 浮动底座碰撞模型、速度策略 ONNX 与部署参数。模型与权重固定在上游提交 `1425b15f73bd4095f0df53709d7c389c3eb9e790`，本地资产许可证位于 `robots/assets/unitree_g1_mjlab/LICENSE`。

## 安装

在项目根目录运行：

```bash
conda env update -f environment.yml --prune
conda activate wmal
```

`environment.yml` 会安装 MuJoCo、NumPy、ONNX Runtime 和项目本身。若不使用该环境文件，等价安装命令为：

```bash
conda activate wmal
python -m pip install -e '.[simulation,walking]'
```

## 走 10 步并验证

```bash
conda activate wmal
PYTHONPATH=src python scripts/walk_g1.py --steps 10 --viewer
```

`--viewer` 会打开 MuJoCo 图形窗口并按接近真实时间播放；默认视角会跟随观察机器人所在区域。窗口可用鼠标旋转/缩放，10 步结束后会自动关闭并在终端打印 JSON 结果。若只需要无窗口验证，可去掉 `--viewer`。

这里的“一步”按 MuJoCo 中检测到的**左右脚交替触地事件**计数，不是只按计时器估算。成功时命令输出 JSON，包括触地步数、向前位移、躯干高度、倾斜角和模拟时长；如果策略超时、模型跌倒或姿态超出保护范围，命令会以非零状态停止，不报告成功。

可选参数：

```bash
PYTHONPATH=src python scripts/walk_g1.py --steps 10 --velocity-x 0.2 --max-duration 12
```

前进速度范围限制为 `0.05–0.5 m/s`。这是隔离的本地仿真测试，不连接真机；不要把它当作真机安全验证或 sim-to-real 结论。测试报告和可复现环境版本建议与实验记录一并保存。

## 实现与验证范围

- 使用 Unitree 发布的 G1 浮动底座 MJCF，保留其专为速度策略配置的碰撞几何；运行时添加平地和由发布部署参数设定的 29 个 PD 位置执行器，并写入上游按电机齿轮参数计算的关节反射惯量。
- ONNX 输入按上游部署观察项组装：机身角速度、投影重力、速度命令、步态相位、关节位置偏差、关节速度和上一动作；输出映射为 29 个限幅位置目标。
- 仿真控制周期为 `0.02 s`，MuJoCo 物理步长为 `0.002 s`。脚步由左右脚碰撞几何与地面接触边沿检测，并对重复碰撞做去抖。
- 已在本项目 Conda `wmal` 环境以 CPU ONNX Runtime 执行 10 步无显示测试：10 次触地、前进约 `0.60 m`、结束机身高度约 `0.78 m`、倾斜约 `0.02 rad`，总模拟时间约 `4.02 s`。完整自动化测试可通过 README 中的 unittest 命令重跑。

**限制：** 策略与 G1/Mjlab 模型来源相同，但本地轻量回放不是上游训练过程本身；无地形随机化、扰动或感知闭环，不用于评估鲁棒性。策略模型和数据集许可需分别核查后再用于再分发或商业用途。
