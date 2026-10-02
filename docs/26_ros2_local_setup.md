# 本机 ROS 2 与项目 wmal 环境

## 配置边界

Ubuntu 24.04 已安装 `/opt/ros/jazzy`。系统 ROS Python 是 3.12，本项目继续使用唯一的 `wmal` 环境（Python 3.10），不升级项目 Python，也不创建其他环境。

系统 `rclpy` 编译扩展不能直接被 Python 3.10 加载。这与 [ROS 官方故障排查](https://repo.test.ros2.org/en/jazzy/How-To-Guides/Installation-Troubleshooting.html) 所述的解释器/编译版本不匹配一致。因此本项目在 `ros2/.python310/` 构建局部 Python ABI overlay：沿用系统 Jazzy 的 C++ 中间件，重编译 `rclpy`、其必要消息依赖及 `wmal_interfaces`。这不是全套 ROS 的 Python 3.10 官方发行版；未重新编译的其他 ROS Python 扩展不能因此自动兼容。

源码取自官方 `ros2/rclpy`（7.1.12）、`ros2/rcl_interfaces`（2.0.4）、`ros2/common_interfaces`（5.3.8）、`ros2/unique_identifier_msgs`（2.5.1），匹配本机安装版本；构建脚本校验系统版本和下载归档 SHA256，不自动追踪 rolling。系统 ROS 更新后需重新审查版本和验证，脚本遇到版本变化会停止。

## 本机使用

在项目根目录，每个运行项目 ROS 节点的终端都执行：

```bash
conda activate wmal
source scripts/setup_ros2.bash
python -c 'import rclpy; from wmal_interfaces.srv import G1Session; print("ROS project bindings ready")'
```

默认设置实验域 `ROS_DOMAIN_ID=73`，中间件为 Fast DDS。已有的域和中间件设置会保留，通信双方须一致；发现范围固定为本机 `LOCALHOST`，覆盖系统默认 `SUBNET`，本指南不配置跨机器通信。这些设置是仿真实验隔离，不是真机认证或安全机制。请勿连接真机控制话题。

不要在这个 overlay 终端直接运行系统 `/usr/bin/ros2`：它的解释器是 Python 3.12，而项目 overlay 的扩展是 3.10。系统 ROS 命令另开只加载 `/opt/ros/jazzy/setup.bash` 的终端；系统端也不能直接使用本 overlay 的项目消息 Python 扩展。本项目节点、训练和 MuJoCo 程序始终用 `wmal` 的 `python`。

## 重建或在另一台同版本机器复现

系统层要求：Ubuntu 24.04、ROS Jazzy、`python3-colcon-common-extensions`、`cmake`、C/C++ 编译器、`pybind11-dev`、`curl`，以及 ROS 相关开发包。当前机器已具备这些依赖，无须重新安装或输入 sudo 密码。新机器若缺失系统依赖，先由管理员按 [Jazzy 安装文档](https://repo.test.ros2.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) 完成安装。

```bash
conda activate wmal
bash scripts/build_ros2_wmal.bash
source scripts/setup_ros2.bash
PYTHONPATH= python -m pip check
```

构建工具的 Python 依赖安装到同一个 `wmal`，清单为 `requirements-ros2-build.txt`。下载、源码、编译日志和安装 overlay 全部在 `ros2/.python310/`，已排除 Git。构建复用现有源码目录，不覆盖本地上游源码改动；这类改动会影响复现，正式实验应记录它们。

默认每包最多两个编译任务，同时限制 Make 与 CMake；可显式设置 `WMAL_ROS_BUILD_JOBS=4` 调整。下载先写临时文件并校验，再成为可复用缓存；损坏的旧缓存移到 `.invalid.*` 备份，失败传输保留 `.part` 文件，重新运行会重新下载，不删除本地文件。

## G1 仿真服务与 Agent

终端一（保持运行，只有此进程拥有仿真）：

```bash
conda activate wmal
source scripts/setup_ros2.bash
python scripts/serve_g1_ros2.py --viewer
```

无显示设备时去掉 `--viewer`，可设置 `export MUJOCO_GL=egl`。观测、动作和复位通过 `/wmal/g1/session` 服务；等待规划时不自动推进物理。Agent 任务完成并不关闭服务进程；用户仍可提交后续任务。

终端二（需事先设置已有的 API 配置，见 `docs/20_llm_coordination.md`，不要把密钥写入仓库）：

```bash
conda activate wmal
source scripts/setup_ros2.bash
python scripts/g1_llm_agent.py --transport ros2 --headless
```

该命令进入连续输入任务模式；例如 `前往世界坐标 (1,0)`。它使用 `configs/g1_indoor.json` 指定的现有底座状态预测器，不能改成不兼容的视觉 RSSM 权重；两条模型路径的状态/动作语义不同。API 调用、导航成功率及真机控制不包含在 ROS 通信配置验收中。

## 验证

```bash
conda activate wmal
source scripts/setup_ros2.bash
MUJOCO_GL=egl python -m unittest discover -s tests -v
```

`test_g1_ros2_integration.py` 中的协议服务测试使用测试会话，另有真实 MuJoCo 子进程启动/退出测试；额外的模型规划双进程验证单独记录，不能混称为世界模型性能实验。

### 2026-10-02 本机验收记录

- 实际构建 13 个 ROS 包；增量重建通过。cp310 `rclpy` 扩展、项目服务/Action、相机消息及所需类型支持均可加载。系统层未改动，未新建 Conda 环境。
- 修复 G1Session 和通用规划器回调覆盖 `rclpy.Node.handle` 属性的问题；实测节点创建曾失败，修复后真实服务请求通过。另修复 G1 服务在 SIGINT 后重复关闭 ROS context 的报错，真实子进程测试覆盖启动、观测与干净退出。
- 真实双进程 G1 MuJoCo：一次 `vx=0.1 m/s`、`duration=0.2 s` 的命令推进仿真 `0.20000000000000012 s`；机器人高度 `0.7795 m`、roll `-0.0122 rad`、pitch `-0.0296 rad`，姿态保护未触发。此短暂起步的 x 位移为 `-0.00644 m`，不声称该命令完成向前行走任务。重复观测确认物理暂停，复位生成新回合，关闭客户端后新客户端仍能连接。
- 强制 LOCALHOST 后，另用已有底座预测器 `g1-ridge-74ccb58ff0c8` 与 G1Agent/NavigationPlanner 执行 4 个 ROS 闭环周期，推进 `2.0 s` 仿真，x 位移 `+0.1524 m`；返回 `budget_exhausted`，服务仍运行，复位和重新连接通过。预测、执行、残差事件保存在本地 `runs/ros2_verification_20261002/agent-events.jsonl`。这是底座模型通信/闭环冒烟验证，不是视觉 RSSM 或导航成功率实验。
- 同一 `wmal` 可加载 MuJoCo `3.12.0`、PyTorch `2.11.0+cu128`、ONNX Runtime `1.23.2`，CUDA 可用。清空 ROS 的 Python 搜索路径后对 `wmal` 执行 `pip check` 通过；这是避免将系统 ROS 的包元数据当作项目 pip 依赖，并不切换环境。
- 默认两个编译任务经实际 colcon 构建命令检查：不再附加 `-j28 -l28`，Make 使用 `MAKEFLAGS=-j2 -l2`。校验坏缓存和中断下载的可重复测试通过，坏缓存保留，不删除文件。

完整测试 **190 项通过、无跳过**（17.907 秒），输出保存在本地 `runs/ros2_verification_20261002/full-tests.log`；目录已排除 Git。无凭据的外部 API 未调用，本次没有真机测试、跨主机实验或通用三节点 Action/VLA 的完整任务验收。
