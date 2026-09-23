# 仿真模块

- `mujoco_backend.py`：MuJoCo 模型绑定、物理步进、机器人状态、RGB 相机渲染和 integration state 快照。
- `mujoco_env.py`：采样/评估用同步步进包装器，返回实际步数、仿真时间与执行控制记录。
- `base.py`：采样器和评估器依赖的机器人仿真协议。

MuJoCo backend 同一时刻应由单一 owner 线程/锁控制。Go2/G1 需要提供配置的 locomotion plugin 和经过验证的资产，不会从机械臂关节接口自动获得步态能力。快照需同一资产与模型版本；渲染依赖可用的 MuJoCo OpenGL 环境。
