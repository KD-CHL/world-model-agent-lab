# communication

ROS communication is optional at import time; nodes are created only by the runtime CLI after ROS 2 interfaces are built.

- `contracts.py`：预留 contracts 模块；接口与验收见 docs 文档。
- `transport.py`：预留 transport 模块；接口与验收见 docs 文档。
- `episode_guard.py`：预留 episode_guard 模块；接口与验收见 docs 文档。
- `observation_history.py`：有界、episode/step 有序的 RGB 与 Observation 配对缓存；支持 RGB/BGR stride 解码。

ROS robot node publishes the optional `RobotSensorFrame` custom message only when a MuJoCo camera is explicitly configured. The image carries the exact same serialized observation as its paired state.
