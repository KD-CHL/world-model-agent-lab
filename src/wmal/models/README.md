# models

规划模型可加载本项目生成的单关节探针检查点；正式机器人模型尚未训练和验证。

- `latent_dynamics.py`：独立编写的动作条件关节位移回归基线、bootstrap 成员和检查点读写；不是论文算法的复现。
- `action_service.py`：可选视觉动作服务客户端，返回动作序列，不向规划器冒充状态预测。
- `skill_outcome.py`：早期场景方案留下的占位模块；当前 Agent 路径不调用。
- `calibration.py`：预留 calibration 模块；接口与验收见 docs 文档。
