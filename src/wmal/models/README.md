# 模型模块

- `latent_dynamics.py`：低维状态、动作目标条件下的 bootstrap 岭回归集成。
- `neural_dynamics.py`：可选 PyTorch MLP 成员集成，训练入口位于 `training/neural_trainer.py`。
- `loader.py`：按 `.json`、`.pt`/`.pth` 加载项目格式的状态模型。
- `visual_prediction.py`：目标图像规划使用的图像特征预测协议和候选序列优化适配器。
- `action_service.py`：语言条件动作序列服务客户端；动作提议与未来状态预测保持不同语义。

神经和视觉功能属于新接入代码，尚未运行验证。视觉模型必须由显式配置的本地插件提供推理能力。
