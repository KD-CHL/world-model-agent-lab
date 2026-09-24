# Agent 模块

- `api_client.py`：兼容 OpenAI 风格 Chat Completions API 的任务目标解析；严格校验 JSON、关节名、关节范围和 API 超时。
- `runner.py`：观察状态、请求世界模型规划、执行第一步、用环境观测核验完成，并记录模型预测残差后重规划。
- `cli.py`：启动 Agent 或通过 ROS 2 服务启动规划器/机器人节点。
- `action_runner.py`：可选图像/状态条件策略闭环；调用独立动作服务，每轮只执行 action chunk 第一项，并强制等待新观测。

Agent 只提出受限目标，不直接写 MuJoCo 状态或生成未校验的关节控制；当前高层目标类型仍是关节目标，物体级技能规划等场景确定后再添加。
高层 Agent 的输出应先通过 `ConstrainedTaskPlanner`，再进入规划器。该边界
允许接入 LLM、规则 Agent 或外部策略，同时统一检查目标关节和关节限位；Agent
不直接调用机器人执行接口。

策略模式使用 `python scripts/run_agent.py --mode wma-policy`，依赖显式相机、状态顺序、动作映射和成功检测插件。它不会加载 WMA checkpoint；训练/推理服务保持外部隔离。
