# Agent 模块

- `api_client.py`：兼容 OpenAI 风格 Chat Completions API 的任务目标解析；严格校验 JSON、关节名、关节范围和 API 超时。
- `runner.py`：观察状态、请求世界模型规划、执行第一步、用环境观测核验完成，并记录模型预测残差后重规划。
- `cli.py`：启动 Agent 或通过 ROS 2 服务启动规划器/机器人节点。

Agent 只提出受限目标，不直接写 MuJoCo 状态或生成未校验的关节控制；当前高层目标类型仍是关节目标，物体级技能规划等场景确定后再添加。
