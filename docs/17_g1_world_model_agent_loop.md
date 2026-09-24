# G1 世界模型 Agent 持续仿真闭环

本入口将任务 Agent、G1 浮动底座世界模型、滚动规划、Unitree 低层行走策略和 MuJoCo 分开。它不会训练模型，也不会把低层 ONNX 行走策略或 MuJoCo 物理仿真伪装成学习世界模型。

## 世界模型接入契约

配置项 `world_model.factory` 必须是 `Python模块:工厂函数`。工厂函数接收 JSON 的 `world_model.config` 字典，并返回具有以下接口的对象：

```python
class MyG1WorldModel:
    version = "checkpoint-or-service-version"
    state_schema = "wmal.g1.base_state.v1"
    action_schema = "wmal.g1.body_velocity.v1"

    def predict(self, state, action, duration_s):
        # state: wmal.locomotion.contracts.G1State
        # action: wmal.locomotion.contracts.G1VelocityAction
        # return G1Prediction(next_state, uncertainty={"position_m": 0.02})
        ...
```

预测必须返回 `G1Prediction` 或 `G1State`；预测状态应保持 episode 与状态字段 schema 一致，且 step/time 前进。`step_id` 按 Agent 可见的已完成动作段递增，不按 MuJoCo 的 2ms 物理步或低层策略的 20ms 控制周期递增。模型还必须声明精确的 `state_schema="wmal.g1.base_state.v1"` 与 `action_schema="wmal.g1.body_velocity.v1"`，否则启动时拒绝接入。`G1State` 提供世界坐标系底座位置、偏航、世界坐标系线速度/角速度、roll/pitch、躯干高度、以弧度表示的关节位置和足端接触。动作是机身坐标系速度 `(vx, vy, yaw_rate)` 和预测时长。适配器可以加载本地检查点，也可以封装推理 HTTP/gRPC 服务；服务密钥只能从环境变量读取，不能放进实验 JSON。

Unitree ONNX 策略是低层速度到关节目标控制器，**不是**上面的状态预测器。若现有开源模型只生成视频或动作块，需另写适配器/状态动力学模型，将其转换为符合此预测契约的、经过语义验证的状态预测；不能直接当作 `predict` 使用。

## 运行

先在已安装 MuJoCo 和 `walking` extra 的 `wmal` 环境中，将示例里的 `factory`、检查点路径和版本改成自己的模型适配器信息，然后运行：

```bash
conda activate wmal
PYTHONPATH=src python scripts/g1_agent_sim.py \
  --config configs/experiments/g1_closed_loop.example.json
```

窗口打开后，在终端输入世界坐标目标：

```text
goal> 1.0 0.0
goal> 1.0 0.5 90
goal> quit
```

Agent 对候选速度序列调用世界模型进行预测，综合目标误差、姿态稳定性、动作代价和模型不确定度选取序列，只执行第一段速度指令。G1 ONNX 策略将该指令转换为关节控制，MuJoCo 执行后返回新状态，Agent 再规划。达到一个目标后仿真窗口保留，可继续输入下一个目标；输入 `quit`、终端 EOF 或按 Ctrl-C 才退出。跌倒/姿态安全保护触发后停止当前会话。

模型调用配置缺失、checkpoint/schema 不兼容或模型预测无效时会拒绝规划，不会退回固定步行动作。若未训练也未提供符合契约的预训练/外部世界模型，目前只能运行模型接口测试，不能进行有意义的世界模型预测实验。实验事件与预测误差默认写入 `runs/g1_agent/events.jsonl`。规划耗时随 `samples × horizon` 次串行模型调用增加；远程服务使用前应评估延迟，并可在适配器内部批量推理。

## 小型接口冒烟测试

纯接口测试使用测试夹具模型，验证配置、Agent、滚动规划和多任务会话生命周期；它不是实验结果，也不是一个可用于研究的训练/预测模型：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_g1_locomotion_runtime.py' -v
```

真实 G1 MuJoCo/ONNX 会话烟测和全项目验证命令见根目录 README。仿真结果不能代表真机安全或 sim-to-real 能力。
