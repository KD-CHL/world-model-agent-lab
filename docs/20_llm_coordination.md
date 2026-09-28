# API 大模型规划与 G1 执行协调

## 分层职责

```text
自然语言 → LanguageMissionPlanner（带 API Key 的 Chat Completions 请求）
        → 严格 JSON、能力、目标边界与路径可达性校验
        → G1Coordinator（任务 ID、单任务互斥、观测版本）
        → MissionAgent（多阶段目标）
        → NavigationPlanner + G1RolloutPlanner（调用配置的预测模型）
        → ExecutionGate（一次性执行许可）
        → G1MuJoCoSession → 实际观测与回执 → 残差反馈与下一周期
```

大模型输出绝对世界坐标目标及可选朝向，不直接产生关节控制或执行代码。当前能力是室内导航；请求抓取、折毛巾等未支持任务时应返回 `unsupported`。十二种操作环境已经具备物理建模，但未接入操作策略或 Dex1 夹爪，不能通过语言规划凭空获得这些能力。

预测模型通过 `world_model.factory` 显式加载，配置可使用 `expected_version` 锁定版本。默认 `configs/g1_indoor.json` 仍加载轻量底座动力学基线，不是视觉世界模型。大型视觉世界模型及其动作语义适配尚未完成；本功能没有宣称已接入 UniFoLM。

## API 配置

支持提供 `/chat/completions` 和 JSON object 输出的兼容服务。`LLM_BASE_URL` 应包含服务要求的版本前缀，例如 `https://你的服务域名/v1`，不要包含 `/chat/completions`。模型名称由服务提供商确定。

```bash
cd /home/chl/GitHub/CodeSpace/world-model-agent-lab-main
conda activate wmal
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export LLM_BASE_URL='https://你的服务域名/v1'
export LLM_MODEL='你的模型名称'
export LLM_TIMEOUT_S=30
export LLM_RETRIES=2

# 在终端隐藏输入，避免密钥进入命令历史
read -rsp 'API Key: ' LLM_API_KEY
export LLM_API_KEY
echo

python scripts/g1_llm_agent.py --config configs/g1_indoor.json
```

输入例子：“到世界坐标 (4,0)，然后回到 (0,0)。”窗口保留并接受下一条指令；输入 `quit` 结束。输入及 API 等待期间 MuJoCo 物理时间暂停。

单任务无窗口运行：

```bash
python scripts/g1_llm_agent.py --headless \
  --instruction '前往世界坐标 (1,0)，到达后结束任务'
```

`.env.example` 提供字段说明。程序直接读取进程环境变量，不会自动加载 `.env`。不要把真实密钥写进配置 JSON、源码或聊天消息。日志不记录 API Key、用户原始指令或 HTTP 响应正文，但会记录目标坐标、执行观测等实验数据。

## 通信与故障行为

- API 使用 HTTPS（本机测试地址允许 HTTP），拒绝自动重定向；限制响应大小，检查生成结束原因与 JSON 类型。
- 仅对 429、500、502、503、504 和网络故障做有限退避重试；401/403 或非法输出直接失败。重试最多三次。
- API timeout 用于剩余等待预算和 socket 超时；本地预测超过配置时限后拒绝动作。Python 同步调用不能强制中断卡死的本地模型，也不提供硬实时调度保证。
- 大模型规划完成后再次检查 episode、step 和仿真时间。观测变化则拒绝旧目标。
- 预测必须对应同一 episode、下一执行步和动作时长，执行前再检查观测。
- 每条动作分配 command ID，执行许可只消费一次，发送及回执写入同一 task ID 的日志。
- 执行抛出异常或回执时序不符时，将协调器锁定为 faulted，停止后续动作，重新启动会话后恢复；动作本身不自动重试。
- 协调器一次只运行一个任务。多目标任务每阶段有周期预算，任一阶段失败即结束任务。

该执行门控是同进程、同步 MuJoCo 会话保护，不是分布式 exactly-once 协议。已有 ROS 2 的 API joint-goal 路径继续可用；新的 G1 协调入口不宣称完成了跨机器网络部署。真机需要机器人端 watchdog 和独立停止接口。

## 日志与验证

默认 `runs/g1_llm/events.jsonl`，事件包含 `language_planning`、`mission_validated`、`plan`、`command_sent`、`command_ack`、`prediction_residual`、`feedback` 和 `mission_result`。执行结果不确定时记录 `command_uncertain`。

测试使用真实本地 HTTP 服务模拟一次 503 再成功，连接真实 MuJoCo，并使用明确标记的测试运动学模型验证协议链路；不消耗外部 API 额度，也不替代真实提供商认证与视觉模型验证。另覆盖非法目标、未知输出字段、重复动作、执行异常锁定和密钥不出现在日志中。

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_g1_coordinator.py' -v
```
