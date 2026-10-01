# Agent 实验观测台

本客户端是本地、只读的实验观察工具，不替代 Agent、世界模型或低层控制器。可以连续查看任务状态、候选计划、模型预测、执行反馈、实时 MuJoCo 画面，以及已有实验工件。完成一个目标不会要求关闭页面或重启视觉 Agent 会话。

## 1. 环境与历史实验启动

只使用本项目 `wmal` 环境；前端不需要 Node、npm、网络 CDN 或额外桌面框架。

```bash
cd /home/chl/GitHub/CodeSpace/world-model-agent-lab-main
conda activate wmal
export PYTHONPATH="$PWD/src"
python -m pip install -e '.[monitor]'
python scripts/agent_monitor.py --runs-root runs --port 8765
```

浏览器访问 `http://127.0.0.1:8765`。也可用 `python -m wmal.monitor.server` 或安装后的 `wmal-monitor` 启动。`--open-browser` 可主动打开系统浏览器，默认不打开。

该命令适用于历史浏览及正在追加的事件日志；**独立监控进程不能读取另一个进程的 MuJoCo 画面**。实时画面需要按下一节在实验进程内启用发布端。

## 2. 真实视觉世界模型 + Agent + 实时画面

下列命令使用本机已有的视觉模型与校准工件。数据、权重和 runs 产物被 Git 忽略，新克隆仓库必须先按 [视觉训练指南](23_visual_world_training_and_agent.md) 采集/训练，或提供语义兼容的检查点与校准文件。

```bash
cd /home/chl/GitHub/CodeSpace/world-model-agent-lab-main
conda activate wmal
export PYTHONPATH="$PWD/src"
export MUJOCO_GL=egl
python scripts/visual_skill_agent.py \
  --checkpoint runs/visual_g1_release_finetuned_20260930/best.pt \
  --calibration runs/visual_g1_release_finetuned_20260930/calibration.json \
  --baseline A3 --goal 0.45 0.95 --error-budget 0.24 \
  --output runs/visual_monitor_run1 \
  --monitor --monitor-port 8766 --realtime --device cpu
```

浏览器访问终端打印的地址 `http://127.0.0.1:8766/?run=...`。8765 与 8766 是不同服务，避免端口占用。输出目录必须新建；再次运行换成 `runs/visual_monitor_run2`，不要覆盖实验证据。

这是**固定底座 G1 工作台、左臂两个关节目标增量 `arm_delta`**的验证，不是全身步行、抓取、堆叠或 UniFoLM 预训练模型演示。模型预测 RGB 和关节/目标状态，Agent 使用 A3 校准误差界选择动作前缀。`--realtime` 用于观察执行过程，不改变物理步长；会增加墙钟耗时。`--device cuda` 可以使用本项目环境已配置的 GPU。

首次目标结束后，在**启动实验的终端**输入下一目标：

```text
0.35 0.85
0.40 0.90
quit
```

每行是肩关节/肘关节目标弧度，范围为 `.05≤shoulder≤.65`、`.55≤elbow≤1.15`；不是自然语言指令。故障锁定后可输入 `reset` 明确重置，再输入新目标。网页不提供执行/重置按钮。输入流结束时服务保持打开，按 Ctrl+C 退出。任务间空闲时无 GUI 的会话只发布当前画面，不额外推进物理或发送未计入预算的动作。

如果还要本地交互式 MuJoCo 窗口，在有桌面的终端移除 `export MUJOCO_GL=egl`（或 `unset MUJOCO_GL`），并增加 `--viewer`；具体桌面 OpenGL 配置需能启动 MuJoCo viewer。

## 3. 浮动底座 G1 接入

`scripts/g1_agent_sim.py` 已支持相同的只读监控发布端。首先按照 [G1 闭环指南](17_g1_world_model_agent_loop.md) 准备可加载的配置和兼容的已训练 G1 状态预测器：

```bash
conda activate wmal
export PYTHONPATH="$PWD/src"
export MUJOCO_GL=egl
python scripts/g1_agent_sim.py \
  --config runs/g1_research/your_validated_experiment.json \
  --log runs/g1_monitor_run1/events.jsonl \
  --headless --monitor --monitor-port 8766
```

`your_validated_experiment.json` 是需要替换的配置路径，**不是随仓库提供的现成文件**；`configs/experiments/g1_closed_loop.example.json` 仍包含必须替换的模型工厂/检查点，不可直接声称已能运行。Unitree ONNX 是低层行走控制器，并不是高层世界模型。

启动后在终端输入 `x y [yaw_degrees]` 和 `quit`，多个目标共用一个会话。单目标可增加 `--goal "1 0" --monitor-keep-open`：目标完成后关闭仿真物理与渲染器，服务保留最后一帧，状态为“来源已停止”，Ctrl+C 关闭服务。无 `--monitor-keep-open` 的单目标保留原有结束行为。浮动 G1 发布端已通过实际 MuJoCo + ONNX 低层策略帧采集测试；本客户端验收不重新证明高层行走预测器的精度。

## 4. 页面功能

| 页面 | 信息与使用方法 |
| --- | --- |
| 实时工作台 | 服务连接与帧来源状态、回合/步号/仿真时间、Agent 目标/子目标/候选/预算、世界模型版本、校准语义、候选评估、最近反馈 |
| 预测与反馈 | 手动选择 `prediction_*.npz`，并排查看初始真实图像、每一步预测图像、实际终端图像；滑动预测步；查看状态残差和校准界、执行前缀变化 |
| 事件与工件 | 按事件/回合/步号/文本筛选，点击记录读取原始证据及关联工件；查看 manifest、任务 JSON、训练报告、history 训练曲线 |
| 实验对比 | 选择两组实验，比较已记录的提交、种子、状态计数、计划时延、预测残差等描述性信息；不自动执行显著性检验 |

“跟随最新记录”只控制页面选中记录，不暂停实验。服务已连接不等于机器人在执行；当前任务成功也不等于实时帧仍在产生。回合/步号/仿真时间均一致才标为精确对齐；缺少时间或段内画面明确提示。只保存终端实测图像的工件，不会伪造中间步实测值。旧工件缺少关联标识时显示“旧工件未记录对应关系”。

## 5. 数据、协议与架构

```text
仿真所有者线程 → FramePublisher → 有界 RGB 缓冲 → 本机 HTTP 帧接口 → 浏览器
Agent / 实验日志 → events.jsonl → 增量读取 / SSE → 浏览器任务与事件视图
已有 manifest / task / prediction / history → 校验后的工件 API → 历史证据视图
```

`src/wmal/monitor/` 包含数据契约、帧缓冲、目录发现、日志读取、状态投影、只读服务、发布端与静态页面。新增视觉日志记录 task/episode/step、仿真时间、wall-clock 时间、decision_id 和预测工件关联；预测 NPZ 添加对应回合/动作前后步号，但保留原有训练/预测数组。

支持目录中的 `events.jsonl`、`run_manifest.json`、`manifest.json`、`task_*.json`、`prediction_*.npz`、`training_report.json`、`history.json`、`results.json` 与图片。G1 benchmark 同目录的 `*-seed*.jsonl` 以独立日志入口列出，共享目录工件；目录入口不会合并不同日志或不同种子。支持包裹式 `{event,wall_time_utc,payload}` 和原有顶层式视觉日志。

只读 API：`/api/runs`、`/api/runs/{id}/events?after=...`、`/stream`、`/artifacts`、`/api/live/{id}/status`、`/frame.png`。没有动作控制 API。只绑定 `127.0.0.1`，校验 Host/Origin，限制工件目录与符号链接，NPZ 禁止对象/pickle，文本不作为 HTML 执行，对凭证/提示词字段脱敏。不要用反向代理将本服务暴露到公网；不提供远程身份认证。

## 6. 性能与能力边界

- 画面默认 320×240、最多约 5 帧/秒，动作边界 G1 帧可额外发布；与世界模型输入分辨率分离。渲染在仿真线程中进行，有额外计算耗时，不保证硬实时控制。
- 帧只在内存中保留最新两帧，满时覆盖旧帧；浏览器慢/断开不会等待消费者，关闭服务不改变动作或物理状态。连续三次捕获错误后禁用画面，事件日志仍可浏览。
- 没有发布端、画面超过 4 秒未更新、来源已停止分别明确显示。历史查看不是完整视频回放；退出实验后未持久化的实时画面不能恢复。
- 服务保留最近 32 个事件源，每源最多 5,000 条事件，时间线展示筛选后的末 250 条。指标也是**保留窗口的描述性指标**，不能作为完整论文统计。完整结果以原始日志和已有评估程序为准。
- 校准界、ensemble spread 与失败概率含义不同；缺少校准就显示缺失，不给出未经依据的“置信度百分比”。
- 目前客户端不训练任何模型、不调用新的 LLM，不修改既有 Agent 决策；未启用 `--monitor` 时不新增画面渲染。

## 7. 验证

```bash
conda activate wmal
MUJOCO_GL=egl PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/check_scaffold.py
git diff --check
```

包含协议/内存覆盖、日志追加/部分行/损坏/轮转、目录越界/符号链接、只读 HTTP/SSE、预测工件、缺失字段、仿真开启/关闭监控结果一致及断开监控后继续执行等测试。浏览器检查使用了已有真实视觉模型的 A3 实验与历史工件，并验证连续目标后的事件更新和预测图片加载。
