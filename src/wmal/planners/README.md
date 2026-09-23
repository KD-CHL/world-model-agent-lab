# 规划模块

- `sequence_optimizer.py`：有界连续动作序列的交叉熵分布搜索器。
- `world_planner.py`：用状态动力学的成员模型滚动候选动作，按目标距离与预测差异排序；输出首步命令、一步预测、终点预测和模型成员差异。

机器人只执行首步，再观察和重规划。ensemble spread 是成员间差异，没有校准为任务失败概率。视觉目标图像规划接口位于 `models/visual_prediction.py`，待连接具体视觉模型插件和相机数据通路。
