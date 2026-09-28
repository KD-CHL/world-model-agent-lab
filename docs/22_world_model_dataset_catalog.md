# 世界模型数据清单

核查日期：2026-09-28。规模来自固定 revision 的 meta/info.json，而非可能滞后的网页预览。完整 SHA、features 和访问状态见 data/manifests/research_dataset_audit_20260928.json。本次仅查询元数据，未下载新增大数据。

| 数据集 | episodes / frames | 格式 |
|---|---:|---|
| [G1_Dex1_MountCameraRedGripper_Dataset](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_MountCameraRedGripper_Dataset) | 201 / 172649 | v2.1 |
| [G1_Dex1_Stack_Block](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Stack_Block) | 690 / 284191 | v3.0 |
| [G1_Dex1_Bag_Insert](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Bag_Insert) | 200 / 136473 | v2.1 |
| [G1_Dex1_Erase_Board](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Erase_Board) | 200 / 127545 | v2.1 |
| [G1_Clean_Table](https://huggingface.co/datasets/unitreerobotics/G1_Clean_Table) | 未确认：HTTP 401 | 未知 |
| [G1_Dex1_Pack_PencilBox](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Pack_PencilBox) | 200 / 162762 | v2.1 |
| [G1_Dex1_Pour_Medicine](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Pour_Medicine) | 200 / 158476 | v2.1 |
| [G1_Dex1_Pack_PingPong](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Pack_PingPong) | 200 / 160593 | v2.1 |
| [G1_Dex1_Prepare_Fruit](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Prepare_Fruit) | 200 / 123617 | v2.1 |
| [G1_Dex1_Organize_Tools](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Organize_Tools) | 200 / 182638 | v2.1 |
| [G1_Dex1_Fold_Towel](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Fold_Towel) | 200 / 310671 | v2.1 |
| [G1_Dex1_Wipe_Table](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Wipe_Table) | 200 / 102695 | v3.0 |
| [G1_Dex1_DualRobot_Clean_Table](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_DualRobot_Clean_Table) | 200 / 171312 | v2.1 |
| [G1_Dex1_DiverseManip_DualArm_256x256](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_DiverseManip_DualArm_256x256) | 525 / 413538 | v3.0 |
| [G1_Dex1_DiverseManip_DualArm_128x128](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_DiverseManip_DualArm_128x128) | 525 / 413538 | v3.0 |
| [G1_Dex1_DiverseManip_SingleArm_256x256](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_DiverseManip_SingleArm_256x256) | 468 / 331555 | v3.0 |
| [G1_Dex1_DiverseManip_SingleArm_128x128](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_DiverseManip_SingleArm_128x128) | 468 / 331555 | v3.0 |

除 Clean Table 未确认条目外，上表匿名元数据读取成功，API 标为 ungated，卡片 license 为 apache-2.0；这不代替实际下载完整性验证。

## 优先顺序

- P0：已有 Pack Camera，先验证预测模式、输入动作语义和帧对齐。
- P1：Pack PencilBox、Pack PingPong、Organize Tools，先选一个，再扩到2—3个任务族；Stack Block 作为接触难度扩展。
- P2：DiverseManip 选择一个分辨率用于数据规模实验；Prepare Fruit、Erase Board、Wipe Table 用于后续泛化。
- P3：Bag Insert、Fold Towel、Pour Medicine、DualRobot，柔性、颗粒或协同控制增加干扰，首篇论文暂缓。
- Clean Table：原链接及 G1_Dex1_Clean_Table 匿名访问均返回401。不能断言已下架或私有，只能标为本次未确认可获取。

## 必须处理的兼容问题

1. 本地 Pack Camera 为 v2.1，合并 state/action 各16维，201 episodes、172649帧、30 fps、四个480×640视角。路径为 `data/raw/lerobot/unitree_g1_pack_camera/`。
2. Pack PencilBox 等 v2.1 数据使用 observation.left_arm/right_arm/left_gripper/right_gripper 等分列。7+7+1+1只说明可组成16维，不能证明与旧数据控制语义相同；需核实关节顺序、绝对/增量目标、夹爪量程和单位。
3. Stack Block、Wipe Table 的当前 v3 字段变为 observation.state.left_arm 等，视频存储布局也不能按 v2.1 假设。DiverseManip 当前版本同样是 v3。需锁定经过验证的旧版本或新增转换器。
4. 元数据同时存在CHW/HWC图像shape；应实际解码确认，不能直接用shape推断图像布局。选择统一主视角；禁止把同步多相机画面作为独立测试数据。
5. Stack Block网页预览与当前meta可能不同；本表采用固定revision中的690 episodes，下载应使用清单SHA。
6. DualRobot旧数据只显示一组左右臂字段，不能凭名称推断完整双机器人同步动作。另发现 [Left](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Two_Robot_Clean_Table_Left) 和 [Right](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Two_Robot_Clean_Table_Right)：各200 episodes，198824/198003帧，均v3；license元数据缺失，配对同步未核实。暂不纳入主实验。
7. DiverseManip的128与256版本计数相同，应视作潜在同源数据，核对轨迹ID与内容哈希后分组，不能跨训练/测试划分。

## 其他来源

| 来源 | 用途与限制 |
|---|---|
| [Z1 StackBox](https://huggingface.co/datasets/unitreerobotics/Z1_StackBox_Dataset/tree/v2.1) | 上游复现实验；不是G1动作 |
| [Z1 Dual StackBox](https://huggingface.co/datasets/unitreerobotics/Z1_Dual_Dex1_StackBox_Dataset/tree/v2.1) | 双臂操作参考 |
| [Z1 Dual StackBox V2](https://huggingface.co/datasets/unitreerobotics/Z1_Dual_Dex1_StackBox_Dataset_V2/tree/v2.1) | 上游训练来源 |
| [Z1 CleanupPencils](https://huggingface.co/datasets/unitreerobotics/Z1_Dual_Dex1_CleanupPencils_Dataset/tree/v2.1) | 整理技能参考 |
| [Open X-Embodiment](https://robotics-transformer-x.github.io/) | Base已有其先验；首阶段不重做跨本体预训练 |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 成熟操作基准备选；若采用必须明确更换机器人本体 |
| [RoboCasa](https://github.com/robocasa/robocasa) | 多步骤日常操作与仿真扩展 |
| 本项目G1新采集 | 同初态多候选、失败、扰动、恢复轨迹；目前尚未拥有完整操作数据 |

Z1四项来自上游README，本次未逐文件审计；外部基准按官方方式获取，分别核查许可、版本与动作接口。不是所有机器人演示都可直接混合训练。

## 下载与存储

示例只下载，不完成分列适配与训练转换：

```bash
hf download unitreerobotics/G1_Dex1_Pack_PencilBox \
  --repo-type dataset --revision a21ef18aeb4538be6190cbb3793c5a9e79213795 \
  --local-dir data/raw/lerobot/g1_pack_pencilbox
```

本次未审计新增数据的精确总字节数，下载前查询文件大小与磁盘空间。建议结构：

```text
data/raw/lerobot/<dataset>/          原始只读数据
data/prepared/wma/<dataset>/<split>/ 转换后隔离数据
data/sim/<task>/<seed>/              同步RGB、状态、动作与结果
data/manifests/                     revision、schema、划分及统计
runs/world_model/<experiment>/      权重、指标与训练配置
```

按episode/采集会话拆分；不同视角、分辨率、近重复演示归为同组。训练集拟合归一化，验证集选参数，独立校准集拟合可信度，测试集只报告结果。本地清单只记录split，现有上游转换/训练链不自动执行这些split，必须真正隔离CSV与统计量。
