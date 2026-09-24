# datasets

`trajectory.py` 提供依赖无关的 `Transition` 与 JSONL 读写格式。它是
LeRobot、robomimic、MuJoCo 采样器等外部数据源的适配目标，不会把第三方
数据集实现或运行时依赖复制进项目。
- `lerobot.py`：只读校验 WMA 预期的 LeRobot V2.1 episode parquet、相机视频、state/action 维数与顺序，并生成 episode 级 split 和来源 manifest。
- `splits.py`：按固定 seed 对完整 episode 确定性划分，避免相邻帧跨 split。
- `provenance.py`：生成文件大小/可选 SHA256 及原子 JSON manifest。

`scripts/prepare_wma_dataset.py` 默认只检查本地数据；只有显式给出 pinned revision 并启用 `--download` 时才下载。原始数据和转换产物不应提交进 Git。
