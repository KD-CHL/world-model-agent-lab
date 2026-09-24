# evaluation

Evaluation utilities keep success, numerical dynamics, and visual prediction metrics separate.

- `runner.py`：预留 runner 模块；接口与验收见 docs 文档。
- `success_detector.py`：预留 success_detector 模块；接口与验收见 docs 文档。
- `perturbations.py`：预留 perturbations 模块；接口与验收见 docs 文档。
- `video_prediction.py`：显式 provider 插件的 held-out RGB 序列 MAE/PSNR，保存汇总和逐样本指标；这些是图像指标，不代表任务成功或仿真真值。
