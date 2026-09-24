# Third-party research references

Reviewed 2026-09-23. The listed projects are cited as technical references. No source files, pretrained weights, or datasets from these repositories were copied into this repository by the module integration change.

| Project | Repository | Upstream code license | Use in this project |
| --- | --- | --- | --- |
| Continuous-control world model | https://github.com/nicklashansen/tdmpc2 | MIT | Architecture and planning interface reference |
| Modular model-based control toolbox | https://github.com/facebookresearch/mbrl-lib | MIT; repository archived | Separation of model, rollout and trajectory search |
| Visual predictive world models | https://github.com/facebookresearch/jepa-wms | CC BY-NC 4.0 | Feature prediction interface reference only; no code or weights bundled |
| Goal-image predictive planning | https://github.com/gaoyuezhou/dino_wm | MIT | Goal-image feature planning interface reference |
| Unitree world-model-action (UnifoLM-WMA) | https://github.com/unitreerobotics/unifolm-world-model-action | CC BY-NC-SA 4.0 | Training/data/inference workflow reference only; no source or weights copied |
| Unitree vision-language-action (UnifoLM-VLA) | https://github.com/unitreerobotics/unifolm-vla | No root LICENSE file found during review | Training/dataset workflow reference only; verify code, model and dataset terms before use |

This list describes upstream repository code licenses only. Checkpoint, dataset, submodule, and base-encoder licenses may have separate terms; verify those before downloading or redistributing them. Project modules were implemented against local interfaces and should not be represented as upstream reproductions.
