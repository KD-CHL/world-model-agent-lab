# Third-party research references and bundled asset notices

Reviewed 2026-09-24. Repository code below is cited as technical reference unless the row explicitly states that an asset is bundled. No UnifoLM weights or datasets are bundled.

| Project | Repository | Upstream code license | Use in this project |
| --- | --- | --- | --- |
| Continuous-control world model | https://github.com/nicklashansen/tdmpc2 | MIT | Architecture and planning interface reference |
| Modular model-based control toolbox | https://github.com/facebookresearch/mbrl-lib | MIT; repository archived | Separation of model, rollout and trajectory search |
| Visual predictive world models | https://github.com/facebookresearch/jepa-wms | CC BY-NC 4.0 | Feature prediction interface reference only; no code or weights bundled |
| Goal-image predictive planning | https://github.com/gaoyuezhou/dino_wm | MIT | Goal-image feature planning interface reference |
| Unitree world-model-action (UnifoLM-WMA) | https://github.com/unitreerobotics/unifolm-world-model-action | CC BY-NC-SA 4.0 | Training/data/inference workflow reference only; no source or weights copied |
| Unitree vision-language-action (UnifoLM-VLA) | https://github.com/unitreerobotics/unifolm-vla | No root LICENSE file found during review | Training/dataset workflow reference only; verify code, model and dataset terms before use |
| Unitree MuJoCo G1 29-DOF model | https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/g1 | BSD 3-Clause; license copy in `robots/assets/unitree_g1/LICENSE` | Bundled source model and its 36 referenced STL meshes; project derivative removes the floating base joint and adds a floor, light, and `pack_camera`; see `robots/assets/unitree_g1/README.md` |
| Unitree RL Mjlab G1 model and velocity policy | https://github.com/unitreerobotics/unitree_rl_mjlab/tree/1425b15f73bd4095f0df53709d7c389c3eb9e790 | Apache-2.0; license copy in `robots/assets/unitree_g1_mjlab/LICENSE` | Bundled G1 29-DOF MJCF/meshes and published velocity `policy.onnx`; model-derived collision geometry and deployment gains are used by the local MuJoCo walking adapter |

This list describes upstream repository code licenses only. Checkpoint, dataset, submodule, and base-encoder licenses may have separate terms; verify those before downloading or redistributing them. Project modules were implemented against local interfaces and should not be represented as upstream reproductions.
