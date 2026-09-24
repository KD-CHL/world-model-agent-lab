# Unitree RL Mjlab G1 assets

`xmls/g1.xml` and its `xmls/assets/` mesh files are from Unitree Robotics' `unitree_rl_mjlab` repository at commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`. The Apache-2.0 license is copied as `LICENSE`.

The upstream G1 policy configuration expects this floating-base model's collision geometry. The local runner adds a flat ground plane and 29 position actuators at runtime using the gains, action scales, default joint pose, and effort limits from Unitree's `deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml` and G1 actuator definitions. The corresponding published ONNX policy is `models/unitree_g1_velocity_policy.onnx`.

The model source is bundled for a repeatable local MuJoCo test; it does not replace the separate fixed-base G1 asset or establish sim-to-real safety. Run instructions and validation scope are in [the local walking guide](../../../docs/16_g1_local_walking.md).
