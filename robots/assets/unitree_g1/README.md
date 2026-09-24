# Unitree G1 29-DOF MuJoCo asset

- Upstream: [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
- Revision: `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`
- Source model: `unitree_robots/g1/g1_29dof.xml`
- License: BSD 3-Clause; see [LICENSE](LICENSE).
- Included meshes: the 36 STL files referenced by the source model.

`g1_29dof_fixed_base.xml` is a project derivative. It removes the upstream
`floating_base_joint` and adds a ground plane, lighting, and a fixed RGB camera
named `pack_camera`. It is a fixed-base manipulation/test rig, not a dynamically
balanced humanoid. Its 29 actuated joints are simulated; the root pelvis is
immobile. The upstream rubber-hand meshes are visual only: there are no finger
or gripper actuators. `pack_camera` is a world-mounted target-view camera for
rendering and interface tests; its extrinsics are not calibrated to the
published pack-camera dataset. Walking, balance, contact-rich grasping, and the
UnifoLM dataset-to-actuator mapping have not been validated by this asset.

To reconstruct the local asset files, check out the revision above and copy the
source XML, its referenced meshes, and its `LICENSE`; then apply the documented
derivative changes to `g1_29dof.xml`.
