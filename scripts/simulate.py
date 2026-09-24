"""Run a bounded MuJoCo simulation or inspect it in the interactive viewer."""
import argparse
import json
import math
from pathlib import Path
import time

from wmal.communication.contracts import RobotProfile
from wmal.envs.keyboard_control import KeyboardJointControl
from wmal.envs.mujoco_backend import MujocoBackend


def parse_joint_target(value):
    name, separator, raw_target = value.partition('=')
    if not separator or not name:
        raise argparse.ArgumentTypeError('Expected JOINT=RADIANS')
    try:
        target = float(raw_target)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Joint target must be a number') from exc
    if not math.isfinite(target):
        raise argparse.ArgumentTypeError('Joint target must be finite')
    return name, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Robot JSON configuration')
    parser.add_argument('--steps', type=int, default=1000, help='Physics steps for headless run (1..500000)')
    parser.add_argument('--joint', action='append', type=parse_joint_target, default=[],
                        metavar='NAME=RAD', help='Optional joint-position target; may be repeated')
    parser.add_argument('--viewer', action='store_true', help='Open interactive MuJoCo viewer')
    parser.add_argument('--camera', help='Lock viewer to this named camera (free orbit is the default)')
    args = parser.parse_args()
    if not 1 <= args.steps <= 500_000:
        parser.error('--steps must be between 1 and 500000')

    config_path = Path(args.config).expanduser().resolve()
    try:
        config = json.loads(config_path.read_text())
        model_path = (config_path.parent / config['model_path']).resolve()
        profile = RobotProfile(**config['profile'])
        backend = MujocoBackend(str(model_path), profile, config['actuator_map'],
                                locomotion=None)
        targets = dict(backend.observe().joints)
        seen = set()
        for name, target in args.joint:
            if name in seen:
                raise ValueError('Duplicate joint target: ' + name)
            seen.add(name)
            targets[name] = target
        if args.joint:
            profile.validate_targets({name: target for name, target in args.joint})
            duration_s = (max(1.0, float(args.steps) * backend.model.opt.timestep)
                          if args.viewer else float(args.steps) * backend.model.opt.timestep)
            command = backend.interface.joint_command(backend.observe(), targets,
                                                      duration_s=min(duration_s, profile.max_duration_s))
            backend.begin(command)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.viewer:
        try:
            import mujoco
            import mujoco.viewer
            camera_name = args.camera
            if camera_name:
                camera_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                if camera_id < 0:
                    parser.error('Configured camera not found: ' + camera_name)
            joint_control = KeyboardJointControl(backend, initial_joint='left_shoulder_pitch_joint'
                                                 if 'left_shoulder_pitch_joint' in profile.joint_limits else None)
            print('Viewer keys: N/P select joint, [/] decrease/increase target by 0.1 rad, '
                  '0 holds the current joint. Close the viewer to exit.', flush=True)
            print(f'Selected {joint_control.selected_joint}; target='
                  f'{joint_control.requested_targets[joint_control.selected_joint]:.3f} rad', flush=True)
            with mujoco.viewer.launch_passive(backend.model, backend.data,
                                              key_callback=joint_control.on_key) as viewer:
                if camera_name:
                    if camera_id >= 0:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = camera_id
                while viewer.is_running():
                    for message in joint_control.update():
                        print(message, flush=True)
                    backend.step()
                    viewer.sync()
                    time.sleep(float(backend.model.opt.timestep))
        except (RuntimeError, ImportError) as exc:
            parser.error('Could not open MuJoCo viewer: ' + str(exc))
        finally:
            backend.stop()
        return

    try:
        for _ in range(args.steps):
            backend.step()
    finally:
        backend.stop()
    observation = backend.observe()
    print(f"MuJoCo simulation complete: {len(observation.joints)} joints, "
          f"{args.steps} steps, {observation.sim_time_s:.3f} simulated seconds")
    if args.joint:
        for name, _ in args.joint:
            print(f"{name}={observation.joints[name]:.4f} rad")


if __name__ == '__main__':
    main()
