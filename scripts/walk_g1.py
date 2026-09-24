"""Safely run the Unitree pretrained G1 velocity policy in local MuJoCo."""
import argparse
import json

from wmal.envs.g1_locomotion import simulate_g1_walk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=10,
                        help='Measured alternating foot touchdowns to complete (1..100)')
    parser.add_argument('--velocity-x', type=float, default=0.2,
                        help='Forward velocity command in m/s (0.05..0.5)')
    parser.add_argument('--max-duration', type=float, default=12.0,
                        help='Safety timeout in simulated seconds')
    parser.add_argument('--viewer', action='store_true',
                        help='Show the walk in a MuJoCo window at real-time playback speed')
    args = parser.parse_args()
    try:
        result = simulate_g1_walk(args.steps, args.velocity_x, args.max_duration, args.viewer)
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'G1 walk failed safely: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
