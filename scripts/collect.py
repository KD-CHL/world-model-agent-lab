"""Collect MuJoCo joint transitions with episode-level data splits."""
import argparse
from wmal.training.collector import collect

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--steps-per-episode', type=int, default=20)
    parser.add_argument('--duration-s', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(collect(args.config, args.output, episodes=args.episodes,
                  steps_per_episode=args.steps_per_episode, duration_s=args.duration_s, seed=args.seed))

if __name__ == "__main__":
    main()
