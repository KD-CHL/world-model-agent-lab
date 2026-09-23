"""Fit the action-conditioned joint model on train episodes only."""
import argparse
import json
from wmal.training.trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--members', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.checkpoint, members=args.members, seed=args.seed)))

if __name__ == "__main__":
    main()
