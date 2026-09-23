"""Command-line entry points for model training."""
import argparse
import json


def train_neural_cli():
    from wmal.training.neural_trainer import train_neural
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--members', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    report = train_neural(args.dataset, args.checkpoint, members=args.members,
                          epochs=args.epochs, batch_size=args.batch_size,
                          learning_rate=args.learning_rate, seed=args.seed,
                          device=args.device)
    print(json.dumps(report))
