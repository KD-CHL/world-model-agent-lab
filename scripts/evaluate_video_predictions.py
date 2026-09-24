"""Evaluate an explicit visual-model plugin on held-out samples."""
import argparse
import json
from wmal.communication.plugins import load_factory
from wmal.evaluation.video_prediction import evaluate_video_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", required=True, help="Explicit module:factory returning a VideoPredictionProvider")
    parser.add_argument("--dataset", required=True, help="Held-out data consumed by the selected plugin")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    provider = load_factory(args.plugin, dataset_path=args.dataset)
    print(json.dumps(evaluate_video_predictions(provider, args.dataset, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
