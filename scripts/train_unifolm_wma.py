"""Build a reproducible WMA fine-tuning config and optionally launch training."""
import argparse
import json
from pathlib import Path

from wmal.training.unifolm_wma import build_wma_config, launch_wma_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--template", help="Default: <upstream-root>/configs/train/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Immutable WMA-0 Base checkpoint")
    parser.add_argument("--prepared-data", required=True)
    parser.add_argument("--dataset-key", required=True, help="Dataset key registered in WMA's loader")
    parser.add_argument("--mode", choices=("decision", "joint"), required=True)
    parser.add_argument("--config-output", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=12366)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--dry-run", action="store_true", help="Print launch metadata without training")
    args = parser.parse_args()

    root = Path(args.upstream_root).expanduser().resolve()
    template = Path(args.template).expanduser() if args.template else root / "configs/train/config.yaml"
    config = build_wma_config(template, checkpoint=args.checkpoint, prepared_data=args.prepared_data,
                              dataset_key=args.dataset_key, mode=args.mode, output_path=args.config_output)
    result = launch_wma_training(root, config, run_name=args.run_name, output_dir=args.run_output,
                                processes=args.processes, master_port=args.master_port,
                                dry_run=args.dry_run, cuda_visible_devices=args.cuda_visible_devices)
    print(json.dumps({**result, "generated_config": str(config)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
