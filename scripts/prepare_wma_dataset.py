"""Validate or explicitly download/convert one LeRobot V2.1 dataset for WMA."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from wmal.datasets.lerobot import inspect_lerobot_v21
from wmal.datasets.provenance import write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Dataset root with meta/, data/, and videos/ folders")
    parser.add_argument("--dataset-id", required=True, help="Hugging Face dataset repo ID")
    parser.add_argument("--revision", required=True, help="Pinned tag or commit revision")
    parser.add_argument("--license-id", required=True, help="Dataset license identifier or explicit unresolved label")
    parser.add_argument("--camera-key", default="observation.images.top")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", help="Manifest output; default <root>/wmal_manifest.json")
    parser.add_argument("--hash-episode-files", action="store_true", help="Stream SHA256 across all parquet/video data")
    parser.add_argument("--download", action="store_true", help="Explicitly download this revision with Hugging Face Hub")
    parser.add_argument("--convert", action="store_true", help="Run the converter in --wma-root after validation")
    parser.add_argument("--wma-root", help="External unifolm-world-model-action checkout")
    parser.add_argument("--prepared-root", help="Output root for WMA prepared data")
    parser.add_argument("--robot-name", default="Unitree G1 with gripper")
    args = parser.parse_args()

    dataset_root = Path(args.root).expanduser().resolve()
    if args.download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            parser.error("Install project extra: python -m pip install -e '.[robot-data]'")
        dataset_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=args.dataset_id, repo_type="dataset", revision=args.revision,
                          local_dir=str(dataset_root))
    manifest = inspect_lerobot_v21(
        dataset_root, dataset_id=args.dataset_id, revision=args.revision,
        license_id=args.license_id, seed=args.seed, camera_key=args.camera_key,
        hash_episode_files=args.hash_episode_files)
    manifest_path = Path(args.manifest).expanduser() if args.manifest else dataset_root / "wmal_manifest.json"
    write_manifest(manifest, manifest_path)

    conversion = None
    if args.convert:
        if not args.wma_root or not args.prepared_root:
            parser.error("--convert requires --wma-root and --prepared-root")
        wma_root = Path(args.wma_root).expanduser().resolve()
        converter = wma_root / "prepare_data" / "prepare_training_data.py"
        if not converter.is_file():
            parser.error("WMA converter not found under --wma-root")
        conversion = [sys.executable, str(converter), "--source_dir", str(dataset_root.parent),
                     "--target_dir", str(Path(args.prepared_root).expanduser().resolve()),
                     "--dataset_name", dataset_root.name, "--robot_name", args.robot_name]
        subprocess.run(conversion, cwd=wma_root, check=True)
    print(json.dumps({"manifest": str(manifest_path), "episodes": len(manifest["episodes"]),
                      "splits": manifest["split_counts"], "conversion_command": conversion},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
