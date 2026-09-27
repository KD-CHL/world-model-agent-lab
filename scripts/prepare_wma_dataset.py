"""Validate or explicitly download/convert one LeRobot V2.1 dataset for WMA."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from wmal.datasets.lerobot import inspect_lerobot_v21
from wmal.datasets.provenance import write_manifest


def finalize_wma_conversion(prepared_root, dataset_name, camera_key, manifest):
    """Keep one camera and repair the upstream converter's skipped non-AV1 videos."""
    prepared_root = Path(prepared_root).expanduser().resolve()
    csv_path = prepared_root / f"{dataset_name}.csv"
    camera_dir = prepared_root / "videos" / dataset_name / camera_key
    transition_dir = prepared_root / "transitions" / dataset_name
    stats_path = transition_dir / "meta_data" / "stats.safetensors"
    if not csv_path.is_file() or not stats_path.is_file():
        raise ValueError("WMA conversion did not produce its CSV metadata and stats file")

    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "data_dir" not in fieldnames or "videoid" not in fieldnames:
        raise ValueError("WMA converter CSV is missing data_dir/videoid columns")
    selected_rows = [row for row in rows if row.get("data_dir") == f"{dataset_name}/{camera_key}"]
    episode_ids = {int(item["episode_index"]) for item in manifest["episodes"]}
    selected_ids = [int(row["videoid"]) for row in selected_rows]
    if set(selected_ids) != episode_ids or len(selected_ids) != len(episode_ids):
        raise ValueError("WMA CSV does not contain exactly one row for every validated episode and selected camera")

    camera_dir.mkdir(parents=True, exist_ok=True)
    source_videos = {int(item["episode_index"]): Path(item["video"]["path"])
                     for item in manifest["episodes"]}
    for episode_id in sorted(episode_ids):
        destination = camera_dir / f"{episode_id}.mp4"
        transition = transition_dir / f"{episode_id}.h5"
        if not transition.is_file() or transition.stat().st_size == 0:
            raise ValueError(f"WMA conversion is missing episode transition data: {episode_id}.h5")
        if not destination.is_file() or destination.stat().st_size == 0:
            shutil.copy2(source_videos[episode_id], destination)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise ValueError(f"WMA conversion is missing camera video: {episode_id}.mp4")

    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                     dir=csv_path.parent, prefix=csv_path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Dataset root with meta/, data/, and videos/ folders")
    parser.add_argument("--dataset-id", required=True, help="Hugging Face dataset repo ID")
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face commit SHA (not main or a movable tag)")
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

    # A provenance manifest is only reproducible when the downloaded source is
    # immutable. Hugging Face exposes commit hashes as 40 hexadecimal chars.
    if len(args.revision) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in args.revision):
        parser.error("--revision must be an immutable 40-character Hugging Face commit SHA")

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
        finalize_wma_conversion(args.prepared_root, dataset_root.name, args.camera_key, manifest)
    print(json.dumps({"manifest": str(manifest_path), "episodes": len(manifest["episodes"]),
                      "splits": manifest["split_counts"], "conversion_command": conversion},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
