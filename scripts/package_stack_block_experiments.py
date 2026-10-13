"""Package local stack-block experiment artifacts for a GitHub release.

The archives are deliberately kept below GitHub's 2 GiB per-asset limit.
Only files under ``runs/stack_block*`` are included; the manifest records a
SHA-256 digest for every original file and archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo


MAX_CONTENT_BYTES = 1_500_000_000
CHUNK_BYTES = 1024 * 1024


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(root: Path, output: Path, manifest_path: Path) -> dict:
    files = sorted(
        path for directory in root.glob("runs/stack_block*")
        if directory.is_dir()
        for path in directory.rglob("*")
        if path.is_file()
    )
    files.extend(sorted(path for path in root.glob("runs/stack_block*") if path.is_file()))
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValueError("No runs/stack_block* artifacts found")

    output.mkdir(parents=True, exist_ok=True)
    groups: list[list[Path]] = [[]]
    group_bytes = 0
    for path in files:
        size = path.stat().st_size
        if size > MAX_CONTENT_BYTES:
            raise ValueError(f"Artifact exceeds archive target: {path}")
        if groups[-1] and group_bytes + size > MAX_CONTENT_BYTES:
            groups.append([])
            group_bytes = 0
        groups[-1].append(path)
        group_bytes += size

    entries: list[dict] = []
    archives: list[dict] = []
    for number, group in enumerate(groups, 1):
        archive_path = output / f"stack-block-experiments-20261013-part{number:02d}.zip"
        with ZipFile(archive_path, "w", compression=ZIP_STORED, allowZip64=True) as archive:
            for path in group:
                relative = path.relative_to(root).as_posix()
                info = ZipInfo(relative, date_time=(2026, 10, 13, 0, 0, 0))
                info.compress_type = ZIP_STORED
                info.external_attr = 0o644 << 16
                digest = hashlib.sha256()
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                        target.write(chunk)
                        digest.update(chunk)
                entries.append({"path": relative, "size_bytes": path.stat().st_size,
                                "sha256": digest.hexdigest(), "archive": archive_path.name})
        archives.append({"name": archive_path.name, "size_bytes": archive_path.stat().st_size,
                         "sha256": digest_file(archive_path), "file_count": len(group)})
        print(f"Packaged {archive_path.name}: {len(group)} files, {archive_path.stat().st_size} bytes", flush=True)

    manifest = {
        "schema": "wmal.stack_block_experiment_release.v1",
        "date": "2026-10-13",
        "status": "development evidence; model and recovery policy not frozen",
        "source_pattern": "runs/stack_block*",
        "file_count": len(entries),
        "total_source_bytes": sum(entry["size_bytes"] for entry in entries),
        "archives": archives,
        "files": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / manifest_path.name).write_bytes(manifest_path.read_bytes())
    print(f"Manifest: {manifest_path} ({len(entries)} files)", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/stack_block_experiments_20261013.json"))
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = arguments.output.resolve()
    manifest = arguments.manifest
    if not manifest.is_absolute():
        manifest = root / manifest
    package(root, output, manifest)


if __name__ == "__main__":
    main()
