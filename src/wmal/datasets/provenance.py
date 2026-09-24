"""Dataset provenance records and deterministic file fingerprints."""
from pathlib import Path

from wmal.logging.manifest import atomic_json, sha256_file


def file_record(path: str | Path, *, hash_content: bool = False) -> dict:
    """Return stable file metadata; hash large episode files only when requested."""
    source = Path(path)
    record = {"path": source.as_posix(), "size_bytes": source.stat().st_size}
    if hash_content:
        record["sha256"] = sha256_file(source)
    return record


def write_manifest(manifest: dict, path: str | Path) -> Path:
    """Atomically persist a JSON-serializable dataset manifest."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest")
    destination = Path(path)
    atomic_json(destination, manifest)
    return destination
