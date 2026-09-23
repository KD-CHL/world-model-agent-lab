"""Small, provider-neutral metadata and atomic JSON output for experiments."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import platform
import subprocess
import tempfile


def related_path(output, label):
    path = Path(output)
    return path.with_name(path.stem + '.' + label + '.json')


def sha256_file(path):
    digest = sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=destination.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        temporary = handle.name
    os.replace(temporary, destination)


def build_manifest(kind, seed, files, parameters=None, **extra):
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                  timeout=2, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                                    capture_output=True, text=True, timeout=2, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision = None
        dirty = None
    try:
        import mujoco
        mujoco_version = mujoco.__version__
    except ImportError:
        mujoco_version = None
    return {'schema_version': 1, 'run_kind': kind,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': revision, 'git_dirty': dirty, 'python_version': platform.python_version(),
            'platform': platform.platform(), 'mujoco_version': mujoco_version,
            'seed': seed, 'parameters': parameters or {},
            'inputs': {name: {'path': str(path), 'sha256': sha256_file(path)} for name, path in files.items()},
            **extra}
