"""Safe, read-only discovery and preview of experiment artifacts."""
from hashlib import sha256
import json
import os
import math
from pathlib import Path, PurePosixPath
import struct
import zlib
import zipfile

import numpy as np


_MAX_JSON_BYTES = 5 * 1024 * 1024
_MAX_NPZ_BYTES = 32 * 1024 * 1024
_MAX_NPZ_UNPACKED = 64 * 1024 * 1024
_MAX_FRAME_PIXELS = 2048 * 2048
_ROOT_MARKERS = {'events.jsonl', 'run_manifest.json', 'manifest.json',
                 'training_report.json', 'history.json', 'results.json'}


def _is_artifact_name(name):
    return (name in _ROOT_MARKERS or
            (name.endswith('.jsonl') and not name.startswith('.')) or
            (name.startswith('task_') and name.endswith('.json')) or
            (name.startswith('prediction_') and name.endswith('.npz')) or
            (name.lower().endswith(('.png', '.jpg', '.jpeg')) and not name.startswith('.')))


def _png_chunk(kind, content):
    payload = kind + content
    return struct.pack('>I', len(content)) + payload + struct.pack('>I', zlib.crc32(payload) & 0xffffffff)


def _encode_png(rgb):
    height, width, _ = rgb.shape
    header = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    scanlines = b''.join(b'\x00' + row.tobytes() for row in rgb)
    return (b'\x89PNG\r\n\x1a\n' + _png_chunk(b'IHDR', header)
            + _png_chunk(b'IDAT', zlib.compress(scanlines, level=3))
            + _png_chunk(b'IEND', b''))


class RunCatalog:
    def __init__(self, runs_root):
        root = Path(runs_root).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError('runs_root must be an existing directory')
        self.root = root
        self._runs = {}

    def _relative(self, directory):
        value = directory.relative_to(self.root).as_posix()
        return value or '.'

    def _run_id(self, relative):
        return sha256(relative.encode('utf-8')).hexdigest()[:20]

    def list_runs(self):
        found = {}
        for directory, subdirs, files in os.walk(self.root, followlinks=False):
            base = Path(directory)
            subdirs[:] = [name for name in subdirs
                          if not (base / name).is_symlink() and not name.startswith('.')]
            names = {name for name in files if not (base / name).is_symlink()}
            if not (names & _ROOT_MARKERS or
                    any(name.startswith(('task_', 'prediction_')) or name.endswith('.jsonl') for name in names)):
                continue
            relative = self._relative(base)
            run_id = self._run_id(relative)
            manifest = {}
            manifest_path = base / 'run_manifest.json'
            if (not manifest_path.is_symlink() and manifest_path.is_file()
                    and manifest_path.stat().st_size <= _MAX_JSON_BYTES):
                try:
                    value = json.loads(manifest_path.read_text(encoding='utf-8'))
                    if isinstance(value, dict):
                        manifest = value
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    pass
            found[run_id] = {'run_id': run_id, 'run_path': relative,
                             'directory': relative, 'events_file': 'events.jsonl',
                             'run_kind': manifest.get('run_kind', manifest.get('kind')),
                             'started_at_utc': manifest.get('created_at_utc'),
                             'git_commit': manifest.get('git_commit'),
                             'git_dirty': manifest.get('git_dirty'),
                             'seed': manifest.get('seed'), 'manifest': manifest}
            for name in sorted(names):
                if name == 'events.jsonl' or name.startswith('.') or not name.endswith('.jsonl'):
                    continue
                source_id = self._run_id(relative + '::' + name)
                found[source_id] = {**found[run_id], 'run_id': source_id,
                                    'run_path': relative + '/' + name, 'events_file': name}
        self._runs = found
        return sorted(found.values(), key=lambda row: (row.get('started_at_utc') or '', row['run_path']))

    def resolve_run(self, run_id):
        if run_id not in self._runs:
            self.list_runs()
        if run_id not in self._runs:
            raise KeyError('Unknown run id')
        row = self._runs[run_id]
        path = (self.root / row.get('directory', row['run_path'])).resolve()
        if not path.is_relative_to(self.root) or not path.is_dir():
            raise ValueError('Run path escaped the configured root')
        return path

    def event_log_path(self, run_id):
        run = self.resolve_run(run_id)
        path = run / self._runs[run_id]['events_file']
        if path.is_symlink():
            raise ValueError('Symbolic-link logs are not served')
        return path

    def _safe_file(self, run_id, relative_path):
        if not isinstance(relative_path, str) or '\x00' in relative_path or '\\' in relative_path:
            raise ValueError('Invalid artifact path')
        path_value = PurePosixPath(relative_path)
        if path_value.is_absolute() or not path_value.parts or any(part in ('..', '.') for part in path_value.parts):
            raise ValueError('Invalid artifact path')
        run = self.resolve_run(run_id)
        candidate = run
        for part in path_value.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError('Symbolic-link artifacts are not served')
        target = candidate.resolve()
        if not target.is_relative_to(run) or not target.is_file():
            raise ValueError('Artifact path is outside the run or is not a regular file')
        if not _is_artifact_name(target.name):
            raise ValueError('Unsupported artifact type')
        return target

    def list_artifacts(self, run_id):
        run = self.resolve_run(run_id)
        rows = []
        for path in run.iterdir():
            if path.is_symlink() or not path.is_file() or not _is_artifact_name(path.name):
                continue
            suffix = path.suffix.lower()
            kind = ('events' if suffix == '.jsonl' else
                    'manifest' if path.name in ('run_manifest.json', 'manifest.json') else
                    'prediction' if suffix == '.npz' else
                    'image' if suffix in ('.png', '.jpg', '.jpeg') else 'json')
            info = path.stat()
            rows.append({'relative_path': path.name, 'kind': kind, 'size_bytes': info.st_size,
                         'modified_time': info.st_mtime})
        return sorted(rows, key=lambda row: row['relative_path'])

    def read_json_artifact(self, run_id, relative_path):
        target = self._safe_file(run_id, relative_path)
        if target.suffix.lower() != '.json' or target.stat().st_size > _MAX_JSON_BYTES:
            raise ValueError('Artifact is not a supported-size JSON file')
        try:
            value = json.loads(target.read_text(encoding='utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError('Artifact contains invalid JSON') from exc
        if not isinstance(value, (dict, list)):
            raise ValueError('JSON artifact must contain an object or array')
        return value

    def read_prediction_frame(self, run_id, relative_path, index=0, key='predicted_rgb'):
        target = self._safe_file(run_id, relative_path)
        if target.suffix.lower() != '.npz' or not target.name.startswith('prediction_'):
            raise ValueError('Artifact is not a supported prediction archive')
        if type(index) is not int or index < 0 or key not in ('predicted_rgb', 'before_rgb', 'observed_rgb'):
            raise ValueError('Invalid prediction image selector')
        self._validate_prediction_archive(target)
        try:
            with np.load(target, allow_pickle=False) as archive:
                if key not in archive.files:
                    raise ValueError(f'Prediction image key {key!r} is unavailable')
                frames = np.asarray(archive[key])
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(('Prediction archive', 'Prediction image')):
                raise
            raise ValueError('Unable to safely read prediction archive') from exc
        if frames.dtype.kind not in ('u', 'i', 'f') or not frames.size:
            raise ValueError('Prediction image array must be numeric and nonempty')
        if frames.ndim == 3:
            selected = frames
            if index != 0:
                raise ValueError('Single-frame image index must be zero')
        elif frames.ndim == 4:
            if index >= frames.shape[0]:
                raise ValueError('Prediction frame index is out of range')
            selected = frames[index]
        else:
            raise ValueError('Prediction image must be CHW/HWC or TCHW/THWC')
        if selected.shape[0] == 3:
            selected = selected.transpose(1, 2, 0)
        if (selected.ndim != 3 or selected.shape[2] != 3
                or selected.shape[0] * selected.shape[1] > _MAX_FRAME_PIXELS):
            raise ValueError('Prediction frame has unsupported image dimensions')
        normalized = selected.dtype.kind == 'f'
        selected = selected.astype(np.float32, copy=False)
        if not np.isfinite(selected).all() or selected.min() < 0 or selected.max() > 255:
            raise ValueError('Prediction frame contains invalid pixel values')
        if normalized and selected.max(initial=0) <= 1:
            selected = selected * 255
        return _encode_png(np.rint(selected).astype(np.uint8))

    def _validate_prediction_archive(self, target):
        if target.stat().st_size > _MAX_NPZ_BYTES:
            raise ValueError('Prediction archive exceeds the byte limit')
        try:
            with zipfile.ZipFile(target) as archive:
                members = archive.infolist()
                if (len(members) > 32 or sum(m.file_size for m in members) > _MAX_NPZ_UNPACKED
                        or any('/' in m.filename or '\\' in m.filename
                               or not m.filename.endswith('.npy') for m in members)):
                    raise ValueError('Prediction archive contains unsafe members')
                for member in members:
                    with archive.open(member) as stream:
                        version = np.lib.format.read_magic(stream)
                        if version not in ((1, 0), (2, 0)):
                            raise ValueError('Unsupported NPY version')
                        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                  else np.lib.format.read_array_header_2_0)
                        shape, _, dtype = reader(stream)
                        size = math.prod(shape) * dtype.itemsize
                        if dtype.hasobject or size > member.file_size or size > _MAX_NPZ_UNPACKED:
                            raise ValueError('Unsafe array size or object dtype')
        except (zipfile.BadZipFile, EOFError, TypeError, ValueError) as exc:
            raise ValueError('Prediction archive is malformed or unsafe') from exc

    def read_prediction_info(self, run_id, relative_path):
        target = self._safe_file(run_id, relative_path)
        if target.suffix != '.npz' or not target.name.startswith('prediction_'):
            raise ValueError('Unsupported prediction archive')
        self._validate_prediction_archive(target)
        with np.load(target, allow_pickle=False) as archive:
            available = [key for key in ('before_rgb', 'predicted_rgb', 'observed_rgb') if key in archive.files]
            count = 0
            if 'predicted_rgb' in available:
                shape = archive['predicted_rgb'].shape
                if len(shape) not in (3, 4):
                    raise ValueError('Prediction image must be CHW/HWC or TCHW/THWC')
                count = int(shape[0]) if len(shape) == 4 else 1
                image_shape = shape[-3:]
                if 3 not in (image_shape[0], image_shape[-1]) or not 1 <= count <= 256:
                    raise ValueError('Invalid prediction frame dimensions or count')
            result = {'available_images': available, 'frame_count': count, 'artifact': relative_path}
            for key in ('predicted_state', 'observed_state', 'actions', 'error_bounds',
                        'episode_id', 'before_step', 'after_step', 'decision_id'):
                if key in archive.files:
                    array = np.asarray(archive[key])
                    if array.size > 16384:
                        raise ValueError('Prediction metadata is too large')
                    result[key] = array.tolist()
        before, after = result.get('before_step'), result.get('after_step')
        result['aligned'] = (isinstance(result.get('episode_id'), str) and bool(result['episode_id'])
            and isinstance(result.get('decision_id'), str) and bool(result['decision_id'])
            and type(before) is int and type(after) is int and 0 <= before <= after)
        return result
