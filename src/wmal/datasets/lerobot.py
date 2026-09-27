"""Read-only validation and provenance for the LeRobot V2.1 layout used by WMA."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from wmal.datasets.provenance import file_record
from wmal.datasets.splits import split_episodes
from wmal.logging.manifest import sha256_file


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON metadata: {Path(path).name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {Path(path).name}")
    return value


def _shape(feature, name):
    if not isinstance(feature, dict):
        raise ValueError(f"Missing feature metadata: {name}")
    shape = feature.get("shape")
    if not isinstance(shape, list) or not shape or any(type(width) is not int or width < 1 for width in shape):
        raise ValueError(f"Invalid feature shape: {name}")
    return shape


def _vector_names(feature, dimension, name):
    names = feature.get("names") if isinstance(feature, dict) else None
    # LeRobot v2.1 stores vector names grouped by a single axis, e.g.
    # [["joint_0", "joint_1"]], while some exporters use a flat list.
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if (not isinstance(names, list) or len(names) != dimension
            or any(not isinstance(value, str) or not value for value in names)
            or len(set(names)) != len(names)):
        raise ValueError(f"{name} feature must declare unique ordered names")
    return names


def inspect_lerobot_v21(root, *, dataset_id, revision, license_id, seed=0,
                        camera_key="observation.images.top", hash_episode_files=False):
    """Validate WMA's expected V2.1 source tree and return a reproducible manifest.

    Large parquet/video contents are not read or mutated. Set ``hash_episode_files``
    to stream full content hashes when an expensive integrity pass is desired.
    """
    root = Path(root).resolve()
    if not dataset_id or not revision or not license_id:
        raise ValueError("Dataset ID, pinned revision, and license identifier are required")
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    info = _read_json(info_path)
    total = info.get("total_episodes")
    if type(total) is not int or total < 5:
        raise ValueError("LeRobot metadata must contain at least five episodes")
    episodes_metadata = info.get("episodes")
    if isinstance(episodes_metadata, dict) and isinstance(episodes_metadata.get("chunk-000"), dict):
        episodes_metadata = episodes_metadata["chunk-000"]
    if isinstance(episodes_metadata, list) and len(episodes_metadata) != total:
        raise ValueError("Episode metadata count does not match total_episodes")
    fps = info.get("fps")
    if type(fps) not in (int, float) or fps <= 0:
        raise ValueError("LeRobot metadata requires a positive fps")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("LeRobot metadata requires a features object")
    state_shape = _shape(features.get("observation.state"), "observation.state")
    action_shape = _shape(features.get("action"), "action")
    if len(state_shape) != 1 or len(action_shape) != 1:
        raise ValueError("State and action features must be one-dimensional vectors")
    if state_shape != action_shape:
        raise ValueError("State/action dimensions differ; an explicit adapter is required")
    state_names = _vector_names(features["observation.state"], state_shape[0], "State")
    action_names = _vector_names(features["action"], action_shape[0], "Action")
    camera = features.get(camera_key)
    camera_shape = _shape(camera, camera_key)
    dimensions = camera.get("names")
    if not isinstance(dimensions, list) or len(dimensions) != 3:
        raise ValueError("Selected camera must declare three ordered image dimensions")
    normalized_dimensions = [str(name).lower() for name in dimensions]
    if normalized_dimensions == ["channels", "height", "width"]:
        channels, height, width = camera_shape
    elif normalized_dimensions == ["height", "width", "channels"]:
        height, width, channels = camera_shape
    else:
        raise ValueError("Camera dimensions must be channels/height/width or height/width/channels")
    if camera.get("dtype") not in ("video", "image") or channels != 3:
        raise ValueError("Selected camera must be an RGB video/image feature")

    tasks = []
    try:
        for line_no, line in enumerate(tasks_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            task = json.loads(line)
            if not isinstance(task, dict) or not isinstance(task.get("task"), str) or not task["task"].strip():
                raise ValueError(f"Invalid task record on line {line_no}")
            tasks.append(task["task"].strip())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read valid LeRobot tasks.jsonl") from exc
    if not tasks:
        raise ValueError("LeRobot tasks.jsonl is empty")

    def resolve_episode_file(kind, episode, filename):
        patterns = info.get("data_path" if kind == "data" else "video_path")
        if isinstance(patterns, str):
            relative = patterns.replace("{episode_index:06d}", f"{episode:06d}")
            relative = relative.replace("{episode_index}", str(episode))
            relative = relative.replace("{chunk_index}", "0").replace("{video_key}", camera_key)
            candidate = root / relative
            if candidate.is_file():
                return candidate
        candidates = ([root / "data" / "chunk-000" / filename]
                      if kind == "data" else
                      [root / "videos" / "chunk-000" / camera_key / filename,
                       root / "videos" / "chunk-000" / camera_key.rsplit(".", 1)[-1] / filename])
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    episode_ids = list(range(total))
    split_by_episode = split_episodes(episode_ids, seed=seed)
    episodes = []
    for episode in episode_ids:
        parquet = resolve_episode_file("data", episode, f"episode_{episode:06d}.parquet")
        video = resolve_episode_file("video", episode, f"episode_{episode:06d}.mp4")
        if not parquet.is_file() or parquet.stat().st_size == 0:
            raise ValueError(f"Missing or empty episode parquet: {parquet.relative_to(root)}")
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError(f"Missing or empty camera video: {video.relative_to(root)}")
        episodes.append({"episode_index": episode, "split": split_by_episode[episode],
                         "parquet": file_record(parquet, hash_content=hash_episode_files),
                         "video": file_record(video, hash_content=hash_episode_files)})

    manifest_payload = {"dataset_id": dataset_id, "revision": revision,
                        "license_id": license_id, "camera_key": camera_key,
                        "episode_count": total}
    manifest_id = hashlib.sha256(json.dumps(manifest_payload, sort_keys=True).encode()).hexdigest()

    return {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "format": "lerobot_v2.1_wma_source",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"repo_id": dataset_id, "revision": revision, "license": license_id,
                    "source_root": str(root), "info_sha256": sha256_file(info_path),
                    "tasks_sha256": sha256_file(tasks_path)},
        "robot": info.get("robot_type", info.get("robot", "unspecified")),
        "fps": float(fps),
        "camera": {"key": camera_key, "width": width, "height": height, "channels": channels},
        "state": {"key": "observation.state", "dimension": state_shape[0],
                  "order": state_names, "dtype": features["observation.state"].get("dtype")},
        "action": {"key": "action", "dimension": action_shape[0],
                   "order": action_names, "dtype": features["action"].get("dtype")},
        "tasks": sorted(set(tasks)), "seed": seed,
        "split_counts": {name: sum(value == name for value in split_by_episode.values())
                         for name in ("train", "validation", "test")},
        "episodes": episodes,
        "conversion": {"upstream": "unitreerobotics/unifolm-world-model-action",
                       "expected_converter": "prepare_data/prepare_training_data.py"},
    }
