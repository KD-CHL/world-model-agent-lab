"""Offline image-space evaluation for visual world-model plugins."""
from pathlib import Path
import json
import math
import numpy as np

from wmal.logging.manifest import atomic_json, sha256_file


def _frames(values, label):
    result = np.asarray(values)
    if result.dtype != np.uint8 or result.ndim != 4 or result.shape[0] < 1 or result.shape[-1] != 3:
        raise ValueError(f"{label} must be a nonempty uint8 [T,H,W,3] RGB sequence")
    return result


def evaluate_video_predictions(provider, dataset_path, output_path):
    if not isinstance(getattr(provider, "version", None), str) or not provider.version:
        raise ValueError("Visual predictor must declare a model version")
    source = Path(dataset_path).expanduser().resolve()
    if not source.exists():
        raise ValueError("Visual evaluation dataset does not exist")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample_path = destination.with_name(destination.stem + ".samples.jsonl")
    count, total_abs, total_sq, pixels, sample_metrics = 0, 0.0, 0.0, 0, []
    with sample_path.open("w", encoding="utf-8") as stream:
        for item in provider.iter_evaluation_samples(str(source)):
            sample_id = item.get("sample_id")
            if not isinstance(sample_id, (str, int)):
                raise ValueError("Visual sample must have a stable sample_id")
            prediction = _frames(provider.predict_video(item["observation_history"],
                                                        item["action_sequence"]), "prediction")
            target = _frames(item["target_frames"], "target")
            if prediction.shape != target.shape:
                raise ValueError("Predicted and target video shapes/timesteps must match exactly")
            difference = prediction.astype(np.float32) - target.astype(np.float32)
            abs_sum = float(np.abs(difference).sum())
            sq_sum = float(np.square(difference).sum())
            n = int(difference.size)
            mse = sq_sum / n
            psnr = None if mse == 0 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)
            row = {"sample_id": sample_id, "model_version": provider.version,
                   "frames": int(prediction.shape[0]), "height": int(prediction.shape[1]),
                   "width": int(prediction.shape[2]), "mae_0_1": abs_sum / n / 255.0,
                   "psnr_db": psnr}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
            total_abs += abs_sum
            total_sq += sq_sum
            pixels += n
            sample_metrics.append(row)
    if count == 0 or pixels == 0:
        raise ValueError("Visual predictor returned no evaluation samples")
    mse = total_sq / pixels
    summary = {"schema_version": 1, "run_kind": "visual_world_model_evaluation",
               "model_version": provider.version, "dataset_path": str(source),
               "dataset_sha256": sha256_file(source) if source.is_file() else None,
               "samples": count, "mae_0_1": total_abs / pixels / 255.0,
               "psnr_db": None if mse == 0 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse),
               "sample_metrics_path": str(sample_path),
               "scope_note": "Image-space prediction metrics only; not task success or simulator ground truth."}
    atomic_json(destination, summary)
    return summary
