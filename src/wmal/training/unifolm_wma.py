"""Reproducible configuration and launch helpers for an external WMA checkout."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from wmal.logging.manifest import atomic_json, sha256_file


def _yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML with `python -m pip install -e '.[robot-data]'`") from exc
    return yaml


def _required_file(value, name):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{name} file does not exist")
    return path


def build_wma_config(template_path, *, checkpoint, prepared_data, dataset_key, mode, output_path):
    """Write a WMA-compatible config overlay without touching upstream files."""
    yaml = _yaml()
    template = _required_file(template_path, "WMA training template")
    base_checkpoint = _required_file(checkpoint, "WMA base checkpoint")
    data_root = Path(prepared_data).expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError("Prepared WMA dataset directory does not exist")
    if mode not in ("decision", "joint"):
        raise ValueError("mode must be 'decision' or 'joint'")
    if not isinstance(dataset_key, str) or not dataset_key.strip() or any(ch.isspace() for ch in dataset_key):
        raise ValueError("dataset_key must be a nonempty token")
    try:
        config = yaml.safe_load(template.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("Cannot parse WMA config template") from exc
    try:
        model = config["model"]
        model_params = model["params"]
        data_params = config["data"]["params"]
        train_data = data_params["train"]["params"]
    except (TypeError, KeyError) as exc:
        raise ValueError("WMA template is missing expected model/data config sections") from exc
    if (not isinstance(model, dict) or not isinstance(model_params, dict)
            or not isinstance(data_params, dict) or not isinstance(train_data, dict)):
        raise ValueError("WMA config sections must be mappings")
    model["pretrained_checkpoint"] = str(base_checkpoint)
    # Upstream defines False as joint decision-making plus simulation training.
    model_params["decision_making_only"] = mode == "decision"
    train_data["data_dir"] = str(data_root)
    data_params["dataset_and_weights"] = {dataset_key: 1.0}
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                     prefix=destination.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_wma_command(upstream_root, config_path, *, run_name, output_dir, processes, master_port=12366):
    root = Path(upstream_root).expanduser().resolve()
    trainer = root / "scripts" / "trainer.py"
    config = _required_file(config_path, "Generated WMA config")
    if not trainer.is_file():
        raise ValueError("WMA trainer not found under upstream root")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Run output directory is not empty; refusing to overwrite a prior run")
    if not isinstance(run_name, str) or not run_name.strip() or any(ch.isspace() for ch in run_name):
        raise ValueError("run_name must be a nonempty token")
    if type(processes) is not int or processes < 1:
        raise ValueError("processes must be a positive integer")
    if type(master_port) is not int or not 1024 <= master_port <= 65535:
        raise ValueError("master_port must be in [1024, 65535]")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={processes}", f"--master_port={master_port}", str(trainer),
               "--base", str(config), "--train", "--name", run_name, "--logdir", str(output),
               "--devices", str(processes), f"--total_gpus={processes}", "lightning.trainer.num_nodes=1"]
    return command


def launch_wma_training(upstream_root, config_path, *, run_name, output_dir, processes,
                        master_port=12366, dry_run=False, cuda_visible_devices=None):
    """Launch a configured WMA run and preserve provenance/logs outside the repo."""
    command = build_wma_command(upstream_root, config_path, run_name=run_name,
                                output_dir=output_dir, processes=processes, master_port=master_port)
    config = Path(config_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    yaml = _yaml()
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    checkpoint = Path(loaded["model"]["pretrained_checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise ValueError("Configured base checkpoint does not exist")
    manifest = {"schema_version": 1, "run_kind": "unifolm_wma_finetune",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_name": run_name,
                "mode": "decision" if loaded["model"]["params"].get("decision_making_only") else "joint",
                "config": {"path": str(config), "sha256": sha256_file(config)},
                "base_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
                "prepared_data": loaded["data"]["params"]["train"]["params"]["data_dir"],
                "dataset_mixture": loaded["data"]["params"].get("dataset_and_weights", {}),
                "processes": processes, "command": command}
    if dry_run:
        return {**manifest, "dry_run": True}
    if cuda_visible_devices is not None and (not isinstance(cuda_visible_devices, str) or not cuda_visible_devices.strip()):
        raise ValueError("cuda_visible_devices must be a nonempty device list")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "wmal_run_manifest.json", manifest)
    environment = os.environ.copy()
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    with (output / "training.log").open("wb") as log:
        result = subprocess.run(command, cwd=Path(upstream_root).expanduser().resolve(),
                                env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        atomic_json(output / "failure.json", {"returncode": result.returncode,
                                               "log": str(output / "training.log")})
        raise RuntimeError(f"WMA training failed with exit code {result.returncode}; inspect training.log")
    return {**manifest, "dry_run": False, "returncode": result.returncode,
            "log": str(output / "training.log")}
