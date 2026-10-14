"""Atomic training snapshots, separate from the provider's inference artifact."""
import os
from pathlib import Path
import platform
import random
import tempfile
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist

SCHEMA = 'wmal.embodied_training.v1'


@contextmanager
def deterministic_execution():
    """Exact same-environment replay; no warn-only or nondeterministic fallback.

    Applied before CUDA math. Restore caller settings even on interruption so
    the independent trainer does not change legacy callers' execution policy.
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        if workspace is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace


def environment_identity(device):
    result = dict(python=platform.python_version(), torch=str(torch.__version__),
        numpy=np.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        platform=platform.platform(), deterministic=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32, device_type=torch.device(device).type)
    result.update(python_executable=sys.executable, torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        backend_config=torch.__config__.show(),
        environment={key: os.environ.get(key) for key in
            ('CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    if torch.device(device).type == 'cuda':
        info = torch.cuda.get_device_properties(device)
        result['hardware'] = dict(name=info.name, total_memory=info.total_memory,
                                  capability=[info.major, info.minor])
    return result


def capture_rng(device='cpu'):
    name, keys, position, gaussian, cached = np.random.get_state()
    result = dict(python=random.getstate(), numpy=[name, keys.tolist(), position, gaussian, cached],
                  torch=torch.get_rng_state().tolist(), cuda=None)
    if torch.device(device).type == 'cuda':
        result['cuda'] = torch.cuda.get_rng_state(device).tolist()
    return result


def restore_rng(state, device='cpu'):
    random.setstate(state['python'])
    name, keys, position, gaussian, cached = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype='uint32'), position, gaussian, cached))
    torch.set_rng_state(torch.tensor(state['torch'], dtype=torch.uint8))
    if state['cuda'] is not None:
        if torch.device(device).type != 'cuda':
            raise ValueError('CUDA RNG cannot resume on CPU')
        torch.cuda.set_rng_state(torch.tensor(state['cuda'], dtype=torch.uint8), device)


def save_training_checkpoint(path, payload):
    """All ranks call at a completed optimizer boundary; rank zero alone writes."""
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload, schema=SCHEMA), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_checkpoint(path, identity):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA:
        raise ValueError('Expected optimizer-boundary training snapshot, not inference weights')
    required = {'identity', 'model', 'optimizer', 'scheduler', 'scaler', 'rank_states',
                'position', 'run', 'normalization', 'schema'}
    if payload.keys() != required or payload['identity'] != identity:
        raise ValueError('Exact resume identity mismatch: config/data/profile/environment/world size/output')
    if len(payload['rank_states']) != identity['world_size']:
        raise ValueError('Exact resume requires every rank RNG and sampler state')
    return payload
