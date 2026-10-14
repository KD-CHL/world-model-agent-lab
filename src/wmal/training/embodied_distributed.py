"""Forward-owned loss wrapping, real rank setup, accumulation and selection."""
from contextlib import nullcontext
from datetime import timedelta
import os
import copy

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def optimizer_update(model, optimizer, microbatches, forward, *, scaler=None):
    """One update. Tail groups divide by their actual length; sync on final forward."""
    if not microbatches:
        raise ValueError('An optimizer update requires real microbatches')
    optimizer.zero_grad(set_to_none=True)
    metrics = []
    for index, batch in enumerate(microbatches):
        sync = (model.no_sync() if index + 1 < len(microbatches) and hasattr(model, 'no_sync')
                else nullcontext())
        with sync:
            loss, values = forward(batch)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            scaled_loss = loss / len(microbatches)
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()
            metrics.append(dict(values, loss=float(loss.detach())))
    if scaler is not None:
        scaler.unscale_(optimizer)
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for group in optimizer.param_groups for parameter in group['params']):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError('Nonfinite gradient: optimizer update refused')
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer); scaler.update()
    return dict(microbatches=len(microbatches), metrics=metrics)


def unique_validation_metrics(records, physical_scale):
    unique = {}
    for sample_id, squared_error in records:
        value = np.asarray(squared_error, dtype='float64')
        if value.ndim != 2 or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError('Validation requires finite squared errors [H,P]')
        if sample_id in unique and not np.array_equal(unique[sample_id], value):
            raise ValueError('Duplicate validation identity has conflicting predictions')
        unique[sample_id] = value
    if not unique:
        raise ValueError('No unique validation samples')
    normalized = np.sqrt(np.stack([unique[key] for key in sorted(unique)]).mean(0))
    raw = normalized * np.asarray(physical_scale)
    return dict(unique_samples=len(unique), selection_metric=float(normalized.mean()),
                normalized_rmse=normalized.tolist(), physical_rmse=raw.tolist())


def select_epoch(metrics):
    if not metrics or not np.isfinite(metrics).all():
        raise ValueError('Finite selection history required')
    return int(np.argmin(metrics)) + 1


class DistributedRuntime:
    def __init__(self, mode, device):
        if mode not in ('none', 'ddp', 'fsdp'):
            raise ValueError('distributed must be none/ddp/fsdp')
        self.mode = mode
        self.rank = int(os.environ.get('RANK', '0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.owns_group = False
        self.device = torch.device(device)
        if mode == 'fsdp' and (self.device.type != 'cuda' or
                              str(torch.__version__).split('+')[0] != '2.11.0'):
            raise ValueError('FSDP requires CUDA and explicitly supported torch 2.11.0; no fallback')
        if mode != 'none' and (self.world_size < 2 or any(
                key not in os.environ for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))):
            raise ValueError('Distributed training requires torchrun ranks and WORLD_SIZE >= 2')
        if mode == 'none' and self.world_size != 1:
            raise ValueError('WORLD_SIZE > 1 requires distributed ddp/fsdp')
        if not 0 <= self.rank < self.world_size or self.local_rank < 0:
            raise ValueError('Invalid torchrun rank')
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA requested but unavailable')
            self.device = torch.device('cuda', self.local_rank if mode != 'none' else (self.device.index or 0))
            torch.cuda.set_device(self.device)
        elif self.device.type != 'cpu':
            raise ValueError('Only CPU gloo and CUDA NCCL are supported')
        if mode != 'none' and not dist.is_initialized():
            backend = 'nccl' if self.device.type == 'cuda' else 'gloo'
            if backend == 'nccl' and not dist.is_nccl_available():
                raise ValueError('NCCL is unavailable; no backend fallback')
            dist.init_process_group(backend=backend, timeout=timedelta(seconds=60))
            self.owns_group = True
        if dist.is_initialized() and (dist.get_rank() != self.rank or dist.get_world_size() != self.world_size):
            raise ValueError('Existing process group differs from torchrun ranks')

    def gather(self, value):
        if self.mode == 'none':
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def rank_zero(self, operation):
        """Propagate rank-zero I/O errors to peers before the next collective."""
        value, error = None, None
        if self.rank == 0:
            try:
                value = operation()
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        if self.mode != 'none':
            message = [value, error]; dist.broadcast_object_list(message, src=0)
            value, error = message
        if error:
            raise ValueError(error)
        return value

    def wrap(self, loss_module):
        if self.mode == 'ddp':
            return DistributedDataParallel(loss_module,
                device_ids=[self.device.index] if self.device.type == 'cuda' else None)
        if self.mode == 'fsdp':
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            return FSDP(loss_module, device_id=self.device, use_orig_params=True)
        return loss_module

    def snapshot(self, wrapped, member, optimizer):
        if self.mode == 'fsdp':
            from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                StateDictType, FullStateDictConfig, FullOptimStateDictConfig)
            # All ranks participate; full CPU snapshots also permit detached selection.
            with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                    FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)):
                state = wrapped.state_dict()
                optim = FSDP.optim_state_dict(wrapped, optimizer)
            state = {key.removeprefix('member.'): value for key, value in state.items()}
        else:
            state, optim = member.state_dict(), optimizer.state_dict()
        def cpu_copy(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {key: cpu_copy(item) for key, item in value.items()}
            if isinstance(value, list):
                return [cpu_copy(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cpu_copy(item) for item in value)
            return copy.deepcopy(value)
        return {key: value.detach().cpu().clone() for key, value in state.items()}, cpu_copy(optim)

    def restore_optimizer(self, wrapped, optimizer, state):
        if self.mode == 'fsdp':
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            state = FSDP.optim_state_dict_to_load(wrapped, optimizer, state)
        optimizer.load_state_dict(state)

    def close(self):
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
