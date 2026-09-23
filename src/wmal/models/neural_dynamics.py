"""Optional neural ensemble for nonlinear state/action transition prediction.

PyTorch is imported lazily so simulation protocols stay usable without it.
The model predicts state deltas and exposes member-wise rollouts to the planner.
"""
from pathlib import Path
import os
import tempfile

import numpy as np


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError('Install the optional learning dependencies to use neural dynamics') from exc
    return torch, nn


def _network(width, output_width):
    torch, nn = _torch_modules()
    return nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 128),
                         nn.SiLU(), nn.Linear(128, output_width))


class NeuralJointDynamics:
    """Bootstrap ensemble with per-member MLP residual dynamics."""

    def __init__(self, joints, state_dicts, version, *, device='cpu'):
        torch, _ = _torch_modules()
        self.joints = tuple(joints)
        if not self.joints or len(set(self.joints)) != len(self.joints) or not state_dicts:
            raise ValueError('Invalid model dimensions')
        self.device = torch.device(device)
        self.models = []
        for state_dict in state_dicts:
            model = _network(1 + 2 * len(self.joints), len(self.joints)).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)
        self.ensemble_size = len(self.models)
        self.version = str(version)

    def _feature(self, state, targets, duration_s):
        if set(state) != set(self.joints) or not set(targets).issubset(self.joints) or duration_s <= 0:
            raise ValueError('State/action mismatch')
        row = [duration_s, *[state[joint] for joint in self.joints],
               *[targets.get(joint, state[joint]) for joint in self.joints]]
        if not np.isfinite(row).all():
            raise ValueError('Nonfinite model input')
        return row

    def predict_member(self, index, state, targets, duration_s):
        torch, _ = _torch_modules()
        if not 0 <= index < self.ensemble_size:
            raise IndexError('Unknown ensemble member')
        feature = torch.as_tensor(self._feature(state, targets, duration_s), dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        with torch.inference_mode():
            delta = self.models[index](feature).squeeze(0).cpu().numpy()
        return {joint: float(state[joint] + delta[i]) for i, joint in enumerate(self.joints)}

    def predict(self, state, targets, duration_s):
        predictions = [self.predict_member(i, state, targets, duration_s)
                       for i in range(self.ensemble_size)]
        return {joint: float(np.mean([item[joint] for item in predictions])) for joint in self.joints}

    def save(self, path):
        torch, _ = _torch_modules()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {'schema_version': 1, 'joints': self.joints, 'version': self.version,
                   'state_dicts': [{key: value.detach().cpu() for key, value in model.state_dict().items()}
                                   for model in self.models]}
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=destination.name + '.')
        os.close(descriptor)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path, *, device='cpu'):
        torch, _ = _torch_modules()
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get('schema_version') != 1:
            raise ValueError('Unknown neural dynamics checkpoint schema')
        return cls(payload['joints'], payload['state_dicts'], payload['version'], device=device)

    @classmethod
    def fit(cls, train_rows, joints, *, validation_rows=(), members=5, epochs=100,
            batch_size=256, learning_rate=3e-4, seed=0, device='cpu', version=None):
        torch, nn = _torch_modules()
        if not train_rows or members < 1 or epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError('Invalid neural training data or parameters')
        joints = tuple(joints)

        def arrays(rows):
            features, deltas = [], []
            for row in rows:
                before, after, target = row['before'], row['after'], row['target']
                if set(before) != set(joints) or set(after) != set(joints) or not set(target).issubset(joints):
                    raise ValueError('Training transition does not match joint schema')
                features.append([row['duration_s'], *[before[j] for j in joints],
                                 *[target.get(j, before[j]) for j in joints]])
                deltas.append([after[j] - before[j] for j in joints])
            x, y = np.asarray(features, dtype=np.float32), np.asarray(deltas, dtype=np.float32)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Nonfinite training data')
            return x, y

        x, y = arrays(train_rows)
        vx, vy = arrays(validation_rows) if validation_rows else (None, None)
        state_dicts, metrics = [], []
        for member_index in range(members):
            member_seed = seed + member_index
            torch.manual_seed(member_seed)
            generator = np.random.default_rng(member_seed)
            sampled = generator.integers(0, len(x), len(x)) if members > 1 else np.arange(len(x))
            x_tensor = torch.as_tensor(x[sampled], device=device)
            y_tensor = torch.as_tensor(y[sampled], device=device)
            model = _network(x.shape[1], y.shape[1]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            best_state, best_validation = None, float('inf')
            for _ in range(epochs):
                model.train()
                permutation = torch.randperm(len(x_tensor), device=device)
                for indices in permutation.split(batch_size):
                    loss = nn.functional.mse_loss(model(x_tensor[indices]), y_tensor[indices])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                model.eval()
                if vx is not None:
                    with torch.inference_mode():
                        valid_loss = float(nn.functional.mse_loss(
                            model(torch.as_tensor(vx, device=device)), torch.as_tensor(vy, device=device)).item())
                    if valid_loss < best_validation:
                        best_validation = valid_loss
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if best_state is None:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_validation = None
            state_dicts.append(best_state)
            metrics.append({'member': member_index, 'validation_mse_rad2': best_validation})
        model_version = version or f'neural-joint-seed{seed}-n{len(train_rows)}'
        return cls(joints, state_dicts, model_version, device=device), metrics
