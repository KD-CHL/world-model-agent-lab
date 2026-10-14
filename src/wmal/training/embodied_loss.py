"""Normalized supervision; candidate priors receive bounded real context only."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from wmal.models.categorical_rssm import balanced_kl, categorical_probs
from wmal.models.embodied_world import trajectory_seed


class _DeterministicPool(torch.autograd.Function):
    """Native forward, fixed-order backward (CUDA adaptive pooling uses atomics)."""
    @staticmethod
    def forward(ctx, value, output_size):
        ctx.input_shape, ctx.output_size = value.shape, output_size
        return F.adaptive_avg_pool2d(value, output_size)

    @staticmethod
    def backward(ctx, gradient):
        height, width = ctx.input_shape[-2:]
        out_h, out_w = ctx.output_size
        # Accumulate low-precision gradients in FP32, then cast once. Each bin
        # is added sequentially, including overlapping bins for small images.
        dtype = torch.float64 if gradient.dtype == torch.float64 else torch.float32
        result = torch.zeros(ctx.input_shape, dtype=dtype, device=gradient.device)
        for row in range(out_h):
            low_h, high_h = row*height//out_h, ((row+1)*height+out_h-1)//out_h
            for col in range(out_w):
                low_w, high_w = col*width//out_w, ((col+1)*width+out_w-1)//out_w
                contribution = gradient[..., row:row+1, col:col+1].to(dtype)
                result[..., low_h:high_h, low_w:high_w] += contribution / ((high_h-low_h)*(high_w-low_w))
        return result.to(gradient.dtype), None


class DeterministicAdaptivePool2d(nn.AdaptiveAvgPool2d):
    """Parameter-free training substitute; inference forward remains bit-identical."""
    def forward(self, value):
        return _DeterministicPool.apply(value, self.output_size)


def masked_rgb_mse(prediction, target, view_mask):
    weights = view_mask[..., None, None, None].to(prediction.dtype)
    pixels = target.shape[-3] * target.shape[-2] * target.shape[-1]
    return ((prediction.float() - target.float()).square() * weights).sum() / (weights.sum() * pixels).clamp_min(1)


def changed_region_loss(prediction, target, origin, view_mask, threshold):
    """Mean-channel |target-origin| > threshold; normalize by changed pixels alone.

    An empty region contributes zero. Threshold is in RGB [0,1] units and the
    fraction uses measured pixels only, never missing cameras or startup padding.
    """
    measured = view_mask[..., None, None].expand(*target.shape[:-3], *target.shape[-2:])
    changed = (target.float() - origin.float()).abs().mean(-3) > threshold
    changed = changed & measured
    squared = (prediction.float() - target.float()).square().mean(-3)
    loss = (squared * changed).sum() / changed.sum().clamp_min(1)
    fraction = changed.sum().float() / measured.sum().clamp_min(1)
    return loss, fraction


def prepare_sample(sample, normalization, config, device):
    """Strip all-false startup slots, retain real T and choose a bounded prefix."""
    mask = np.asarray(sample['view_mask'], dtype=bool)
    real = np.flatnonzero(mask.any(-1))
    if not len(real) or not mask[real[0]:].any(-1).all():
        raise ValueError('Only contiguous startup padding may be stripped')
    low = int(real[0]); origin = int(sample.get('origin', sample['prediction_offset']))
    if origin < low or origin + config.horizon >= len(mask):
        raise ValueError('Origin requires a real observation and complete future horizon')
    if len(sample['actions']) != len(mask)-1:
        raise ValueError('Training requires T observations / T-1 actions')
    tensor = lambda x, dtype=torch.float32: torch.as_tensor(np.array(x, copy=True), dtype=dtype, device=device)[None]
    def normalized(values, prefix):
        values = (np.asarray(values, dtype='float64') - normalization[prefix+'_mean']) / normalization[prefix+'_scale']
        if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo('float32').max):
            raise ValueError('Nonfinite normalized training coordinates')
        return tensor(values)
    rgb = tensor(sample['rgb'][low:])
    physical = normalized(sample['physical'][low:], 'physical')
    commands = normalized(sample['commands'][low:], 'command')
    actions = normalized(sample['actions'][low:], 'action')
    views = tensor(mask[low:], torch.bool)
    resets = tensor(sample['is_first'][low:], torch.bool)
    end = origin + config.horizon
    begin = max(low, origin-config.context_steps)
    # Infer only from the raw origin command and future actions; target commands
    # cannot become a shortcut into imagination, even under nonzero normalization.
    width = config.model.command_dim
    if width:
        if config.command_rule != 'joint_delta' or width != config.model.action_dim:
            raise ValueError('No known future commands for this action profile')
        future = np.asarray(sample['commands'][origin], dtype='float64') + np.cumsum(
            np.asarray(sample['actions'][origin:end], dtype='float64'), axis=0)
    else:
        future = np.empty((config.horizon, 0))
    o, b = origin-low, begin-low
    context = (rgb[:, b:o+1], physical[:, b:o+1], commands[:, b:o+1], actions[:, b:o],
               actions[:, o:o+config.horizon], normalized(future, 'command'))
    return dict(rgb=rgb, physical=physical, commands=commands, actions=actions,
                view_mask=views, is_first=resets, origin=o, context=context,
                context_view_mask=views[:, b:o+1], context_is_first=resets[:, b:o+1])


def imagine_sample(member, prepared, config, *, seed):
    return member.imagine_context(*prepared['context'], view_mask=prepared['context_view_mask'],
        is_first=prepared['context_is_first'], sample=config.mode == 'sample', seed=seed)


def embodied_loss(member, batch, normalization, config):
    """Complete posterior + multi-step prior loss, executed inside wrapper.forward.

    Batch is a list of aligned Task2 samples. Raw T sequences carry an explicitly
    sampled `origin`; H/K validation windows carry prediction_offset. Only loss
    targets use future observations, view masks, changed regions and event masks.
    """
    if not isinstance(batch, (list, tuple)) or not batch:
        raise ValueError('Expected a nonempty batch of aligned embodied samples')
    totals, rows = [], []
    device = next(member.parameters()).device
    for sample in batch:
        values = prepare_sample(sample, normalization, config, device)
        origin, horizon = values['origin'], config.horizon
        posterior = member.observe(values['rgb'], values['physical'], values['commands'], values['actions'],
            view_mask=values['view_mask'], is_first=values['is_first'],
            sample=config.mode == 'sample', seed=trajectory_seed(int(sample.get('seed', 0)), 0, 0))
        # Earlier real prefix is burn-in only. No startup padding survives prepare_sample.
        recon = (F.mse_loss(posterior['physical'][:, origin:].float(), values['physical'][:, origin:]) +
            masked_rgb_mse(posterior['rgb'][:, origin:], values['rgb'][:, origin:], values['view_mask'][:, origin:]))
        qlog, plog = posterior['posterior_logits'][:, origin:].float(), posterior['prior_logits'][:, origin:].float()
        dyn, rep = balanced_kl(qlog, plog, free_nats=config.free_nats, unimix=member.config.unimix)
        q, p = categorical_probs(qlog, member.config.unimix), categorical_probs(plog, member.config.unimix)
        raw_kl = (q * (q.log()-p.log())).sum((-1, -2))
        target_p = values['physical'][:, origin+1:origin+horizon+1]
        target_rgb = values['rgb'][:, origin+1:origin+horizon+1]
        future_mask = values['view_mask'][:, origin+1:origin+horizon+1]
        predictions, frames, regions, events, fractions = [], [], [], [], []
        event_start = int(sample.get('origin', 0)) if 'origin' in sample else 0
        event_mask = torch.as_tensor(np.array(sample['event_mask'][event_start:event_start+horizon], copy=True),
                                     dtype=torch.bool, device=device)[None]
        event_target = torch.as_tensor(np.array(sample['events'][event_start:event_start+horizon], copy=True),
                                       dtype=torch.float32, device=device)[None]
        for index in range(config.sample_count):
            imagined = imagine_sample(member, values, config,
                seed=trajectory_seed(int(sample.get('seed', 0)), 0, index))
            predictions.append(imagined['physical'].float())
            frames.append(masked_rgb_mse(imagined['rgb'], target_rgb, future_mask))
            region, fraction = changed_region_loss(imagined['rgb'], target_rgb,
                values['rgb'][:, origin:origin+1], future_mask, config.change_threshold)
            regions.append(region); fractions.append(fraction)
            if imagined['event_logits'] is not None:
                ce = F.binary_cross_entropy_with_logits(imagined['event_logits'].float(), event_target, reduction='none')
                events.append((ce * event_mask).sum() / event_mask.sum().clamp_min(1))
        predicted = torch.stack(predictions)
        state = (predicted-target_p).square().mean()
        frame, changed = torch.stack(frames).mean(), torch.stack(regions).mean()
        event = torch.stack(events).mean() if events else state.new_zeros(())
        loss = (config.reconstruction_weight*recon + config.state_weight*state +
                config.frame_weight*frame + config.change_weight*changed +
                config.dyn_weight*dyn.mean() + config.rep_weight*rep.mean() + config.event_weight*event)
        mean_prediction = predicted.mean(0)
        error = (mean_prediction-target_p).square()[0]
        baseline = (values['physical'][:, origin:origin+1]-target_p).square()[0]
        scale = torch.tensor(normalization['physical_scale'], device=device)
        totals.append(loss)
        rows.append(dict(state=float(state.detach()), frame=float(frame.detach()), changed_region=float(changed.detach()),
            reconstruction=float(recon.detach()), event=float(event.detach()),
            raw_kl=float(raw_kl.detach().mean()), clipped_kl=float(dyn.detach().mean()),
            kl_clipping_rate=float((raw_kl.detach() < config.free_nats).float().mean()),
            prior_entropy=float(posterior['prior_entropy'][:, origin:].detach().mean()),
            posterior_entropy=float(posterior['posterior_entropy'][:, origin:].detach().mean()),
            changed_fraction=float(torch.stack(fractions).detach().mean()),
            physical_rmse=(error.detach().sqrt()*scale).cpu().tolist(),
            baseline_rmse=(baseline.detach().sqrt()*scale).cpu().tolist(),
            event_seen=event_mask.any((0, 1)).cpu().tolist(), sample_id=sample['sample_id'],
            episode_id=sample['episode_id'], source_episode_id=sample['source_episode_id'],
            original_group=sample.get('original_group', sample['source_episode_id']),
            offset=sample['offset'], seed=int(sample.get('seed', 0)),
            origin=int(sample.get('origin', sample['prediction_offset'])), sequence_length=len(values['rgb'][0])))
    return torch.stack(totals).mean(), dict(samples=rows)


class EmbodiedLossForward(nn.Module):
    """DDP/FSDP wrap this module, so every training loss traverses forward hooks."""
    def __init__(self, member, normalization, config):
        super().__init__()
        self.member, self.normalization, self.config = member, normalization, config
        for name, layer in member.encoder.backbone.named_children():
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                member.encoder.backbone[int(name)] = DeterministicAdaptivePool2d(layer.output_size)

    def forward(self, batch):
        device = next(self.member.parameters()).device
        enabled = self.config.precision != 'float32'
        dtype = torch.float16 if self.config.precision == 'float16' else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            return embodied_loss(self.member, batch, self.normalization, self.config)
