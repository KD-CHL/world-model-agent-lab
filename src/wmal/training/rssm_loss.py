"""Posterior representation learning + deployment-matched prior-only supervision."""
import torch
from torch.nn import functional as F

from wmal.models.categorical_rssm import balanced_kl, symlog


def rssm_loss(member, rgb, state, action, batch, config, normalization):
    # Posterior sampling only during training. Validation uses the same repeatable
    # soft categorical proxy as deployed prediction and independent calibration.
    k = config.context_steps
    if k != member.config.context_steps:
        raise ValueError('Loss and RSSM context protocols differ')
    if rgb.shape[1] != k + config.horizon + 1:
        raise ValueError('Loss requires aligned burn-in context and full future horizon')
    is_first = batch.get('is_first')
    if is_first is not None:
        is_first = is_first.to(rgb.device)
    observed = member.observe(rgb, state, action, is_first=is_first, sample=member.training)
    imagined = member.imagine_context(rgb[:, :k+1], state[:, :k+1], action[:, :k], action[:, k:],
                                      is_first=None if is_first is None else is_first[:, :k+1], sample=False)
    future_rgb, future_state = rgb[:, k+1:], state[:, k+1:]
    weights = 1. + config.change_weight * (future_rgb - rgb[:, k:k+1]).abs().mean(2, keepdim=True)
    frame = ((imagined['rgb'] - future_rgb).square() * weights).mean()
    state_loss = F.mse_loss(imagined['state_symlog'], symlog(future_state))
    object_loss = torch.zeros((), device=rgb.device)
    observation_loss = torch.zeros((), device=rgb.device)
    if imagined.get('object_state') is not None:
        if 'object_truth' not in batch or 'object_mean' not in normalization or 'object_scale' not in normalization:
            raise ValueError('RSSM object prediction requires normalized future object targets')
        object_mean=torch.as_tensor(normalization['object_mean'],device=rgb.device,dtype=torch.float32)
        object_scale=torch.as_tensor(normalization['object_scale'],device=rgb.device,dtype=torch.float32)
        target=(batch['object_truth'].to(rgb.device)[:,k+1:]-object_mean)/object_scale
        object_loss=F.mse_loss(imagined['object_state'],target)
        if imagined.get('object_observation_state') is not None:
            current=(batch['object_truth'].to(rgb.device)[:,k]-object_mean)/object_scale
            observation_loss=F.mse_loss(imagined['object_observation_state'],current)
    # Burn-in constructs carry; neither startup padding nor past context are loss targets.
    reconstruction = (F.mse_loss(observed['rgb'][:, k:], rgb[:, k:]) +
                      F.mse_loss(observed['state_symlog'][:, k:], symlog(state[:, k:])))
    dyn, rep = balanced_kl(observed['posterior_logits'], observed['prior_logits'],
                          free_nats=config.free_nats, unimix=member.config.unimix)
    dyn, rep = dyn[:, k:].mean(), rep[:, k:].mean()
    event = torch.zeros((), device=rgb.device)
    if imagined['event_logits'] is not None:
        mask = batch['event_mask'].to(rgb.device)
        values = F.binary_cross_entropy_with_logits(imagined['event_logits'],
                                                    batch['events'].to(rgb.device), reduction='none')
        event = (values * mask).sum() / mask.sum().clamp_min(1.)
    total = (config.frame_weight * frame + config.state_weight * state_loss +
             config.reconstruction_weight * reconstruction + config.dyn_weight * dyn +
             config.rep_weight * rep + config.event_weight * event + config.object_weight*object_loss+
             config.object_observation_weight*observation_loss)
    components = dict(frame=frame, state=state_loss, reconstruction=reconstruction,
                      dyn_kl=dyn, rep_kl=rep, event=event, object_state=object_loss,
                      object_observation=observation_loss,
                      prior_entropy=observed['prior_entropy'][:, k:].mean(),
                      posterior_entropy=observed['posterior_entropy'][:, k:].mean())
    return total, {name: float(value.detach()) for name, value in components.items()}
