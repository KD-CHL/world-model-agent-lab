"""Posterior representation learning + deployment-matched prior-only supervision."""
import torch
from torch.nn import functional as F

from wmal.models.categorical_rssm import balanced_kl, symlog


def rssm_loss(member, rgb, state, action, batch, config):
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
             config.rep_weight * rep + config.event_weight * event)
    components = dict(frame=frame, state=state_loss, reconstruction=reconstruction,
                      dyn_kl=dyn, rep_kl=rep, event=event,
                      prior_entropy=observed['prior_entropy'][:, k:].mean(),
                      posterior_entropy=observed['posterior_entropy'][:, k:].mean())
    return total, {name: float(value.detach()) for name, value in components.items()}
