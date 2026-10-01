"""Posterior representation learning + deployment-matched prior-only supervision."""
import torch
from torch.nn import functional as F

from wmal.models.categorical_rssm import balanced_kl, symlog


def rssm_loss(member, rgb, state, action, batch, config):
    # Posterior sampling only during training. Validation uses the same repeatable
    # soft categorical proxy as deployed prediction and independent calibration.
    observed = member.observe(rgb, state, action, sample=member.training)
    imagined = member.imagine(rgb[:, 0], state[:, 0], action, sample=False)
    weights = 1. + config.change_weight * (rgb[:, 1:] - rgb[:, :1]).abs().mean(2, keepdim=True)
    frame = ((imagined['rgb'] - rgb[:, 1:]).square() * weights).mean()
    state_loss = F.mse_loss(imagined['state_symlog'], symlog(state[:, 1:]))
    reconstruction = (F.mse_loss(observed['rgb'], rgb) +
                      F.mse_loss(observed['state_symlog'], symlog(state)))
    dyn, rep = balanced_kl(observed['posterior_logits'], observed['prior_logits'],
                          free_nats=config.free_nats, unimix=member.config.unimix)
    dyn, rep = dyn.mean(), rep.mean()
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
                      prior_entropy=observed['prior_entropy'].mean(),
                      posterior_entropy=observed['posterior_entropy'].mean())
    return total, {name: float(value.detach()) for name, value in components.items()}
