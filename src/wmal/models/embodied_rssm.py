"""Scalable categorical RSSM: real posterior context and action-only imagination.

All state, command and action tensors are already in provider/training coordinates.
No coordinate fitting or command integration belongs to this network. Sampling
uses common random numbers across batch rows (candidate branches), indexed by
seed, posterior/prior stream and episode-relative step. It never uses global RNG.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .categorical_rssm import categorical_probs, symlog
from .embodied_blocks import GroupedRecurrentCore, SharedViewDecoder, SharedViewEncoder, mlp
from .embodied_config import PARAMETER_BUDGETS, ScalableConfig


class ScalableRSSMMember(nn.Module):
    def __init__(self, config: ScalableConfig):
        super().__init__()
        self.config = config
        cfg = config
        latent_dim = cfg.stoch * cfg.classes
        feature_dim = cfg.deter + latent_dim
        self.encoder = SharedViewEncoder(cfg)
        self.physical_encoder = mlp(cfg.physical_dim, cfg.fusion_width, cfg.fusion_width, eps=cfg.norm_eps)
        self.command_encoder = (mlp(cfg.command_dim, cfg.fusion_width, cfg.fusion_width, eps=cfg.norm_eps)
                                if cfg.command_dim else None)
        fusion_input = cfg.fusion_width * (cfg.views + 1 + bool(cfg.command_dim)) + cfg.views
        self.fusion = mlp(fusion_input, cfg.fusion_width, cfg.fusion_width, eps=cfg.norm_eps)
        self.action_encoder = mlp(cfg.action_dim + 2 * cfg.command_dim, cfg.hidden, cfg.hidden, eps=cfg.norm_eps)
        self.transition = GroupedRecurrentCore(latent_dim, cfg)
        self.prior = mlp(cfg.deter, cfg.hidden, latent_dim, eps=cfg.norm_eps)
        self.posterior = mlp(cfg.deter + cfg.fusion_width, cfg.hidden, latent_dim, eps=cfg.norm_eps)
        self.physical_head = mlp(feature_dim, cfg.hidden, cfg.physical_dim, eps=cfg.norm_eps)
        self.event_head = nn.Linear(feature_dim, cfg.event_dim) if cfg.event_dim else None
        self.decoder = SharedViewDecoder(feature_dim, cfg)

    def _validate_float(self, value, shape, name):
        parameter = next(self.parameters())
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or not value.is_floating_point() or value.device != parameter.device
                or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
            raise ValueError(f'{name} must be finite floating {tuple(shape)} on the model device/dtype')

    def _context(self, rgb, physical, commands, actions, view_mask, is_first):
        cfg = self.config
        if not isinstance(physical, torch.Tensor) or physical.ndim != 3:
            raise ValueError('physical must be [B,T,P]')
        batch, steps = physical.shape[:2]
        if batch < 1 or steps < 1:
            raise ValueError('Real context must contain at least one observation')
        self._validate_float(physical, (batch, steps, cfg.physical_dim), 'physical')
        self._validate_float(commands, (batch, steps, cfg.command_dim), 'commands')
        self._validate_float(actions, (batch, steps - 1, cfg.action_dim), 'past actions')
        self._validate_float(rgb, (batch, steps, cfg.views, 3, cfg.image_size, cfg.image_size), 'rgb')
        if (rgb < 0).any() or (rgb > 1).any():
            raise ValueError('RGB must be in [0,1]')
        if view_mask is None:
            view_mask = torch.ones(batch, steps, cfg.views, dtype=torch.bool, device=physical.device)
        if (not isinstance(view_mask, torch.Tensor) or view_mask.shape != (batch, steps, cfg.views)
                or view_mask.dtype != torch.bool or view_mask.device != physical.device
                or not view_mask.any(-1).all()):
            raise ValueError('view_mask must be boolean [B,T,V] with at least one real view per step')
        if is_first is None:
            is_first = torch.zeros(batch, steps, dtype=torch.bool, device=physical.device)
        if (not isinstance(is_first, torch.Tensor) or is_first.shape != (batch, steps)
                or is_first.dtype != torch.bool or is_first.device != physical.device):
            raise ValueError('is_first must be a boolean [B,T] reset mask')
        # Every call has a clean origin even when a caller passes False at t=0.
        is_first = is_first.clone()
        is_first[:, 0] = True
        return view_mask, is_first

    def _sampling(self, sample, seed):
        if type(sample) is not bool or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('sample must be boolean and seed an integer in [0,2**63)')

    def _latent(self, logits, sample, seed, ages, stream):
        # Probabilities/entropy stay FP32 under autocast, with ST returned in core dtype.
        probs = categorical_probs(logits.float(), self.config.unimix)
        if not sample:
            return probs.flatten(-2).to(logits.dtype)
        uniforms = torch.empty(logits.shape[0], self.config.stoch, 1, device=logits.device)
        for age in ages.unique().tolist():
            generator = torch.Generator(device=logits.device)
            generator.manual_seed((seed + 1_000_003 * stream + 104_729 * age) % (2**63))
            noise = torch.rand(1, self.config.stoch, 1, device=logits.device, generator=generator)
            uniforms[ages == age] = noise
        indices = (uniforms > probs.cumsum(-1)).sum(-1).clamp_max(self.config.classes - 1)
        hard = F.one_hot(indices, self.config.classes).to(probs.dtype)
        return (hard + (probs - probs.detach())).flatten(-2).to(logits.dtype)

    def _entropy(self, logits):
        probs = categorical_probs(logits.float(), self.config.unimix)
        return -torch.special.xlogy(probs, probs).sum((-1, -2))

    def _logits(self, head, value):
        return head(value).reshape(*value.shape[:-1], self.config.stoch, self.config.classes)

    def _token(self, rgb, physical, commands, mask):
        parts = [self.encoder(rgb, mask), self.physical_encoder(symlog(physical))]
        if self.command_encoder is not None:
            parts.append(self.command_encoder(symlog(commands)))
        parts.append(mask.to(physical.dtype))
        return self.fusion(torch.cat(parts, -1))

    def _advance(self, h, z, action, current_command, next_command):
        token = self.action_encoder(torch.cat((symlog(action), symlog(current_command), symlog(next_command)), -1))
        return self.transition(z, token, h)

    def _decode(self, features, physical_origin, rgb_origin):
        delta = self.physical_head(features)
        return dict(physical=physical_origin + delta, physical_delta=delta,
                    rgb=self.decoder(features, rgb_origin),
                    event_logits=self.event_head(features) if self.event_head is not None else None)

    def _observe(self, rgb, physical, commands, actions, mask, resets, sample, seed):
        batch, steps = physical.shape[:2]
        h = physical.new_zeros(batch, self.config.deter)
        z = physical.new_zeros(batch, self.config.stoch * self.config.classes)
        ages = torch.zeros(batch, dtype=torch.long, device=physical.device)
        features, priors, posts, physical_origins, rgb_origins = [], [], [], [], []
        physical_origin, rgb_origin = physical[:, 0], rgb[:, 0]
        for index in range(steps):
            reset = resets[:, index]
            h, z = h.masked_fill(reset[:, None], 0), z.masked_fill(reset[:, None], 0)
            ages = ages.masked_fill(reset, 0)
            action = physical.new_zeros(batch, self.config.action_dim) if index == 0 else actions[:, index-1]
            action = action.masked_fill(reset[:, None], 0)
            previous = commands[:, max(0, index - 1)]
            previous = torch.where(reset[:, None], commands[:, index], previous)
            h = self._advance(h, z, action, previous, commands[:, index])
            prior = self._logits(self.prior, h)
            token = self._token(rgb[:, index], physical[:, index], commands[:, index], mask[:, index])
            post = self._logits(self.posterior, torch.cat((h, token), -1))
            z = self._latent(post, sample, seed, ages, stream=0)
            features.append(torch.cat((h, z), -1))
            priors.append(prior)
            posts.append(post)
            physical_origin = torch.where(reset[:, None], physical[:, index], physical_origin)
            rgb_origin = torch.where(reset[:, None, None, None, None], rgb[:, index], rgb_origin)
            physical_origins.append(physical_origin)
            rgb_origins.append(rgb_origin)
            ages = ages + 1
        return (torch.stack(features, 1), torch.stack(priors, 1), torch.stack(posts, 1),
                torch.stack(physical_origins, 1), torch.stack(rgb_origins, 1))

    def observe(self, rgb, physical, commands, actions, *, view_mask=None, is_first=None, sample=True, seed=0):
        """Reconstruct real T observations with exactly T-1 intervening actions.

        Prior at t may use known commands at t, but never RGB/physical at t.
        A reset clears both latent states, incoming action and old command carry.
        """
        self._sampling(sample, seed)
        mask, resets = self._context(rgb, physical, commands, actions, view_mask, is_first)
        features, prior, post, p0, rgb0 = self._observe(rgb, physical, commands, actions, mask, resets, sample, seed)
        result = self._decode(features, p0, rgb0)
        result.update(features=features, prior_logits=prior, posterior_logits=post,
                      prior_entropy=self._entropy(prior), posterior_entropy=self._entropy(post))
        return result

    def imagine_context(self, rgb, physical, commands, past_actions, actions, future_commands,
                        *, view_mask=None, is_first=None, sample=True, seed=0):
        """Only bounded real history, proposed actions and provider-computed commands.

        Physical deltas are relative to the last measured state, never accumulated.
        Residual RGB uses the last real image, never a future observation.
        Different rows share categorical uniforms for candidate comparison; use
        distinct per-call seeds for independent member/sample trajectories.
        """
        self._sampling(sample, seed)
        mask, resets = self._context(rgb, physical, commands, past_actions, view_mask, is_first)
        batch, steps = physical.shape[:2]
        if steps > self.config.context_steps + 1:
            raise ValueError('Real history exceeds configured context_steps+1')
        if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
            raise ValueError('Future actions must be [B,H,A]')
        horizon = actions.shape[1]
        if not 1 <= horizon <= self.config.horizon:
            raise ValueError('Rollout horizon must be within the configured bound')
        self._validate_float(actions, (batch, horizon, self.config.action_dim), 'future actions')
        self._validate_float(future_commands, (batch, horizon, self.config.command_dim), 'future commands')
        start, _, post, _, _ = self._observe(rgb, physical, commands, past_actions, mask, resets, sample, seed)
        h, z = start[:, -1, :self.config.deter], start[:, -1, self.config.deter:]
        features, priors = [], []
        current = commands[:, -1]
        for index in range(horizon):
            h = self._advance(h, z, actions[:, index], current, future_commands[:, index])
            prior = self._logits(self.prior, h)
            ages = torch.full((batch,), index, dtype=torch.long, device=h.device)
            z = self._latent(prior, sample, seed, ages, stream=1)
            features.append(torch.cat((h, z), -1))
            priors.append(prior)
            current = future_commands[:, index]
        features, priors = torch.stack(features, 1), torch.stack(priors, 1)
        result = self._decode(features, physical[:, -1:], rgb[:, -1:])
        result.update(features=features, prior_logits=priors, prior_entropy=self._entropy(priors),
                      posterior_logits=post[:, -1], posterior_entropy=self._entropy(post[:, -1]))
        return result


def parameter_report(config):
    """Count all unique weights on meta; no giant CPU/CUDA weights or RNG draws."""
    with torch.device('meta'):
        member = ScalableRSSMMember(config)
    unique = {id(parameter): parameter for parameter in member.parameters()}
    total = sum(parameter.numel() for parameter in unique.values())
    trainable = sum(parameter.numel() for parameter in unique.values() if parameter.requires_grad)
    lower, upper = PARAMETER_BUDGETS[config.preset]
    return dict(total=total, trainable=trainable, frozen=total-trainable,
                preset=config.preset, budget_min=lower, budget_max=upper,
                within_budget=lower <= total <= upper, device='meta')
