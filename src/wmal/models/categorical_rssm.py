"""Compact, independently implemented categorical recurrent state-space model.

Mechanisms informed by DreamerV3 (see the design source table), not its weights,
block GRU or actor-critic. Observation updates and action-only imagination have
different APIs. Soft categorical deployment is repeatable, NOT an exact expected
stochastic trajectory, and entropy is NOT a calibrated error/success probability.
"""
import torch
from torch import nn
from torch.nn import functional as F


def symlog(value):
    return value.sign() * torch.log1p(value.abs())


def symexp(value):
    return value.sign() * torch.expm1(value.abs())


def categorical_probs(logits, unimix):
    return (1. - unimix) * logits.softmax(-1) + unimix / logits.shape[-1]


def balanced_kl(posterior, prior, *, free_nats, unimix):
    """Sum over independent variables; dynamics and representation gradients split."""
    q, p = categorical_probs(posterior, unimix), categorical_probs(prior, unimix)
    dyn = (q.detach() * (q.detach().log() - p.log())).sum((-1, -2))
    rep = (q * (q.log() - p.detach().log())).sum((-1, -2))
    return dyn.clamp_min(free_nats), rep.clamp_min(free_nats)


class CategoricalRSSMMember(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        z, h = config.stoch * config.classes, config.hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(512, config.latent_dim), nn.LayerNorm(config.latent_dim))
        self.state_encoder = nn.Sequential(nn.Linear(config.state_dim, config.latent_dim),
                                           nn.LayerNorm(config.latent_dim), nn.SiLU())
        self.transition = nn.GRUCell(z + config.action_dim, h)
        self.prior = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU(), nn.Linear(h, z))
        self.posterior = nn.Sequential(nn.Linear(h + 2*config.latent_dim, h),
                                       nn.LayerNorm(h), nn.SiLU(), nn.Linear(h, z))
        self.state_head = nn.Sequential(nn.Linear(h + z, h), nn.SiLU(), nn.Linear(h, config.state_dim))
        self.event_head = nn.Linear(h + z, config.event_dim) if config.event_dim else None
        self.decoder_input = nn.Linear(h + z, 32*4*4)
        layers, size, channels = [], 4, 32
        while size < config.image_size:
            output = 3 if size*2 == config.image_size else 16
            layers.extend([nn.ConvTranspose2d(channels, output, 4, 2, 1),
                           nn.Sigmoid() if output == 3 else nn.SiLU()])
            size, channels = size*2, output
        self.decoder = nn.Sequential(*layers)

    def _logits(self, head, features):
        return head(features).reshape(*features.shape[:-1], self.config.stoch, self.config.classes)

    def _latent(self, logits, sample):
        probabilities = categorical_probs(logits, self.config.unimix)
        if not sample:
            return probabilities.flatten(-2)
        indices = torch.distributions.Categorical(probs=probabilities).sample()
        one_hot = F.one_hot(indices, self.config.classes).to(probabilities.dtype)
        return (one_hot + probabilities - probabilities.detach()).flatten(-2)

    def _entropy(self, logits):
        probabilities = categorical_probs(logits, self.config.unimix)
        return -(probabilities * probabilities.log()).sum((-1, -2))

    def _token(self, rgb, state):
        return torch.cat([self.encoder(rgb), self.state_encoder(symlog(state))], -1)

    def _initial(self, state):
        return (state.new_zeros(len(state), self.config.hidden_dim),
                state.new_zeros(len(state), self.config.stoch*self.config.classes))

    def _decode(self, features):
        shape = features.shape[:-1]
        frame = self.decoder(self.decoder_input(features).reshape(-1, 32, 4, 4))
        return {'rgb': frame.reshape(*shape, 3, self.config.image_size, self.config.image_size),
                'state_symlog': self.state_head(features),
                'event_logits': self.event_head(features) if self.event_head is not None else None}

    def observe(self, rgb, states, actions, *, is_first=None, sample=False):
        """Observe T actual frames with T-1 intervening actions, no cross-reset carry.

        Calls start with a clean carry. is_first[t] resets both carry and incoming
        action (including t=0). These are real frames, never imagined observations.
        """
        b, t = states.shape[:2]
        if (t < 1 or states.shape != (b, t, self.config.state_dim)
                or actions.shape != (b, t-1, self.config.action_dim)
                or rgb.shape != (b, t, 3, self.config.image_size, self.config.image_size)):
            raise ValueError('RSSM expects aligned real observation sequence and T-1 actions')
        if is_first is None:
            is_first = torch.zeros((b, t), dtype=torch.bool, device=states.device)
            is_first[:, 0] = True
        if is_first.shape != (b, t) or is_first.dtype != torch.bool:
            raise ValueError('is_first must be a boolean [B,T] episode reset mask')
        h, z = self._initial(states[:, 0])
        features, priors, posts = [], [], []
        for i in range(t):
            reset = is_first[:, i, None]
            incoming = states.new_zeros(b, self.config.action_dim) if i == 0 else actions[:, i-1]
            h = h.masked_fill(reset, 0.)
            z = z.masked_fill(reset, 0.)
            incoming = incoming.masked_fill(reset, 0.)
            h = self.transition(torch.cat([z, incoming], -1), h)
            prior = self._logits(self.prior, h)
            post = self._logits(self.posterior, torch.cat([h, self._token(rgb[:, i], states[:, i])], -1))
            z = self._latent(post, sample)
            features.append(torch.cat([h, z], -1))
            priors.append(prior)
            posts.append(post)
        features = torch.stack(features, 1)
        prior, post = torch.stack(priors, 1), torch.stack(posts, 1)
        result = self._decode(features)
        result.update(features=features, prior_logits=prior, posterior_logits=post,
                      prior_entropy=self._entropy(prior), posterior_entropy=self._entropy(post))
        return result

    def imagine(self, rgb, state, actions, *, sample=False):
        """Only initial real observation is consumed; future steps use the prior."""
        if actions.ndim != 3 or actions.shape[0] != len(state) or actions.shape[1] < 1:
            raise ValueError('RSSM candidate actions must have shape [B,H,A], H>=1')
        start = self.observe(rgb[:, None], state[:, None], actions[:, :0], sample=sample)
        feature = start['features'][:, 0]
        h, z = feature[:, :self.config.hidden_dim], feature[:, self.config.hidden_dim:]
        features, entropy = [], []
        for action in actions.unbind(1):
            h = self.transition(torch.cat([z, action], -1), h)
            logits = self._logits(self.prior, h)
            z = self._latent(logits, sample)
            features.append(torch.cat([h, z], -1))
            entropy.append(self._entropy(logits))
        result = self._decode(torch.stack(features, 1))
        result.update(state=symexp(result['state_symlog']),
                      prior_entropy=torch.stack(entropy, 1),
                      posterior_entropy=start['posterior_entropy'][:, 0])
        return result
