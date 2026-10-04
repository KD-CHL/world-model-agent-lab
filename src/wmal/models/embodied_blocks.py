"""Independent shared vision and grouped recurrent building blocks."""
import math

import torch
from torch import nn


def mlp(input_dim, width, output_dim, *, eps):
    return nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width, eps=eps), nn.SiLU(),
                         nn.Linear(width, width), nn.LayerNorm(width, eps=eps), nn.SiLU(),
                         nn.Linear(width, output_dim))


class GroupLinear(nn.Module):
    """Independent dense maps within groups, without a padded dense matrix."""
    def __init__(self, input_dim, output_dim, groups):
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(groups, output_dim // groups, input_dim // groups))
        self.bias = nn.Parameter(torch.empty(groups, output_dim // groups))
        bound = 1 / math.sqrt(input_dim // groups)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, value):
        grouped = value.reshape(*value.shape[:-1], self.groups, -1)
        return torch.einsum('...gi,goi->...go', grouped, self.weight) + self.bias


class GroupedRecurrentCore(nn.Module):
    """Global input mixing followed by normalized, independently gated groups."""
    def __init__(self, latent_dim, config):
        super().__init__()
        self.mix = nn.Sequential(nn.Linear(latent_dim + config.hidden, config.hidden),
                                 nn.LayerNorm(config.hidden, eps=config.norm_eps), nn.SiLU())
        self.input_gates = GroupLinear(config.hidden, 3 * config.deter, config.blocks)
        self.state_gates = GroupLinear(config.deter, 3 * config.deter, config.blocks)
        self.norm = nn.LayerNorm(config.deter, eps=config.norm_eps)

    def forward(self, latent, action_token, state):
        token = self.mix(torch.cat((latent, action_token), -1))
        xr, xu, xn = self.input_gates(token).chunk(3, -1)
        hr, hu, hn = self.state_gates(state).chunk(3, -1)
        reset, update = (xr + hr).sigmoid(), (xu + hu).sigmoid()
        proposal = (xn + reset * hn).tanh()
        next_state = update * state.reshape_as(proposal) + (1 - update) * proposal
        return self.norm(next_state.flatten(-2))


class SharedViewEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers, previous = [], 3
        for channels in config.encoder_channels:
            layers.extend((nn.Conv2d(previous, channels, 4, 2, 1), nn.SiLU()))
            previous = channels
        layers.extend((nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                       nn.Linear(previous * 16, config.fusion_width),
                       nn.LayerNorm(config.fusion_width, eps=config.norm_eps)))
        self.backbone = nn.Sequential(*layers)
        self.view_embedding = nn.Embedding(config.views, config.fusion_width)

    def forward(self, rgb, mask):
        batch, views = rgb.shape[:2]
        token = self.backbone(rgb.reshape(-1, *rgb.shape[2:])).reshape(batch, views, -1)
        token = token + self.view_embedding.weight[None]
        weights = mask.to(token.dtype)[..., None]
        # Keep named camera slots through fusion. Averaging additive IDs would
        # make swapping two real camera images invisible to the posterior.
        return (token * weights).flatten(1)


class SharedViewDecoder(nn.Module):
    def __init__(self, feature_dim, config):
        super().__init__()
        self.config = config
        self.view_embedding = nn.Embedding(config.views, feature_dim)
        self.project = nn.Linear(feature_dim, config.decoder_width * 16)
        layers, size, channels = [], 4, config.decoder_width
        while size < config.image_size:
            next_channels = 3 if size * 2 == config.image_size else max(16, channels // 2)
            layers.append(nn.ConvTranspose2d(channels, next_channels, 4, 2, 1))
            if next_channels != 3:
                layers.append(nn.SiLU())
            size, channels = size * 2, next_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, features, reference):
        cfg = self.config
        conditioned = features[..., None, :] + self.view_embedding.weight
        hidden = self.project(conditioned).reshape(-1, cfg.decoder_width, 4, 4)
        raw = self.layers(hidden).reshape(*features.shape[:-1], cfg.views, 3, cfg.image_size, cfg.image_size)
        if cfg.rgb_mode == 'full':
            return raw.sigmoid()
        return (reference + raw.tanh()).clamp(0, 1)
