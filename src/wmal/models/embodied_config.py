"""Validated dimensions for the independent, normalized embodied model family."""
from dataclasses import dataclass
import math


PARAMETER_BUDGETS = {
    'smoke': (1, 1_000_000),
    'small': (30_000_000, 50_000_000),
    'base': (120_000_000, 180_000_000),
    'large': (240_000_000, 300_000_000),
}


@dataclass(frozen=True)
class ScalableConfig:
    physical_dim: int = 2
    command_dim: int = 2
    action_dim: int = 2
    views: int = 1
    image_size: int = 128
    event_dim: int = 0
    # K connecting actions means at most K+1 real observations, including origin.
    context_steps: int = 8
    horizon: int = 8
    deter: int = 64
    hidden: int = 128
    stoch: int = 8
    classes: int = 8
    blocks: int = 4
    encoder_channels: tuple[int, ...] = (16, 32, 64)
    fusion_width: int = 64
    decoder_width: int = 32
    unimix: float = .01
    norm: str = 'layer'
    norm_eps: float = 1e-5
    rgb_mode: str = 'residual'
    preset: str = 'smoke'

    def __post_init__(self):
        positive = ('physical_dim', 'action_dim', 'views', 'image_size', 'horizon',
                    'deter', 'hidden', 'stoch', 'classes', 'blocks',
                    'fusion_width', 'decoder_width')
        for name in positive + ('command_dim', 'event_dim', 'context_steps'):
            value = getattr(self, name)
            lower = 1 if name in positive else 0
            if type(value) is not int or value < lower:
                raise ValueError(f'{name} must be an integer >= {lower}')
        if self.classes < 2:
            raise ValueError('Categorical variables require at least two classes')
        if self.image_size < 16 or self.image_size & (self.image_size - 1):
            raise ValueError('image_size must be a power of two >= 16')
        if self.deter % self.blocks or self.hidden % self.blocks:
            raise ValueError('deter and hidden must be divisible by blocks')
        if not isinstance(self.encoder_channels, (tuple, list)) or not self.encoder_channels:
            raise ValueError('encoder_channels must be a nonempty sequence')
        if any(type(value) is not int or value < 1 for value in self.encoder_channels):
            raise ValueError('encoder_channels must contain positive integer widths')
        if 2**len(self.encoder_channels) > self.image_size:
            raise ValueError('Encoder cannot downsample beyond one pixel')
        object.__setattr__(self, 'encoder_channels', tuple(self.encoder_channels))
        if (isinstance(self.unimix, bool) or not isinstance(self.unimix, (int, float))
                or not math.isfinite(self.unimix) or not 0 <= self.unimix < 1):
            raise ValueError('unimix must be finite in [0,1)')
        if (self.norm != 'layer' or isinstance(self.norm_eps, bool)
                or not isinstance(self.norm_eps, (int, float))
                or not math.isfinite(self.norm_eps) or self.norm_eps <= 0):
            raise ValueError('Only per-sample layer normalization with positive epsilon is supported')
        if self.rgb_mode not in ('full', 'residual'):
            raise ValueError('rgb_mode must be full or residual')
        if self.preset not in PARAMETER_BUDGETS:
            raise ValueError('Unknown preset budget')


def network_preset(name, **semantic_dimensions):
    """Meaningful capacity presets; semantic overrides also affect budget counts.

    Counts refer to the default 2D joint profile, one 128px camera and no events.
    Explicit wider profiles must recheck parameter_report before allocation.
    """
    widths = {
        'smoke': dict(deter=64, hidden=128, stoch=8, classes=8, blocks=4,
                      encoder_channels=(16, 32, 64), fusion_width=64, decoder_width=32,
                      image_size=32, context_steps=2, horizon=4),
        'small': dict(deter=1024, hidden=1536, stoch=16, classes=32, blocks=8,
                      encoder_channels=(64, 128, 256, 512), fusion_width=512, decoder_width=256),
        'base': dict(deter=2048, hidden=3072, stoch=32, classes=32, blocks=8,
                     encoder_channels=(96, 192, 384, 768), fusion_width=1024, decoder_width=512),
        'large': dict(deter=3072, hidden=4096, stoch=48, classes=32, blocks=8,
                      encoder_channels=(128, 256, 512, 1024), fusion_width=1536, decoder_width=768),
    }
    if name not in widths:
        raise ValueError(f'Unknown network preset: {name}')
    semantic_fields = {'physical_dim', 'command_dim', 'action_dim', 'views',
                       'image_size', 'event_dim', 'context_steps', 'horizon', 'rgb_mode'}
    if semantic_dimensions.keys() - semantic_fields:
        raise ValueError('Preset overrides must describe observation/action semantics')
    return ScalableConfig(**(widths[name] | semantic_dimensions), preset=name)
