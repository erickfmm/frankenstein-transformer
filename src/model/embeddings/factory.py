"""Positional encoding factory.

Dispatches construction of positional encoding modules based on the
``positional_encoding`` field of a model configuration object. The factory
returns an ``nn.Module`` (or ``None`` for the ``none``/``nope`` aliases, which
resolve to :class:`NoPE`) compatible with the ``pos_encoder(x,
logical_layer_idx=0)`` call signature used throughout the attention blocks.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn


def build_pos_encoder(config) -> Optional[nn.Module]:
    """Build a positional encoding module from a model configuration.

    Dispatches on ``config.positional_encoding`` (case-insensitive). Unknown
    values raise ``ValueError``. The returned module exposes the
    ``forward(self, x, logical_layer_idx=0)`` interface; ALiBi, PaPE and
    learned/sinusoidal absolute encodings additionally expose ``add`` or
    ``encode_qk``/``bias`` methods for use by attention blocks that need
    additive or augmented positional information.

    Args:
        config: Configuration object exposing at least ``hidden_size``,
            ``num_heads`` and ``positional_encoding`` attributes. Additional
            optional attributes (``rope_base``, ``rope_scaling``,
            ``hope_base``, ``hope_damping``, ``alibi_num_heads``,
            ``pape_num_parabolas``, ``pape_num_positions``,
            ``sinusoidal_max_len``, ``sinusoidal_base``,
            ``sinusoidal_scale``, ``learned_max_len``, ``learned_init_std``)
            are read with sensible defaults via ``getattr``.

    Returns:
        A positional encoding ``nn.Module``, or ``None``-style identity module
        for the ``none``/``nope`` aliases.

    Raises:
        ValueError: If ``config.positional_encoding`` is not a recognized
            value.
    """
    pe = str(getattr(config, "positional_encoding", "rope")).lower()
    head_dim = config.hidden_size // config.num_heads
    if pe == "none" or pe == "nope":
        from .nope import NoPE

        return NoPE()
    elif pe == "rope":
        from .rope import RoPE

        return RoPE(
            head_dim,
            base=getattr(config, "rope_base", 10_000.0),
            scaling=getattr(config, "rope_scaling", 1.0),
        )
    elif pe == "hope":
        from .hope import HoPE

        return HoPE(
            head_dim,
            base=getattr(config, "hope_base", 10_000.0),
            damping=getattr(config, "hope_damping", 0.01),
        )
    elif pe == "alibi":
        from .alibi import ALiBi

        return ALiBi(getattr(config, "alibi_num_heads", config.num_heads))
    elif pe == "pape":
        from .pape import PaPE

        return PaPE(
            config.hidden_size,
            config.num_heads,
            head_dim,
            num_parabolas=getattr(config, "pape_num_parabolas", 4),
            num_positions=getattr(config, "pape_num_positions", 1),
        )
    elif pe == "pape_efficient":
        from .pape_efficient import PaPEEfficient

        return PaPEEfficient(
            config.hidden_size,
            config.num_heads,
            head_dim,
            num_parabolas=getattr(config, "pape_num_parabolas", 4),
            num_positions=getattr(config, "pape_num_positions", 1),
        )
    elif pe == "pape_ri":
        from .pape_ri import PaPERI

        return PaPERI(
            config.hidden_size,
            config.num_heads,
            head_dim,
            num_positions=getattr(config, "pape_num_positions", 2),
        )
    elif pe == "sinusoidal_absolute":
        from .sinusoidal import SinusoidalAbsolute

        return SinusoidalAbsolute(
            config.hidden_size,
            max_len=getattr(config, "sinusoidal_max_len", 512),
            base=getattr(config, "sinusoidal_base", 10_000.0),
            scale=getattr(config, "sinusoidal_scale", 1.0),
        )
    elif pe == "sinusoidal_rotary":
        from .sinusoidal import SinusoidalRotary

        return SinusoidalRotary(
            head_dim,
            max_len=getattr(config, "sinusoidal_max_len", 512),
            base=getattr(config, "sinusoidal_base", 10_000.0),
            scale=getattr(config, "sinusoidal_scale", 1.0),
        )
    elif pe == "learned_absolute":
        from .learned_absolute import LearnedAbsolutePE

        return LearnedAbsolutePE(
            config.hidden_size,
            max_len=getattr(config, "learned_max_len", 512),
            init_std=getattr(config, "learned_init_std", 0.02),
        )
    else:
        raise ValueError(f"Unknown positional_encoding: {pe}")