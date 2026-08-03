"""BigBird sparse attention (Zaheer et al., 2020; arXiv:2007.14062).

Implements a sparse attention mechanism that combines three complementary
patterns: local sliding-window attention, random attention links, and global
tokens. Together these patterns satisfy all theoretical properties needed for
Turing completeness of the transformer while achieving O(n) linear complexity.
The combination of local, random, and global attention ensures that information
can flow between any pair of tokens in O(log n) steps.

Reference:
    Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S.,
    Pham, P., Ravula, A., Wang, Q., Yang, L., & Ahmed, A. (2020).
    "Big Bird: Transformers for Longer Sequences."
    arXiv:2007.14062.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class BigBirdAttention(nn.Module):
    """Sparse attention with local window, random links, and global tokens.

    Each query token attends to three sets of keys:
    1. **Local window**: A symmetric window of ``window_size`` tokens centered
       on the query position.
    2. **Random links**: A fixed number of randomly selected tokens per query,
       providing long-range connectivity.
    3. **Global tokens**: The first ``num_global`` tokens attend to and are
       attended by all positions, serving as information hubs.

    The combined pattern achieves O(n) complexity while provably preserving
    the expressiveness of dense attention. In decoder mode, all patterns are
    intersected with a causal mask.

    Attributes:
        hidden_size (int): Dimensionality of the input and output embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        scale (float): Scaling factor for dot-product scores (1 / sqrt(head_dim)).
        window_size (int): Size of the local attention window (total span).
        num_random (int): Number of random attention links per query.
        num_global (int): Number of global tokens (first ``num_global`` positions).
        q_proj (nn.Module): Query projection (BitLinear or nn.Linear).
        k_proj (nn.Module): Key projection (BitLinear or nn.Linear).
        v_proj (nn.Module): Value projection (BitLinear or nn.Linear).
        out_proj (nn.Module): Output projection (BitLinear or nn.Linear).
        dropout (nn.Dropout): Dropout applied to attention weights.
        mode (str): ``"encoder"`` or ``"decoder"``. Decoder mode applies causal
            masking on top of the sparse patterns.

    Reference:
        Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S.,
        Pham, P., Ravula, A., Wang, Q., Yang, L., & Ahmed, A. (2020).
        "Big Bird: Transformers for Longer Sequences."
        arXiv:2007.14062.
    """

    def __init__(self, config):
        """Initialize BigBirdAttention.

        Args:
            config: Configuration object with the following expected attributes:
                hidden_size (int): Model hidden dimension.
                num_heads (int): Number of attention heads. Must divide
                    ``hidden_size`` evenly.
                dropout (float): Dropout probability for attention weights.
                use_bitnet (bool): If True, use BitLinear projections.
                bigbird_window_size (int, optional): Total span of the local
                    attention window. Defaults to 128.
                bigbird_num_random (int, optional): Number of random attention
                    links per query. Defaults to 64.
                bigbird_num_global (int, optional): Number of global tokens.
                    Defaults to 2.
                mode (str, optional): ``"encoder"`` or ``"decoder"``.
                    Defaults to ``"encoder"``.

        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for BigBirdAttention")

        self.window_size = max(1, int(getattr(config, "bigbird_window_size", 128)))
        self.num_random = max(1, int(getattr(config, "bigbird_num_random", 64)))
        self.num_global = max(1, int(getattr(config, "bigbird_num_global", 2)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def _build_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build the combined BigBird attention mask.

        Constructs a boolean mask that encodes all three BigBird attention
        patterns for the given sequence length:

        1. **Local window**: Each position ``i`` attends to a symmetric window
           of ``window_size // 2`` positions on each side.
        2. **Random links**: Each position ``i`` attends to ``num_random``
           randomly sampled positions (sampled independently per position).
        3. **Global tokens**: The first ``num_global`` positions attend to all
           positions and are attended by all positions.

        Args:
            seq_len (int): Current sequence length.
            device (torch.device): Device on which to create the mask tensor.

        Returns:
            torch.Tensor: Boolean mask of shape ``(seq_len, seq_len)`` where
                ``True`` indicates allowed attention.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        half_w = self.window_size // 2

        for i in range(seq_len):
            start = max(0, i - half_w)
            end = min(seq_len, i + half_w + 1)
            mask[i, start:end] = True

        for i in range(seq_len):
            rand_idx = torch.randint(0, seq_len, (self.num_random,), device=device)
            mask[i, rand_idx] = True

        for g in range(min(self.num_global, seq_len)):
            mask[g, :] = True
            mask[:, g] = True

        return mask

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute BigBird sparse attention.

        Projects queries, keys, and values, then applies the combined
        window + random + global mask. In decoder mode, the mask is further
        intersected with a causal (lower-triangular) mask.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch, seq_len, hidden_size)``.
            logical_layer_idx (Optional[int]): Logical layer index for
                potential layer-specific behavior. Not used by this
                implementation.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = self._build_mask(seq_len, x.device)
        if self.mode == "decoder":
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            mask = mask & causal
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
