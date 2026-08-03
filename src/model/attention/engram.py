"""
Engram: Conditional Memory via Scalable Lookup
Paper: arXiv:2601.07372  (DeepSeek AI, 2026)

Engram augments a transformer with an N-gram conditional memory module that
performs O(1) deterministic lookup into a set of learnable embedding tables.
Each position retrieves a bigram/trigram (up to *max_ngram_size*) fingerprint,
fuses the retrieved vectors with the hidden state through a learned gate, and
returns a corrected hidden state.

This file adapts the official demo (deepseek-ai/Engram) to the Frankenstein
architecture:
  - Works with standard (B, T, D) tensors (no hyper-connection dimension).
  - Uses raw token IDs instead of a normalizing tokenizer wrapper.
  - Plugs in as a standard `layer_type` in HybridLayer.
"""

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BitLinear


# ---------------------------------------------------------------------------
# Primality helpers (used only at __init__ time, not on the hot path)
# ---------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    """Test whether an integer is prime using trial division up to sqrt(n).

    Handles small cases (n < 2, n < 4) directly, then checks divisibility by
    2 and 3 before iterating over candidates of the form 6k ± 1.

    Args:
        n: Integer to test for primality.

    Returns:
        ``True`` if ``n`` is prime, ``False`` otherwise.
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _next_prime(start: int, seen: set) -> int:
    """Return the smallest prime > *start* that is not in *seen*."""
    candidate = start + 1
    while not (_is_prime(candidate) and candidate not in seen):
        candidate += 1
    return candidate


# ---------------------------------------------------------------------------
# N-gram hasher (pure-numpy, runs on CPU)
# ---------------------------------------------------------------------------

class NgramHasher:
    """
    Computes deterministic N-gram hashes from raw token IDs.

    For each N-gram size n in [2, max_ngram_size] and each head j, the hash is:

        hash_j(i) = XOR(token[i-k] * multiplier[k]  for k in 0..n-1) % prime_j

    Multipliers are seeded odd random integers drawn at construction time.
    Each (n, j) pair uses a distinct prime modulus so hash collisions across
    heads are statistically independent.
    """

    def __init__(
        self,
        max_ngram_size: int,
        n_heads_per_ngram: int,
        base_vocab_size: int,
        seed: int = 42,
    ) -> None:
        self.max_ngram_size = max_ngram_size
        self.n_heads_per_ngram = n_heads_per_ngram

        # Seeded odd multipliers (length = max_ngram_size)
        rng = np.random.default_rng(seed)
        max_long = np.iinfo(np.int64).max
        half = max(1, (max_long // max(base_vocab_size, 1)) // 2)
        r = rng.integers(0, half, size=(max_ngram_size,), dtype=np.int64)
        self.multipliers: np.ndarray = r * 2 + 1  # ensure odd

        # Per-(n_gram, head) prime moduli starting just above base_vocab_size
        seen_primes: set = set()
        self.moduli: List[List[int]] = []
        for n in range(2, max_ngram_size + 1):
            heads: List[int] = []
            search_start = base_vocab_size - 1
            for _ in range(n_heads_per_ngram):
                p = _next_prime(search_start, seen_primes)
                seen_primes.add(p)
                heads.append(p)
                search_start = p
            self.moduli.append(heads)

    def hash(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Args:
            input_ids: int64 array of shape (B, T)

        Returns:
            hashes: int64 array of shape (B, T, total_heads)
            where total_heads = (max_ngram_size - 1) * n_heads_per_ngram
        """
        B, T = input_ids.shape

        # Precompute right-shifted (lagged) token arrays, padded with 0
        shifts = [input_ids]
        for k in range(1, self.max_ngram_size):
            padded = np.pad(input_ids, ((0, 0), (k, 0)), constant_values=0)[:, :T]
            shifts.append(padded)

        all_hashes: List[np.ndarray] = []
        for n_idx, n in enumerate(range(2, self.max_ngram_size + 1)):
            # XOR-mix for this n-gram order
            mix = shifts[0] * self.multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, shifts[k] * self.multipliers[k])
            # One hash index per head (different prime modulus)
            for j, mod in enumerate(self.moduli[n_idx]):
                all_hashes.append((mix % mod).astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)  # (B, T, total_heads)


# ---------------------------------------------------------------------------
# Multi-head embedding (single table with per-head offsets)
# ---------------------------------------------------------------------------

class MultiHeadEmbedding(nn.Module):
    """
    Packs *num_heads* separate embedding tables of sizes N_0, N_1, … into a
    single nn.Embedding by adding per-head offsets to the lookup indices.
    """

    def __init__(self, vocab_sizes: List[int], embed_dim: int) -> None:
        super().__init__()
        self.num_heads = len(vocab_sizes)
        offsets = [0]
        for n in vocab_sizes[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(sum(vocab_sizes), embed_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (B, T, num_heads) int64 – raw per-head hash indices

        Returns:
            (B, T, num_heads, embed_dim)
        """
        shifted = indices + self.offsets  # (B, T, H)
        return self.embedding(shifted)    # (B, T, H, D_head)


# ---------------------------------------------------------------------------
# Depthwise short convolution for local context fusion
# ---------------------------------------------------------------------------

class ShortConv(nn.Module):
    """
    Causal depthwise Conv1d that mixes the retrieved N-gram embeddings across a
    small local window before gating.  The causal padding ensures position i
    only sees N-gram information from positions ≤ i.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            dilation=dilation,
            padding=self.causal_pad,
            bias=False,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args / returns: (B, T, C)
        """
        out = self.conv(x.transpose(1, 2))  # (B, C, T + causal_pad)
        out = out[..., : x.size(1)]         # causal trim → (B, C, T)
        return F.silu(self.norm(out.transpose(1, 2)))


# ---------------------------------------------------------------------------
# EngramLayer
# ---------------------------------------------------------------------------

class EngramLayer(nn.Module):
    """
    Engram: Conditional Memory via Scalable Lookup  (arXiv:2601.07372).

    Given hidden states ``x`` of shape (B, T, hidden_size) and the original
    token IDs (B, T), the module:

    1. Hashes token N-grams into embedding-table indices.
    2. Looks up per-head N-gram embeddings and concatenates them.
    3. Passes the concatenated embedding through a causal depthwise convolution.
    4. Computes a scalar gate: σ(dot(norm(key), norm(query)) / √D).
    5. Returns ``gate ⊙ value_proj(engram_emb)`` projected to hidden_size.

    The layer is registered as ``"engram_attn"`` in HybridLayer and can appear
    at any position in *layer_pattern*.  When no input_ids are available the
    module returns a zero tensor (graceful degradation).

    Config knobs (all optional, sensible defaults):
      - engram_max_ngram_size  (int, default 3): highest N-gram order (2 and 3).
      - engram_n_heads_per_ngram (int, default 4): hash heads per N-gram order.
      - engram_embed_dim_per_head (int, default 32): embedding dim per head.
      - engram_kernel_size (int, default 4): ShortConv kernel width.
      - engram_seed (int, default 42): RNG seed for hash multipliers.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        max_ngram = getattr(config, "engram_max_ngram_size", 3)
        n_heads = getattr(config, "engram_n_heads_per_ngram", 4)
        head_dim = getattr(config, "engram_embed_dim_per_head", 32)
        kernel = getattr(config, "engram_kernel_size", 4)
        seed = getattr(config, "engram_seed", 42)

        n_gram_types = max_ngram - 1                    # e.g. 2 for max_ngram=3
        total_heads = n_gram_types * n_heads            # total hash heads
        embed_per_gram = n_heads * head_dim             # per N-gram-order dim
        engram_dim = n_gram_types * embed_per_gram      # full concatenated dim

        # N-gram hasher (CPU-side, constructed once)
        self.hasher = NgramHasher(
            max_ngram_size=max_ngram,
            n_heads_per_ngram=n_heads,
            base_vocab_size=config.vocab_size,
            seed=seed,
        )

        # One embedding slot per (n_gram, head) pair
        all_vocab_sizes = [
            self.hasher.moduli[n_idx][j]
            for n_idx in range(n_gram_types)
            for j in range(n_heads)
        ]
        self.multi_head_emb = MultiHeadEmbedding(all_vocab_sizes, head_dim)

        # Local context fusion
        self.short_conv = ShortConv(
            channels=engram_dim,
            kernel_size=kernel,
            dilation=max_ngram,
        )

        # Projections
        proj_cls = BitLinear if getattr(config, "use_bitnet", False) else nn.Linear
        self.value_proj = proj_cls(engram_dim, self.hidden_size, bias=False)
        self.key_proj = proj_cls(engram_dim, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)

        self.norm_key = nn.LayerNorm(self.hidden_size)
        self.norm_query = nn.LayerNorm(self.hidden_size)

        self._scale = math.sqrt(self.hidden_size)
        self._total_heads = total_heads

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        logical_layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, T, hidden_size) – incoming hidden states
            input_ids:      (B, T) int64 – token indices (required for lookup)
            logical_layer_idx: ignored, kept for HybridLayer interface parity

        Returns:
            (B, T, hidden_size) – Engram-corrected hidden states
        """
        if input_ids is None:
            return torch.zeros_like(x)

        B, T, D = x.shape
        device = x.device

        # 1. Hash: CPU numpy → GPU tensor.
        #    The hashing is deterministic and fast on CPU; the result is a
        #    small int64 tensor of shape (B, T, total_heads) so the transfer
        #    overhead is proportional to batch × seq_len × total_heads × 8 bytes
        #    (e.g. ~6 KB for batch=2, seq=128, 6 heads), which is negligible.
        ids_np = input_ids.detach().cpu().numpy().astype(np.int64)
        hashes_np = self.hasher.hash(ids_np)              # (B, T, total_heads)
        hashes = torch.from_numpy(hashes_np).to(device)   # (B, T, total_heads)

        # 2. Embedding lookup → (B, T, total_heads, head_dim)
        emb4d = self.multi_head_emb(hashes)

        # 3. Flatten heads → (B, T, engram_dim)
        engram_emb = emb4d.flatten(start_dim=2)

        # 4. ShortConv for local context fusion
        engram_emb = engram_emb + self.short_conv(engram_emb)

        # 5. Gate: σ(dot(norm(key), norm(query)) / √D)
        key = self.norm_key(self.key_proj(engram_emb))    # (B, T, D)
        query = self.norm_query(x)                         # (B, T, D)
        gate = (key * query).sum(dim=-1, keepdim=True) / self._scale  # (B, T, 1)
        # Smooth sqrt-gate from the paper (Section 3.3 of arXiv:2601.07372).
        # Plain sigmoid collapses to near-0 or near-1 for large dot products,
        # leading to vanishing gradients. The sqrt(|s|) * sign(s) remapping
        # compresses the dynamic range before sigmoid so the gate stays in a
        # more gradient-friendly region while preserving the sign of the score.
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = torch.sigmoid(gate)

        # 6. Value and output projection
        value = gate * self.value_proj(engram_emb)         # (B, T, D)
        return self.out_proj(value)
