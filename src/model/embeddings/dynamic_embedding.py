"""EELBERT-style dynamic (hash) token embeddings — arXiv:2310.20144.

Replaces the trainable ``vocab_size × hidden_size`` embedding lookup table
with an on-the-fly *n-gram pooling hash* of the token **string** (EELBERT,
Cohn et al., EMNLP 2023 Industry Track). Dynamic embeddings are deterministic
functions of one stored seed plus the tokenizer vocabulary: they are never
trained and the table never appears in checkpoints or exported models, which
is where the size reduction in the paper comes from.

Algorithm (paper §3.2), for a token string ``w`` of length ``l`` producing a
``d = hidden_size`` vector:

1. Draw ``d`` hash seeds ``h ∈ [0, B)^d`` from a fixed random state (only the
   integer seed is persisted).
2. Compute ``i``-gram signatures ``s_i ∈ [0, B)^{k_i}`` with a polynomial
   rolling hash, where ``k_i = l - i + 1`` and ``i = 1..N`` (``N = 3``).
3. Partition ``h`` into ``N`` vectors ``h_i`` (``Σ d_i = d``, larger ``i``
   gets a larger ``d_i``) and form ``P_i = s_i × h_i^T``.
4. Fold ``P_i`` into ``[-1, 1]``: ``P %= B``; ``P -= (P > B/2)·B``;
   ``P /= B/2`` with ``B = 10^9 + 7``.
5. Average ``P_i`` across its ``k_i`` rows and concatenate the ``N`` parts.

With ``precompute=True`` (default) the whole ``vocab_size × hidden_size``
table is computed once and kept in a **non-persistent** buffer: it is not
saved, not exported, and simply moves with ``.to(device)``. That is pure
memoisation of a deterministic function, matching the paper's suggestion of
caching frequent n-grams (§5.5); set ``precompute=False`` for on-the-fly
per-batch computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention.common import BitConv1d

if TYPE_CHECKING:
    from ..config import FrankensteinModelConfig


# Polynomial rolling-hash base. 257 is prime and larger than any UTF-8 byte
# value, which keeps distinct byte strings well separated modulo the bucket.
_ROLLING_HASH_BASE = 257

# Bound on the ``(block, k_i, d_i)`` intermediate tensor used while hashing
# the vocabulary (elements; ~32 MB of int64 per gram size).
_CHUNK_ELEMENTS = 4_000_000


def _partition_sizes(hidden_size: int, num_parts: int) -> List[int]:
    """Split ``hidden_size`` into ``num_parts`` parts, larger index → larger part.

    Mirrors the paper's ``Σ d_i = d`` requirement with ``d_i`` increasing with
    the n-gram size ``i`` (weights ``1, 2, ..., N``).

    Args:
        hidden_size: Total embedding dimension ``d``.
        num_parts: Number of parts ``N`` (one per n-gram size).

    Returns:
        A list of ``num_parts`` positive integers summing to ``hidden_size``.

    Raises:
        ValueError: If ``num_parts < 1`` or ``hidden_size < num_parts``.
    """
    if num_parts < 1:
        raise ValueError(f"num_parts must be >= 1, got {num_parts}")
    if hidden_size < num_parts:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be >= the number of n-gram sizes "
            f"({num_parts}); each partition needs at least one dimension"
        )
    if num_parts == 1:
        return [hidden_size]

    total_weight = num_parts * (num_parts + 1) // 2
    sizes = [max(1, (hidden_size * weight) // total_weight) for weight in range(1, num_parts + 1)]

    # Reconcile rounding drift: return the surplus to the largest part, or
    # shave the deficit off parts that can still give up dimensions.
    diff = hidden_size - sum(sizes)
    while diff > 0:
        sizes[-1] += diff
        diff = 0
    index = len(sizes) - 1
    while diff < 0:
        if sizes[index] <= 1:
            index -= 1
            continue
        take = min(-diff, sizes[index] - 1)
        sizes[index] -= take
        diff += take
    return sizes


class DynamicEmbedding(nn.Module):
    """Dynamic token embedding computed from token strings (EELBERT).

    Holds no trainable parameters and no persisted lookup table. The token
    vocabulary (id-ordered token strings) is supplied through
    ``config.token_vocab``; the training engine, deployment and inference
    paths inject it automatically from the run/deploy tokenizer.

    Attributes:
        vocab: Id-ordered list of token strings.
        hidden_size: Embedding dimension ``d``.
        ngram_max: Maximum n-gram size ``N``.
        bucket: Hash bucket size ``B`` (paper default ``10^9 + 7``).
        seed: Random state used to derive the hash seeds.
        precompute: Whether the full table is memoised in a non-persistent
            buffer (``_table``).
        conv: Optional Conv1d (or :class:`BitConv1d`) applied to the
            embedding stream; mirrors :class:`FactorizedEmbedding`.
    """

    def __init__(self, config: "FrankensteinModelConfig"):
        """Initialise the dynamic embedding from a FrankensteinModelConfig.

        Args:
            config: Model configuration. Reads ``hidden_size``, ``vocab_size``,
                ``dynamic_embedding_ngram_max``, ``dynamic_embedding_bucket``,
                ``dynamic_embedding_seed``, ``dynamic_embedding_precompute``,
                ``use_embedding_conv``, ``embedding_conv_kernel``,
                ``use_bitnet``, ``use_bitnet_conv`` and ``token_vocab``.

        Raises:
            ValueError: If ``token_vocab`` is missing or its length does not
                match ``vocab_size``, or if ``ngram_max`` exceeds
                ``hidden_size``.
        """
        super().__init__()
        vocab = getattr(config, "token_vocab", None)
        if not vocab:
            raise ValueError(
                "embedding.dynamic.enabled requires the tokenizer vocabulary "
                "(FrankensteinModelConfig.token_vocab). The training engine, "
                "deploy and inference paths inject it automatically from the "
                "run/deploy tokenizer; when constructing the model directly, "
                "set config.token_vocab to an id-ordered list of token strings."
            )
        self.vocab: List[str] = [str(token) for token in vocab]
        if len(self.vocab) != int(config.dims.vocab_size):
            raise ValueError(
                f"embedding.dynamic: token_vocab has {len(self.vocab)} entries but "
                f"vocab_size is {config.dims.vocab_size}; the embedding ids must match "
                "the tokenizer vocabulary exactly"
            )

        self.hidden_size = int(config.dims.hidden_size)
        self.ngram_max = int(config.embedding.dynamic.ngram_max)
        self.bucket = int(config.embedding.dynamic.bucket)
        self.seed = int(config.embedding.dynamic.seed)
        self.precompute = bool(config.embedding.dynamic.precompute)

        if self.ngram_max < 1:
            raise ValueError(f"dynamic_embedding_ngram_max must be >= 1, got {self.ngram_max}")
        if self.bucket < 2:
            raise ValueError(f"dynamic_embedding_bucket must be >= 2, got {self.bucket}")
        if self.ngram_max > self.hidden_size:
            raise ValueError(
                f"dynamic_embedding_ngram_max ({self.ngram_max}) must be <= "
                f"hidden_size ({self.hidden_size})"
            )

        # One random state -> d hash seeds partitioned per n-gram size. The
        # seeds are a plain CPU tensor (tiny, recomputed from ``seed``): the
        # hash runs on CPU and only its output is moved to the module device.
        generator = torch.Generator().manual_seed(self.seed)
        self._seeds = torch.randint(
            0, self.bucket, (self.hidden_size,), generator=generator, dtype=torch.long
        )
        self._part_sizes: List[int] = _partition_sizes(self.hidden_size, self.ngram_max)

        # Optional Conv1d over the embedding stream (same composition as
        # FactorizedEmbedding: BitConv1d under BitNet when requested).
        self.use_conv = bool(config.embedding.conv.enabled)
        kernel = max(int(config.embedding.conv.kernel), 1)
        padding = kernel // 2
        use_bitconv = bool(getattr(config.bitnet, "conv", False))
        conv_cls = BitConv1d if (config.bitnet.enabled and use_bitconv) else nn.Conv1d
        self.conv = (
            conv_cls(self.hidden_size, self.hidden_size, kernel_size=kernel, padding=padding)
            if self.use_conv
            else None
        )

        # Memoised table (non-persistent: never saved nor exported).
        if self.precompute:
            self.register_buffer(
                "_table", self._embed_matrix(self.vocab).contiguous(), persistent=False
            )
        else:
            self._table: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def embedding_table(self) -> torch.Tensor:
        """Return the full ``(vocab_size, hidden_size)`` embedding table.

        Computed on demand when ``precompute`` is disabled. Used by the
        HuggingFace export path to materialise a frozen lookup table.

        Returns:
            A float tensor of shape ``(vocab_size, hidden_size)`` on the
            module's device.
        """
        if self._table is not None:
            return self._table
        return self._embed_matrix(self.vocab)

    # ------------------------------------------------------------------ #
    # Hash computation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _encode(strings: Sequence[str], max_len: int) -> torch.Tensor:
        """Encode UTF-8 token strings into a zero-padded byte matrix.

        Args:
            strings: Token strings.
            max_len: Pad/truncate width (the longest string's byte length).

        Returns:
            A ``(len(strings), max_len)`` int64 tensor of byte values.
        """
        matrix = torch.zeros((len(strings), max_len), dtype=torch.long)
        for row, string in enumerate(strings):
            payload = string.encode("utf-8")
            if payload:
                matrix[row, : len(payload)] = torch.tensor(
                    list(payload), dtype=torch.long
                )
        return matrix

    def _embed_chunk(self, strings: Sequence[str], max_len: int) -> torch.Tensor:
        """Hash a chunk of token strings into ``(len(strings), hidden_size)``.

        Args:
            strings: Token strings of the chunk.
            max_len: Byte width the chunk was padded to (matrix column count).

        Returns:
            A float32 embedding tensor of shape ``(len(strings), hidden_size)``.
        """
        bytes_matrix = self._encode(strings, max_len)
        lengths = torch.tensor(
            [len(string.encode("utf-8")) for string in strings], dtype=torch.long
        )
        out = torch.zeros((len(strings), self.hidden_size), dtype=torch.float32)

        hashes: Optional[torch.Tensor] = None
        offset = 0
        for ngram in range(1, self.ngram_max + 1):
            part = self._part_sizes[ngram - 1]
            windows = max_len - ngram + 1
            if windows <= 0:
                offset += part
                continue

            if ngram == 1:
                hashes = bytes_matrix[:, :windows].clone()
            else:
                # Incremental polynomial hash: append the next byte.
                hashes = (
                    hashes[:, :windows] * _ROLLING_HASH_BASE
                    + bytes_matrix[:, ngram - 1 : ngram - 1 + windows]
                ) % self.bucket

            # Projection matrix P_i = s_i x h_i^T, folded into [-1, 1].
            projection = hashes.unsqueeze(-1) * self._seeds[offset : offset + part].view(
                1, 1, -1
            )
            projection = projection % self.bucket
            projection = projection - (
                projection > (self.bucket // 2)
            ).to(projection.dtype) * self.bucket
            projection = projection.to(torch.float32) / (self.bucket / 2.0)

            # Only windows fully inside the token contribute; tokens shorter
            # than the current n-gram size contribute a zero vector.
            positions = torch.arange(windows)
            counts = (lengths - ngram + 1).clamp(min=0)
            valid = positions.unsqueeze(0) < counts.unsqueeze(1)
            projection = projection * valid.unsqueeze(-1).to(projection.dtype)
            out[:, offset : offset + part] = projection.sum(dim=1) / counts.clamp(
                min=1
            ).to(projection.dtype).unsqueeze(1)

            offset += part

        return out

    def _embed_matrix(self, strings: Sequence[str]) -> torch.Tensor:
        """Hash a list of token strings into ``(len(strings), hidden_size)``.

        Args:
            strings: Token strings.

        Returns:
            A float32 embedding tensor of shape ``(len(strings), hidden_size)``.
        """
        if not strings:
            return torch.zeros((0, self.hidden_size), dtype=torch.float32)

        max_len = max(1, max(len(string.encode("utf-8")) for string in strings))
        block = max(1, _CHUNK_ELEMENTS // (max_len * max(self._part_sizes)))
        chunks = [
            self._embed_chunk(strings[start : start + block], max_len)
            for start in range(0, len(strings), block)
        ]
        return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs through the dynamic hash computation.

        Args:
            input_ids: Integer token indices of shape ``(B, S)``.

        Returns:
            Tensor of shape ``(B, S, hidden_size)``.
        """
        if self._table is not None:
            x = F.embedding(input_ids, self._table)
        else:
            unique, inverse = torch.unique(input_ids, return_inverse=True)
            embeddings = self._embed_matrix(
                [self.vocab[int(token_id)] for token_id in unique.tolist()]
            ).to(input_ids.device)
            x = embeddings[inverse].view(*input_ids.shape, self.hidden_size)

        if self.conv is not None:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            if x.size(1) != input_ids.size(1):
                target_len = input_ids.size(1)
                if x.size(1) > target_len:
                    x = x[:, :target_len, :]
                else:
                    pad_len = target_len - x.size(1)
                    x = F.pad(x, (0, 0, 0, pad_len))
        return x
