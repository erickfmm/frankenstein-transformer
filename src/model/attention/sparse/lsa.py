"""Latent Sparse Attention (Kubicek et al., 2026; arXiv:2610.42147).

Implements **Native-LSA (N-LSA)**, the natively trainable variant of Latent
Sparse Attention: keys are L2-normalised, pooled every ``pool_size`` tokens
and Ward-clustered per complete ``cluster_block`` into buckets whose
unit-norm centroids are cached. Each query scores the bucket centroids
(``s_{i,b} = max_h q̂_{i,h} · c_b``, eq. 3), keeps the top-``p`` buckets,
expands them best-first into ``k`` long-range keys and attends to
``num_sinks`` sinks + selection + a one-block window. A single index list is
shared by every query head.

N-LSA adds **no parameters** (it reuses the attention's own projections),
needs **no positional encoding** (``lsa_use_pe`` defaults to ``False``)
and **stops the gradient through the selection**: the clustering, centroid
scoring and expansion run under ``torch.no_grad()``, so the next-token loss
trains the projections only through the keys each query actually attends to.

Cached decoding is exact: a block's buckets are immutable once the block
completes ("Once a cluster block is clustered, its buckets never change"),
so the incremental branch reuses the same centroids and index tables as the
full forward pass. Buckets live in a :class:`~src.model.cache.slots.BucketKVSlot`.

Reference:
    Kubicek, A., Caciolai, A., Peng, L., Sumant, S., Hoover, B., Greenlee,
    F., Battleday, R. M., Urban, R., Hoefler, T., & Whittington, J. C. R.
    (2026). "Latent Sparse Attention: Towards Subquadratic Ultra-Long
    Context Attention." arXiv:2610.42147.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from ...numerics import module_softmax
from ...norm import attach_qk_norm, record_qk_max_logits

from ..common import BitLinear, apply_pe_to_qk


class LSAAttention(nn.Module):
    """Latent Sparse Attention (N-LSA): semantic key clustering + top-k selection.

    Args:
        config: Configuration object. Relevant attributes: ``hidden_size``,
            ``num_heads``, ``num_kv_heads`` (paper default: 1), ``dropout``,
            ``use_bitnet``, ``mode``, and the optional ``lsa_*`` knobs
            (``num_long_range`` 2048, ``num_buckets`` 512, ``num_sinks`` 4,
            ``pool_size`` 4, ``cluster_block`` 512, ``keys_per_bucket`` 8,
            ``max_keys_per_bucket`` 16).
        pos_encoder: Optional shared positional encoding module.

    Attributes:
        hidden_size: Input dimensionality ``H``.
        num_heads: Number of query heads ``n_h``.
        num_kv_heads: Number of key/value heads ``n_kv`` (paper: 1).
        head_dim: Per-head dimensionality ``d_h = H / n_h``.
        num_long_range: Long-range key budget ``k``.
        num_buckets: Buckets scored per query ``p``.
        num_sinks: Always-attended leading sink keys ``n_sink``.
        pool_size: Keys summed per pool key ``n_pool``.
        cluster_block: Cluster block size ``n_cluster`` (also the window).
        keys_per_bucket: Average keys per bucket ``n̄_b``; produces
            ``m = cluster_block / keys_per_bucket`` buckets per block.
        max_keys_per_bucket: Merge cap ``n_max_b``.
        q_proj: Query projection ``(H, H)``.
        k_proj: Key projection ``(H, n_kv · d_h)``.
        v_proj: Value projection ``(H, n_kv · d_h)``.
        out_proj: Output projection ``(H, H)``.
        last_num_buckets: Number of cached bucket centroids used by the most
            recent forward pass (0 when no cluster block was complete).

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``, or
            ``num_heads`` is not divisible by ``num_kv_heads``.
    """

    def __init__(self, config, pos_encoder=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for LSAAttention")
        self.num_kv_heads = int(
            getattr(config, "num_kv_heads", max(1, self.num_heads // 8))
        )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_kv_heads must divide num_heads for LSAAttention")
        self.group_size = self.num_heads // self.num_kv_heads

        # Paper Table 2 / §4.4 (N-LSA bucket column).
        self.num_long_range = max(1, int(getattr(config, "lsa_num_long_range", 2048)))
        self.num_buckets = max(1, int(getattr(config, "lsa_num_buckets", 512)))
        self.num_sinks = max(0, int(getattr(config, "lsa_num_sinks", 4)))
        self.pool_size = max(1, int(getattr(config, "lsa_pool_size", 4)))
        self.cluster_block = max(2, int(getattr(config, "lsa_cluster_block", 512)))
        self.keys_per_bucket = max(1, int(getattr(config, "lsa_keys_per_bucket", 8)))
        self.max_keys_per_bucket = max(
            self.pool_size,
            int(getattr(config, "lsa_max_keys_per_bucket", 16)),
        )
        self._buckets_per_block = max(
            1, self.cluster_block // self.keys_per_bucket
        )

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, kv_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, kv_dim, bias=False)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.mode = str(getattr(config, "mode", "encoder"))
        self.scale = self.head_dim ** -0.5
        self.sink_eps = 1e-6
        self.last_num_buckets = 0

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        # N-LSA trains without positional encoding (§3.4); the per-mixer flag
        # stays overridable for hybrid experiments.
        self.use_pe = bool(getattr(config, "lsa_use_pe", False))
        attach_qk_norm(self, config, self.head_dim)

    # ------------------------------------------------------------------ #
    # Clustering (paper §3.1, Algorithm 1)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """L2-normalise the last dimension (``k_j ← k_j / ‖k_j‖₂``)."""
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _ward_single(
        sums: torch.Tensor,
        sizes: torch.Tensor,
        target_m: int,
        max_keys: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        """Ward agglomeration of one cluster block (Algorithm 1, one sample).

        Args:
            sums: Pool-key sums ``(N, D)`` (one bucket per pool key at start).
            sizes: Key counts ``(N,)`` (``n_pool`` per pool key).
            target_m: Stop once ``m`` buckets remain.
            max_keys: ``n_max_b``; merges whose combined key count exceeds it
                are skipped.

        Returns:
            ``(bucket_sums, bucket_sizes, members)`` where ``members[b]`` is
            the sorted list of pool-key indices inside bucket ``b``.
        """
        n = int(sums.size(0))
        device = sums.device
        sums = sums.clone()
        sizes = sizes.clone()
        alive = torch.ones(n, dtype=torch.bool, device=device)
        members: List[List[int]] = [[i] for i in range(n)]

        while int(alive.sum()) > target_m:
            index = alive.nonzero(as_tuple=False).flatten()
            local = int(index.numel())
            if local <= 1:
                break
            a = sums[index]
            # a_b = sqrt(G_bb) = ‖k_sum_b‖; a_{b∪b'} = ‖k_sum_b + k_sum_b'‖
            norms = a.norm(dim=-1)
            merged_norm = (a[:, None, :] + a[None, :, :]).norm(dim=-1)
            cost = norms[:, None] + norms[None, :] - merged_norm
            # Δ(b,b') = a_b + a_b' − a_{b∪b'}; skip merges above n_max_b.
            allowed = (sizes[index][:, None] + sizes[index][None, :]) <= max_keys
            allowed.fill_diagonal_(False)
            cost = cost.masked_fill(~allowed, float("inf"))
            partner = cost.argmin(dim=1)

            pairs: List[Tuple[int, int]] = []
            for li in range(local):
                lj = int(partner[li])
                if lj == li:
                    continue
                if int(partner[lj]) == li and li < lj:
                    if bool(torch.isfinite(cost[li, lj])):
                        pairs.append((int(index[li]), int(index[lj])))
            if not pairs:
                # No mutual pair (possible under the n_max_b cap): fall back
                # to the single cheapest allowed pair to guarantee progress.
                flat = int(cost.view(-1).argmin())
                li, lj = divmod(flat, local)
                if not bool(torch.isfinite(cost[li, lj])):
                    break  # every merge exceeds n_max_b
                pairs = [(int(index[li]), int(index[lj]))]
            pairs.sort()
            remaining = int(alive.sum()) - target_m
            for i, j in pairs[:remaining]:
                sums[i] = sums[i] + sums[j]
                sizes[i] = sizes[i] + sizes[j]
                members[i] = members[i] + members[j]
                alive[j] = False

        keep = alive.nonzero(as_tuple=False).flatten().tolist()
        return (
            sums[keep],
            sizes[keep],
            [sorted(members[i]) for i in keep],
        )

    def _cluster_block(
        self,
        keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool and Ward-cluster one complete cluster block.

        Args:
            keys: Index-head keys of the block ``(B, n_cluster, d_h)`` (must
                already be post-QK-norm, post-PE, **un-normalised**).
            positions: Absolute int64 positions of the block ``(n_cluster,)``.

        Returns:
            ``(centroids, table, sizes)`` with ``centroids`` ``(B, m, d_h)``
            (unit norm), ``table`` ``(B, m, n_max_b)`` absolute key positions
            padded with ``-1`` and ``sizes`` ``(B, m)`` key counts.
        """
        bsz, n_keys, head_dim = keys.shape
        device = keys.device
        n_pool = self.pool_size
        n_groups = (n_keys + n_pool - 1) // n_pool
        pad = n_groups * n_pool - n_keys
        if pad:
            keys = torch.cat([keys, keys.new_zeros(bsz, pad, head_dim)], dim=1)
        # 1) L2-normalise, 2) sum every n_pool consecutive keys.
        pool = self._l2norm(keys).view(bsz, n_groups, n_pool, head_dim).sum(dim=2)
        group_sizes = torch.full(
            (n_groups,), n_pool, dtype=torch.long, device=device
        )
        if n_keys % n_pool:
            group_sizes[-1] = n_keys - (n_groups - 1) * n_pool
        group_positions = [
            positions[g * n_pool : min((g + 1) * n_pool, n_keys)]
            for g in range(n_groups)
        ]
        target_m = min(self._buckets_per_block, n_groups)

        centroids: List[torch.Tensor] = []
        tables: List[torch.Tensor] = []
        sizes_out: List[torch.Tensor] = []
        for bi in range(bsz):
            bucket_sums, bucket_sizes, members = self._ward_single(
                pool[bi], group_sizes, target_m, self.max_keys_per_bucket
            )
            n_buckets = int(bucket_sums.size(0))
            centroids.append(self._l2norm(bucket_sums))
            table = torch.full(
                (n_buckets, self.max_keys_per_bucket),
                -1,
                dtype=torch.long,
                device=device,
            )
            for k, member in enumerate(members):
                if not member:
                    continue
                pos = torch.cat([group_positions[g] for g in member])
                pos = pos[: self.max_keys_per_bucket]
                table[k, : pos.numel()] = pos
            tables.append(table)
            sizes_out.append(bucket_sizes)
        return torch.stack(centroids), torch.stack(tables), torch.stack(sizes_out)

    def _build_bucket_cache(
        self,
        index_keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Build the bucket cache for every complete block of a sequence.

        Args:
            index_keys: Index-head keys ``(B, T, d_h)``.
            positions: Absolute positions ``(T,)``.

        Returns:
            ``(centroids, table, sizes, block_ids)`` or ``None`` when no
            cluster block is complete. ``block_ids`` is a per-bucket int64
            tensor ``(C,)``.
        """
        total = int(index_keys.size(1))
        n_blocks = total // self.cluster_block
        if n_blocks <= 0:
            return None
        centroids: List[torch.Tensor] = []
        tables: List[torch.Tensor] = []
        sizes: List[torch.Tensor] = []
        block_ids: List[int] = []
        for bi in range(n_blocks):
            start = bi * self.cluster_block
            stop = start + self.cluster_block
            cent, table, size = self._cluster_block(
                index_keys[:, start:stop, :], positions[start:stop]
            )
            centroids.append(cent)
            tables.append(table)
            sizes.append(size)
            block_ids.extend([bi] * int(cent.size(1)))
        return (
            torch.cat(centroids, dim=1),
            torch.cat(tables, dim=1),
            torch.cat(sizes, dim=1),
            torch.tensor(block_ids, dtype=torch.long, device=index_keys.device),
        )

    # ------------------------------------------------------------------ #
    # Semantic search + sparse attention (paper §3.2)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_selection(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        kv_positions: torch.Tensor,
        bucket_cache,
        query_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the shared index list ``J_i`` (paper §3.2, steps A–C).

        Runs under ``torch.no_grad()``: "We stop the gradient through the
        selection" (paper §3.4).

        Args:
            q: Queries ``(B, H, S, d_h)`` (post-QK-norm, post-PE).
            k: Cached keys ``(B, n_kv, T, d_h)`` (post-PE).
            kv_positions: Absolute positions of the cached keys ``(T,)``.
            bucket_cache: ``(centroids, table, sizes, block_ids)`` or ``None``.
            query_positions: Absolute positions of the queries ``(S,)``.

        Returns:
            ``(index, keep_mask)`` of shapes ``(B, S, W)``; ``index`` holds
            clamped key indices and ``keep_mask`` marks the attended slots.
        """
        bsz, _, seq_len, _ = q.shape
        device = q.device
        total = int(k.size(2))
        block = self.cluster_block
        n_sink = self.num_sinks
        k_budget = self.num_long_range

        # --- Candidate set J_i: sinks ∪ top-k long-range keys ∪ own-block window.
        sink_pos = torch.arange(n_sink, device=device)
        sink_valid = sink_pos.unsqueeze(0).expand(seq_len, n_sink) < total
        if self.mode == "decoder":
            sink_valid = sink_valid & (sink_pos.unsqueeze(0) <= query_positions.unsqueeze(1))

        own_start = (query_positions // block) * block
        own_end = torch.clamp(own_start + block, max=total)
        win_pos = own_start.unsqueeze(1) + torch.arange(block, device=device).unsqueeze(0)
        win_valid = (win_pos < own_end.unsqueeze(1)) & (win_pos < total)
        if self.mode == "decoder":
            win_valid = win_valid & (win_pos <= query_positions.unsqueeze(1))

        lr_width = k_budget + 1
        lr_pos = torch.zeros(bsz, seq_len, lr_width, dtype=torch.long, device=device)
        lr_valid = torch.zeros(bsz, seq_len, lr_width, dtype=torch.bool, device=device)
        centroids = bucket_cache[0] if bucket_cache is not None else None
        self.last_num_buckets = int(centroids.size(1)) if centroids is not None else 0
        if centroids is not None and centroids.size(1) > 0:
            table = bucket_cache[1]
            block_ids = torch.as_tensor(
                bucket_cache[3], dtype=torch.long, device=device
            )
            num_cached = int(centroids.size(1))
            # Eq. (3): s_{i,b} = max_h q̂_{i,h} · c_b (single shared index list).
            scores = torch.einsum(
                "bhsd,bcd->bhsc", self._l2norm(q), centroids
            )  # (B, H, S, C)
            bucket_scores = scores.amax(dim=1)  # (B, S, C)

            query_blocks = query_positions // block  # (S,)
            if self.mode == "decoder":
                candidate = block_ids.unsqueeze(0) < query_blocks.unsqueeze(1)
            else:
                candidate = block_ids.unsqueeze(0) != query_blocks.unsqueeze(1)
            bucket_scores = bucket_scores.masked_fill(
                ~candidate.unsqueeze(0), float("-inf")
            )

            # "p is chosen such that p · n_pool ≥ k"; never exceed the cache.
            min_buckets = math.ceil(k_budget / self.pool_size)
            p = min(num_cached, max(self.num_buckets, min_buckets))
            _, top_ids = bucket_scores.topk(p, dim=-1)  # (B, S, p), best first

            table_width = int(table.size(-1))
            gather_ids = (
                top_ids.reshape(bsz, seq_len * p, 1).expand(bsz, seq_len * p, table_width)
            )
            selected_pos = table.gather(1, gather_ids).reshape(
                bsz, seq_len, p, table_width
            )  # (B, S, p, n_max_b)
            flat_pos = selected_pos.reshape(bsz, seq_len, p * table_width)

            valid = flat_pos >= 0
            in_own = (flat_pos >= own_start.view(1, seq_len, 1)) & (
                flat_pos < own_end.view(1, seq_len, 1)
            )
            valid = valid & (~in_own) & (flat_pos >= n_sink) & (flat_pos < total)
            if self.mode == "decoder":
                valid = valid & (flat_pos <= query_positions.view(1, seq_len, 1))

            # Best bucket first, ascending inside each bucket: keep the first k.
            rank = torch.cumsum(valid.to(torch.long), dim=-1)
            keep = valid & (rank <= k_budget)
            target = (rank - 1).clamp_min(0).clamp_max(k_budget)
            lr_pos = lr_pos.scatter(2, target, flat_pos)
            lr_valid = lr_valid.to(torch.uint8).scatter(
                2, target, keep.to(torch.uint8)
            ).to(torch.bool)
            lr_pos = lr_pos[..., :k_budget].clamp_min(0)
            lr_valid = lr_valid[..., :k_budget]

        index = torch.cat(
            [
                sink_pos.unsqueeze(0).expand(bsz, seq_len, n_sink),
                win_pos.unsqueeze(0).expand(bsz, seq_len, block).clamp_max(
                    max(total - 1, 0)
                ),
                lr_pos,
            ],
            dim=-1,
        )
        keep_mask = torch.cat(
            [
                sink_valid.unsqueeze(0).expand(bsz, seq_len, n_sink),
                win_valid.unsqueeze(0).expand(bsz, seq_len, block),
                lr_valid,
            ],
            dim=-1,
        )
        return index, keep_mask

    def _select_and_attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_positions: torch.Tensor,
        bucket_cache,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Score buckets, expand the top-p into ``J_i`` and attend over it.

        Args:
            q: Queries ``(B, H, S, d_h)`` (post-QK-norm, post-PE).
            k: Cached keys ``(B, n_kv, T, d_h)`` (post-PE).
            v: Cached values ``(B, n_kv, T, d_h)``.
            kv_positions: Absolute positions of the cached keys ``(T,)``.
            bucket_cache: ``(centroids, table, sizes, block_ids)`` or ``None``.
            query_positions: Absolute positions of the queries ``(S,)``.

        Returns:
            Attended values ``(B, H, S, d_h)`` (pre out-projection).
        """
        bsz, _, seq_len, head_dim = q.shape
        total = int(k.size(2))
        index, keep_mask = self._build_selection(
            q, k, kv_positions, bucket_cache, query_positions
        )
        gather_index = index.unsqueeze(-1).expand(
            bsz, seq_len, index.size(-1), head_dim
        )

        outputs: List[torch.Tensor] = []
        score_chunks: List[torch.Tensor] = []
        for g in range(self.num_kv_heads):
            k_head = k[:, g].unsqueeze(1).expand(bsz, seq_len, total, head_dim)
            v_head = v[:, g].unsqueeze(1).expand(bsz, seq_len, total, head_dim)
            k_sel = torch.gather(k_head, 2, gather_index)  # (B, S, W, d_h)
            v_sel = torch.gather(v_head, 2, gather_index)
            q_group = q[:, g * self.group_size : (g + 1) * self.group_size]
            scores = torch.einsum("bgsd,bsjd->bgsj", q_group, k_sel) * self.scale
            scores = scores.masked_fill(~keep_mask.unsqueeze(1), float("-inf"))
            if getattr(self, "_record_qk_max_logits", False):
                score_chunks.append(scores)
            attn = module_softmax(self, scores, dim=-1)
            attn = self.dropout(attn)
            outputs.append(torch.einsum("bgsj,bsjd->bgsd", attn, v_sel))
        if score_chunks:
            record_qk_max_logits(self, torch.cat(score_chunks, dim=1))
        return torch.cat(outputs, dim=1)

    # ------------------------------------------------------------------ #
    # Cache plumbing
    # ------------------------------------------------------------------ #
    def make_cache_slot(self, meta, cache):
        """Build the bucket-carrying KV slot for incremental decoding."""
        from ...cache.slots import BucketKVSlot

        kwargs = cache.codec_slot_kwargs()
        return BucketKVSlot(
            meta,
            codec=kwargs["codec"],
            policy=None,
            num_sinks=kwargs["num_sinks"],
            max_keys_per_bucket=self.max_keys_per_bucket,
            cluster_block=self.cluster_block,
        )

    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute Latent Sparse Attention (N-LSA).

        The indexer (clustering + centroid scoring + expansion) runs under
        ``torch.no_grad()`` — the paper stops the gradient through the
        selection — while the attention over the selected keys is fully
        differentiable.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index forwarded to the positional
                encoder.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder``.
            cache: Optional ``BucketKVSlot`` for incremental decoding
                (``None`` runs the legacy full-recompute path, D5).
            cache_position: Absolute index of the first token in ``x``.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder
        bsz, seq_len, hidden = x.shape
        index = logical_layer_idx or 0

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None
        if slot is not None:
            from ...cache.hooks import kv_decode_append, positions_for

            offset = cache_position
            positions = positions_for(offset, seq_len, x.device)
            q, k = apply_pe_to_qk(
                pe,
                self.pe_type,
                q,
                k,
                x,
                index,
                self.use_pe,
                position_offset=offset,
            )
            K, V, kv_positions, _ = kv_decode_append(slot, k, v, positions)
            total = int(K.size(2))
            n_blocks = total // self.cluster_block
            if hasattr(slot, "append_block") and hasattr(slot, "read_bucket_cache"):
                # Cluster each newly completed block once; buckets are immutable.
                with torch.no_grad():
                    while slot.num_completed_blocks < n_blocks:
                        bi = slot.num_completed_blocks
                        start = bi * self.cluster_block
                        stop = start + self.cluster_block
                        cent, table, sizes = self._cluster_block(
                            K[:, 0, start:stop, :], kv_positions[start:stop]
                        )
                        slot.append_block(
                            block_id=bi,
                            centroids=cent,
                            positions=table,
                            sizes=sizes,
                        )
                bucket_cache = slot.read_bucket_cache()
            else:
                # D5 defence: a plain KVSlot (slot factory bypassed) still
                # yields exact output — re-cluster the cached keys on the fly.
                with torch.no_grad():
                    bucket_cache = self._build_bucket_cache(K[:, 0], kv_positions)
            out = self._select_and_attend(
                q, K, V, kv_positions, bucket_cache, positions
            )
        else:
            q, k = apply_pe_to_qk(pe, self.pe_type, q, k, x, index, self.use_pe)
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            with torch.no_grad():
                bucket_cache = self._build_bucket_cache(k[:, 0], positions)
            out = self._select_and_attend(
                q, k, v, positions, bucket_cache, positions
            )

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
