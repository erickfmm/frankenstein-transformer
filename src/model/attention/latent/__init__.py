"""Latent and KV-compression attention family.

This subpackage groups nine dense-attention variants that reduce the
Key-Value cache footprint or mix information across attention heads,
generalising Grouped-Query Attention (GQA) along complementary axes:

* :class:`MLAAttention` -- Multi-Head Latent Attention (MLA + RoPE),
  arXiv:2506.09342. Jointly compresses keys and values into a low-rank
  latent vector; the canonical MLA used by DeepSeek-V2/V3.
* :class:`GQLAAttention` -- Group-Query Latent Attention, arXiv:2605.15250.
  Exposes two algebraically equivalent decoding paths (MQA-absorb and
  per-group GQA) over the same trained weights so a runtime can pin the
  roofline of both H100-class and commodity inference GPUs.
* :class:`MLRAAttention` -- Multi-Head Low-Rank Attention, arXiv:2603.02188.
  Partitionable latent states enabling efficient 4-way tensor-parallel
  decoding (each device loads a disjoint latent head slice).
* :class:`TuckerAttention` -- Tucker Attention, arXiv:2603.30033.
  Generalises GQA/MLA/MHA through a Tucker-style factorisation of the
  query/key/value weight tensors, exposing the actual ranks achieved by
  each method; an order of magnitude fewer parameters for comparable
  validation metrics.
* :class:`IHAAttention` -- Interleaved Head Attention, arXiv:2602.21371.
  Constructs ``P`` pseudo-heads per head (typically ``P = H``) where each
  pseudo query/key/value is a learned linear combination of all ``H``
  original heads, enabling cross-head mixing and up to ``P**2`` attention
  patterns per head with modest ``O(H**2 * P)`` parameter overhead.
* :class:`GTAAttention` -- Grouped-head laTenT Attention, arXiv:2506.17286.
  Combines a shared attention-map mechanism (reuses attention scores
  across multiple heads to shrink the key cache) with a nonlinear value
  decoder that compresses the value cache into a latent space.
* :class:`MTLAAttention` -- Multi-head Temporal Latent Attention,
  arXiv:2505.13544. Extends MLA along the temporal axis with a
  hyper-network that dynamically merges temporally adjacent KV cache
  vectors and a stride-aware causal mask that keeps parallel training
  consistent with inference.
* :class:`CCAAttention` -- Compressed Convolutional Attention,
  arXiv:2510.04476. Down-projects q/k/v into a shared latent and
  performs the entire attention operation inside that latent (no
  up-projections), with causal convolutions, q-k-mean, and value-shift
  to recover quality. Reduces parameters, KV-cache, **and** FLOPs by
  the compression factor.
* :class:`CCGQAAttention` -- Compressed Convolutional Grouped Query
  Attention, arXiv:2510.04476. Extends CCA with GQA-style head sharing
  inside the latent and decoupled query/KV compression rates.

All modules follow the project attention-mixer interface:
``forward(x: torch.Tensor, logical_layer_idx: Optional[int] = None) ->
torch.Tensor`` and accept an :class:`UltraConfig`-like ``config`` object
with ``hidden_size``, ``num_heads``, ``dropout``, ``use_bitnet`` and
``mode`` attributes. BitNet ternary weights are honoured through
:class:`~src.model.attention.common.BitLinear`.
"""

from .mla_attn import MLAAttention
from .gqla_attn import GQLAAttention
from .mlra_attn import MLRAAttention
from .tucker_attn import TuckerAttention
from .iha_attn import IHAAttention
from .gta_attn import GTAAttention
from .mtla_attn import MTLAAttention
from .cca_attn import CCAAttention, CCGQAAttention

__all__ = [
    "MLAAttention",
    "GQLAAttention",
    "MLRAAttention",
    "TuckerAttention",
    "IHAAttention",
    "GTAAttention",
    "MTLAAttention",
    "CCAAttention",
    "CCGQAAttention",
]