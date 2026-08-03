"""Nested model-config flattener.

Converts the hierarchical ``model:`` YAML block (introduced in the schema
refactor) into the flat keyword-argument shape expected by the
:class:`UltraConfig` dataclass, which remains the internal flat
representation used throughout the model, attention, deploy and export
code paths.

The hierarchical schema groups keys as:

    model:
      dims:        {vocab_size, hidden_size, num_layers, num_loops, num_heads,
                    num_kv_heads, retention_heads, dropout, layer_pattern, mode}
      norm:        {type, partial_ratio}
      embedding:
        factorized: {enabled, dim}
        conv:       {enabled, kernel}
      attention:
        titan:      {positional_encoding, use_hope, hope: {base, damping},
                     rope: {base, scaling}}
        mla:        {latent_rank}
        gqla:       {latent_rank, num_groups, decode_path}
        mlra:       {latent_rank, num_latent_heads}
        tucker:     {query_rank, key_rank, value_rank}
        iha:        {num_pseudo_heads}
        gta:        {num_shared_groups, value_latent_rank}
        mtla:       {latent_rank, merge_factor, stride}
        cca:        {latent_rank, num_conv_layers, conv_kernel_seq,
                    conv_kernel_ch, qk_mean, value_shift}
        ccgqa:      {query_latent_rank, kv_latent_rank, num_kv_heads,
                    num_conv_layers, conv_kernel_seq, conv_kernel_ch,
                    qk_mean, value_shift}
        msa:        {block_size, topk_blocks, index_dim, kl_loss_weight}
        sparda:     {block_size, topk_blocks, forecast_dim}
        engram:     {max_ngram_size, n_heads_per_ngram, embed_dim_per_head,
                    kernel_size, seed}
      # flat keys that were never moved (use_moe, use_bitnet, ffn_*, ode_*, ...)

The flattener is tolerant: if the input ``model_data`` is already flat
(legacy shape, e.g. an old checkpoint ``config.json``), it is returned
unchanged. This makes on-disk JSON checkpoints forward-compatible with
the new YAML schema without an explicit migration step.
"""

from __future__ import annotations

from typing import Any, Dict

# Leaf-name remap table for the per-mixer attention sub-objects.
# Maps (mixer_key, leaf_name) -> flat_key.
# Only mixers whose flat-key prefix differs from a trivial copy need an entry.
_ATTENTION_MIXER_RENAMES: Dict[str, Dict[str, str]] = {
    "mla": {"latent_rank": "mla_latent_rank"},
    "gqla": {
        "latent_rank": "gqla_latent_rank",
        "num_groups": "gqla_num_groups",
        "decode_path": "gqla_decode_path",
    },
    "mlra": {
        "latent_rank": "mlra_latent_rank",
        "num_latent_heads": "mlra_num_latent_heads",
    },
    "tucker": {
        "query_rank": "tucker_query_rank",
        "key_rank": "tucker_key_rank",
        "value_rank": "tucker_value_rank",
    },
    "iha": {"num_pseudo_heads": "iha_num_pseudo_heads"},
    "gta": {
        "num_shared_groups": "gta_num_shared_groups",
        "value_latent_rank": "gta_value_latent_rank",
    },
    "mtla": {
        "latent_rank": "mtla_latent_rank",
        "merge_factor": "mtla_merge_factor",
        "stride": "mtla_stride",
    },
    "cca": {
        "latent_rank": "cca_latent_rank",
        "num_conv_layers": "cca_num_conv_layers",
        "conv_kernel_seq": "cca_conv_kernel_seq",
        "conv_kernel_ch": "cca_conv_kernel_ch",
        "qk_mean": "cca_qk_mean",
        "value_shift": "cca_value_shift",
    },
    "ccgqa": {
        "query_latent_rank": "ccgqa_query_latent_rank",
        "kv_latent_rank": "ccgqa_kv_latent_rank",
        "num_kv_heads": "ccgqa_num_kv_heads",
        "num_conv_layers": "ccgqa_num_conv_layers",
        "conv_kernel_seq": "ccgqa_conv_kernel_seq",
        "conv_kernel_ch": "ccgqa_conv_kernel_ch",
        "qk_mean": "ccgqa_qk_mean",
        "value_shift": "ccgqa_value_shift",
    },
    "msa": {
        "block_size": "msa_block_size",
        "topk_blocks": "msa_topk_blocks",
        "index_dim": "msa_index_dim",
        "kl_loss_weight": "msa_kl_loss_weight",
    },
    "sparda": {
        "block_size": "sparda_block_size",
        "topk_blocks": "sparda_topk_blocks",
        "forecast_dim": "sparda_forecast_dim",
    },
    "engram": {
        "max_ngram_size": "engram_max_ngram_size",
        "n_heads_per_ngram": "engram_n_heads_per_ngram",
        "embed_dim_per_head": "engram_embed_dim_per_head",
        "kernel_size": "engram_kernel_size",
        "seed": "engram_seed",
    },
}

# Sub-mixers that should be skipped entirely (they are grouping containers
# like titan's hope/rope, not direct flat-key producers).
_MIXER_GROUP_KEYS = {"titan"}


def _is_nested_shape(model_data: Dict[str, Any]) -> bool:
    """Heuristic: detect whether ``model_data`` uses the new nested schema.

    Returns True if any of the top-level grouping keys is present.
    Legacy flat dicts (e.g. old checkpoint config.json) do not contain
    ``dims``, ``norm``, ``embedding`` (as a dict), or ``attention`` (as a
    dict) and are passed through unchanged.
    """
    for key in ("dims", "norm", "embedding", "attention"):
        if key in model_data and isinstance(model_data[key], dict):
            return True
    return False


def _flatten_attention(attn: Dict[str, Any], out: Dict[str, Any]) -> None:
    """Flatten the ``model.attention`` sub-tree into flat keys in ``out``."""
    for mixer, spec in attn.items():
        if mixer == "titan":
            _flatten_titan(spec, out)
        else:
            renames = _ATTENTION_MIXER_RENAMES.get(mixer)
            if renames is None:
                # Unknown mixer sub-key: skip (defensive — the schema
                # `additionalProperties: false` already rejects unknowns).
                continue
            for leaf, flat_key in renames.items():
                if leaf in spec:
                    out[flat_key] = spec[leaf]


def _flatten_titan(titan: Dict[str, Any], out: Dict[str, Any]) -> None:
    """Flatten ``model.attention.titan`` (positional encoding) sub-tree."""
    if "positional_encoding" in titan:
        out["positional_encoding"] = titan["positional_encoding"]
    if "use_hope" in titan:
        out["use_hope"] = titan["use_hope"]
    hope = titan.get("hope")
    if isinstance(hope, dict):
        if "base" in hope:
            out["hope_base"] = hope["base"]
        if "damping" in hope:
            out["hope_damping"] = hope["damping"]
    rope = titan.get("rope")
    if isinstance(rope, dict):
        if "base" in rope:
            out["rope_base"] = rope["base"]
        if "scaling" in rope:
            out["rope_scaling"] = rope["scaling"]


def flatten_model_dict(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a (possibly nested) ``model:`` block to flat UltraConfig kwargs.

    Args:
        model_data: The ``model:`` mapping from a YAML config or a
            ``config.json`` dict. May be either the new hierarchical shape
            (with ``dims``, ``norm``, ``embedding``, ``attention`` sub-keys)
            or a legacy flat shape.

    Returns:
        A flat dictionary suitable for ``UltraConfig(**result)``. If the
        input is already flat, it is returned as-is (shallow copy).
    """
    if not isinstance(model_data, dict):
        return {}

    if not _is_nested_shape(model_data):
        # Legacy flat shape (e.g. old checkpoint) — pass through.
        return dict(model_data)

    out: Dict[str, Any] = {}

    # Pass through the staying-flat keys (use_moe, use_bitnet, ffn_*, ...).
    # We do this by copying everything that is NOT a known grouping key.
    grouping_keys = {"dims", "norm", "embedding", "attention"}
    for key, value in model_data.items():
        if key not in grouping_keys:
            out[key] = value

    # dims.* — leaf names unchanged.
    dims = model_data.get("dims")
    if isinstance(dims, dict):
        for leaf in (
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_loops",
            "num_heads",
            "num_kv_heads",
            "retention_heads",
            "dropout",
            "layer_pattern",
            "mode",
        ):
            if leaf in dims:
                out[leaf] = dims[leaf]

    # norm.* — leaf renames.
    norm = model_data.get("norm")
    if isinstance(norm, dict):
        if "type" in norm:
            out["norm_type"] = norm["type"]
        if "partial_ratio" in norm:
            out["prms_partial_ratio"] = norm["partial_ratio"]
        if "flashnorm_partial_ratio" in norm:
            out["flashnorm_partial_ratio"] = norm["flashnorm_partial_ratio"]

    # embedding.factorized.* + embedding.conv.* — leaf renames.
    embedding = model_data.get("embedding")
    if isinstance(embedding, dict):
        fact = embedding.get("factorized")
        if isinstance(fact, dict):
            if "enabled" in fact:
                out["use_factorized_embedding"] = fact["enabled"]
            if "dim" in fact:
                out["factorized_embedding_dim"] = fact["dim"]
        conv = embedding.get("conv")
        if isinstance(conv, dict):
            if "enabled" in conv:
                out["use_embedding_conv"] = conv["enabled"]
            if "kernel" in conv:
                out["embedding_conv_kernel"] = conv["kernel"]

    # attention.<mixer>.*
    attn = model_data.get("attention")
    if isinstance(attn, dict):
        _flatten_attention(attn, out)

    return out