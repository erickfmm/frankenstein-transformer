"""Nested model-config flattener.

Converts the hierarchical ``model:`` YAML block (introduced in the schema
refactor) into the flat keyword-argument shape expected by the
:class:`FrankensteinModelConfig` dataclass, which remains the internal flat
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
        gma:        {num_components, routing_dim, epsilon, sigma_eps,
                    init_mean_std}
        ssog:       {num_atoms, lookat, max_offset, cold_init, sigma_floor,
                    grid_h, grid_w}
      positional_encoding: string enum (rope, hope, nope, alibi, pape, ...)
      positional_encoding_parameters: {rope: {base, scaling}, hope: {base, damping},
                         pape: {num_parabolas, num_positions, rotation_invariant},
                         sinusoidal: {max_len, base, scale}, learned: {max_len, init_std},
                         use_pe: {<mixer>: bool, ...}}
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
    "gma": {
        "num_components": "gma_num_components",
        "routing_dim": "gma_routing_dim",
        "epsilon": "gma_epsilon",
        "sigma_eps": "gma_sigma_eps",
        "init_mean_std": "gma_init_mean_std",
    },
    "ssog": {
        "num_atoms": "ssog_num_atoms",
        "lookat": "ssog_lookat",
        "max_offset": "ssog_max_offset",
        "cold_init": "ssog_cold_init",
        "sigma_floor": "ssog_sigma_floor",
        "grid_h": "ssog_grid_h",
        "grid_w": "ssog_grid_w",
    },
    "falcon": {
        "chunk_size": "falcon_chunk_size",
        "qk_norm": "falcon_qk_norm",
        "beta_mode": "falcon_beta_mode",
        "lambda_mode": "falcon_lambda_mode",
        "beta": "falcon_beta",
        "lambda": "falcon_lambda",
        "window": "falcon_window",
        "short_conv": "falcon_short_conv",
        "conv_kernel": "falcon_conv_kernel",
        "eps": "falcon_eps",
        "eps_gamma": "falcon_eps_gamma",
    },
}

# Sub-mixers that should be skipped entirely (they are grouping containers
# like titan's hope/rope, not direct flat-key producers).
_MIXER_GROUP_KEYS = {"titan"}


def _is_nested_shape(model_data: Dict[str, Any]) -> bool:
    """Heuristic: detect whether ``model_data`` uses the new nested schema.

    Returns True if any of the top-level grouping keys is present.
    Legacy flat dicts (e.g. old checkpoint config.json) do not contain
    ``dims``, ``norm``, ``embedding`` (as a dict), ``attention`` (as a
    dict), ``mhc`` (as a dict) or ``residuals`` (as a dict) and are
    passed through unchanged.
    """
    for key in ("dims", "norm", "embedding", "attention", "mhc", "residuals", "positional_encoding_parameters"):
        if key in model_data and isinstance(model_data[key], dict):
            return True
    return False


def flatten_image_dict(image_data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the top-level ``image:`` block into flat FrankensteinModelConfig kwargs.

    This is separate from :func:`flatten_model_dict` because the ``image:``
    block is a top-level YAML key (sibling of ``model:``), not a sub-key of
    ``model:``. The resulting flat keys (``image_height``, ``patch_size``,
    etc.) are merged into the model config kwargs before constructing
    :class:`FrankensteinModelConfig`.

    Args:
        image_data: The ``image:`` mapping from a YAML config.

    Returns:
        A flat dictionary of vision config kwargs. Empty if ``image_data``
        is not a dict.
    """
    if not isinstance(image_data, dict):
        return {}

    out: Dict[str, Any] = {}

    # image_size.{height, width} -> image_height, image_width
    image_size = image_data.get("image_size")
    if isinstance(image_size, dict):
        if "height" in image_size:
            out["image_height"] = image_size["height"]
        if "width" in image_size:
            out["image_width"] = image_size["width"]

    # Direct leaf mappings (flat keys with same name).
    for leaf in (
        "patch_size",
        "in_channels",
        "to_grayscale",
        "pos_embedding_type",
        "cls_token",
        "pooling_mode",
        "mask_ratio",
        "mask_token_strategy",
        "prediction_target",
        "seg_head_type",
        "num_classes",
        "num_seg_classes",
        "seg_num_queries",
        "seg_l2_blocks",
        "seg_mask_annealing",
    ):
        if leaf in image_data:
            out[leaf] = image_data[leaf]

    return out


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


def _flatten_pe_parameters(pe_params: Dict[str, Any], out: Dict[str, Any]) -> None:
    """Flatten the ``model.positional_encoding_parameters`` sub-tree into flat keys."""
    rope = pe_params.get("rope")
    if isinstance(rope, dict):
        if "base" in rope:
            out["rope_base"] = rope["base"]
        if "scaling" in rope:
            out["rope_scaling"] = rope["scaling"]

    hope = pe_params.get("hope")
    if isinstance(hope, dict):
        if "base" in hope:
            out["hope_base"] = hope["base"]
        if "damping" in hope:
            out["hope_damping"] = hope["damping"]

    pape = pe_params.get("pape")
    if isinstance(pape, dict):
        if "num_parabolas" in pape:
            out["pape_num_parabolas"] = pape["num_parabolas"]
        if "num_positions" in pape:
            out["pape_num_positions"] = pape["num_positions"]
        if "rotation_invariant" in pape:
            out["pape_rotation_invariant"] = pape["rotation_invariant"]

    sinusoidal = pe_params.get("sinusoidal")
    if isinstance(sinusoidal, dict):
        if "max_len" in sinusoidal:
            out["sinusoidal_max_len"] = sinusoidal["max_len"]
        if "base" in sinusoidal:
            out["sinusoidal_base"] = sinusoidal["base"]
        if "scale" in sinusoidal:
            out["sinusoidal_scale"] = sinusoidal["scale"]

    learned = pe_params.get("learned")
    if isinstance(learned, dict):
        if "max_len" in learned:
            out["learned_max_len"] = learned["max_len"]
        if "init_std" in learned:
            out["learned_init_std"] = learned["init_std"]

    bam = pe_params.get("bam")
    if isinstance(bam, dict):
        if "learn_mu" in bam:
            out["bam_learn_mu"] = bam["learn_mu"]
        if "theta_init" in bam:
            out["bam_theta_init"] = bam["theta_init"]
        if "eps" in bam:
            out["bam_eps"] = bam["eps"]
        if "mu_init" in bam:
            out["bam_mu_init"] = bam["mu_init"]

    use_pe = pe_params.get("use_pe")
    if isinstance(use_pe, dict):
        for mixer_name, flag in use_pe.items():
            flat_key = f"{mixer_name}_use_pe"
            out[flat_key] = flag


def flatten_model_dict(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a (possibly nested) ``model:`` block to flat FrankensteinModelConfig kwargs.

    Args:
        model_data: The ``model:`` mapping from a YAML config or a
            ``config.json`` dict. May be either the new hierarchical shape
            (with ``dims``, ``norm``, ``embedding``, ``attention`` sub-keys)
            or a legacy flat shape.

    Returns:
        A flat dictionary suitable for ``FrankensteinModelConfig(**result)``. If the
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
    grouping_keys = {"dims", "norm", "embedding", "attention", "mhc", "residuals", "positional_encoding_parameters"}
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

    if "positional_encoding" in model_data:
        out["positional_encoding"] = model_data["positional_encoding"]

    pe_params = model_data.get("positional_encoding_parameters")
    if isinstance(pe_params, dict):
        _flatten_pe_parameters(pe_params, out)

    # mhc.* — Manifold-Constrained Hyper-Connections (arXiv:2512.24880).
    mhc = model_data.get("mhc")
    if isinstance(mhc, dict):
        if "enabled" in mhc:
            out["use_mhc"] = mhc["enabled"]
        if "expansion_rate" in mhc:
            out["mhc_expansion_rate"] = mhc["expansion_rate"]
        if "sinkhorn_iters" in mhc:
            out["mhc_sinkhorn_iters"] = mhc["sinkhorn_iters"]
        if "gating_init" in mhc:
            out["mhc_gating_init"] = mhc["gating_init"]
        if "checkpoint" in mhc:
            out["mhc_checkpoint"] = mhc["checkpoint"]
        if "full_prec_under_bitnet" in mhc:
            out["mhc_full_prec_under_bitnet"] = mhc["full_prec_under_bitnet"]

    # residuals.* — Attention Residuals (AttnRes, arXiv:2603.15031).
    residuals = model_data.get("residuals")
    if isinstance(residuals, dict):
        if "type" in residuals:
            out["residual_type"] = residuals["type"]
        full_attn = residuals.get("full_attn")
        if isinstance(full_attn, dict):
            if "init_query_zero" in full_attn:
                out["full_attn_init_query_zero"] = full_attn["init_query_zero"]
            if "use_rmsnorm_keys" in full_attn:
                out["full_attn_use_rmsnorm_keys"] = full_attn["use_rmsnorm_keys"]
        block_attn = residuals.get("block_attn")
        if isinstance(block_attn, dict):
            if "num_blocks" in block_attn:
                out["block_attn_num_blocks"] = block_attn["num_blocks"]
            if "init_query_zero" in block_attn:
                out["block_attn_init_query_zero"] = block_attn["init_query_zero"]
            if "use_rmsnorm_keys" in block_attn:
                out["block_attn_use_rmsnorm_keys"] = block_attn["use_rmsnorm_keys"]
        if "mhc_stream_mode" in residuals:
            out["attnres_mhc_stream_mode"] = residuals["mhc_stream_mode"]
        if "gradient_checkpoint" in residuals:
            out["attnres_gradient_checkpoint"] = residuals["gradient_checkpoint"]

    return out