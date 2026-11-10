"""Cross-field validation for :class:`FrankensteinModelConfig`.

Ported from the historical flat dataclass so the runtime invariants (and the
error messages the test suite asserts on) survive the nested-config rewrite.
"""

from __future__ import annotations

from typing import Any


def _validate_ffn_activation_config(activation: str, cfg: dict) -> None:
    """Delegate to the config module's validator (single source of truth)."""
    from src.model.config import _validate_ffn_activation_config as impl

    impl(activation, cfg)


def validate_config(cfg: Any) -> None:
    """Validate and normalise the nested configuration in place.

    Args:
        cfg: A :class:`~src.model.config.FrankensteinModelConfig` instance.

    Raises:
        ValueError: If a cross-field or per-component invariant is violated.
    """
    dims = cfg.dims
    heads = max(1, int(dims.num_heads))

    def _put(mixer: str, key: str, value: Any) -> None:
        cfg.attention.setdefault(mixer, {})[key] = value

    def _resolved(mixer: str, key: str) -> Any:
        return (cfg.attention.get(mixer) or {}).get(key)

    def _round_up_to(value: int, unit: int) -> int:
        return max(unit, ((value + unit - 1) // unit) * unit)

    def _param(mixer: str, key: str) -> Any:
        value = _resolved(mixer, key)
        if value is not None:
            return value
        from src.registry import ComponentFamily, ensure_builtins, registry

        ensure_builtins()
        spec = registry.get(ComponentFamily.MIXER, mixer)
        return dict(spec.defaults).get(key)

    # ---- Derived per-mixer defaults ----
    if _resolved("cca", "latent_rank") is None:
        _put("cca", "latent_rank", _round_up_to(max(1, dims.hidden_size // 4), 2 * heads))
    if _resolved("ccgqa", "num_kv_heads") is None:
        kv_heads = max(1, heads // 4)
        while heads % kv_heads != 0:
            kv_heads -= 1
        _put("ccgqa", "num_kv_heads", kv_heads)
    kv_heads = max(1, int(_resolved("ccgqa", "num_kv_heads")))
    q_rank = _resolved("ccgqa", "query_latent_rank")
    kv_rank = _resolved("ccgqa", "kv_latent_rank")
    if q_rank is None and kv_rank is None:
        q_dh = _round_up_to(max(2, dims.hidden_size // (2 * heads)), 2)
        _put("ccgqa", "query_latent_rank", q_dh * heads)
        _put("ccgqa", "kv_latent_rank", q_dh * kv_heads)
    elif q_rank is None:
        kv_dh = max(1, int(kv_rank) // kv_heads)
        _put("ccgqa", "query_latent_rank", kv_dh * heads)
    elif kv_rank is None:
        q_dh = max(1, int(q_rank) // heads)
        _put("ccgqa", "kv_latent_rank", q_dh * kv_heads)
    if _resolved("gma", "routing_dim") is None:
        _put("gma", "routing_dim", max(1, dims.hidden_size // heads))
    image = cfg.image
    patch = max(1, int(image.patch_size if image is not None else 16))
    height = int(image.height if image is not None else 224)
    width = int(image.width if image is not None else 224)
    if _resolved("ssog", "grid_h") is None:
        _put("ssog", "grid_h", max(1, height // patch))
    if _resolved("ssog", "grid_w") is None:
        _put("ssog", "grid_w", max(1, width // patch))

    # ---- GMA ----
    if int(_param("gma", "routing_dim")) < 1:
        raise ValueError(f"gma_routing_dim must be >= 1, got {_param('gma', 'routing_dim')}")
    if int(_param("gma", "num_components")) < 1:
        raise ValueError(f"gma_num_components must be >= 1, got {_param('gma', 'num_components')}")
    if float(_param("gma", "epsilon")) <= 0:
        raise ValueError(f"gma_epsilon must be > 0, got {_param('gma', 'epsilon')}")
    if float(_param("gma", "sigma_eps")) <= 0:
        raise ValueError(f"gma_sigma_eps must be > 0, got {_param('gma', 'sigma_eps')}")

    # ---- SSOG ----
    if int(_param("ssog", "grid_h")) < 1 or int(_param("ssog", "grid_w")) < 1:
        raise ValueError(
            "ssog_grid_h/ssog_grid_w must be >= 1, got "
            f"{_param('ssog', 'grid_h')}x{_param('ssog', 'grid_w')}"
        )
    if int(_param("ssog", "num_atoms")) < 1:
        raise ValueError(f"ssog_num_atoms must be >= 1, got {_param('ssog', 'num_atoms')}")
    if float(_param("ssog", "sigma_floor")) < 0.0:
        raise ValueError(f"ssog_sigma_floor must be >= 0, got {_param('ssog', 'sigma_floor')}")
    if float(_param("ssog", "max_offset")) <= 0.0:
        raise ValueError(f"ssog_max_offset must be > 0, got {_param('ssog', 'max_offset')}")
    if "ssog" in (dims.layer_pattern or []) and dims.mode == "decoder":
        raise ValueError(
            "ssog is encoder-only: the separable Gaussian field has no causal "
            "formulation. Remove ssog from layer_pattern for decoder models."
        )

    # ---- SWA ----
    if int(_param("swa", "window_size")) < 2:
        raise ValueError(f"swa_window_size must be >= 2, got {_param('swa', 'window_size')}")
    if int(_param("swa", "num_sinks")) < 1:
        raise ValueError(f"swa_num_sinks must be >= 1, got {_param('swa', 'num_sinks')}")
    if int(_param("swa", "num_sinks")) >= int(_param("swa", "window_size")):
        raise ValueError(
            "swa_num_sinks must be smaller than swa_window_size, got "
            f"num_sinks={_param('swa', 'num_sinks')}, "
            f"window_size={_param('swa', 'window_size')}"
        )

    # ---- CA / FPA ----
    if float(_param("ca", "gain_init")) <= 0.0:
        raise ValueError(f"ca_gain_init must be > 0, got {_param('ca', 'gain_init')}")
    if float(_param("fpa", "beta_init")) <= 0.0:
        raise ValueError(f"fpa_beta_init must be > 0, got {_param('fpa', 'beta_init')}")
    if float(_param("fpa", "tau")) <= 0.0:
        raise ValueError(f"fpa_tau must be > 0, got {_param('fpa', 'tau')}")

    # ---- Fast Conformer ----
    if int(_param("fastconformer", "window_size")) < 2:
        raise ValueError(
            "fastconformer_window_size must be >= 2, got "
            f"{_param('fastconformer', 'window_size')}"
        )
    if int(_param("fastconformer", "global_tokens")) < 0:
        raise ValueError(
            "fastconformer_global_tokens must be >= 0, got "
            f"{_param('fastconformer', 'global_tokens')}"
        )
    if int(_param("fastconformer", "global_tokens_spacing")) < 1:
        raise ValueError(
            "fastconformer_global_tokens_spacing must be >= 1, got "
            f"{_param('fastconformer', 'global_tokens_spacing')}"
        )

    # ---- LSA ----
    for key, minimum in (
        ("num_long_range", 1),
        ("num_buckets", 1),
        ("num_sinks", 1),
        ("pool_size", 1),
        ("cluster_block", 2),
        ("keys_per_bucket", 1),
    ):
        if int(_param("lsa", key)) < minimum:
            raise ValueError(f"lsa_{key} must be >= {minimum}, got {_param('lsa', key)}")
    if int(_param("lsa", "num_sinks")) >= int(_param("lsa", "cluster_block")):
        raise ValueError(
            "lsa_num_sinks must be smaller than lsa_cluster_block, got "
            f"num_sinks={_param('lsa', 'num_sinks')}, "
            f"cluster_block={_param('lsa', 'cluster_block')}"
        )
    if int(_param("lsa", "cluster_block")) % int(_param("lsa", "keys_per_bucket")):
        raise ValueError(
            "lsa_cluster_block must be divisible by lsa_keys_per_bucket, got "
            f"cluster_block={_param('lsa', 'cluster_block')}, "
            f"keys_per_bucket={_param('lsa', 'keys_per_bucket')}"
        )
    if int(_param("lsa", "max_keys_per_bucket")) < int(_param("lsa", "keys_per_bucket")):
        raise ValueError(
            "lsa_max_keys_per_bucket must be >= lsa_keys_per_bucket, got "
            f"max_keys_per_bucket={_param('lsa', 'max_keys_per_bucket')}, "
            f"keys_per_bucket={_param('lsa', 'keys_per_bucket')}"
        )

    # ---- Conformer ----
    if int(_param("conformer", "kernel_size")) < 1:
        raise ValueError(
            f"conformer_kernel_size must be >= 1, got {_param('conformer', 'kernel_size')}"
        )
    if str(_param("conformer", "conv_norm")).lower() not in {"batch", "group", "layer"}:
        raise ValueError(
            "conformer_conv_norm must be one of ['batch', 'group', 'layer'], got "
            f"{_param('conformer', 'conv_norm')!r}"
        )

    # ---- Positional encoding ----
    if cfg.positional_encoding is None:
        cfg.positional_encoding = "rope"
    else:
        cfg.positional_encoding = str(cfg.positional_encoding).lower()
        from src.registry import ComponentFamily, ensure_builtins, registry

        ensure_builtins()
        valid = set(registry.names(ComponentFamily.POSITIONAL_ENCODING)) | set(
            registry.alias_index(ComponentFamily.POSITIONAL_ENCODING)
        )
        if cfg.positional_encoding not in valid:
            raise ValueError(
                f"positional_encoding must be one of {sorted(valid)}, got "
                f"{cfg.positional_encoding!r}"
            )
    if int(cfg.pape_num_parabolas) < 1:
        raise ValueError(f"pape_num_parabolas must be >= 1, got {cfg.pape_num_parabolas}")
    if int(cfg.pape_num_positions) < 1:
        raise ValueError(f"pape_num_positions must be >= 1, got {cfg.pape_num_positions}")
    if float(cfg.bam_eps) <= 0.0:
        raise ValueError(f"bam_eps must be > 0, got {cfg.bam_eps}")
    if cfg.ssmax.enabled and float(cfg.ssmax.s_init) <= 0.0:
        raise ValueError(
            f"ssmax_s_init must be > 0 when use_ssmax=True, got {cfg.ssmax.s_init}"
        )

    # ---- Norm / QK norm ----
    if not 0.0 < float(cfg.norm.partial_ratio) <= 1.0:
        raise ValueError("prms_partial_ratio must be in the range (0, 1]")
    if not 0.0 <= float(cfg.norm.flashnorm_partial_ratio) <= 1.0:
        raise ValueError("flashnorm_partial_ratio must be in the range [0, 1]")
    qk = cfg.norm.qk_norm
    qk.method = str(qk.method).lower()
    if qk.method not in {"rms_norm", "layer_norm", "l2_norm", "qk_clip", "quack"}:
        raise ValueError(
            "qk_norm_method must be one of {'rms_norm', 'layer_norm', 'l2_norm', "
            f"'qk_clip', 'quack'}}, got {qk.method!r}"
        )
    if float(qk.clip_tau) <= 0.0:
        raise ValueError("qk_clip_tau must be > 0")
    if not 0.0 <= float(qk.clip_alpha) <= 1.0:
        raise ValueError("qk_clip_alpha must be in the range [0, 1]")
    if float(qk.quack_tau) <= 0.0:
        raise ValueError("quack_tau must be > 0")
    if float(qk.eps) <= 0.0:
        raise ValueError("qk_norm_eps must be > 0")

    # ---- Embeddings ----
    emb = cfg.embedding
    if emb.dynamic.enabled and emb.factorized.enabled:
        raise ValueError(
            "use_dynamic_embedding and use_factorized_embedding are mutually "
            "exclusive: dynamic embeddings replace the factorized lookup table "
            "(embedding.dynamic.enabled vs embedding.factorized.enabled)"
        )
    if int(emb.dynamic.ngram_max) < 1:
        raise ValueError("dynamic_embedding_ngram_max must be >= 1")
    if int(emb.dynamic.bucket) < 2:
        raise ValueError("dynamic_embedding_bucket must be >= 2")
    if emb.dynamic.enabled and int(emb.dynamic.ngram_max) > int(dims.hidden_size):
        raise ValueError(
            f"dynamic_embedding_ngram_max ({emb.dynamic.ngram_max}) must be <= "
            f"hidden_size ({dims.hidden_size}): the n-gram partition needs at "
            "least one embedding dimension per gram size"
        )

    # ---- mHC / Hyperloop ----
    mhc = cfg.mhc
    if int(mhc.expansion_rate) < 1:
        raise ValueError("mhc_expansion_rate must be >= 1")
    if int(mhc.sinkhorn_iters) < 1:
        raise ValueError("mhc_sinkhorn_iters must be >= 1")
    if float(mhc.gating_init) <= 0.0:
        raise ValueError("mhc_gating_init must be > 0")
    if mhc.hyperloop:
        if not mhc.enabled:
            raise ValueError(
                "mhc_hyperloop=True requires use_mhc=True: the Hyperloop "
                "transformer replaces the per-sublayer mHC wiring with "
                "loop-level hyper-connections (arXiv:2604.21254)"
            )
        if int(dims.num_loops) < 2:
            raise ValueError(
                "mhc_hyperloop=True requires num_loops >= 2 (the paper uses 3), "
                f"got num_loops={int(dims.num_loops)}"
            )
        if int(mhc.hyperloop_begin_layers) < 0:
            raise ValueError("mhc_hyperloop_begin_layers must be >= 0")
        if int(mhc.hyperloop_end_layers) < 0:
            raise ValueError("mhc_hyperloop_end_layers must be >= 0")
        if int(mhc.hyperloop_begin_layers) + int(mhc.hyperloop_end_layers) >= int(
            dims.num_layers
        ):
            raise ValueError(
                "mhc_hyperloop_begin_layers + mhc_hyperloop_end_layers "
                f"({int(mhc.hyperloop_begin_layers)} + "
                f"{int(mhc.hyperloop_end_layers)}) must be strictly less than "
                f"num_layers ({int(dims.num_layers)}): the looped middle block "
                "must contain at least one layer"
            )
        rp = str(mhc.hyperloop_res_parameterization).lower()
        if rp not in {"diagonal", "sinkhorn", "identity"}:
            raise ValueError(
                "mhc_hyperloop_res_parameterization must be one of "
                f"['diagonal', 'identity', 'sinkhorn'], got {rp!r}"
            )
        mhc.hyperloop_res_parameterization = rp
        if str(cfg.residuals.type).lower() in {"full_attn", "block_attn"}:
            raise ValueError(
                "mhc_hyperloop is incompatible with AttnRes residuals "
                "(residual_type='full_attn'/'block_attn'): AttnRes applies "
                "depth-wise attention per logical layer on the n-stream "
                "residual, while Hyperloop applies hyper-connections only at "
                "the loop level"
            )

    # ---- Residuals ----
    rt = str(cfg.residuals.type).lower()
    if rt not in {"standard", "none", "full_attn", "block_attn"}:
        raise ValueError(
            "residual_type must be one of ['block_attn', 'full_attn', 'none', "
            f"'standard'], got {cfg.residuals.type!r}"
        )
    cfg.residuals.type = rt
    if int(cfg.residuals.block_attn.num_blocks) < 1:
        raise ValueError("block_attn_num_blocks must be >= 1")
    max_blocks = int(dims.num_layers) * int(dims.num_loops)
    if rt == "block_attn" and int(cfg.residuals.block_attn.num_blocks) > max_blocks:
        raise ValueError(
            f"block_attn_num_blocks ({cfg.residuals.block_attn.num_blocks}) cannot "
            f"exceed total logical depth ({max_blocks})"
        )
    if cfg.residuals.mhc_stream_mode not in {"independent", "joint"}:
        raise ValueError(
            "attnres_mhc_stream_mode must be 'independent' or 'joint', got "
            f"{cfg.residuals.mhc_stream_mode!r}"
        )

    # ---- Falcon (shared knobs) ----
    if (
        _param("falcon1", "chunk_size") is not None
        and int(_param("falcon1", "chunk_size")) < 1
    ):
        raise ValueError("falcon_chunk_size must be >= 1 or null")
    fq = str(_param("falcon1", "qk_norm")).lower()
    if fq not in {"rms_norm", "l2_norm"}:
        raise ValueError(f"falcon_qk_norm must be 'rms_norm' or 'l2_norm', got {fq!r}")
    fb = str(_param("falcon1", "beta_mode")).lower()
    if fb not in {"static", "ctx_beta", "ctx_eta"}:
        raise ValueError(
            f"falcon_beta_mode must be 'static', 'ctx_beta' or 'ctx_eta', got {fb!r}"
        )
    fl = str(_param("falcon1", "lambda_mode")).lower()
    if fl not in {"static", "ctx"}:
        raise ValueError(f"falcon_lambda_mode must be 'static' or 'ctx', got {fl!r}")
    if not 0.0 < float(_param("falcon1", "beta")) < 2.0:
        raise ValueError("falcon_beta must be in (0, 2)")
    if float(_param("falcon1", "lambda")) < 0.0:
        raise ValueError("falcon_lambda must be >= 0")
    if int(_param("falcon1", "window")) < 1:
        raise ValueError("falcon_window must be >= 1")
    if int(_param("falcon1", "conv_kernel")) < 1:
        raise ValueError("falcon_conv_kernel must be >= 1")
    if float(_param("falcon1", "eps")) <= 0.0:
        raise ValueError("falcon_eps must be > 0")
    if not 0.0 < float(_param("falcon1", "eps_gamma")) < 1.0:
        raise ValueError("falcon_eps_gamma must be in (0, 1)")

    # ---- MoE: softmax top-1 keeps the router trainable via pre-softmax ----
    import warnings

    moe = cfg.moe
    if (
        moe.enabled
        and moe.router.scoring == "softmax"
        and moe.top_k == 1
        and not moe.router.pre_softmax
    ):
        moe.router.pre_softmax = True
        warnings.warn(
            "moe.router.scoring='softmax' with top_k=1: enabling "
            "moe.router.pre_softmax automatically (a top-1 post-softmax weight "
            "is always 1 and gives the router no gradient).",
            UserWarning,
            stacklevel=2,
        )
    if moe.enabled and moe.balancing.bias_update != "none" and moe.router.scoring == "softmax":
        raise ValueError(
            "moe.balancing.bias_update requires router scoring 'sigmoid' or "
            "'softplus_sqrt': the expert bias is added to the scores before "
            "top-k and is not tunable under softmax normalisation "
            "(Wang et al., arXiv:2408.15664)."
        )

    # ---- FFN ----
    from src.registry import ComponentFamily, ensure_builtins, registry

    ensure_builtins()
    if cfg.ffn.activation not in registry.names(ComponentFamily.ACTIVATION):
        raise ValueError(
            f"ffn_activation must be one of "
            f"{sorted(registry.names(ComponentFamily.ACTIVATION))}, got "
            f"{cfg.ffn.activation!r}"
        )
    if cfg.ffn.activation_config is not None:
        if not isinstance(cfg.ffn.activation_config, dict):
            raise ValueError(
                "ffn_activation_config must be a mapping (dict) when provided"
            )
        _validate_ffn_activation_config(cfg.ffn.activation, cfg.ffn.activation_config)

    # ---- Vision Transformer ----
    if image is not None:
        valid_vit_pe = {
            "learned_1d", "none", "learned_absolute", "sinusoidal_absolute",
            "sinusoidal_rotary", "rope", "hope", "nope", "alibi", "bam", "pape",
            "pape_efficient", "pape_ri",
        }
        if image.pos_embedding_type not in valid_vit_pe:
            raise ValueError(
                f"pos_embedding_type must be one of {sorted(valid_vit_pe)}, got "
                f"{image.pos_embedding_type!r}"
            )
        if image.pos_embedding_type == "learned_1d":
            image.pos_embedding_type = "learned_absolute"
        if image.pooling_mode not in {"cls", "gap"}:
            raise ValueError(
                f"pooling_mode must be 'cls' or 'gap', got {image.pooling_mode!r}"
            )
        if image.mask_token_strategy not in {"bert", "mask_only", "random_only"}:
            raise ValueError(
                "mask_token_strategy must be 'bert', 'mask_only', or 'random_only', "
                f"got {image.mask_token_strategy!r}"
            )
        if image.prediction_target not in {
            "mean_color_3bit", "downsampled_3bit", "full_patch_l2"
        }:
            raise ValueError(
                "prediction_target must be 'mean_color_3bit', 'downsampled_3bit', "
                f"or 'full_patch_l2', got {image.prediction_target!r}"
            )
        if image.seg_head_type not in {"pixel", "eomt"}:
            raise ValueError(
                f"seg_head_type must be 'pixel' or 'eomt', got {image.seg_head_type!r}"
            )
        if int(image.patch_size) < 1:
            raise ValueError(f"patch_size must be >= 1, got {image.patch_size}")
        if int(image.height) % int(image.patch_size) != 0:
            raise ValueError(
                f"image_height ({image.height}) must be divisible by patch_size "
                f"({image.patch_size})"
            )
        if int(image.width) % int(image.patch_size) != 0:
            raise ValueError(
                f"image_width ({image.width}) must be divisible by patch_size "
                f"({image.patch_size})"
            )
        if not 0.0 < float(image.mask_ratio) < 1.0:
            raise ValueError(
                f"mask_ratio must be in the range (0, 1), got {image.mask_ratio}"
            )
        if int(image.num_classes) < 1:
            raise ValueError(f"num_classes must be >= 1, got {image.num_classes}")
        if int(image.num_seg_classes) < 1:
            raise ValueError(f"num_seg_classes must be >= 1, got {image.num_seg_classes}")
        if int(image.seg_num_queries) < 1:
            raise ValueError(f"seg_num_queries must be >= 1, got {image.seg_num_queries}")
        if int(image.seg_l2_blocks) < 1:
            raise ValueError(f"seg_l2_blocks must be >= 1, got {image.seg_l2_blocks}")

    # ---- Classification head ----
    head = cfg.classification_head
    if head.enabled and head.num_labels is None:
        raise ValueError("num_labels is required when classification_head=True")
    if head.num_labels is not None and int(head.num_labels) < 1:
        raise ValueError(f"num_labels must be >= 1, got {head.num_labels}")
    if str(head.pooling_mode) not in ("cls", "gap"):
        raise ValueError(
            f"encoder_pooling_mode must be 'cls' or 'gap', got {head.pooling_mode}"
        )

    # ---- Audio ----
    audio = cfg.audio
    if audio is not None:
        if str(audio.feature).lower() not in {"logmel", "spectrogram", "raw"}:
            raise ValueError(
                f"audio_feature must be one of ['logmel', 'raw', 'spectrogram'], got "
                f"{audio.feature!r}"
            )
        audio.feature = str(audio.feature).lower()
        for name in ("sample_rate", "n_mels", "n_fft", "in_channels", "conv_channels"):
            if int(getattr(audio, name)) < 1:
                raise ValueError(f"audio_{name} must be >= 1, got {getattr(audio, name)}")
        for name in ("window_ms", "stride_ms"):
            if float(getattr(audio, name)) <= 0.0:
                raise ValueError(f"audio_{name} must be > 0, got {getattr(audio, name)}")
        window_samples = int(
            round(float(audio.window_ms) * int(audio.sample_rate) / 1000.0)
        )
        if int(audio.n_fft) < max(1, window_samples):
            raise ValueError(
                f"audio_n_fft ({audio.n_fft}) must be >= the window length in "
                f"samples ({max(1, window_samples)} samples for {audio.window_ms} ms "
                f"at {audio.sample_rate} Hz)"
            )
        if str(audio.asr_loss).lower() not in {"ctc", "transducer"}:
            raise ValueError(
                f"asr_loss must be one of ['ctc', 'transducer'], got {audio.asr_loss!r}"
            )
        audio.asr_loss = str(audio.asr_loss).lower()
        if int(audio.asr_decoder_dim) < 1:
            raise ValueError(f"asr_decoder_dim must be >= 1, got {audio.asr_decoder_dim}")
        if int(audio.asr_decoder_layers) < 1:
            raise ValueError(
                f"asr_decoder_layers must be >= 1, got {audio.asr_decoder_layers}"
            )
        if not 0.0 <= float(audio.asr_decoder_dropout) < 1.0:
            raise ValueError(
                "asr_decoder_dropout must be in the range [0, 1), got "
                f"{audio.asr_decoder_dropout}"
            )
        spec_augment = audio.spec_augment
        if int(spec_augment.freq_mask_width) < 1:
            raise ValueError("audio_spec_augment_freq_mask_width must be >= 1")
        if int(spec_augment.time_masks) < 1:
            raise ValueError("audio_spec_augment_time_masks must be >= 1")
        if not 0.0 < float(spec_augment.time_mask_ratio) <= 1.0:
            raise ValueError("audio_spec_augment_time_mask_ratio must be in (0, 1]")
