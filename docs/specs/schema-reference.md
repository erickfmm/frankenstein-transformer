# Schema Field Reference

> Cross-references: [Architecture](architecture.md) · [Attention Mixers](attention-mixers.md) · [Optimizers](optimizers.md) · [CLI Reference](cli-reference.md) · [Training Safety](training-safety.md) · [SBERT Workflows](sbert-workflows.md) · [Deployment](deployment.md) · [mHC](mhc.md) · [Vision](vision.md)

## Strict Validation Policy

The schema (`src/schema.yaml`) enforces **`additionalProperties: false`** at all levels — top-level and every nested object. Unknown keys fail fast instead of being silently ignored. The `training.optimizer.parameters` object is additionally constrained by optimizer-specific prefix rules through `allOf` + `if/then` pattern checks.

## Top-Level Sections

| Section | Required | Description |
|---|---|---|
| `model_class` | No (optional when `base_model` set) | Architecture variant: `frankenstein`, `frankensteindecoder` |
| `model` | No (optional when `base_model` set) | FrankensteinModelConfig model parameters |
| `training` | **Yes** | Training hyperparameters and runtime configuration |
| `base_model` | No | HuggingFace model identifier for continual pretraining |
| `tokenizer` | No | Tokenizer config when `base_model` is used |

## Model Fields

| Field | Type | Required | Range/Enum | Default | Description |
|---|---|---|---|---|---|
| `vocab_size` | int | Yes | ≥ 1 | — | Vocabulary size |
| `hidden_size` | int | Yes | ≥ 1, divisible by `num_heads` | — | Hidden dimension |
| `num_layers` | int | Yes | ≥ 1 | — | Physical layer count |
| `num_loops` | int | Yes | ≥ 1 | — | Logical loop count |
| `num_heads` | int | Yes | ≥ 1, divides `hidden_size` | — | Attention heads |
| `retention_heads` | int | Yes | ≥ 1 | — | Retention heads for RetNet |
| `num_experts` | int | Yes | ≥ 1 | — | MoE expert count |
| `top_k_experts` | int | Yes | ≥ 1, ≤ `num_experts` | — | Top-k expert routing |
| `dropout` | float | Yes | [0, 1] | — | Global dropout rate |
| `layer_pattern` | array[enum] | Yes | 36 valid mixer codes | — | Ordered block list |
| `ode_solver` | enum | Yes | `rk4`, `euler` | — | ODE integration method |
| `ode_steps` | int | Yes | ≥ 1 | — | ODE integration steps |
| `use_bitnet` | bool | Yes | — | — | Enable BitLinear path |
| `bitnet_routers` | bool | Yes | — | `false` | Also quantize routing/scoring projections |
| `use_bitnet_conv` | bool | Yes | — | `false` | Also quantize the embedding Conv1d (opt-in) |
| `norm_type` | enum | Yes | `layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`, `flash_norm` | — | Normalization strategy |
| `prms_partial_ratio` | float | No | (0, 1] | `0.0625` | Fraction of leading dims used for partial RMS (`prms_norm`) |
| `flashnorm_partial_ratio` | float | No | [0, 1] | `0.0` | Partial-RMS composition for `flash_norm` |
| `use_factorized_embedding` | bool | Yes | — | — | Enable factorized embeddings |
| `factorized_embedding_dim` | int | Yes | ≥ 1 | — | Reduced embedding dimension |
| `use_embedding_conv` | bool | Yes | — | — | Enable Conv1d over embeddings |
| `embedding_conv_kernel` | int | Yes | ≥ 1 | — | Conv1d kernel size |
| `use_hope` | bool | Yes | — | — | **Deprecated** legacy HoPE flag (mapped to `positional_encoding`) |
| `use_moe` | bool | Yes | — | — | Enable MoE FFN routing |
| `ffn_hidden_size` | int | Yes | ≥ 1 | — | FFN intermediate width |
| `ffn_activation` | enum | Yes | 43 values (see [Activations](activations.md)) | `silu` | FFN non-linearity |
| `ffn_activation_config` | object | No | `additionalProperties: false` | — | Learnable-activation params (RAF degrees/version, PReLU init, parametric alphas) |
| `mode` | enum | No | `encoder`, `decoder` | — | Attention masking mode |
| `positional_encoding` | enum | No | `rope`, `hope`, `nope`, `alibi`, `pape`, `pape_efficient`, `pape_ri`, `sinusoidal_absolute`, `sinusoidal_rotary`, `learned_absolute`, `none` | `rope` | Model-wide positional encoding (see [PE annex](../paper/appendices/annex-12-positional-encodings.tex)) |
| `hope_base` | number | No | ≥ 0 | — | HoPE base frequency |
| `hope_damping` | number | No | ≥ 0 | — | HoPE damping coefficient |
| `rope_base` | number | No | ≥ 0 | — | RoPE base frequency |
| `rope_scaling` | number | No | ≥ 0 | — | RoPE scaling factor |
| `alibi_num_heads` | int | No | ≥ 1 | `num_heads` | ALiBi slope-schedule head count |
| `pape_num_parabolas` | int | No | ≥ 1 | 4 | PaPE number of parabolas |
| `pape_num_positions` | int | No | ≥ 1 | 1 | PaPE number of positions per parabola |
| `pape_rotation_invariant` | bool | No | — | `false` | PaPE-RI rotation-invariant flag |
| `sinusoidal_max_len` | int | No | ≥ 1 | 512 | Sinusoidal precomputed table length |
| `sinusoidal_base` | number | No | > 0 | 10000 | Sinusoidal base frequency |
| `sinusoidal_scale` | number | No | ≥ 0 | 1.0 | Sinusoidal embedding scale |
| `learned_max_len` | int | No | ≥ 1 | 512 | Learned-absolute max sequence length |
| `learned_init_std` | number | No | ≥ 0 | 0.02 | Learned-absolute init std |
| `positional_encoding_parameters` | object | No | `additionalProperties: false` | — | Nested PE sub-config (`rope`, `hope`, `alibi`, `pape`, `sinusoidal`, `learned`, `use_pe`) |
| `<mixer>_attn_use_pe` | bool | No | — | `True`/`False` | Per-mixer PE opt-out (see [PE annex](../paper/appendices/annex-12-positional-encodings.tex)); `False` default for recurrent/decay mixers |
| `use_mixture_of_depths` | bool | No | — | — | Enable MoD token routing |
| `mixture_of_depths_capacity_ratio` | number | No | (0, 1] | — | MoD token update fraction |
| `mixture_of_depths_router_aux_loss_weight` | number | No | ≥ 0 | — | MoD router aux loss weight |
| `engram_max_ngram_size` | int | No | ≥ 2 | — | Max N-gram order for Engram |
| `engram_n_heads_per_ngram` | int | No | ≥ 1 | — | Hash heads per N-gram |
| `engram_embed_dim_per_head` | int | No | ≥ 1 | — | Embed dim per hash head |
| `engram_kernel_size` | int | No | ≥ 1 | — | ShortConv kernel width |
| `engram_seed` | int | No | — | — | Hash seed |
| `gma_num_components` | int | No | ≥ 1 | 8 | Number K of Gaussian mixture components per head (GMA) |
| `gma_routing_dim` | int | No | ≥ 1 | head_dim | Routing dimension d_r for GMA responsibilities |
| `gma_epsilon` | number | No | > 0 | 1e-6 | Read-normalizer ε for GMA |
| `gma_sigma_eps` | number | No | > 0 | 1e-4 | Diagonal covariance floor ε_σ for GMA |
| `gma_init_mean_std` | number | No | — | 1.0 | Init std for GMA component means μ |
| `use_mhc` | bool | No | — | `false` | Enable the mHC n-stream residual (see [mHC](mhc.md)) |
| `mhc_expansion_rate` | int | No | ≥ 1 | `4` | Stream expansion factor `n` |
| `mhc_sinkhorn_iters` | int | No | ≥ 1 | `20` | Sinkhorn-Knopp normalisation rounds |
| `mhc_gating_init` | float | No | > 0 | `0.01` | Initial gating scalar value |
| `mhc_checkpoint` | bool | No | — | `false` | Gradient checkpointing on mHC layers |
| `mhc_full_prec_under_bitnet` | bool | No | — | `true` | Keep mHC projection full-precision under BitNet |
| `residual_type` | enum | No | `standard`, `none`, `full_attn`, `block_attn` | — | Residual connection strategy |

### Layer Pattern Valid Values

`retnet`, `retnet_attn`, `mamba`, `ode`, `titan_attn`, `standard_attn`, `sigmoid_attn`, `sparse_transformer_attn`, `longformer_attn`, `bigbird_attn`, `sparsek_attn`, `nsa_attn`, `sparge_attn`, `fasa_attn`, `gla_attn`, `deltanet_attn`, `gated_deltanet_attn`, `gated_deltanet2_attn`, `hgrn2_attn`, `fox_attn`, `gated_softmax_attn`, `kda_attn`, `engram_attn`, `gqa_attn`, `mla_attn`, `gqla_attn`, `mlra_attn`, `tucker_attn`, `iha_attn`, `gta_attn`, `mtla_attn`, `cca_attn`, `ccgqa_attn`, `gma_attn`, `msa_attn`, `sparda_attn`

> **Eval-only:** `sparge_attn` and `fasa_attn` cannot be used during training.

### Normalization Valid Values

`layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`, `flash_norm`

### Vision / Dataset / Tokenizer fields

Vision, dataset, and tokenizer configs are top-level blocks. For the full
vision field reference see [Vision](vision.md):

| Section | Fields | Notes |
|---|---|---|
| `image` | `image_size`, `patch_size`, `in_channels`, `to_grayscale`, `pos_embedding_type`, `cls_token`, `pooling_mode`, `mask_ratio`, `mask_token_strategy`, `prediction_target`, `seg_head_type`, `num_classes`, `num_seg_classes`, `seg_num_queries`, `seg_l2_blocks`, `seg_mask_annealing` | Required for vision tasks |
| `dataset` | source, columns, rescaling, augmentations | Required for vision tasks |
| `tokenizer` | `name_or_path`, plus tokenizer options | Required when using `base_model` for MLM |

All of these enforce `additionalProperties: false`.

### Vision Task Sub-Blocks

Vision tasks (`patch_prediction`, `classification`, `segmentation`) reuse the
top-level `training:` fields (`batch_size`, `num_epochs`, `optimizer`,
`scheduler_type`, `grad_clip_max_norm`, checkpointing, etc.) — the same fields
used by `mlm` and `causal_lm`. The optional `training.<task>:` sub-block only
carries task-specific knobs:

| Sub-block | Fields | Description |
|---|---|---|
| `training.classification` | `label_smoothing` (float, default 0) | Label smoothing for the classification cross-entropy loss |
| `training.segmentation` | `seg_loss_bce_weight` (default 5), `seg_loss_dice_weight` (default 5), `seg_loss_ce_weight` (default 2), `mask_annealing_factor` (default 0.9) | EoMT loss weights and mask annealing decay factor |
| `training.patch_prediction` | _(none)_ | Empty sub-block kept for schema consistency |

## Training Fields

| Field | Type | Required | Range/Enum | Default | Description |
|---|---|---|---|---|---|
| `task` | enum | **Yes** | `mlm`, `sbert`, `causal_lm`, `patch_prediction`, `classification`, `segmentation` | — | Training objective |
| `num_epochs` | int | No | ≥ 1 | — | Training epochs (MLM, causal_lm, and vision tasks) |
| `batch_size` | int | No | ≥ 1 | — | Loader batch size (MLM, causal_lm, and vision tasks) |
| `dataloader_workers` | int | No | ≥ 0 | — | PyTorch dataloader workers |
| `max_length` | int | No | ≥ 1 | — | Sequence length cap |
| `mlm_probability` | float | No | [0, 1] | — | MLM masking probability |
| `max_samples` | int | No | ≥ 1 | — | Maximum streamed samples |
| `dataset_batch_size` | int | No | ≥ 1 | — | Internal streaming chunk size |
| `num_workers` | int | No | ≥ 0 | — | Streaming dataset workers |
| `cache_dir` | string | No | — | — | Dataset cache directory |
| `local_parquet_dir` | string | No | — | — | Local parquet path |
| `prefer_local_cache` | bool | No | — | — | Prefer local cache |
| `stream_local_parquet` | bool | No | — | — | Stream from local parquet |
| `join_temp_data_context_window` | int | No | ≥ 0 | — | Join cached chunks to this length |
| `join_temp_data_min_remainder_tokens` | int | No | ≥ 0 | — | Min tokens in last partial window |
| `use_amp` | bool | No | — | — | Mixed precision toggle |
| `gradient_accumulation_steps` | int | No | ≥ 1 | — | Effective batch multiplier |
| `scheduler_total_steps` | int | No | ≥ 1 | — | Scheduler horizon |
| `scheduler_warmup_ratio` | float | No | [0, 1] | — | Warmup ratio |
| `scheduler_type` | enum | No | `cosine`, `constant`, `linear_warmup_then_constant` | — | LR schedule type |
| `grad_clip_max_norm` | float | No | ≥ 0 | — | Global norm clipping threshold |
| `inf_post_clip_threshold` | float | No | ≥ 0 | — | Post-clip explosion guard |
| `max_nan_retries` | int | No | ≥ 0 | — | NaN/Inf retry budget |
| `checkpoint_every_n_steps` | int | No | ≥ 1 | — | Rolling checkpoint frequency |
| `max_rolling_checkpoints` | int | No | ≥ 1 | — | Rolling checkpoints to keep |
| `num_best_checkpoints` | int | No | ≥ 1 | — | Best checkpoints tracked |
| `nan_check_interval` | int | No | ≥ 1 | — | NaN/Inf check cadence |
| `log_gradient_stats` | bool | No | — | — | Enable gradient stats logging |
| `gradient_log_interval` | int | No | ≥ 1 | — | Gradient logging cadence |
| `csv_log_path` | string | No | — | — | Step-level CSV output path |
| `csv_rotate_on_schema_change` | bool | No | — | — | Rotate CSV on schema change |
| `gpu_metrics_backend` | enum | No | `nvml`, `none` | — | GPU telemetry backend |
| `nvml_device_index` | int | No | ≥ 0 | — | NVML device index |
| `enable_block_grad_norms` | bool | No | — | — | Per-block gradient norm telemetry |
| `telemetry_log_interval` | int | No | ≥ 1 | — | Heavy telemetry interval |
| `use_galore` | bool | No | — | — | Enable GaLore strategy |
| `galore_rank` | int | No | ≥ 1 | — | GaLore low-rank projection dim |
| `galore_update_interval` | int | No | ≥ 1 | — | Projection refresh interval |
| `galore_scale` | float | No | ≥ 0 | — | Gradient scaling in projected space |
| `galore_max_dim` | int | No | ≥ 1 | — | Max tensor dim for GaLore projection |

## Optimizer Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `training.optimizer.optimizer_class` | enum | **Yes** | One of 23 optimizer codes |
| `training.optimizer.parameters` | object | **Yes** | Prefixed hyperparameter groups |

### Valid `optimizer_class` Values

`sgd_momentum`, `adamw`, `adafactor`, `galore_adamw`, `prodigy`, `lion`, `sophia`, `muon`, `turbo_muon`, `radam`, `adan`, `adopt`, `ademamix`, `mars_adamw`, `cautious_adamw`, `lamb`, `schedulefree_adamw`, `shampoo`, `soap`, `anon`, `apollo`, `apollo_mini`, `q_apollo`

## Scheduler Types

| Type | Behavior |
|---|---|
| `cosine` | Cosine decay from initial LR to near-zero over `scheduler_total_steps` |
| `constant` | Fixed LR throughout training |
| `linear_warmup_then_constant` | Linear warmup over `warmup_ratio × total_steps`, then constant |

## GPU Metrics Backends

| Backend | Description |
|---|---|
| `nvml` | NVIDIA Management Library — GPU temperature, utilization, memory |
| `none` | Disable GPU telemetry |

## SBERT Dataset Types

| Type | Description |
|---|---|
| `paired_similarity` | Sentence pairs with similarity scores (cosine regression) |
| `triplets` | Anchor-positive-negative triplets for contrastive learning |
| `qa` | Question-answer pairs for semantic search training |

## Critical Gotchas

1. **`hidden_size` must be divisible by `num_heads`** — per-head dimension = `hidden_size / num_heads` should be ≥ 64.
2. **`norm_type`** accepts `layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`, `flash_norm`. When using `prms_norm`, set `prms_partial_ratio` (default `0.0625`, range `(0, 1]`). `flash_norm` uses `flashnorm_partial_ratio` (default `0.0`, range `[0, 1]`).
3. **`fasa_attn` and `sparge_attn`** are eval-only blocks. Training with either raises a runtime error.
4. **Optimizer parameters** use prefixed keys: `<optimizer_class>-<group>_<param>`.
5. **`training.task`** is required (`mlm`, `sbert`, `causal_lm`, or a vision task). Legacy top-level optimizer keys are not accepted.
6. **`model_class: frankensteindecoder`** forces `mode: decoder` at runtime.
7. **`layer_pattern` length must equal `num_layers`**.
8. **mHC is incompatible with MoD** (`use_mixture_of_depths`) — enabling both raises a `ValueError`.
9. **`bitnet_routers: true` requires `use_bitnet: true`** — enforced at config load.
