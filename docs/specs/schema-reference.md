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
| `tokenizer` | No | Tokenizer config for NLP tasks (`hf_repo` load or `train_from_dataset`) — available to both `base_model` and custom-model paths |

## Model Fields

The `model:` block is hierarchical. Each sub-object enforces
`additionalProperties: false`; the paths below are the canonical YAML keys.

### `model.dims.*`

| Field | Type | Required | Range/Enum | Default | Description |
|---|---|---|---|---|---|
| `vocab_size` | int | Yes | ≥ 1 | `50000` | Vocabulary size |
| `hidden_size` | int | Yes | ≥ 1, divisible by `num_heads` | `2048` | Hidden dimension |
| `num_layers` | int | Yes | ≥ 1 | `12` | Physical layer count |
| `num_loops` | int | Yes | ≥ 1 | `2` | Logical loop count |
| `num_heads` | int | Yes | ≥ 1, divides `hidden_size` | `16` | Attention heads |
| `num_kv_heads` | int | No | ≥ 1, divides `num_heads` | `1` | KV heads for GQA/MSA/SparDA |
| `dropout` | float | No | [0, 1] | `0.1` | Global dropout rate |
| `layer_pattern` | array[enum] | Yes | 43 valid mixer codes | — | Ordered block list |
| `mode` | enum | No | `encoder`, `decoder` | `encoder` | Attention masking mode |

### `model.norm.*`

| Field | Type | Required | Range/Enum | Default | Description |
|---|---|---|---|---|---|
| `type` | enum | No | `layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`, `flash_norm` | `dynamic_tanh` | Normalization strategy |
| `partial_ratio` | float | No | (0, 1] | `0.0625` | Fraction of leading dims for `prms_norm` |
| `flashnorm_partial_ratio` | float | No | [0, 1] | `0.0` | Partial-RMS composition for `flash_norm` |

### `model.embedding.*`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `factorized.enabled` | bool | No | `false` | Enable factorized embeddings |
| `factorized.dim` | int | No | `128` | Reduced embedding dimension |
| `conv.enabled` | bool | No | `true` | Enable Conv1d over embeddings |
| `conv.kernel` | int | No | `3` | Conv1d kernel size |

### `model.positional_encoding` / `model.positional_encoding_parameters.*`

| Field | Type | Required | Range/Enum | Default | Description |
|---|---|---|---|---|---|
| `positional_encoding` | enum | No | `rope`, `hope`, `nope`, `alibi`, `bam`, `pape`, `pape_efficient`, `pape_ri`, `sinusoidal_absolute`, `sinusoidal_rotary`, `learned_absolute`, `none` | `rope` | Model-wide positional encoding |
| `parameters.rope.base` | number | No | ≥ 0 | `10000.0` | RoPE base frequency |
| `parameters.rope.scaling` | number | No | ≥ 0 | `1.0` | RoPE position scaling |
| `parameters.hope.base` | number | No | ≥ 0 | `10000.0` | HoPE base frequency |
| `parameters.hope.damping` | number | No | ≥ 0 | `0.01` | HoPE damping coefficient |
| `parameters.alibi.num_heads` | int/null | No | ≥ 1 or `null` | `null` (= `dims.num_heads`) | ALiBi slope-schedule head count |
| `parameters.bam.learn_mu` | bool | No | — | `false` | Make BAM θ_µ learnable |
| `parameters.bam.theta_init` | number | No | — | `0.0` | BAM θ_α/θ_β init (0 = Uniform prior) |
| `parameters.bam.mu_init` | number | No | — | `0.0` | BAM θ_µ init |
| `parameters.bam.eps` | number | No | > 0 | `1e-5` | BAM β<0 stability floor |
| `parameters.pape.num_parabolas` | int | No | ≥ 1 | `4` | PaPE parabolas |
| `parameters.pape.num_positions` | int | No | ≥ 1 | `1` | PaPE position dimensions |
| `parameters.pape.rotation_invariant` | bool | No | — | `false` | PaPE-RI flag (equivalent to `positional_encoding: pape_ri`) |
| `parameters.sinusoidal.max_len` | int | No | ≥ 1 | `512` | Sinusoidal table length |
| `parameters.sinusoidal.base` | number | No | ≥ 0 | `10000.0` | Sinusoidal base |
| `parameters.sinusoidal.scale` | number | No | — | `1.0` | Sinusoidal scale |
| `parameters.learned.max_len` | int | No | ≥ 1 | `512` | Learned-absolute max length |
| `parameters.learned.init_std` | number | No | ≥ 0 | `0.02` | Learned-absolute init std |
| `parameters.use_pe.<mixer>` | bool | No | — | true (attention) / false (recurrent) | Per-mixer PE opt-out |

### `model.attention.*`

Per-mixer sub-objects (`mla`, `gqla`, `mlra`, `tucker`, `iha`, `gta`, `mtla`,
`cca`, `ccgqa`, `msa`, `sparda`, `engram`, `gma`, `ssog`, `falcon`, `ode`,
`retnet`). Optional rank fields are `int | null`; `null` means "derive from
`hidden_size` / `num_heads`". See `configs/README.md` for the per-mixer
parameter catalogue.

### Feature blocks

| Field | Type | Required | Range/Enum | Default | Description |
|---|---|---|---|---|---|
| `moe.enabled` | bool | No | — | `true` | Enable MoE FFN routing |
| `moe.num_experts` | int | No | ≥ 1 | `8` | MoE expert count |
| `moe.top_k` | int | No | ≥ 1, ≤ `moe.num_experts` | `2` | Top-k expert routing |
| `bitnet.enabled` | bool | No | — | `true` | Enable BitLinear (BitNet b1.58) |
| `bitnet.routers` | bool | No | — | `false` | Also quantize routing/scoring projections |
| `bitnet.conv` | bool | No | — | `false` | Also quantize the embedding Conv1d |
| `mixture_of_depths.enabled` | bool | No | — | `false` | Enable MoD token routing |
| `mixture_of_depths.capacity_ratio` | number | No | (0, 1] | `0.5` | MoD token update fraction |
| `mixture_of_depths.router_aux_loss_weight` | number | No | ≥ 0 | `0.0` | MoD router aux loss weight |
| `ssmax.enabled` | bool | No | — | `false` | Enable Scalable Softmax |
| `ssmax.s_init` | number | No | > 0 | `1.0` | Per-head SSMax scalar init |
| `ffn.hidden_size` | int | No | ≥ 1 | `null` (2× hidden) | FFN intermediate width |
| `ffn.activation` | enum | No | 43 values (see [Activations](activations.md)) | `silu` | FFN non-linearity |
| `ffn.activation_config` | object | No | `additionalProperties: false` | — | Learnable-activation params |
| `classification_head.enabled` | bool | No | — | `false` | NLP sequence-classification head |
| `classification_head.num_labels` | int/null | No | ≥ 1 or `null` | `null` | Number of classes |
| `classification_head.pooling_mode` | enum | No | `cls`, `gap` | `cls` | Head pooling |

### `model.mhc.*` / `model.residuals.*`

| Field | Type | Required | Range/Enum | Default | Description |
|---|---|---|---|---|---|
| `mhc.enabled` | bool | No | — | `false` | Enable the mHC n-stream residual (see [mHC](mhc.md)) |
| `mhc.expansion_rate` | int | No | ≥ 1 | `4` | Stream expansion factor `n` |
| `mhc.sinkhorn_iters` | int | No | ≥ 1 | `20` | Sinkhorn-Knopp rounds |
| `mhc.gating_init` | float | No | > 0 | `0.01` | Initial gating scalar |
| `mhc.checkpoint` | bool | No | — | `false` | Gradient checkpointing on mHC layers |
| `mhc.full_prec_under_bitnet` | bool | No | — | `true` | Keep mHC projection full-precision under BitNet |
| `residuals.type` | enum | No | `standard`, `none`, `full_attn`, `block_attn` | `standard` | Residual strategy |
| `residuals.mhc_stream_mode` | enum | No | `independent`, `joint` | `independent` | AttnRes × mHC interaction |
| `residuals.gradient_checkpoint` | bool | No | — | `false` | Checkpoint AttnRes attention |

### Layer Pattern Valid Values

`retnet`, `retnet_attn`, `mamba`, `ode`, `titan_attn`, `standard_attn`, `sigmoid_attn`, `sparse_transformer_attn`, `longformer_attn`, `bigbird_attn`, `sparsek_attn`, `nsa_attn`, `sparge_attn`, `fasa_attn`, `gla_attn`, `deltanet_attn`, `gated_deltanet_attn`, `gated_deltanet2_attn`, `hgrn2_attn`, `fox_attn`, `gated_softmax_attn`, `kda_attn`, `engram_attn`, `gqa_attn`, `mla_attn`, `gqla_attn`, `mlra_attn`, `tucker_attn`, `iha_attn`, `gta_attn`, `mtla_attn`, `cca_attn`, `ccgqa_attn`, `gma_attn`, `ssog_attn`, `msa_attn`, `sparda_attn`, `falcon1_attn`, `falcon2_attn`, `falcon3_attn`, `falcon1a_attn`, `falcon2a_attn`, `falcon3a_attn`

> **Eval-only:** `sparge_attn` and `fasa_attn` cannot be used during training.
> **Encoder-only:** `ssog_attn` requires `mode: encoder` and a fixed
> `grid_h × grid_w` raster (ViT: `cls_token: false`).

### Normalization Valid Values

`layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`, `flash_norm`

### Vision / Dataset / Tokenizer fields

Vision, dataset, and tokenizer configs are top-level blocks. For the full
vision field reference see [Vision](vision.md):

| Section | Fields | Notes |
|---|---|---|
| `image` | `image_size`, `patch_size`, `in_channels`, `to_grayscale`, `pos_embedding_type`, `cls_token`, `pooling_mode`, `mask_ratio`, `mask_token_strategy`, `prediction_target`, `seg_head_type`, `num_classes`, `num_seg_classes`, `seg_num_queries`, `seg_l2_blocks`, `seg_mask_annealing` | Required for vision tasks |
| `vision_dataset` | source, columns, rescaling, augmentations | Required for vision tasks |
| `text_dataset` | `dataset_name`, `split`, `text_column`, `label_column`, `data_dir`, `streaming`, `use_labels`, `max_samples` | Host-owned NLP corpus (also the `train_from_dataset` tokenizer corpus) |
| `tokenizer` | `source`, `name_or_path`, `use_fast`, `trust_remote_code`, `training` | Required when using `base_model` for MLM; available for custom models too |

All of these enforce `additionalProperties: false`.

### Tokenizer Fields

The `tokenizer` block is available to both `base_model` (continual pretraining /
fine-tuning) and custom from-scratch model paths. Two modes are selected via
`source`:

| Field | Type | Required | Range/Enum | Default | Description |
|---|---|---|---|---|---|
| `source` | enum | No | `hf_repo`, `train_from_dataset` | `hf_repo` | Tokenizer source mode |
| `name_or_path` | string | Yes (when `source=hf_repo`) | — | — | HuggingFace tokenizer id or local path (`AutoTokenizer.from_pretrained`) |
| `use_fast` | bool | No | — | `true` | Use the Rust fast tokenizer implementation |
| `trust_remote_code` | bool | No | — | `false` | Allow custom tokenizer code from the repo (security risk) |
| `training` | object | Yes (when `source=train_from_dataset`) | — | — | Tokenizer-training config (see below) |

**Mode `hf_repo` (default):** loads a pretrained tokenizer from a repo id or
local path. Required for `base_model` MLM training.

**Mode `train_from_dataset`:** trains a NEW tokenizer from the corpus text
(HuggingFace `tokenizers` library). The corpus comes from the `text_dataset`
block (or the DashAI run dataset when the plugin's *Use DashAI dataset for
tokenizer* checkbox is checked). The trained vocabulary size
(`len(tokenizer)`) is injected into `model.dims.vocab_size` before model
construction, so the embedding matrix always matches. `name_or_path` must be
omitted in this mode.

#### `tokenizer.training.*`

| Field | Type | Required | Range/Enum | Default | Description |
|---|---|---|---|---|---|
| `algorithm` | enum | No | `bpe`, `wordpiece`, `wordlevel`, `unigram` | `bpe` | Tokenizer training algorithm |
| `vocab_size` | int | Yes (bpe/wordpiece/wordlevel) | ≥ 1 | — | Target vocabulary size (optional for `unigram`, which auto-prunes) |
| `min_frequency` | int | No | ≥ 1 | `2` | Minimum token/subword frequency to keep |
| `special_tokens` | array[str] | No | — | `[PAD]`,`[UNK]`,`[CLS]`,`[SEP]`,`[MASK]` | Special tokens registered in the trained tokenizer |
| `save_dir` | string | No | — | — | Directory persisted via `save_pretrained()` |
| `lowercase` | bool | No | — | `false` | Lowercase all text before training |
| `strip_accents` | bool | No | — | `false` | Strip diacritics (`á → a`) before training |
| `pre_tokenizer` | enum | No | `whitespace`, `bytelevel`, `bert` | `whitespace` | Pre-tokenization strategy |
| `max_token_length` | int | No | ≥ 0 | `0` (no limit) | Per-token character cap |

**Critical gotcha:** when `source=train_from_dataset`, `dims.vocab_size` is
overridden by the trained tokenizer's actual vocabulary — an explicit
`dims.vocab_size` in the config is ignored. The `text_dataset` block (or the
DashAI run dataset) is required as the training corpus.

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
2. **`model.norm.type`** accepts `layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`, `flash_norm`. When using `prms_norm`, set `model.norm.partial_ratio` (default `0.0625`, range `(0, 1]`). `flash_norm` uses `model.norm.flashnorm_partial_ratio` (default `0.0`, range `[0, 1]`).
3. **`fasa_attn` and `sparge_attn`** are eval-only blocks. Training with either raises a runtime error.
4. **Optimizer parameters** use prefixed keys: `<optimizer_class>-<group>_<param>`.
5. **`training.task`** is required (`mlm`, `sbert`, `causal_lm`, or a vision task). Legacy top-level optimizer keys are not accepted.
6. **`model_class: frankensteindecoder`** forces `mode: decoder` at runtime.
7. **`layer_pattern` length must equal `num_layers`**.
8. **mHC is incompatible with MoD** (`mixture_of_depths.enabled`) — enabling both raises a `ValueError`.
9. **`bitnet.routers: true` requires `bitnet.enabled: true`** — enforced at config load.
10. **`tokenizer.source: train_from_dataset`** requires `tokenizer.training` (with `vocab_size` for bpe/wordpiece/wordlevel), is mutually exclusive with `name_or_path`, and overrides `dims.vocab_size` with the trained vocabulary.
