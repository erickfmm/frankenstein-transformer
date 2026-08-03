# Training Configurations (YAML)

This directory contains YAML presets. Each file describes the model (`model`) and training/dataset parameters (`training`).

## Files

- `frankenstein.yaml`
- `mini.yaml`
- `frankesteindecoder.yaml`
- `standard.yaml`
- `standard_hope.yaml`
- `tinybert.yaml`
- `embbert.yaml`
- `modernbert_continual.yaml`
- `modernbert_sbert.yaml`
- `schema.yaml` (JSON Schema in YAML form)

## General Structure

```yaml
# Option A: train custom Tormented model
model_class: frankenstein|mini|frankesteindecoder
model: # UltraConfig
  ...

# Option B: continual pretraining/finetuning from any HF/local base model
base_model: answerdotai/ModernBERT-base
tokenizer:
  name_or_path: answerdotai/ModernBERT-base
  use_fast: true
  trust_remote_code: false

training:
  task: mlm|sbert
  # TrainingConfig + runtime
  batch_size: 4
  dataloader_workers: 2
  max_length: 512
  mlm_probability: 0.15
  max_samples: 20000000
  dataset_batch_size: 25000
  num_workers: 8
  cache_dir: "./temp_data/v2_dataset_cache"
  local_parquet_dir: "/path/to/parquet"  # optional
  prefer_local_cache: true
  stream_local_parquet: true
  join_temp_data_context_window: 0   # e.g. 2000 or 8192 to join cached temp_data
  join_temp_data_min_remainder_tokens: 128
  use_amp: false
  gradient_accumulation_steps: 4
  optimizer:
    optimizer_class: adamw
    parameters:
      adamw-lr_embeddings: 1e-6
      adamw-lr_norms: 5e-6
      adamw-lr_attention: 3e-6
      adamw-lr_other: 2e-6
      adamw-wd_embeddings: 0.01
      adamw-wd_norms: 0.001
      adamw-wd_attention: 0.01
      adamw-wd_other: 0.01
      adamw-betas_embeddings: [0.9, 0.95]
      adamw-betas_norms: [0.9, 0.95]
      adamw-betas_attention: [0.9, 0.95]
      adamw-betas_other: [0.9, 0.95]
      adamw-eps_embeddings: 1e-8
      adamw-eps_norms: 1e-8
      adamw-eps_attention: 1e-8
      adamw-eps_other: 1e-8
  scheduler_total_steps: 10000
  scheduler_warmup_ratio: 0.1
  scheduler_type: cosine
  grad_clip_max_norm: 5.0
  inf_post_clip_threshold: 100.0
  max_nan_retries: 3
  checkpoint_every_n_steps: 500
  max_rolling_checkpoints: 3
  num_best_checkpoints: 2
  nan_check_interval: 10
  log_gradient_stats: true
  gradient_log_interval: 10
  csv_log_path: "training_metrics.csv"
  csv_rotate_on_schema_change: true
  gpu_metrics_backend: nvml
  nvml_device_index: 0
  enable_block_grad_norms: true
  telemetry_log_interval: 1
  gpu_temp_guard_enabled: true
  switch_on_thermal: false
  gpu_temp_pause_threshold_c: 90.0
  gpu_temp_resume_threshold_c: 80.0
  gpu_temp_critical_threshold_c: null
  gpu_temp_poll_interval_seconds: 30.0
  use_galore: false
  galore_rank: 64
  galore_update_interval: 1
  galore_scale: 1.0
  galore_max_dim: 4096

  # Used when task == sbert
  sbert:
    dataset_name: "erickfmm/agentlans__multilingual-sentences__paired_10_sts"
    dataset_type: paired_similarity
    columns:
      sentence1: sentence1
      sentence2: sentence2
      similarity: similarity
    query_prefix: ""
    document_prefix: ""
    output_dir: "./output/sbert_base_model"
    batch_size: 16
    epochs: 4
    warmup_steps: 1000
    evaluation_steps: 5000
    checkpoint_save_steps: 1000
    resume_from_checkpoint: false
    learning_rate: 2e-5
    max_train_samples: null
    max_eval_samples: 10000
    max_seq_length: 512
    pooling_mode: mean
    trust_remote_code: false
    use_amp: true
    resample_balanced: true
    standardize_scores: false
    resample_std: 0.3
```

## Optimizer Chooser

`training.optimizer.optimizer_class` supported values:
- `sgd_momentum`
- `adamw`
- `adafactor`
- `galore_adamw`
- `prodigy`
- `lion`
- `sophia`
- `muon`
- `turbo_muon`
- `radam`
- `adan`
- `adopt`
- `ademamix`
- `mars_adamw`
- `cautious_adamw`
- `lamb`
- `schedulefree_adamw`
- `shampoo`
- `soap`

`training.optimizer.parameters` must use the selected optimizer prefix. Example: when `optimizer_class: adamw`, all keys must start with `adamw-`.

### Optimizer Parameters Reference

All optimizers support these shared per-group parameter suffixes (use with optimizer prefix, e.g. `adamw-lr_embeddings`):

- `lr_embeddings`, `lr_norms`, `lr_attention`, `lr_other`
- `wd_embeddings`, `wd_norms`, `wd_attention`, `wd_other`
- `betas_embeddings`, `betas_norms`, `betas_attention`, `betas_other`
- `eps_embeddings`, `eps_norms`, `eps_attention`, `eps_other`

Optimizer-specific global parameter suffixes (also prefixed):

- `sgd_momentum`: `momentum`, `nesterov`
- `adamw`: none
- `adafactor`: `beta2_decay`, `clip_threshold`, `eps1`, `eps2`
- `galore_adamw`: `rank`, `update_proj_gap`
- `prodigy`: `d_coef`
- `lion`: none
- `sophia`: `rho`, `update_k`
- `muon`: `momentum`, `nesterov`, `ns_steps`, `ns_eps`
- `turbo_muon`: `momentum`, `nesterov`, `ns_steps`, `ns_eps`
- `radam`: none
- `adan`: none
- `adopt`: none
- `ademamix`: none
- `mars_adamw`: none
- `cautious_adamw`: `cautious_clip`
- `lamb`: none
- `schedulefree_adamw`: none
- `shampoo`: none
- `soap`: none

Examples:
- `adafactor-beta2_decay: 0.8`
- `galore_adamw-rank: 128`
- `sophia-update_k: 10`
- `muon-ns_steps: 5`

## Available Fields (Detailed)

### model_class
- `frankenstein`, `mini`, or `frankesteindecoder`.
- `frankesteindecoder` builds the autoregressive causal decoder variant (`FrankensteinDecoder`).
- Ignored when `base_model` is set.

### base_model
- Optional HF model id/local path.
- When set, `model_class` and `model` can be omitted and ignored by runtime.

### tokenizer
- Required for `training.task: mlm` when `base_model` is set.
- Uses `AutoTokenizer` via `tokenizer.name_or_path`.

### model (UltraConfig)

The `model` block is organized into hierarchical sub-objects: `dims`, `norm`,
`embedding`, `attention`, plus flat top-level keys for cross-cutting features
(MoE, BitNet, Mixture-of-Depths, FFN, ODE).

#### model.dims

- `vocab_size`: vocabulary size.
- `hidden_size`: hidden dimension.
- `num_layers`: physical layers.
- `num_loops`: logical loops.
- `num_heads`: attention heads.
- `num_kv_heads`: key-value heads for GQA.
- `retention_heads`: retention heads.
- `dropout`: global dropout.
- `layer_pattern`: list of blocks.
- Legacy blocks: `retnet`, `retnet_attn`, `mamba`, `ode`, `titan_attn`, `standard_attn`, `sigmoid_attn`.
- Sparse blocks: `sparse_transformer_attn`, `longformer_attn`, `bigbird_attn`, `sparsek_attn`, `nsa_attn`, `sparge_attn`, `fasa_attn`, `msa_attn` (MiniMax Sparse Attention, arXiv:2606.13392), `sparda_attn` (SparDA, arXiv:2606.04511).
- Gated blocks: `gla_attn`, `deltanet_attn`, `gated_deltanet_attn`, `hgrn2_attn`, `fox_attn`, `gated_softmax_attn`, `kda_attn` (Kimi Delta Attention, arXiv:2510.26692).
- Latent blocks (KV-compression / head-mixing, generalising GQA):
  - `mla_attn` — Multi-Head Latent Attention + RoPE (arXiv:2506.09342).
  - `gqla_attn` — Group-Query Latent Attention, two decoding paths (arXiv:2605.15250).
  - `mlra_attn` — Multi-Head Low-Rank Attention, partitionable latent (arXiv:2603.02188).
  - `tucker_attn` — Tucker Attention, generalised low-rank factorisation (arXiv:2603.30033).
  - `iha_attn` — Interleaved Head Attention, cross-head pseudo-heads (arXiv:2602.21371).
  - `gta_attn` — Grouped-head laTenT Attention, shared map + latent values (arXiv:2506.17286).
  - `mtla_attn` — Multi-head Temporal Latent Attention, temporal KV merge (arXiv:2505.13544).
  - `cca_attn` — Compressed Convolutional Attention, attention in compressed latent + convs (arXiv:2510.04476).
  - `ccgqa_attn` — Compressed Convolutional Grouped Query Attention, CCA + decoupled GQA in latent (arXiv:2510.04476).
- Training-free policy: `sparge_attn` and `fasa_attn` are eval-only; training mode raises an explicit runtime error.
- `mode`: `encoder` or `decoder` (if omitted, defaults to encoder unless `model_class: frankesteindecoder`, which forces decoder).

#### model.norm

- `type`: `layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`.
- `partial_ratio`: fraction of dims for pRMSNorm (default 0.0625).

#### model.embedding

- `factorized.enabled`: enable factorized embeddings.
- `factorized.dim`: reduced embedding dimension.
- `conv.enabled`: optional Conv1d on embeddings.
- `conv.kernel`: Conv1d kernel size.

#### model.attention.titan

- `positional_encoding`: titan positional encoding (`hope` or `rope`). If omitted, legacy `use_hope` is used.
- `use_hope`: legacy fallback for `titan_attn` when `positional_encoding` is omitted.
- `hope.base`: HoPE base.
- `hope.damping`: HoPE damping.
- `rope.base`: RoPE base.
- `rope.scaling`: RoPE position scaling.

#### Flat model.* keys

- `num_experts`: MoE experts.
- `top_k_experts`: top-k routing in MoE.
- `ode_solver`: `rk4` or `euler`.
- `ode_steps`: integration steps.
- `use_bitnet`: BitLinear on/off (BitNet b1.58). When true, all primary + gate
  projections (Q/K/V/O, FFN, embeddings `proj`, LM head, and recurrent-state
  gates) become ternary `{-1,0,1}`.
- `bitnet_routers`: when `true` (and `use_bitnet`), also quantize
  routing/scoring projections (MoE router, MoD router, sparse block-index,
  forecast, top-k score nets). Default `false` keeps them full-precision for
  routing stability.
- `use_bitnet_conv`: when `true` (and `use_bitnet` and `model.embedding.conv.enabled`),
  also quantize the embedding Conv1d to `BitConv1d`. Default `false` (opt-in).
- `use_moe`: use MoE in FFN.
- `use_mixture_of_depths`: enable Mixture-of-Depths token routing.
- `mixture_of_depths_capacity_ratio`: fraction of tokens updated per MoD layer.
- `mixture_of_depths_router_aux_loss_weight`: MoD router aux loss weight.
- `ffn_hidden_size`: FFN intermediate dimension.
- `ffn_activation`: `silu` or `gelu` (or any of the 43 supported activations).
- `ffn_activation_config`: nested object for learnable/parametric activations.

### Latent / KV-compression family (model.attention.*)

- `model.attention.mla.latent_rank`: rank of the joint KV latent for `mla_attn` (default `hidden_size // 2`).
- `model.attention.gqla.latent_rank`: latent rank for `gqla_attn` (default `hidden_size // 2`).
- `model.attention.gqla.num_groups`: number of GQA groups for the expanded path (default `num_heads // 4`).
- `model.attention.gqla.decode_path`: `"mqa_absorb"` or `"gqa"` (default `"gqa"`).
- `model.attention.mlra.latent_rank`: total latent rank for `mlra_attn` (default `hidden_size // 2`).
- `model.attention.mlra.num_latent_heads`: number of disjoint latent sub-heads (default 4).
- `model.attention.tucker.query_rank`, `model.attention.tucker.key_rank`, `model.attention.tucker.value_rank`: Tucker ranks (default `hidden_size` for Q, `hidden_size // 2` for K/V).
- `model.attention.iha.num_pseudo_heads`: number of pseudo-heads per head for `iha_attn` (default `num_heads`).
- `model.attention.gta.num_shared_groups`: number of head groups sharing an attention map for `gta_attn` (default `num_heads // 4`).
- `model.attention.gta.value_latent_rank`: value-cache latent rank for `gta_attn` (default `hidden_size // 2`).
- `model.attention.mtla.latent_rank`: feature-axis latent rank for `mtla_attn` (default `hidden_size // 2`).
- `model.attention.mtla.merge_factor`: consecutive KV entries merged per temporal slot (default 2).
- `model.attention.mtla.stride`: stride between merged slots (default `mtla_merge_factor`).
- `model.attention.cca.latent_rank`: latent width ẽ for `cca_attn` (default `hidden_size // 4`, i.e. C=4).
- `model.attention.cca.num_conv_layers`: 0, 1, or 2 convolution layers for `cca_attn` (default 2).
- `model.attention.cca.conv_kernel_seq`: depth-wise causal seq conv kernel for `cca_attn` (default 4).
- `model.attention.cca.conv_kernel_ch`: head-wise grouped channel conv kernel for `cca_attn` (default 3).
- `model.attention.cca.qk_mean`: enable q-k-mean bias for `cca_attn` (default true).
- `model.attention.cca.value_shift`: enable value-shift for `cca_attn` (default true; requires even `num_heads`).
- `model.attention.ccgqa.query_latent_rank`: query latent width E/C₁ for `ccgqa_attn` (default `hidden_size // 2`).
- `model.attention.ccgqa.kv_latent_rank`: KV latent width E/C₂ for `ccgqa_attn` (default `hidden_size // 8`).
- `model.attention.ccgqa.num_kv_heads`: number of KV group heads for `ccgqa_attn` (default `num_heads // 4`).
- `model.attention.ccgqa.num_conv_layers`: 0, 1, or 2 conv layers for `ccgqa_attn` (default 2).
- `model.attention.ccgqa.conv_kernel_seq`: seq conv kernel for `ccgqa_attn` (default 4).
- `model.attention.ccgqa.conv_kernel_ch`: channel conv kernel for `ccgqa_attn` (default 3).
- `model.attention.ccgqa.qk_mean`: enable q-k-mean with B_group/E_group for `ccgqa_attn` (default true).
- `model.attention.ccgqa.value_shift`: enable value-shift for `ccgqa_attn` (default true).

### MiniMax Sparse Attention & SparDA (model.attention.msa / model.attention.sparda)

- `model.attention.msa.block_size`: KV block size for `msa_attn` (default 128).
- `model.attention.msa.topk_blocks`: blocks selected per GQA group per query (default 16).
- `model.attention.msa.index_dim`: Index Branch head dim (default 64).
- `model.attention.msa.kl_loss_weight`: KL alignment loss weight (default 0.0).
- `model.attention.sparda.block_size`: KV block size for `sparda_attn` (default 128).
- `model.attention.sparda.topk_blocks`: blocks selected per GQA group (default 16).
- `model.attention.sparda.forecast_dim`: Forecast projection dim (default 64).

## Decoder Examples (Famous Architecture-Inspired)

Autoregressive decoder examples are available in `configs/examples/`:

- `decoder_gpt_like.yaml` — GPT-style stacked causal transformer blocks.
- `decoder_llama_like.yaml` — LLaMA-inspired causal stack (`silu`, `rope`).
- `decoder_retnet_mamba_like.yaml` — hybrid RetNet + Mamba causal decoder.

You can also run the named decoder preset directly:

```yaml
model_class: frankesteindecoder
# file: configs/frankesteindecoder.yaml
```

### training (TrainingConfig + runtime)
- `task`: `mlm` or `sbert` (required).
  - `sbert` task requires `base_model`.
- `batch_size`: dataloader batch size.
- `dataloader_workers`: dataloader workers.
- `max_length`: max sequence length.
- `mlm_probability`: MLM mask probability.
- `max_samples`: sample limit.
- `dataset_batch_size`: internal streaming dataset batch size.
- `num_workers`: internal streaming dataset workers.
- `cache_dir`: cache folder.
- `local_parquet_dir`: local parquet path (optional).
- `prefer_local_cache`: prefer local cache.
- `stream_local_parquet`: stream from local parquet.
- `join_temp_data_context_window`: if >0, joins cached `temp_data` chunks into this context size.
- `join_temp_data_min_remainder_tokens`: keep final joined remainder only if content tokens >= this value.
- `use_amp`: mixed precision.
- `gradient_accumulation_steps`: gradient accumulation steps.
- `optimizer`: optimizer chooser object with `optimizer_class` and prefixed `parameters`.
- `scheduler_total_steps`: total steps for the scheduler.
- `scheduler_warmup_ratio`: warmup fraction of total steps.
- `scheduler_type`: `cosine`, `constant`, or `linear_warmup_then_constant`.
- `grad_clip_max_norm`: gradient clipping max norm.
- `inf_post_clip_threshold`: threshold for exploding gradients after clipping.
- `max_nan_retries`: max retries before stopping on NaN.
- `checkpoint_every_n_steps`: checkpoint frequency.
- `max_rolling_checkpoints`: rolling checkpoints to keep.
- `num_best_checkpoints`: best checkpoints to keep.
- `nan_check_interval`: NaN/Inf check interval.
- `log_gradient_stats`: enable gradient logging.
- `gradient_log_interval`: gradient logging interval.
- `csv_log_path`: CSV path for training metrics.
- `csv_rotate_on_schema_change`: rotate old CSV when header schema changes.
- `gpu_metrics_backend`: GPU telemetry backend (`nvml` or `none`).
- `nvml_device_index`: GPU index for NVML metrics.
- `enable_block_grad_norms`: include fixed per-block grad norm columns.
- `telemetry_log_interval`: interval in optimizer steps for heavy telemetry fields.
- `gpu_temp_guard_enabled`: enable strict thermal guard for CUDA training.
- `switch_on_thermal`: when `true`, critical thermal events move training to CPU and resume on GPU after cooldown.
- `gpu_temp_pause_threshold_c`: pause when GPU temp exceeds this threshold.
- `gpu_temp_resume_threshold_c`: resume when GPU temp is at/below this threshold.
- `gpu_temp_critical_threshold_c`: optional critical threshold used by `switch_on_thermal` for GPU->CPU fallback.
- `gpu_temp_poll_interval_seconds`: cooldown polling interval while paused.
- `use_galore`: enable GaLore.
- `galore_rank`: low-rank projection rank.
- `galore_update_interval`: projection update frequency.
- `galore_scale`: projected gradient scale.
- `galore_max_dim`: max tensor size for GaLore.
- `sbert`: block for SBERT-specific settings when `task: sbert`.

## SBERT Dataset Types

`training.sbert.dataset_type` controls how rows are converted and which objective/evaluator is used:

- `paired_similarity`:
- Required columns: `sentence1`, `sentence2`, `similarity` (defaults in runtime: `sentence1`, `sentence2`, `score`).
- Loss: `CosineSimilarityLoss`.
- Evaluator: `EmbeddingSimilarityEvaluator`.
- Resampling: supported (`resample_balanced` and `resample_std`).
- Score standardization: optional via `standardize_scores: true` (keeps all rows, rescales score mean/std).
- Rolling checkpoints: configurable via `checkpoint_save_steps` (default: `1000`, set `0` to disable).
- Resume support: `resume_from_checkpoint: true` resumes from latest checkpoint inside `output_dir/checkpoints`.

- `triplets`:
- Required columns: `query`, `positive`, `negatives`.
- `negatives` must be a list of strings; malformed rows are skipped with warning.
- Loss: `MultipleNegativesRankingLoss` (using explicit negatives).
- Evaluator: disabled by default.
- Resampling: auto-disabled with warning.

- `qa`:
- Required columns: `question`, `answer`.
- Loss: `MultipleNegativesRankingLoss`.
- Evaluator: disabled by default.
- Resampling: auto-disabled with warning.

`training.sbert.columns` is optional and lets you rename dataset columns without code changes.
`training.sbert.query_prefix` and `training.sbert.document_prefix` are optional text prefixes for asymmetric query/document style training.

### Example: Paired Similarity

```yaml
training:
  task: sbert
  sbert:
    dataset_name: "erickfmm/agentlans__multilingual-sentences__paired_10_sts"
    dataset_type: paired_similarity
    columns:
      sentence1: sentence1
      sentence2: sentence2
      similarity: similarity
```

### Example: Triplets (`Q`, `POS`, `NEGs`)

```yaml
training:
  task: sbert
  sbert:
    dataset_name: "your-org/triplets-dataset"
    dataset_type: triplets
    columns:
      query: Q
      positive: POS
      negatives: NEGs
    query_prefix: "query: "
    document_prefix: "passage: "
```

### Example: QA (`Question`, `Answer`)

```yaml
training:
  task: sbert
  sbert:
    dataset_name: "your-org/qa-pairs"
    dataset_type: qa
    columns:
      question: Question
      answer: Answer
    query_prefix: "query: "
    document_prefix: "answer: "
```

## Migration Note

This is a hard migration.
- `training.task` is required.
- Legacy top-level optimizer keys (`lr_*`, `wd_*`, `betas_*`, `eps_*`) are no longer accepted in `training`.

## Notes

- `ffn_hidden_size` is required to match BERT/TinyBERT/EmbBERT.
- `embedding_conv_kernel` allows replicating EmbBERT with kernel 32.
- `standard_attn` and `sigmoid_attn` are compatible with any positional-encoding setting because they do not apply HoPE/RoPE.
- Full GPU power/utilization/memory telemetry uses `pynvml` when available; non-guard telemetry may fall back to zeros.
- When `gpu_temp_guard_enabled: true`, temperature reads are strict (`NVML` then `nvidia-smi`), and telemetry failure stops training immediately.
- When `switch_on_thermal: true`, critical temperature can switch training GPU->CPU and return CPU->GPU at resume temperature; CUDA/NVIDIA runtime failures may force CPU-only mode.
