# Training Configurations (YAML)

> **Component schema entries are generated.** Optimizer enum entries, the
> `by_optimizer` suffix reference and the `^<optimizer>-` schema rules come
> from each optimizer's `OptimizerSpec` (`src/registry/builtin/optimizer.py`);
> mixer/norm/activation/residual/PE/embedding enumerations are likewise
> registry-derived. Adding a component never means editing the schema enum
> or the flattener (there is no flattener: the runtime config is nested).


This directory contains YAML presets. Each file describes the model (`model`) and training/dataset parameters (`training`).

Keys that are absent from `src/schema/**` (for example a field added to a config before the schema is updated) are logged as `WARNING [schema] … unknown key(s) ignored` and dropped at load time. The presets shipped here are still validated strictly by `tests/test_schema_consistency.py`; never rely on an undocumented key being honoured.

## Files

- `frankenstein_text_encoder.yaml`
- `mini.yaml`
- `frankenstein_text_decoder.yaml`
- `standard.yaml`
- `standard_hope.yaml`
- `tinybert.yaml`
- `embbert.yaml`
- `modernbert_continual.yaml`
- `modernbert_sbert.yaml`
- `schema.yaml` (JSON Schema in YAML form)

## General Structure

```yaml
# Option A: train custom Frankenstein model
model_class: frankenstein_text_encoder|frankenstein_text_decoder
model: # FrankensteinModelConfig
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
- `sgd`
- `sgd_momentum`
- `rmsprop`
- `adamw`
- `adam_pm`
- `adafactor`
- `galore_adamw`
- `prodigy`
- `lion`
- `sophia`
- `muon`
- `turbo_muon`
- `af_muon`
- `ada_muon`
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
- `anon`
- `apollo`
- `apollo_mini`
- `q_apollo`
- `dion3`
- `eve`
- `ea4llm`
- `root`
- `muon2`
- `nor_muon`
- `conda`
- `tuon`
- `namo`
- `namo_d`

`training.optimizer.parameters` must use the selected optimizer prefix. Example: when `optimizer_class: adamw`, all keys must start with `adamw-`.

`eve` and `ea4llm` require `gradient_accumulation_steps: 1` and are not available for SBERT (closure-based step: the trainer re-evaluates the current batch on several candidate points inside `optimizer.step()`).

### Optimizer Parameters Reference

All optimizers support these shared per-group parameter suffixes (use with optimizer prefix, e.g. `adamw-lr_embeddings`). Groups match the canonical block taxonomy used by the parameter estimator and the `grad_norm_<block>` training logs:

- `lr_embeddings`, `lr_pos`, `lr_attention`, `lr_ffn`, `lr_norms`, `lr_head`, `lr_mhc`, `lr_router`, `lr_residual`, `lr_other`
- `wd_embeddings`, `wd_pos`, `wd_attention`, `wd_ffn`, `wd_norms`, `wd_head`, `wd_mhc`, `wd_router`, `wd_residual`, `wd_other`
- `betas_embeddings`, `betas_pos`, `betas_attention`, `betas_ffn`, `betas_norms`, `betas_head`, `betas_mhc`, `betas_router`, `betas_residual`, `betas_other`
- `eps_embeddings`, `eps_pos`, `eps_attention`, `eps_ffn`, `eps_norms`, `eps_head`, `eps_mhc`, `eps_router`, `eps_residual`, `eps_other`

Block routing (`embeddings` = token/patch tables, `pos` = learned positional
encodings and cls/mask tokens, `attention` = mixers and their internal norms,
`ffn` = feed-forward bodies and MoE experts, `norms` = standalone block norms,
`head` = task heads, `mhc` = mHC/Hyperloop, `router` = MoE/MoD routers,
`residual` = AttnRes, `other` = everything else).

Optimizer-specific global parameter suffixes (also prefixed):

- `sgd_momentum`: `momentum`, `nesterov`
- `sgd`: none (zero optimizer state; only the shared `lr_*`/`wd_*` apply, with `betas_*`/`eps_*` accepted but unused)
- `rmsprop`: `alpha` (EMA decay of the squared-gradient running average — the slides' 0.9/0.1 split, default 0.9; the shared `eps_*` is the denominator guard and `wd_*` is L2-coupled into the gradient)
- `adamw`: none
- `adam_pm`: `group_size`, `pool_dim` (group-mean pooling of adjacent second-moment entries along rows or columns; second-moment state 1/s; `s=1` → vanilla Adam; first moment untouched; shared `lr_*`/`wd_*`/`betas_*`/`eps_*` per group)
- `adafactor`: `beta2_decay`, `clip_threshold`, `eps1`, `eps2`
- `galore_adamw`: `rank`, `update_proj_gap`
- `prodigy`: `d_coef`
- `lion`: none
- `sophia`: `rho`, `update_k`
- `muon`: `momentum`, `nesterov`, `ns_steps`, `ns_eps`, `per_head`, `per_head_heads`, `clip_value` (orthogonalizes hidden-layer 2-D matrices only; `per_head: true` + `per_head_heads` splits Q/K/V (and latent `uk_proj`/`uv_proj`) attention projections per head, and `clip_value` applies K2-style element-wise weight clipping to the orthogonalized matrices; `embeddings`, `pos`, `head` and non-2-D params use a decoupled AdamW fallback driven by the shared `lr_*`, `wd_*`, `betas_*`, `eps_*` keys)
- `turbo_muon`: `momentum`, `nesterov`, `ns_steps`, `ns_eps` (same hybrid routing as `muon`)
- `af_muon`: `momentum`, `nesterov`, `ns_steps`, `ns_eps`, `vector_lr`, `tied_cap`, `tied_scale`, `rho_hidden`, `rho_output`, `chunk_rows`, `bisection_steps`, `max_bracket_steps`
- `ada_muon`: `momentum`, `ns_steps`, `ns_eps`
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
- `anon`: `gamma`
- `apollo`: `rank`, `update_proj_gap`, `scale`, `scale_type`, `proj_type`, `scale_front`, `disable_nl`
- `apollo_mini`: `update_proj_gap`, `scale`, `proj_type`, `scale_front`, `disable_nl`
- `q_apollo`: `rank`, `update_proj_gap`, `scale`, `scale_type`, `proj_type`, `scale_front`, `disable_nl`, `quant_bits`
- `dion3`: `fraction`, `momentum`, `ns_steps`, `ns_eps`, `normuon`
- `eve`: `de_weight`, `de_jitter`, `init_sigma`, `adam_steps`, `use_pm`, `pm_eta`, `seed` (population-of-4 differential evolution with a targeted Adam fallback; the shared `lr_*`/`wd_*`/`betas_*`/`eps_*` configure the fallback per group, with `wd` folded into the fallback gradient as L2; requires `gradient_accumulation_steps: 1`, not available for SBERT)
- `ea4llm`: `population`, `sigma`, `antithetic`, `rank_shaping`, `nes_scaling`, `seed` (gradient-free evolution strategies; the shared `lr_*` is the per-group step size and `wd_*` a decoupled decay extension, `betas_*`/`eps_*` accepted but unused; N forward evaluations per step; requires `gradient_accumulation_steps: 1`, not available for SBERT)
- `root`: `momentum`, `nesterov`, `ns_steps`, `ns_eps`, `quantile`, `softthresh`, `ada_newton` (same hybrid routing as `muon`, with per-shape calibrated Newton-Schulz coefficients and a dynamic soft-threshold `epsilon_t = Quantile(|M_t|, p)` on the momentum; `quantile` defaults to 0.90)
- `muon2`: `ns_steps`, `ns_eps`, `factored` (Adam-style second-moment preconditioning of the momentum before Newton-Schulz; β₁/β₂ come from the shared `betas_*`, the preconditioner epsilon from the shared `eps_*`; update scaled by η√(m/n); same hybrid routing as `muon`; `factored: true` switches to the memory-efficient Muon²-F variant)
- `nor_muon`: `momentum`, `ns_steps`, `ns_eps` (neuron-wise second-moment normalization after Newton-Schulz; β2 from the shared `betas_*`, ε from the shared `eps_*`; update rescaled to Adam's RMS via η̂ = 0.2η√(mn)/‖O_t‖_F; same hybrid routing as `muon`)
- `conda`: `update_proj_gap`, `scale`, `correct_bias` (Adam in the SVD subspace of the first moment; the second moment is an EMA of the squared projected gradients `(P^T G)^2` with β from the shared `betas_*` and ε from the shared `eps_*`; the SVD basis is refreshed every `update_proj_gap` steps — default 2000; the paper's pretraining uses `scale: 0.25` and `betas: [0.9, 0.99]`; every 2-D parameter takes the Conda path, and non-2-D parameters use the decoupled AdamW fallback)
- `tuon`: `momentum`, `ns_steps`, `ns_eps`, `refresh_interval`, `rms_eps`, `transform`, `stack` (tensor-aware Muon: same-shape hidden matrices per group are stacked, mixed with a unitary layer transform — `decorq` eigendecomposes the layer Gram matrix refreshed every `refresh_interval` steps, `dct`/`haar`/`dft` are fixed — slice-polarized with Newton-Schulz, inverse-transformed and rescaled to RMS 1/√max(m,n) via `rms_eps`; `stack: full|pairs`; K=1 reduces to `muon`; the shared `betas_*`/`eps_*` configure the AdamW fallback and `wd_*` stays decoupled)
- `namo`: `betas`, `ns_steps`, `ns_eps` (norm-based Adam-type adaptive stepsize α_t on the orthogonalized momentum: `betas` is (μ₁, μ₂) of the momentum and the scalar norm EMA — default [0.95, 0.99], the paper's GPT-2 values, distinct from the shared per-group `betas_*` that configure only the AdamW fallback; the shared `eps_*` is α_t's denominator epsilon; the shared `wd_*` is decoupled and scaled by α_t; same hybrid routing as `muon`)
- `namo_d`: `betas`, `ns_steps`, `ns_eps`, `clamp` (diagonal extension of `namo`: a per-column second-moment EMA gives neuron-wise stepsizes that right-multiply the orthogonalized momentum; `clamp` is the bound parameter c ∈ (0,1] — default 0.1, the paper's GPT-2 124M value — keeping the diagonal well-conditioned via `d̃ = clamp(d, c·d̄, d̄/c)`; the shared `eps_*`/`betas_*` configure the AdamW fallback and `wd_*` stays decoupled but is right-multiplied by the diagonal)

Examples:
- `adafactor-beta2_decay: 0.8`
- `galore_adamw-rank: 128`
- `sophia-update_k: 10`
- `muon-ns_steps: 5`
- `af_muon-tied_cap: 3.0`
- `ada_muon-ns_steps: 5`
- `dion3-fraction: 0.25`
- `eve-adam_steps: 3`
- `ea4llm-population: 30`
- `ea4llm-sigma: 1.0e-3`
- `root-quantile: 0.90`
- `muon2-ns_steps: 3`
- `nor_muon-momentum: 0.95`
- `conda-update_proj_gap: 2000`
- `conda-scale: 0.25`
- `tuon-refresh_interval: 200`
- `tuon-transform: decorq`
- `adam_pm-group_size: 8`
- `adam_pm-pool_dim: col`
- `namo-betas: [0.95, 0.99]`
- `namo-ns_steps: 5`
- `namo_d-clamp: 0.1`

## Available Fields (Detailed)

### model_class
- `frankenstein_text_encoder` or `frankenstein_text_decoder`.
- `frankenstein_text_decoder` builds the autoregressive causal decoder variant (`FrankensteinDecoder`).
- Ignored when `base_model` is set.

### base_model
- Optional HF model id/local path.
- When set, `model_class` and `model` can be omitted and ignored by runtime.

### tokenizer
- Required for `training.task: mlm` when `base_model` is set.
- Uses `AutoTokenizer` via `tokenizer.name_or_path`.

### model (FrankensteinModelConfig)

The `model` block is fully hierarchical: `dims`, `norm`, `embedding`,
`positional_encoding`/`positional_encoding_parameters`, `attention`, `moe`,
`bitnet`, `mixture_of_depths`, `ssmax`, `numerics`, `ffn`, `classification_head`,
`mhc`, and `residuals`. Flat `model.*` feature keys are no longer accepted.

#### model.dims

- `vocab_size`: vocabulary size.
- `hidden_size`: hidden dimension.
- `num_layers`: physical layers.
- `num_loops`: logical loops.
- `num_heads`: attention heads.
- `num_kv_heads`: key-value heads for GQA (also used by `msa`/`sparda`/`lsa`).
- `dropout`: global dropout.
- `layer_pattern`: list of blocks.
- `mode`: `encoder` or `decoder` (if omitted, defaults to encoder unless `model_class: frankenstein_text_decoder`, which forces decoder).
- Legacy blocks: `retnet`, `mamba`, `ode`, `standard`, `sigmoid`, `xsa` (Exclusive Self Attention, arXiv:2603.09078).
- Sparse blocks: `sparse_transformer`, `longformer`, `bigbird`, `sparsek`, `nsa`, `sparge`, `fasa`, `msa` (MiniMax Sparse Attention, arXiv:2606.13392), `sparda` (SparDA, arXiv:2606.04511), `lsa` (Latent Sparse Attention, arXiv:2610.42147).
- Gated blocks: `gla`, `deltanet`, `gated_deltanet`, `gated_deltanet2`, `hgrn2`, `fox`, `fpa` (First Passage Attention, arXiv:2610.22714), `gated_softmax`, `kda` (Kimi Delta Attention, arXiv:2510.26692).
- Latent blocks (KV-compression / head-mixing, generalising GQA):
  - `mla` — Multi-Head Latent Attention + RoPE (arXiv:2506.09342).
  - `gmla` — Gated MLA: DeepSeek-style latent attention with q-LoRA, NoPE/rope split and a full-rank output gate (arXiv:2607.24653).
  - `gqla` — Group-Query Latent Attention, two decoding paths (arXiv:2605.15250).
  - `mlra` — Multi-Head Low-Rank Attention, partitionable latent (arXiv:2603.02188).
  - `tucker` — Tucker Attention, generalised low-rank factorisation (arXiv:2603.30033).
  - `iha` — Interleaved Head Attention, cross-head pseudo-heads (arXiv:2602.21371).
  - `gta` — Grouped-head laTenT Attention, shared map + latent values (arXiv:2506.17286).
  - `mtla` — Multi-head Temporal Latent Attention, temporal KV merge (arXiv:2505.13544).
  - `cca` — Compressed Convolutional Attention, attention in compressed latent + convs (arXiv:2510.04476).
  - `ccgqa` — Compressed Convolutional Grouped Query Attention, CCA + decoupled GQA in latent (arXiv:2510.04476).
- Convolution-augmented: `conformer` — Conformer, macaron FFN₁ + MHSA + convolution module (arXiv:2005.08100); `fastconformer` — Fast Conformer limited-context attention, local window + head-pinned global token (arXiv:2305.05084).
- Memory blocks: `titan` — Titans attention window (the neural-memory module is not implemented, arXiv:2501.00663); `engram` — conditional N-gram memory lookup (arXiv:2601.07372); `memory` — Memory Attention, token-ID memory values `V = K + Norm(E[s])` replacing the value projection (arXiv:2609.28399).
- Training-free policy: `sparge` and `fasa` are eval-only; training mode raises an explicit runtime error.

#### model.norm

- `type`: `layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`, `flash_norm`.
- `partial_ratio`: fraction of dims for pRMSNorm (default 0.0625).
- `flashnorm_partial_ratio`: partial-RMS composition for FlashNorm (default 0.0).
- `qk_norm.enabled`: enable query–key attention-logit stabilization (default `false`).
- `qk_norm.method`: one of `rms_norm` (default), `layer_norm`, `l2_norm` — forward-time
  per-head QK normalization (arXiv:2010.04245, arXiv:2606.16310); `qk_clip` —
  per-head post-step weight rescaling with `gamma_h = min(1, tau/S_max^h)`
  (arXiv:2507.20534); or `quack` — parameter-dependent query/key learning rates
  (arXiv:2511.21377).
- `qk_norm.clip_tau`: QK-Clip logit threshold (default 100.0; Kimi K2).
- `qk_norm.clip_alpha`: QK-Clip query/key balancing exponent (default 0.5).
- `qk_norm.quack_tau`: QuacK initial relative learning rate (default 1.0).
- `qk_norm.eps`: RMS/L2 stabilizer for the forward methods (default 1e-6).

  QK stabilization requires every mixer in `dims.layer_pattern` to be one of
  `standard`, `sigmoid`, `gqa`, `mla`, `swa`,
  `longformer`, `bigbird`, `sparse_transformer`, `gta`,
  `iha`, `mtla`, `gma`, `nsa`, `sparsek`, `msa`,
  `sparda`, `lsa`, `fastconformer`, `conformer`, `fox`, `fpa`, `gated_softmax`, `xsa`,
  `titan`, `gqla`, `mlra`, `memory`; `qk_clip` additionally excludes
  `gma`. Recurrent/state-space mixers (`retnet`, `gla`, `mamba`, …)
  are rejected at load time. Examples: `norm_qknorm_rms.yaml`,
  `norm_qknorm_clip_mla.yaml`, `norm_qknorm_quack.yaml`.

#### model.embedding

- `factorized.enabled`: enable factorized embeddings.
- `factorized.dim`: reduced embedding dimension.
- `dynamic.enabled`: EELBERT dynamic embeddings (arXiv:2310.20144) — replaces the `vocab_size × hidden_size` lookup table with an on-the-fly n-gram hash of the token strings. Mutually exclusive with `factorized.enabled`; requires a tokenizer (the pipeline injects its vocabulary).
- `dynamic.ngram_max`: maximum n-gram size (default 3).
- `dynamic.bucket`: hash bucket size (default `1000000007`).
- `dynamic.seed`: hash-seed random state (default 42); changing it invalidates checkpoints trained with the previous seed.
- `dynamic.precompute`: memoise the computed table in a non-persistent buffer (default `true`; never saved or exported).
- `conv.enabled`: optional Conv1d on embeddings.
- `conv.kernel`: Conv1d kernel size.

#### model.positional_encoding / model.positional_encoding_parameters

- `positional_encoding`: model-wide PE enum (`rope`, `hope`, `nope`, `alibi`, `bam`, `pape`, `pape_efficient`, `pape_ri`, `sinusoidal_absolute`, `sinusoidal_rotary`, `learned_absolute`, `none`).
- `positional_encoding_parameters.rope`: `base`, `scaling`.
- `positional_encoding_parameters.hope`: `base`, `damping`.
- `positional_encoding_parameters.alibi`: `num_heads` (null = `dims.num_heads`).
- `positional_encoding_parameters.bam`: `learn_mu`, `theta_init`, `mu_init`, `eps`.
- `positional_encoding_parameters.pape`: `num_parabolas`, `num_positions`, `rotation_invariant`.
- `positional_encoding_parameters.sinusoidal`: `max_len`, `base`, `scale`.
- `positional_encoding_parameters.learned`: `max_len`, `init_std`.
- `positional_encoding_parameters.use_pe.<mixer>_use_pe`: per-mixer opt-out.

#### model.attention.<mixer>

Per-mixer sub-objects: `ca`, `kda`, `mla`, `gmla`, `gqla`, `mlra`, `tucker`, `iha`, `gta`, `mtla`,
`cca`, `ccgqa`, `msa`, `sparda`, `lsa`, `fastconformer`, `engram`, `gma`, `ssog`, `conformer`,
`falcon`, `ode` (`solver`, `steps`), `retnet` (`heads`). Optional rank fields
accept `null` (= derive from `hidden_size` / `num_heads`).

#### model.moe / model.bitnet / model.mixture_of_depths / model.ssmax / model.numerics

- `moe`: router, routing, balancing, experts and layer layout. Core keys:
  `enabled`, `num_experts`, `top_k`, `expert_hidden_size`; `router.scoring`
  (`softmax`/`sigmoid`/`softplus_sqrt`), `pre_softmax`, `renormalize`,
  `routed_scaling_factor`, `dtype`, `jitter_eps`, `z_loss_coeff`;
  `routing.type` (`token_choice`/`expert_choice`), `capacity_factor`,
  `hash_layers`, `hash_table`, `hash_seed`; `balancing.aux_loss`
  (`none`/`batch`/`sequence`), `aux_loss_coeff`, `bias_update`
  (`none`/`sign`/`centered_sign`), `bias_update_rate`; `shared.count`,
  `shared.hidden_size`, `shared.gate`; `latent.enabled`, `latent.dim`,
  `latent.norm_before_up_proj`; `layout.dense_layers`,
  `layout.every_n_layers`; `experts.activation_clamp`. The named MoE presets
  (`deepseek_v3`, `kimi_k2`, `kimi_k3`, `glm_5`, `gpt_oss_120b`, `llama4_*`,
  `qwen3_*`, `minimax_*`) mirror the published architectures; see
  `docs/paper/appendices/annex-22-moe.tex` for the per-model table and the
  documented single-device deviations.
- `bitnet`: `enabled`, `routers`, `conv`. When `enabled`, all primary + gate
  projections (Q/K/V/O, FFN, embeddings `proj`, LM head, and recurrent-state
  gates) become ternary `{-1,0,1}`. `routers` also quantizes routing/scoring
  projections (default `false` for routing stability). `conv` also quantizes
  the embedding Conv1d (requires `model.embedding.conv.enabled`).
- `mixture_of_depths`: `enabled`, `capacity_ratio`, `router_aux_loss_weight`.
- `ssmax`: `enabled`, `s_init` (Scalable Softmax, arXiv:2505.22842).
- `numerics`: `fixed_softmax`, `fixed_cross_entropy` (default `false`).
  Cancellation-free softmax/log-softmax and zero-sum-gradient hard-target
  cross-entropy from arXiv:2610.31642 ("A Fix for Softmax and Cross-Entropy").
  Forward values are mathematically identical to the standard operations; the
  flags only change the backward-pass numerics past the precision cliff.
  Example: `configs/examples/numerics_fixed.yaml`.

#### model.ffn

- `hidden_size`: FFN intermediate dimension (null = 2× `dims.hidden_size`).
- `activation`: `silu`, `gelu`, or any of the 44 supported activations.
- `activation_config`: nested object for learnable/parametric activations.

#### model.classification_head

- `enabled`: enable the NLP sequence-classification head (DashAI).
- `num_labels`: number of classes.
- `pooling_mode`: `cls` or `gap`.

#### model.mhc / model.residuals

- `mhc`: `enabled`, `expansion_rate`, `sinkhorn_iters`, `gating_init`,
  `checkpoint`, `full_prec_under_bitnet`, `hyperloop*`.
- `residuals`: `type`, `full_attn.*`, `block_attn.*`, `mhc_stream_mode`,
  `gradient_checkpoint`.

### Continuum Attention (model.attention.ca)

- `model.attention.ca.learned_gain`: learn the per-head positive score gain
  `γ = exp(ρ)` (default `true`, the paper's 101M flagship; each layer adds one
  scalar per head). `false` keeps the gain fixed and is parameter-identical to
  `standard` (the paper's CA-P configuration).
- `model.attention.ca.gain_init`: initial (learned) or fixed score gain
  `γ = 1/τ` (default `1.0`; CA-P uses `3.3333` ≈ 1/0.30).
- `positional_encoding_parameters.use_pe.ca_use_pe`: per-mixer PE opt-out (default
  `true`; the paper uses RoPE / learned absolute positions).
- Exact segment integration over piecewise-linear key/value trajectories
  (arXiv:2610.20586); classic KV cache. Examples:
  `configs/examples/es_arch_ca_adamw.yaml` and the 101M reference preset
  `configs/ca.yaml`.

### First Passage Attention (model.attention.fpa)

- `model.attention.fpa.beta_init`: initial value of the learned per-head
  first-passage strength `β` (default `1.0`, the paper's initialization). `β`
  always stays learnable and multiplies the deepest-dip depth in
  `b_ts = β·min(0, min_{s<u≤t} φ_u − ψ_s)`.
- `model.attention.fpa.tau`: temperature `τ` of the soft-minimum
  **straight-through gradient** (default `1.0`). The forward pass and
  evaluation always use the hard range minimum; `τ` only scales the
  differentiable surrogate.
- `positional_encoding_parameters.use_pe.fpa_use_pe`: per-mixer PE opt-out (default
  `false`; FPA trains without positional encoding — the learned walk is
  positional).
- Softmax attention with a range-minimum (path-extremum) forget bias
  (arXiv:2610.22714); contains FoX as the non-rising-walk special case. The
  bias adds `3·nH·(H+1) + nH` parameters per layer and the cache stores two
  extra scalars per key and head. Examples:
  `configs/examples/es_arch_fpa_adamw.yaml` and the 110M reference preset
  `configs/fpa.yaml`.

### Kimi Delta Attention (model.attention.kda)

- `model.attention.kda.num_heads`, `model.attention.kda.head_dim`: decouple the head geometry from `dims.num_heads`/`hidden_size` (Kimi K3: 96 x 128).
- `model.attention.kda.decay_mode`: `kimilinear` (default) or `k3` (input-dependent bounded decay, `g_min` = `decay_min_log`).
- `model.attention.kda.decay_min_log`, `decay_low_rank`: bound `g_min` (default -5) and the low-rank width of the decay logits.
- `model.attention.kda.short_conv_kernel`: causal depthwise short conv + SiLU before Q/K/V (default 0 = off; K3: 4).
- `model.attention.kda.use_full_rank_gate`: full-rank `silu(W_g x)` output gate (default true).

### Latent / KV-compression family (model.attention.*)

- `model.attention.gmla`: `latent_rank`, `q_lora_rank`, `num_heads`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `use_nope`, `use_output_gate` (defaults: `hidden_size // 2`, off, `dims.num_heads`, 128, 64, 128, true, true).
- `model.attention.mla.latent_rank`: rank of the joint KV latent for `mla` (default `hidden_size // 2`).
- `model.attention.gqla.latent_rank`: latent rank for `gqla` (default `hidden_size // 2`).
- `model.attention.gqla.num_groups`: number of GQA groups for the expanded path (default `num_heads // 4`).
- `model.attention.gqla.decode_path`: `"mqa_absorb"` or `"gqa"` (default `"gqa"`).
- `model.attention.mlra.latent_rank`: total latent rank for `mlra` (default `hidden_size // 2`).
- `model.attention.mlra.num_latent_heads`: number of disjoint latent sub-heads (default 4).
- `model.attention.tucker.query_rank`, `model.attention.tucker.key_rank`, `model.attention.tucker.value_rank`: Tucker ranks (default `hidden_size` for Q, `hidden_size // 2` for K/V).
- `model.attention.iha.num_pseudo_heads`: number of pseudo-heads per head for `iha` (default `num_heads`).
- `model.attention.gta.num_shared_groups`: number of head groups sharing an attention map for `gta` (default `num_heads // 4`).
- `model.attention.gta.value_latent_rank`: value-cache latent rank for `gta` (default `hidden_size // 2`).
- `model.attention.mtla.latent_rank`: feature-axis latent rank for `mtla` (default `hidden_size // 2`).
- `model.attention.mtla.merge_factor`: consecutive KV entries merged per temporal slot (default 2).
- `model.attention.mtla.stride`: stride between merged slots (default `mtla_merge_factor`).
- `model.attention.cca.latent_rank`: latent width ẽ for `cca` (default `hidden_size // 4`, i.e. C=4).
- `model.attention.cca.num_conv_layers`: 0, 1, or 2 convolution layers for `cca` (default 2).
- `model.attention.cca.conv_kernel_seq`: depth-wise causal seq conv kernel for `cca` (default 4).
- `model.attention.cca.conv_kernel_ch`: head-wise grouped channel conv kernel for `cca` (default 3).
- `model.attention.cca.qk_mean`: enable q-k-mean bias for `cca` (default true).
- `model.attention.cca.value_shift`: enable value-shift for `cca` (default true; requires even `num_heads`).
- `model.attention.ccgqa.query_latent_rank`: query latent width E/C₁ for `ccgqa` (default `hidden_size // 2`).
- `model.attention.ccgqa.kv_latent_rank`: KV latent width E/C₂ for `ccgqa` (default `hidden_size // 8`).
- `model.attention.ccgqa.num_kv_heads`: number of KV group heads for `ccgqa` (default `num_heads // 4`).
- `model.attention.ccgqa.num_conv_layers`: 0, 1, or 2 conv layers for `ccgqa` (default 2).
- `model.attention.ccgqa.conv_kernel_seq`: seq conv kernel for `ccgqa` (default 4).
- `model.attention.ccgqa.conv_kernel_ch`: channel conv kernel for `ccgqa` (default 3).
- `model.attention.ccgqa.qk_mean`: enable q-k-mean with B_group/E_group for `ccgqa` (default true).
- `model.attention.ccgqa.value_shift`: enable value-shift for `ccgqa` (default true).

### MiniMax Sparse Attention & SparDA (model.attention.msa / model.attention.sparda)

- `model.attention.msa.block_size`: KV block size for `msa` (default 128).
- `model.attention.msa.topk_blocks`: blocks selected per GQA group per query (default 16).
- `model.attention.msa.index_dim`: Index Branch head dim (default 64).
- `model.attention.msa.kl_loss_weight`: KL alignment loss weight (default 0.0).
- `model.attention.sparda.block_size`: KV block size for `sparda` (default 128).
- `model.attention.sparda.topk_blocks`: blocks selected per GQA group (default 16).
- `model.attention.sparda.forecast_dim`: Forecast projection dim (default 64).

### Latent Sparse Attention (model.attention.lsa)

- `model.attention.lsa.num_long_range`: long-range keys selected per query $k$ (default 2048).
- `model.attention.lsa.num_buckets`: buckets scored per query $p$; raised to `ceil(k / pool_size)` when smaller, clamped to the cached buckets (default 512).
- `model.attention.lsa.num_sinks`: always-attended leading sink keys (default 4).
- `model.attention.lsa.pool_size`: keys summed per pool key before Ward clustering (default 4).
- `model.attention.lsa.cluster_block`: cluster block size, also the window $n_w$ (default 512).
- `model.attention.lsa.keys_per_bucket`: average keys per bucket; yields `cluster_block / keys_per_bucket` buckets per block and must divide `cluster_block` (default 8).
- `model.attention.lsa.max_keys_per_bucket`: Ward merge cap per bucket, at least `keys_per_bucket` (default 16).

`lsa` uses `num_kv_heads` (the paper's N-LSA uses a single shared KV head),
adds no parameters, and opts out of the model-wide positional encoding by
default (N-LSA trains without PE).

### Fast Conformer limited-context attention (model.attention.fastconformer)

- `model.attention.fastconformer.window_size`: total local span $w$; the per-side context is `w / 2` (default 256, the paper's 128 steps on each side).
- `model.attention.fastconformer.global_tokens`: number of leading global attention tokens $g$; they attend to every position and are attended by every position (default 1, the paper's single global token; `0` is the pure limited-context variant).
- `model.attention.fastconformer.global_tokens_spacing`: stride $s$ between global positions $\{0, s, \dots, (g{-}1)s\}$ (default 1).
- `model.attention.fastconformer.global_attn_separate`: use a separate set of global query/key/value projections (paper §2.2, Longformer-style, adds $3H^2$ parameters) instead of the shared local ones (paper §3.4, the evaluated long-form model; default `false`).

`fastconformer` uses plain multi-head projections (it does not read
`num_kv_heads`), adds no parameters by default, trains its windowed + global
pattern in both training and inference, and caches in a `RingSlot` whose pinned
head prefix covers the global positions (exact incremental decoding).

### SSOG — Separable Sum of Gaussians (model.attention.ssog)

- `model.attention.ssog.num_atoms`: Gaussian atoms per head for `ssog` (default 4; each atom is 5 numbers — μy, μx, σy, σx, λ).
- `model.attention.ssog.lookat`: content-conditioned steering on μ/σ/λ (default true; `false` = fixed content-blind field).
- `model.attention.ssog.max_offset`: bound on per-token μ travel in grid cells (default 4.0; small grids want 1–2).
- `model.attention.ssog.cold_init`: start steering gates at ≈0, frozen geometry at init (default true; warm-start loses ~1 pt).
- `model.attention.ssog.sigma_floor`: minimum atom width in grid cells (default 0.25).
- `model.attention.ssog.grid_h` / `model.attention.ssog.grid_w`: token grid raster (default: derived from `image_height // patch_size` × `image_width // patch_size`; set explicitly for non-vision grids — `grid_h: 1, grid_w: max_len` gives a 1D positional field).
- Constraints: `ssog` is encoder-only (`mode: encoder`), sequence length must equal `grid_h * grid_w`, and `frankenstein_vit` configs must use `image.cls_token: false` (with `pooling_mode: gap`).
- Examples: `configs/examples/vit_ssog_mnist.yaml` (ViT), `configs/examples/es_arch_ssog_adamw.yaml` (1D NLP grid).

### Conformer (model.attention.conformer)

`conformer` (arXiv:2005.08100) owns the macaron FFN₁, multi-head
self-attention and the convolution module with internal pre-norm residuals;
the shared `HybridLayer` norm + FFN plays FFN₂ and the outer residual.
Examples: `configs/examples/es_arch_conformer_adamw.yaml` (8-layer Spanish
MLM preset), `configs/examples/mixer_conformer.yaml` (minimal smoke preset),
`vit_conformer.yaml` (ViT), `audio_asr_conformer.yaml` and
`audio_classification_conformer.yaml` (speech/audio).

- `model.attention.conformer.kernel_size`: depth-wise conv kernel `k` for `conformer` (default 32; paper best of {3, 7, 17, 32, 65}).
- `model.attention.conformer.macaron`: enable the macaron FFN₁ branch (default true).
- `model.attention.conformer.conv_norm`: normalization after the depth-wise conv — `batch` (BatchNorm1d, paper default), `group` (GroupNorm) or `layer` (LayerNorm).
- `positional_encoding_parameters.use_pe.conformer_use_pe`: per-mixer PE opt-out (default true).
- Decoder mode pads the depth-wise conv causally; encoder mode uses symmetric padding.
- `conformer` is not KV-cacheable (depth-wise conv + macaron state); it always runs the legacy full-recompute path.

### Memory Attention (model.attention.memory)

`memory` (arXiv:2609.28399) keeps the standard softmax attention operator
but drops the dedicated value projection: values are the contextual keys plus a
per-layer, token-ID-indexed memory table, `V = K + RMSNorm(E[s])`, with
`d_v = dims.num_kv_heads * (hidden_size / num_heads)`. RoPE applies only to the
scoring pair. There are no per-mixer sub-keys (`dims.num_kv_heads` and
`dims.vocab_size` are the only inputs); `positional_encoding_parameters.use_pe.memory`
(default true) is the only flag. The mixer needs token IDs, so it is text-only
(mask `input_ids`; ViT/audio presets have none). At inference it uses a plain
KV slot, so `inference.kv_cache.enabled: true` works as for `standard`.
Examples: `configs/examples/es_arch_memory_adamw.yaml` (small Spanish MLM
preset) and `configs/memory.yaml` (the paper's L24-D2048 causal-LM
reference).

### Audio presets (`frankenstein_audio`, tasks `asr` / `audio_classification`)

`frankenstein_audio` (arXiv:2005.08100) replaces the token embedding with a
waveform frontend and adds ASR / audio-classification heads. New presets:

- `configs/examples/audio_asr_conformer.yaml` — CTC ASR, 4-layer
  Conformer-heavy encoder, log-mel frontend + SpecAugment, Chilean-jokes HF
  dataset (`astroza/chilean-jokes-festival-de-vina`, split `test`).
- `configs/examples/audio_asr_transducer.yaml` — RNN-Transducer ASR
  (arXiv:1211.3711), same encoder; `training.asr.loss: transducer` overrides
  `audio.asr_loss`.
- `configs/examples/audio_classification_conformer.yaml` — audio
  classification with the synthetic dummy dataset (`dataset_name: ""`).
- `configs/examples/audio_eeg_classification.yaml` — generic multichannel 1-D
  signals: `audio.feature: raw`, `audio.in_channels: 8` (EEG-style).
- `configs/examples/vit_conformer.yaml` — ViT with `conformer` in
  `layer_pattern`.

Top-level blocks:

- `audio.feature`: `logmel` (80-channel log-mel, paper recipe), `spectrogram`
  (log-power STFT) or `raw` (multichannel 1-D signals; skips mel/STFT).
- `audio.sample_rate` / `window_ms` / `stride_ms` / `n_fft` / `n_mels`:
  frontend geometry (defaults 16 kHz, 25 ms, 10 ms, 400, 80).
- `audio.in_channels` / `conv_channels`: raw-signal channels and convolutional
  subsampling width (4× time reduction).
- `audio.pooling_mode` / `num_classes`: classification head (`gap` or learned
  `cls` token).
- `audio.asr_loss`: `ctc` (default) or `transducer`; the transducer also reads
  `asr_decoder_dim` / `asr_decoder_layers` / `asr_decoder_dropout`.
- `audio.spec_augment`: `{enabled, freq_mask_width (27), time_masks (10),
  time_mask_ratio (0.05)}`; applied inside the frontend only while the model is
  in `train()` mode (arXiv:1904.08779).
- `audio_dataset.dataset_name` / `dataset_dir` / `split` / `audio_column` /
  `text_column` / `label_column` / `max_duration_sec` / `resample`. The ASR
  character vocabulary is built from `text_column` at runtime; an empty
  `dataset_name` selects the synthetic dummy dataset for smoke tests.
- `training.asr.loss` (optional): `ctc` | `transducer`; overrides
  `audio.asr_loss` and must match the built head.

### Manifold-Constrained Hyper-Connections (model.mhc)

`model.mhc` configures mHC (arXiv:2512.24880), an opt-in replacement for the
standard residual connection that expands the residual stream width by a factor
`n` and constrains the stream-mixing matrix `H[res]` to the Birkhoff polytope
(doubly stochastic) via Sinkhorn-Knopp. See
[`docs/architecture.md`](../docs/architecture.md#78-residuals-mhc-and-hyperloop-modelresiduals-modelmhc).

- `model.mhc.enabled`: enable mHC (`use_mhc`). Default `false`.
- `model.mhc.expansion_rate`: stream expansion factor `n` (`mhc_expansion_rate`). Default `4`.
- `model.mhc.sinkhorn_iters`: Sinkhorn-Knopp normalisation rounds (`mhc_sinkhorn_iters`). Default `20`.
- `model.mhc.gating_init`: initial value of the gating scalars `α` (`mhc_gating_init`). Default `0.01`.
- `model.mhc.checkpoint`: gradient checkpointing on mHC layers (`mhc_checkpoint`). Default `false`.
- `model.mhc.full_prec_under_bitnet`: keep `φ_l` full-precision under BitNet (`mhc_full_prec_under_bitnet`). Default `true`.

Example: `configs/examples/es_arch_mhc_adamw.yaml`. ⚠️ mHC is incompatible with
`mixture_of_depths.enabled`.

#### Hyperloop mode (arXiv:2604.21254)

`model.mhc.hyperloop: true` switches mHC to the **Hyperloop Transformer**
wiring: loop-level hyper-connections (once per loop iteration, with per-loop
parameters `{W_l, b_l, α_l, e_l}`) over a **middle-cycle** partitioned stack —
begin block (runs once) → looped middle block → end block (runs once). The
stream is created by copying `expansion_rate` times and collapsed by averaging
(no learned in/out projections), and `H^res` uses the paper's diagonal-sigmoid
parameterization by default. The paper matches depth-matched Transformers with
~50% fewer parameters.

- `model.mhc.hyperloop`: enable Hyperloop mode (`mhc_hyperloop`). Default `false`.
  Requires `enabled: true` and `dims.num_loops >= 2`.
- `model.mhc.hyperloop_begin_layers`: begin-block layers (`mhc_hyperloop_begin_layers`).
  Default `0` (loop the whole stack); paper uses ~25% e.g. `2`.
- `model.mhc.hyperloop_end_layers`: end-block layers (`mhc_hyperloop_end_layers`).
  Default `0`; paper uses ~25% e.g. `2`.
- `model.mhc.hyperloop_res_parameterization`: `H^res` mixing
  (`mhc_hyperloop_res_parameterization`): `diagonal` (paper default & best),
  `sinkhorn` (mHC-style doubly stochastic), or `identity`. Default `diagonal`.

Example: `configs/examples/hyperloop_mhc_adamw.yaml`. ⚠️ Incompatible with
`residuals.type: full_attn|block_attn`; unlike per-layer mHC,
`mixture_of_depths.enabled` **stays compatible** (middle-block layers run the
standard C-dim path).

## Decoder Examples (Famous Architecture-Inspired)

Autoregressive decoder examples are available in `configs/examples/`:

- `decoder_gpt_like.yaml` — GPT-style stacked causal transformer blocks.
- `decoder_llama_like.yaml` — LLaMA-inspired causal stack (`silu`, `rope`).
- `decoder_retnet_mamba_like.yaml` — hybrid RetNet + Mamba causal decoder.

You can also run the named decoder preset directly:

```yaml
model_class: frankenstein_text_decoder
# file: configs/frankenstein_text_decoder.yaml
```

### training (TrainingConfig + runtime)
- `task`: `mlm` or `sbert` (required).
  - `sbert` task requires `base_model`.
- `seed`: reproducibility seed (default `42`). Initializes Python/NumPy/PyTorch/CUDA RNGs, model weight init, DataLoader shuffle generators and workers, the held-out test split and the SBERT dataset shuffle; enables deterministic cuDNN.
- `test_percentage`: percentage of the task dataset held out before training and evaluated after the last epoch (`0` disables; written to `test_metrics.json`, with balanced accuracy, macro/weighted F1 and per-class metrics for classification tasks).
- `test_stratify`: stratify the held-out split by class label for `classification`, `text_classification` and `audio_classification` (default `true`); rare classes stay represented on both sides. The exact indices are written to `test_split.json` and reused by the `full_tests` inference evaluation.
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
- `enable_block_grad_norms`: include one `grad_norm_<block>` column per canonical block (`embeddings`, `pos`, `attention`, `ffn`, `norm`, `head`, `mhc`, `router`, `residual`, `other`) and log them to W&B as `grad_norm/<block>` when `wandb_project` is set.
- `wandb_project`: Weights & Biases project name; unset/`null` disables W&B. Requires the optional `wandb` extra (`pip install -e ".[wandb]"`).
- `wandb_mode`: W&B run mode (`online`, `offline`, `disabled`); default `online`.
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

## Inference / KV cache (`inference:`)

The optional top-level `inference:` block is **runtime** configuration
(never part of `FrankensteinModelConfig`). It is validated by
`src/schema/_inference.yaml` and materialised as
`src/deploy/inference_config.InferenceConfig`.

```yaml
inference:
  generation:
    max_new_tokens: 64
    temperature: 0.8
    top_k: 50
    top_p: 1.0
    greedy: false
  kv_cache:
    enabled: true                       # opt-in (default false: legacy path)
    quantization:
      method: int4_kivi                 # none | int8 | int4_kivi
      k_bits: 4
      v_bits: 4
      group_size: 32
      protected_window: 32
    compression:
      method: window_sinks              # none | window_sinks
      num_sinks: 4
      window: 512
      budget: null
      per_layer_budgets: []
    per_mixer: []                       # [{mixer: standard, enabled: false}]
```

Precedence (D2): **CLI flags > YAML `inference:` > schema defaults**.
Every CLI flag defaults to `None`, so only explicitly passed values patch
the preset:

```bash
frankenstein-transformer infer --model artifacts/ --task causal_lm \
  --config configs/examples/kv_cache_generate.yaml \
  --prompt "Hola" --max-new-tokens 64 --greedy \
  --kv-cache --kv-quant int4_kivi --kv-group-size 32 \
  --kv-compress window_sinks --kv-num-sinks 4 --kv-window 512
```

Semantics:
- `kv_cache.enabled: false` (default) keeps generation bit-identical to
  the legacy full-recompute loop.
- Mixers without an implemented incremental branch (documented in
  `src/model/cache/capabilities.py`) silently fall back to the uncached
  path — inference never fails because of the cache (D5). The loader
  prints a warning naming them.
- PaPE-configured layers fall back per layer (content-dependent q/k
  augmentation; caching exactness unproven).
- `compression.method != none` requires `enabled: true`, and
  `compression.window >= quantization.protected_window`.
- `int8` fixes 8 bits (ignores `k_bits`/`v_bits`); `int4_kivi` uses them;
  `none` ignores all quantization sub-keys.

`ModelDeployer` serialises the resolved block into the artifact
`config.json` (`"inference": {...}`) so cached inference is reproducible
without the original YAML.

## Notes

- `ffn_hidden_size` is required to match BERT/TinyBERT/EmbBERT.
- `embedding_conv_kernel` allows replicating EmbBERT with kernel 32.
- `standard` and `sigmoid` are compatible with any positional-encoding setting because they do not apply HoPE/RoPE.
- Full GPU power/utilization/memory telemetry uses `pynvml` when available; non-guard telemetry may fall back to zeros.
- When `gpu_temp_guard_enabled: true`, temperature reads are strict (`NVML` then `nvidia-smi`), and telemetry failure stops training immediately.
- When `switch_on_thermal: true`, critical temperature can switch training GPU->CPU and return CPU->GPU at resume temperature; CUDA/NVIDIA runtime failures may force CPU-only mode.
