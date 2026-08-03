# Spanish Adaptation Examples (P40-Oriented)

## Purpose and Scope
This document provides a curated set of **31 schema-compliant YAML presets** for Spanish-focused adaptation workflows in this repository:

- **25 MLM presets** for continual pretraining (`training.task: mlm`) with full optimizer coverage.
- **5 SBERT presets** for sentence embedding finetuning (`training.task: sbert`).
- **4 decoder presets** (`model_class: frankesteindecoder`) for autoregressive causal language models.
- Coverage includes Spanish-first checkpoints, multilingual fallback models, multiple optimizers, all supported norm types, and all attention block families (legacy, gated, sparse).

All examples are tuned around practical stability on **NVIDIA Tesla P40 (24GB VRAM)**.

## Quick Start Commands
The example presets are stored in `configs/examples/`.

### CLI (terminal)
```bash
# Run MLM example
python -m src.cli train \
  --config configs/examples/es_mlm_beto_adamw.yaml \
  --device cuda \
  --gpu-temp-guard \
  --gpu-temp-pause-threshold-c 90 \
  --gpu-temp-resume-threshold-c 80 \
  --gpu-temp-critical-threshold-c 95 \
  --gpu-temp-poll-interval-seconds 30

# Run custom architecture MLM example
python -m src.cli train \
  --config configs/examples/es_arch_moe_titan_ademamix.yaml \
  --device cuda \
  --gpu-temp-guard \
  --gpu-temp-pause-threshold-c 90 \
  --gpu-temp-resume-threshold-c 80 \
  --gpu-temp-critical-threshold-c 95 \
  --gpu-temp-poll-interval-seconds 30

# Run SBERT finetuning example
python -m src.cli train \
  --config configs/examples/es_sbert_modernbert_mean.yaml \
  --device cuda \
  --gpu-temp-guard \
  --gpu-temp-pause-threshold-c 90 \
  --gpu-temp-resume-threshold-c 80 \
  --gpu-temp-critical-threshold-c 95 \
  --gpu-temp-poll-interval-seconds 30
```

### Python interface
```python
# Option 1: Call the training entrypoint directly
from src.training.main import main as train_main

train_main([
    "--config",
    "configs/examples/es_mlm_beto_adamw.yaml",
    "--device",
    "cuda",
    "--gpu-temp-guard",
    "--gpu-temp-pause-threshold-c",
    "90",
    "--gpu-temp-resume-threshold-c",
    "80",
    "--gpu-temp-critical-threshold-c",
    "95",
    "--gpu-temp-poll-interval-seconds",
    "30",
])

# Option 2: Call the CLI programmatically
from src.cli import main as cli_main

cli_main([
    "train",
    "--config",
    "configs/examples/es_sbert_beto_mean.yaml",
    "--device",
    "cuda",
    "--gpu-temp-guard",
    "--gpu-temp-pause-threshold-c",
    "90",
    "--gpu-temp-resume-threshold-c",
    "80",
    "--gpu-temp-critical-threshold-c",
    "95",
    "--gpu-temp-poll-interval-seconds",
    "30",
])
```

## ArXiv Research Summary and Mapping
| Paper | Key idea | Why relevant for Spanish adaptation on P40 | Mapped YAML presets |
|---|---|---|---|
| [BERT (1810.04805)](https://arxiv.org/abs/1810.04805) | Bidirectional MLM pretraining | Core objective used by all MLM configs | `es_mlm_beto_adamw`, `es_mlm_bertin_lion`, `es_mlm_maria_roberta_adafactor`, all `es_arch_*` |
| [RoBERTa (1907.11692)](https://arxiv.org/abs/1907.11692) | Robust BERT training scaling recipe | Informs stable long-run continual MLM defaults | `es_mlm_maria_roberta_adafactor`, `es_mlm_bertin_lion`, `es_mlm_xlmr_radam` |
| [ALBERT (1909.11942)](https://arxiv.org/abs/1909.11942) | Parameter sharing/factorization for efficiency | Good lower-memory baseline for P40 throughput | `es_mlm_albert_sophia` |
| [DistilBERT (1910.01108)](https://arxiv.org/abs/1910.01108) | Knowledge distillation for compact transformers | Lower VRAM pressure while keeping multilingual coverage | `es_mlm_distilmbert_lamb` |
| [ELECTRA (2003.10555)](https://arxiv.org/abs/2003.10555) | Efficient discriminative pretraining | Inspires efficiency-first optimizer sweeps under fixed VRAM | `es_arch_standard_layer_norm_prodigy`, `es_arch_bitnet_factorized_mars_adamw` |
| [DeBERTa (2006.03654)](https://arxiv.org/abs/2006.03654) | Enhanced attention + disentangled representations | Strong multilingual transfer and robust continual adaptation baselines | `es_mlm_mdeberta_shampoo`, `es_mlm_deberta_soap` |
| [Sentence-BERT (1908.10084)](https://arxiv.org/abs/1908.10084) | Siamese sentence embedding training | Foundation for all `training.task: sbert` presets | `es_sbert_beto_mean`, `es_sbert_bertin_cls`, `es_sbert_maria_max`, `es_sbert_modernbert_mean`, `es_sbert_xlmr_mean` |
| [MiniLM (2002.10957)](https://arxiv.org/abs/2002.10957) | Small model distillation with strong transfer | Guides compact/high-efficiency experiment profiles | `es_mlm_distilmbert_lamb`, `es_arch_mini_cautious_adamw` |
| [XLM-R (1911.02116)](https://arxiv.org/abs/1911.02116) | Large multilingual pretraining | Strong fallback when Spanish-only model coverage is limited | `es_mlm_xlmr_radam`, `es_sbert_xlmr_mean` |
| [LaBSE (2007.01852)](https://arxiv.org/abs/2007.01852) | Language-agnostic sentence embeddings | Supports multilingual retrieval-style SBERT settings | `es_sbert_xlmr_mean`, `es_sbert_modernbert_mean` |
| [BERTIN (2107.07253)](https://arxiv.org/abs/2107.07253) | Spanish RoBERTa model family | Spanish-first continual pretraining and embedding adaptation | `es_mlm_bertin_lion`, `es_sbert_bertin_cls` |
| [MarIA / Spanish LM family (2308.02976)](https://arxiv.org/abs/2308.02976) | Spanish-focused model and adaptation insights | Strong Spanish-domain fit for both MLM and SBERT | `es_mlm_maria_roberta_adafactor`, `es_sbert_maria_max` |
| [ModernBERT (2412.13663)](https://arxiv.org/abs/2412.13663) | Modernized BERT encoder stack | Good high-quality base for continual Spanish adaptation | `es_mlm_modernbert_galore_adamw`, `es_sbert_modernbert_mean` |

## P40 Practical Tuning Guide
### Batch/Sequence/Accumulation Heuristics
- Start with `use_amp: false` for MLM on P40.
- Use `max_length: 512` for lighter encoders and compact custom configs.
- Drop to `max_length: 256` for heavier multilingual or MoE/hybrid settings.
- Keep micro-batch in the `1..8` range and scale effective batch with `gradient_accumulation_steps` (`4..16`).

### OOM Fallback Sequence
1. Reduce `training.batch_size` by half.
2. Increase `training.gradient_accumulation_steps` to preserve effective batch.
3. Reduce `training.max_length` from `512` to `384`, then `256`.
4. Reduce `training.dataset_batch_size` and `training.num_workers` if host RAM pressure appears.
5. Use compact presets (`es_mlm_albert_sophia`, `es_mlm_distilmbert_lamb`, `es_arch_mini_cautious_adamw`).

### Stability Knobs
- Keep `grad_clip_max_norm` at `5.0` as baseline.
- Increase warmup by raising `scheduler_warmup_ratio` (for unstable loss spikes).
- Prefer `scheduler_type: cosine`; use `constant` only for conservative optimizer experiments.
- Keep `inf_post_clip_threshold` and `max_nan_retries` defaults unless reproducible instability requires stricter values.

## Full Preset Catalog
| filename | task | base model or architecture family | optimizer | norm | attention pattern | expected VRAM tier | stability tag |
|---|---|---|---|---|---|---|---|
| `es_mlm_beto_adamw.yaml` | mlm | `dccuchile/bert-base-spanish-wwm-cased` | `adamw` | n/a | base-model MLM | medium | `p40_safe` |
| `es_mlm_bertin_lion.yaml` | mlm | `bertin-project/bertin-roberta-base-spanish` | `lion` | n/a | base-model MLM | medium | `p40_safe` |
| `es_mlm_maria_roberta_adafactor.yaml` | mlm | `PlanTL-GOB-ES/roberta-base-bne` | `adafactor` | n/a | base-model MLM | medium | `p40_safe` |
| `es_mlm_modernbert_galore_adamw.yaml` | mlm | `answerdotai/ModernBERT-base` | `galore_adamw` | n/a | base-model MLM | medium-high | `p40_experimental` |
| `es_mlm_xlmr_radam.yaml` | mlm | `xlm-roberta-base` | `radam` | n/a | base-model MLM | high | `p40_safe` |
| `es_mlm_mbert_schedulefree_adamw.yaml` | mlm | `bert-base-multilingual-cased` | `schedulefree_adamw` | n/a | base-model MLM | high | `p40_safe` |
| `es_mlm_distilmbert_lamb.yaml` | mlm | `distilbert-base-multilingual-cased` | `lamb` | n/a | base-model MLM | low-medium | `p40_safe` |
| `es_mlm_mdeberta_shampoo.yaml` | mlm | `microsoft/mdeberta-v3-base` | `shampoo` | n/a | base-model MLM | high | `p40_experimental` |
| `es_mlm_deberta_soap.yaml` | mlm | `microsoft/deberta-v3-base` | `soap` | n/a | base-model MLM | high | `p40_experimental` |
| `es_mlm_albert_sophia.yaml` | mlm | `albert-base-v2` | `sophia` | n/a | base-model MLM | low | `p40_safe` |
| `es_arch_titan_dynamic_tanh_sgd_momentum.yaml` | mlm | custom `frankenstein` | `sgd_momentum` | `dynamic_tanh` | `[titan_attn]` | medium | `p40_safe` |
| `es_arch_standard_layer_norm_prodigy.yaml` | mlm | custom `frankenstein` | `prodigy` | `layer_norm` | `[standard_attn]` | medium | `p40_safe` |
| `es_arch_sigmoid_derf_muon.yaml` | mlm | custom `frankenstein` | `muon` | `derf` | `[sigmoid_attn]` | medium | `p40_experimental` |
| `es_arch_hybrid_turbo_muon.yaml` | mlm | custom `frankenstein` hybrid | `turbo_muon` | `layer_norm` | `[retnet, titan_attn, mamba, ode]` | high | `p40_experimental` |
| `es_arch_titan_hope_adan.yaml` | mlm | custom `frankenstein` | `adan` | `dynamic_tanh` | `[titan_attn]` | high | `p40_experimental` |
| `es_arch_standard_sigmoid_adopt.yaml` | mlm | custom `frankenstein` | `adopt` | `layer_norm` | `[standard_attn, sigmoid_attn]` | medium | `p40_safe` |
| `es_arch_moe_titan_ademamix.yaml` | mlm | custom `frankenstein` MoE | `ademamix` | `layer_norm` | `[titan_attn, retnet]` | high | `p40_experimental` |
| `es_arch_bitnet_factorized_mars_adamw.yaml` | mlm | custom `frankenstein` BitNet+factorized | `mars_adamw` | `derf` | `[retnet, standard_attn]` | medium | `p40_safe` |
| `es_arch_mini_cautious_adamw.yaml` | mlm | custom `mini` | `cautious_adamw` | `layer_norm` | `[retnet, titan_attn, mamba, ode, standard_attn, sigmoid_attn]` | low-medium | `p40_safe` |
| `es_arch_gla_deltanet_lion.yaml` | mlm | custom `frankenstein` gated | `lion` | `layer_norm` | `[gla_attn, deltanet_attn]` | medium | `p40_safe` |
| `es_arch_hgrn2_fox_gated_softmax_adamw.yaml` | mlm | custom `frankenstein` gated | `adamw` | `dynamic_tanh` | `[hgrn2_attn, fox_attn, gated_softmax_attn]` | medium | `p40_safe` |
| `es_arch_all_gated_muon.yaml` | mlm | custom `frankenstein` all-gated BitNet | `muon` | `derf` | all 6 gated types | high | `p40_experimental` |
| `es_arch_sparse_longformer_adamw.yaml` | mlm | custom `frankenstein` sparse | `adamw` | `layer_norm` | `[sparse_transformer_attn, longformer_attn]` | medium-high | `p40_safe` |
| `es_arch_bigbird_sparsek_lamb.yaml` | mlm | custom `frankenstein` sparse | `lamb` | `layer_norm` | `[bigbird_attn, sparsek_attn]` | medium-high | `p40_safe` |
| `es_arch_nsa_retnet_radam.yaml` | mlm | custom `frankenstein` sparse+retnet | `radam` | `dynamic_tanh` | `[nsa_attn, retnet_attn]` | medium | `p40_safe` |
| `es_sbert_beto_mean.yaml` | sbert | `dccuchile/bert-base-spanish-wwm-cased` | n/a (SBERT trainer) | n/a | sentence pooling `mean` | medium | `p40_safe` |
| `es_sbert_bertin_cls.yaml` | sbert | `bertin-project/bertin-roberta-base-spanish` | n/a (SBERT trainer) | n/a | sentence pooling `cls` | medium | `p40_safe` |
| `es_sbert_maria_max.yaml` | sbert | `PlanTL-GOB-ES/roberta-base-bne` | n/a (SBERT trainer) | n/a | sentence pooling `max` | medium | `p40_safe` |
| `es_sbert_modernbert_mean.yaml` | sbert | `answerdotai/ModernBERT-base` | n/a (SBERT trainer) | n/a | sentence pooling `mean` | medium-high | `p40_experimental` |
| `es_sbert_xlmr_mean.yaml` | sbert | `xlm-roberta-base` | n/a (SBERT trainer) | n/a | sentence pooling `mean` | high | `p40_safe` |

## Selector Guides
### Decoder autoregressive presets (famous architecture-inspired)
- `decoder_gpt_like.yaml`
- `decoder_llama_like.yaml`
- `decoder_retnet_mamba_like.yaml`
- `decoder_gated_mixed_adamw.yaml`  ← GLA/DeltaNet/HGRN2 causal decoder

### Gated attention presets
- `es_arch_gla_deltanet_lion.yaml` — GLA + DeltaNet
- `es_arch_hgrn2_fox_gated_softmax_adamw.yaml` — HGRN2, ForgettingAttn, GatedSoftmax
- `es_arch_all_gated_muon.yaml` — all 6 gated types

### Sparse attention presets
- `es_arch_sparse_longformer_adamw.yaml` — SparseTransformer + Longformer
- `es_arch_bigbird_sparsek_lamb.yaml` — BigBird + SparseK
- `es_arch_nsa_retnet_radam.yaml` — NSA + RetNetAttention

### Best first run
- `es_mlm_beto_adamw.yaml`
- `es_mlm_distilmbert_lamb.yaml`
- `es_arch_standard_layer_norm_prodigy.yaml`

### Lowest VRAM
- `es_mlm_albert_sophia.yaml`
- `es_mlm_distilmbert_lamb.yaml`
- `es_arch_mini_cautious_adamw.yaml`

### Fastest SBERT adaptation
- `es_sbert_beto_mean.yaml`
- `es_sbert_bertin_cls.yaml`

### Broad optimizer benchmarking
Use all 19 MLM presets (`es_mlm_*` + `es_arch_*`) to benchmark one optimizer per preset under matched data and runtime settings.

## Troubleshooting Notes (Schema and Runtime)
- For MLM with `base_model`, `tokenizer.name_or_path` is required.
- For MLM, `training.optimizer` is required and parameter keys must match optimizer prefix exactly.
- For SBERT, `training.sbert` is required and `base_model` must be set.
- Unknown keys are rejected by schema (`additionalProperties: false` in top-level and major blocks).
- If training becomes unstable on P40, keep `use_amp: false`, lower `max_length`, and increase `gradient_accumulation_steps`.
