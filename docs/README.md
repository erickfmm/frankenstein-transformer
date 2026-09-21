# Toolkit Reference

This document describes the project as a configurable library and CLI toolchain. It is rendered in Read the Docs under **Toolkit Reference**; the full hosted documentation (specs, API autodoc, bibliography, technical reports) lives at <https://frankenstein-transformer.readthedocs.io>.

## Building the documentation locally

The documentation pipeline is **Sphinx + MyST + Read the Docs**. There is no `pdoc` or other pipeline anymore.

```bash
# Build HTML locally (installs Sphinx deps if missing, compiles LaTeX papers if pdflatex is available)
bash docs/build_docs.sh
# Output: docs/_build/html/index.html
```

The Read the Docs build (`.readthedocs.yaml`) reuses the same Sphinx sources under `docs/source/` and copies the generated HTML to the hosting output.

## Academic reports (LaTeX)

The LaTeX sources are split by section under `docs/paper/` (English) and `docs/paper-es/` (Spanish). Each folder is self-contained and references the shared `docs/bibliography/` via a relative `../bibliography/...` path, so compilation must be run **from inside the paper folder**.

English (run from `docs/paper/`):

```bash
cd docs/paper
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

Spanish (run from `docs/paper-es/`):

```bash
cd docs/paper-es
pdflatex -interaction=nonstopmode paper-es.tex
bibtex paper-es
pdflatex -interaction=nonstopmode paper-es.tex
pdflatex -interaction=nonstopmode paper-es.tex
```

Each main file only contains the preamble, front matter, and `\input` links to the per-section files in `sections/` and `appendices/`. The compiled PDFs (`docs/paper/paper.pdf`, `docs/paper-es/paper-es.pdf`) are embedded in the Read the Docs pages **Technical Report (English)** and **Informe Tecnico (Espanol)**.

## 1. CLI Command Surface

The entrypoint is:

```bash
frankenstein-transformer
```

Subcommands:

- `train` — run schema-validated training from a YAML config (MLM, SBERT, causal LM, vision, ...)
- `deploy` — convert a checkpoint into deployment artifacts (`quantized`/`standard`) or export formats (`transformers`, `gguf`)
- `infer` — run inference with a deployed model (`--task mlm`) or a trained SBERT model (`--task sbert`)
- `web-server` — run the Streamlit schema-driven YAML builder

Common execution device choice:

```bash
--device auto|cpu|cuda|mps
```

## 2. Train Command

Usage patterns:

```bash
frankenstein-transformer train --config-name frankenstein --device auto
frankenstein-transformer train --config path/to/config.yaml --device auto
frankenstein-transformer train --list-configs
```

Key train flags:

- `--config` — path to a YAML config file
- `--config-name` — name of a preset under `configs/` (without extension)
- `--list-configs` — list available named presets
- `--device`

Everything else (batch size, model class, thermal guard, resume checkpoint, ...)
is defined in the YAML config — the CLI does not duplicate schema settings.

## 3. Configuration Schema

The authoritative schema is `src/schema.yaml` (a JSON-Schema that `$ref`s into the modular `src/schema/` directory). **There is no `configs/schema.yaml`** — older references to it are stale. The schema is `additionalProperties: false`, so every config key must exist in the schema or validation fails.

Top-level required keys:

- `model_class`
- `model`
- `training`

### 3.1 `model_class`

Allowed values:

- `frankenstein` — full-featured mixed-architecture encoder (17+ mixers, MoE, advanced normalization); best for MLM pre-training
- `frankensteindecoder` — autoregressive causal decoder for LLM-style next-token generation; **forces `mode: decoder`** at runtime

### 3.2 `model` section

Required fields include: `vocab_size`, `hidden_size`, `num_layers`, `num_loops`, `num_heads`, `retention_heads`, `num_experts`, `top_k_experts`, `dropout`, `layer_pattern`, `ode_solver`, `ode_steps`, `use_bitnet`, `norm_type`, `use_factorized_embedding`, `factorized_embedding_dim`, `use_embedding_conv`, `embedding_conv_kernel`, `use_hope`, `use_moe`, `ffn_hidden_size`, `ffn_activation`.

Key enums:

- `layer_pattern` items (35 mixers): `retnet`, `retnet_attn`, `mamba`, `ode`, `titan_attn`, `standard_attn`, `sigmoid_attn`, `sparse_transformer_attn`, `longformer_attn`, `bigbird_attn`, `sparsek_attn`, `nsa_attn`, `sparge_attn`, `fasa_attn`, `gla_attn`, `deltanet_attn`, `gated_deltanet_attn`, `gated_deltanet2_attn`, `hgrn2_attn`, `fox_attn`, `gated_softmax_attn`, `engram_attn`, `gqa_attn`, `mla_attn`, `gqla_attn`, `mlra_attn`, `tucker_attn`, `iha_attn`, `gta_attn`, `mtla_attn`, `cca_attn`, `ccgqa_attn`, `msa_attn`, `sparda_attn`, `kda_attn`. **Note:** `sparge_attn` and `fasa_attn` are eval-only and raise during training.
- `ode_solver`: `rk4`, `euler`
- `norm_type`: `layer_norm` (recommended), `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm` (see `docs/specs/architecture.md`)
- `ffn_activation`: 43 values including `silu`, `gelu`, `swiglu`, `geglu`, `reglu`, plus learnable variants (see `docs/specs/activations.md`)

Optional model keys: `hope_base`, `hope_damping`, `prms_partial_ratio` (for `prms_norm`), `base_model` (pretrained init, requires `training.task` and, for MLM, `tokenizer.name_or_path`).

### 3.3 `training` section

Required core fields: `batch_size`, `dataloader_workers`, `max_length`, `mlm_probability`, `max_samples`, `dataset_batch_size`, `num_workers`, `cache_dir`, `use_amp`, `gradient_accumulation_steps`, `optimizer` (for `task: mlm`) or `sbert` (for `task: sbert`), `scheduler_total_steps`, `scheduler_warmup_ratio`, `scheduler_type`, `grad_clip_max_norm`, `inf_post_clip_threshold`, `max_nan_retries`, `checkpoint_every_n_steps`, `max_rolling_checkpoints`, `num_best_checkpoints`, `nan_check_interval`, `log_gradient_stats`, `gradient_log_interval`, `csv_log_path`, `csv_rotate_on_schema_change`, `gpu_metrics_backend`, `nvml_device_index`, `enable_block_grad_norms`, `telemetry_log_interval`, `gpu_temp_guard_enabled`, `gpu_temp_pause_threshold_c`, `gpu_temp_resume_threshold_c`, `gpu_temp_critical_threshold_c`, `gpu_temp_poll_interval_seconds`, `use_galore`, `galore_rank`, `galore_update_interval`, `galore_scale`, `galore_max_dim`.

`task` enum: `mlm`, `sbert`. Legacy top-level optimizer keys are gone — use `training.optimizer` (MLM) or `training.sbert` (SBERT).

Optional dataset locality fields: `local_parquet_dir`, `prefer_local_cache`, `stream_local_parquet`.

Scheduler enum: `cosine`, `constant`, `linear_warmup_then_constant`.

GPU metrics backend enum: `nvml`, `none`.

### 3.4 Optimizer configuration

`training.optimizer` requires:

- `optimizer_class`
- `parameters`

Allowed `optimizer_class` values (23 families):

`sgd_momentum`, `adamw`, `adafactor`, `galore_adamw`, `prodigy`, `lion`, `sophia`, `muon`, `turbo_muon`, `radam`, `adan`, `adopt`, `ademamix`, `mars_adamw`, `cautious_adamw`, `lamb`, `schedulefree_adamw`, `shampoo`, `soap`, `anon`, `apollo`, `apollo_mini`, `q_apollo`.

Parameter keys must use the prefix for the selected optimizer class, e.g. `adamw-lr_embeddings`, `muon-ns_steps`, `sgd_momentum-momentum`. The prefix list lives in `src/schema/_optimizer.yaml` under `by_optimizer`; adding a new optimizer requires adding a `prefix` entry there.

## 4. Deployment Commands

Deploy a checkpoint to artifacts:

```bash
frankenstein-transformer deploy \
  --checkpoint path/to/checkpoint.pt \
  --output deployed_model \
  --format quantized \
  --validate \
  --device auto
```

Deploy flags:

- `--checkpoint` (required)
- `--output` (required)
- `--format` (`quantized|standard|transformers|gguf`)
- `--yaml` (required for `transformers`/`gguf`)
- `--validate` (standard/quantized)
- `--check` (gguf compatibility check only)
- `--device`

`quantized` is the default format, so `quantize` needs no separate command:

```bash
frankenstein-transformer deploy --checkpoint ckpt.pt --output deployed_model_quantized --validate
```

### 4.1 Hugging Face Transformers export

```bash
frankenstein-transformer deploy --checkpoint deployed_model --yaml configs/frankenstein.yaml --output hf_model --format transformers
```

Flags: `--checkpoint` (required), `--yaml` (required), `--output` (required), `--format transformers`.

### 4.2 BitNet GGUF export

```bash
frankenstein-transformer deploy --checkpoint deployed_model.pt --yaml configs/mini.yaml --output model.gguf --format gguf
# Compatibility check only:
frankenstein-transformer deploy --checkpoint deployed_model.pt --yaml configs/frankenstein.yaml --output model.gguf --format gguf --check
```

Flags: `--checkpoint` (required), `--yaml` (required), `--output` (required), `--format gguf`, `--check`.

## 5. Inference Command

```bash
frankenstein-transformer infer --model deployed_model --text "hola" --device auto
```

Infer flags:

- `--model` (required)
- `--task` (`mlm|sbert`, default `mlm`)
- `--text`
- `--input`
- `--output`
- `--device`
- `--fp16`
- `--batch-size`
- `--benchmark`

## 6. SBERT Workflows

SBERT training is regular training: set `training.task: sbert` (with the
`training.sbert` block) in the YAML and run `train` with a preset such as
`configs/modernbert_sbert.yaml`.

SBERT inference is the `sbert` task of `infer`:

```bash
frankenstein-transformer infer --model ./output/sbert_model --task sbert --mode similarity --sentence1 "a" --sentence2 "b"
```

Flags: `--model` (required), `--task sbert`, `--mode` (`similarity|search|cluster|encode`, required), `--sentence1`, `--sentence2`, `--query`, `--corpus-file`, `--top-k`, `--sentences-file`, `--n-clusters`, `--input-file`, `--output-file`, `--batch-size`, `--device`.

## 7. Web Server (Streamlit)

Run the schema-driven YAML builder:

```bash
frankenstein-transformer web-server
frankenstein-transformer web-server --server-port 8501 --server-address 0.0.0.0 --server-headless
```

Flags: `--server-port` (default 8501), `--server-address` (default `localhost`), `--server-headless`, `--development-mode`.

## 8. Recommended Workflow

1. Select or create a YAML config that validates against `src/schema.yaml`. Named presets live in `configs/` (`frankenstein`, `frankensteindecoder`, `standard`, `tinybert`, `embbert`, …); optimizer × architecture combos live in `configs/examples/`.
2. Run `train`.
3. Export with `deploy --format transformers` (Hugging Face folder) or `deploy --format gguf` (BitNet GGUF file); plain `deploy` produces quantized/standard artifacts.
4. Run `infer` for runtime validation and benchmark.
5. Train/evaluate sentence embeddings via `training.task: sbert` + `train`, then `infer --task sbert`.