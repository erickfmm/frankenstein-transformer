# CLI Command Reference

> Cross-references: [Schema Reference](schema-reference.md) · [Training Safety](training-safety.md) · [Deployment](deployment.md) · [SBERT Workflows](sbert-workflows.md)

## Entrypoint

```
frankenstein-transformer <subcommand> [flags]
```

## Subcommand Overview

| Subcommand | Purpose |
|---|---|
| `train` | Run schema-validated training from a YAML config (MLM, SBERT, causal LM, vision, ...) |
| `deploy` | Convert checkpoint to deployment artifacts (`quantized`/`standard`) or export formats (`transformers`, `gguf`) |
| `infer` | Run inference with a deployed model (`--task mlm`) or a trained SBERT model (`--task sbert`) |
| `web-server` | Launch Streamlit configuration builder |

### Global flags

Every subcommand accepts the standard Python argparse flags:

| Flag | Description |
|---|---|
| `--help` / `-h` | Show the subcommand's help and exit |
| `--version` | Print the version and exit (where supported) |

Run `frankenstein-transformer <subcommand> --help` to see the exact flags and
defaults for that subcommand.

## Design Rules

- **The YAML config is the single source of truth.** Every model/training
  setting (batch size, model class, thermal guard, resume checkpoint, ...) is
  defined in the YAML schema (`src/schema.yaml`) — the CLI does not duplicate
  them.
- **`--device` is the only runtime override.** All model-executing commands
  accept `--device` because it selects the execution environment, not a
  training hyperparameter:

  | Value | Behavior |
  |---|---|
  | `auto` | Auto-detect best available device (CUDA > MPS > CPU) |
  | `cpu` | Force CPU execution |
  | `cuda` | Force CUDA GPU |
  | `mps` | Force Apple Metal Performance Shaders |

## `train` — Main Training

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | string | — | Path to custom YAML config file |
| `--config-name` | string | `frankenstein` | Named preset from `configs/` directory |
| `--list-configs` | flag | — | List available config presets and exit |
| `--device` | choice | `auto` | `auto`, `cpu`, `cuda`, `mps` |

### Examples

```bash
# Train with default frankenstein preset
frankenstein-transformer train

# Train with custom config
frankenstein-transformer train --config my_experiment.yaml

# Train a SBERT model (training.task: sbert lives in the YAML)
frankenstein-transformer train --config-name modernbert_sbert

# List available presets
frankenstein-transformer train --list-configs
```

Thermal-guard, batch size and resume settings are YAML keys under
`training.*` (e.g. `training.gpu_temp_guard_enabled`,
`training.resume_from_checkpoint`) — see
[Training Safety](training-safety.md).

## `deploy` — Deployment Artifacts & Export Formats

| Flag | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | string | **Required** | Path to trained checkpoint (.pt) |
| `--output` | string | **Required** | Output directory (or `.gguf` path for `gguf`) |
| `--format` | choice | `quantized` | `quantized`, `standard`, `transformers`, `gguf` |
| `--yaml` | string | — | Training YAML (required for `transformers`/`gguf`) |
| `--validate` | flag | — | Validate artifact after creation (`standard`/`quantized`) |
| `--check` | flag | — | Only run the GGUF compatibility check and exit (`gguf` only) |
| `--device` | choice | `auto` | `auto`, `cpu`, `cuda`, `mps` |

### Examples

```bash
# Quantized deployment (default format — this replaces the old `quantize` command)
frankenstein-transformer deploy --checkpoint checkpoints/model.pt --output ./deployed --validate

# Standard FP32 artifacts
frankenstein-transformer deploy --checkpoint checkpoints/model.pt --output ./deployed --format standard

# HuggingFace Transformers export (replaces the old `transformers-export` command)
frankenstein-transformer deploy --checkpoint checkpoints/model.pt --yaml config.yaml --output ./hf-export --format transformers

# BitNet GGUF export (replaces the old `bitnet-gguf` command)
frankenstein-transformer deploy --checkpoint ckpt.pt --yaml cfg.yaml --output out.gguf --format gguf

# GGUF compatibility check only
frankenstein-transformer deploy --checkpoint ckpt.pt --yaml cfg.yaml --output out.gguf --format gguf --check
```

> ⚠️ The `gguf` exporter only supports BitNet models with a single
> `standard_attn` mixer (`use_bitnet: true`) — see
> [Deployment](deployment.md) for the limitations.

## `infer` — Model & SBERT Inference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | string | **Required** | Path to deployed model dir (mlm) or trained SBERT model (sbert) |
| `--task` | choice | `mlm` | `mlm` (deployed encoder) or `sbert` (sentence embeddings) |
| `--text` | string | — | Single text input (mlm) |
| `--input` | string | — | Input file path, one text per line (mlm) |
| `--output` | string | — | Output file for results (mlm) |
| `--fp16` | flag | — | Use FP16 precision (mlm) |
| `--batch-size` | int | `8` | Batch size for file processing (mlm) / SBERT (`32` forwarded) |
| `--benchmark` | flag | — | Run inference benchmark (mlm) |
| `--mode` | choice | — | `similarity`, `search`, `cluster`, `encode` (required with `--task sbert`) |
| `--sentence1` / `--sentence2` | string | — | Pairwise similarity inputs (sbert) |
| `--query` | string | — | Search query (sbert search) |
| `--corpus-file` | string | — | Corpus file (sbert search) |
| `--top-k` | int | `5` | Top-k results (sbert search) |
| `--sentences-file` | string | — | Sentences file (sbert cluster/encode) |
| `--n-clusters` | int | `5` | Number of clusters (sbert cluster) |
| `--input-file` | string | — | Input file (sbert encode) |
| `--output-file` | string | — | Output file (sbert encode) |
| `--device` | choice | `auto` | `auto`, `cpu`, `cuda`, `mps` |

### Examples

```bash
# Single text inference
frankenstein-transformer infer --model ./deployed --text "Hello world"

# Batch file inference
frankenstein-transformer infer --model ./deployed --input texts.txt --output results.json

# Benchmark
frankenstein-transformer infer --model ./deployed --benchmark --fp16

# Pairwise similarity (replaces the old `sbert-infer` command)
frankenstein-transformer infer --model ./sbert --task sbert --mode similarity --sentence1 "Hello" --sentence2 "Hi"

# Corpus search
frankenstein-transformer infer --model ./sbert --task sbert --mode search --query "machine learning" --corpus-file docs.txt --top-k 10

# Clustering
frankenstein-transformer infer --model ./sbert --task sbert --mode cluster --sentences-file texts.txt --n-clusters 8

# Encode and export
frankenstein-transformer infer --model ./sbert --task sbert --mode encode --input-file texts.txt --output-file embeddings.npz
```

## `web-server` — Streamlit Configuration Builder

| Flag | Type | Default | Description |
|---|---|---|---|
| `--server-port` | int | `8501` | Streamlit server port |
| `--server-address` | string | `localhost` | Bind address |
| `--server-headless` | flag | — | Run without opening browser |
| `--development-mode` | flag | — | Enable debug logging |

### Examples

```bash
frankenstein-transformer web-server
frankenstein-transformer web-server --server-port 8080 --server-headless
```

## Migration From the Old 9-Command Surface

| Removed command / flag | Replacement |
|---|---|
| `quantize` | `deploy` (quantized is the default format) |
| `transformers-export` | `deploy --format transformers` |
| `bitnet-gguf` | `deploy --format gguf` |
| `sbert-train` | `train` with `training.task: sbert` in the YAML (e.g. `configs/modernbert_sbert.yaml`) |
| `sbert-infer` | `infer --task sbert --mode ...` |
| `train --transformers-export` | Run `deploy --format transformers` after training |
| `train --batch-size` | `training.batch_size` YAML key |
| `train --model-mode` | top-level `model_class` YAML key |
| `train --gpu-temp-*`, `--switch-on-thermal` | `training.gpu_temp_*` / `training.switch_on_thermal` YAML keys |
| `train --resume-from-checkpoint` | `training.resume_from_checkpoint` YAML key |
| `deploy --config` (JSON) | Unneeded — checkpoints embed their config |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `error: unrecognized arguments` | Wrong flags for the subcommand | Run `<subcommand> --help` to confirm flag names |
| Config fails validation with "additional properties" | An unknown YAML key | Remove the key or add it to the schema |
| `--yaml required` error on `deploy` | `--format transformers\|gguf` needs the source YAML | Pass `--yaml config.yaml` |
| `--mode required` error on `infer` | `--task sbert` requires an inference mode | Add `--mode similarity\|search\|cluster\|encode` |
| Model loads but produces garbage | Vocab mismatch between checkpoint and tokenizer | Ensure `vocab_size` matches the tokenizer |
| FP16 ignored on `infer` | The model uses BitNet (`BitLinear`) | FP16 is disabled for BitNet models by design |
| Training aborts at a temperature | GPU thermal guard hit `critical_threshold` | Raise `training.gpu_temp_critical_threshold_c` in the YAML or improve cooling |
