# CLI Command Reference

> Cross-references: [Schema Reference](schema-reference.md) · [Training Safety](training-safety.md) · [Deployment](deployment.md) · [SBERT Workflows](sbert-workflows.md)

## Entrypoint

```
frankestein-transformer <subcommand> [flags]
```

## Subcommand Overview

| Subcommand | Purpose |
|---|---|
| `train` | Run main MLM/decoder training |
| `finetune` | Fine-tune a pretrained model |
| `deploy` | Convert checkpoint to deployment artifacts |
| `quantize` | Export checkpoint in quantized format |
| `infer` | Run deployed model inference |
| `sbert-train` | Train SBERT sentence embedding model |
| `sbert-infer` | Run SBERT inference (similarity/search/cluster/encode) |
| `web-server` | Launch Streamlit configuration builder |
| `transformers-export` | Export checkpoint + YAML to HuggingFace Transformers format |

## Device Choices

All subcommands that involve model computation accept `--device`:

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
| `--config-name` | string | `mini` | Named preset from `configs/` directory |
| `--list-configs` | flag | — | List available config presets and exit |
| `--batch-size` | int | — | Override config batch size |
| `--model-mode` | choice | — | Override model class: `frankenstein`, `mini`, `frankesteindecoder` |
| `--device` | choice | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `--gpu-temp-guard` / `--no-gpu-temp-guard` | flag | — | Enable/disable GPU thermal guard |
| `--gpu-temp-pause-threshold-c` | float | — | Temperature to pause training (°C) |
| `--gpu-temp-resume-threshold-c` | float | — | Temperature to resume training (°C) |
| `--gpu-temp-critical-threshold-c` | float | — | Temperature to abort training (°C) |
| `--gpu-temp-poll-interval-seconds` | float | — | Polling interval for temperature checks |
| `--switch-on-thermal` / `--no-switch-on-thermal` | flag | — | Enable/disable thermal-based device switching |
| `--transformers-export` | flag | `false` | Also export to HuggingFace format after training |

### Examples

```bash
# Train with default mini preset
frankestein-transformer train

# Train with custom config
frankestein-transformer train --config my_experiment.yaml

# Train with GPU thermal guard
frankestein-transformer train --config-name frankenstein --gpu-temp-guard --gpu-temp-pause-threshold-c 80

# List available presets
frankestein-transformer train --list-configs
```

## `deploy` — Deployment Artifact Creation

| Flag | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | string | **Required** | Path to trained checkpoint (.pt) |
| `--output` | string | **Required** | Output directory for artifacts |
| `--format` | choice | `quantized` | `quantized` or `standard` |
| `--validate` | flag | — | Validate artifact after creation |
| `--config` | string | — | Path to YAML config for metadata |
| `--yaml` | string | — | YAML path (required for `--transformers-export`) |
| `--device` | choice | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `--transformers-export` | flag | `false` | Also export to HuggingFace format |

### Examples

```bash
frankestein-transformer deploy --checkpoint checkpoints/model.pt --output ./deployed
frankestein-transformer deploy --checkpoint checkpoints/model.pt --output ./deployed --format standard --validate
```

## `quantize` — Quantized Export

| Flag | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | string | **Required** | Path to trained checkpoint |
| `--output` | string | **Required** | Output directory |
| `--validate` | flag | — | Validate after quantization |
| `--config` | string | — | YAML config path |
| `--yaml` | string | — | YAML path (required for `--transformers-export`) |
| `--device` | choice | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `--transformers-export` | flag | `false` | Also export to HuggingFace format |

### Examples

```bash
frankestein-transformer quantize --checkpoint checkpoints/model.pt --output ./quantized --validate
```

## `infer` — Model Inference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | string | **Required** | Path to deployed model artifact |
| `--text` | string | — | Single text input for inference |
| `--input` | string | — | Input file path (one text per line) |
| `--output` | string | — | Output file for results |
| `--device` | choice | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `--fp16` | flag | — | Use FP16 precision |
| `--batch-size` | int | `8` | Batch size for file processing |
| `--benchmark` | flag | — | Run inference benchmark |

### Examples

```bash
# Single text inference
frankestein-transformer infer --model ./deployed --text "Hello world"

# Batch file inference
frankestein-transformer infer --model ./deployed --input texts.txt --output results.json

# Benchmark
frankestein-transformer infer --model ./deployed --benchmark --fp16
```

## `sbert-train` — SBERT Training

| Flag | Type | Default | Description |
|---|---|---|---|
| `--base-model` | string | — | HuggingFace model identifier |
| `--pretrained` | string | — | Path to pretrained checkpoint |
| `--output_dir` | string | `./output/sbert_tormented_v2` | Output directory |
| `--dataset_name` | string | `erickfmm/agentlans__multilingual-sentences__paired_10_sts` | HuggingFace dataset |
| `--batch_size` | int | `16` | Training batch size |
| `--epochs` | int | `4` | Training epochs |
| `--warmup_steps` | int | `1000` | LR warmup steps |
| `--evaluation_steps` | int | `5000` | Evaluation frequency |
| `--learning_rate` | float | `2e-5` | Learning rate |
| `--max_train_samples` | int | — | Max training samples |
| `--max_eval_samples` | int | `10000` | Max eval samples |
| `--max_seq_length` | int | `512` | Max sequence length |
| `--hidden_size` | int | `768` | Hidden dimension |
| `--num_layers` | int | `12` | Number of layers |
| `--pooling_mode` | choice | `mean` | `mean`, `cls`, `max` |
| `--trust_remote_code` | flag | — | Trust remote code execution |
| `--no_amp` | flag | — | Disable mixed precision |
| `--no_resample` | flag | — | Disable data resampling |
| `--resample_std` | float | `0.3` | Resampling standard deviation |
| `--device` | choice | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `--switch-on-thermal` / `--no-switch-on-thermal` | flag | — | Thermal-based device switching |

### Examples

```bash
frankestein-transformer sbert-train --base-model answerdotai/ModernBERT-base
frankestein-transformer sbert-train --pretrained checkpoints/model.pt --pooling_mode cls
```

## `sbert-infer` — SBERT Inference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model_path` | string | **Required** | Path to trained SBERT model |
| `--mode` | choice | **Required** | `similarity`, `search`, `cluster`, `encode` |
| `--sentence1` | string | — | First sentence (similarity mode) |
| `--sentence2` | string | — | Second sentence (similarity mode) |
| `--query` | string | — | Search query (search mode) |
| `--corpus_file` | string | — | Corpus file (search mode) |
| `--top_k` | int | `5` | Top-k results for search |
| `--sentences_file` | string | — | Sentences file (cluster/encode mode) |
| `--n_clusters` | int | `5` | Number of clusters (cluster mode) |
| `--input_file` | string | — | Input file (encode mode) |
| `--output_file` | string | — | Output file for results |
| `--batch_size` | int | `32` | Batch size |
| `--device` | choice | `auto` | `auto`, `cpu`, `cuda`, `mps` |

### Examples

```bash
# Pairwise similarity
frankestein-transformer sbert-infer --model_path ./output/sbert --mode similarity --sentence1 "Hello" --sentence2 "Hi"

# Corpus search
frankestein-transformer sbert-infer --model_path ./output/sbert --mode search --query "machine learning" --corpus_file docs.txt --top_k 10

# Clustering
frankestein-transformer sbert-infer --model_path ./output/sbert --mode cluster --sentences_file texts.txt --n_clusters 8

# Encode and export
frankestein-transformer sbert-infer --model_path ./output/sbert --mode encode --input_file texts.txt --output_file embeddings.npy
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
frankestein-transformer web-server
frankestein-transformer web-server --server-port 8080 --server-headless
```

## `transformers-export` — HuggingFace Export

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | string | **Required** | Path to checkpoint |
| `--yaml` | string | **Required** | Path to training YAML |
| `--output` | string | **Required** | Output directory |

### Examples

```bash
frankestein-transformer transformers-export --model checkpoints/model.pt --yaml config.yaml --output ./hf-export
```

## GPU Thermal Guard Flags

Available on `train` and `sbert-train` subcommands:

| Flag | Description |
|---|---|
| `--gpu-temp-guard` | Enable GPU temperature monitoring |
| `--no-gpu-temp-guard` | Disable GPU temperature monitoring |
| `--gpu-temp-pause-threshold-c` | Temperature (°C) at which training pauses |
| `--gpu-temp-resume-threshold-c` | Temperature (°C) at which training resumes |
| `--gpu-temp-critical-threshold-c` | Temperature (°C) at which training aborts |
| `--gpu-temp-poll-interval-seconds` | Seconds between temperature checks |
| `--switch-on-thermal` | Enable automatic device switching on thermal events |
| `--no-switch-on-thermal` | Disable automatic device switching |
