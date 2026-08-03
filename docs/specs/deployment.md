# Deployment and Quantization Specification

> Cross-references: [Architecture](architecture.md) · [Schema Reference](schema-reference.md) · [CLI Reference](cli-reference.md)

## Deploy Pipeline

```
Trained checkpoint → Weight packing (ternary / low-bit) → Activation scaling (INT8) → Deployable artifact
```

The codebase treats quantization as a deploy-stage transformation rather than a separate model family. The pipeline is accessed via the `deploy` and `quantize` CLI subcommands.

## Quantization Methods

### Ternary Weight Packing (BitNet b1.58)

Given weight tensor `W`, the per-tensor absmean scaling follows BitNet b1.58:

```
s = mean(|W|)
W̃ = clip(round(W / s), −1, 1)      # ternary {-1, 0, 1}
stored weight = W̃ · s
```

The packed mapping uses **two bits per weight symbol** (4 values per byte;
`-1 → 0b00`, `0 → 0b01`, `1 → 0b10`, `0b11` reserved). This yields
~1.58-bit storage (log₂(3) ≈ 1.58 bits per ternary symbol).

Reference: Ma et al. (2024), "The Era of 1-bit LLMs", arXiv:2402.17764.

### Which layers are quantized?

The quantizer only packs **BitNet modules** — `BitLinear` and (when enabled)
`BitConv1d` — identified by `isinstance`. Full-precision parameters (norms,
embeddings lookup, biases, and routing/scoring projections when
`bitnet_routers` is false) are stored verbatim. The model itself decides at
construction time which layers become BitNet modules, so the quantizer
automatically honours the schema flags:

| Schema flag | Default | What it controls |
|---|---|---|
| `use_bitnet` | `true` (UltraConfig) / `false` (example) | Q/K/V/O, FFN up/down, embeddings `proj`, LM head, and all recurrent-state **gates** (alpha/beta/forget/erase/write/merge/gk) → `BitLinear` |
| `bitnet_routers` | `false` | When `true`, also quantizes routing/scoring: MoE router, MoD router, sparse block-index/forecast, top-k score nets. Default `false` keeps them full-precision for routing stability |
| `use_bitnet_conv` | `false` | When `true` (and `use_embedding_conv`), replaces the embedding `Conv1d` with `BitConv1d`. Off by default; the conv runs over the reduced embedding stream where ternary noise can be costly |

### Baking faithful ternary weights

During training, `BitLinear`/`BitConv1d` keep a full-precision **master** weight
and apply ternary quantization via the straight-through estimator (STE) on
every forward. At export/deploy time, `bake_bitnet_weights()` replaces the
master with the quantized value `{-1, 0, 1} · s` once, so the checkpoint
becomes compact and self-describing. Baking is **idempotent** (a no-op on
already-ternary weights) and is applied automatically by `deploy --format
quantized` and `transformers-export`. Training should not continue after
baking (the STE gradient path is gone).

### INT8 Activation Quantization

For activations `x` (per-token, per the BitNet formulation):

```
α = 127 / (max(|x|) + ε)
q = clip(round(x · α), −128, 127)       q ∈ [−128, 127]
```

Dequantization: `x ≈ q / α`. Applied inside `BitLinear`/`BitConv1d.forward`
via STE; the inputs are LayerNorm-normalized over the last dimension (`H`)
before quantization.

## Size Estimates

For `N` parameters:

| Format | Approximate Size | Formula |
|---|---|---|
| FP32 | 4N bytes | `4 × N` |
| FP16 | 2N bytes | `2 × N` |
| 1.58-bit (ternary packed) | ~0.1975N bytes | `(1.58/8) × N` |

Before metadata and packing overhead. A 110M parameter model:
- FP32: ~440 MB
- FP16: ~220 MB
- 1.58-bit: ~22 MB

## Deployment Artifact Contents

A deployment artifact directory contains:

| File | Description |
|---|---|
| `config.json` | Model architecture configuration (JSON) |
| `model_quantized.pt` | Quantized model weights (PyTorch format, ternary-packed) |
| `deployment_info.json` | Metadata: `use_bitnet`, `bitnet_routers`, `use_bitnet_conv`, `baked_ternary_weights`, `baked_bitlinear_layers`, `quantization` description |

`deployment_info.json` reflects the **actual** quantization flags rather
than assuming BitNet unconditionally. For `--format standard`, weights are
stored as FP32 without ternary packing.

## Inference Modes

The `infer` subcommand supports three modes:

| Mode | Flag | Description |
|---|---|---|
| Single text | `--text "input"` | Run inference on one text string, print result |
| File batch | `--input file.txt --output results.json` | Process a file (one text per line), write JSON results |
| Benchmark | `--benchmark` | Measure throughput and latency over multiple runs |

### Inference Options

| Flag | Effect |
|---|---|
| `--fp16` | Load model in FP16 (reduces memory ~50%). **Disabled** for BitNet models: `BitLinear`'s STE activation/weight quantization assumes float32 statistics, so FP16 would corrupt the ternary scales. A warning is logged and FP32 is kept |
| `--batch-size N` | Process N texts simultaneously in file mode |
| `--device` | Select compute device (auto/cpu/cuda/mps) |

## Quantization CLI

The `quantize` subcommand is a convenience wrapper that calls `deploy` with `--format quantized`:

```bash
# These are equivalent:
frankestein-transformer quantize --checkpoint model.pt --output ./out
frankestein-transformer deploy --checkpoint model.pt --output ./out --format quantized
```

## Validation

The `--validate` flag on `deploy`/`quantize` runs a forward pass with the quantized model and checks:
- Output tensor shapes match expectations
- No NaN/Inf in activations
- Numerical consistency within tolerance vs FP32 reference

## HuggingFace Transformers Export

The `transformers-export` subcommand (and `--transformers-export` flag on `train`/`deploy`/`quantize`) converts a checkpoint + YAML config into a HuggingFace Transformers-compatible folder with:
- `config.json` (HF format)
- `pytorch_model.bin` or `model.safetensors`
- Tokenizer files (if available)

Compatibility is validated before export — only standard-compatible architectures (no custom mixers) can be exported.

### BitNet-aware export

When `use_bitnet: true`, the export additionally:

1. **Bakes** every `BitLinear`/`BitConv1d` weight to faithful `{-1, 0, 1} · s`
   (via `bake_bitnet_weights`) so `pytorch_model.bin` carries self-describing
   ternary values instead of full-precision masters.
2. Writes a `quantization_config` block into `config.json`:
   ```json
   "quantization_config": {
     "quant_method": "bitnet",
     "bits": 1.58,
     "ternary_weights": true,
     "activation_bits": 8,
     "bitnet_routers": false
   }
   ```
   This lets HF tooling identify the model as a 1.58-bit BitNet model.
3. Emits a warning in the compatibility report when `bitnet_routers` is false
   (routing/scoring projections remain full-precision — the recommended
   default for routing stability).

The generated `modeling_frankestein.py` rebuilds `BitLinear`/`BitConv1d` from
source and applies runtime quantization via STE on load, so the exported
model is loadable with `trust_remote_code=True` and produces ternary-aware
forward passes. Encoder (MLM) and decoder (causal) variants are both
supported; `FrankensteinDecoder` forces `mode=decoder`.

## BitNet.cpp / GGUF Export (best-effort)

[`src/deploy/bitnet_gguf_export.py`](../../src/deploy/bitnet_gguf_export.py)
provides a **best-effort** exporter targeting
[Microsoft/BitNet](https://github.com/microsoft/BitNet) (bitnet.cpp), which
is built on llama.cpp and consumes the **GGUF** format with the `i2_s`
quantization type (ternary `{-1, 0, 1}` weights + per-tensor fp scale).

### Scope and limitations

bitnet.cpp only supports architectures registered in llama.cpp (Llama,
Falcon3, Falcon-E, and the official BitNet b1.58 models). The Frankenstein
**hybrid** model (33 mixer families, custom code) **cannot run in
bitnet.cpp** because its compute graph is not registered. Therefore the GGUF
exporter only accepts models configured with a **single mixer type
`standard_attn`** (pure encoder or decoder), whose attention/FFN graph maps
to the Llama-style tensor naming. Any other `layer_pattern` (e.g.
`retnet`, `mamba`, `gla_attn`, …) is rejected by `check_gguf_compatibility`.

```bash
# Compatibility check (returns is_compatible + reason)
frankestein-transformer bitnet-gguf --yaml cfg.yaml --output out.gguf --check

# Export a standard_attn-only BitNet model to i2_s GGUF
frankestein-transformer bitnet-gguf --model ckpt.pt --yaml cfg.yaml --output out.gguf
```

The writer is a self-contained GGUF v3 emitter (no dependency on the
optional `gguf` package): it packs ternary indices 2 bits per value and
records `general.quantization = "i2_s (BitNet b1.58 ternary)"`. For full
fidelity / production use, prefer the official llama.cpp / `gguf` tooling
applied to a Llama-compatible checkpoint.

> ⚠️ This is **experimental**. The hybrid mixers have no equivalent in
> llama.cpp and are intentionally unsupported by this exporter.
