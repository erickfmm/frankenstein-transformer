# Transformers export compatibility

This document describes the compatibility rules for the CLI command:

```bash
frankenstein-transformer deploy --checkpoint <checkpoint.pt> --yaml <train.yaml> --output <output_dir> --format transformers
```

The compatibility pre-check runs automatically before the export; if the
config is incompatible, the export fails with the report described below.

## Export output layout

When compatible, the command creates:

- `config.json` (Transformers config with `auto_map`; includes
  `frankenstein_quantization` when `--quant` is not `none`)
- `model.safetensors` (weights; the only weight container — no
  `pytorch_model.bin` or bitsandbytes artifact is produced)
- `quantization_frankenstein.py` (block codec copied into the artifact)
- `configuration_frankenstein.py` (custom `PretrainedConfig`)
- `modeling_frankenstein.py` (custom `PreTrainedModel` wrappers)
- `load_frankenstein.py` (`load_pretrained()` helper; imports the nested
  `model/` package that HF's `trust_remote_code` copier does not handle)
- `model/` (Frankenstein architecture source copied from `src/model/`)
- `__init__.py`
- `compatibility_report.json`
- `export_info.json`

## Quantized safetensors

`deploy --format transformers` accepts:

```bash
--quant {none,q8_0,q4_0,i2_s}
--quant-runtime {dequantize_on_load,quantized_linear}
```

| `--quant` | Layout | Bits/weight | Runtime |
|---|---|---|---|
| `q8_0` | 32-weight affine blocks, fp16 scale + 32 int8 | 8.5 | v1/v2 |
| `q4_0` | 32-weight affine blocks, fp16 scale + 16 packed nibbles | 4.5 | v1/v2 |
| `i2_s` | BitNet ternary `{-1,0,1}`, 4 values/byte + fp32 scale | 2.0 | v1/v2 |

`--quant-runtime dequantize_on_load` (v1) expands the packed blocks to fp32
parameters while loading. `--quant-runtime quantized_linear` (v2) keeps
`PackedLinear` modules whose buffers stay packed and dequantize inside
`forward` (preserving `BitLinear` activation quantization when applicable).

Only `nn.Linear` weights whose last dimension is a multiple of 32 are
quantized; embeddings, norms, biases and convolutions stay fp32. Load the
artifact with the bundled helper (HF's dynamic-module copier cannot copy the
nested `model/` source tree):

```python
import sys
sys.path.insert(0, "<model-dir>")
from load_frankenstein import load_pretrained

model = load_pretrained("<model-dir>")
```

It does not use HF's built-in quantization runtimes.

## Compatibility matrix

### High-level training/task compatibility

| Input setting | Compatible | Notes |
|---|---|---|
| `training.task: mlm` | ✅ | Exported as `FrankensteinForMaskedLM` unless decoder mode is active. |
| `model_class: frankenstein_text_decoder` or `model.mode: decoder` | ✅ | Exported as `FrankensteinForCausalLM`. |
| `training.task: sbert` | ❌ | Not exported by this command (SBERT wrapper is not an AutoModel LM head). |
| `base_model` set in YAML | ❌ | Command does not export base-model fine-tune configs for this custom architecture wrapper. |

### `layer_pattern` compatibility (from `src/schema.yaml` enum)

All schema-declared layer types are recognized by the exporter.  
Special case:

- `fasa`, `sparge`: exported for inference, but they are eval-only layers for training.

If a checkpoint/YAML includes layer types outside the schema enum, the command marks the export as incompatible and writes details to `compatibility_report.json`.
