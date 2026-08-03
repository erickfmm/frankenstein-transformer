# Transformers export compatibility

This document describes the compatibility rules for the CLI command:

```bash
frankestein-transformer transformers-export --model <checkpoint.pt> --yaml <train.yaml> --output <output_dir>
```

## Integrated `--transformers-export` flag

The same `--transformers-export` flag is available in other CLI commands.

- `train --transformers-export`
  - Performs a compatibility pre-check using the resolved training YAML.
  - If compatible, exports to `checkpoints/transformers-export/` after successful training.
- `deploy --transformers-export --yaml <train.yaml>`
  - Performs compatibility pre-check before deploy.
  - On success, exports to `<deploy-output>/transformers-export/`.
- `quantize --transformers-export --yaml <train.yaml>`
  - Performs compatibility pre-check before quantization.
  - On success, exports to `<quantize-output>/transformers-export/`.

Unsupported with this flag (returns error and stops): `infer`, `sbert-train`, `sbert-infer`.

## Export output layout

When compatible, the command creates:

- `config.json` (Transformers config with `auto_map`)
- `pytorch_model.bin` (weights)
- `configuration_frankestein.py` (custom `PretrainedConfig`)
- `modeling_frankestein.py` (custom `PreTrainedModel` wrappers)
- `model/` (Frankenstein architecture source copied from `src/model/`)
- `__init__.py`
- `compatibility_report.json`
- `export_info.json`

## Compatibility matrix

### High-level training/task compatibility

| Input setting | Compatible | Notes |
|---|---|---|
| `training.task: mlm` | ✅ | Exported as `FrankesteinForMaskedLM` unless decoder mode is active. |
| `model_class: frankesteindecoder` or `model.mode: decoder` | ✅ | Exported as `FrankesteinForCausalLM`. |
| `training.task: sbert` | ❌ | Not exported by this command (SBERT wrapper is not an AutoModel LM head). |
| `base_model` set in YAML | ❌ | Command does not export base-model fine-tune configs for this custom architecture wrapper. |

### `layer_pattern` compatibility (from `src/schema.yaml` enum)

All schema-declared layer types are recognized by the exporter.  
Special case:

- `fasa_attn`, `sparge_attn`: exported for inference, but they are eval-only layers for training.

If a checkpoint/YAML includes layer types outside the schema enum, the command marks the export as incompatible and writes details to `compatibility_report.json`.
