# `full_tests` — Tests end-to-end exhaustivos

Esta carpeta contiene un harness independiente que entrena modelos **reales pero diminutos** usando el CLI `frankenstein-transformer` como subproceso, con datos sintéticos y un tokenizador entrenado **al vuelo** una sola vez y persistido en `tmp/`.

**No forma parte de la suite CI/CD** (vive fuera de `tests/`) y está pensado para dejarse corriendo varias horas en una máquina de desarrollo.

## Qué prueba

- **Atenciones de a pares**: `itertools.combinations(ATTENTIONS, 2)` (~561 modelos de 2 capas). Incluye `gma_attn` (Gaussian Mixture Attention). `fasa_attn` y `sparge_attn` se omiten (solo evaluación).
- **Cada atención sola**: una capa con cada mezclador entrenable.
- **Todos los optimizadores** del schema (`src/schema/_optimizer.yaml`).
- **Todas las normas** (`layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`, `flash_norm`).
- **Todas las codificaciones posicionales** a nivel de modelo (`rope`, `hope`, `nope`, `alibi`, `bam`, `pape`, `pape_efficient`, `pape_ri`, `sinusoidal_absolute`, `sinusoidal_rotary`, `learned_absolute`, `none`) — una config cada una sobre un patrón base fijo `[standard_attn, titan_attn]` (sin cross product).
- **Todas las `pos_embedding_type` del ViT** (`learned_1d`, `none`, `learned_absolute`, `sinusoidal_absolute`, `sinusoidal_rotary`, `pape`, `pape_efficient`, `pape_ri`, `rope`, `hope`, `nope`, `alibi`, `bam`) a través de las 3 tareas de visión (classification, patch_prediction, segmentation) — 39 configs, sin cross product con mixers.
- **Transversales**: BitNet (incluido routers/conv), embeddings factorizados/conv, MoE, MoD, mHC, residuos/AttnRes (`standard`/`none`/`full_attn`/`block_attn`), RoPE vs HoPE, SSMax (`use_ssmax`), loops duplicados, activaciones de FFN, etc.
- **Tareas**: `mlm`/encoder y `causal_lm`/decoder; visión (`classification`, `patch_prediction`, `segmentation`).
- **Deploy / infer / cuantización / transformers-export / bitnet-gguf** (smoke tests sobre el primer entrenamiento exitoso).

## Cómo ejecutar

Desde la raíz del repo, dentro del entorno `frankenstein` de conda:

```bash
conda run -n frankenstein python full_tests/run_e2e.py
```

Ejecución rápida de solo 3 optimizadores:

```bash
conda run -n frankenstein python full_tests/run_e2e.py --category opt --limit 3
```

Solo atenciones de a pares:

```bash
conda run -n frankenstein python full_tests/run_e2e.py --category attn
```

Saltar el barrido más lento de atenciones de a pares:

```bash
conda run -n frankenstein python full_tests/run_e2e.py --skip-attn-pairs
```

Solo las codificaciones posicionales a nivel de modelo:

```bash
conda run -n frankenstein python full_tests/run_e2e.py --category pe
```

Solo las `pos_embedding_type` del ViT (3 tareas × 12 PEs):

```bash
conda run -n frankenstein python full_tests/run_e2e.py --category vision_pe
```

### Selección de dispositivo

Por defecto todo corre en `cpu`. Para ejecutar los tests en otro dispositivo (entrenamiento, deploy e inferencia):

```bash
# En GPU
conda run -n frankenstein python full_tests/run_e2e.py --device cuda

# En Apple Silicon (Metal)
conda run -n frankenstein python full_tests/run_e2e.py --device mps

# Dejar que el CLI resuelva el dispositivo automáticamente
conda run -n frankenstein python full_tests/run_e2e.py --device auto
```

Valores válidos: `auto`, `cpu`, `cuda`, `mps` (por defecto `cpu`).

### Guarda térmica de GPU

La guarda térmica está **desactivada por defecto** (se pasa `--no-gpu-temp-guard`). Para activarla durante el entrenamiento en GPU y ajustar sus umbrales:

```bash
conda run -n frankenstein python full_tests/run_e2e.py --device cuda \
  --gpu-temp-guard \
  --gpu-temp-pause-threshold-c 80 \
  --gpu-temp-resume-threshold-c 70 \
  --gpu-temp-critical-threshold-c 90 \
  --gpu-temp-poll-interval-seconds 5 \
  --gpu-temp-checkpoint-grace-seconds 30
```

Cada umbral es opcional; si se omite, el CLI usa su valor por defecto. Solo tienen efecto cuando `--gpu-temp-guard` está activo.

## Filosofía de los resultados

- `OK`: el entrenamiento terminó sin error.
- `GRAD_EXPLODED`: el proceso falló con NaN/inf/gradiente. **Esto es tolerado y esperado** para algunas combinaciones; se registra pero no detiene la ejecución.
- `FAILED` / `TIMEOUT`: fallo inesperado. El script termina con exit code != 0 si hay alguno.

## Arhivos

- `_helpers.py` — entrenamiento del tokenizador SPM, generación del parquet, runner de subprocesos, helpers de deploy/infer/export/gguf, persistencia de resultados.
- `run_e2e.py` — bucles for que generan todas las combinaciones y lanzan el CLI.
- `tmp/` — tokenizador, datos, checkpoints por run, resultados, deploys. **Git-ignorado.**

## Notas importantes

- `max_samples` y `dataset_batch_size` son pequeños para que el corpus de 7 frases produzca unos pocos batches.
- Cada run tiene su propio `cwd` (`full_tests/tmp/runs/<id>/`) para aislar los checkpoints.
- El tokenizador se entrena una sola vez y se copia al CWD de cada run (`es_redpajama_<vocab_size>.model`) para evitar que el loader legacy intente bajar 100 GB de datos.
- Se fija semilla vía un `sitecustomize.py` generado en `tmp/` y añadido a `PYTHONPATH`, sembrando `torch`, `numpy` y `random` en cada subproceso sin modificar el proyecto.
