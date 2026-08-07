# `full_tests` — Tests end-to-end exhaustivos

Esta carpeta contiene un harness independiente que entrena modelos **reales pero diminutos** usando el CLI `frankenstein-transformer` como subproceso, con datos sintéticos y un tokenizador entrenado **al vuelo** una sola vez y persistido en `tmp/`.

**No forma parte de la suite CI/CD** (vive fuera de `tests/`) y está pensado para dejarse corriendo varias horas en una máquina de desarrollo.

## Qué prueba

- **Atenciones de a pares**: `itertools.combinations(ATTENTIONS, 2)` (~528 modelos de 2 capas).
- **Cada atención sola**: una capa con cada mezclador entrenable.
- **Todos los optimizadores** del schema (`src/schema/_optimizer.yaml`).
- **Todas las normas** (`layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`, `flash_norm`).
- **Transversales**: BitNet (incluido routers/conv), embeddings factorizados/conv, MoE, MoD, mHC, residuos (`standard/none/full_attn/block_attn`), RoPE vs HoPE, loops duplicados, activaciones de FFN, etc.
- **Tareas**: `mlm`/encoder y `causal_lm`/decoder.
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
