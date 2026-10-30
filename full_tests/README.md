# `full_tests` — Tests end-to-end exhaustivos

Esta carpeta contiene un harness independiente que entrena modelos **reales pero diminutos** usando el CLI `frankenstein-transformer` como subproceso. Por defecto (`--realonall`) **todas** las categorías entrenan contra datasets reales de HuggingFace Hub; con `--no-realonall` se usa el corpus sintético y el tokenizador entrenado al vuelo que se persiste en `tmp/`.

**No forma parte de la suite CI/CD** (vive fuera de `tests/`) y está pensado para dejarse corriendo varias horas en una máquina de desarrollo.

## Qué prueba

- **Atenciones de a pares**: `itertools.combinations(ATTENTIONS, 2)` (~820 modelos de 2 capas). Incluye `gma_attn` (Gaussian Mixture Attention) y `ssog_attn` (SSOG — a las configs NLP con `ssog_attn` se les inyecta automáticamente una rejilla `1×max_length`) y la familia Falcon de fast-weight attention (`falcon1_attn`, `falcon2_attn`, `falcon3_attn`, `falcon1a_attn`, `falcon2a_attn`, `falcon3a_attn`). `fasa_attn` y `sparge_attn` se omiten (solo evaluación).
- **Cada atención sola**: una capa con cada mezclador entrenable.
- **Todos los optimizadores** del schema (`src/schema/_optimizer.yaml`).
- **Todas las normas** (`layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`, `flash_norm`).
- **Todas las codificaciones posicionales** a nivel de modelo (`rope`, `hope`, `nope`, `alibi`, `bam`, `pape`, `pape_efficient`, `pape_ri`, `sinusoidal_absolute`, `sinusoidal_rotary`, `learned_absolute`, `none`) — una config cada una sobre un patrón base fijo `[standard_attn, titan_attn]` (sin cross product).
- **Todas las `pos_embedding_type` del ViT** (`learned_1d`, `none`, `learned_absolute`, `sinusoidal_absolute`, `sinusoidal_rotary`, `pape`, `pape_efficient`, `pape_ri`, `rope`, `hope`, `nope`, `alibi`, `bam`) a través de las 3 tareas de visión (classification, patch_prediction, segmentation) — 39 configs, sin cross product con mixers.
- **Transversales**: BitNet (incluido routers/conv), embeddings factorizados/conv, MoE, MoD, mHC, residuos/AttnRes (`standard`/`none`/`full_attn`/`block_attn`), RoPE vs HoPE, SSMax (`use_ssmax`), SSOG campo fijo (`lookat: false`) y SSOG+BitNet, loops duplicados, activaciones de FFN, etc.
- **Tareas**: `mlm`/encoder y `causal_lm`/decoder; visión (`classification`, `patch_prediction`, `segmentation`) — cada tarea de visión tiene además una variante `ssog_attn` (`cls_token: false` + `pooling_mode: gap`; la rejilla 2×2 se deriva de la imagen). El mezclador `conformer_attn` (Conformer, arXiv:2005.08100) se barre en el encoder (pares, singles) y en el decoder (`causal_lm`, padding causal de la convolución) y dentro del ViT (`configs/examples/vit_conformer.yaml`).
- **Audio**: `frankenstein_audio` con `conformer_attn` en `audio_asr_ctc`, `audio_asr_transducer` (RNN-T), `audio_classification` y `audio_eeg_classification` (`feature: raw`, 8 canales). Con `--no-realonall` el harness limpia `audio_dataset.dataset_name`/`dataset_dir` en el YAML temporal de cada run para forzar `DummyAudioDataset` (sin red). Con `--realonall` (por defecto) los casos ASR usan `astroza/chilean-jokes-festival-de-vina` y la clasificación usa `danielquillanroxas/EMOTION_VOICES` (7 clases); el caso EEG `raw` sigue siendo sintético. La decodificación de audio real con la versión actual de `datasets` requiere `torchcodec` instalado en el entorno.
- **Deploy / inferencia / export / GGUF**: **cada** combo ejecuta su pipeline completo en el mismo worker y de forma secuencial — entrenamiento → deploy `standard` → evaluación de inferencia → deploy `quantized` → evaluación de inferencia → export a HuggingFace Transformers → persistencia de resultados — antes de pasar al siguiente combo. Los checkpoints de audio deben cargar y validar en ambos formatos; su export a Transformers debe fallar con el mensaje documentado (`frankenstein_audio` no exportable) y se puntúa como `OK` cuando lo hace. El export GGUF BitNet se ejecuta en el combo dedicado (`bitnet_std_attn_for_gguf`). Las comparaciones inferencia-vs-real se guardan en `inference.csv`.

## Cómo ejecutar

Desde la raíz del repo, con el entorno uv del proyecto:

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py
```

Ejecución rápida de solo 3 optimizadores:

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py --category opt --limit 3
```

Solo atenciones de a pares:

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py --category attn
```

Saltar el barrido más lento de atenciones de a pares:

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py --skip-attn-pairs
```

Solo las codificaciones posicionales a nivel de modelo:

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py --category pe
```

Solo las `pos_embedding_type` del ViT (3 tareas × 12 PEs):

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py --category vision_pe
```

Solo las tareas de audio (`asr` CTC + RNN-T, clasificación, EEG `raw`) y su
smoke de deploy:

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py --category audio
```

### Datos reales por defecto (`--realonall`)

`--realonall` está **activo por defecto**: cada categoría (pares de atenciones, singles, optimizadores, normas, codificaciones posicionales, transversales, tareas, batch size, tokenizer, visión, audio y realdata) se reescribe para entrenar con datos reales del Hub, reutilizando los mismos datasets de `--category realdata`:

| Modalidad | Dataset / tokenizer |
|---|---|
| Texto | `adleme94/borges` (columna `poem`) + tokenizer `dccuchile/bert-base-spanish-wwm-cased` |
| Visión — classification / patch_prediction | `as-cle-bert/breastcanc-ultrasound-class` (2 clases, 64×64) |
| Visión — segmentation | `SerdarHelli/SegmentationOfTeethPanoramicXRayImages` (máscara en `label`, 2 clases, 64×64) |
| Audio — ASR | `astroza/chilean-jokes-festival-de-vina` |
| Audio — classification | `danielquillanroxas/EMOTION_VOICES` (7 clases) |
| Audio — EEG `raw` | `DummyAudioDataset` (sigue sintético; no hay corpus EEG real) |
| Tokenizer | `train_from_dataset` entrena sobre un parquet real materializado desde `adleme94/borges` |

Los datasets se descargan la primera vez (caché de `datasets`), así que conviene un `--timeout` generoso (p. ej. `--timeout 1800`). Para volver al camino totalmente offline:

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py --no-realonall
```

Número de muestras reales comparadas por artefacto desplegado:

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py --infer-samples 8
```

### Retención de artefactos (`--save-to`)

Por defecto se guarda todo en `full_tests/tmp/`:

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py --save-to disk
```

- **`disk`** (por defecto): se conservan los directorios de run, los checkpoints, los modelos desplegados y los datos bajo `full_tests/tmp/`.
- **`delete`**: se ejecuta igual en disco, pero al terminar el pipeline de cada combo (entrenamiento + deploy + inferencia + export) se borran su directorio de run, sus modelos y sus datos; solo sobreviven los resultados (`training_metrics/*.csv`, `inference.csv`, `param_metrics.json`, `results.json|md`). Al final del barrido se elimina todo el árbol de scratch.
- **`memory`**: el directorio de run y los modelos desplegados viven en un tmpfs en RAM (`/dev/shm/frankenstein-full-tests-<pid>`); ningún artefacto del harness toca el disco. Los resultados siguen escribiéndose en `full_tests/tmp/results/`. El tmpfs se elimina al terminar. Requiere `/dev/shm` escribible y es incompatible con `--resume`; la caché de `datasets` de HuggingFace (fuera del control del harness) no se redirige.

```bash
# Solo resultados, borrando modelos y datos al terminar cada combo
uv run --extra cu118 --extra train python full_tests/run_e2e.py --save-to delete

# Modelo y deploy solo en RAM
uv run --extra cu118 --extra train python full_tests/run_e2e.py --save-to memory
```

### Evaluación con partición de test (`--test-percentage`)

Por defecto se reserva el **20%** de los datos de cada combo antes de entrenar; al terminar se evalúa el modelo final sobre esa partición reservada y la precisión se escribe en `results.json` y en la columna `test_acc` de `results.md`:

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py --test-percentage 20
```

El valor se escribe en `training.test_percentage`; el entrenador guarda la pérdida y precisión reservadas en `test_metrics.json` junto al CSV de métricas del run. Aplica a MLM, causal LM, clasificación de texto, las tres tareas de visión, ASR (donde la precisión reportada es la tasa de error de caracteres, `metric: cer`) y clasificación de audio. SBERT gestiona su propia partición de validación (`sbert.max_eval_samples`) y reporta `n/a`. Usar `0` desactiva la partición y restaura el comportamiento sobre todos los datos:

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py --test-percentage 0
```

Nótese que la partición reduce los ejemplos de entrenamiento (~20% menos), por lo que el número de pasos del optimizador baja en la misma proporción.

### Selección de dispositivo

Por defecto todo corre en `cpu`. Para ejecutar los tests en otro dispositivo (entrenamiento, deploy e inferencia):

```bash
# En GPU
uv run --extra cu118 --extra train python full_tests/run_e2e.py --device cuda

# En Apple Silicon (Metal)
uv run --extra cu118 --extra train python full_tests/run_e2e.py --device mps

# Dejar que el CLI resuelva el dispositivo automáticamente
uv run --extra cu118 --extra train python full_tests/run_e2e.py --device auto
```

Valores válidos: `auto`, `cpu`, `cuda`, `mps` (por defecto `cpu`).

### Guarda térmica de GPU

La guarda térmica se configura **en el YAML** (`training.gpu_temp_guard_enabled` y
los umbrales `training.gpu_temp_*`) — el CLI ya no acepta overrides térmicos.
El harness **la desactiva por defecto** (escribe `gpu_temp_guard_enabled: false`
en el YAML de cada run). Para activarla durante el entrenamiento en GPU y ajustar
sus umbrales:

```bash
uv run --extra cu118 --extra train python full_tests/run_e2e.py --device cuda \
  --gpu-temp-guard \
  --gpu-temp-pause-threshold-c 80 \
  --gpu-temp-resume-threshold-c 70 \
  --gpu-temp-critical-threshold-c 90 \
  --gpu-temp-poll-interval-seconds 5 \
  --gpu-temp-checkpoint-grace-seconds 30
```

Cada umbral del harness es opcional; si se omite, no se escribe la clave y rige el
default del schema. Solo tienen efecto cuando `--gpu-temp-guard` está activo.

## Artefactos de resultados

Todo se escribe bajo `full_tests/tmp/results/` (git-ignorado):

- **`results.json` / `results.md`** — tabla de estados por run (entrenamiento, deploy, eval, export). Cada fila de entrenamiento incluye `test_accuracy` (en JSON) y la columna `test_acc` (en Markdown) con la precisión medida sobre la partición de test reservada; `n/a` cuando no aplica (SBERT, o `--test-percentage 0`).
- **`inference.csv`** — una fila por comparación inferencia-vs-real de cada artefacto desplegado (formato `standard` y `quantized`). Columnas: `timestamp, combo_id, model_class, task, deploy_format, sample_id, input_excerpt, predicted, real, metric_name, metric_value, status, notes`. Métricas por tarea:
  - `mlm` → `mlm_top1` (acierto top-1; `notes` incluye `top5_hit`), enmascarando un token real del corpus.
  - `causal_lm` → `causal_next_token` (acierto del siguiente token; `notes` incluye la accuracy por token).
  - `text_classification` / ViT `classification` / `audio_classification` → accuracy etiqueta predicha vs real.
  - `segmentation`, `patch_prediction`, `asr` → fila `SKIPPED_NO_INFER` (no hay métrica de etiqueta equivalente).
  - `SKIPPED_NO_DATASET` cuando no hay dataset/tokenizer HF (p. ej. con `--no-realonall`).
- **`param_metrics.json`** — `{modelo: {bloque: nparams, ..., "total": n}}` con las 10 categorías canónicas de `src/utils/param_categories.py` (`embeddings, pos, attention, ffn, norm, head, mhc, router, residual, other`). Se calcula con `build_model` sobre `torch.device("meta")` desde el `config.yaml` de cada run.
- **`training_metrics/<modelo>.csv`** — copia lista para procesar del `training_metrics.csv` de cada run, renombrada con el id del modelo.

## Filosofía de los resultados

- `OK`: el entrenamiento terminó sin error.
- `GRAD_EXPLODED`: el proceso falló con una señal numérica inequívoca (NaN/Inf, pérdida no finita o gradientes explotados). **Esto es tolerado y esperado** para algunas combinaciones; se registra pero no detiene la ejecución. La detección es por palabra completa y solo sobre líneas no indentadas, así que un `RuntimeError` cualquiera (p. ej. un desajuste de formas) o los banners `INFO` / `Gradient Accumulation` ya **no** se confunden con una explosión de gradiente.
- `FAILED` / `TIMEOUT`: fallo inesperado. El script termina con exit code != 0 si hay alguno.

## Archivos

- `_helpers.py` — entrenamiento del tokenizador SPM, generación del parquet sintético y del corpus real, runner de subprocesos, helpers de deploy/infer/export/gguf, persistencia de resultados y artefactos (`inference.csv`, `param_metrics.json`, `training_metrics/`), lectura de `test_metrics.json` y limpieza por combo (`--save-to delete|memory`).
- `_infer_eval.py` — subproceso de evaluación inferencia-vs-real (texto, ViT y audio).
- `run_e2e.py` — bucles for que generan todas las combinaciones, aplican `--realonall` y ejecutan el pipeline completo de cada combo (train → deploy → infer → export) en un mismo worker, con `--save-to` y `--test-percentage`.
- `tmp/` — tokenizador, datos, checkpoints por run, resultados, deploys. **Git-ignorado.** Con `--save-to memory` este árbol se ubica en `/dev/shm` (salvo `tmp/results/`).

## Notas importantes

- `max_samples` y `dataset_batch_size` son pequeños para que el corpus produzca unos pocos batches.
- Cada run tiene su propio `cwd` (`full_tests/tmp/runs/<id>/`) para aislar los checkpoints.
- Con `--no-realonall` el tokenizador se entrena una sola vez y se copia al CWD de cada run (`es_redpajama_<vocab_size>.model`) para evitar que el loader legacy intente bajar 100 GB de datos.
- La decodificación de audio real requiere `torchcodec` en el entorno con la versión actual de `datasets`; si falta, los combos de audio real fallan con `ImportError` y el harness continúa.
- Se fija semilla vía un `sitecustomize.py` generado en `tmp/` y añadido a `PYTHONPATH`, sembrando `torch`, `numpy` y `random` en cada subproceso sin modificar el proyecto.
