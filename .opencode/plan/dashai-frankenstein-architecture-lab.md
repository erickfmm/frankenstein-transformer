# Plan: dashai-frankenstein — Frankenstein Architecture Lab (estratégico a largo plazo)

**Estado:** Propuesta (aprobada por el autor: ecosistema completo, Architecture Designer visual primero, frontend híbrido).
**Scope:** `dashai-frankenstein/` (plugin) + `frankenstein-transformer` (base) + web GitHub Pages + PR upstream a DashAI.
**Decisión de branding:** estandarizar a `frankenstein-transformer` (corregir typo `frankestein` en URLs/UI; mantener URL vieja como redirect de GitHub Pages).

## Visión y posicionamiento

Transformar `dashai-frankenstein` de un **Model plugin** (el usuario pega JSON y entrena) en un **Frankenstein Architecture Lab**: laboratorio visual para diseñar, combinar, entrenar y comparar arquitecturas híbridas Transformer/NLP. El texto estratégico lo deja claro: el espacio que ninguna de Orange/KNIME/Weka/LlamaFactory/Unsloth/Oumi ocupa es *"dibujar una arquitectura que probablemente no existe, entrenarla y comparar si funciona"*.

**Pitch ancla:**
> *Frankenstein Architecture Lab is a visual research environment for designing, training and comparing custom neural architectures by composing attention, state-space, recurrent, sparse and hybrid mixing mechanisms.*

**No competir** en fine-tuning de LLM (LlamaFactory/Unsloth) ni en ML visual genérico (Orange/KNIME). El diferenciador es **`layer_pattern` + los 36 mixers + 23 optimizadores como objeto manipulable**.

## Arquitectura de dos capas (decisión estratégica 1)

```
                    ┌──────────────────────────┐
                    │          dashAI          │  Workbench host: datasets,
                    │  datasets · experiments │  notebooks, runs, métricas,
                    │  reproducibility · UI    │  comparación, reproducibilidad
                    └────────────┬─────────────┘
                                 │ plugin (entry points)
                                 ▼
             ┌───────────────────────────────────┐
             │   Frankenstein Architecture Lab   │  Plugin: componentes Model/Task
             │   (dashai-frankenstein)            │  + motor de analítica + export
             └───────────────┬───────────────────┘
                             ▼
                Frankenstein Transformer (PyPI)  Schema source of truth + engine API
                             ▼
                         PyTorch

  GitHub Pages web  ──"Open in dashAI"──►  dashAI  ◄──"Export to YAML"──  web
  (Architecture Designer visual, drag&drop, tarjetas educativas, diagrama)
```

La web (repo `erickfmm.github.io`) y el plugin son complementarios:
- **Web** = demo/educación/diseño rápido, exporta YAML/JSON, botón "Open in dashAI".
- **dashAI plugin** = experimentación reproducible (train/evaluate/compare).

## Decisiones estratégicas

1. **El Architecture Lab vive en la web independiente** (drag&drop, widgets custom) porque DashAI no soporta widgets custom en plugins. Un **PR upstream a DashAI** se abre en paralelo para multiline textarea / custom widgets (largo plazo, fuera de nuestro control).
2. **El plugin dashAI se refuerza**: nuevos componentes de **analítica** (param count, FLOPs, VRAM) + **experiment manager** + **comparison**, además de los Model/Task actuales.
3. **Frankenstein base** expone un **engine de analítica** (sin instanciar el modelo: estimación de params/FLOPs/VRAM por mixer) y **metadatos de mixers** (familia, complejidad O(n), descripción, referencia) — auditoría §7.7 ("field-metadata export") generalizada a mixers.
4. **Branding**: estandarizar a `frankenstein-transformer` (corregir `frankestein` typo en URLs/UI; mantener la URL vieja como redirect de GitHub Pages).
5. **No reemplazar el trainer** ni competir con HF Trainer; el plugin sigue usando `src.engine` in-process.

## Fases (roadmap a largo plazo)

### Fase 0 — Cierre de fundaciones (pre-requisito)
Objetivo: el plugin actual corre contra una instancia viva de DashAI y Frankenstein `1.1.0` está en PyPI.
- [ ] Publicar `frankenstein-transformer==1.1.0` en PyPI (audit §7.6; único item Phase 0 pendiente).
- [ ] Publicar `dashai-frankenstein==0.2.0` en PyPI (o editable-install para QA).
- [ ] Smoke test end-to-end contra DashAI real: los 5 componentes aparecen, MLM entrena en una tarea de clasificación de texto, métricas fluyen, save/load/predict funcionan.
- [ ] Corregir typo `frankestein` → `frankenstein` en URLs/UI (mantener redirect).
- [ ] Abrir issue/PR upstream en DashAI para **multiline textarea + custom widgets** (híbrido).

**Salida:** plugin v0.2.0 funcional en DashAI viva; upstream issue abierto.

### Fase 1 — Architecture Designer visual (prueba de valor, elegida primero)
Objetivo: la web se convierte en un laboratorio visual de diseño de arquitecturas.
- [ ] **Repositorio web** (`erickfmm.github.io`): panel de 3 columnas — Components / Architecture / Inspector.
- [ ] **Library de mixers** drag&drop: 36 mixers agrupados por familia (dense/gated/latent/sparse/recurrent), cada uno con tarjeta educativa (tipo, complejidad O(n), ventajas/desventajas, referencia paper).
- [ ] **Diagrama interactivo**: Embedding → [mixers...] → LM Head; click en un nodo abre el Inspector (tipo, complejidad, params, referencia).
- [ ] **Generación de YAML** + **JSON de una línea** (para el campo passthrough del plugin) + botón **"Open in dashAI"** (deep link / copia comando).
- [ ] **Validación client-side** contra el schema Frankenstein (versión JS de `src/schema.yaml`).
- [ ] **Branding Frankenstein** en toda la web.

**Dependencia de Frankenstein base:**
- [ ] Exportar **metadatos de mixers** como JSON (`src/model/attention/registry.json` generado): `name, family, complexity, description{en,es}, reference, training_ok`. Generado desde `src/schema/_model/_dims.yaml` (enum `layer_pattern`) + docstrings.

**Salida:** web v2 con Architecture Designer; usuarios diseñan arquitecturas híbridas y las exportan a dashAI.

### Fase 2 — Model Stats en vivo (params/FLOPs/VRAM)
Objetivo: el diseñador muestra el coste de cada arquitectura mientras se construye.
- [ ] **Engine de analítica en Frankenstein base** (`src/analytic.py` o `src/engine.py`): `estimate_params(config_dict)`, `estimate_flops(config_dict, seq_len)`, `estimate_vram(config_dict, batch, seq_len)`. Sin instanciar el modelo (modo `meta` de torch o cálculo simbólico por mixer).
- [ ] Plugin: exponer los mismos vía un componente de analítica o un endpoint que la web consume.
- [ ] Web: panel "Model Statistics" reactivo (Parameters, Trainable, Layers, Context, FLOPs estimados, VRAM estimado) que se actualiza al cambiar `layer_pattern` / dims.
- [ ] Tarjetas por mixer con params/FLOPs estimados individuales.

**Salida:** el usuario entiende experimentalmente el coste de cada arquitectura antes de entrenar.

### Fase 3 — Experiment Manager + Comparison (matriz arquitectura × optimizador × dataset)
Objetivo: dashAI deja de ser solo "train" y pasa a ser "experiment & compare".
- [ ] Plugin: **Experiment Manager** — dispara una matriz de runs (arquitectura × optimizador × dataset) reutilizando el job queue de DashAI.
- [ ] Plugin: **Comparison component** — tabla lado a lado (params, FLOPs, val loss, perplexity, attention/mamba/retnet layer count) entre runs.
- [ ] Web: botón "⚗️ Compare" (diseño A vs B, side-by-side stats + diagrama).
- [ ] Web: sección "Experiment" (loss curve, val loss, perplexity por run) consumiendo métricas de dashAI.
- [ ] Reproducibilidad: cada experimento registra config YAML + dataset split + seed + métricas (ya provisto por dashAI).

**Salida:** Frankenstein = herramienta para experimentos reproducibles de arquitectura.

### Fase 4 — Curated native fields (v2 schema, UX en dashAI)
Objetivo: reemplazar el passthrough JSON por campos nativos para los ~15 knobs de alto impacto, sin perder el schema como source of truth.
- [ ] Audit §7.7: exportar metadatos de campos (`src/schema/_field_metadata.json` generado en build).
- [ ] Plugin: generar campos pydantic nativos para `model_class, hidden_size, num_layers, num_heads, layer_pattern, task, optimizer_class, ffn_activation, norm_type, use_moe, use_bitnet, ...`.
- [ ] Serialización al mismo YAML Frankenstein bajo el capó.
- [ ] No traducir los 151+38 fields (frágil ante schema evolution).

**Salida:** UX de form nativo en dashAI para los knobs principales, YAML sigue siendo source of truth.

### Fase 5 — Ecosistema y adopción
- [ ] Docs/specs: `docs/specs/architecture-lab.md` (arquitectura del lab), `docs/specs/web-bridge.md` (puente web↔dashAI).
- [ ] Paper / blog: "Frankenstein Architecture Lab: visual research environment for hybrid transformers".
- [ ] Plantillas de experimentos reproducibles (matrices arquitectura × optimizador predefinidas).
- [ ] Integración con datasets de dashAI (dataset → Architecture Lab → train → evaluate → compare).
- [ ] Marketing: charlas técnicas con el diagrama interactivo como material educativo.

## Reparto de trabajo por repositorio

| Repositorio | Trabajo |
|---|---|
| `frankenstein-transformer` (base) | Engine de analítica (params/FLOPs/VRAM), metadatos de mixers exportados, field metadata export (§7.7), publicar `1.1.0`, docs/specs del Architecture Lab. |
| `dashai-frankenstein` (plugin) | Componentes de analítica + Experiment Manager + Comparison; curated native fields (v2); exponer analítica vía endpoint; publicar en PyPI. |
| `erickfmm.github.io` (web) | Architecture Designer visual (drag&drop, tarjetas, diagrama), Model Stats, Compare UI, "Open in dashAI", "Export to YAML". |
| DashAI (upstream) | PR/issue para multiline textarea + custom widgets (largo plazo, fuera de nuestro control). |

## Hard constraints (se respetan siempre)

1. **Schema es source of truth** (`src/schema/_*.yaml`): cada config field debe existir en el schema; la web valida client-side contra el mismo schema.
2. **Cross-component compatibility** (divisibilidad hidden_size/num_heads, num_kv_heads|num_heads, vocab match, task↔optimizer/sbert, frankensteindecoder→mode:decoder).
3. **BitNet defaults True**; bitnet_routers requiere use_bitnet.
4. **fasa_attn/sparge_attn eval-only** (no en layer_pattern de training).
5. **Optimizador prefijado** `<class>-<group>_<param>`.
6. **training.task requerido** (mlm/sbert/vision/causal).
7. **Nuevos mixers/optimizadores requieren example YAML** (CI smoke test).

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| PR upstream a DashAI no aceptado | Plan híbrido: la web independiente es el Architecture Lab; el PR es bonus, no bloquea. |
| Analítica de FLOPs/VRAM imprecisa por mixer | Modo `meta` de torch + fórmulas por familia; documentar como "estimación"; comparar vs real en smoke tests. |
| Schema drift entre web y plugin | Web valida contra el mismo schema exportado; el plugin revalida server-side (ya lo hace). |
| Sobrecarga de mantenimiento 3 repos | Fases escalonadas; la web y el plugin comparten el schema exportado (single source of truth). |
| Competir accidentalmente con Unsloth/LlamaFactory | Mantener foco en "diseño de arquitectura", no en fine-tuning masivo. |

## Métricas de éxito

- Un usuario diseña una arquitectura híbrida en la web, exporta a dashAI, entrena, compara — todo sin escribir YAML a mano.
- La matriz arquitectura × optimizador produce runs reproducibles comparables en dashAI.
- El diagrama interactivo es usable como material educativo en una charla.

## Estado actual del plugin (referencia)

`dashai-frankenstein` v0.2.0 ya implementado con 5 entry points (patrón passthrough JSON, validación contra schema + config loader):
- `FrankensteinMLMModel` → `TextClassificationTask`
- `FrankensteinDecoderModel` → `TextToTextGenerationTask`
- `FrankensteinViTClassifier` → `ImageClassificationTask`
- `FrankensteinViTSegmenter` → `SegmentationTask` (nuevo, provisto por el plugin)
- `SegmentationTask` → `BaseTask`

Audit completo en `docs/dashai-plugin-audit.md` (Phase 0–3 marcadas, falta publicar Frankenstein `1.1.0` a PyPI y correr contra instancia viva de DashAI). Este plan extiende esa base hacia el Architecture Lab.