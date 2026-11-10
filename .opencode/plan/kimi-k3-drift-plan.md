# Kimi K3 — Drift de `configs/kimi_k3.yaml` y plan de implementación

> **Fecha:** 2026-11-10
> **Estado:** análisis completo; implementación no iniciada.
> **Fuentes contrastadas:**
> - Paper: *Kimi K3: Open Frontier Intelligence*, arXiv:2607.24653 (Tabla 1; §2.1 Hybrid Attention; §2.1.1 KDA; §2.1.2 Gated MLA; §2.2 AttnRes; §2.3 Stable LatentMoE; §2.3.1 Norm; §2.3.2 SiTU-GLU; §2.3.3 Quantile Balancing; §2.5 Per-Head Muon; §3.3 Training Recipe; §3.4 Long-Context).
> - Pesos oficiales: `config.json` de `huggingface.co/moonshotai/Kimi-K3` (fuente autoritativa de dims exactos).
> - Código/schema de este repo (`src/schema/**`, `src/model/**`).

---

## 0. Veredicto

El YAML captura **exactamente** la anatomía macro de K3 (patrón 3:1 KDA:MLA con MLA final, LatentMoE 3584 + RMSNorm pre-up-proj, 896/16, sigmoid + scale 1.0, 2 shared, 1 capa densa, vocab 163840). Los drifts se concentran en: **AttnRes ausente** (y sin soporte en el decoder), **SiTU-GLU y Quantile Balancing como aproximaciones documentadas**, **`num_heads: 56` vs 96** (bloqueado por el invariante `hidden % num_heads == 0`), **MLA con RoPE en vez de NoPE**, **AdamW en vez de Per-Head Muon**, y una **KDA "estática" estilo Kimi Linear** que no incluye los cambios de K3 (decay acotado input-dependiente, ShortConv, RMSNorm por cabeza).

---

## 1. Verificado correcto (no requiere cambios)

| Clave YAML | Valor | K3 real | Fuente |
|---|---|---|---|
| `dims.vocab_size` | 163840 | 163840 | Tabla 1 ("160K"), HF |
| `dims.hidden_size` | 7168 | 7,168 | Tabla 1, HF |
| `dims.num_layers` | 93 | 93 | Tabla 1 |
| `dims.num_loops` | 1 | 1 (sin looped depth) | — |
| `layer_pattern` | 69 `kda` + 24 `mla`, MLA extra al final | 69 KDA + 24 MLA, "additional Gated MLA at the end" | §2.1; HF `full_attn_layers=[4,8,…,92,93]` — **match exacto** |
| `attention.mla.latent_rank` | 512 | `kv_lora_rank: 512` | HF |
| `moe.num_experts` / `top_k` | 896 / 16 | 896 / 16 | Tabla 1, abstract |
| `moe.expert_hidden_size` | 3072 | `moe_intermediate_size: 3072` | Tabla 1 |
| `moe.latent.dim` | 3584 | `routed_expert_hidden_size: 3584` (0.5×) | Tabla 1, §2.3 |
| `moe.latent.norm_before_up_proj` | true | `latent_moe_use_norm: true` | §2.3.1, HF |
| `moe.shared.count` | 2 | `num_shared_experts: 2` (full-width, Ns=2) | §2.3, HF |
| `moe.router.scoring` / `routed_scaling_factor` / `renormalize` (default true) | sigmoid / 1.0 / true | `sigmoid` / `1.0` / `moe_renormalize: true` | §2.3.3, HF |
| `moe.layout.dense_layers` / `every_n_layers` (default 1) | 1 / 1 | `first_k_dense_replace: 1` / `moe_layer_freq: 1` | Tabla 1, HF |
| `norm.type: rms_norm` | ✓ | `norm_type: rmsnorm` | HF |
| `bitnet.enabled: false` | ✓ (explícito; el default del repo es true) | — | — |
| `dropout: 0.0` | ✓ | — | — |
| `scheduler_type: cosine` | ✓ | "cosine learning rate schedule" | §3.3 |
| Approx `centered_sign` y `activation_clamp: 4.0` | documentadas como desviación en comentarios del propio YAML | — | ver B2/B1 |

---

## 2. Drifts corregibles solo con YAML (el schema ya lo soporta)

**Esfuerzo total: ~0.5 día** (editar + correr los gates de schema/presets).

| # | Drift | YAML actual | K3 real | Fix |
|---|---|---|---|---|
| A1 | MLA con RoPE ≠ **NoPE** | `use_pe.mla` default `true` → aplica RoPE a Q/K descomprimidos | NoPE en **todas** las capas MLA (§2.1.2, §3.4); extrapolación a 1M sin retuning RoPE | `positional_encoding: {use_pe: {mla: false}}` (`kda` ya defaultea `false` ✓) |
| A2 | `training.task: mlm` ≠ causal LM | `mlm` | next-token prediction multimodal (§3.3); con `frankenstein_text_decoder` la tarea fiel es `causal_lm` (regla válida en `config_loader`) | `training.task: causal_lm` |
| A3 | Warmup 10% ≠ **1%** | `scheduler_warmup_ratio: 0.1` | "cosine schedule with a **1% linear warmup**" (§3.3) | `scheduler_warmup_ratio: 0.01` |
| A4 | Capa densa 3072 ≠ 33792 | `ffn.hidden_size: 3072` | `intermediate_size: 33792` (HF) para la capa 1 densa | `ffn.hidden_size: 33792`, o dejar 3072 como compromiso de memoria **documentado en comentario** |
| A5 | AdamW ≠ **Per-Head Muon** | `adamw-*`, lr 1e-6…5e-6 (placeholder), wd 0.001–0.01 | Muon per-head para matrices Q/K/V + weight-clipping K2, **wd 0.1** (§3.3, §2.5); peak LR no publicada (scaling-law) | mientras llega P1.4: cambiar a familia `muon-*` existente con `wd 0.1` y lr razonable (p. ej. 0.02 default de `muon.py`); borrar betas/eps de adamw |
| A6 | `max_length: 4096` vs curriculum 8K→64K→256K→1M (§3.4) | 4096 | 8k arranque | mantener 4096 (single-device) con **comentario** que documente el curriculum del paper |
| A7 | `rms_norm_eps: 1e-5` (HF) | no seteado (default del repo) | 1e-5 | verificar si existe clave de eps para la norm de bloque; si existe, setear 1e-5 (si no, entra en P2.10) |

**Notas:**
- A2 implica usar el pipeline `causal_lm` (dataset/tokenizer ya soportados por `config_loader`). El resto del bloque `training` (batch, acumulación, `scheduler_total_steps`) son placeholders de experimento local, no de fidelidad K3 — dejarlos fuera del alcance.
- `num_heads: 56` **no** es corregible por YAML: 7168 % 96 ≠ 0 y el invariante `hidden_size % num_heads == 0` lo impide (ver P2.5).

---

## 3. Faltantes de implementación — EN FOCO

Escala de esfuerzo: **S** ≤ 4h · **M** 4–12h · **L** 12–32h · **XL** > 32h.

### P1.1 — Activación `situ` (SiTU-GLU) · **S–M (4–8h)**

**Gap (§2.3.2 + Fig. 4):** K3 usa `hidden_act: "situ"` con softcap en **ambas** ramas del GLU:

```
gate(u) = β₁·tanh(u/β₁) · σ(u)        # u = W_g x   (Swish con factor lineal softcap-eado)
up(v)   = β₂·tanh(v/β₂)               # v = W_u x
|f| ≤ β₁·β₂ = 100                     # β₁ = 4, β₂ = 25 (HF: activation_situ_beta / _linear_beta)
```

El YAML aproxima con `swiglu` + `moe.experts.activation_clamp: 4.0`: clamp **duro** (no tanh) y **solo en la puerta** (falta el tope de la rama up).

**Cambios (receta "Activation" de `docs/architecture.md`):**
1. `src/model/activation_function/glu.py` — nuevo kind `"situ"` en `GatedFFN`/builder: gate branch `softcap(silu-input)*sigmoid` con β₁, up branch softcap β₂. Los betas llegan como kwargs del builder (como hoy llega `clamp_value`).
2. `src/model/activation_function/factory.py` — añadir `"situ"` a `GLU_VARIANTS` (y por ende a `ALL_ACTIVATIONS`); NO a `get_activation` (es unidad GLU, no elementwise).
3. `src/schema/_model/_ffn.yaml` — enum de activación + sub-bloque `activation_config` con `situ_beta_gate` (default 4.0) y `situ_beta_up` (default 25.0), descripciones bilingües y referencias (paper: arxiv:2607.24653).
4. `src/utils/config_flatten.py` / `src/model/config.py` — campos planos `ffn_activation_situ_beta_gate/up` + validación (betas > 0) + paso al builder de `GatedFFN` (dense y expertos vía `expert_factory`).
5. `configs/examples/activation_situ.yaml` + actualizar `configs/kimi_k3.yaml` (`ffn.activation: situ`; `moe.experts.activation_clamp` → 0).
6. Tests: `tests/test_activation_functions.py` (nuevo caso: valores del softcap, cota β₁β₂, cercanía a SwiGLU cerca del origen), `tests/test_schema_consistency.py` (enum ↔ dataclass ↔ defaults), `tests/test_yaml_examples.py` (auto-glob).
7. `docs/architecture.md` §7 + tabla de activaciones (el enum es fuente de verdad: 43 → 44 valores).

**Riesgos:** la cuenta "43 activaciones" aparece en docs/tests; el clamp de expertos hoy se aplica dentro de `GatedFFN.forward` — con `situ` el clamp debe quedar deshabilitado (softcap ya acota).

### P1.2 — `bias_update: quantile` (Quantile Balancing) · **M (6–10h)**

**Gap (§2.3.3, Eq. 12–14 + Fig. 5):** router sigmoid, bias solo en la selección (no en los pesos ✓ ya así en el repo). QB reemplaza la regla de paso fijo:

```
s_i = sigmoid(W_r x_i)                     # scores del batch (m tokens, n expertos)
selección sobre s_i + b^t con Top-(k+1)     # el (k+1)-ésimo es el cutoff α_i
bias_j^{t+1} = Quantil_{1−k/n}({ s_ij − α_i }_{i=1..m})   # (q+1)-ésimo mayor margen, q = m·k/n
```

- Causal: el bias derivo de un forward **no se usa en ese mismo batch** (efecto al paso siguiente) — igual que `sign`/`centered_sign` hoy.
- Inferencia: bias congelado (buffer, no parámetro entrenable).
- El paper usa histogramas solo porque el cuantil abarca el batch global multi-rank; en runtime single-device el **cuantil exacto** es viable (los histogramas quedan out of scope).

**Cambios (receta MoE — el bloque vive en `_moe.yaml` + `src/model/moe/`):**
1. `src/model/moe/router.py` — en `forward`, retener `last_scores` (sigmoid cruda) y computar el cutoff αᵢ del Top-(k+1) sobre `s + b`; extender `update_expert_bias` con la rama `quantile` (torch.quantile / kthvalue por experto sobre los márgenes `s_ij − α_i`; definir manejo de empates y de `q < 1`, i.e. batch chico). Quitar dependencia de `bias_update_rate` para esta rama.
2. `src/schema/_model/_moe.yaml` — enum `balancing.bias_update: [none, sign, centered_sign, quantile]` + descripciones bilingües (referencia paper: arxiv:2607.24653; nota: histogramas cross-rank out of scope).
3. `src/model/config.py` — validar `quantile` igual que `sign` (requiere `router.scoring ≠ softmax`).
4. `configs/kimi_k3.yaml` — `balancing: {aux_loss: none, bias_update: quantile}` (eliminar `bias_update_rate`).
5. Tests: suite MoE existente + caso nuevo: batch sintético con desbalance conocido → tras un update QB la carga del paso siguiente es exactamente `q` por experto (sin empates); empates y `m·k/n < 1`; bias congelado en `eval()`.

**Riesgos:** interacción con `gradient_accumulation_steps: 16` (¿update por micro-batch o por paso de optimizer?). El paper deriva el bias de "a single forward pass" (micro-batch); documentar la elección (por micro-batch) en el schema.

### P1.3 — AttnRes (Block, 8 bloques × 12 capas) en el decoder · **M (6–12h)**

**Gap (§2.2):** K3 usa Block AttnRes: pseudo-query aprendida `w_l`, keys `RMSNorm(f_i(h_i))` con el embedding como fuente 0, softmax sobre las N representaciones de bloque; **8 bloques de 12 capas** (93 = 7×12 + 9 → bloque final parcial; 9 fuentes contando el embedding). HF: `attn_res_block_size: 12`.

**Estado del repo:** Full/Block AttnRes **implementados** (`src/model/residuals/{full_attn_res,block_attn_res}.py`, schema `_residuals.yaml` con `block_attn.num_blocks`, validación en `config.py` L1671+, y cableado completo en `frankenstein_encoder.py`, `_vit.py`, `_audio.py`). `frankenstein_decoder.py` **no** construye el residual ni aplica el paso por capa (`self.residual(idx, x)`); solo menciona AttnRes para deshabilitar caché (L255).

**Cambios:**
1. `src/model/frankenstein_decoder.py` — portar del encoder (template: L220–334): construir `self.residual` vía `residuals/factory.py`; en `forward`: `register_state/reset_state/set_embedding` antes del stack; después de cada capa `x = self.residual(idx, x)` cuando `is_attn_res`; fallback cache+AttnRes → warning + `cache = None` (igual que encoder, D5/F2).
2. Interacciones: verificar Hyperloop/mHC en el decoder (la incompatibilidad mHC-hyperloop + AttnRes ya se valida en `config.py`; mHC está excluido de kimi_k3 así que no aplica).
3. `configs/kimi_k3.yaml` — `residuals: {type: block_attn, block_attn: {num_blocks: 8}}` (N=8 ↔ block size 12 con 93 capas; equivalente al `attn_res_block_size: 12` de HF).
4. Tests: `tests/test_residuals.py` con caso decoder (forward shape, init en cero de `w_l` ↔ equivalencia con residual standard en t=0, gradiente de las pseudo-queries), y caso decoder+KV-cache → espera warning y fallback a recompute.

**Riesgos:** el decoder forward tiene particiones begin/middle/end (Hyperloop) — replicar el hook en cada rama de ejecución. Inferencia incremental con AttnRes queda como F2 (paper usa online-softmax para el merge; no bloquea).

### P1.4 — Per-Head Muon (+ weight clipping K2) · **M–L (8–16h)**

**Gap (§2.5, §3.3):** K3 optimiza matrices con Muon, refinado a **variante per-head**: particionar los momentum de las proyecciones **Q, K, V** por cabeza y correr Newton–Schulz por bloque (equaliza la escala de update entre cabezas). Además usa el **weight clipping de Kimi K2** y **wd 0.1** global.

**Cambios (receta "Optimizer"):**
1. `src/model/optimizer/muon.py` — opción `per_head: bool` + `head_dim`/heurística: si el nombre del parámetro matchea proyecciones de atención (`q_proj`, `k_proj`, `v_proj`; en MLA `q_proj`/`dkv_proj`), reshapar el momentum `(out, in) → (heads, head_dim, in)` y orthogonalizar por cabeza. Alternativa: archivo nuevo `per_head_muon.py` si se prefiere familia separada.
2. Weight clipping K2: parámetro `clip_value` (clip elementwise post-update aplicado solo a matrices ortogonalizadas).
3. `src/model/optimizer/factory.py` — `OPTIMIZER_REGISTRY` + `OPTIMIZER_ALLOWED_KEYS` (`per_head`, `clip_value`, `ns_steps`, …).
4. `src/schema/_optimizer.yaml` — enum (si familia nueva) + bloque `by_optimizer` con prefijos `muon-<group>_<param>`; `src/schema/_training.yaml` — regla `if/then` de prefijos.
5. `configs/kimi_k3.yaml` — `optimizer_class: muon` con `muon-wd_other: 0.1`, `muon-lr_*` (mantener placeholders documentados; el paper no publica la peak LR).
6. Tests: `tests/test_optimizer_factory.py`, `test_optimizer_base.py` (paso per-head ≈ paso full cuando hay 1 cabeza; clipping acota la norma; fallback AdamW de no-matrices intacto). `configs/README.md`.

**Riesgos:** clasificar parámetros por nombre a través de ~40 mixers (los nombres de proyección varían); la semántica K3 aplica solo a Q/K/V de atención — documentar qué nombres matchea.

### P2.5 — Cabezas por mixer desacopladas de `hidden_size` · **L (12–20h)**

**Gap:** K3 usa **96 cabezas** en MLA (`num_attention_heads: 96`) **y** en KDA (`linear_attn_config.num_heads: 96, head_dim: 128`; 96×128 = 12288 > 7168, expansión propia). El repo fuerza `head_dim = hidden_size // num_heads` y el invariante `hidden_size % num_heads == 0` (7168 % 96 = 64 → 96 inexpresable; por eso el YAML usa 56).

**Cambios:**
1. `src/schema/_model/_attention_kda.yaml` (nuevo, sub-bloque `attention.kda`): `num_heads`, `head_dim` (null → global). Receta Mixer: `$ref` en `_model.yaml` + `_dims.yaml` enum/tablas.
2. `src/schema/_model/_attention_latent.yaml` — extender `attention.mla` con `num_heads`, `q_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `use_nope`, `use_output_gate` (estos 4 últimos se consumen en P2.7).
3. `src/utils/config_flatten.py` (`_ATTENTION_MIXER_RENAMES` / flatten de attention) + `src/model/config.py` — campos `kda_num_heads/kda_head_dim/mla_num_heads/...`; **relajar el invariante** solo cuando el mixer declara overrides propios (validación: si `kda ∈ layer_pattern` y `kda_num_heads` seteado, no exigir `hidden % kda_num_heads == 0`; idem MLA). Regla dura #2 de AGENTS/docs queda reformulada para mixers con overrides.
4. `src/model/attention/gated/kda.py` (+ `latent/mla.py` en P2.7) y `src/model/cache/capabilities.py` — `CacheSpec` de `kda` debe usar las cabezas del mixer (el estado del cache escala con ellas).
5. `configs/kimi_k3.yaml` — `dims.num_heads` queda como default de los mixers sin override; `attention.kda.num_heads: 96, head_dim: 128`; `attention.mla.num_heads: 96`.
6. Tests: `test_schema_attention_layers.py`, `test_schema_consistency.py`, `test_kv_cache_mixers.py` (parametriza kda → paridad con nuevas dims), ejemplo yaml.

**Riesgos:** el invariante es una regla dura con tests asociados; tocar `config.py` validate + docs. Cambia el conteo de parámetros del preset (96 cabezas → más parámetros de atención: acercar el total a los 2.78T/104B del paper).

### P2.6 — KDA "v2" (los cambios de K3, §2.1.1) · **L (16–32h)** (chunkwise aparte)

**Gaps** (contra `src/model/attention/gated/kda.py` actual):

| K3 (paper §2.1.1) | Repo actual |
|---|---|
| Decay **input-dependiente**: logits `z_t = W_lowrank x_t + b_α` (per-head bias) | `log_decay_base/delta` son **parámetros libres**, no dependen de x |
| Mapeo **acotado**: `g = g_min·σ(e^{A_h}·z)`, `g_min = −5`, `A_h` per-head (init 0) → retención `α > e⁻⁵ ≈ 6.7e−3` | `exp(base − softplus(delta))` no acotado (estilo Kimi Linear) |
| **ShortConv (kernel 4) + Swish** antes de las proyecciones Q/K/V (`short_conv_kernel_size: 4`) | sin convolución |
| RMSNorm **por cabeza** antes del gate de salida | `nn.LayerNorm(total_dim)` global |
| Gate de salida full-rank input-dependiente ✓ | `g_proj` full-rank silu — **coincide** ✓ |
| L2 norm q/k ✓, silu(v) ✓ | ✓ |
| Forma chunkwise (recurrente entre chunks, paralela dentro; los tiles diagonales en Tensor Core gracias al decay acotado) | bucle O(T) token a token en Python |

**Cambios:**
1. `src/model/attention/gated/kda.py`:
   - Proyección low-rank del logit de decay (`hidden → r → num_heads·head_dim`) + `b_α` per-head + `A_h` (log-scale por cabeza).
   - Mapeo `g_min·sigmoid(e^{A_h} z)` con `g_min` configurable (default −5); mantener la forma actual como modo legacy (`decay_mode: kimilinear | k3`).
   - ShortConv depthwise (kernel 4, groups=hidden) + silu antes de q/k/v.
   - Reemplazar LayerNorm por RMSNorm head-wise.
   - (Opcional, esfuerzo aparte **+L**): chunkwise reference en PyTorch puro con test de paridad contra el bucle secuencial — es la vía para eficiencia, no requerida para fidelidad funcional.
2. `src/schema/_model/_attention_kda.yaml` (creado en P2.5) — claves: `decay_mode`, `decay_min_log` (−5), `decay_low_rank`, `short_conv_kernel` (4), `use_full_rank_gate` (true).
3. `config_flatten.py` + `model/config.py` + `_DEFAULT_PAIRS` si aplica defaults; `_dims.yaml` tablas descriptivas de `kda` (ya existen → actualizar).
4. `configs/examples/attention_kda*.yaml`; `configs/kimi_k3.yaml` setea el modo K3.
5. Tests: paridad decay acotado vs legacy (con `g_min → −∞` debe converger al mapping actual), cota de retención `α > e⁻⁵`, estado de cache con conv (la conv corta añade estado al slot: extender `CacheSpec`), `test_kv_cache_mixers.py`.

**Riesgos:** la conv corta agrega estado (kernel−1 tokens) al cache incremental; el bucle secuencial con 12288-dim de estado × 96 cabezas × T es lento — aceptable en CPU para tests chicos, documentarlo.

### P2.7 — Gated MLA "DeepSeek-style" (NoPE + q_lora + output gate) · **L (16–24h)**

**Gap (§2.1.2 + HF):** el `mla.py` actual es el de Mehta et al. (arXiv:2506.09342) para modelos chicos (latent único, RoPE sobre Q/K descomprimidos). K3 usa MLA DeepSeek-style: `q_lora_rank: 1536` (queries low-rank), `qk_nope_head_dim: 128` + `qk_rope_head_dim: 64` (con `mla_use_nope: true` la parte rope queda sin usar), `v_head_dim: 128`, 96 cabezas, **gate de salida full-rank input-dependiente** (`mla_use_output_gate: true`), salida de atención en FP32 durante training (nota de kernel).

**Recomendación:** implementarlo como **mixer nuevo** (p. ej. `gmla`) siguiendo la receta Mixer completa, para no romper la semántica de `mla` existente; `configs/kimi_k3.yaml` pasa su `layer_pattern` de `mla` a `gmla`. Alternativa (más invasiva): flags dentro de `attention.mla` que cambian la parametrización.

**Cambios (ruta mixer nuevo):**
1. `src/model/attention/latent/gmla.py` — c_KV = W_DKV x (512); q = W_UQ(W_DQ x) (1536); k/v descomprimidos por cabeza (nope 128, v 128; rope 64 solo si `use_pe`); sin PE por default; `out = SDPA(...) ⊙ silu(W_g x)`; `make_cache_slot` → `LatentKVSlot` de c_KV (reutilizar hook `cached_latent_attention`, extensión para gate).
2. Receta Mixer completa: `latent/__init__.py` + `attention/__init__.py` → `hybrid_layer.py` registry (`"gmla"`); `_dims.yaml` enum + **ambas** tablas descriptivas; `_attention_latent.yaml` sub-bloque nuevo (o `_attention_gmla.yaml`); `_model.yaml` `$ref`; `config_flatten._ATTENTION_MIXER_RENAMES`; `model/config.py` campos; `capabilities.py` CacheSpec; `configs/examples/attention_gmla.yaml`; tests (`test_kv_cache_mixers` parametriza `IMPLEMENTED_MIXERS` → paridad prefill+decode).
3. `configs/kimi_k3.yaml` — `layer_pattern` con `gmla`, `attention.gmla.{latent_rank: 512, q_lora_rank: 1536, qk_nope_head_dim: 128, v_head_dim: 128, num_heads: 96}` + NoPE (no requiere `use_pe` — default false).

**Riesgos:** decode incremental con forma absorbida (W_UK absorbida en W_Q) es la vía eficiente de MLA; mínimo viable = expandir K/V por paso (correcto, más lento); documentar.

### P2.10 — Numérica menor (eps de norm, FP32) · **S (2–4h)**

- Verificar/agregar clave `eps` para la norm de bloque (`_norm.yaml` tiene eps solo para QK-norm) y setear `1e-5` en el preset (HF `rms_norm_eps: 1e-5`). Si hay que agregarla: `_norm.yaml` + factory de normas + `config.py` + `_DEFAULT_PAIRS` + `test_common_modules.py`.
- FP32 en la salida de atención durante training (§2.1.2): **nota de documentación solamente** (detalle de kernel; SDPA de torch ya acumula en fp32) — no planificar código.

---

## 4. Fuera de foco (documentado, no planeado)

### P3.8 — MTP (Multi-Token Prediction, 1 capa) · **XL (~40–60h)**

- **Paper:** Tabla 1 — K2 y K3 entrenan con **1 capa MTP**; los pesos liberados la traen desactivada (`num_nextn_predict_layers: 0`).
- **Si se hiciera:** módulo draft (capa extra que reusa embedding + head), `trainer.py` (loss MTP con pesos por profundidad), pipeline de datos (targets desplazados), schema (`_model.yaml`/`_training.yaml`), export/`deploy/` (self-speculative decoding), tests de paridad.
- **Por qué fuera de foco:** el checkpoint publicado no la usa (fidelidad del modelo liberado = sin MTP); costo alto y beneficio principal en velocidad de training/serving, no en la arquitectura del backbone.

### P3.9 — Vía visual nativa (MoonViT-V2 + projector) · **XL (~60–100h)**

- **Paper/HF:** MoonViT-V2 (401M, 27 capas, patch 14, 12 cabezas, hidden 1024), projector `patchmergerv2` (pixel-shuffle 2×2 → 7168), placeholder `<|kimi_image_placeholder|>` (id 163605), entrenamiento **intercalado** texto+visión desde el inicio (§2.4, §3.3).
- **Si se hiciera:** torre vision (nuevo modelo o reusar `frankenstein_vit`), fusión de embeddings en el decoder, dataset builder multimodal (interleaved), schema (`_vision.yaml`/`_model_class.yaml`), export. Cambia el contrato `model_class` (hoy el decoder es text-only).
- **Por qué fuera de foco:** repo CPU-first single-device; la fidelidad text-only del backbone (KDA/MLA/AttnRes/LatentMoE) es el objetivo; la vía visual duplica el scope de datos+trainer.

---

## 5. Anotaciones para el YAML (aplicar junto a cada milestone)

Bloque de comentario sugerido al tope de `configs/kimi_k3.yaml` (resumen de foco/drift):

```yaml
# Kimi K3 (arXiv:2607.24653) — spec-fidelity preset.
#
# VERIFICADO EXACTO vs paper Tabla 1 + config.json de moonshotai/Kimi-K3:
#   dims (vocab 163840, hidden 7168, 93 capas), layer_pattern 69 kda + 24 mla
#   (MLA final extra), mla.latent_rank 512, moe 896/16, expert 3072,
#   latent 3584 + norm pre-up-proj, 2 shared, sigmoid + scale 1.0,
#   1 capa densa, rms_norm, cosine.
#
# DESVIACIONES DOCUMENTADAS (con plan .opencode/plan/kimi-k3-drift-plan.md):
#   - num_heads 56 (K3: 96 en MLA y KDA) — bloqueado por el invariante
#     hidden % num_heads == 0 hasta P2.5 (heads por mixer).
#   - quantile balancing aproximado con centered_sign (P1.2 lo implementa).
#   - SiTU-GLU aproximado con swiglu + clamp 4.0 (P1.1 lo implementa).
#   - KDA estilo Kimi Linear sin decay acotado/ShortConv de K3 (P2.6).
#   - MLA Mehta-style con RoPE en vez de Gated MLA NoPE (P2.7; NoPE ya
#     desactivado abajo vía use_pe.mla=false).
#   - Optimizador AdamW placeholder en vez de Per-Head Muon + wd 0.1 (P1.4).
#   - max_length 4096: límite práctico single-device; el paper usa
#     curriculum 8K→64K (pretrain) → 256K→1M (cooldown).
#
# FUERA DE FOCO (documentado en el plan, no perseguido):
#   - MTP (1 capa entrenada en el paper; desactivada en los pesos liberados).
#   - Vía visual nativa MoonViT-V2 + projector (multimodal interleaved).
#   - Contexto 1M y su curriculum de datos.
```

Diffs concretos por milestone:

- **M0 (drift YAML-only, §2):**
  ```yaml
  positional_encoding: {use_pe: {mla: false}}      # A1 NoPE
  training.task: causal_lm                          # A2
  training.scheduler_warmup_ratio: 0.01             # A3
  ffn.hidden_size: 33792                            # A4 (o comentario de compromiso)
  # A5: optimizer_class: muon + muon-* (wd 0.1) cuando P1.4 caiga; interim ver §2
  ```
- **M1:** `ffn.activation: situ` (+ betas 4/25), `moe.balancing.bias_update: quantile` (sin `bias_update_rate`), `activation_clamp` → 0.
- **M2:** `residuals: {type: block_attn, block_attn: {num_blocks: 8}}`.
- **M3:** `optimizer_class: muon` con `per_head: true` y `clip_value` (K2), `wd 0.1`.
- **M4:** `attention.kda.{num_heads: 96, head_dim: 128, decay_mode: k3, short_conv_kernel: 4}`.
- **M5:** `layer_pattern` `mla` → `gmla` con dims DeepSeek-style (96 cabezas, q_lora 1536, nope 128, v 128).

---

## 6. Milestones y esfuerzo total (en foco)

| Milestone | Ítems | Esfuerzo estimado | Entregable |
|---|---|---|---|
| **M0** | A1–A7 (YAML-only) | ~0.5 d | preset alineado en lo que el schema permite + comentarios |
| **M1** | P1.1 `situ` + P1.2 `quantile` (+ P2.10 eps) | 1.5–3 d | activación y balanceo fieles; YAML los usa |
| **M2** | P1.3 AttnRes decoder | 1–1.5 d | Block AttnRes ×8 en el preset |
| **M3** | P1.4 Per-Head Muon | 1.5–2 d | optimizador K3 en el preset |
| **M4** | P2.5 heads por mixer + P2.6 KDA v2 | 4–7 d | 96 cabezas reales + KDA de K3 |
| **M5** | P2.7 Gated MLA | 2–3 d | patrón completo `gmla` |
| **Total en foco** | | **~11–17 días hábiles** | |
| Fuera de foco | P3.8 MTP (~40–60h), P3.9 visión (~60–100h) | no agendado | |

Orden sugerido = M0→M5 (cada milestone deja el preset validado y los gates verdes). P2.5 antes de P2.6/P2.7 porque ambos consumen las claves de heads por mixer.

## 7. Gates que deben quedar verdes (por milestone)

- Todos: `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings`
- M1: `test_activation_functions.py`, `test_schema_consistency.py`, `test_yaml_examples.py` (+ suite MoE).
- M2: `test_residuals.py`, `test_model_variants.py`, suites `test_kv_cache_*` (fallback AttnRes+cache).
- M3: `test_optimizer_factory.py`, `test_optimizer_base.py`, reglas `if/then` de prefijos en `_training.yaml`.
- M4/M5: `test_schema_attention_layers.py`, `test_kv_cache_mixers.py` (parametriza `IMPLEMENTED_MIXERS`), `test_common_modules.py`, `test_naming_conventions.py`.
