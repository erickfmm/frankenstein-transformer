# Kimi K3 — Drift de `configs/kimi_k3.yaml` y plan de implementación

> **Fecha:** 2026-11-10 · **Revisión:** 2026-11-11 (tras el rollout del registro y la
> eliminación del flattener, commits `d1f6ad0`…`8d93194`; ver §8 changelog).
> **Estado:** análisis revisado contra el código actual; implementación no iniciada.
> **Fuentes contrastadas:**
> - Paper: *Kimi K3: Open Frontier Intelligence*, arXiv:2607.24653v2 (Tabla 1; §2.1 Hybrid Attention; §2.1.1 KDA; §2.1.2 Gated MLA; §2.2 AttnRes; §2.3 Stable LatentMoE; §2.3.1 Norm; §2.3.2 SiTU-GLU; §2.3.3 + App. C Quantile Balancing; §2.5 Per-Head Muon; §3.3 Training Recipe; §3.4 Long-Context). **Re-verificado 2026-11-11** contra el PDF: β₁=4/β₂=25 (|f|≤100), warmup lineal 1%, wd 0.1, 96 cabezas (Tabla 1), AttnRes 8 bloques × 12 capas (bloque final parcial, 9 fuentes con el embedding), QB con Top-(k+1) y margen (q+1)-ésimo mayor, per-head Muon sobre Q/K/V + weight clipping K2. Los números del plan son correctos.
> - Pesos oficiales: `config.json` de `huggingface.co/moonshotai/Kimi-K3` (fuente autoritativa de dims exactos).
> - Código/schema de este repo (`src/schema/**`, `src/registry/**`, `src/model/**`) — **verificado 2026-11-11**, era del registro: el schema se ensambla desde `src/registry/schema_assembly.py` y **el flattener ya no existe** (D15).

---

## 0. Veredicto (revisado)

El YAML captura **exactamente** la anatomía macro de K3 (patrón 3:1 KDA:MLA con MLA final, LatentMoE 3584 + RMSNorm pre-up-proj, 896/16, sigmoid + scale 1.0, 2 shared, 1 capa densa, vocab 163840). Los drifts que quedan son: **SiTU-GLU y Quantile Balancing como aproximaciones documentadas**, **`num_heads: 56` vs 96** (bloqueado por el invariante `hidden % num_heads == 0`), **MLA con RoPE en vez de NoPE** (corregible por YAML, ver A1), **AdamW en vez de Per-Head Muon**, y una **KDA "estática" estilo Kimi Linear** sin los cambios de K3.

**Corrección mayor de esta revisión:** el item que era P1.3 (AttnRes en el decoder) **ya está resuelto en el repo**. `FrankensteinDecoder` delega en `FrankensteinEncoder` como backbone (`frankenstein_decoder.py:117`), que ya construye el residual (`build_residual`, L153), aplica el paso por capa (`x = self.residual(idx, x)`, L313–314) y hace el fallback cache→recompute (L219–228). Está testeado end-to-end en decoder (`tests/test_residuals.py::EndToEndDecoderTests`, L441+: `test_full_attn_decoder`, `test_block_attn_decoder` train/eval) y se verificó con un smoke test el 2026-11-11 (decoder `mode=decoder` + `layer_pattern [kda, mla, kda, mla]` + MoE sigmoid + `block_attn` num_blocks=2: forward/backward train y forward eval OK, `is_attn_res=True`). Queda **solo el cambio YAML** (nuevo A8).

La mecánica de implementación también cambió con los refactors del registro: los enums (mixers, optimizadores, activaciones) y los fragmentos `attention.<mixer>` se **inyectan desde los specs** al ensamblar el schema; no hay flattener ni campos planos; las knobs de componentes viven en `params_schema` del spec o en bloques anidados (`ffn.activation_config`, `model.attention.<mixer>`).

---

## 1. Verificado correcto (no requiere cambios)

| Clave YAML | Valor | K3 real | Fuente |
|---|---|---|---|
| `dims.vocab_size` | 163840 | 163840 | Tabla 1 ("160K"), HF |
| `dims.hidden_size` | 7168 | 7,168 | Tabla 1, HF |
| `dims.num_layers` | 93 | 93 | Tabla 1 |
| `dims.num_loops` | 1 | 1 (sin looped depth) | — |
| `layer_pattern` | 69 `kda` + 24 `mla`, MLA extra al final | 69 KDA + 24 MLA, "additional Gated MLA at the end" | §2.1; HF `full_attn_layers` — **match exacto** |
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
| `scheduler_type: cosine` | ✓ | "cosine learning rate schedule" (§3.2/§3.3) | paper |
| Approx `centered_sign` y `activation_clamp: 4.0` | documentadas como desviación en comentarios del propio YAML | — | ver P1.2/P1.1 |
| `kda` sin PE | default del spec (`needs_pos_encoder: false`, `registry/builtin/attention.py:582`) | — | ✓ ya correcto |

---

## 2. Drifts corregibles solo con YAML (el schema ya lo soporta)

**Esfuerzo total: ~0.5 día** (editar + correr los gates de schema/presets).

| # | Drift | YAML actual | K3 real | Fix |
|---|---|---|---|---|
| A1 | MLA con RoPE ≠ **NoPE** | `mla` es attention mixer → default `mla_use_pe: true` (RoPE sobre Q/K descomprimidos, arXiv:2506.09342) | NoPE en **todas** las capas MLA (§2.1.2, §3.4); extrapolación a 1M sin retuning RoPE | `positional_encoding_parameters: {use_pe: {mla_use_pe: false}}` — **shape verificado**: las claves son `<mixer>_use_pe`, inyectadas del registro (`schema_assembly._inject_mixers`, `_positional_encoding.yaml:376+`); ejemplo vigente: `configs/examples/pe_mixed_use_pe.yaml` (`kda` ya defaultea false ✓) |
| A2 | `training.task: mlm` ≠ causal LM | `mlm` (+ `mlm_probability: 0.15`) | next-token prediction multimodal (§3.3); con `frankenstein_text_decoder` la tarea fiel es `causal_lm` (regla válida en `config_loader`) | `training.task: causal_lm` y **quitar `mlm_probability`** (solo aplica a `mlm`, `_training.yaml:220`) |
| A3 | Warmup 10% ≠ **1%** | `scheduler_warmup_ratio: 0.1` | "cosine schedule with a **1% linear warmup**" (§3.3) | `scheduler_warmup_ratio: 0.01` |
| A4 | Capa densa 3072 ≠ 33792 | `ffn.hidden_size: 3072` | `intermediate_size: 33792` (HF) para la capa 1 densa | `ffn.hidden_size: 33792`, o dejar 3072 como compromiso de memoria **documentado en comentario** (el ancho de expertos ya está aparte en `moe.expert_hidden_size` ✓) |
| A5 | AdamW ≠ **Muon (interim)** | `adamw-*`, lr 1e-6…5e-6 (placeholder), wd 0.001–0.01, betas/eps | Muon para matrices (§2.5), **wd 0.1** (§3.3); peak LR no publicada (scaling-law) | mientras llega P1.4: `optimizer_class: muon` con `muon-lr_*` razonables y `muon-wd_other: 0.1`; el spec `muon` existe (`optimizer.py:259`, root: `momentum`, `nesterov`, `ns_steps`, `ns_eps`; fallback AdamW vía sufijos compartidos `lr/wd/betas/eps` por grupo). K3 per-head + clip K2 = **P1.4** |
| A6 | `max_length: 4096` vs curriculum 8K→64K→256K→1M (§3.4) | 4096 | 8k arranque | mantener 4096 (single-device) con **comentario** que documente el curriculum del paper |
| A7 | `rms_norm_eps: 1e-5` (HF) | no seteado | 1e-5 | **confirmado 2026-11-11: no existe** `norm.eps` de bloque (el único `eps` de `_norm.yaml` es el de QK-norm, L400; `RMSNorm.__init__` acepta `eps=1e-6` pero la factory no lo pasa desde config) → la clave hay que agregarla: **P2.10** |
| A8 | **AttnRes ausente** (nuevo; ex-P1.3) | sin bloque `residuals` (default `standard`) | Block AttnRes, 8 bloques × 12 capas, 9 fuentes con embedding (§2.2; HF `attn_res_block_size: 12`) | `residuals: {type: block_attn, block_attn: {num_blocks: 8}}` — **soporte decoder ya implementado y testeado** (ver §0); validación vigente: `num_blocks ≤ num_layers×num_loops` (`_validation.py:317-324`), incompatible solo con mHC-hyperloop (no aplica). Smoke test decoder+MoE+kda/mla OK |

**Notas:**
- A2 implica el pipeline `causal_lm` (ya soportado por `config_loader`). El resto del bloque `training` (batch, acumulación, `scheduler_total_steps`) son placeholders de experimento local, no de fidelidad K3 — fuera del alcance.
- `num_heads: 56` **no** es corregible por YAML: 7168 % 96 = 64 ≠ 0 y el invariante `hidden_size % num_heads == 0` (chequeado por mixer en su `__init__`, p. ej. `kda.py:128`, `mla.py:92`) lo impide (ver P2.5).

---

## 3. Faltantes de implementación — EN FOCO

Escala de esfuerzo: **S** ≤ 4h · **M** 4–12h · **L** 12–32h · **XL** > 32h.

> Contexto registry-era para todos los items: el schema ensamblado inyecta enums y
> fragmentos desde `src/registry/**`; no existe `src/utils/config_flatten.py` (D15);
> los knobs de mixers llegan como `ComponentParams` bindrados (`src/model/attention/params.py`).

### P1.1 — Activación `situ` (SiTU-GLU) · **S–M (4–8h)**

**Gap (§2.3.2 + Fig. 4 + App. B):** K3 usa softcap en **ambas** ramas del GLU:

```
gate(u) = β₁·tanh(u/β₁) · σ(u)        # u = W_g x   (Swish con factor lineal softcap-eado)
up(v)   = β₂·tanh(v/β₂)               # v = W_u x
|f| ≤ β₁·β₂ = 100                     # β₁ = 4, β₂ = 25 (HF: activation_situ_beta / _linear_beta)
```

El YAML aproxima con `swiglu` + `moe.experts.activation_clamp: 4.0`: clamp **duro** (no tanh, el paper lo contrasta explícitamente en App. B) y **solo en la puerta** (`GatedFFN.forward` clampea `gate_proj(x)`; `up_proj(x)` pasa sin transform — `glu.py:86-93`).

**Cambios (receta §11.1 "Add or change a component" de `docs/architecture.md`):**
1. `src/model/activation_function/glu.py` — extender `GatedFFN`/`make_gated_ffn` con el kind `"situ"`: gate branch `β₁tanh(u/β₁)·σ(u)` y **cap de la rama up** `β₂tanh(v/β₂)` (nuevos kwargs `beta_gate`/`beta_up`, como hoy llega `clamp_value`; con `situ` el clamp duro debe quedar deshabilitado).
2. `src/registry/builtin/activation.py` — nuevo `ActivationSpec` `situ` con `is_glu=True`, builder `src.model.activation_function.glu:make_gated_ffn` y defaults de los betas. **El enum se auto-inyecta** (`schema_assembly._inject_simple_enum` sobre `model.ffn.activation`); la línea `enum:` de `_ffn.yaml` es sombra sobreescrita en el ensamblado (actualizarla por higiene solo si `test_schema_consistency` lo exige).
3. `src/schema/_model/_ffn.yaml` — **solo** el sub-bloque `activation_config`: propiedades `situ_beta_gate` (default 4.0) y `situ_beta_up` (default 25.0), `exclusiveMinimum: 0`, descripciones bilingües y referencias (arxiv:2607.24653). `activation_config` SÍ es hand-maintained.
4. `src/model/config.py` — extender `_validate_ffn_activation_config` (L56+) con las claves nuevas (números > 0). **No hay campos planos** (flattener eliminado): el builder lee el dict anidado `ffn.activation_config`.
5. Plumbing de los betas al builder: `hybrid_layer.py` (call site del GatedFFN denso, ~L206–220, donde hoy pasa `clamp_value`) y `src/model/moe/layer.py` (fábrica de expertos/shared, L129/L139) → `make_gated_ffn(..., beta_gate=…, beta_up=…)`.
6. `configs/examples/activation_situ.yaml` + actualizar `configs/kimi_k3.yaml` (`ffn.activation: situ`; `ffn.activation_config: {situ_beta_gate: 4.0, situ_beta_up: 25.0}`; `moe.experts.activation_clamp` → 0).
7. Tests: `tests/test_activation_functions.py` (softcap exacto, cota β₁β₂, recuperación de SwiGLU cuando β→∞ y primer orden en el origen), `tests/test_schema_consistency.py` (enum registro ↔ dataclass ↔ defaults), `tests/test_yaml_examples.py` (auto-glob).
8. Docs: `docs/architecture.md` §7.6 (FFN y activaciones; 43 → 44 valores, tablas generadas D17) + regenerar `docs/schema.assembled.yaml`.

**Riesgos:** la cuenta "43 activaciones" aparece en censos/docs (D17 la regenera); `activation_clamp` + `situ` son excluyentes (validar).

### P1.2 — `bias_update: quantile` (Quantile Balancing) · **M (6–10h)**

**Gap (§2.3.3, Eq. 13–14, Fig. 5 + App. C):** router sigmoid, bias solo en la selección (no en los pesos ✓ ya así en el repo). QB reemplaza la regla de paso fijo:

```
s_i = sigmoid(W_r x_i)                     # scores del batch (m tokens, n expertos)
selección sobre s_i + b^t con Top-(k+1)     # el (k+1)-ésimo es el cutoff α_i
bias_j^{t+1} = Quantil_{1−k/n}({ s_ij − α_i }_{i=1..m})   # (q+1)-ésimo mayor margen, q = m·k/n
```

- Causal: el bias derivado de un forward **no se usa en ese mismo batch** (efecto al paso siguiente) — igual que `sign`/`centered_sign` hoy.
- Inferencia: bias congelado (buffer, no parámetro entrenable).
- El paper usa histogramas solo porque el cuantil abarca el batch global multi-rank; en runtime single-device el **cuantil exacto** es viable (histogramas out of scope).

**Cambios (MoE no es familia del registro: `_moe.yaml` + `src/model/moe/` hand-maintained es correcto):**
1. `src/model/moe/router.py` — en `forward`, retener `last_scores` (sigmoid cruda) y computar el cutoff αᵢ del Top-(k+1) sobre `s + b`; extender `update_expert_bias` (~L245–260, hoy `sign`/`centered_sign` con `bias_update_rate`) con la rama `quantile` (`torch.quantile`/`kthvalue` por experto sobre los márgenes `s_ij − α_i`; definir empates y `q < 1`/batch chico). Sin `bias_update_rate` para esta rama.
2. `src/schema/_model/_moe.yaml` — enum `balancing.bias_update: [none, sign, centered_sign, quantile]` (hoy L494) + descripciones bilingües (arxiv:2607.24653; nota: histogramas cross-rank out of scope).
3. `src/model/config.py` — validar `quantile` igual que `sign` (requiere `router.scoring ≠ softmax`).
4. `src/model/moe/layer.py` (L118–121 pasa `bias_update`/`bias_update_rate`) — no pasar rate cuando `quantile`.
5. `configs/kimi_k3.yaml` — `balancing: {aux_loss: none, bias_update: quantile}` (eliminar `bias_update_rate`).
6. Tests: `tests/test_moe.py` + caso nuevo: batch sintético con desbalance conocido → tras un update QB la carga del paso siguiente es exactamente `q` por experto (sin empates); empates y `m·k/n < 1`; bias congelado en `eval()`.

**Riesgos:** interacción con `gradient_accumulation_steps: 16` (¿update por micro-batch o por paso de optimizer?). El paper deriva el bias de "a single forward pass" (micro-batch); documentar la elección en el schema.

### P1.3 — ~~AttnRes en el decoder~~ · **RESUELTO (obsoleto)**

El diagnóstico original era incorrecto/queda desactualizado: el decoder **delega en `FrankensteinEncoder`** (`self.backbone = FrankensteinEncoder(self.config)`, `frankenstein_decoder.py:117`), por lo que hereda todo el wiring de AttnRes del encoder: construcción (`build_residual`, encoder L153), estado (`register_state`/`reset_state`/`set_embedding`, L265–268), paso por capa (`x = self.residual(idx, x)`, L313–314), `finalize` (L356) y fallback cache→recompute (L219–228; el comentario del decoder en L259 es ese path).

Evidencia: `tests/test_residuals.py::EndToEndDecoderTests` (L441+) cubre `full_attn` y `block_attn` en decoder train/eval; ejemplo `configs/examples/block_attn_res_adamw.yaml`; y smoke test del 2026-11-11 (decoder + `[kda, mla]` + MoE sigmoid + `block_attn`) pasa forward/backward.

**Restante:** solo el YAML (item A8). Opcional (S): example preset decoder+MoE+AttnRes y caso de test decoder+KV-cache→warning+recompute si no existe ya.

### P1.4 — Per-Head Muon (+ weight clipping K2) · **M–L (8–16h)**

**Gap (§2.5, §3.3):** K3 optimiza matrices con Muon, refinado a **variante per-head**: particionar los momentum de las proyecciones **Q, K, V** por cabeza y correr Newton–Schulz por bloque (equaliza la escala de update entre cabezas; además reduce overhead: bloques per-head más baratos que la matriz completa). Además usa el **weight clipping de Kimi K2** y **wd 0.1** global.

**Estado:** `muon.py` existe (Newton–Schulz + fallback AdamW, root `momentum/nesterov/ns_steps/ns_eps`) pero **sin** `per_head` ni `clip_value` (grep 2026-11-11: cero matches de `per_head` en src/). No confundir con las variantes `muon2/ada_muon/af_muon/nor_muon/turbo_muon/tuon` — K3 usa Muon vanilla + per-head + clip K2.

**Cambios (receta "Optimizer" — spec-first):**
1. `src/model/optimizer/muon.py` — opción `per_head: bool` + `clip_value` (clip elementwise post-update aplicado solo a las matrices ortogonalizadas): si el nombre del parámetro matchea proyecciones de atención (`q_proj`, `k_proj`, `v_proj`; en MLA `q_proj`/`dkv_proj`), reshapar el momentum `(out, in) → (heads, head_dim, in)` y orthogonalizar por cabeza.
2. `src/registry/builtin/optimizer.py` — añadir `per_head` y `clip_value` a los `param_bindings` del spec `muon` (L259+). Alternativa: spec nuevo (p. ej. `k3_muon`) si se prefiere familia separada; `custom_build` no debería hacer falta.
3. `src/schema/_optimizer.yaml` / `_training.yaml` — **no hand-editar**: el enum, `by_optimizer` y las reglas `if/then` de prefijos se generan del `OptimizerSpec` (AGENTS #7). Las claves resultantes serían `muon-per_head`, `muon-clip_value` (root) + `muon-<group>_*` existentes.
4. `configs/kimi_k3.yaml` — `optimizer_class: muon` con `muon-per_head: true`, `muon-clip_value: <K2>`, `muon-wd_*: 0.1`, `muon-lr_*` placeholders documentados (peak LR no publicada).
5. Tests: `tests/test_optimizer_factory.py`, `test_optimizer_base.py` (paso per-head == paso full con 1 cabeza; clipping acota la norma; fallback AdamW de no-matrices intacto). `configs/README.md`.

**Riesgos:** clasificar parámetros por nombre a través de 49 mixers (los nombres de proyección varían); la semántica K3 aplica solo a Q/K/V de atención — documentar qué nombres matchea.

### P2.5 — Cabezas por mixer desacopladas de `hidden_size` · **L (12–20h)**

**Gap:** K3 usa **96 cabezas** en MLA (`num_attention_heads: 96`) **y** en KDA (`linear_attn_config: num_heads 96, head_dim 128`; 96×128 = 12288 > 7168, expansión propia). El repo fuerza `head_dim = hidden_size // num_heads` y el chequeo por mixer `hidden_size % num_heads == 0` (7168 % 96 = 64 → 96 inexpresable; por eso el YAML usa 56).

**Cambios (registry-era):**
1. **kda**: el spec (`registry/builtin/attention.py:571`) hoy **no tiene** `params_schema` (cero knobs). Crear el fragmento `src/schema/_model/_attention_kda.yaml` con el sub-bloque (`num_heads`, `head_dim` null→global, y las claves de P2.6) y referenciarlo: `params_schema="schema/_model/_attention_kda.yaml#/properties/kda"` + `defaults`.
2. **mla**: el spec (L643) ya referencia `params_schema="schema/_model/_attention_latent.yaml#/properties/mla"` con `defaults={"latent_rank": None}` → extender ese mismo fragmento con `num_heads`, `q_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `use_nope`, `use_output_gate` (los 4 últimos se consumen en P2.7).
3. Runtime: los mixers leen los knobs vía `ComponentParams` (`mixer_params`, `src/model/attention/params.py`); relajar el chequeo interno de cada mixer (`kda.py:128`, `mla.py:92`, …) cuando el mixer declara `num_heads`/`head_dim` propios, y reformular la regla dura #2 de AGENTS/docs (`_dims.yaml:45/58` "Must be divisible by `num_heads`") para mixers con override.
4. Cache: `CACHE_SPECS` se **deriva del registro** (`capabilities.py:_derive_cache_specs`, no hay dict hand-maintained desde `66f020d`) — el `CacheSpec` del spec kda debe reflejar que el estado escala con las cabezas del mixer. `kda`/`mla` ∈ `IMPLEMENTED_MIXERS` (`capabilities.py:47-86`) → `test_kv_cache_mixers` los parametriza automáticamente con las nuevas dims (D17).
5. `configs/kimi_k3.yaml` — `dims.num_heads` queda como default de los mixers sin override; `attention.kda: {num_heads: 96, head_dim: 128}`; `attention.mla: {num_heads: 96}`.
6. Tests: `test_schema_attention_layers.py`, `test_schema_consistency.py`, `test_kv_cache_mixers.py`, ejemplo yaml.

**Riesgos:** la regla dura #2 tiene tests asociados; cambia el conteo de parámetros del preset (96 cabezas acerca el total a los 2.78T/104B del paper). Docs: reglas #2 y #12 de AGENTS.

### P2.6 — KDA "v2" (los cambios de K3, §2.1.1) · **L (16–32h)** (chunkwise aparte)

**Gaps** (contra `src/model/attention/gated/kda.py` actual):

| K3 (paper §2.1.1, Fig. 2) | Repo actual |
|---|---|
| Decay **input-dependiente**: logits `z_t = W_lowrank x_t + b_α` (per-head bias) | `log_decay_base/delta` son **parámetros libres**, no dependen de x (kda.py:135-136) |
| Mapeo **acotado**: `g = g_min·σ(e^{A_h}·z)`, `g_min = −5`, `A_h` per-head (init 0) → retención `α > e⁻⁵ ≈ 6.7e−3` | `exp(base − softplus(delta))` no acotado (kda.py:194-197, estilo Kimi Linear) |
| **ShortConv (kernel 4) + Swish** antes de las proyecciones Q/K/V (`short_conv_kernel_size: 4`) | sin convolución |
| RMSNorm **por cabeza** antes del gate de salida | `nn.LayerNorm(total_dim)` global (kda.py:139) |
| Gate de salida full-rank input-dependiente ✓ | `g_proj` full-rank silu — **coincide** ✓ |
| L2 norm q/k ✓, silu(v) ✓ | ✓ |
| Forma chunkwise (recurrente entre chunks, paralela dentro) | bucle O(T) token a token en Python (CacheSpec notes: "plain recurrence, no DPLR kernel") |

**Cambios:**
1. `src/model/attention/gated/kda.py`:
   - Proyección low-rank del logit de decay (`hidden → r → num_heads·head_dim`) + `b_α` per-head + `A_h` (log-scale por cabeza).
   - Mapeo `g_min·sigmoid(e^{A_h} z)` con `g_min` configurable (default −5); mantener la forma actual como modo legacy (`decay_mode: kimilinear | k3`).
   - ShortConv depthwise (kernel 4, groups=hidden) + silu antes de q/k/v.
   - Reemplazar LayerNorm por RMSNorm head-wise.
   - (Opcional, esfuerzo aparte **+L**): chunkwise reference en PyTorch puro con test de paridad contra el bucle secuencial — vía de eficiencia, no requerida para fidelidad funcional.
2. Knobs en el fragmento creado en P2.5 (`_attention_kda.yaml`) + `defaults` del spec: `decay_mode`, `decay_min_log` (−5), `decay_low_rank`, `short_conv_kernel` (4), `use_full_rank_gate` (true).
3. La conv corta añade estado al cache incremental (kernel−1 tokens): extender el `CacheSpec` del spec (`extras`, p. ej. `("conv_state",)`, como `longformer`/`lsa` hacen hoy) + `make_cache_slot`.
4. `configs/examples/attention_kda*.yaml`; `configs/kimi_k3.yaml` setea el modo K3.
5. Tests: paridad decay acotado vs legacy (con `g_min → −∞` converge al mapping actual), cota de retención `α > e⁻⁵`, cache con conv, `test_kv_cache_mixers.py`.

**Riesgos:** bucle secuencial con estado 12288-dim × 96 cabezas × T es lento — aceptable en CPU para tests chicos, documentarlo.

### P2.7 — Gated MLA "DeepSeek-style" (NoPE + q_lora + output gate) · **L (16–24h)**

**Gap (§2.1.2 + HF):** el `mla.py` actual es el de Mehta et al. (arXiv:2506.09342; el propio CacheSpec lo dice: "applies RoPE to the decompressed q,k, so DeepSeek's absorb trick is unavailable"). K3 usa MLA DeepSeek-style: `q_lora_rank: 1536`, `qk_nope_head_dim: 128` + `qk_rope_head_dim: 64` (con NoPE la parte rope queda sin usar), `v_head_dim: 128`, 96 cabezas, **gate de salida full-rank input-dependiente** (`W_g` full rank, igual que KDA — §2.1.2), salida de atención en FP32 durante training (nota de kernel).

**Recomendación:** implementarlo como **mixer nuevo** (`gmla`) vía spec, para no romper la semántica de `mla` existente; `configs/kimi_k3.yaml` pasa su `layer_pattern` de `mla` a `gmla`. Alternativa (más invasiva): flags dentro de `attention.mla` que cambian la parametrización.

**Cambios (ruta mixer nuevo, registry-era):**
1. `src/model/attention/latent/gmla.py` — c_KV = W_DKV x (512); q = W_UQ(W_DQ x) (1536); k/v descomprimidos por cabeza (nope 128, v 128; rope 64 solo si `use_pe`); sin PE por default; `out = SDPA(...) ⊙ silu(W_g x)`; `make_cache_slot` → `LatentKVSlot` de c_KV (reutilizar hook `cached_latent_attention`, extensión para gate).
2. Spec `MixerSpec(name="gmla", …)` en `registry/builtin/attention.py` con `params_schema` (fragmento propio o `_attention_latent.yaml#/properties/gmla`), `needs_pos_encoder=False`, `CacheSpec(slot_type="latent_kv", …)`. `HybridLayer` resuelve el builder vía `resolve_builder(spec.builder)` (hybrid_layer.py:162-169) — **no hay registry dict manual que editar**. Añadir `gmla` a `IMPLEMENTED_MIXERS` cuando el branch incremental exista (capabilities.py:47) → `test_kv_cache_mixers` lo cubre solo. Docs `_dims.yaml` (ambas tablas) se generan/verifican por D17.
3. `configs/kimi_k3.yaml` — `layer_pattern` con `gmla`, `attention.gmla.{latent_rank: 512, q_lora_rank: 1536, qk_nope_head_dim: 128, v_head_dim: 128, num_heads: 96}` (NoPE por default, sin `use_pe`).

**Riesgos:** decode incremental con forma absorbida es la vía eficiente de MLA; mínimo viable = expandir K/V por paso (correcto, más lento); documentar.

### P2.10 — Numérica menor (eps de norm, FP32) · **S (2–4h)**

- **Agregar `norm.eps`** (confirmado ausente): `_norm.yaml` (clave `eps` a nivel `norm`, hoy solo existe bajo `qk`, L400) + la factory/spec builders de normas (`RMSNorm` ya acepta `eps`, `rms.py:59`; falta pasarlo desde config) + `model/config.py` (bloque norm) + `test_common_modules.py`. Setear `1e-5` en el preset (HF `rms_norm_eps: 1e-5`).
- FP32 en la salida de atención durante training (§2.1.2): **nota de documentación solamente** (detalle de kernel para corregir el rounding de flash attention; SDPA de torch ya acumula en fp32) — no planificar código.

---

## 4. Fuera de foco (documentado, no planeado)

### P3.8 — MTP (Multi-Token Prediction, 1 capa) · **XL (~40–60h)**

- **Paper:** Tabla 1 — K2 y K3 entrenan con **1 capa MTP**; los pesos liberados la traen desactivada (`num_nextn_predict_layers: 0`).
- **Si se hiciera:** módulo draft (capa extra que reusa embedding + head), `trainer.py` (loss MTP con pesos por profundidad), pipeline de datos (targets desplazados), schema (`_model.yaml`/`_training.yaml`), export/`deploy/` (self-speculative decoding), tests de paridad.
- **Por qué fuera de foco:** el checkpoint publicado no la usa (fidelidad del modelo liberado = sin MTP); costo alto y beneficio principal en velocidad de training/serving, no en la arquitectura del backbone.

### P3.9 — Vía visual nativa (MoonViT-V2 + projector) · **XL (~60–100h)**

- **Paper/HF:** MoonViT-V2 (401M, 27 capas, patch 14, 12 cabezas, hidden 1024), projector `patchmergerv2` (pixel-shuffle 2×2 → 7168), placeholder `<|kimi_image_placeholder|>`, entrenamiento **intercalado** texto+visión desde el inicio (§2.4, §3.3).
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
#   1 capa densa, rms_norm, cosine, block_attn 8x12 (AttnRes).
#
# DESVIACIONES DOCUMENTADAS (con plan .opencode/plan/kimi-k3-drift-plan.md):
#   - num_heads 56 (K3: 96 en MLA y KDA) — bloqueado por el invariante
#     hidden % num_heads == 0 hasta P2.5 (heads por mixer).
#   - quantile balancing aproximado con centered_sign (P1.2 lo implementa).
#   - SiTU-GLU aproximado con swiglu + clamp 4.0 (P1.1 lo implementa).
#   - KDA estilo Kimi Linear sin decay acotado/ShortConv de K3 (P2.6).
#   - MLA Mehta-style con RoPE en vez de Gated MLA NoPE (P2.7; NoPE ya
#     desactivado abajo vía use_pe.mla_use_pe=false).
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
  positional_encoding_parameters: {use_pe: {mla_use_pe: false}}  # A1 NoPE (shape <mixer>_use_pe)
  training.task: causal_lm                                      # A2 (+ quitar mlm_probability)
  training.scheduler_warmup_ratio: 0.01                         # A3
  ffn.hidden_size: 33792                                        # A4 (o comentario de compromiso)
  residuals: {type: block_attn, block_attn: {num_blocks: 8}}    # A8 (ex-P1.3; ya soportado)
  # A5 interim: optimizer_class: muon + muon-lr_*/muon-wd_* (wd 0.1); per-head+clip en P1.4
  ```
- **M1:** `ffn.activation: situ` + `ffn.activation_config: {situ_beta_gate: 4.0, situ_beta_up: 25.0}`; `moe.balancing: {aux_loss: none, bias_update: quantile}` (sin `bias_update_rate`); `moe.experts.activation_clamp: 0.0`; `norm.eps: 1e-5` (P2.10).
- **M2:** `muon-per_head: true`, `muon-clip_value: <K2>`, `muon-wd_*: 0.1` (claves root `<optimizer>-<param>`, generadas del spec).
- **M3:** `attention.kda: {num_heads: 96, head_dim: 128, decay_mode: k3, short_conv_kernel: 4}`.
- **M4:** `layer_pattern` `mla` → `gmla` con dims DeepSeek-style (96 cabezas, q_lora 1536, nope 128, v 128).

---

## 6. Milestones y esfuerzo total (en foco)

| Milestone | Ítems | Esfuerzo estimado | Entregable |
|---|---|---|---|
| **M0** | A1–A8 (YAML-only; A8 = ex-P1.3) | ~0.5 d | preset alineado en lo que el schema permite + comentarios |
| **M1** | P1.1 `situ` + P1.2 `quantile` + P2.10 `norm.eps` | 1.5–2.5 d | activación, balanceo y eps fieles; YAML los usa |
| **M2** | P1.4 Per-Head Muon | 1.5–2 d | optimizador K3 en el preset |
| **M3** | P2.5 heads por mixer + P2.6 KDA v2 | 4–7 d | 96 cabezas reales + KDA de K3 |
| **M4** | P2.7 Gated MLA (`gmla`) | 2–3 d | patrón completo `gmla` |
| **Total en foco** | | **~9.5–15 días hábiles** | |
| Fuera de foco | P3.8 MTP (~40–60h), P3.9 visión (~60–100h) | no agendado | |

Orden sugerido = M0→M4 (cada milestone deja el preset validado y los gates verdes). P2.5 antes de P2.6/P2.7 porque ambos consumen las claves de heads por mixer. *(Re-numeración 2026-11-11: el viejo M2 AttnRes-decoder desaparece — resuelto; los demás corren un lugar.)*

## 7. Gates que deben quedar verdes (por milestone)

- Todos: `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings`
- M0: `test_yaml_examples.py` (auto-glob), `test_schema_consistency.py` (presets validan contra el schema ensamblado).
- M1: `test_activation_functions.py`, `test_schema_consistency.py`, `test_yaml_examples.py`, `test_moe.py`, `test_common_modules.py`; al tocar specs → `test_registry_core.py`, `test_registry_binding.py`, `test_schema_assembly.py`; regenerar `docs/schema.assembled.yaml`.
- M2 (Muon): `test_optimizer_factory.py`, `test_optimizer_base.py`, reglas `if/then` de prefijos (generadas — verificar en `test_schema_consistency.py`).
- M3/M4: `test_schema_attention_layers.py`, `test_kv_cache_mixers.py` (parametriza `IMPLEMENTED_MIXERS`), `test_common_modules.py`, `test_naming_conventions.py`, `test_registry_core.py`.
- Docs: cualquier cambio de contrato del registro/schema actualiza `INSTALL.md`, `HOWTO_MAKE_PLUGIN.md`, `README.md` y `docs/architecture.md` en el mismo change (AGENTS #12).

## 8. Changelog de esta revisión (2026-11-11)

Qué cambió respecto a la versión del 2026-11-10 y por qué:

1. **P1.3 (AttnRes decoder) marcado RESUELTO/obsoleto**: el decoder delega en `FrankensteinEncoder` (`frankenstein_decoder.py:117`); wiring + tests end-to-end ya existen (`test_residuals.py::EndToEndDecoderTests`); smoke verificado hoy (decoder + kda/mla + MoE + `block_attn`, train/eval). Pasa a ser el item YAML **A8**; el milestone M2 desaparece y la tabla se re-numera (total baja de ~11–17 d a ~9.5–15 d).
2. **Mecánica registry-era** (commits `70082e4` flattener eliminado, `9155e9f` shadow blocks + `use_pe`, `66f020d` CACHE_SPECS derivado): P1.1 ya no crea campos planos (`ffn_activation_situ_beta_*`) sino claves anidadas en `ffn.activation_config` (hand-maintained) + spec en el registro (enum auto-inyectado); P1.4 explicita que enum/`by_optimizer`/`if/then` se generan del `OptimizerSpec`; P2.5/P2.6/P2.7 usan `MixerSpec.params_schema` (kda hoy sin fragmento; mla referencia `_attention_latent.yaml#/properties/mla`) y `IMPLEMENTED_MIXERS` (frozenset hand-listado) para la cobertura de cache.
3. **A1 corregido al shape real**: `positional_encoding_parameters.use_pe.<mixer>_use_pe` (no `positional_encoding.use_pe.mla`); el ejemplo `pe_mixed_use_pe.yaml` usa el shape vigente.
4. **A7 confirmado ausente** (no existe `norm.eps` de bloque; solo `qk.eps`; `RMSNorm` acepta eps no expuesto) → receta concreta en P2.10.
5. **A5 refinado**: el spec `muon` existe (root `momentum/nesterov/ns_steps/ns_eps`, fallback AdamW con sufijos compartidos); `per_head`/`clip_value` no existen (grep 0 matches) — P1.4 los agrega.
6. **Cifras verificadas hoy**: 49 mixers (`MixerSpec` count), 43 activaciones (registro), 36 optimizadores; `situ`, `quantile` (como bias_update), `gmla`, `per_head` ausentes del código.
7. **Paper re-verificado contra arXiv:2607.24653v2** (App. B/C incluidas): todos los números citados siguen en pie; sin cambios de fondo en los gaps.
8. Añadida nota A2: quitar `mlm_probability` al pasar a `causal_lm`; referencias de docs actualizadas a §7.6/§11.1–§11.3 de `docs/architecture.md`.
