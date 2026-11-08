# Plan: add Fast Conformer mixer (arXiv:2305.05084)

**Paper:** *Fast Conformer with Linearly Scalable Attention for Efficient Speech
Recognition* — Dima Rekesh, Nithin Rao Koluguri, Samuel Kriman, Somshubra Majumdar,
Vahid Noroozi, He Huang, Oleksii Hrinchuk, Krishna Puvvada, Ankur Kumar,
Jagadeesh Balam, Boris Ginsburg (NVIDIA + UCLA), ICASSP/ASRU 2023,
[arXiv:2305.05084v6](https://arxiv.org/abs/2305.05084).

**One-paragraph summary.** The paper redesigns the Conformer encoder for
efficiency (8× depthwise-separable subsampling) and, for long-form inputs,
replaces global self-attention with **limited-context attention**: every token
attends only to a fixed-size local window, augmented with **global attention
token(s)** pinned at the head of the sequence that attend to everything and are
attended by everything (§2.2: with an optional *separate* Q/K/V projection set,
Longformer-style; §3.4's evaluated long-form model: *shared* projections, window
128 on each side). This makes attention cost linear in sequence length — a
single A100 moves from 15-minute to 675-minute inputs (paper Table 3) and the
global token improves long-form WER (TED-LIUM 9.18 → 7.5, Earnings-21 18.26 →
11.85, Table 8). The paper's attention mechanism is a domain-agnostic sequence
mixer — exactly the Sparse-family pattern this repo already hosts
(`longformer_attn`, `swa_attn`, `bigbird_attn`) — and fills the family's
"window + head-pinned global token with optional dedicated projections" slot.
The Fast Conformer *downsampling schema* (§2.1) is audio-frontend work and is
**out of scope** for a mixer addition.

---

## 2. Paper extract

Everything downstream depends on this. **Fidelity note (D11):** the paper is an
ICASSP-format short paper and gives **no explicit equations** for the attention
mechanism — it defines it precisely in prose (§2.2, §3.4) and explicitly builds
on Longformer (arXiv:2004.05150, "the overlapping chunks approach introduced in
Longformer"), with the authors' NeMo implementation as the open-source recipe
("The models and training recipes have been open sourced through NVIDIA NeMo").
The formulations below are the exact encoding of the paper's prose, with the
global-attention dataflow taken from the paper's own NeMo implementation
(`nemo/collections/asr/parts/submodules/multi_head_attention.py`, class at line
371: "Multi-Head Attention layer … with sliding window local+global attention
from Longformer"). The annexes must present them as the paper's formulation,
citing Longformer for the inherited global-attention pattern, and the module
docstring must state the same.

### 2.1 Mechanism (verbatim paper anchors)

- §2.2: *"We decided to use an alternative approach inspired by Longformer -
  local attention augmented with global tokens. We use a single global attention
  token, which attends to all other tokens and has all other tokens attend to
  it, using a separate set of query, key and value linear projections, while
  others attend in a fixed-size window surrounding each token (see Fig. 2). …
  In order to efficiently compute this attention, we utilize the overlapping
  chunks approach introduced in Longformer."*
- §3.4 (the evaluated long-form model): *"Fast Conformer with limited context
  and global token for long audio was trained in the following way. This model
  has shared query, key and value projection layers that are used for global
  and local attention. The encoder is initialized with the checkpoint
  pre-trained … with full context. We then do additional fine-tuning … with
  limited context attention for 10k steps. … The size of the limited context was
  set to 128 steps on each side of a token, which corresponds to around 10
  seconds."*
- Abstract: *"we replaced global attention with limited context attention
  post-training, while also improving accuracy through fine-tuning with the
  addition of a global token."*
- NeMo reference implementation (the paper's own recipe), global-token
  selection: `is_index_global_attn[:, : global_tokens * global_tokens_spacing :
  global_tokens_spacing]` → global positions are **the head of the sequence**,
  `G = {0, s, 2s, …, (g−1)·s}`; knobs `global_tokens` (count, default 0 in
  NeMo), `global_tokens_spacing` (default 1), `global_attn_separate` (default
  False; when True creates `global_q/global_k/global_v` `nn.Linear(n_feat,
  n_feat)`).

### 2.2 Forward equations (paper-faithful reconstruction)

Notation: input `X ∈ R^{n×H}`, `h` heads, head dim `d = H/h`; window span `w`
(total, per side `w/2`; paper §3.4: `w/2 = 128`), global count `g`, spacing `s`,
`G = {0, s, …, (g−1)s} ∩ [0, n)`.

**Projections (shared path, §3.4):**

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V,\qquad W_\ast \in \mathbb{R}^{H\times H}$$

**Projections (global path, §2.2, optional):**

$$Q^{g} = XW_Q^{g},\quad K^{g} = XW_K^{g},\quad V^{g} = XW_V^{g}$$

**Attention pattern.** Query `i` may attend to key `j` iff

$$M_{ij} = \underbrace{[\,|i-j| \le w/2\,]}_{\text{local window}} \;\vee\; \underbrace{[\,j \in G\,]}_{\text{global column}} \;\vee\; \underbrace{[\,i \in G\,]}_{\text{global row}}$$

(intersected with the causal mask `j ≤ i` in decoder mode; the paper is
encoder/bidirectional — the causal form is this repo's standard decoder
extension, identical to `longformer_attn`'s).

**Scores and output (shared-projection variant).** With effective keys/values
that swap in the global projections at global positions when
`global_attn_separate=true`:

$$\tilde{k}_j = \begin{cases} k^g_j & j \in G \wedge \text{separate}\\ k_j & \text{else}\end{cases}
\qquad \tilde{v}_j = \begin{cases} v^g_j & j \in G \wedge \text{separate}\\ v_j & \text{else}\end{cases}$$

$$z_{ij} = \frac{q_i \cdot \tilde{k}_j}{\sqrt{d}}\ \text{ over } M_{ij},\qquad
o_i = \sum_{j:\,M_{ij}} \mathrm{softmax}_j(z_{ij})\,\tilde{v}_j,\qquad
Y = O\,W_O$$

**Global rows (separate variant).** For `i ∈ G` the paper's Longformer-style
global attention replaces the whole row: the global token's query
`q^g_i` scores **all** positions using the global key projection and aggregates
the global values (NeMo `_compute_out_global_to_all`, lines 729–805):

$$z_{ij} = \frac{q^g_i \cdot k^g_j}{\sqrt{d}}\ \ \forall j \le i,\qquad
o_i = \sum_{j \le i} \mathrm{softmax}_j(z_{ij})\,v^g_j$$

With `global_attn_separate=false` (§3.4) both formulas collapse to one shared
Q/K/V — the global tokens are simply positions with unmasked rows + columns.

**Masked-dense reference vs paper kernel.** The repo computes the full score
matrix and masks it (`O(n^2)` compute, mathematically identical) — the same
CPU-first reference choice as `longformer_attn`/`swa_attn` (architecture.md
line ~1091 states this style of caveat for `lsa_attn`). The paper's
overlapping-chunk kernel (`sliding_chunks`, Longformer) is an optimization and
is **out of scope**.

### 2.3 Pseudocode

**Parallel / training form (masked-dense reference):**

```
G ← {0, s, …, (g−1)s} ∩ [0, n)
Q,K,V ← xW_q, xW_k, xV_v            # BitLinear when bitnet.enabled
if separate: Qg,Kg,Vg ← xW_qg, xW_kg, xW_vg
k̃, ṽ ← K,V;  k̃[G], ṽ[G] ← Kg[G], Vg[G]   (if separate)
M ← window(w) ∪ columns(G) ∪ rows(G)      (∩ causal if decoder)
Z ← (Q·k̃ᵀ/√d) masked by M;  if separate: Z[G,:] ← Qg[G]·Kgᵀ/√d (causal-clipped)
O ← softmax(Z)·ṽ;  Y ← O·W_o
```

**Incremental / decode form (RingSlot, exact):**

```
state: ring of last (capacity−pinned) (k,v) + pinned head region covering G
per step: append (k_t, v_t); evict oldest non-pinned
q_t attends: all cached positions  (retention policy == attention pattern:
             pinned G + recent window — decode needs no extra mask,
             swa_attn precedent; multi-token chunks pattern-mask over
             cached absolute positions, B5)
```

### 2.4 Parameter inventory and exact parameter-count formula

| Tensor | Shape | Bias | Count |
|---|---|---|---|
| `q_proj` (`W_Q`) | H×H | no | H² |
| `k_proj` (`W_K`) | H×H | no | H² |
| `v_proj` (`W_V`) | H×H | no | H² |
| `out_proj` (`W_O`) | H×H | no | H² |
| `global_q/k/v_proj` | H×H each | no | 3H² — **only when `global_attn_separate=true`** |

No global-token embeddings exist (global tokens are **existing head positions**
of the input sequence — NeMo's design; the paper adds no new token vector).
No biases (repo convention). Head count does not enter the formula (plain MHA
H→H projections, like `standard_attn`/`longformer_attn`/`swa_attn`).

$$P_{\text{fastconformer}} = 4H^2 + 3H^2\cdot[\texttt{global\_attn\_separate}]$$

`ft-param-estimator.js` case (bias- and knob-aware):

```js
case 'fastconformer_attn': {
  var fcSep = toBool(pick(m, 'attention.fastconformer.global_attn_separate',
                          'fastconformer_global_attn_separate', false), false);
  return 4 * H * H + (fcSep ? 3 * H * H : 0);   // Q,K,V,O (+ global Q,K,V when separate)
}
```

`full_tests/param_count_check.py` needs no code change (it counts the real
model); `param_truth.json` is regenerated and the **intentional diff** includes
the new example + preset. To make the ground truth cover **both** parameter
paths, the example YAML sets `global_attn_separate: true` (the §2.2 variant)
while the full preset keeps the §3.4 shared default — see §3.

### 2.5 Knobs (schema → dataclass)

| Schema key | Flat dataclass field | Type | Default | Constraints / coercion |
|---|---|---|---|---|
| `model.attention.fastconformer.window_size` | `fastconformer_window_size` | int | `256` | ≥ 2; total local span, per-side = `//2`; paper §3.4: 128 per side → 256 |
| `model.attention.fastconformer.global_tokens` | `fastconformer_global_tokens` | int | `1` | ≥ 0; paper: "a single global attention token"; `0` = pure limited context (Table 8 row 2) |
| `model.attention.fastconformer.global_tokens_spacing` | `fastconformer_global_tokens_spacing` | int | `1` | ≥ 1; `G = {0, s, …, (g−1)s}` |
| `model.attention.fastconformer.global_attn_separate` | `fastconformer_global_attn_separate` | bool | `false` | §2.2 separate global Q/K/V (+3H²) vs §3.4 shared (user-confirmed default) |
| `model.positional_encoding_parameters.use_pe.fastconformer_attn` | `fastconformer_attn_use_pe` | bool | `true` | softmax-attention-like: consumes the model-wide PE |

### 2.6 State & cache

- **Compatibility class:** (iv) ring / sliding buffer (`src/model/cache/capabilities.py`
  lines 22–29). Global tokens live in a **bounded head region**, so eviction is
  safe — unlike `longformer_attn` (unbounded `KVSlot`, capabilities.py line
  229–236 "global tokens may sit anywhere (no safe eviction)").
- **Slot:** `RingSlot` (`src/model/cache/slots.py` line 571), `make_cache_slot`
  hook in the mixer (swa_attn.py lines 161–172 precedent):
  `num_sinks = max(0, (g−1)·s + 1)` (pinned prefix covering `G`; `0` when `g=0`),
  `capacity = num_sinks + w//2 + 1` (decoder recent span `i−w/2 ≤ j ≤ i` plus
  pinned globals). Retention policy == attention pattern, so single-token
  decode attends the whole cached set with **no mask** (swa precedent);
  multi-token prefill chunks pattern-mask over cached absolute `positions`
  (B5, longformer_attn.py lines 200–224 precedent). **KV-cache decision 1** —
  full incremental branch + `IMPLEMENTED_MIXERS` + auto parity via
  `tests/test_kv_cache_mixers.py` (which parametrizes `IMPLEMENTED_MIXERS`,
  line 27). The ring capacity formula must be verified by the parity test; if
  parity fails, widen `capacity` by the off-by-one (see §9).
- **Per-step decode cost:** O(w + g) scores over retained keys; O(1) state
  update (append + evict).

### 2.7 Complexity (for `tab:all-mixers`)

- Training: **O(n·(w+g)·d)** with the paper's overlapping-chunk kernel;
  the repo's masked-dense reference is O(n²) compute (mathematically identical;
  same caveat as longformer/swa rows).
- Inference decode: **O(w+g)** per token; bounded cache memory O(w+g) —
  the paper's headline (15 min → 675 min on one A100, Table 3).

### 2.8 Trainability

Trainable from scratch (mask change on the shared path + optional projections).
**Not** `TRAINING_FREE_LAYERS` (hard rule 5 does not apply): like `swa_attn`
("training-free but a normal trainable mixer", section 04 line 71), the
windowed+global pattern runs in **both** train and eval. The paper fine-tunes
from a full-attention checkpoint (§3.4); from-scratch training is the repo's
default path and is what the e2e smoke does. Joins `full_tests/run_e2e.py`
`ATTENTIONS` (trainable list, lines 224–245).

### 2.9 Positional encoding interaction

NeMo's implementation uses Transformer-XL relative PE; in this repo PEs are
transversal and injected per-mixer (`apply_pe_to_qk` / `apply_pe_to_scores`),
so the mixer consumes the model-wide PE exactly like `longformer_attn` /
`swa_attn` (both default `use_pe=true`). `per_mixer_use_pe` default:
**`true`**.

### 2.10 Mode constraints

Encoder: symmetric window (paper-native — the ASR encoder is bidirectional).
Decoder: causal clipping of window + global columns + global rows (repo
extension, longformer_attn.py lines 242–244 precedent). No grid/vision
restriction (unlike `ssog_attn`); composes with `frankenstein_audio` and the
ViT as a generic HybridLayer mixer, but no audio/vision preset is added (the
paper's audio frontend changes are out of scope). Rule 6: no forced modes.

### 2.11 BibTeX data

- Authors: Dima Rekesh, Nithin Rao Koluguri, Samuel Kriman, Somshubra Majumdar,
  Vahid Noroozi, He Huang, Oleksii Hrinchuk, Krishna Puvvada, Ankur Kumar,
  Jagadeesh Balam, Boris Ginsburg.
- Title: *Fast Conformer with Linearly Scalable Attention for Efficient Speech
  Recognition*. Year: 2023. arXiv: 2305.05084 (eess.AS). Key:
  **`rekesh_fastconformer_2023`** (entry drafted in row 21 below, following the
  `kubicek_lsa_2026` style with `eprint` fields — attention_types.bib lines
  495–503).

### 2.12 Empirical numbers (for the annex "Empirical Performance")

- 2.8× faster inference than the equivalent Conformer; 2.9× fewer MACs;
  LS test-other WER 5.19 (Conformer-RNNT) → 4.99 (FC-RNNT); encoder params
  115 M → 109 M (Table 2/4).
- Long-form (Table 8, normalized greedy WER): TED-LIUM v3 9.18 (Conformer) /
  9.15 (FC buffered) / 8.25 (FC + limited context) / **7.5 (FC + limited
  context + global token)**; Earnings-21: 18.26 / 17.65 / 16.08 / **11.85**.
- Max single-pass duration (Table 3): 15 → 25 → 135 → **675 min** (A100, bs=1).
- §3.4 fine-tune recipe: 10k steps, LR warmup 1k, peak 1e-6, cosine to 0;
  window 128/side (~10 s); shared projections.

---

## 3. Decisions (Phase 3, user-confirmed where marked)

1. **Registry name:** `fastconformer_attn` — class **`FastConformerAttention`**,
   file `src/model/attention/sparse/fastconformer_attn.py`.
   *(User-confirmed.)* The paper's mechanism acronym "LSA" (Linearly Scalable
   Attention) collides with the existing `lsa_attn` (Native-LSA,
   arXiv:2610.42147, registry line 231) — grep confirms `fastconformer` is
   unused everywhere in the repo.
2. **Family:** `sparse/` directory, census label **Sparse** (12th member).
   Windowed masked attention with pinned global tokens is the family's core
   pattern (peers: `sparse_transformer_attn`, `longformer_attn`, `swa_attn`,
   `bigbird_attn`). No new family ⇒ no new `mixer-taxonomy.mmd` branch class,
   no new family-map figure, no new `ft-diagram.js` `STYLES` class.
3. **Closest peers & what is mirrored:**
   - `src/model/attention/sparse/swa_attn.py` (primary): constructor signature
     `(config, pos_encoder=None)`, knob validation, `_build_mask`, BitLinear
     projections, `attach_qk_norm` + `record_qk_max_logits`, `module_softmax`,
     PE wiring (`apply_pe_to_qk`/`apply_pe_to_scores`), `make_cache_slot`
     (RingSlot), cached forward with `kv_decode_append` + `pattern_mask`.
   - `src/model/attention/sparse/longformer_attn.py` (global-token semantics):
     global rows ∪ columns in the mask (lines 121–133, 200–214), decoder causal
     intersection (lines 242–244), cached `_pattern` over absolute positions.
   - Extra beyond both peers: the optional separate global Q/K/V projections
     (§2.2) and the deterministic head-placement rule for global tokens.
4. **Knobs:** 4 (§2.5). `global_attn_separate` default **`false`**
   (§3.4 production form, NeMo default; user-confirmed) — the §2.2
   separate-projection variant is opt-in. `window_size=256` (paper 128/side,
   longformer `window_size//2` convention), `global_tokens=1` (paper: single
   global token), `global_tokens_spacing=1`.
5. **`use_pe` default:** `true` (+ `fastconformer_attn_use_pe: bool = True`).
6. **KV-cache decision 1 (user-confirmed):** `CacheSpec(supported=True,
   slot_type="ring", extras=("global_tokens",), notes=…)` + `IMPLEMENTED_MIXERS`
   membership + incremental branch (RingSlot) + auto parity
   (`test_kv_cache_mixers.py`). Evidence: global tokens occupy a bounded head
   region, so ring eviction is exact; swa_attn precedent (capabilities.py line
   237–244); the paper's bounded-memory claim makes `ring` the faithful class.
7. **Training-free?** No. Trainable; not in `TRAINING_FREE_LAYERS`; joins
   e2e `ATTENTIONS`.
8. **Example preset:** `configs/examples/es_arch_fastconformer_adamw.yaml`
   (small Spanish MLM encoder, `es_arch_lsa_adamw.yaml` shape) — sets
   `global_attn_separate: true` **on purpose** so the param ground truth and
   estimator both exercise the 7H² path; a header comment states it is the
   §2.2 variant. **Full preset (user-confirmed):** `configs/fastconformer.yaml`
   — Fast Conformer-L proportions (hidden 512, 17 layers, Table 9) as a text
   MLM encoder with the §3.4 evaluated attention recipe (shared projections,
   window 256, single global token), mirroring `configs/lsa.yaml`'s
   paper-recipe header style. Both join the smoke-tested preset set.
9. **Census bumps** (actual numbers read from the repo — the LSA commit
   `6c5868b` is the reference blueprint):
   - Global mixer census **47 → 48** (enum, `test_schema_attention_layers.py`
     line 155, `test_kv_cache_fallback.py` line 62, capabilities.py docstring
     lines 6–7 "47 total: 46 mixer classes + mamba", architecture.md §7.4 line
     351, README badge line 20 + lines 97/114/187, annex-summary-tables line 4
     EN + ES line 4, abstracts `paper.tex:63` + `paper-es.tex:63`,
     `01-introduction.tex:14` + `01-introduccion.tex:14`,
     `04-architecture-taxonomy.tex:7,24` + `04-taxonomia-arquitectura.tex:7,24`,
     `10-discussion.tex:16` + `10-discusion.tex:16`,
     `11-conclusion.tex:8` + `11-conclusion.tex:8`,
     `mixer-taxonomy.mmd` root label line 3).
   - Sparse family **11 → 12** (README line 139, taxonomy line 9
     `s["Sparse (11)"]`, annex-summary-tables line 45 EN + ES line 45,
     annex-4 line 771 "eleven sparse attention methods" + ES counterpart at
     ~line 771, `04-architecture-taxonomy.tex:71` "eleven sparse blocks" +
     `04-taxonomia-arquitectura.tex:66` "once bloques dispersos", plus the
     exec summary at annex-4 line 6 which is stale at "ten" — it misses LSA
     and must become "twelve" including LSA and Fast Conformer).
   - Cache census (annex-15 line 20 EN + ES line 20): "forty-six dispatchable
     layer types" → "forty-seven"; "Twenty-eight have parity-tested" →
     "Twenty-nine". architecture.md §7.13 line 639 "(45)" is **stale**
     (pre-existing drift; should already be 47) — bump to 48 in the census
     pass. architecture.md §1 line 25 "**46** (45 concrete mixers + mamba)" →
     "**48** (47 concrete mixers + `mamba`)", line 38 "46 mixers" → "48
     mixers", line 82 "46 mixers" → "47 mixers" (class count; all three are
     stale-at-46 pre-existing drift, listed here because row 28 requires
     every census occurrence).
   - See §9 for occurrences deliberately **not** touched (related-work "43",
     Sphinx missing `swa_attn.rst`, `swa_attn` missing from the annex-schema
     mixer table, `test_kv_cache_fallback.py` docstring "46 values").

---

## 4. File-by-file plan table

Execution order top to bottom. "n/a" rows are omitted. Anchors are current
line numbers; the implementer must re-locate by content (line drift risk, §9).

| # | File | Planned change (be concrete) |
|---|---|---|
| 1 | `src/model/attention/sparse/fastconformer_attn.py` | **New module** `FastConformerAttention` (≈330 lines). Header docstring: paper citation (Rekesh et al., 2023, arXiv:2305.05084), one-paragraph mechanism description quoting §2.2/§3.4, `Reference:` block like swa_attn.py lines 17–19. `from __future__ import annotations`; imports mirror swa_attn.py lines 22–32 (`BitLinear, apply_pe_to_qk, apply_pe_to_scores` from `..common`; `attach_qk_norm, record_qk_max_logits` from `...norm`; `module_softmax` from `...numerics`). `__init__(self, config, pos_encoder=None)`: read `fastconformer_window_size` (default 256, `int(getattr(config, "fastconformer_window_size", 256))`), `fastconformer_global_tokens` (1), `fastformer_global_tokens_spacing` (1), `fastconformer_global_attn_separate` (False), `mode`; validate `window_size ≥ 2`, `global_tokens ≥ 0`, `spacing ≥ 1` (raise `ValueError`, swa_attn.py lines 106–114 style); store `self.global_positions = [(i*s) for i in range(g)]` as a rule (computed per forward against `seq_len`, never a fixed list); build `q_proj/k_proj/v_proj/out_proj` = `proj_cls(H, H, bias=False)` (`BitLinear` when `use_bitnet`); **when separate**: also `global_q_proj/global_k_proj/global_v_proj` (3 more H×H, BitLinear under BitNet — this is what `test_bitnet_coverage.py` greps); `self.pos_encoder`, `self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()`, `self.use_pe = bool(getattr(config, "fastconformer_attn_use_pe", True))`, `attach_qk_norm(self, config, self.head_dim)`. `_build_mask(seq_len, device)`: vectorized (arange outer ops, swa_attn.py lines 146–159 style, **not** longformer's python loop): `window = |i−j| ≤ w//2`; `is_global_col = j ∈ G`; `is_global_row = i ∈ G`; encoder mask = `window ∪ col ∪ row`; decoder = `(window ∪ col ∪ row) ∩ (j ≤ i)` — i.e. global columns causal-clipped, global rows full-causal. `make_cache_slot(meta, cache)`: `RingSlot(meta, capacity=self.num_sinks + self.window_size // 2 + 1, num_sinks=self.num_sinks, **codec/policy kwargs)` where `num_sinks = max(0, (g−1)*s + 1)`; swa_attn.py lines 161–172 pattern; docstring states retention == pattern. `forward(x, logical_layer_idx=None, pos_encoder=None, cache=None, cache_position=None)`: (a) project q/k/v (and qg/kg/vg when separate), `qk_norm`, head split `(B,h,n,d)`; (b) **cached path** (slot kind `kv`/`ring`, swa_attn.py lines 220–259 pattern): `positions_for(offset, seq_len)`, `apply_pe_to_qk(..., position_offset=offset)`, `kv_decode_append`, scores over cached K, `_pattern(q_pos, kv_pos)` = `(window over positions) ∪ (kv_pos ∈ G ∧ kv_pos ≤ q_pos) ∪ (q_pos ∈ G ∧ kv_pos ≤ q_pos)`; when separate, cached K must hold the *effective* keys: append `k̃` (globals swapped to kg) — simplest exact approach: swap global rows of `k`/`v` with `kg`/`vg` **before** append (works because global positions are always in the cached pinned region; note it in the docstring); `seq_len == 1` → no mask (retention == pattern); else `pattern_mask(...)`; `apply_pe_to_scores(...)`, `record_qk_max_logits`, `module_softmax`, dropout, `out_proj`. Global-row decode: new decode tokens are never global (G at head), so cached decode never needs the global-row override — full-causal global rows only occur during prefill/multi-token chunks, handled by `_pattern`; (c) **dense path**: `apply_pe_to_qk`, build `k̃/ṽ` (swap globals when separate), `Z = q·k̃ᵀ·scale`; when separate, recompute rows `Z[G,:] = qg[G]·kgᵀ·scale` (all columns; NeMo `_compute_out_global_to_all` semantics) and use `ṽ = vg` for those rows' value aggregation — implement as: `out_local = softmax(mask(Z_local))·ṽ`; `out_global_rows = softmax(mask(qg[G]·kgᵀ))·vg`; scatter back; single softmax per row either way. `~mask → -inf`, decoder causal intersection, `apply_pe_to_scores`, `record_qk_max_logits`, `module_softmax`, dropout, `out_proj`, return `(B,n,H)`. Call shapes S1/S2 per §7.13 (same signature as peers). CPU-safe (no flash kernels). |
| 2 | `src/model/attention/sparse/__init__.py` + `src/model/attention/__init__.py` | sparse `__init__.py`: add `from .fastconformer_attn import FastConformerAttention` (alphabetical, after `.fasa_attn` import line 2) and `"FastConformerAttention",` in `__all__` (list at lines 13–25). Top-level `__init__.py`: add to the `from .sparse import (` tuple (lines 5–17, after `FASAAttention,`) and `"FastConformerAttention",` in `__all__` (after `"FASAAttention",` line 73). |
| 3 | `src/model/hybrid_layer.py` | Import `FastConformerAttention` in the `from .attention.sparse import (...)` block at line 53; register `"fastconformer_attn": FastConformerAttention,` in `mixer_registry` after `"lsa_attn": LSAAttention,` (line 231). **Not** added to `TRAINING_FREE_LAYERS` (line 123) — the mixer trains its own pattern. |
| 4 | `src/model/config.py` | (a) Docstring: add `fastconformer_attn_use_pe: Whether ``fastconformer_attn`` layers consume the model-wide positional encoding (windowed softmax attention, default True).` in the per-mixer use_pe doc block (after the `swa_attn_use_pe` line ~322). (b) Fields: `fastconformer_attn_use_pe: bool = True` in the use_pe block (after line 618); and a new knob block after the SWA block (lines 792–797), quoting swa's style: `# ---- Fast Conformer limited-context attention (arXiv:2305.05084) ----` with `fastconformer_window_size: int = 256` (comment: total local span w, per-side w/2; paper §3.4: 128 per side), `fastconformer_global_tokens: int = 1` (leading global attention tokens g; paper: single), `fastconformer_global_tokens_spacing: int = 1` (stride s between global positions), `fastconformer_global_attn_separate: bool = False` (§2.2 separate global Q/K/V, +3H²; §3.4 shared default). (c) Validation in `__post_init__` next to the swa checks (lines 1122–1134): `window_size < 2` → raise; `global_tokens < 0` → raise; `global_tokens_spacing < 1` → raise. |
| 5 | `src/utils/config_flatten.py` | (a) Comment inventory at line ~42: add `fastconformer: {window_size, global_tokens, global_tokens_spacing, global_attn_separate}`. (b) `_ATTENTION_MIXER_RENAMES` (line 76): after the `"swa"` block (lines 163–165) add `"fastconformer": {"window_size": "fastconformer_window_size", "global_tokens": "fastconformer_global_tokens", "global_tokens_spacing": "fastconformer_global_tokens_spacing", "global_attn_separate": "fastconformer_global_attn_separate"}`. |
| 6 | `src/schema/_model/_dims.yaml` | (a) Enum (items.enum, line 334): add `- fastconformer_attn` after `- lsa_attn` (line 381). (b) EN table: new row after the `lsa_attn` row (line 263): `` | `fastconformer_attn` | Fast Conformer limited-context attention — local window + head-pinned global token(s), optional separate global projections (paper: arxiv:2305.05084) | ✅ | ✅ | ``. (c) ES table: new row after the ES `lsa_attn` row (line 321): `` | `fastconformer_attn` | Atención de contexto limitado Fast Conformer — ventana local + token(s) global(es) anclado(s) a la cabecera, proyecciones globales separadas opcionales (paper: arxiv:2305.05084) | ✅ | ✅ | ``. Both tables are scanned by `test_layer_pattern_enum_matches_documentation` (test_schema_consistency.py lines 66–73). No ⚠️ wording (trainable). |
| 7 | `src/schema/_model/_attention_fastconformer.yaml` | **New knob file** (template: `_attention_swa.yaml`). Drafted fully in §4.1 below: `type: object`, `additionalProperties: false`, `properties.fastconformer` with the 4 leaves, `title`/"title_es", bilingual `description`/`description_es` with `**References:** (paper: arxiv:2305.05084)` + `(paper: arxiv:2004.05150)` where the Longformer pattern is referenced, per-leaf `minimum`/`default`/`examples`, examples-explained bullets citing §2.2/§3.4 values (window 256 = 128/side, g=1, s=1, separate=false=§3.4, true=§2.2 +3H²). Defaults must match the dataclass exactly (`_DEFAULT_PAIRS` gate). |
| 8 | `src/schema/_model.yaml` | Under `attention.properties` (line 69): add after the `swa:` entry (lines 102–103): `fastconformer:` + `$ref: "_model/_attention_fastconformer.yaml#/properties/fastconformer"`. **No description-list edit**: the `description`/`description_es` lists (lines 50–54 / 60–64) enumerate mixers *without* knobs only (`fastconformer_attn` has a sub-object; the LSA commit likewise did not add `lsa_attn` there). |
| 9 | `src/schema/_model/_positional_encoding.yaml` | `use_pe` map (lines 376–422): add after the `swa_attn` entry (line 415): `fastconformer_attn: { type: boolean, title: "fastconformer_attn", description: "Model-wide PE for `fastconformer_attn` layers (paper: arxiv:2305.05084).", default: true, description_es: "PE a nivel de modelo para capas `fastconformer_attn` (paper: arxiv:2305.05084)." }`. The false-default mixer lists (lines 24, 385, 393) are **not** touched (fastconformer defaults true). |
| 10 | `src/model/cache/capabilities.py` | (a) Docstring count (lines 6–7): "47 total: 46 mixer classes + mamba" → "48 total: 47 mixer classes + the ``mamba`` linear placeholder". (b) `IMPLEMENTED_MIXERS` (lines 68–105): add `"fastconformer_attn",  # F1 (ring retention == window + pinned globals)` in the sparse group after `"swa_attn"`. (c) `CACHE_SPECS` sparse section (after the `swa_attn` spec, lines 237–244): `CacheSpec(supported=True, slot_type="ring", extras=("global_tokens",), notes="Fast Conformer limited-context attention (arXiv:2305.05084); window + global tokens pinned at the head (Longformer-inspired, separate global projections optional). Global positions live in a bounded head region, so a ring slot with the pinned prefix covering G retains exactly the attention pattern (swa precedent); decode needs no mask.")`. The `test_cache_specs_match_layer_pattern_enum` gate requires `set(CACHE_SPECS) == enum` with len 48 (test_schema_attention_layers.py lines 144–156). |
| 11 | `configs/examples/es_arch_fastconformer_adamw.yaml` | **New example** (mirror `es_arch_lsa_adamw.yaml`, 84 lines): `model_class: frankenstein_text_encoder`; `model.dims`: vocab 50000, hidden 512, num_layers 10, num_loops 1, num_heads 8 (no `num_kv_heads` — plain MHA), `layer_pattern: [fastconformer_attn]`; `norm: layer_norm`; `embedding.conv.enabled: false`; `attention:` block with `fastconformer: {window_size: 256, global_tokens: 1, global_tokens_spacing: 1, global_attn_separate: true}` (header comment: §2.2 Longformer-style separate global projections — exercises +3H² in the param gates) and `retnet: {heads: 8}` (peer convention, es_arch_lsa lines 26–27 / es_arch_gma line 24); `moe.enabled: false`, `bitnet.enabled: false`, `ffn: {hidden_size: 2048, activation: gelu}`; `training:` block copied verbatim from `es_arch_lsa_adamw.yaml` lines 35–84 (mlm, seed 42, max_length 512, the adamw per-group parameters, `csv_log_path: training_metrics_es_arch_fastconformer_adamw.csv`). Must pass `load_training_config` + `test_all_presets_validate` + `test_yaml_examples` + `param_count_check.py`. |
| 12 | `configs/fastconformer.yaml` | **New full preset** (user-approved; `configs/lsa.yaml` shape, ~116 lines). Header comment: "Fast Conformer reference configuration (arXiv:2305.05084 §3.4, Table 9 FC-L proportions adapted to text MLM): every layer is limited-context attention — 128-token window per side (`window_size: 256`) plus a single head-pinned global attention token with shared projections (the paper's evaluated long-form recipe); the paper's RNNT/8×-subsampled audio frontend is out of scope for the text model class." `model_class: frankenstein_text_encoder`; `model.dims`: vocab 32000, hidden 512, num_layers 17, num_heads 8, dropout 0.1, mode encoder, `layer_pattern: [fastconformer_attn ×17]`; `norm: layer_norm`; `positional_encoding: rope`; `attention.fastconformer: {window_size: 256, global_tokens: 1, global_tokens_spacing: 1, global_attn_separate: false}`; `ffn: {hidden_size: 2048, activation: gelu}`; moe/bitnet off; `training:` mirrors `configs/lsa.yaml` lines 77–116 with `task: mlm`, `mlm_probability: 0.15`, `max_length: 512`, the same adamw per-group parameter block, `scheduler_type: cosine`, `scheduler_total_steps: 10000`. Joins the smoke-tested preset set (`test_all_presets_validate` auto-globs `configs/*.yaml`). |
| 13 | `full_tests/run_e2e.py` | Append to `ATTENTIONS` (lines 224–245, after `"lsa_attn",` line 244): `# Fast Conformer limited-context attention (arXiv:2305.05084): local`, `# window + head-pinned global token; trains its windowed pattern natively in`, `# encoder and decoder modes; exact ring-cache incremental decoding.` + `"fastconformer_attn",`. This auto-generates `attn_single_fastconformer` (via `make_single_attention_configs`, line 756) and every pair `attn_<a>__fastconformer` / `attn_fastconformer__<b>` (via `make_attention_pair_configs`, line 741, `itertools.combinations`) — **no builder edits**. |
| 14 | `tests/test_schema_attention_layers.py` | (a) `expected` set (lines 18–66): add `"fastconformer_attn",`. (b) `expected_per_mixer` (lines 90–128): add `"fastconformer": {"window_size", "global_tokens", "global_tokens_spacing", "global_attn_separate"},`. (c) `test_cache_specs_match_layer_pattern_enum`: `self.assertEqual(len(enum_values), 47)` → `48` (line 155). |
| 15 | `tests/test_sparse_attention.py` | New `FastConformerAttentionTests(unittest.TestCase)` after `SlidingWindowAttentionTests` (class at line 108) / `LongformerAttentionTests` (line 76), mirroring their helpers: `test_forward_shape_encoder` and `test_forward_shape_decoder` (output `(1, n, vocab)`), `test_global_token_attended_by_all` (longformer lines 97–107 style: column of the mask fully true), `test_global_token_attends_to_all` (row fully true, causal-clipped in decoder), `test_decoder_causal_clips_future` (no future key visible, incl. future globals), `test_shared_projections_add_no_params` (param count == 4H² by construction — module has exactly q/k/v/out), `test_separate_projections_add_three_projections` (+3H² params, all BitLinear under BitNet), `test_window_larger_than_sequence_is_full_plus_globals`, `test_invalid_window_raises` (`window_size < 2`), `test_invalid_global_tokens_raises` (negative), `test_no_nan_or_inf` (fp32 forward), `test_bitnet_uses_bitlinear` (all six projections BitLinear when `use_bitnet=True`). |
| 16 | `tests/test_bitnet_coverage.py` | Append `"fastconformer_attn",` to the `_make` `layer_pattern` list (lines 37–66, after `"gma_attn",`). The gate asserts every projection child is BitLinear — covers the optional global projections when the fixture enables them (add `fastconformer_global_attn_separate=True` to `_make`'s config if the shared fixture path does not exercise them; keep the edit minimal and mirror how lsa's knobs were covered). |
| 17 | `tests/test_kv_cache_mixers.py` | **No structural edit expected**: the suite parametrizes `IMPLEMENTED_MIXERS` (line 27) so `fastconformer_attn` joins `test_every_implemented_mixer_parity` (line 92) automatically across `rope/nope/alibi`. Only add a fixture/method if parity reveals a need (e.g. a ring-retention assertion class like LSA's `LSABucketParityTests`, commit 6c5868b); keep it minimal. |
| 18 | `tests/test_attention_refactor.py` | `test_layer_type_coverage_trainable` list (lines 141–172): add `"fastconformer_attn",` (after `"swa_attn",` line 151). |
| 19 | `tests/test_kv_cache_fallback.py` | `self.assertEqual(len(enum), 47)` → `48` (line 62). The docstring "46 values" (line 28) is stale — leave it or fix in the same edit (flagged in §9; fixing is a one-word count correction in a file already being touched). |
| 20 | `tests/test_schema_consistency.py` | `_DEFAULT_PAIRS` (lines 153–233): add after the swa pairs (lines 199–200): `"properties.model.properties.attention.properties.fastconformer.properties.window_size": "fastconformer_window_size"`, `…global_tokens": "fastconformer_global_tokens"`, `…global_tokens_spacing": "fastconformer_global_tokens_spacing"`, `…global_attn_separate": "fastconformer_global_attn_separate"`. (The enum↔table test and preset validation need no edit — they scan the schema.) |
| 21 | `src/model/norm/qk.py` | Add `"fastconformer_attn",` to `QK_NORM_MIXERS` (frozenset at lines 46–…) next to `"swa_attn"` / `"longformer_attn"` (lines 52–53) — the module calls `attach_qk_norm` + `record_qk_max_logits`, so QK-norm must validate. `QK_CLIP_MIXERS = QK_NORM_MIXERS - {"gma_attn"}` (line 79) inherits it automatically. Update `configs/README.md`'s QK list accordingly (row 28). |
| 22 | `docs/paper/figures/mixers/fastconformer.mmd` | **New mermaid** in house style (swa.mmd template, init header `%%{init: {"theme":"neutral","flowchart":{"curve":"basis","nodeSpacing":18,"rankSpacing":26,"useMaxWidth":false}}}%%`, `flowchart TD`, tensor-shaped labels, labelled edges): nodes `x["x<br/>[B,n,H]"]`, `qkv["Q, K, V<br/>[B,n,H]"]`, `gproj["Qg, Kg, Vg<br/>[B,n,H]<br/>(optional)"]`, `win["local<br/>window w"]`, `glo["head tokens 0,s,…<br/>g global"]`, `s["masked scores =<br/>(w+g)⊗QKᵀ"]`, `sm["softmax"]`, `o["O = Wo·(softmax·V)<br/>[B,n,H]"]`; edges `x -- "Wqkv" --> qkv`, `x -- "Wg (separate)" --> gproj`, `x -- "window" --> win`, `x -- "global rows/cols" --> glo`, `qkv -- "mask" --> s`, `gproj -- "global rows/cols" --> s`, `win -- "mask" --> s`, `glo -- "mask" --> s`, `s -- "softmax" --> sm`, `sm -- "Wo" --> o`, `qkv -- "V" --> o`. Rendered by `build-figures.sh` (auto-globs every `.mmd`) into `.pdf`+`.svg`. |
| 23 | `docs/paper/figures/mixer-taxonomy.mmd` | Line 3: root label `"Sequence Mixer Registry (47 variants)"` → `"(48 variants)"`. Line 9: `s["Sparse (11)"]` → `s["Sparse (12)"]` and extend the chain: `… --> s11["lsa_attn"] --> s12["fastconformer_attn"]`. Re-render pdf+svg. |
| 24 | `docs/paper/figures/sparse-map.mmd` | Line 6: `local["Local / windowed<br/>Longformer / SWA with sinks"]` → `local["Local / windowed<br/>Longformer / SWA with sinks<br/>FastConformer"]`. Re-render. |
| 25 | `docs/bibliography/attention_types.bib` | Add after `jolicoeur_sliding_window_2026` (lines 98–105) or in arXiv-id order near the sparse entries: `@misc{rekesh_fastconformer_2023,` `title = {Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition},` `author = {Rekesh, Dima and Koluguri, Nithin Rao and Kriman, Samuel and Majumdar, Somshubra and Noroozi, Vahid and Huang, He and Hrinchuk, Oleksii and Puvvada, Krishna and Kumar, Ankur and Balam, Jagadeesh and Ginsburg, Boris},` `year = {2023},` `eprint = {2305.05084},` `archivePrefix = {arXiv},` `primaryClass = {eess.AS},` `url = {https://arxiv.org/abs/2305.05084},` `publisher = {{arXiv}}}` (kubicek_lsa_2026 field style, lines 495–503). |
| 26 | `docs/paper/appendices/annex-4-sparse-attention.tex` | (a) Executive Summary (line 6): "ten major sparse attention families---Sparse Transformer, Longformer, SWA with attention sinks, BigBird, SparseK, NSA, FASA, SpargeAttn, MSA, and SparDA" is **stale (misses LSA)** — rewrite to "twelve major sparse attention methods---Sparse Transformer, Longformer, SWA with attention sinks, Fast Conformer, BigBird, SparseK, NSA, FASA, SpargeAttn, MSA, SparDA, and LSA". (b) New `\subsection{Fast Conformer: Limited-Context Attention with a Head-Pinned Global Token}` inserted after the LSA subsection (ends line 767) and before `\subsection{Design Space and Selection Criteria}` (line 769), mirroring the LSA subsection structure exactly (lines 689–767): `\begin{figure}[H]` + `\fitfigure{\includegraphics{figures/mixers/fastconformer.pdf}}` + caption + `\label{fig:mixer-fastconformer}`; `\subsubsection{Mathematical Formulation}` with the §2.2 prose quote, the equations from plan §2.2 (pattern M_ij, shared scores, separate global rows, effective keys/values swap), a remark that the ICASSP paper defines the mechanism in prose and the global-attention dataflow follows Longformer's design \citep{beltagy_longformer_2020} as the paper directs; `\subsubsection{Algorithmic Pseudocode}` `algorithm` float (masked-dense parallel form + ring decode form, plan §2.3); `\subsubsection{Key Characteristics}` itemize: Complexity O(n·(w+g)·d) chunked / masked-dense reference caveat; Trainable (yes, natively; the paper fine-tunes from full attention in 10k steps at LR 1e-6); Strength (Table 8: TED-LIUM 9.18→7.5, Earnings-21 18.26→11.85; Table 3: 15→675 min single-pass; 2.8× encoder speedup); Limitation (no closed-form equations in the paper — reconstruction documented; single global token by default; masked-dense reference is O(n²) compute). `\citep{rekesh_fastconformer_2023}` throughout. (c) Design Space (line 771): "The eleven sparse attention methods" → "The twelve sparse attention methods"; add one itemize bullet: `\item \textbf{Window + pinned global broadcast} (Fast Conformer): the Longformer pattern with global tokens fixed at the head — bounded-memory long-form decoding with a single always-visible summary channel, proven at 11-hour scale.` |
| 27 | `docs/paper/sections/04-architecture-taxonomy.tex` | Line 7: "forty-seven" → "forty-eight". Line 24 caption: "forty-seven" → "forty-eight". Line 71: "eleven sparse blocks" → "twelve sparse blocks"; append `rekesh_fastconformer_2023` to the cite list; append one sentence after the LSA sentence: "Fast Conformer is the family's long-form broadcast pattern: a fixed local window plus a global attention token pinned at the sequence head, optionally with its own query/key/value projections, giving bounded-memory decoding at hours-long context." |
| 28 | `docs/paper/appendices/annex-summary-tables.tex` | Line 4: "47 sequence mixers across ten families" → "48 …". Line 45: `\textbf{Sparse (11)}` → `\textbf{Sparse (12)}`. New row after the `lsa_attn` row (line 57): `fastconformer\_attn & Sparse & $\mathcal{O}(n \cdot (w + g) \cdot d)$ & $\mathcal{O}(w + g)$ & Limited-context window + head-pinned global token; optional separate global projections; linear-memory long-form & \citep{rekesh_fastconformer_2023} \\` |
| 29 | `docs/paper/appendices/annex-schema.tex` | (a) Knob block after the `model.attention.lsa.*` block (lines 146–154): `\multicolumn{3}{l}{\textit{model.attention.fastconformer.*}} \\` + 4 rows: `attention.fastconformer.window\_size & int $\ge 2$ & Total local span $w$; per-side context $w/2$ (paper: 128 each side). Default: 256. \\`, `attention.fastconformer.global\_tokens & int $\ge 0$ & Leading global attention tokens $g$ (paper: 1; 0 = pure limited context). Default: 1. \\`, `attention.fastconformer.global\_tokens\_spacing & int $\ge 1$ & Stride $s$ between global positions $\{0, s, \dots, (g{-}1)s\}$. Default: 1. \\`, `attention.fastconformer.global\_attn\_separate & bool & Separate global Q/K/V projections ($+3H^2$; §2.2) vs shared (§3.4). Default: false. \\`. (b) Mixer table: add after the `lsa\_attn` row (line 261): `fastconformer\_attn & Fast Conformer --- limited-context window + head-pinned global token & \checkmark & \checkmark \\` |
| 30 | `docs/paper/appendices/annex-15-kv-cache.tex` | Line 20: "forty-six dispatchable layer types" → "forty-seven dispatchable layer types"; "Twenty-eight have parity-tested incremental branches" → "Twenty-nine have parity-tested incremental branches". (fastconformer joins the implemented majority; the fallback table is untouched.) |
| 31 | `docs/paper-es/appendices/annex-4-sparse-attention.tex` | Faithful Spanish translation of row 26 (same insert point: after the LSA subsection ending before `\subsection{Espacio de Diseño y Criterios de Selección}` at line 769): `\subsection{Fast Conformer: Atención de Contexto Limitado con un Token Global Anclado a la Cabecera}`, figures referenced from `../paper/figures/` exactly as the other ES annexes do (`\fitfigure` path unchanged — verify how annex-4 ES includes figures and mirror it), equations identical to EN, `\citep{rekesh_fastconformer_2023}`; also translate the exec-summary "twelve" fix and the design-space bullet ("Ventana + difusión global anclada (Fast Conformer): …"). |
| 32 | `docs/paper-es/sections/04-taxonomia-arquitectura.tex` | Line 7: "cuarenta y siete" → "cuarenta y ocho"; line 24 caption: same; line 66: "once bloques dispersos" → "doce bloques dispersos" + append `rekesh_fastconformer_2023` to the cite list + the Spanish sentence counterpart of row 27. |
| 33 | `docs/paper-es/appendices/annex-summary-tables.tex` | Line 4: "47 mezcladores" → "48 mezcladores"; line 45: `\textbf{Disperso (11)}` → `\textbf{Disperso (12)}`; new row after the `lsa\_attn` row (line 57): `fastconformer\_attn & Disperso & $\mathcal{O}(n \cdot (w + g) \cdot d)$ & $\mathcal{O}(w + g)$ & Ventana de contexto limitado + token global anclado a la cabecera; proyecciones globales separadas opcionales; forma larga de memoria lineal & \citep{rekesh_fastconformer_2023} \\` |
| 34 | `docs/paper-es/appendices/annex-schema.tex` | Spanish counterparts of row 29 (knob block after the `model.attention.lsa.*` block at lines 146–152; mixer-table row after `lsa\_attn`): translated descriptions, identical types/defaults. |
| 35 | `docs/paper-es/appendices/annex-15-cache-kv.tex` | Line 20: "cuarenta y seis tipos de capa despachables" → "cuarenta y siete tipos de capa despachables"; "Veintiocho tienen ramas incrementales verificadas por paridad" → "Veintinueve tienen ramas incrementales verificadas por paridad". |
| 36 | `docs/paper/paper.tex` + `docs/paper/sections/01-introduction.tex`, `10-discussion.tex`, `11-conclusion.tex` + `docs/paper-es/paper-es.tex` + `01-introduccion.tex`, `10-discusion.tex`, `11-conclusion.tex` | Census pass, one number per file: "forty-seven sequence mixers" → "forty-eight" (EN abstract paper.tex:63, 01-introduction:14, 10-discussion:16 "The forty-seven implemented mixers" → "forty-eight", 11-conclusion:8); ES: "cuarenta y siete" → "cuarenta y ocho" (paper-es.tex:63, 01-introduccion:14, 10-discusion:16, 11-conclusion:8). (These six + the two section-04 files are the same surfaces the LSA commit touched — its stat is the scope blueprint.) |
| 37 | `README.md` | Line 20 badge: `**47**` → `**48**` (Mixers column). Line 97: "47 across 10 categories" → "48 across 10 categories". Line 114: "all 47 mixer types" → "all 48 mixer types". Line 139 Sparse row: `**Sparse** (11)` → `**Sparse** (12)` and append `fastconformer_attn` to the code-name list + extend the description ("…and head-pinned global-token long-form patterns"). Line 187: "Mixer registry (47 variants / 10 families)" → "(48 variants / 10 families)". No ⚠️ note (trainable). |
| 38 | `docs/architecture.md` | (a) §1 line 25: "**46** (45 concrete mixers + `mamba`)" → "**48** (47 concrete mixers + `mamba`)" (fixes pre-existing staleness in the same census pass). (b) Line 38: "46 mixers / 11 positional encodings" → "48 mixers / 11 positional encodings". (c) Line 82: "46 mixers:" → "47 mixers:". (d) §7.4 line 351: "47 enum values in 10 families" → "48 enum values in 10 families". (e) §7.4 table: new row after `lsa_attn` (line 402): `` | `fastconformer_attn` | Sparse | `fastconformer.*` | yes | ``. (f) §7.4 prose: add a paragraph after the `lsa_attn` paragraph (lines 424–435), swa-paragraph style: "`fastconformer_attn` implements the limited-context attention of Fast Conformer from arXiv:2305.05084: every query attends to a symmetric local window (`fastconformer.window_size`, paper: 128 per side) plus `fastconformer.global_tokens` global tokens pinned at the head positions {0, s, …, (g−1)s} (`s = fastconformer.global_tokens_spacing`), which attend to everything and are attended by everything; `fastconformer.global_attn_separate` gives the global token its own q/k/v projections (§2.2, +3H²) instead of sharing the local ones (§3.4). The reference implementation masks a dense score matrix; the paper's overlapping-chunk kernel is out of scope. Cache: exact `RingSlot` (pinned head region covers the globals)." (g) §7.13 line 639: "(45)" → "(48)" (pre-existing staleness). (h) §7.13 implemented list (lines 645–654): add `fastconformer_attn` after `swa_attn`/`longformer_attn` entries. |
| 39 | `configs/README.md` | (a) Line 269 Sparse blocks list: append `, fastconformer_attn` (Fast Conformer limited-context attention, arXiv:2305.05084). (b) QK list (lines 302–305): add `fastconformer_attn` to the supported set. (c) `model.attention.<mixer>` sub-object list (line 338): add `fastconformer` after `lsa`. (d) New knob section after the LSA section (lines 418–428), same shape: `### Fast Conformer limited-context attention (model.attention.fastconformer)` with the 4 knobs, defaults, and the paper values; note `fastconformer_attn` uses plain MHA projections (no `num_kv_heads`). |
| 40 | `docs/source/api/model/attention/sparse/fastconformer_attn.rst` + `docs/source/api/model/attention/sparse/index.rst` | New rst (mirror `lsa_attn.rst`): title `FastConformer Attention` + `.. automodule:: src.model.attention.sparse.fastconformer_attn` `:members:` `:undoc-members:` `:show-inheritance:`. Add `fastconformer_attn` to the index toctree (after `lsa_attn`). |
| 41 | `full_tests/param_truth.json` | Regenerated (not hand-edited) by `full_tests/param_count_check.py` — the diff adding the two new configs is intentional and must be committed. |
| 42 | *(mirror, planned only)* `erickfmm.github.io/frankenstein-transformer/ft-diagram.js` | Site-only edit: (a) `CATS`: add `fastconformer_attn: { c: 'sparse', l: 'FastConformer Attn' },` next to the sparse entries (lines 22–40, e.g. after `lsa_attn` line 40). (b) `SUBSTRUCT`: add a tensor-flow template after `longformer_attn` (lines 283–291): `fastconformer_attn: [ { n: 'x [B,n,H]', c: 'sparse', id: 'x' }, { n: 'Q,K,V [B,n,H]', c: 'sparse', e: 'Wqkv', from: ['x'] }, { n: 'Qg,Kg,Vg [B,n,H] (optional)', c: 'sparse', e: 'Wg', from: ['x'] }, { n: 'local window w', c: 'sparse', e: 'local', from: ['x'] }, { n: 'head tokens g global', c: 'sparse', e: 'global', from: ['x'] }, { n: 'masked scores = (w+g)⊗QKᵀ', c: 'sparse', e: 'mask', from: ['Q,K,V [B,n,H]', 'local window w', 'head tokens g global', 'Qg,Kg,Vg [B,n,H] (optional)'] }, { n: 'softmax', c: 'sparse', e: 'softmax', from: ['masked scores = (w+g)⊗QKᵀ'] }, { n: 'O = Wo·softmax·V', c: 'sparse', e: 'Wo', from: ['softmax', 'Q,K,V [B,n,H]'] } ]`. (c) **No `STYLES`/`STYLES_PRINT` change** — the `sparse` class already exists (family reused). |
| 43 | *(mirror, planned only)* `erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js` | Site-only edit: add the `case 'fastconformer_attn':` block from plan §2.4 in `mixerParams` (switch at lines 83–95, next to the `longformer_attn`/sparse cases). |
| 44 | *(mirror, planned only)* `erickfmm.github.io/frankenstein-transformer/index.html` | **No mixer-list edit exists to make**: grep of the live page shows zero occurrences of peer mixer names (`longformer_attn`, `swa_attn`, `lsa_attn`) — the Builder reads the mirrored `schema.yaml`. Only probe-driven fixes per `/sync-website` §4 if any probe fails after the sync. |

### 4.1 Draft: `src/schema/_model/_attention_fastconformer.yaml`

```yaml
type: object
additionalProperties: false
properties:
  fastconformer:
    type: object
    additionalProperties: false
    title: "Fast Conformer Limited-Context Attention"
    title_es: "Atención de Contexto Limitado Fast Conformer"
    description: |-
      Hyperparameters for `fastconformer_attn` (Fast Conformer limited-context
      attention with a head-pinned global token (paper: arxiv:2305.05084)).
      Every query attends to a symmetric local window around it plus the first
      `global_tokens` positions of the sequence (spaced `global_tokens_spacing`
      apart), which attend to all positions and are attended by all positions.
      With `global_attn_separate: true` the global token uses its own
      query/key/value projections (+3*H^2 parameters, the paper's §2.2
      Longformer-style variant); with `false` it shares the local projections
      (the paper's §3.4 long-form model). In decoder mode the pattern is
      causal. Only applies when `fastconformer_attn` appears in `layer_pattern`.

      **References:** (paper: arxiv:2305.05084) (paper: arxiv:2004.05150)
    description_es: |-
      Hiperparámetros para `fastconformer_attn` (atención de contexto limitado
      Fast Conformer con un token global anclado a la cabecera (paper:
      arxiv:2305.05084)). Cada consulta atiende a una ventana local simétrica a
      su alrededor más las primeras `global_tokens` posiciones de la secuencia
      (espaciadas `global_tokens_spacing`), que atienden a todas las posiciones
      y son atendidas por todas. Con `global_attn_separate: true` el token
      global usa sus propias proyecciones query/key/value (+3*H^2 parámetros,
      la variante §2.2 del paper, estilo Longformer); con `false` comparte las
      proyecciones locales (el modelo de forma larga §3.4 del paper). En modo
      decoder el patrón es causal. Solo aplica cuando `fastconformer_attn`
      aparece en `layer_pattern`.

      **Referencias:** (paper: arxiv:2305.05084) (paper: arxiv:2004.05150)
    properties:
      window_size:
        type: integer
        minimum: 2
        title: "Fast Conformer Window Size"
        title_es: "Tamaño de Ventana Fast Conformer"
        description: |-
          Total local span `w` each query attends to in `fastconformer_attn`
          (paper: arxiv:2305.05084); the per-side context is `window_size / 2`.
          The paper's long-form model uses 128 steps on each side of a token
          (~10 seconds of audio after 8x subsampling), i.e. `window_size: 256`.
          Larger windows extend local context at linear cost in the attention
          budget; smaller windows are faster but lose local context.

          **Examples explained:**
          - `window_size=256`: the paper's §3.4 setting (128 per side).
          - `window_size=64`: short-context variant; the global token carries
            all long-range information.

          **References:** (paper: arxiv:2305.05084)
        description_es: |-
          Tramo local total `w` al que atiende cada consulta en
          `fastconformer_attn` (paper: arxiv:2305.05084); el contexto por lado
          es `window_size / 2`. El modelo de forma larga del paper usa 128
          pasos a cada lado del token (~10 segundos de audio tras el
          submuestreo 8x), es decir, `window_size: 256`. Ventanas mayores
          extienden el contexto local a costo lineal; ventanas menores son más
          rápidas pero pierden contexto local.

          **Ejemplos explicados:**
          - `window_size=256`: el ajuste §3.4 del paper (128 por lado).
          - `window_size=64`: variante de contexto corto; el token global
            transporta toda la información de largo alcance.

          **Referencias:** (paper: arxiv:2305.05084)
        default: 256
        examples: [64, 256]
      global_tokens:
        type: integer
        minimum: 0
        title: "Fast Conformer Global Tokens"
        title_es: "Tokens Globales Fast Conformer"
        description: |-
          Number `g` of leading global attention tokens in `fastconformer_attn`
          (paper: arxiv:2305.05084). Global tokens are the first `g` positions
          of the input sequence (spaced `global_tokens_spacing` apart): they
          attend to all positions and are attended by all positions. The paper
          uses "a single global attention token" (`g = 1`), which improves
          long-form WER (TED-LIUM 9.18 -> 7.5, Earnings-21 18.26 -> 11.85);
          `g = 0` is the paper's pure limited-context variant (Table 8 row 2).

          **References:** (paper: arxiv:2305.05084)
        description_es: |-
          Número `g` de tokens de atención globales iniciales en
          `fastconformer_attn` (paper: arxiv:2305.05084). Los tokens globales
          son las primeras `g` posiciones de la secuencia de entrada
          (espaciadas `global_tokens_spacing`): atienden a todas las
          posiciones y son atendidos por todas. El paper usa "un único token de
          atención global" (`g = 1`), que mejora el WER de forma larga
          (TED-LIUM 9.18 -> 7.5, Earnings-21 18.26 -> 11.85); `g = 0` es la
          variante de contexto limitado puro del paper (fila 2 de la Tabla 8).

          **Referencias:** (paper: arxiv:2305.05084)
        default: 1
        examples: [0, 1]
      global_tokens_spacing:
        type: integer
        minimum: 1
        title: "Fast Conformer Global Token Spacing"
        title_es: "Espaciado de Tokens Globales Fast Conformer"
        description: |-
          Stride `s` between the leading global attention tokens of
          `fastconformer_attn` (paper: arxiv:2305.05084): the global positions
          are `{0, s, 2s, ..., (g-1)*s}`. The paper uses a single global token
          (`g = 1`), where the spacing is irrelevant; `s > 1` spreads several
          global tokens across the head of the sequence (the NeMo recipe's
          `global_tokens_spacing`).

          **References:** (paper: arxiv:2305.05084)
        description_es: |-
          Paso `s` entre los tokens de atención globales iniciales de
          `fastconformer_attn` (paper: arxiv:2305.05084): las posiciones
          globales son `{0, s, 2s, ..., (g-1)*s}`. El paper usa un único token
          global (`g = 1`), donde el espaciado es irrelevante; `s > 1` reparte
          varios tokens globales por la cabecera de la secuencia (el
          `global_tokens_spacing` de la receta NeMo).

          **Referencias:** (paper: arxiv:2305.05084)
        default: 1
        examples: [1]
      global_attn_separate:
        type: boolean
        title: "Separate Global Projections"
        title_es: "Proyecciones Globales Separadas"
        description: |-
          Whether the global attention tokens of `fastconformer_attn` use a
          separate set of query/key/value linear projections (paper:
          arxiv:2305.05084 §2.2, the Longformer-style global attention design;
          adds 3*H^2 parameters) or share the local projections (the paper's
          §3.4 long-form model, fine-tuned from a full-attention checkpoint;
          the NeMo default). The default `false` keeps the mixer
          parameter-neutral (4*H^2).

          **References:** (paper: arxiv:2305.05084) (paper: arxiv:2004.05150)
        description_es: |-
          Si los tokens de atención globales de `fastconformer_attn` usan un
          conjunto separado de proyecciones lineales query/key/value (paper:
          arxiv:2305.05084 §2.2, el diseño de atención global estilo Longformer;
          añade 3*H^2 parámetros) o comparten las proyecciones locales (el
          modelo de forma larga §3.4 del paper, ajustado por fine-tuning desde
          un checkpoint de atención completa; el valor por defecto de NeMo). El
          valor por defecto `false` mantiene el mixer sin parámetros extra
          (4*H^2).

          **Referencias:** (paper: arxiv:2305.05084) (paper: arxiv:2004.05150)
        default: false
        examples: [false, true]
```

### 4.2 Hard rules to restate (for the implementer)

- **The schema is the contract** (`additionalProperties: false` everywhere):
  every key the new YAMLs use must exist in `src/schema/_model/**` before the
  presets validate; never read an undocumented key.
- Invariants: `hidden_size % num_heads == 0`; `num_kv_heads` divides
  `num_heads` (fastconformer uses plain MHA — do not read `num_kv_heads`);
  `len(layer_pattern) == num_layers` (both new configs satisfy it).
- **Inference never fails because of the cache (D5):** unknown/unsupported
  mixers fall back to full recompute; `spec_for` never raises.
- **Hard rule 5 does not apply** — `fastconformer_attn` is trainable and runs
  its windowed+global pattern in both `train()` and `eval()`; it must NOT be
  added to `TRAINING_FREE_LAYERS` and MUST be in the e2e `ATTENTIONS` list.
- **D11:** the papers are the theory source of truth. The annex formulations
  must match arXiv:2305.05084 (and, for the inherited global-attention
  pattern, Longformer arXiv:2004.05150 which the paper explicitly builds on);
  `docs/paper-es` must translate them faithfully — equations identical, prose
  Spanish.
- BitNet is on by default: all six projections must be `BitLinear` under
  `use_bitnet` (gate: `tests/test_bitnet_coverage.py`).

---

## 5. Ordered execution checklist (implementation time — do NOT run while planning)

Hard rules to restate at execution time are in §4.2. Preconditions:
`git status --short` clean; `uv sync --extra cpu --extra train`; TeX present
(`command -v pdflatex bibtex` — verified available on this machine); Chromium
present (`command -v chromium` — verified available).

### 5.1 Implementation (rows 1–12 of the table)

1. Module + exports (rows 1–2), registry (row 3), config fields + validation
   (row 4), flatten renames (row 5).
2. Schema: `_dims.yaml` enum + both tables (row 6), `_attention_fastconformer.yaml`
   (row 7, §4.1 draft), `_model.yaml` `$ref` (row 8), `use_pe` map (row 9),
   `capabilities.py` (row 10), `qk.py` (row 21).
3. Examples: `es_arch_fastconformer_adamw.yaml` (row 11), `fastconformer.yaml`
   (row 12).
4. e2e list: `run_e2e.py` `ATTENTIONS` (row 13).

### 5.2 Tests (rows 14–21)

Edit `test_schema_attention_layers.py`, `test_sparse_attention.py` (new class),
`test_bitnet_coverage.py`, `test_attention_refactor.py`,
`test_kv_cache_fallback.py`, `test_schema_consistency.py`
(`_DEFAULT_PAIRS`); `test_kv_cache_mixers.py` auto-extends.

### 5.3 Targeted gates

```bash
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_attention_refactor.py \
  tests/test_sparse_attention.py tests/test_bitnet_coverage.py \
  tests/test_kv_cache_mixers.py tests/test_kv_cache_fallback.py \
  -v --tb=short -p no:warnings
```

### 5.4 Full suite

```bash
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

### 5.5 e2e smoke, incl. targeted real-data training

```bash
# enumerate the new mixer's combos (build-only, no training)
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --list-combos | grep -F fastconformer

# e2e TARGETED SMOKE — ONLY the new mixer: actual training on REAL data.
# --realonall is the DEFAULT: every selected combo trains on the real Borges
# corpus (not the synthetic toy one). First invocation downloads the
# tokenizer/corpus from the HF Hub. '--only fastconformer' substring-matches
# the auto-generated combo ids: attn_single_fastconformer plus every pair
# attn_<a>__fastconformer / attn_fastconformer__<b>.
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only fastconformer

# minimal real-data run if the pair sweep is too heavy: the single solo combo
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only attn_single_fastconformer

# offline synthetic fallback (no HF download), same selection
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only fastconformer --no-realonall

# sizing knobs when needed: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800
# GPU smoke (real training box): swap the extra for --extra cu118 --extra train.
```

**Expected evidence:** every selected combo reports PASS with a finite
training loss over real corpus windows; the run directory keeps the generated
`config.yaml` + metrics. NaN/Inf loss, schema rejection or timeout is a
**blocker**, not a warning. Record the "N/N runs OK" number for the commit
message (LSA precedent: "real-data sweep: 264/264 runs OK" — expect the
count to grow by 47 combos: 1 solo + 46 pairs).

### 5.6 Figures

```bash
( cd docs/paper/figures && ./build-figures.sh both )
```

(auto-globs every `.mmd`, including the three edited/new ones; needs
Chrome/Chromium — verified present as `/usr/bin/chromium`; falls back to
`PUPPETEER_EXECUTABLE`). Verify `mixers/fastconformer.pdf`,
`mixers/fastconformer.svg`, `mixer-taxonomy.pdf`, `sparse-map.pdf` are newer
than their `.mmd`.

### 5.7 Param ground truth + JS estimator

```bash
uv run --extra cpu --extra train python full_tests/param_count_check.py
node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01
```

Expected: `… configs counted, 0 failures` (the regenerated `param_truth.json`
diff is intentional) and `ALL PASS` ≤1% — this only passes **after** the
mirror's `ft-param-estimator.js` gained the `fastconformer_attn` case (§6),
so run the estimator check again after the website sync; before the sync,
verify the Python ground truth only.

### 5.8 Sphinx

```bash
bash docs/build_docs.sh
```

### 5.9 PDF rebuild (only at implementation time)

```bash
cd docs/paper
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
rm -f *.aux *.log *.out *.toc *.blg

cd ../paper-es
pdflatex -interaction=nonstopmode paper-es.tex
bibtex paper-es
pdflatex -interaction=nonstopmode paper-es.tex
pdflatex -interaction=nonstopmode paper-es.tex
rm -f *.aux *.log *.out *.toc *.blg
```

**Verification after compilation:** no `Fatal error`, no
`Citation .* undefined` (specifically `rekesh_fastconformer_2023` resolves),
`paper.pdf` / `paper-es.pdf` newer than the edited `.tex`/`.bbl`,
`git status --short docs/paper docs/paper-es` shows only intended
`.tex`/`.bbl`/`.pdf`/figure files. Fallbacks: if `pdflatex`/`bibtex` were
unavailable, leave sources updated and note the rebuild must happen on a TeX
machine before the next release (the website sync copies prebuilt PDFs); if
no Chrome is available for mermaid, commit the `.mmd` sources and note the
`.pdf`/`.svg` renders must be produced before the paper/website sync.

---

## 6. Website sync (planned, not executed)

Run `/sync-website` (its own guardrails apply; state script first):

1. **Bootstrap/refresh the mirror** (currently NOT cloned — verified):
   `git clone https://github.com/erickfmm/erickfmm.github.io.git erickfmm.github.io`
   (path is gitignored in this repo), then `bash .opencode/command/sync-website-state.sh`.
2. **Copy per the file map** (`/sync-website` §1 table): `src/schema.yaml` +
   `src/schema/**` (incl. `_attention_fastconformer.yaml`, updated `_dims.yaml`,
   `_model.yaml`, `_positional_encoding.yaml`), `configs/*.yaml` +
   `configs/README.md` (adds `fastconformer.yaml`), `configs/examples/**` (adds
   `es_arch_fastconformer_adamw.yaml`), `docs/paper/**` + `docs/paper-es/**`
   (tex, bbl, pdf, figures incl. `mixers/fastconformer.*`,
   `mixer-taxonomy.*`, `sparse-map.*`), `docs/bibliography/**`,
   `docs/architecture.md`.
3. **Site-only edits** (mirror is the source of truth for these):
   - `ft-diagram.js`: add `CATS["fastconformer_attn"]` and the
     `SUBSTRUCT["fastconformer_attn"]` tensor-flow template exactly as drafted
     in row 42; **no `STYLES`/`STYLES_PRINT` class** (Sparse class reused).
   - `ft-param-estimator.js`: add the `case 'fastconformer_attn':` block
     drafted in §2.4 (reads `attention.fastconformer.global_attn_separate`).
   - `index.html`: no mixer lists exist (verified: zero peer-mixer-name
     occurrences in the live page); only fix what a probe flags.
4. **Paper tab HTML rebuild + top-level PDFs**:
   `flatten_tex.py --web` + `pandoc` + `postprocess_html.py` for en + es
   (exact commands in `/sync-website` §3), then `cp paper/en/paper.pdf
   paper/en.pdf` and `cp paper/es/paper-es.pdf paper/es.pdf`.
5. **Validation Gate** (`/sync-website` §5): `param_count_check.py` →
   `0 failures` (regenerates `param_truth.json` — intentional diff);
   `node full_tests/param_estimate_check.mjs … --tol 0.01` → `ALL PASS`
   (both new configs, incl. the 7H² example); the JSON round-trip script;
   the CLI parse script; the state-script probes (`schema-refs`,
   `example-yaml`, `cli-flags`, `cmd-binary-name`, …) all PASS.
6. The mirror stays **uncommitted** until the commit phase (§7).

---

## 7. Commit plan (planned, not executed)

Follow `/make-a-commit` conventions: English, conventional tag, structured
body; **never** stage `*.aux *.log *.out *.toc *.blg`, caches or mermaid temp
files; `param_truth.json` changes are intentional.

1. **Main repo** — `feat(attention): add Fast Conformer mixer (arXiv:2305.05084)`

   Body (mirror the LSA commit message structure):
   - mechanism summary: limited-context attention + head-pinned global token,
     optional separate global projections (§2.2) / shared (§3.4, default);
     window 128/side; Longformer-inspired; linearly scalable long-form
   - module: `src/model/attention/sparse/fastconformer_attn.py`
     (`FastConformerAttention`): window+global mask, separate global path,
     QK-norm capable, BitNet projections, encoder + decoder modes
   - HybridLayer registry entry (trainable; not training-free)
   - cache: `CacheSpec(supported=True, slot_type="ring", extras=global_tokens)`
     + `IMPLEMENTED_MIXERS`; exact ring branch (pinned head region covers the
     globals; swa precedent); parity auto-extended
   - schema: layer_pattern enum (48 values), EN/ES mixer tables,
     `model.attention.fastconformer.*` (4 knobs), `use_pe.fastconformer_attn`
     (default true), QK-capable list
   - config: 4 flat knobs + validation + `fastconformer_attn_use_pe`
   - presets: `configs/examples/es_arch_fastconformer_adamw.yaml` (small MLM,
     §2.2 separate-projection variant for param coverage) and
     `configs/fastconformer.yaml` (FC-L proportions, §3.4 recipe)
   - tests: schema/fallback counts 47→48, `FastConformerAttentionTests`
     (shapes, decoder causal clip, global row/column, separate-projection
     params, BitNet), bitnet coverage, refactor list, consistency pairs
   - e2e: `ATTENTIONS += fastconformer_attn` (real-data sweep: N/N runs OK)
   - figures: `mixers/fastconformer` + taxonomy (48 mixers, 12 sparse) +
     sparse-map, rendered PDF/SVG
   - docs: annex-4 EN/ES (new subsection + exec-summary/design-space),
     section 04 EN/ES, summary tables, schema annex + fastconformer block,
     KV-cache annex counts, abstract/intro/discussion/conclusion counts EN/ES,
     bibliography, README/architecture/configs-README census, Sphinx rst
   - rebuilt `paper.pdf` / `paper-es.pdf`
   - param ground truth + website estimator agree on
     `4H² + 3H²·[separate]`

   Paths to stage (main repo):
   `src/model/attention/sparse/fastconformer_attn.py`,
   `src/model/attention/sparse/__init__.py`, `src/model/attention/__init__.py`,
   `src/model/hybrid_layer.py`, `src/model/config.py`,
   `src/model/norm/qk.py`, `src/model/cache/capabilities.py`,
   `src/utils/config_flatten.py`, `src/schema/_model.yaml`,
   `src/schema/_model/_dims.yaml`,
   `src/schema/_model/_attention_fastconformer.yaml`,
   `src/schema/_model/_positional_encoding.yaml`,
   `configs/fastconformer.yaml`, `configs/examples/es_arch_fastconformer_adamw.yaml`,
   `configs/README.md`, `full_tests/run_e2e.py`,
   `full_tests/param_truth.json`, `tests/test_schema_attention_layers.py`,
   `tests/test_sparse_attention.py`, `tests/test_bitnet_coverage.py`,
   `tests/test_attention_refactor.py`, `tests/test_kv_cache_fallback.py`,
   `tests/test_schema_consistency.py`,
   (+ `tests/test_kv_cache_mixers.py` only if it needed a fixture),
   `docs/bibliography/attention_types.bib`,
   `docs/paper/figures/mixers/fastconformer.{mmd,pdf,svg}`,
   `docs/paper/figures/{mixer-taxonomy,sparse-map}.{mmd,pdf,svg}`,
   `docs/paper/appendices/{annex-4-sparse-attention,annex-summary-tables,annex-schema,annex-15-kv-cache}.tex`,
   `docs/paper/sections/{01-introduction,04-architecture-taxonomy,10-discussion,11-conclusion}.tex`,
   `docs/paper/{paper.tex,paper.bbl,paper.pdf}`,
   the ES counterparts (`docs/paper-es/appendices/annex-4-sparse-attention.tex`,
   `annex-summary-tables.tex`, `annex-schema.tex`, `annex-15-cache-kv.tex`,
   `docs/paper-es/sections/{01-introduccion,04-taxonomia-arquitectura,10-discusion,11-conclusion}.tex`,
   `docs/paper-es/{paper-es.tex,paper-es.bbl,paper-es.pdf}`),
   `README.md`, `docs/architecture.md`,
   `docs/source/api/model/attention/sparse/{fastconformer_attn.rst,index.rst}`.
   Then `git push origin HEAD:main`.

2. **Mirror** — `git -C erickfmm.github.io …` commit
   `docs(site): sync Fast Conformer mixer (arXiv:2305.05084)` staging the
   synced tree (schema, configs, examples, paper sources/PDFs/figures,
   bibliography, architecture.md spec) **plus** the two site-only JS files
   (`ft-diagram.js`, `ft-param-estimator.js`), the rebuilt
   `paper/paper-en.html`, `paper/paper-es.html`, `paper/en.pdf`,
   `paper/es.pdf`; then
   `git -C erickfmm.github.io push origin HEAD:main`.

---

## 8. Verification matrix

| Gate | Command | Expected result |
|---|---|---|
| Schema attention layers | `uv run --extra cpu --extra train python -m pytest tests/test_schema_attention_layers.py -v` | all pass; enum len 48; `attention.fastconformer` leaves present; `set(CACHE_SPECS) == enum` |
| Schema consistency | `… pytest tests/test_schema_consistency.py -v` | enum ↔ both bilingual tables; declared defaults ↔ dataclass (4 new pairs); all presets validate |
| YAML examples | `… pytest tests/test_yaml_examples.py -v` | both new YAMLs load via `load_training_config` |
| Refactor coverage | `… pytest tests/test_attention_refactor.py -v` | trainable list builds a 6-token encoder forward for `fastconformer_attn` |
| Family tests | `… pytest tests/test_sparse_attention.py -v` | new `FastConformerAttentionTests` all green |
| BitNet coverage | `… pytest tests/test_bitnet_coverage.py -v` | all projections BitLinear under BitNet (incl. global set) |
| KV parity | `… pytest tests/test_kv_cache_mixers.py -v` | `fastconformer_attn` × {rope, nope, alibi}: prefill+decode == full forward |
| KV fallback census | `… pytest tests/test_kv_cache_fallback.py -v` | enum count 48 |
| Full suite | `… pytest tests/ --continue-on-collection-errors -v` | no new failures |
| e2e enumeration | `full_tests/run_e2e.py --category attn --list-combos \| grep -F fastconformer` | `attn_single_fastconformer` + all pairs listed |
| e2e real-data training | `full_tests/run_e2e.py --category attn --only fastconformer` | every selected combo PASS, finite loss on the real Borges corpus |
| e2e solo variant | `full_tests/run_e2e.py --category attn --only attn_single_fastconformer` | solo combo PASS |
| e2e offline fallback | `full_tests/run_e2e.py --category attn --only fastconformer --no-realonall` | PASS without HF download (use only if Hub unreachable) |
| Figures | `( cd docs/paper/figures && ./build-figures.sh both )` | `fastconformer.{pdf,svg}` created; taxonomy/sparse-map re-rendered |
| Param ground truth | `full_tests/param_count_check.py` | `… configs counted, 0 failures`; truth includes both new configs (4H² and 7H²) |
| JS estimator | `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` | `ALL PASS` ≤1% (after the mirror edit) |
| Sphinx | `bash docs/build_docs.sh` | build succeeds; `fastconformer_attn` page generated |
| PDFs | pdflatex/bibtex ×3 per paper (§5.9) | no `Fatal error`, no undefined citations, fresh `paper.pdf`/`paper-es.pdf` |
| Mirror probes | `bash .opencode/command/sync-website-state.sh` | `schema-refs`, `example-yaml`, `cli-flags`, … all PASS |

---

## 9. Risks, assumptions and open questions

**Assumptions to re-verify at implementation time**

- **Line drift:** every anchor (path:line) was read at planning time on a
  clean tree at `b583287`; re-locate by quoted content, not line number.
- **NeMo semantics:** the global-token placement/projection dataflow was
  recovered from `nemo/collections/asr/parts/submodules/multi_head_attention.py`
  (paper: "open sourced through NVIDIA NeMo"). Re-check that file if the
  implementation disagrees with the mask behavior expected here.
- **Ring capacity off-by-one:** `capacity = num_sinks + w//2 + 1` is derived,
  not tested; the parity test is the arbiter (widen by 1 if prefill/decode
  parity fails at the window boundary). Decode-step queries are never global
  rows (G is at the head) — verify with a prefill shorter than `(g−1)·s+1`.
- **Separate-projection cached path:** appending pre-swapped global k̃/ṽ keeps
  the cached branch exact **only because** global positions are always inside
  the pinned head region; assert this in the module docstring and rely on
  parity across `rope/nope/alibi`.
- **HF Hub reachability** for the real-data smoke; offline fallback in §5.5.
- `pdflatex`/`bibtex`/`pandoc`/`node`/`chromium` verified present on this
  machine; re-check on the execution box.

**Naming collisions:** `fastconformer` grep across repo + mirror files is
empty; `lsa_attn` (Native-LSA) collision avoided by the user-confirmed name.

**Paper ambiguity:** the ICASSP paper gives no equations for the attention
mechanism — the reconstruction (§2.2) is anchored on its prose + Longformer
(which it cites) + its own NeMo recipe. If the user prefers a stricter
reading (e.g. treat the pooled-token NeMo variant as canonical), that is a
different mixer and this plan does not cover it. The paper's *other* content
(8× downsampling schema, RNNT/CTC decoders, SSL pretraining) is out of scope.

**Mode/task restrictions:** none (rule 6 untouched); works in encoder and
decoder; composes with ViT/audio stacks as a generic mixer (no preset added).

**BitNet interactions:** the optional global projections must be BitLinear
under BitNet (row 16 gate); `bitnet.routers` untouched.

**Pre-existing drift noticed (flagged, decide before/at implementation):**

1. `docs/paper/sections/02-related-work.tex:44` says "43 sequence mixers" —
   stale census **not** touched by the LSA commit; fixing it exceeds the
   blueprint scope → leave unless the user approves.
2. `docs/architecture.md` lines 25/38/82 (say 46) and §7.13 line 639 (says
   45) are stale vs the current 47 — these **are** mixer/cache census lines
   covered by the row-38 census pass and are bumped to the post-change values
   there (48 / 47 classes); noted here because they were already wrong.
3. `docs/paper/appendices/annex-4` exec summary says "ten" sparse families,
   missing LSA — fixed in row 26 (it must become twelve).
4. `swa_attn` has no `docs/source/api/model/attention/sparse/swa_attn.rst`
   and no `annex-schema.tex` mixer row (pre-existing gaps) — **not** fixed by
   this plan; flag to the user separately.
5. `tests/test_kv_cache_fallback.py` docstring says "46 values" (stale) —
   row 19 fixes the assertion; the docstring word is a one-word fix in an
   already-touched file (implementer's discretion).

**Open questions:** none blocking. The full-preset interpretation
(`configs/fastconformer.yaml` as a text-MLM FC-L encoder rather than the
paper's RNNT audio model) is interpretive by necessity and documented in the
preset header.

---

## 10. Out of scope

- **No code, schema, preset, test, figure, doc, PDF, mirror or git change was
  made by this planning command.** The only file written is this plan
  (`.opencode/plan/add-mixer-2305.05084.md`).
- The paper's downsampling schema (§2.1: 8× depthwise-separable subsampling,
  kernel 9, 256 channels), the RNNT/CTC decoder recipes, the Speech
  Translation/SLU heads, the SSL-pretrained XL/XXL scaling recipe, and the
  Longformer overlapping-chunk kernel are **not** implemented by this plan.
- No changes to `TRAINING_FREE_LAYERS`, no renaming/deletion of any existing
  mixer, no family re-organization, no unrelated drift fixes beyond the
  census occurrences enumerated in rows 26–38 and flagged in §9.
- No commits, pushes, or mirror writes (all §6/§7 steps are planned only).
