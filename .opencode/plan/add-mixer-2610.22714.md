# Plan: add First Passage Attention (FPA) mixer (arXiv:2610.22714)

**Paper**: Yuxuan Liu, *First Passage Attention*, arXiv:2610.22714, October 2026
(Dept. of Mathematics, University of Bergen).

**One-paragraph summary.** FoX-style forget gates sum decay contributions
along the key→query path, so suppression is monotonic in age: a recent token
cannot be erased without erasing everything older. First Passage Attention
(FPA) replaces the path **sum** with a path **extremum**: every head computes
a scalar *field* φ at each position and a scalar *level* ψ per key, and a
key's logit is lowered by β · (depth of the deepest dip of φ below ψ over the
span after the key). The bias is `b_ts = β·min(0, min_{s<u≤t} φ_u − ψ_s)`,
with φ = W + g, ψ = W + h and W a learned walk (prefix sum of per-token
increments δ). FoX, a bottleneck bias, and Expire-Span drop out as special
cases; in the stack form (g = h = 0) the alive keys evolve as a stack, so a
token can erase newer keys while preserving older ones. The paper proves one
constant-dimension FPA head realises every dip-induced alive set (softmax
heads with positional biases/token-level gates need head dim linear in key
count), trains a 110M LM matching FoX in perplexity with perfect passkey
retrieval at 8× train length, and defines an **exact O(t) decode recurrence**
storing two scalars per cached key/head. It belongs in this repo as a
gated-family sibling of `fox` (its own special case b), natively trainable in
encoder and decoder modes with a plain-KV incremental branch.

---

## 2. Paper extract

### 2.1 Forward equations (verbatim, §2 + Table 14)

Causal head of dimension *d*, queries `q_t`, keys `k_s`, values `v_s`:

```
O = Σ_{s≤t} w_ts v_s,   w_ts ∝ exp(ℓ_ts)
ℓ_ts = q_tᵀ k_s / √d + b_ts                       (content + bias)
b_ts = β · min(0,  min_{s<u≤t} φ_u − ψ_s),  b_tt = 0     (first-passage bias)
φ̄_ts := min_{s<u≤t} φ_u                            (range minimum)
```

Field/level parameterisation (eq. 2; one linear map gives all three scalars
per head from the same normalised input as Q/K/V):

```
W_t = Σ_{u≤t} δ_u        (walk: prefix sum of increments)
φ_u = W_u + g_u          (field)
ψ_s = W_s + h_s          (level of key s)
β > 0                    (learned strength, one scalar per head)
```

Learned tensors per layer of width D with H heads: one `Linear(D, 3H)` with
bias for (δ, g, h) plus H strengths → **3H(D+1) + H extra parameters**
(27,696 per layer at D=768, H=12; 332,352 over 12 layers — paper §2/Table 14).

Special cases (Prop. 2.2, useful for the annex):
- **(a)** g = h = 0 → `b_ts = −β·L_ts`, `L_ts = max(0, max_{s<u≤t}(W_s − W_u))`
  (Lindley/Skorokhod queue regulator; "stack form").
- **(b)** additionally δ_u ≤ 0 → `b_ts = Σ_{s<u≤t} log f_u`, f_u = exp(βδ_u) ∈ (0,1]
  — **exactly FoX** (this repo's `fox`).
- **(c)** δ = 0, h = 0, g ≤ 0, β = 1 → bottleneck bias `min_{s<u≤t} g_u`.
- **(d)** δ_u = −1, g = 0, ψ_s = W_s − e_s → Expire-Span `β·min(0, e_s − (t−s))`.

Properties (Prop. 2.3): death is permanent (`b_{t′s} ≤ b_ts` for t′ > t); in
the stack form the alive keys evolve as a **stack** (a dip kills a time-suffix
of alive keys, old keys survive).

### 2.2 Training / gradient (straight-through, §2 "Training")

The bias is piecewise linear in φ, ψ; the inner range minimum can close
early in training and stay closed, so the paper trains with an STE:

- **Forward**: hard range minimum φ̄ (eq. above). Evaluation always uses the
  hard bias unchanged.
- **Backward**: gradient of the **soft minimum**
  `−τ · log Σ_{s<u≤t} exp(−φ_u/τ)` with **τ = 1** in place of the inner min.
- **The outer `min(0, ·)` acts on the hard value** (ψ and β receive gradients
  through the active piecewise-linear path).

### 2.3 Parallel (training) and recurrent (decode) algorithms

**Parallel/blocked form (Prop. 2.4a; paper's Triton kernel, Appendix E):**
with block size B, the bias between query block I and earlier key block J
needs only three quantities: suffix minima of φ within J, the minimum of φ
over blocks strictly between, and prefix minima of φ within I. O(B²) per
block pair (same as an element-wise bias like FoX's) + O(T/B) block minima.
The paper fuses this with online-softmax attention in deterministic Triton
kernels (blocks of 64; no atomics).

**Reference (this repo, CPU-first v1):** the exact O(T²) bias matrix,
vectorised — per head:
```
M[t,u] = φ_u if u ≤ t else +inf                     (lower-triangular mask)
S = flip(cummin(flip(M, dim=u), dim=u))             S[t,u] = min_{u≤v≤t} φ_v
G[t,s] = S[t, s+1]  (shift one column; pad +inf)     G[t,s] = min_{s<u≤t} φ_u
b = β · clamp(G − ψ, max=0)                          # b_tt = 0 falls out (G[t,t] = +inf)
```
STE variant: compute `G_soft = −τ · logcumsumexp(−φ/τ)` with the same
mask/shift, then `G_st = G_hard.detach() + (G_soft − G_soft.detach())`
(forward = hard, backward = soft). One flip + `cummin`/`logcumsumexp` + flip;
no T×T× anything intermediates beyond the bias matrix itself.

**Recurrent/decode form (Prop. 2.4b — exact):**
```
φ̄_{t,s} = min(φ̄_{t−1,s}, φ_t)   for s < t−1;   φ̄_{t,t−1} = φ_t
b_ts     = β · min(0, φ̄_{t,s} − ψ_s)
```
Each step updates the bias row in **O(t)** and stores **two scalars per
cached key and head** (the per-token field φ_u and level ψ_s, from which φ̄ is
maintained / recomputed as a suffix min). Since `φ̄_{t,s}` never increases,
dead keys stay dead.

### 2.4 Parameter inventory (for the exact count formula)

| Tensor | Module | Shape | Bias |
|---|---|---|---|
| `q_proj` | `BitLinear/Linear` | `H × H` (nH·hd out) | no |
| `k_proj` | `BitLinear/Linear` | `H × H` | no |
| `v_proj` | `BitLinear/Linear` | `H × H` | no |
| `out_proj` | `BitLinear/Linear` | `H × H` | no |
| `field_proj` | `BitLinear/Linear` | `H × 3·nH` | **yes** (paper Table 15: "except the field projection of FPA") |
| `beta` | `nn.Parameter` | `(nH,)` init `beta_init` (paper: 1) | — |

**Exact count** in repo terms (H = hidden_size, nH = num_heads, hd = H/nH,
nKv and fH unused):

```
mixer_params(fpa) = 4·H²  +  3·nH·(H + 1)  +  nH
                    QKV+O   field δ,g,h+bias  per-head β
```

Cross-check vs paper: extra over softmax = 3H(D+1) + H ✓ (D=768, nH=12 →
27,696/layer; ×12 layers = 332,352 ✓ "332 thousand"). Cache overhead: two
scalars per key per head = 2/(2·hd) ≈ **+1.56%** over K/V at hd=64 ✓ (§4.3).

### 2.5 Knobs (→ `model.attention.fpa.*`)

| Schema key | Flat field | Type | Default | Constraints / coercion | Notes |
|---|---|---|---|---|---|
| `attention.fpa.beta_init` | `fpa_beta_init` | number | `1.0` | `> 0` (`exclusiveMinimum: 0`), float | Initial value of the learned per-head strength β. β is always learned (raw `nn.Parameter`); paper Table 16 init β = 1 and excludes strengths from weight decay (training detail, not enforced by the repo). |
| `attention.fpa.tau` | `fpa_tau` | number | `1.0` | `> 0` (`exclusiveMinimum: 0`), float | Temperature τ of the soft-minimum **straight-through gradient only** (paper §2 Training, τ = 1). The forward pass and eval always use the hard range minimum. |

No other knobs: the paper fixes the field width at 3 scalars/head, keeps MHA
(no GQA in the reference), and the blocked kernel is an implementation detail.
(Chosen over "no knobs" per user decision D-C4.)

### 2.6 State & cache

- **Compatibility class (i)**: classic per-token KV attention, extended with
  **per-token extras** — exactly the FoX precedent
  (`fox` caches `(K, V, c_j)`; FPA caches `(K, V, φ_u, ψ_s)`).
- Slot: `KVSlot` (`slot_type="kv"`), `extras=("field_phi", "key_level_psi")`.
- Decode cost: O(t) per step (suffix min over cached φ + one bias row);
  paper §4.3: 22.0 ms/token at 8k ctx vs 20.2 FoX / 23.9 SDPA.
- Cache size: +2 scalars per key per head (~1.56% over K/V at hd=64).
- **Eviction** (paper §4.3, *not* in scope for v1): a key whose bias falls
  below −θ never returns (Prop. 2.3a), so it can be dropped — θ=10 keeps
  26–34% of keys at +0.0005 nats. Natural follow-up as a CachePolicy over the
  KVSlot (D14 `sparsek` precedent) — flagged in §9, needs explicit approval.

### 2.7 Complexity (for `tab:all-mixers`)

- Training: **O(n²·d)** attention + O(n² ) per-head scalar bias (element-wise,
  same class as FoX's D matrix); paper kernel ≈ 0.97× SDPA time.
- Inference: **O(n) KV cache** + O(t)/step decode, two extra scalars per
  key/head.

### 2.8 Trainability

Natively trainable from scratch (softmax + additive bias matrix; STE for the
inner min). **Not** training-free — no `TRAINING_FREE_LAYERS` membership, no
`dense_forward` surrogate, appears in `ATTENTIONS`. (Paper note: an FPA model
without a bias in the **first** layer fixed multi-query associative recall —
recorded as guidance in the annex, not a repo restriction.)

### 2.9 Positional encoding

The paper's FPA models use **no positional encoding** (Table 14: "none";
"FPA uses no other positional information") — the walk *is* positional.
⇒ `per_mixer_use_pe.fpa` default **false**; `fpa_use_pe: bool = False`
dataclass default (same as `fox`). The mixer still honours a forced-on PE via
`apply_pe_to_qk` / `apply_pe_to_scores` like `fox` does.

### 2.10 Mode constraints

Both encoder and decoder. The bias is defined for the **causal** head
(range `(s, t]`). Decoder mode: paper-exact causal bias + causal mask.
Encoder mode (repo generalisation, mirroring how `fox` reuses its bias matrix
bidirectionally): symmetric extension — `φ̄_{ts} = min over the open span
between s and t` (element-wise min of the causal G and its transpose); must
be stated as a repo-side choice in the annex (the paper never defines a
bidirectional form). No grid/vision requirement (ViT-compatible like any
softmax mixer); audio-compatible.

### 2.11 BibTeX (for `docs/bibliography/attention_types.bib`, house format)

```bibtex
@misc{liu_first_passage_2026,
	title = {First Passage Attention},
	author = {Liu, Yuxuan},
	year = {2026},
	eprint = {2610.22714},
	archivePrefix = {arXiv},
	primaryClass = {cs.LG},
	url = {https://arxiv.org/abs/2610.22714},
	publisher = {{arXiv}}
}
```

Reference LM (for the preset + annex "Empirical Performance"): 110M decoder,
12 layers × width 768, 12 heads × 64, SwiGLU-2048, RMSNorm (pre-norm), **no
PE**, tied 32k (Mistral) embeddings, context 2048; SlimPajama 2.7B tokens,
AdamW (0.9, 0.95), lr 2e-3, 512-step warmup, cosine→0, wd 0.1 (weights +
embeddings only), clip 1.0, β init 1, τ = 1; field in fp32 under autocast.
Results: SlimPajama 2.8640 vs FoX 2.8832 nats; 100% passkey at 8×/16k train
length (FoX 20%); flat PG-19 loss to 16,384 ctx; ~26–34% cache at θ=10.

---

## 3. Decisions

| # | Decision | Evidence |
|---|---|---|
| D1 | Registry name **`fpa`**; class **`FirstPassageAttention`** (`src/model/attention/gated/fpa.py`). No `_attn` suffix (removed in 1.2, commit 9b10768). | Paper short name "FPA"; snake_case peers `fox`, `kda`. |
| D2 | Family directory **`gated/`**; census label **Gated**. | User decision (D-C1). FoX — FPA's own special case (Prop. 2.2b) — is `gated/fox.py`, annex-5 subsection 5, taxonomy node g6. No new family infrastructure (taxonomy branch / family map / `STYLES` class). |
| D3 | Closest peer **`gated/fox.py`** (`ForgettingAttention`). Mirror: constructor `(config, pos_encoder=None)`; `proj_cls` BitNet dispatch; `f_proj`-analogue `field_proj` **with bias**; `use_pe` via `getattr(config, "fpa_use_pe", False)`; `attach_qk_norm` + `record_qk_max_logits`; `module_softmax` (honours `self.use_fixed_softmax`, hybrid_layer.py:249); decoder causal mask; cached branch via `kv_decode_append` with extras (fox.py:166–204); non-cached branch (fox.py:206–222) with the range-min bias in place of the cumulative log-forget bias. |
| D4 | Knobs **`fpa.beta_init` = 1.0, `fpa.tau` = 1.0** (user decision D-C4) → knob file + `$ref` + renames + dataclass fields + validation + `_DEFAULT_PAIRS`. |
| D5 | `per_mixer_use_pe.fpa` default **false**; `fpa_use_pe: bool = False`. | Paper Table 14 "Positions: none"; §2.10 above; `fox` precedent (false). |
| D6 | **KV-cache decision 1 (full)**: `CacheSpec(supported=True, slot_type="kv", extras=("field_phi", "key_level_psi"))` **and** `IMPLEMENTED_MIXERS` membership **and** the incremental branch in the module **and** automatic parity coverage in `tests/test_kv_cache_mixers.py` (its `MIXERS` is derived from `IMPLEMENTED_MIXERS`, no test edit needed). | Paper Prop. 2.4(b) defines the exact decode recurrence and the two-per-key/head state; the extras mechanism already carries per-token scalars (fox `cum_log_f`). Class (i) + extras. |
| D7 | Not training-free: no `TRAINING_FREE_LAYERS`, no `dense_forward`, **joins `ATTENTIONS`** (currently 42 entries / 41 unique). | §2.8; list header comment reserves it for trainable mixers (run_e2e.py:223). |
| D8 | Example preset **`configs/examples/es_arch_fpa_adamw.yaml`** (user decision D-C2; recent precedent `es_arch_ca/kda/lsa_*`), encoder-MLM baseline, `load_training_config`-clean. |
| D9 | Full preset **`configs/fpa.yaml`** (user decision D-C3) — paper's 110M reference decoder, `configs/ca.yaml` shape. Joins the smoke-tested preset set (test_yaml_examples globs `configs/*.yaml`). |
| D10 | **Census bumps (verified current numbers)**: schema enum **48 → 49** (`_dims.yaml:338–386`); capabilities docstring "48 total: 47 mixer classes + mamba" → "49 total: 48 mixer classes…" (capabilities.py:5–7) and `architecture.md:664` "(47)" → "(48)"; §7.4 header "48 enum values in 10 families" → **49** (architecture.md:351); README.md:28 "48 mixer architectures" → 49, :75 "**47** across 10 families" → **48**, :95 "all 48" → all 49, :113–121 Mixer Categories **Gated (8) → (9)** + append `fpa`, :130 "48 mixer values" → 49; `mixer-taxonomy.mmd` root "Sequence Mixer Registry (48 variants)" → **(49)** + `Gated (8)` → **Gated (9)** + chain `g8["kda"] --> g9["fpa"]`; `annex-summary-tables.tex:4` "48 sequence mixers" → 49 + `:62` "**Gated (8)**" → (9) + row after `:71` (`kda`); ES twin (`annex-summary-tables.tex:4` "48 mezcladores", Gated ES header, `fox &` row `:69`); section 04 EN `:12` "(8)" → (9) + append FPA, `:24` caption "forty-eight" → **forty-nine**, `:81` "eight blocks" → **nine** + `\citep{liu_first_passage_2026}`; ES 04 `:12` "(8)", `:76` "ocho bloques"; abstract census in `paper.tex` / `paper-es.tex` ("forty-eight sequence mixers" / "cuarenta y ocho mezcladores" → forty-nine / cuarenta y nueve) + the same sweep the CA commit made in `sections/01/10/11` (EN + ES). Final pass must `grep -rn "48 mixer\|forty-eight\|cuarenta y ocho\|(48 variants)\|Gated (8)\|(8)"` across README/docs and fix mixer-census hits only. |
| D11 | Sphinx page `docs/source/api/model/attention/gated/fpa.rst` + toctree entry after `kda`. | gated/index.rst current list. |

**Hard rules restate (must hold):** the schema is the contract
(`additionalProperties: false` everywhere — every new key lives in
`_attention_fpa.yaml`); `hidden_size % num_heads == 0` (validated in the
module like fox.py:102); `num_kv_heads` unused (MHA per paper);
`len(layer_pattern) == num_layers`; inference never fails because of the
cache (D5 — unsupported/quantized paths fall back); not training-free so rule
5 does not apply; papers are the theory source of truth (D11) — annex
formulations must match arXiv:2610.22714 and `paper-es` must translate them
faithfully (encoder-mode symmetric extension labelled as repo-side).

---

## 4. File-by-file plan

Execution order top-down. Line anchors are current (Nov 2026, HEAD 6dc1170).

| # | File | Planned change |
|---|---|---|
| 1 | `src/model/attention/gated/fpa.py` | **New module** `FirstPassageAttention` (paper cited in module + class docstrings; `from __future__ import annotations`; Google-style docstrings; CPU-safe pure PyTorch — no Triton). Constructor `(config, pos_encoder=None)` mirroring fox.py:84–117: `hidden/num_heads/head_dim/total_dim/scale`; `ValueError` if `H % nH`; `proj_cls = BitLinear if config.use_bitnet else nn.Linear`; `q/k/v_proj(hidden, H, bias=False)`, `out_proj(H, hidden, bias=False)`; `field_proj = proj_cls(hidden, 3·num_heads, bias=True)` (paper Table 15); `self.beta = nn.Parameter(torch.full((num_heads,), float(fpa_beta_init)))`; `self.tau = float(getattr(config, "fpa_tau", 1.0))`; `mode`; `pos_encoder`/`pe_type`; `self.use_pe = bool(getattr(config, "fpa_use_pe", False))`; `attach_qk_norm(self, config, head_dim)`. `forward(x, logical_layer_idx=None, pos_encoder=None, cache=None, cache_position=None)` (call shape S1, §7.13): project q/k/v (+qk_norm), field `dgh = field_proj(x)` → δ,g,h `(B,S,nH)`; `W = cumsum(δ, dim=1)`; `phi = W + g`, `psi = W + h` (**fp32**: cast `x.float()` for the field path when autocast/half, per paper Table 16). Hard suffix-min G via mask+flip+`cummin`+flip+shift (§2.3); STE `G_st = G_hard.detach() + (G_soft − G_soft.detach())` with `G_soft` from masked `logcumsumexp` at τ; `bias = beta·clamp(G_st − psi, max=0)` (per head, broadcast `(B,nH,S,S)`); decoder: causal `triu` mask → `−inf`; encoder: symmetric extension `G_sym = minimum(G, Gᵀ)` (documented repo generalisation); `attn = q@kᵀ·scale + bias`; `apply_pe_to_scores` when PE forced on; `record_qk_max_logits`; `module_softmax`; dropout; `out_proj`. **Cached branch** (mirror fox.py:166–204) when `getattr(cache, "kind", "") in ("kv", "ring")`: `positions_for`, `apply_pe_to_qk(..., position_offset=offset)`, `kv_decode_append(slot, k, v, positions, extras={"field_phi": phi_new, "key_level_psi": psi_new})`; read cached `phi_c/psi_c (B,H,T)`; suffix-min over cached φ (reverse cummin), shift one, `bias_row = beta·clamp(min_phi − psi_c, max=0)`; `causal_mask_from_positions` → `−inf`; same softmax path. Exact by Prop. 2.4b. |
| 2 | `src/model/attention/gated/__init__.py` + `src/model/attention/__init__.py` | `from .fpa import FirstPassageAttention` (after `from .fox import ForgettingAttention`, gated/__init__.py:10); `"FirstPassageAttention"` in `__all__` (after `"ForgettingAttention"`, :24); same import + `__all__` entry inside the `from .gated import (...)` block of `attention/__init__.py:34–48`. |
| 3 | `src/model/hybrid_layer.py` | Import at :33–48 block; `mixer_registry` entry `"fpa": FirstPassageAttention,` after `"kda": KDAAttention,` (:208). **No** `TRAINING_FREE_LAYERS` change (rule 5 n/a). Generic `slot is not None` call path (:462–469) already forwards `cache`/`cache_position` — no dispatcher edit. |
| 4 | `src/model/config.py` | After the `ca_*` block (:814–822): `# ---- First Passage Attention (arXiv:2610.22714) ----` `fpa_beta_init: float = 1.0` (init of the learned per-head strength β, paper Table 16), `fpa_tau: float = 1.0` (STE soft-min temperature, paper §2); plus `fpa_use_pe: bool = False` next to `fox_use_pe` (:637). Validation next to `ca_gain_init` (:1163): raise `ValueError` if `fpa_beta_init <= 0` or `fpa_tau <= 0`. |
| 5 | `src/utils/config_flatten.py` | `_ATTENTION_MIXER_RENAMES["fpa"] = {"beta_init": "fpa_beta_init", "tau": "fpa_tau"}` after the `"ca"` entry (:170–173). |
| 6 | `src/schema/_model/_dims.yaml` | Enum: append `- fpa` after `- fastconformer` (:386). EN table after `:265` (`fastconformer` row): `` | `fpa` | First Passage Attention — softmax + first-passage (range-min) forget bias β·min(0, φ̄ₜₛ − ψₛ); contains FoX as δ≤0 special case (paper: arxiv:2610.22714) | ✅ | ✅ | ``. ES table after `:325`: `` | `fpa` | First Passage Attention — softmax + sesgo de olvido de primer paso (mínimo de rango) β·min(0, φ̄ₜₛ − ψₛ); contiene a FoX como caso especial δ≤0 (paper: arxiv:2610.22714) | ✅ | ✅ | ``. No ⚠️ (trainable). |
| 7 | `src/schema/_model/_attention_fpa.yaml` | **New knob file** (template `_attention_ca.yaml`): `type: object; additionalProperties: false; properties: fpa:` with `title "First Passage Attention"` / `title_es "Atención de Primer Paso"`, bilingual descriptions (mechanism, "Only applies when `fpa` is in `layer_pattern`", `**References:** (paper: arxiv:2610.22714)`), and: `beta_init` (number, `exclusiveMinimum: 0`, default 1.0, examples [1.0], bilingual leaf docs: initial value of the learned per-head strength β; paper Table 16 init β = 1) and `tau` (number, `exclusiveMinimum: 0`, default 1.0, examples [1.0], bilingual leaf docs: temperature of the soft-minimum straight-through gradient; forward/eval always use the hard minimum; paper §2 uses τ = 1). |
| 8 | `src/schema/_model.yaml` | Under `attention.properties` after the `ca:` `$ref` (:112–113): `fpa: {$ref: "_model/_attention_fpa.yaml#/properties/fpa"}`. **Leave both attention description lists unchanged** (they enumerate knob-less mixers; fpa has knobs) — verify only. |
| 9 | `src/schema/_model/_positional_encoding.yaml` | In `positional_encoding_parameters.use_pe.properties` after `fox` (:434): `fpa: { type: boolean, title: "fpa", description: "Model-wide PE for `fpa` layers (FPA trains without positional encoding; the walk is positional) (paper: arxiv:2610.22714).", default: false, description_es: "PE a nivel de modelo para capas `fpa` (FPA entrena sin codificación posicional; la caminata es posicional) (paper: arxiv:2610.22714)." }`. Add `fpa` to the false-default prose lists (:24 EN, :385 EN; ES :391 if enumerated). |
| 10 | `src/model/cache/capabilities.py` | (a) `CACHE_SPECS["fpa"] = CacheSpec(supported=True, slot_type="kv", extras=("field_phi", "key_level_psi"), notes="First Passage Attention (arXiv:2610.22714); class (i) with per-token extras: the logit bias is β·min(0, min_{s<u≤t} φ_u − ψ_s), a range minimum over the cached field, so decode stores two scalars per key/head (φ_u, ψ_s) and updates φ̄_{t,s} = min(φ̄_{t−1,s}, φ_t) in O(t) (Prop. 2.4b) — exact. Dead keys (bias < −θ) are evictable in principle (paper §4.3: θ=10 keeps ~26–34% of keys); eviction policy is a D14 follow-up.")` — placed after `"fox"` (:151–158). (b) `IMPLEMENTED_MIXERS` += `"fpa"` in the F1 dense/softmax group (after `"fox"`, :75). (c) Header census "48 total: 47 mixer classes + the `mamba` linear placeholder" → "49 total: 48 mixer classes…". |
| 11 | `configs/examples/es_arch_fpa_adamw.yaml` | **New example** (shape = `es_arch_ca_adamw.yaml`): `model_class: frankenstein_text_encoder`; dims vocab 50000 / hidden 512 / layers 10 / loops 1 / heads 8 / kv 1 / `layer_pattern: [fpa]`; norm layer_norm; embedding.conv disabled; `attention: {fpa: {beta_init: 1.0, tau: 1.0}, retnet: {heads: 8}}`; moe/bitnet off; ffn 2048 gelu; training = the CA example's mlm/adamw block (seed 42, epochs 4, batch 4, max_length 512, per-group adamw params, scheduler_total_steps 9000, `csv_log_path: training_metrics_es_arch_fpa_adamw.yaml.csv`, cache_dir `./temp_data/v2_dataset_cache`). Drop the machine-specific `local_parquet_dir` line (present in CA's by accident of reuse; `prefer_local_cache/stream_local_parquet` need it — omit those three lines and keep the plain HF path so `load_training_config` stays machine-independent). |
| 12 | `configs/fpa.yaml` | **New preset** (shape = `configs/ca.yaml`): header comment "First Passage Attention (arXiv:2610.22714) — paper reference configuration (110M): 12 layers, width 768, 12 heads, SwiGLU-2048, RMSNorm, **no positional encoding**, tied 32k vocab, context 2048; β init 1, STE τ = 1." `model_class: frankenstein_text_decoder`; dims vocab 32000 / hidden 768 / layers 12 / loops 1 / heads 12 / kv 12 / dropout 0.0 / mode decoder / `layer_pattern: [fpa ×12]`; norm rms_norm; `attention: {fpa: {beta_init: 1.0, tau: 1.0}, retnet: {heads: 12}}`; moe/bitnet off; `ffn: {hidden_size: 2048, activation: swiglu}`; `positional_encoding: none`; training causal_lm, seed 42, batch 8, workers 2, max_length 2048, grad-accum 8, adamw `lr_other 3.0e-4` / `lr_attention 3.0e-4`, betas (0.9, 0.95), eps 1e-8, `wd_other 0.1`, `wd_attention 0.01`, scheduler_total_steps 5150 (paper step count), `scheduler_warmup_ratio` ≈ 0.0994 (512/5150) — or `scheduler_warmup_steps` if the schema supports it, else round to 0.1 with a comment, cosine, `grad_clip_max_norm: 1.0`, `csv_log_path: training_metrics_fpa.csv`. |
| 13 | `full_tests/run_e2e.py` | Append to `ATTENTIONS` (after `"ca",` :254) with the house comment: `# First Passage Attention (arXiv:2610.22714): softmax + path-extremum (range-min) forget bias; trains natively in encoder and decoder modes; exact KV-cache incremental decoding (two per-key/head scalars).` + `"fpa",`. Auto-generates `attn_single_fpa` (make_single_attention_configs :788–795) and every pair `attn_<a>__fpa` / `attn_fpa__<b>` (make_attention_pair_configs :773–785) — **no builder edits**. |
| 14 | `tests/test_schema_attention_layers.py` | Add `"fpa"` to the expected enum set (:18–68, next to `"fastconformer"`); `test_cache_specs_match_layer_pattern_enum` assertion `len(enum_values) == 48` → **49** (:162) and its docstring "(47)" → "(48)". |
| 15 | `tests/test_gated_attention.py` | New `class FirstPassageAttentionTests(unittest.TestCase)` (peers at :119 `ForgettingAttentionTests`): `test_forward_encoder_shape`, `test_forward_decoder_causal` (bias is −inf-masked; last-token output changes when a mid-sequence token dips the field — sanity via a crafted φ, optional), `test_output_shape_and_finite` (no NaN/Inf), `test_knobs_honored` (`fpa_beta_init=2.5` → `beta` param all 2.5; `fpa_tau=0.5` accepted), `test_bitnet_on` (build + forward with `use_bitnet=True`), `test_special_case_fox_like` (δ ≤ 0, g = h = 0 ⇒ bias equals the cumulative log-forget form of a FoX head with f = exp(βδ) — small-tensor equivalence check, paper Prop. 2.2b). |
| 16 | `tests/test_bitnet_coverage.py` | Add `"fpa"` to the big `layer_pattern` list (:42 area, next to `"fox"`). |
| 17 | `tests/test_kv_cache_mixers.py` | **No edit required** — `MIXERS = sorted(m for m in IMPLEMENTED_MIXERS if m != "mamba")` (:37) auto-parametrizes fpa once it joins `IMPLEMENTED_MIXERS`; full-forward vs prefill+decode parity then runs over `PES = ("rope", "nope", "alibi")`. (If the `fpa_use_pe=False` default makes RoPE columns the interesting case, that is already covered — the test builds its own configs.) |
| 18 | `tests/test_attention_refactor.py` | Add `"fpa"` to `test_layer_type_coverage_trainable.layer_types` (:141–180, next to `"fox"`). |
| 19 | `src/model/norm/qk.py` | Add `"fpa"` to `QK_NORM_MIXERS` (after `"fox"`, :65) — fpa materializes explicit per-head q/k and calls `attach_qk_norm`. |
| 20 | `docs/paper/figures/mixers/fpa.mmd` | **New figure**, house style (init header like fox.mmd:1, `flowchart TD`, tensor-shaped labels): nodes `x[B,n,H]` → `Q,K,V = W·x` (Wqkv) and `δ,g,h = Wf·x` (Wf, bias) → `walk Wₜ = Σδ` (cumsum) → `field φ = W+g / level ψ = W+h` → `range-min φ̄ₜₛ = min_{s<u≤t} φ_u` → `bias bₜₛ = β·min(0, φ̄ₜₛ − ψₛ)` (β learned per head) → `scores = QKᵀ/√d + b` → `softmax` → `O = Wo·(softmax·V)[B,n,H]`; dotted edge `cumsum → walk` recurrence annotation. |
| 21 | `docs/paper/figures/mixer-taxonomy.mmd` | Root label `(48 variants)` → `(49 variants)` (:3); `g["Gated (8)"]` → `g["Gated (9)"]` and chain `… --> g8["kda"] --> g9["fpa"]` (:11). No gated family-map exists (only dense/latent/recurrent/sparse/memory) — family map row n/a. Rebuild pdf+svg via `build-figures.sh`. |
| 22 | `docs/bibliography/attention_types.bib` | `@misc{liu_first_passage_2026, …}` (§2.11), placed near the FoX entry (:219). |
| 23 | `docs/paper/appendices/annex-5-gated-attention.tex` | New `\subsection{9. First Passage Attention (FPA): Forgetting by Path Extrema}` after KDA's trade-off table (:556), before `\subsection{Unified Gating Principle}` (:560): `\begin{figure}[H] \fitfigure{figures/mixers/fpa.pdf} \caption{…} \label{fig:mixer-fpa}`; **Mathematical Foundation** — eqs of §2.1 verbatim (logits, bias, walk/field/level) + the four special cases (Prop. 2.2: stack/Lindley, **FoX**, bottleneck bias, Expire-Span) + permanence/stack property (Prop. 2.3) + STE training paragraph (§2.2) + blocked/decode cost (Prop. 2.4); `algorithm` float (parallel reference form: project → field → cumsum walk → suffix range-min → bias → softmax; note the exact decode recurrence); **Empirical Performance and Trade-offs** — 110M results (perplexity table vs FoX/Selective/RoPE, passkey 100% at 8×/16k, cost ratios 0.97× SDPA / 1.50× FoX, +1.56% cache, eviction θ=10 keeps 26–34%) + `tabularx` strength/limitation table (recall vs softmax lower bounds, depth-not-duration vs Selective Attention, constant-dimension alive sets; limitations: STE needed, one-seed 110M evidence, first-layer bias hurt MQAR) + `\citep{liu_first_passage_2026}`; note the encoder-mode symmetric extension is a repo-side generalisation. Extend the **Unified Gating Principle** list (:560–563, "All eight…" → "All nine…") with FPA's shift from path sums to path extrema. |
| 24 | `docs/paper/appendices/annex-15-kv-cache.tex` | Add an FPA bullet next to FoX's (:163): range-minimum bias; cache = KV + two scalars per key/head (φ, ψ); decode `φ̄_{t,s} = min(φ̄_{t−1,s}, φ_t)` exact (Prop. 2.4b); dead keys evictable (follow-up). |
| 25 | `docs/paper/appendices/annex-16-fixed-numerics.tex` | Append `fpa` to the gated list at :164 (`gated_softmax`, `fox` → `fox`, `fpa`) — fpa routes its softmax through `numerics.module_softmax`. |
| 26 | `docs/paper/appendices/annex-12-positional-encodings.tex` | Add `fpa` to the no-PE list (:228, `hgrn2`, `fox`, `kda` → … `kda`, `fpa`). Also fix the 1.2 rename artifact at :224–225: the "False (recurrent/decay mixers)" list reads `\texttt{retnet},` / `\texttt{retnet}, \texttt{mamba}, \texttt{ode},` — collapse the duplicated `retnet` to one (leftover of the blind `retnet_attn` → `retnet` replacement, commit 9b10768; see row 39). |
| 27 | `docs/paper/appendices/annex-summary-tables.tex` | Census `:4` "48 sequence mixers" → 49; header `:62` "**Gated (8)**" → (9); row after `kda` (`:71`): `fpa & Gated & $\mathcal{O}(n^2 d)$ & $\mathcal{O}(n)$ KV & Path-extremum (range-min) forget bias; contains FoX; two extra scalars per cached key/head & \citep{liu_first_passage_2026} \\`. |
| 28 | `docs/paper/appendices/annex-schema.tex` | Mixer row after the `fox` row (:257): `fpa & First Passage Attention with a range-minimum forget bias & \checkmark & \checkmark \\`. Also :75: "Retention heads for \texttt{retnet}/\texttt{retnet} layers." → "for \texttt{retnet} layers." (same rename artifact as row 39). |
| 29 | `docs/paper/sections/04-architecture-taxonomy.tex` | `:12` itemize "**Gated mechanisms** (8): … KDA" → "(9): … KDA, and First Passage Attention (FPA)"; `:24` caption "forty-eight sequence mixers" → "forty-nine"; `:80–81` "contains eight blocks" → "nine blocks", append FPA to the `\citep` list + a sentence (path-extrema gating in the full attention path; alive keys form a stack; strictly contains FoX). |
| 30 | `docs/paper/paper.tex` + `sections/01/10/11` (EN) | Abstract census "forty-eight sequence mixers" → "forty-nine" (the CA commit's exact sweep: paper.tex abstract, 01-introduction, 10-discussion, 11-conclusion — grep `forty-eight`). |
| 31 | `docs/paper-es/appendices/annex-5-gated-attention.tex` | **Faithful Spanish translation** of #23 (same structure/numbers/equations; `\fitfigure{figures/mixers/fpa.pdf}` — figures live under `../paper/figures/`), subsection "9. First Passage Attention (FPA): olvido por extremos de camino", `Fundamento Matemático`, `Rendimiento Empírico y Compromisos`, trade-off table, `Principio Unificado de Compuerta` ("las ocho…" → "las nueve…"). |
| 32 | `docs/paper-es/appendices/annex-15/16/12/summary/schema.tex` | Spanish twins of #24–#28 (ES summary census :4 "48 mezcladores" → 49; Gated ES header (8)→(9); row after `fox & Con compuerta …` :69: `fpa & Con compuerta & $\mathcal{O}(n^2 d)$ & $\mathcal{O}(n)$ KV & Sesgo de olvido por extremos de camino (mínimo de rango); contiene a FoX & \citep{liu_first_passage_2026} \\`; ES annex-schema row after :257; ES kv-cache/fixed-numerics/positional lists). Plus the ES rename artifacts: `annex-12-codificaciones-posicionales.tex:240` (`\texttt{retnet}, \texttt{retnet}, \texttt{mamba}` → single `retnet`) and `annex-schema.tex:75` (`capas \texttt{retnet}/\texttt{retnet}` → `capas \texttt{retnet}`) — row 39 family. |
| 33 | `docs/paper-es/sections/04-taxonomia-arquitectura.tex`, `01/10/11`, `paper-es.tex` | Spanish twins of #29–#30 ("(8)" → "(9)" + "y First Passage Attention (FPA)"; "ocho bloques" → "nueve bloques"; abstract "cuarenta y ocho mezcladores" → "cuarenta y nueve"; grep `cuarenta y ocho`). |
| 34 | `README.md` | Census (D10): :28 → 49; :75 → "**48** across 10 families: … Gated, …" ; :95 → all 49; Mixer Categories `:121` `**Gated** (8)` → `(9)`, list + `fpa` (word: "linear attention with multiplicative gates, delta rules, gated softmax, and path-extremum forgetting"); :130 → 49. |
| 35 | `docs/architecture.md` | §7.4 header :351 → "49 enum values in 10 families"; table row after `kda` (:382): ``| `fpa` | Gated | `fpa.*` | yes |`` + a description paragraph (pattern of the `memory`/`lsa`/`ca` paragraphs :418–460): equations, knobs, FoX special case, no-PE default, exact KV decode; §7.13 :664 "(47)" → "(48)"; implemented list :670–680 append `fpa` (KV + per-token φ/ψ extras, exact by Prop 2.4b) **and dedup the accidental `` `retnet`, `retnet` `` pair at :675** (the list names each cached mixer once); TRAINING_FREE note untouched. |
| 36 | `configs/README.md` | `:288` gated list append `fpa` (First Passage Attention, arXiv:2610.22714); `:318–323` QK-stabilization supported list append `fpa`; add `fpa` knob rows to the attention-parameters section if one enumerates `ca.*` (grep `learned_gain`). Also :286: the legacy-blocks list reads ``- Legacy blocks: `retnet`, `retnet`, `mamba`, `ode`, …`` — collapse the duplicated `retnet` (rename artifact, row 39 family). |
| 37 | `docs/source/api/model/attention/gated/fpa.rst` + `gated/index.rst` | `fpa.rst` = "First Passage Attention\n===============\n\n.. automodule:: src.model.attention.gated.fpa\n   :members:\n   :undoc-members:\n   :show-inheritance:" (fox.rst shape); toctree entry `fpa` after `kda` (index.rst). |
| 38 | `full_tests/param_truth.json` | **Regenerated, not hand-edited** (`param_count_check.py` rebuilds it from every preset+example; the two new YAMLs add entries — intentional diff, like the CA commit). |
| 39 | `docs/paper/appendices/annex-2-recurrent-attention.tex` + `docs/paper-es/appendices/annex-2-recurrent-attention.tex` | **`retnet` rename-artifact dedup** (user-requested; execute with the docs pass, rows 22–36). The HybridLayer dispatch algorithm still carries the pre-1.2 alias pair: EN :347 and ES :348 read `\ElsIf{$m\in\{\texttt{retnet},\texttt{retnet}\}$}` → collapse to `\ElsIf{$m=\texttt{retnet}$}` (style of the neighbouring `$m=\texttt{mamba}$` branch; the `retnet_attn` alias was removed in 9b10768 and the blind replace turned `retnet`/`retnet_attn` into `retnet`/`retnet`). Same artifact family, fixed inside rows 26/28/32/35/36: annex-12 EN :224–225 + ES :240, annex-schema EN/ES :75, `architecture.md:675`, `configs/README.md:286`. Deliberately untouched: `annex-21`/`09` EN+ES mentions of "the now-removed `retnet_attn` alias" — intentional historical record. |

---

## 5. Ordered execution checklist

```bash
# ── Implementation (rows 1–13) ──────────────────────────────────────────────
# 1  src/model/attention/gated/fpa.py
# 2  gated/__init__.py + attention/__init__.py
# 3  hybrid_layer.py registry
# 4  config.py fields + validation
# 5  config_flatten.py renames
# 6  _dims.yaml enum + both tables
# 7  _attention_fpa.yaml
# 8  _model.yaml $ref
# 9  _positional_encoding.yaml use_pe
# 10 capabilities.py (spec + IMPLEMENTED_MIXERS + census)
# 11 configs/examples/es_arch_fpa_adamw.yaml
# 12 configs/fpa.yaml
# 13 full_tests/run_e2e.py ATTENTIONS

# ── Tests (rows 14–19) ─────────────────────────────────────────────────────
# 14 tests/test_schema_attention_layers.py
# 15 tests/test_gated_attention.py
# 16 tests/test_bitnet_coverage.py
# 17 tests/test_kv_cache_mixers.py   (no edit; auto-parametrized)
# 18 tests/test_attention_refactor.py
# 19 src/model/norm/qk.py QK_NORM_MIXERS

# ── Targeted gates (run at implementation time; NOT now) ───────────────────
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_attention_refactor.py \
  tests/test_gated_attention.py tests/test_bitnet_coverage.py \
  tests/test_kv_cache_mixers.py \
  -v --tb=short -p no:warnings

# ── Full suite ─────────────────────────────────────────────────────────────
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

```bash
# ── e2e — enumerate the new mixer's combos (build-only, no training) ───────
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --list-combos | grep -F fpa
# expect: attn_single_fpa plus attn_<a>__fpa / attn_fpa__<b> for every other mixer

# ── e2e TARGETED SMOKE — ONLY the new mixer: real training ─────────────────
# --realonall is the DEFAULT: selected combos train on the real Borges corpus
# (adleme94/borges) with the BETO tokenizer; first run downloads from the HF Hub.
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only fpa
# Every selected combo must report PASS with finite loss. A NaN/Inf loss,
# schema rejection or timeout is a BLOCKER.

# minimal fallback if the pair sweep is too heavy: the solo combo only
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only attn_single_fpa

# offline synthetic fallback (no HF download), same selection
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only fpa --no-realonall

# sizing knobs when needed: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800
# GPU smoke (real training box): swap the extra for --extra cu118 --extra train.
# Evidence to keep: the run dir's generated config.yaml + metrics (finite loss).
```

```bash
# ── Figures (needs Chrome/Chromium — /usr/bin/chromium verified — or PUPPETEER_EXECUTABLE)
( cd docs/paper/figures && ./build-figures.sh both )   # renders fpa.pdf/svg + taxonomy

# ── Param ground truth + JS estimator (after rows 11–12 land)
uv run --extra cpu --extra train python full_tests/param_count_check.py   # → "… configs counted, 0 failures"
node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01                                  # → ALL PASS

# ── Sphinx API docs
bash docs/build_docs.sh
```

```bash
# ── PDF rebuild (repo root → docs/paper, then docs/paper-es) — implementation time only
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

**Checklist order**: rows 1–13 → tests 14–19 → targeted pytest → full pytest →
e2e enumeration → e2e real-data smoke → figures (20–21) → docs EN/ES
(22–36, **including the row-39 `retnet` rename-artifact dedup pass**) →
param truth + estimator → Sphinx (37) → PDFs → website sync (§6) →
commit/push (§7).

**Preconditions / fallbacks**: `pdflatex`, `bibtex`, `chromium` verified on
this machine — if unavailable at implementation time, leave sources updated
and note the rebuild must happen before release (website copies prebuilt
PDFs); if Chrome is missing, commit the `.mmd` and render later; if the HF Hub
is unreachable, record the `--no-realonall` PASS and re-run the real-data pass
later. **PDF verification**: no `Fatal error`, no `Citation .* undefined`
(`liu_first_passage_2026` must resolve), `paper.pdf`/`paper-es.pdf` newer than
the edited `.tex`/`.bbl`, and `git status --short docs/paper docs/paper-es`
shows only intended `.tex`/`.bbl`/`.pdf`/figure files.

---

## 6. Website sync (planned, not executed — `/sync-website`)

1. Bootstrap/refresh the mirror (`git clone … erickfmm.github.io` if the
   state script says MISSING) and copy per the file map: `src/schema.yaml` +
   `src/schema/**` (new `_attention_fpa.yaml`), `configs/*.yaml`
   (`fpa.yaml`) + `configs/README.md`, `configs/examples/**`
   (`es_arch_fpa_adamw.yaml`), `docs/paper/**` + `docs/paper-es/**`
   (tex/bbl/pdf + `figures/mixers/fpa.{mmd,pdf,svg}` + rebuilt taxonomy
   pdf/svg), `docs/bibliography/**`, `docs/architecture.md`.
2. **Site-only edits** (mirror is their source of truth):
   - `ft-diagram.js`: `CATS["fpa"] = { c: 'attn', l: 'First Passage Attention' }`
     (next to `fox`, :19) and a `SUBSTRUCT["fpa"]` node list (next to fox's,
     :234):
     ```js
     fpa: [
         { n: 'x [B,n,H]', c: 'attn', id: 'x' },
         { n: 'Q,K,V = W·x [B,n,H]', c: 'attn', e: 'Wqkv', from: ['x'] },
         { n: 'δ,g,h = Wf·x (3 scalars/head)', c: 'attn', e: 'Wf+b', from: ['x'] },
         { n: 'walk Wₜ = Σδ; φ = W+g, ψ = W+h', c: 'attn', e: 'cumsum', from: ['δ,g,h = Wf·x (3 scalars/head)'] },
         { n: 'range-min φ̄ₜₛ = min_{s<u≤t} φ_u', c: 'attn', e: 'RangeMin', from: ['walk Wₜ = Σδ; φ = W+g, ψ = W+h'] },
         { n: 'bias bₜₛ = β·min(0, φ̄ₜₛ − ψₛ)', c: 'attn', e: 'β (per head)', from: ['range-min φ̄ₜₛ = min_{s<u≤t} φ_u'] },
         { n: 'scores = QKᵀ/√d + b', c: 'attn', e: 'add bias', from: ['Q,K,V = W·x [B,n,H]', 'bias bₜₛ = β·min(0, φ̄ₜₛ − ψₛ)'] },
         { n: 'softmax(scores)', c: 'attn', e: 'softmax', from: ['scores = QKᵀ/√d + b'] },
         { n: 'O = Wo·(softmax·V) [B,n,H]', c: 'attn', e: 'Wo', from: ['softmax(scores)', 'Q,K,V = W·x [B,n,H]'] }
     ],
     ```
     No `STYLES` change (stays in the existing `attn` class).
   - `ft-param-estimator.js`: add to `mixerParams` (near the `ca` case, ~:97):
     ```js
     // ---- First Passage Attention (arXiv:2610.22714) ----
     // Q,K,V,O (bias-free) = 4H²; field projection H→3·nH WITH bias
     // (paper Table 15) = 3·nH·(H+1); learned per-head strength β = nH.
     case 'fpa':
       return 4 * H * H + 3 * nH * (H + 1) + nH;
     ```
     (knob-independent — `beta_init`/`tau` change no counts.)
   - `index.html`: grep a peer (`fox`, `kda`) and extend every hardcoded mixer
     list found with `fpa` (+ its `attention.fpa` knob block if the Builder
     renders per-mixer sub-objects — it is schema-driven, so likely only
     census/diagram lists need it).
3. Paper tab HTML rebuild (`flatten_tex.py --web` + `pandoc` +
   `postprocess_html.py`, en + es) and top-level `paper/en.pdf` / `es.pdf`.
4. **Validation Gate**: `param_count_check.py` → `0 failures` (regenerates
   `param_truth.json` — intentional diff); estimator `--tol 0.01` → `ALL PASS`;
   JSON round-trip script; CLI parse script; state-script probes
   (`schema-refs`, `example-yaml`, `cli-flags`, …) all PASS.
5. Mirror stays uncommitted until the commit phase.

---

## 7. Commit plan (planned, not executed — `/make-a-commit` conventions)

Precedent: the CA mixer landed as **two** commits — `2517182 docs(plan): add
Continuum Attention mixer plan…` then `6dc1170 feat(attention): add Continuum
Attention mixer…`.

1. *(optional, user's call — this plan file)*
   `docs(plan): add First Passage Attention mixer plan (arXiv:2610.22714)` —
   `.opencode/plan/add-mixer-2610.22714.md`.
2. Main repo: `feat(attention): add First Passage Attention mixer (arXiv:2610.22714)`
   — body: module + exports + registry; schema (enum + both bilingual tables +
   `_attention_fpa.yaml` + `$ref` + renames + dataclass fields/validation +
   use_pe); cache decision (spec + `IMPLEMENTED_MIXERS` + exact decode branch,
   Prop. 2.4b); example + `configs/fpa.yaml`; tests (schema sets/49, gated
   class, bitnet, refactor coverage, kv parity auto); e2e `ATTENTIONS` +
   real-data smoke result; figures (`fpa.mmd` + rendered pdf/svg + taxonomy
   bump); docs (annex-5 §9 EN/ES, summary tables + census 49, schema annex,
   kv-cache/fixed-numerics/positional annex lists, section 04 EN/ES + abstract
   census, README/architecture/configs README, Sphinx rst; plus the `retnet` rename-artifact dedup — annex-2/12/schema EN+ES, architecture.md §7.13, configs/README.md); rebuilt PDFs;
   `param_truth.json` (intentional). Stage exactly:
   ```
   src/model/attention/gated/fpa.py  src/model/attention/gated/__init__.py
   src/model/attention/__init__.py  src/model/hybrid_layer.py
   src/model/config.py  src/utils/config_flatten.py  src/model/norm/qk.py
   src/schema/_model/_dims.yaml  src/schema/_model/_attention_fpa.yaml
   src/schema/_model.yaml  src/schema/_model/_positional_encoding.yaml
   src/model/cache/capabilities.py
   configs/examples/es_arch_fpa_adamw.yaml  configs/fpa.yaml
   full_tests/run_e2e.py  full_tests/param_truth.json
   tests/test_schema_attention_layers.py  tests/test_gated_attention.py
   tests/test_bitnet_coverage.py  tests/test_attention_refactor.py
   docs/paper/figures/mixers/fpa.{mmd,pdf,svg}
   docs/paper/figures/mixer-taxonomy.{mmd,pdf,svg}
   docs/bibliography/attention_types.bib
   docs/paper/appendices/annex-{5,15,16,12}*.tex docs/paper/appendices/annex-summary-tables.tex docs/paper/appendices/annex-schema.tex
   docs/paper/sections/*.tex docs/paper/paper.{tex,bbl,pdf}
   docs/paper-es/appendices/annex-{5,15,16,12}*.tex docs/paper-es/appendices/annex-summary-tables.tex docs/paper-es/appendices/annex-schema.tex
   docs/paper-es/sections/*.tex docs/paper-es/paper-es.{tex,bbl,pdf}
   README.md  docs/architecture.md  configs/README.md
   docs/source/api/model/attention/gated/{fpa.rst,index.rst}
   ```
   **Never stage** `*.aux *.log *.out *.toc *.blg`, caches, mermaid temp
   files. Then `git push origin HEAD:main`.
3. Mirror: `git -C erickfmm.github.io …` commit
   `docs(site): sync First Passage Attention mixer (arXiv:2610.22714)` —
   schema tree, presets/examples, paper trees + figures + PDFs/HTML,
   bibliography, architecture.md, `ft-diagram.js`, `ft-param-estimator.js`,
   `index.html`; then `git -C erickfmm.github.io push origin HEAD:main`.

---

## 8. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Schema mixer gates | `pytest tests/test_schema_attention_layers.py tests/test_schema_consistency.py -v` | expected-set ⊇ enum; `len(enum) == 49`; `CACHE_SPECS` keys == enum; defaults ↔ dataclass pairs (2 new `fpa` entries); both `_dims.yaml` tables match enum |
| Examples/presets | `pytest tests/test_yaml_examples.py -v` | `es_arch_fpa_adamw.yaml` + `fpa.yaml` load via `load_training_config` |
| Mixer unit tests | `pytest tests/test_gated_attention.py tests/test_attention_refactor.py -v` | new class green; coverage list includes `fpa`; encoder+decoder shapes finite |
| BitNet coverage | `pytest tests/test_bitnet_coverage.py -v` | `fpa` in the mixed 18-mixer stack, forward finite with `use_bitnet=True` |
| KV parity (decision 1) | `pytest tests/test_kv_cache_mixers.py -v` | `fpa` auto-parametrized; full vs prefill+decode greedy match (rtol 1e-4 fallback) for rope/nope/alibi |
| Full suite | `pytest tests/ --continue-on-collection-errors -v` | 77 files green (no regression) |
| e2e enumeration | `run_e2e.py --category attn --list-combos \| grep -F fpa` | `attn_single_fpa` + every `attn_<a>__fpa` / `attn_fpa__<b>` |
| **e2e real-data training** | `run_e2e.py --category attn --only fpa` | all selected combos **PASS**, finite loss on the real Borges corpus; run dir keeps `config.yaml` + metrics |
| e2e solo variant | `run_e2e.py --category attn --only attn_single_fpa` | PASS, finite loss |
| e2e offline fallback | `run_e2e.py --category attn --only fpa --no-realonall` | PASS on synthetic (only if HF Hub unreachable) |
| Figures | `build-figures.sh both` | `fpa.pdf`/`fpa.svg` + taxonomy re-rendered, no mermaid errors |
| Param truth | `python full_tests/param_count_check.py` | `… configs counted, 0 failures`; `param_truth.json` gains both configs with `attention` category matching 4H²+3nH(H+1)+nH per fpa layer |
| Estimator | `node full_tests/param_estimate_check.mjs … --tol 0.01` | `ALL PASS` ≤ 1% |
| Sphinx | `bash docs/build_docs.sh` | `gated/fpa.html` generated, no warnings about missing module |
| PDF EN/ES | pdflatex/bibtex ×3 each | no `Fatal error`, no undefined citation of `liu_first_passage_2026`, PDFs newer than sources, `git status` clean of aux files |
| Mirror probes | state-script + Validation Gate | `schema-refs`, `example-yaml`, `cli-flags`, estimator, JSON round-trip, CLI parse — all PASS |

---

## 9. Risks, assumptions and open questions

**Assumptions to verify at implementation time**
- Line numbers drift (anchors taken at HEAD `6dc1170`, clean tree).
- `torch.cummin`/`logcumsumexp` on flipped tensors: confirm dtype/device
  behaviour and that the +inf masking gives `b_tt = 0` exactly (unit-test it).
- fp32 field under AMP: repo trains CPU/fp32 by default; if a CUDA/AMP run
  NaNs, cast the field/bias path to `float()` explicitly (paper Table 16).
- `scheduler_warmup_ratio` vs the paper's 512 fixed steps — check whether the
  schema supports absolute warmup steps before rounding (row 12).
- HF Hub reachability for the real-data smoke (offline fallback documented).
- Drop the `local_parquet_dir`/`prefer_local_cache`/`stream_local_parquet`
  trio from the example (CA's copy is machine-specific) — confirm the example
  still loads without them (`load_training_config`).

**Risks**
- STE subtlety: gradient must flow to δ, g (via soft suffix-min), ψ and β (via
  the active hard outer min). A wrong detach silently under-trains — the
  `test_special_case_fox_like` + a small gradcheck (field params receive
  non-zero grads) should be added if the peer tests leave doubt.
- Encoder-mode symmetric extension is a repo invention (paper defines causal
  only) — must be labelled as such in annex-5 and architecture.md.
- O(T²) memory for the bias matrix matches `fox`'s house pattern; e2e sizes
  (≤512) are safe, but note it in the module docstring (Triton kernel out of
  scope, CPU-first D7).
- `beta` under BitNet: raw `nn.Parameter` (not BitLinear) — BitNet weight
  quantization does not apply to it; keep it out of any bitnet-specific
  param sweeps (BitNet coverage test just checks finite forward).
- Naming collisions: `fpa` grep'd clean across `src/`, `configs/`, `full_tests/`.
- Task/mode rules (rule 6): `fpa` is a plain mixer — no model-class or task
  restrictions; `causal_lm` preset forces decoder (already handled).

**Open questions / optional follow-ups (need explicit approval)**
1. **Code-level `retnet` duplicates** (observed, *not* planned — user scoped the
   dedup to the paper/md artifacts): `full_tests/run_e2e.py:225`
   (`ATTENTIONS = ["retnet", "retnet", …]` — via `itertools.combinations` this
   generates one spurious `attn_retnet__retnet` pair combo in the e2e sweep),
   `tests/test_attention_refactor.py:144–145` and
   `tests/test_schema_attention_layers.py:19–20` (harmless literal
   duplicates). Removing the e2e one changes the sweep composition — needs
   explicit approval before it rides along with any commit.
2. **Cache-eviction policy** (paper §4.3): drop keys with `b < −θ`
   (θ≈10, keeps 26–34%) as a `CachePolicy` over the KVSlot (D14 `sparsek`
   precedent) + e2e/eval measurement. Out of scope for v1.
3. Triton/blocked kernel (paper Appendix E) for GPU training throughput —
   out of scope (CPU-first, D7).
4. First-layer bias ablation guidance (paper's MQAR fix) — documentation
   only, no repo behaviour.

---

## 10. Out of scope

This command is a planning command: **no** code, schema, preset, test,
figure, doc, PDF, mirror or git change was made. Nothing outside
`.opencode/plan/add-mixer-2610.22714.md` was written; no mutating command
(pytest, training, pdflatex, uv sync, git add/commit/push, website copy) was
executed. Implementation is a separate, explicitly requested step.
