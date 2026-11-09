# Plan: add Continuum Attention mixer (arXiv:2610.20586)

> **Status: PLANNED — nothing outside `.opencode/plan/` has been touched.**
> Registry name: `ca` · Class: `ContinuumAttention` · Family: **Dense** (top-level module) ·
> Registry name and gain defaults approved by the user (2026-11-09).

## 1. Header

**Paper**: *Continuum Attention*, Afshin Khadangi (SnT, University of Luxembourg),
arXiv:2610.20586 (v2, 26 Oct 2026, working paper). Conventional causal attention aggregates
values only at discrete token positions; **Continuum Attention (CA)** instead interpolates
adjacent key/value vectors into continuous, piecewise-linear memory trajectories
`k(t)`, `v(t)` and lets each query retrieve by **normalized integration** of an exponential
relevance density `exp(s_i(t))/Z_i` over its visible prefix `[-1/2, i+1/2]` (constant
half-token extensions at both ends). Because the score is affine inside each segment, the
integrals have **closed forms in exponential moments** `J_n(Δ) = ∫₀¹ uⁿ e^{Δu} du` (n = 0, 1),
so the operator is exactly computable without quadrature, analytically differentiable (via
`J_2`), preserves causal masking and normalized non-negative effective weights, and drops
into a standard Transformer attention block (usual Q/K/V/O projections). The paper trains a
101M byte-level decoder on 26.2B FineWeb-Edu bytes: 42% fewer bytes than a matched
Transformer to reach 1.10 BPB, and a FLORES eight-language macro-average of 1.8915 BPB
(5.8% better than the strongest baseline, winning every language).

**Why it belongs in this repo**: CA is a fully trainable dense attention variant — a
drop-in replacement of the softmax operator with identical projections (fixed-gain variant
adds **zero** parameters), O(n²d) training like `standard`, classic per-token KV caching
(class i), and a per-head learned score gain. It slots into the Dense family beside
`standard` and `sigmoid` and exercises exactly the axis this toolkit exists to compare.

---

## 2. Paper extract

All equations verbatim from arXiv:2610.20586v2 (HTML). Indices: layer/head omitted; head
dim `d_h`; positions 0-based; visible prefix for causal query `i` is `j ≤ i`.

### 2.1 Forward equations

Projections (Eq. 1) — `q_i, k_i, v_i ∈ R^{d_h}`:

```latex
\mathbf{q}_i = W_Q \mathbf{x}_i, \qquad
\mathbf{k}_i = W_K \mathbf{x}_i, \qquad
\mathbf{v}_i = W_V \mathbf{x}_i
```

Piecewise-linear memory (Eqs. 2–3), for `u = t − (j−1) ∈ [0,1]`, `1 ≤ j ≤ i`, plus constant
half-token extensions on `[-1/2, 0]` (value `v_0`, score `s_i(0)`) and `[i, i+1/2]`
(value `v_i`, score `s_i(i)`):

```latex
\mathbf{k}_j(u) = (1-u)\,\mathbf{k}_{j-1} + u\,\mathbf{k}_j, \qquad
\mathbf{v}_j(u) = (1-u)\,\mathbf{v}_{j-1} + u\,\mathbf{v}_j
```

Query-adaptive relevance density (Eqs. 4–8); `γ > 0` is the score gain (`γ = 1/τ`):

```latex
s_i(t) = \frac{\gamma}{\sqrt{d_h}}\,\mathbf{q}_i^\top \mathbf{k}(t)                     \tag{4}
a_{ij} = s_i(j-1), \quad b_{ij} = s_i(j), \quad \Delta_{ij} = b_{ij} - a_{ij}            \tag{5}
s_{ij}(u) = a_{ij} + u\,\Delta_{ij}, \quad u \in [0,1]                                    \tag{6}
p_i(t) = \frac{\exp(s_i(t))}{Z_i}, \quad
Z_i = \int_{-1/2}^{\,i+1/2} \exp(s_i(t))\,dt                                              \tag{7}
\boxed{\ \mathbf{o}_i = \int_{-1/2}^{\,i+1/2} p_i(t)\,\mathbf{v}(t)\,dt\ }                \tag{8}
```

Query adaptation is the segment geometry (Eq. 9):

```latex
\Delta_{ij} = \frac{\gamma}{\sqrt{d_h}}\,\mathbf{q}_i^\top (\mathbf{k}_j - \mathbf{k}_{j-1})
```

Exponential moments (Eqs. 10–13); continuous limits `J_0(0)=1, J_1(0)=1/2, J_2(0)=1/3`:

```latex
J_n(\Delta) = \int_0^1 u^n \exp(\Delta u)\,du, \quad n \in \{0,1,2\}                      \tag{10}
J_0(\Delta) = \frac{e^{\Delta}-1}{\Delta}                                                 \tag{11}
J_1(\Delta) = \frac{(\Delta-1)e^{\Delta}+1}{\Delta^2}                                     \tag{12}
J_2(\Delta) = \frac{(\Delta^2-2\Delta+2)e^{\Delta}-2}{\Delta^3}                           \tag{13}
```

Exact segment integration (Eqs. 14–18) — segment mass, value contribution, adaptive
retrieval position:

```latex
M_{ij}  = \int_0^1 \exp(a_{ij} + u\Delta_{ij})\,du = e^{a_{ij}} J_0(\Delta_{ij})          \tag{14}
\mathbf{U}_{ij} = e^{a_{ij}}\big[(J_0 - J_1)\,\mathbf{v}_{j-1} + J_1\,\mathbf{v}_j\big]   \tag{15}
\mu(\Delta_{ij}) = \frac{J_1(\Delta_{ij})}{J_0(\Delta_{ij})}                              \tag{16}
\mu(\Delta) = \frac{1}{1-e^{-\Delta}} - \frac{1}{\Delta}, \quad \mu(0) = \tfrac12        \tag{17}
\mathbf{U}_{ij} = M_{ij}\big[(1-\mu)\,\mathbf{v}_{j-1} + \mu\,\mathbf{v}_j\big]           \tag{18}
```

Boundary caps and normalization (Eqs. 19–23) — the boxed output:

```latex
Z_i^{\mathrm{cap}} = \tfrac12 e^{s_i(0)} + \tfrac12 e^{s_i(i)}                            \tag{19}
\mathbf{U}_i^{\mathrm{cap}} = \tfrac12 e^{s_i(0)}\mathbf{v}_0 + \tfrac12 e^{s_i(i)}\mathbf{v}_i \tag{20}
Z_i = Z_i^{\mathrm{cap}} + \sum_{j=1}^{i} M_{ij}                                          \tag{21}
\mathbf{U}_i = \mathbf{U}_i^{\mathrm{cap}} + \sum_{j=1}^{i} \mathbf{U}_{ij}               \tag{22}
\boxed{\ \mathbf{o}_i = \frac{\mathbf{U}_i^{\mathrm{cap}} + \sum_{j=1}^{i}\mathbf{U}_{ij}}
                       {Z_i^{\mathrm{cap}} + \sum_{j=1}^{i} M_{ij}}\ }                     \tag{23}
```

Properties (Eq. 24 + §2.4): `i = 0 ⇒ o_0 = v_0`; equal visible scores ⇒ **exactly uniform
causal attention** (identical total mass per token); and the equivalent effective-weight
form:

```latex
\mathbf{o}_i = \sum_{j=0}^{i} w_{ij}\mathbf{v}_j, \qquad w_{ij} \ge 0, \qquad \sum_{j=0}^{i} w_{ij} = 1
```

### 2.2 Analytic differentiation (§2.5, Eqs. 25–33) — reference only

With `g_i = ∂L/∂o_i` and `r_{ij} = g_i^⊤(v_j − o_i)`, segment score gradients use
`(J_0 − 2J_1 + J_2)`, `(J_1 − J_2)`, `J_2` at `Δ_ij`; caps add `½e^{s}/Z_i · r` at the
first/last visible scores; value gradients are `∂L/∂v_j = Σ_i w_{ij} g_i`; query/key
gradients accumulate `g^s_{ij}` with factor `γ/√d_h`; the learned gain obeys
`∂L/∂ρ = Σ_i Σ_j g^s_{ij} s_i(j)`.

**Repo implementation note**: PyTorch autograd over the closed-form moments is *exact*
(the paper's analytic backward is a kernel-efficiency optimization, not a correctness
requirement). No custom `backward()` is needed; `J_2` is therefore not required in the
module (only `J_0`, `J_1`), but the plan keeps a `J_2` helper for the moment unit test.

### 2.3 Numerical evaluation (§2.6, Eq. 34)

```latex
J_n(\Delta) = \sum_{r=0}^{\infty} \frac{\Delta^r}{r!\,(n+r+1)}
```

* Evaluate `J_0/J_1` with the closed forms **away from** `Δ ≈ 0` and the truncated series
  (≈3 terms) **near** `Δ = 0` (catastrophic cancellation in `e^Δ − 1`).
* Compute the normalizer after **subtracting the row max** of the visible scores.
* All segment integrals are independent ⇒ fully parallel; moment math in fp32.

### 2.4 Algorithm / pseudocode

**Parallel (training) form**, per head — this is what the module implements:

```
Input: Q, K, V ∈ R^{B×nH×n×hd}, gain γ ∈ R^{nH}
S ← γ ⊙ (Q·Kᵀ)/√hd                      # endpoint score matrix [B,nH,n,n]
S ← S + PE-score-bias                    # ALiBi/BAM act here (apply_pe_to_scores)
mask S to the visible prefix             # decoder: j ≤ i; encoder: all j
row-max m ← max over visible entries of S
A ← S[:, :-1], B ← S[:, 1:]              # segment endpoints (within visible prefix)
Δ ← B − A;  J0, J1 ← moments(Δ)          # Eqs. 11–12 (+ Eq. 34 near 0), fp32
M ← exp(A − m)·J0                        # Eq. 14, per segment
μ ← J1/J0 (μ←½ where J0=0)               # Eq. 16
Z ← ½e^{S[:,0]−m} + ½e^{S[own]−m} + Σ_seg M        # Eqs. 19+21
w ← 0;  w[:, 0] += ½e^{S[:,0]−m};  w[:, own] += ½e^{S[own]−m}
w[:, :-1] += M(1−μ);  w[:, 1:] += M·μ;  w /= Z      # Eq. 23 → effective weights
o ← w·V                                  # (B,nH,n,hd)
```

`own` = the row's last visible position: decoder → diagonal `i`; encoder → `n−1`.

**Recurrent / decode form** (§2.6): at step `i` with cached `(K, V)` of the visible prefix,
recompute the single endpoint-score row and the same segment integrals — i.e. a
row-local re-normalization over the cache (sigmoid's L1 re-sum pattern), **exact**.

### 2.5 Parameter inventory and exact count formula

| Tensor | Shape | Bias | Notes |
|---|---|---|---|
| `q_proj`, `k_proj`, `v_proj`, `out_proj` | `H × H` each | **False** | BitLinear when BitNet on |
| `log_gain` (only when `ca.learned_gain: true`) | `(nH,)` | — | `γ = exp(log_gain)`, init `log(gain_init)`; paper: "learned positive gain … one scalar per layer and head" |
| — (fixed-gain variant) | — | — | paper §2.7: "introduces **no additional trainable parameters** relative to a conventional softmax attention layer" |

**Parameter-count formula** (per CA layer, in the repo's `H, nH, nKv, hd, fH` terms —
independent of `nKv`, `hd`, `fH`; CA is plain MHA-width):

```
attention(ca) = 4·H² + (ca.learned_gain ? nH : 0)
```

Cross-checks: paper large model — 8 layers × 16 heads = 128 learned log-gains ("All 128
head specific log score gains are learned and initialized at zero"); small model —
"Variants with learned attention gains introduce 24 additional scalar parameters"
(6 layers × 4 heads). Fixed-gain `B0.30`↔`P` parity: identical parameter counts.

### 2.6 Knobs (→ `model.attention.ca`)

| Schema key | Flat field | Type | Default | Constraints | Meaning |
|---|---|---|---|---|---|
| `attention.ca.learned_gain` | `ca_learned_gain` | bool | **true** | — | CA-H/101M flagship: learn `γ = exp(ρ)` per head, init at `gain_init`; `false` = fixed-gain (CA-P style, zero extra params) |
| `attention.ca.gain_init` | `ca_gain_init` | number | **1.0** | `> 0` | initial (learned) or fixed score gain `γ = 1/τ`. Paper: 101M init `γ=1`; CA-P `γ = 1/0.30 ≈ 3.3333`; CA-H-small `γ = 1/0.30` |

Plus the standard per-mixer PE flag `ca_use_pe` (default `true`).

### 2.7 State & cache

* No recurrent state; per-step decode needs all past (post-RoPE) K/V plus the current
  query — **compatibility class (i) classic per-token K/V → `KVSlot`**.
* Per-step decode cost: `O(i · d_h)` — one endpoint-score row + row-local moments over the
  cached prefix (same asymptotics as `sigmoid`'s re-summed L1 normalizer).
* Effective weights are a closed-form function of the endpoint-score row, so the cached
  branch is exact via the generic `cached_qk_attention(..., normalize=<CA weights>)` F1
  recipe (sigmoid precedent, `src/model/cache/hooks.py:143`).

### 2.8 Complexity

* **Training**: `O(T² d_h)` per head — "matching the asymptotic complexity of dense softmax
  attention"; `O(T²)` attention storage if materialized (tiled kernels can avoid it — out
  of scope; the repo materializes, like every other dense mixer).
* **Inference**: `O(n)` time per token with KV cache, `O(n)` cache space.
* Summary-table cells: Training `O(n² d)`, Inference `O(n)` cache.

### 2.9 Trainability

Fully trainable from scratch in **both** modes (the parallel form above is the training
form; autograd through the closed-form moments is exact). **Not** training-free/eval-only:
no `TRAINING_FREE_LAYERS` membership, no `dense_forward` surrogate, no ⚠️ rows.

### 2.10 Positional encoding

Consumes PE: the 101M model uses **RoPE**; the small model uses learned absolute. PE
applies to the scoring pair exactly as in `standard` (rotary PEs rotate q/k before the
score matrix; score-bias PEs — ALiBi/BAM — bias the endpoint score matrix `S`).
⇒ `per_mixer_use_pe.ca` default **true**, dataclass `ca_use_pe: bool = True`.

### 2.11 Mode constraints

* **Decoder**: paper-native (causal; §2 is derived causally; causality verified bitwise in
  §3.6 — future-position perturbations leave prefix outputs unchanged).
* **Encoder**: the paper defines no bidirectional form. **Repo extension (documented
  deviation)**: with all positions visible, the trajectory domain is `[-1/2, n−1/2]` with
  segments `j = 1..n−1` and caps `½e^{s_i(0)} + ½e^{s_i(n−1)}` — the natural symmetric
  extension, needed because the e2e attn sweep trains every mixer as encoder MLM. Flagged
  in the module docstring and the annex ("encoder-mode extension follows the paper's
  uniform-mass convention over the full trajectory").
* Not grid/vision-specific, not audio-specific: any task/mode combination allowed by the
  backbone (hard rule 6 imposes no extra constraint on `ca`).

### 2.12 BibTeX data

```bibtex
@misc{khadangi_continuum_2026,
	title = {Continuum Attention},
	author = {Khadangi, Afshin},
	year = {2026},
	eprint = {2610.20586},
	archivePrefix = {arXiv},
	primaryClass = {cs.LG},
	url = {https://arxiv.org/abs/2610.20586},
	publisher = {{arXiv}}
}
```

Empirical numbers for the annex (three-seed, 101M, 26.2B bytes): FineWeb-Edu 1.0229 BPB,
PG-19 1.4717, Stack-smol 3.3451 (worst of the five — domain weakness), FLORES macro 1.8915
(lowest in all 8 languages; 5.8% avg BPB reduction); 42% fewer training bytes to 1.10 BPB
(8.9B vs 15.4B, ratio ≈1.74); HellaSwag 30.70 / PIQA 60.23 / ARC-E 34.04 / ARC-C 23.08;
robustness: clean 30.70, drop 28.58, repeat 30.04, antspeak 25.76, uppercase 30.39,
randomcase 29.37. Reference code: `github.com/akhadangi/continuum-attention` (Metal/CUDA).

---

## 3. Decisions (Phase 3)

| Decision | Value | Evidence |
|---|---|---|
| Registry name | **`ca`** | Paper's short name ("Continuum Attention (CA)"); `kda`/`swa`/`gla`/`fox` precedent. No collision (`cca` ≠ `ca`; grep clean). **User-approved.** |
| Class / file | `ContinuumAttention` / `src/model/attention/ca.py` | Top-level Dense family like `standard.py`, `sigmoid.py` (§11.1: "gated/, latent/, sparse/, or top level") |
| Family label | **Dense** | §2.7: "retains the surrounding decoder architecture"; full visibility, O(n²d); convex aggregation semantics |
| Peer modules | `standard.py` (constructor, `use_pe`, qk-norm, causal mask, KV branch) + `sigmoid.py` (custom `normalize` callback in the cached path, `src/model/attention/sigmoid.py:121-149`) | closest by family and shape |
| Knobs | `ca.learned_gain` (true) + `ca.gain_init` (1.0) | 101M flagship config; **user-approved**. CA-P = `{learned_gain: false, gain_init: 3.3333}` |
| `use_pe` | default **true** (`ca_use_pe: bool = True`) | paper uses RoPE / learned absolute; softmax-attention-like |
| KV cache | **Decision 1 (simple decode form)**: `CacheSpec(supported=True, slot_type="kv")` + incremental branch via `cached_qk_attention` with CA `normalize` callback + `IMPLEMENTED_MIXERS` + auto-parametrized parity | class (i); weights are a closed-form row-local function of the endpoint scores → exact; `hooks.cached_qk_attention` docstring: "Covers every quadratic-softmax-style mixer … Sigmoid attention passes its sigmoid+L1 normalizer" |
| Example preset | `configs/examples/es_arch_ca_adamw.yaml` (encoder MLM, `es_arch_kda_adamw.yaml` shape) | command row 11; Dense-family precedent |
| Full preset | **`configs/ca.yaml`** (decoder causal_lm at the paper's 101M geometry, adapted) | **user-approved**; joins smoke-tested presets (hard rule 9) |
| Training-free? | **No** — fully trainable | §2.9; no ⚠️ wording anywhere |
| Census bumps | enum **47 → 48**; README Dense (2→3) & "47 mixer values"→48; architecture §7.4 "47 enum values"→"48"; taxonomy mmd "(47 variants)"→48 & "Dense (3)"→"(4)"; `dense-map.mmd` +node; paper EN/ES "forty-seven/cuarenta y siete"→"forty-eight/cuarenta y ocho" (§01:14, §04:7,24, §10:16, §11:8, paper.tex:63 / ES counterparts); annex-summary-tables "47 sequence mixers"→48 & "Dense and convolution-augmented (4)"→"(5)" (ES: "Densos y aumentados con convolución (4)"→"(5)"); annex-15 census + "Twenty-nine"→"Thirty" parity-tested branches (EN/ES); capabilities.py docstrings; `test_schema_attention_layers.py` `len == 47`→`48` | actual repo census read on 2026-11-09 |
| Fixed numerics | `use_fixed_softmax` accepted & **ignored** (documented): CA has no softmax; its integral normalizer is intentionally outside the arXiv:2610.31642 coverage, like sigmoid's L1 (annex-16 §Scope precedent) | `numerics.module_softmax` only swaps softmax/log-softmax |

**Hard rules restated** (must hold): the schema is the contract (`additionalProperties:
false` everywhere; unknown keys warn `WARNING [schema]` and are dropped, never read);
`hidden_size % num_heads == 0`; `num_kv_heads | num_heads`; `len(layer_pattern) ==
num_layers`; inference never fails because of the cache (D5 — `cached_qk_attention` is
opt-in via slot kind, legacy path otherwise); BitNet on by default (BitLinear for the four
projections; the scalar gain stays a plain fp parameter, `kda` decay-param precedent);
papers are the theory source of truth (D11) — annex formulations must match
arXiv:2610.20586 and `paper-es` must translate them faithfully.

---

## 4. File-by-file plan

Anchors are line numbers read on 2026-11-09; re-verify at implementation time (drift).

| # | File | Planned change (concrete) |
|---|---|---|
| 1 | `src/model/attention/ca.py` | **New module** (~330 lines), docstring citing arXiv:2610.20586 with Eqs. (2)–(23) mapped to code; `from __future__ import annotations`; Google-style docstrings; CPU-safe. Class `ContinuumAttention(nn.Module)`, `__init__(self, config, pos_encoder=None)` mirroring `standard.py:62-82`: `hidden/num_heads/head_dim/scale = head_dim ** -0.5`; div-by-`num_heads` `ValueError`; `proj_cls = BitLinear if config.use_bitnet else nn.Linear`; `q/k/v/out_proj = proj_cls(H, H, bias=False)`; `self.ca_learned_gain = bool(getattr(config, "ca_learned_gain", True))`; `gain_init = float(getattr(config, "ca_gain_init", 1.0))` → learned: `self.log_gain = nn.Parameter(torch.full((num_heads,), math.log(gain_init)))`, fixed: plain float `self.gain_init` (no tensor ⇒ fixed variant param-identical to `standard`); `dropout`, `mode`, `pos_encoder`, `pe_type`, `use_pe = bool(getattr(config, "ca_use_pe", True))`; `attach_qk_norm(self, config, self.head_dim)`. **Forward** (S1/S2 shapes per §7.13, signature `(x, logical_layer_idx=None, pos_encoder=None, cache=None, cache_position=None)`): project → qk_norm → `slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None`; cached branch → `cached_qk_attention(slot, q, k, v, cache_position=cache_position, scale=self.scale, pe=pe, pe_type=self.pe_type, use_pe=self.use_pe, x_for_pe=x, logical_layer_idx=logical_layer_idx or 0, normalize=self._effective_weights, dropout=self.dropout, causal=True, rotate=True, use_fixed_softmax=False)` (sigmoid.py:132 pattern) → `out_proj`; dense branch → `apply_pe_to_qk` (standard.py:155) → `S = (q @ k.transpose(-2,-1)) * self.scale * γ_view` (γ per head, `(nH,1,1)`) → `apply_pe_to_scores` (score-bias PEs) → `record_qk_max_logits(self, S)` → causal triu mask when `mode == "decoder"` → `W = self._effective_weights(S)` → `dropout(W)` → `W @ v` → reshape → `out_proj`. **`_effective_weights(S)`**: visible-prefix mask handling (replace `-inf` by a finite sentinel — e.g. each row's own-position score — before forming `Δ`, to avoid `-inf − -inf = NaN`; zero masked weights at the end); row-max subtraction; `A = S[..., :-1]`, `B = S[..., 1:]`, `Δ = B − A`; `_moments(Δ)` → `J0, J1` via Eqs. (11)–(12) in fp32 with the Eq.-34 3-term Taylor fallback for `|Δ| < 1e-3` (and `J2` helper for tests); `M = exp(A − m)·J0`; `μ = J1/J0` with `μ←½` where `J0==0`; caps `½e^{S[...,0]−m}` and `½e^{S[...,own]−m}` (`own` = diagonal in decoder, `n−1` in encoder); accumulate `w` (left endpoint `M(1−μ)`, right `M·μ`); `Z = caps + ΣM`; `w /= Z`; zero masked entries. Docstring notes: `use_fixed_softmax` is accepted and ignored (no softmax in CA); encoder-mode is the repo's symmetric extension of the causal-only paper. |
| 2 | `src/model/attention/__init__.py` | `from .ca import ContinuumAttention` (after the `.standard` import, line 22) + `"ContinuumAttention"` in `__all__` (after `"StandardAttention"`, line 60). |
| 3 | `src/model/hybrid_layer.py` | Import `ContinuumAttention` (after the `standard` import, line 66) and `mixer_registry["ca"] = ContinuumAttention` after `"sigmoid": SigmoidAttention,` (line 190). **Not** added to `TRAINING_FREE_LAYERS` (line 123). |
| 4 | `src/model/config.py` | After the Fast Conformer block (~line 809), add:<br>`# ---- Continuum Attention (arXiv:2610.20586) ----`<br>`# Learn the per-head score gain γ = exp(ρ) (paper's 101M flagship:`<br>`# learned, initialized at γ=1). False keeps CA-P's fixed gain (zero`<br>`# extra parameters).`<br>`ca_learned_gain: bool = True`<br>`# Initial (learned) or fixed score gain γ = 1/τ. Paper: CA-P fixes`<br>`# γ = 1/0.30; the 101M model learns γ initialized at 1.0.`<br>`ca_gain_init: float = 1.0`<br>Validation in `__post_init__` (near the swa checks, line ~1134): `if self.ca_gain_init <= 0: raise ValueError(f"ca_gain_init must be > 0, got {self.ca_gain_init}")`. Plus `ca_use_pe: bool = True` next to `standard_use_pe`/`sigmoid_use_pe` (line ~597) with docstring lines beside lines 280–284. |
| 5 | `src/utils/config_flatten.py` | In `_ATTENTION_MIXER_RENAMES` (line 78) add `"ca": {"learned_gain": "ca_learned_gain", "gain_init": "ca_gain_init"},` (after the `"swa"` entry) and one docstring line in the header table (~line 42): `ca: {learned_gain, gain_init}`. |
| 6 | `src/schema/_model/_dims.yaml` | (a) `items.enum` (lines 336–383): add `- ca` right after `- sigmoid` (line 342) → 48 values. (b) EN table (line 219): new row after `sigmoid` (line 220): `` \| `ca` \| Continuum Attention — exact normalized integration of exp(score) over piecewise-linear key/value trajectories (paper: arxiv:2610.20586) \| ✅ \| ✅ \| ``. (c) ES table (line 278): new row after `sigmoid` (line 279): `` \| `ca` \| Atención Continua — integración normalizada exacta de exp(puntaje) sobre trayectorias clave/valor lineales por tramos (paper: arxiv:2610.20586) \| ✅ \| ✅ \| ``. Both tables keep the `test_layer_pattern_enum_matches_documentation` invariant (`\`ca\`` backticked in both). |
| 7 | `src/schema/_model/_attention_ca.yaml` | **New bilingual knob file** (swa template, `additionalProperties: false` at both levels). Full draft:<br><br>```yaml<br>type: object<br>additionalProperties: false<br>properties:<br>  ca:<br>    type: object<br>    additionalProperties: false<br>    title: "Continuum Attention"<br>    title_es: "Atención Continua"<br>    description: |-<br>      Hyperparameters for `ca` (Continuum Attention (paper: arxiv:2610.20586)).<br>      Adjacent keys and values define a piecewise-linear memory trajectory;<br>      each query retrieves by normalized integration of exp(score) over its<br>      visible prefix, computed exactly through the exponential moments J0/J1<br>      (no quadrature). With `learned_gain: true` every head learns a positive<br>      score gain γ = exp(ρ) initialized at `gain_init` (the paper's 101M<br>      flagship); with `learned_gain: false` the gain stays fixed at<br>      `gain_init` (CA-P uses γ = 1/0.30) and the mixer adds no parameters<br>      beyond the standard Q/K/V/O projections. Only applies when `ca`<br>      appears in `layer_pattern`.<br><br>      **References:** (paper: arxiv:2610.20586)<br>    description_es: |-<br>      Hiperparámetros para `ca` (Atención Continua (paper: arxiv:2610.20586)).<br>      Las claves y valores adyacentes definen una trayectoria de memoria<br>      lineal por tramos; cada consulta recupera mediante la integración<br>      normalizada de exp(puntaje) sobre su prefijo visible, calculada de<br>      forma exacta con los momentos exponenciales J0/J1 (sin cuadratura).<br>      Con `learned_gain: true` cada cabeza aprende una ganancia de puntaje<br>      positiva γ = exp(ρ) inicializada en `gain_init` (el modelo insignia de<br>      101M del paper); con `learned_gain: false` la ganancia queda fija en<br>      `gain_init` (CA-P usa γ = 1/0.30) y el mixer no añade parámetros más<br>      allá de las proyecciones Q/K/V/O estándar. Solo aplica cuando `ca`<br>      está en `layer_pattern`.<br><br>      **Referencias:** (paper: arxiv:2610.20586)<br>    properties:<br>      learned_gain:<br>        type: boolean<br>        title: "Learned Score Gain"<br>        title_es: "Ganancia de Puntaje Aprendida"<br>        description: |-<br>          Learn the per-head positive score gain γ = exp(ρ) of `ca`<br>          (paper: arxiv:2610.20586, §2.7/§3.1). The paper's 101M byte-level<br>          decoder learns all 128 head gains initialized at γ = 1; the<br>          small-model CA-H variant learns them initialized at γ = 1/0.30<br>          (set `gain_init`). `false` keeps the gain fixed at `gain_init`,<br>          which is the parameter-free CA-P configuration. Only applies when<br>          `ca` is in `layer_pattern`.<br><br>          **References:** (paper: arxiv:2610.20586)<br>        description_es: |-<br>          Aprende la ganancia de puntaje positiva por cabeza γ = exp(ρ) de<br>          `ca` (paper: arxiv:2610.20586, §2.7/§3.1). El decoder de bytes de<br>          101M del paper aprende las 128 ganancias inicializadas en γ = 1;<br>          la variante CA-H del modelo pequeño las aprende inicializadas en<br>          γ = 1/0.30 (ajusta `gain_init`). `false` mantiene la ganancia fija<br>          en `gain_init`, la configuración CA-P sin parámetros extra. Solo<br>          aplica cuando `ca` está en `layer_pattern`.<br><br>          **Referencias:** (paper: arxiv:2610.20586)<br>        default: true<br>        examples: [true, false]<br>      gain_init:<br>        type: number<br>        exclusiveMinimum: 0<br>        title: "Initial / Fixed Score Gain"<br>        title_es: "Ganancia de Puntaje Inicial / Fija"<br>        description: |-<br>          Initial (when learned) or fixed (when not) score gain γ = 1/τ of<br>          `ca` (paper: arxiv:2610.20586). The paper uses γ = 1 for the 101M<br>          model (effective τ = 1), γ = 1/0.30 ≈ 3.3333 for the fixed CA-P<br>          and the CA-H small-model initialization. Larger γ sharpens the<br>          relevance density (lower temperature τ). Must be positive. Only<br>          applies when `ca` is in `layer_pattern`.<br><br>          **References:** (paper: arxiv:2610.20586)<br>        description_es: |-<br>          Ganancia de puntaje inicial (si se aprende) o fija (si no) γ = 1/τ<br>          de `ca` (paper: arxiv:2610.20586). El paper usa γ = 1 para el<br>          modelo de 101M (τ efectiva = 1), γ = 1/0.30 ≈ 3.3333 para la CA-P<br>          fija y la inicialización CA-H del modelo pequeño. Un γ mayor<br>          afila la densidad de relevancia (menor temperatura τ). Debe ser<br>          positiva. Solo aplica cuando `ca` está en `layer_pattern`.<br><br>          **Referencias:** (paper: arxiv:2610.20586)<br>        default: 1.0<br>        examples: [1.0, 3.3333]<br>``` |
| 8 | `src/schema/_model.yaml` | Under `model.attention.properties` add after `retnet` (line 111): `ca: $ref: "_model/_attention_ca.yaml#/properties/ca"`. **Do NOT** append `ca` to the EN/ES "mixers without config knobs" description lists (lines 50–54/60–64) — those enumerate knob-less mixers and `ca` **has** a sub-object (documented deviation from the generic recipe; no test reads those lists, correctness wins). |
| 9 | `src/schema/_model/_positional_encoding.yaml` | In `use_pe.properties` add after `sigmoid` (line 397): `ca: { type: boolean, title: "ca", description: "Model-wide PE for `ca` layers (paper: arxiv:2610.20586).", default: true, description_es: "PE a nivel de modelo para capas `ca` (paper: arxiv:2610.20586)." }` |
| 10 | `src/model/cache/capabilities.py` | (a) In `CACHE_SPECS` dense section (after `"sigmoid"`, line ~128):<br>`"ca": CacheSpec(supported=True, slot_type="kv", notes="Continuum Attention (arXiv:2610.20586); class (i) — the effective weights are a closed-form (moment) function of the endpoint score row, so decode recomputes the segment integrals over the whole cached prefix each step; row-local like sigmoid's re-summed L1 normalizer, hence exact."),`<br>(b) `IMPLEMENTED_MIXERS`: add `"ca",` in the F1 dense block (after `"gated_softmax"`, line ~76). (c) Header docstring: line 6 "(48 total: 47 mixer classes + the ``mamba`` linear placeholder)" stays as-is numerically but becomes *correct* (48 enum entries = 47 mixer classes + mamba) — re-verify; line 108 "(47)" → "(48)". |
| 11 | `src/model/norm/qk.py` | Add `"ca"` to `QK_NORM_MIXERS` (line 46; CA exposes per-head q/k projections and records per-head max logits). `QK_CLIP_MIXERS` derives automatically (line 80). Extend `tests/test_qk_norm.py` mixer coverage only if it enumerates mixer names (grep at implementation time; current file enumerates classes, likely no edit). |
| 12 | `configs/examples/es_arch_ca_adamw.yaml` | **New example** — `es_arch_kda_adamw.yaml` verbatim with: `num_kv_heads: 1`, `layer_pattern: [ca]`, and an `attention:` block adding `ca: {learned_gain: true, gain_init: 1.0}` before `retnet: {heads: 8}`; `csv_log_path: training_metrics_es_arch_ca_adamw.yaml`'s sibling → `training_metrics_es_arch_ca_adamw.csv`. Full YAML = `configs/examples/es_arch_kda_adamw.yaml` lines 1–76 with those four edits. Must pass `load_training_config` (auto-globbed by `test_yaml_examples.py`). |
| 13 | `configs/ca.yaml` | **New full preset** (user-approved) — paper's 101M geometry adapted to the repo:<br><br>```yaml<br>model_class: frankenstein_text_decoder<br>model:<br>  dims:<br>    vocab_size: 32000<br>    hidden_size: 1024<br>    num_layers: 8<br>    num_loops: 1<br>    num_heads: 16<br>    num_kv_heads: 16<br>    dropout: 0.0<br>    mode: decoder<br>    layer_pattern:<br>    - ca<br>  norm:<br>    type: rms_norm<br>  attention:<br>    ca:<br>      learned_gain: true<br>      gain_init: 1.0<br>    retnet:<br>      heads: 16<br>  moe:<br>    enabled: false<br>  bitnet:<br>    enabled: false<br>  ffn:<br>    hidden_size: 4096        # paper: feedforward expansion factor 4<br>    activation: gelu<br>  positional_encoding: rope<br>training:<br>  task: causal_lm<br>  seed: 42<br>  batch_size: 8<br>  dataloader_workers: 2<br>  max_length: 2048          # paper context length<br>  gradient_accumulation_steps: 8<br>  optimizer:<br>    optimizer_class: adamw<br>    parameters:<br>      adamw-lr_other: 3.0e-4<br>      adamw-lr_attention: 3.0e-4<br>      adamw-betas_other: [0.9, 0.95]<br>      adamw-betas_attention: [0.9, 0.95]<br>      adamw-eps_other: 1.0e-8<br>      adamw-eps_attention: 1.0e-8<br>      adamw-wd_other: 0.1<br>      adamw-wd_attention: 0.01<br>  scheduler_total_steps: 50000<br>  scheduler_warmup_ratio: 0.04   # 2000-step warmup of the paper<br>  scheduler_type: cosine<br>  grad_clip_max_norm: 1.0<br>  max_nan_retries: 4<br>  gradient_log_interval: 10<br>  csv_log_path: training_metrics_ca.csv<br>  telemetry_log_interval: 5<br>```<br>Untied input/output embeddings = paper ("independent input/output byte embeddings"). Schema-validate against `_training.yaml` keys at implementation time (all keys above appear in existing presets). |
| 14 | `full_tests/run_e2e.py` | Append to `ATTENTIONS` (line 224, after `"fastconformer"` line 249):<br>`# Continuum Attention (arXiv:2610.20586): exact integration of an`<br>`# exponential relevance density over piecewise-linear key/value`<br>`# trajectories; trains natively in encoder and decoder modes; exact`<br>`# KV-cache incremental decoding.`<br>`"ca",`<br>Auto-generates `attn_single_ca` (1-layer solo) and every `attn_<a>__ca` / `attn_ca__<b>` pair via `make_attention_pair_configs`/`make_single_attention_configs` (lines 768–790). No builder edits. |
| 15 | `tests/test_schema_attention_layers.py` | (a) `"ca"` in the expected enum set (line ~67, after `"fastconformer"`). (b) `test_cache_specs_match_layer_pattern_enum`: `assertEqual(len(enum_values), 47)` → `48` (line 160). (c) `expected_per_mixer["ca"] = {"learned_gain", "gain_init"}` (line ~133). |
| 16 | `tests/test_schema_consistency.py` | Two `_DEFAULT_PAIRS` entries (after the swa pair, line ~200): `"properties.model.properties.attention.properties.ca.properties.learned_gain": "ca_learned_gain"` and `"...ca.properties.gain_init": "ca_gain_init"`. |
| 17 | `tests/test_attention_modules.py` | New `ContinuumAttentionTests(unittest.TestCase)` (after `SigmoidAttentionTests`, line ~127), config built like `StandardAttentionTests` (BitNet off + on variant): `test_encoder_output_shape`, `test_decoder_output_shape`, `test_decoder_is_causal` (perturb a future token ⇒ prefix outputs unchanged; paper §3.6 check), `test_equal_scores_reduce_to_uniform` (constant S ⇒ row weights uniform over the visible prefix), `test_first_token_weight_is_identity` (`o_0 = v_0`: row-0 weight is 1 on position 0), `test_moments_match_quadrature` (`J0/J1/J2` closed forms vs `torch.trapz(exp(Δu))` at Δ ∈ {−8, −0.9, −1e-4, 0, 1e-4, 0.9, 8}), `test_learned_gain_parameter_count` (learned ⇒ `4H² + nH`, grad reaches `log_gain`), `test_fixed_gain_parameter_parity` (fixed ⇒ exactly the `standard` count), `test_gradient_flows`, `test_with_bitnet`, `test_no_nans` (n=64), `test_use_pe_flag` (`ca_use_pe=False` ⇒ PE module unused). |
| 18 | `tests/test_bitnet_coverage.py` | Add `"ca"` to the `layer_pattern` list (~line 49, after `"sigmoid"`). |
| 19 | `tests/test_kv_cache_mixers.py` | **No edit** — `MIXERS = sorted(IMPLEMENTED_MIXERS …)` (line 27) auto-parametrizes `ca` over `PES = (rope, nope, alibi)` once row 10 lands. Verify the new subTest appears in the run output. |
| 20 | `docs/paper/figures/mixers/ca.mmd` | **New diagram** (house style of `sigmoid.mmd`):<br><br>```<br>%%{init: {"theme":"neutral","flowchart":{"curve":"basis","nodeSpacing":18,"rankSpacing":26,"useMaxWidth":false}}}%%<br>flowchart TD<br>  x["x<br/>[B,n,H]"]<br>  q["Q = Wq·x<br/>[B,nH,n,hd]"]<br>  k["K = Wk·x<br/>[B,nH,n,hd]"]<br>  v["V = Wv·x<br/>[B,nH,n,hd]"]<br>  s["a,b = γ·QKᵀ/√hd at endpoints<br/>[B,nH,n,n]"]<br>  j["M = eᵃ·J₀(Δ), μ = J₁/J₀<br/>[B,nH,n,n−1]"]<br>  w["w = ½e^{s(0)} + ½e^{s(i)} + Σ segments<br/>Σw = 1, w ≥ 0"]<br>  o["O = Wo·(w·V)<br/>[B,n,H]"]<br><br>  x -- "Wq" --> q<br>  x -- "Wk" --> k<br>  x -- "Wv" --> v<br>  q -- "γQKᵀ/√hd" --> s<br>  k -- "γQKᵀ/√hd" --> s<br>  s -- "Δ = b−a, moments" --> j<br>  j -- "caps + normalize (Z)" --> w<br>  v -- "w·V" --> o<br>  w -- "w·V" --> o<br>``` |
| 21 | `docs/paper/figures/mixer-taxonomy.mmd` | Line 3: `"Sequence Mixer Registry (47 variants)"` → `(48 variants)`. Line 5: `d["Dense (3)"] --> d1["standard"] --> d2["sigmoid"] --> d3["gqa"]` → `d["Dense (4)"] --> d1["standard"] --> d2["sigmoid"] --> d2b["ca"] --> d3["gqa"]` (node id `d2b` keeps the chain unique; any consistent id works). |
| 22 | `docs/paper/figures/dense-map.mmd` | Add node `ca["Continuum<br/>∫ exp(γ·qᵀk(t)/√d)·v(t) dt / Z"]` and extend line 7 chain: `standard ~~~ sigmoid ~~~ gqa ~~~ ca`. |
| 23 | `docs/bibliography/attention_types.bib` | Entry from §2.12, keyed `khadangi_continuum_2026` (format matches `rekesh_fastconformer_2023`, lines 107–116). |
| 24 | `docs/paper/appendices/annex-2-recurrent-attention.tex` | (a) Rename subsection (line 6) to `Dense Attention Baselines: Standard, Sigmoid, and Continuum` (keep the label-free flow). (b) New `\subsubsection{Continuum Attention}` after the GQA block (line 86), figure `\fitfigure{\includegraphics{figures/mixers/ca.pdf}}` + `\label{fig:mixer-ca}`, Mathematical Foundation with Eqs. (2)–(8) + (11)–(23) (LaTeX from §2.1), algorithm float (parallel form of §2.4), Empirical Performance paragraph (§2.12 numbers: 42% data-efficiency, FLORES 1.8915 BPB / 5.8%, MC scores) and a strength/limitation itemize:<br>`\begin{itemize}[leftmargin=*]`<br>`\item \textbf{Training complexity}: $\mathcal{O}(n^2 d)$ — identical to dense softmax; closed-form moments replace the softmax.<br>`\item \textbf{Inference complexity}: $\mathcal{O}(n)$ per token with a KV cache (weights recomputed over the prefix each step, row-local and exact).<br>`\item \textbf{Strengths}: Exact, quadrature-free continuous retrieval; normalized non-negative effective weights; zero extra parameters in the fixed-gain form; strong multilingual and zero-shot gains (FLORES −5.8% BPB; HellaSwag/ARC-C best).<br>`\item \textbf{Weaknesses}: Quadratic attention arithmetic persists; weaker on source code (Stack-smol 3.3451 BPB, worst of the five baselines); reduced robustness to byte deletion and "antspeak" perturbations.`<br>`\end{itemize}`<br>`\citep{khadangi_continuum_2026}` throughout. (c) Extend the dispatch pseudocode (line 294): add `\ElsIf{$m=\texttt{ca}$} \State $H \gets \text{continuum-attention}(H)$` next to the `standard`/`sigmoid` branches. (d) Note the encoder-mode symmetric extension in one sentence. |
| 25 | `docs/paper/sections/04-architecture-taxonomy.tex` | Line 7: "forty-seven" → "forty-eight". Line 9: `\item \textbf{Dense baselines} (3): …` → `(4): standard softmax attention, sigmoid attention, Continuum Attention, and Grouped-Query Attention (GQA)…`. Line 24 caption: "forty-seven" → "forty-eight". Dense-and-Recurrent paragraph (line 29): append "Continuum Attention \citep{khadangi_continuum_2026} replaces the softmax with exact integration of an exponential relevance density over piecewise-linear key/value trajectories \citep{khadangi_continuum_2026}." (one sentence, cite once). |
| 26 | `docs/paper/appendices/annex-summary-tables.tex` | Line 4: "47 sequence mixers" → "48 sequence mixers". Line 28: `\textbf{Dense and convolution-augmented (4)}` → `(5)`. New row after `gqa` (line 32):<br>`ca & Dense & $\mathcal{O}(n^2 d)$ & $\mathcal{O}(n)$ cache & Exact integration over piecewise-linear key/value trajectories; half-token caps; nonnegative normalized effective weights & \citep{khadangi_continuum_2026} \\` |
| 27 | `docs/paper/appendices/annex-schema.tex` | New row after `sigmoid` (line 245): `ca & Continuum Attention --- exact continuous integration over piecewise-linear memory trajectories & \checkmark & \checkmark \\` |
| 28 | `docs/paper/appendices/annex-15-kv-cache.tex` | Line 20: "forty-seven dispatchable layer types" → "forty-eight"; "Twenty-nine have parity-tested incremental branches" → "Thirty". New item after the sigmoid item (line 160): `\item \textbf{Continuum attention.} The normalizer is replaced by the exact segment integration: the effective weights are a closed-form function (exponential moments of the endpoint-score differences) of the full score row, so decode recomputes them over the whole cached prefix each step --- row-local, hence exact, in the sigmoid re-normalization pattern \citep{khadangi_continuum_2026}.` |
| 29 | `docs/paper/appendices/annex-12-positional-encodings.tex` | Add `ca` to the `\textbf{True}` attention-mixer list (line 211–213). |
| 30 | `docs/paper/appendices/annex-16-fixed-numerics.tex` | Optional one-line Scope addition (line 191): mention CA's integral normalizer is intentionally unchanged alongside sigmoid's L1 (no softmax in CA). No table change (CA has no softmax surface). |
| 31 | `docs/paper-es/appendices/annex-2-recurrent-attention.tex` | Faithful Spanish translation of row 24: `\subsubsection{Atención Continua}` after the GQA block (~line 86); figures from `../paper/figures/` per the ES convention (`figures/mixers/ca.pdf` — the ES paper.tex path resolves into the mirrored figures; verify the ES includegraphics root at implementation time); subsection title `Atención Densa de Referencia: Estándar, Sigmoide y Continua`; extend the dispatch pseudocode (~line 294 ES). |
| 32 | `docs/paper-es/sections/04-taxonomia-arquitectura.tex` | Lines 7 & 24: "cuarenta y siete" → "cuarenta y ocho". Line 9: `Líneas base densas (3)` → `(4)` + "Atención Continua" added to the list. Dense-family paragraph (~line 29): one translated CA sentence with `\citep{khadangi_continuum_2026}`. |
| 33 | `docs/paper-es/appendices/annex-summary-tables.tex` | Line 4: "47 mezcladores" → "48 mezcladores". Line 28: `Densos y aumentados con convolución (4)` → `(5)`. Row after `gqa`: `ca & Denso & $\mathcal{O}(n^2 d)$ & $\mathcal{O}(n)$ caché & Integración exacta sobre trayectorias clave/valor lineales por tramos; extensiones de medio token; pesos efectivos normalizados no negativos & \citep{khadangi_continuum_2026} \\` |
| 34 | `docs/paper-es/appendices/annex-schema.tex` | Row after `sigmoid` (line 245): `ca & Atención Continua --- integración continua exacta sobre trayectorias de memoria lineales por tramos & \checkmark & \checkmark \\` |
| 35 | `docs/paper-es/appendices/annex-15-cache-kv.tex` | Line 20: "los cuarenta y siete tipos" → "los cuarenta y ocho"; "Veintinueve" → "Treinta". New item after the sigmoid item (~line 161): translated CA paragraph. |
| 36 | `docs/paper-es/appendices/annex-12-codificaciones-posicionales.tex` | Add `ca` to the `use_pe=True` list (ES counterpart of row 29, §"Banderas `use_pe` Por Mezclador"). |
| 37 | `docs/paper-es/appendices/annex-16-numericas-corregidas.tex` | Optional translated scope note (row 30). |
| 38 | `docs/paper-es/paper.tex` + `01-introduccion.tex:14`, `10-discusion.tex:16`, `11-conclusion.tex:8` | "cuarenta y siete mezcladores" → "cuarenta y ocho" (grep all occurrences; paper-es.tex intro paragraph mirrors paper.tex:63). |
| 39 | `README.md` | Mixer Categories (line 117): `\| **Dense** (2) \| \`standard\`, \`sigmoid\` \| Full quadratic attention variants \|` → `(3)` + `, \`ca\`` (mention: "standard softmax, sigmoid gating, exact continuum integration"). Line 130: "The schema accepts 47 mixer values." → "48 mixer values." |
| 40 | `docs/architecture.md` | §7.4 line 351: "47 enum values in 10 families" → "48 enum values in 10 families". Table (line 356): add `\| \`ca\` \| Dense \| \`ca.*\` \| yes \|` after the `sigmoid` row (line 357). New prose paragraph after the `fastconformer` paragraph (~line 448): "`ca` implements Continuum Attention from arXiv:2610.20586: adjacent keys/values define a piecewise-linear memory trajectory and each query retrieves by exact normalized integration of an exponential relevance density over its visible prefix (half-token endpoint caps included). The closed-form exponential moments J0/J1 replace the softmax (J2 only for the paper's analytic backward); with `ca.learned_gain: true` each head learns a gain γ = exp(ρ) initialized at `ca.gain_init` (+nH parameters per layer); the fixed-gain form adds no parameters. The KV cache is a plain `KVSlot` — the effective weights are recomputed row-locally over the cached prefix each decode step (sigmoid precedent)." §7.13 implemented list (line 658): add `ca` (e.g. after `gated_softmax`). |
| 41 | `configs/README.md` | Line 355: add `ca` to the per-mixer sub-object enumeration. New section `### Continuum Attention (model.attention.ca)` (before `### Latent / KV-compression family`, line 396): documents `learned_gain` (default true, 101M flagship; false = parameter-free CA-P), `gain_init` (default 1.0; CA-P uses 3.3333 = 1/0.30), `positional_encoding_parameters.use_pe.ca` (default true), pointers to `configs/examples/es_arch_ca_adamw.yaml` and `configs/ca.yaml`. |
| 42 | `docs/source/api/model/attention/ca.rst` + `index.rst` | New rst (standard.rst template): title `Continuum Attention`, `.. automodule:: src.model.attention.ca` with `:members:` `:undoc-members:` `:show-inheritance:`. `index.rst` toctree: add `ca` after `sigmoid` (line 18). |
| 43 | Mirror `erickfmm.github.io/frankenstein-transformer/ft-diagram.js` | (site-only) `CATS.ca = { c: 'attn', l: 'Continuum Attn' }` (after `sigmoid`, line 6); `SUBSTRUCT.ca = [ { n: 'x [B,n,H]', c: 'attn', id: 'x' }, { n: 'Q = Wq·x [B,n,H]', c: 'attn', e: 'Wq', from: ['x'] }, { n: 'K = Wk·x [B,n,H]', c: 'attn', e: 'Wk', from: ['x'] }, { n: 'V = Wv·x [B,n,H]', c: 'attn', e: 'Wv', from: ['x'] }, { n: 'a,b = γQKᵀ/√d endpoints [B,nh,n,n]', c: 'attn', e: 'γQKᵀ', from: ['Q = Wq·x [B,n,H]', 'K = Wk·x [B,n,H]'] }, { n: 'M = eᵃJ₀(Δ), μ = J₁/J₀ [B,nh,n,n]', c: 'attn', e: 'moments', from: ['a,b = γQKᵀ/√d endpoints [B,nh,n,n]'] }, { n: 'w = caps + segments, Σw=1', c: 'attn', e: 'normalize', from: ['M = eᵃJ₀(Δ), μ = J₁/J₀ [B,nh,n,n]'] }, { n: 'O = Wo·(w·V) [B,n,H]', c: 'attn', e: 'Wo', from: ['w = caps + segments, Σw=1', 'V = Wv·x [B,n,H]'] } ]` (after `standard`, line 104ff). No `STYLES`/`STYLES_PRINT` change — CA reuses the `attn` class. |
| 44 | Mirror `ft-param-estimator.js` | (site-only) In `mixerParams` (line 77), in the Dense section after `case 'sigmoid':` (line 87):<br>`// ---- Continuum Attention (arXiv:2610.20586) ----`<br>`// Q,K,V,O (bias-free) = 4H²; the learned per-head log score gain`<br>`// γ = exp(ρ) adds nH when attention.ca.learned_gain is true (default).`<br>`case 'ca': {`<br>`  var caLearn = toBool(pick(m, 'attention.ca.learned_gain', 'ca_learned_gain', true), true);`<br>`  return 4 * H * H + (caLearn ? nH : 0);`<br>`}` |
| 45 | Mirror `index.html` | Verify-only (no expected edit): the Builder renders the mixer enum from the mirrored schema, so `ca` appears automatically after the sync. Grep for hardcoded mixer arrays (e.g. `mixers.includes('standard')`, line 2679) and extend only lists that enumerate mixers exhaustively; the `loadExample('standard')` buttons are curated presets, not mixer catalogues. |

Auto-covered, **no edit needed** (verify only): `tests/test_all_mixers_smoke.py`
(parametrizes over the schema enum → `ca` covered), `tests/test_yaml_examples.py`
(auto-globs both config dirs), `test_schema_consistency.test_all_presets_validate`
(globs presets), `full_tests/param_count_check.py` (globs example configs —
`param_truth.json` diff is intentional), Sphinx `automodule` (reflects the module),
`docs/paper/appendices/annex-16` coverage table (CA has no softmax surface).

---

## 5. Ordered execution checklist

Implementation-time commands (do **not** run in plan mode). CPU-first: every command uses
`uv run --extra cpu --extra train …`.

**Phase A — implementation (rows 1–13)**

- [ ] `src/model/attention/ca.py` (row 1)
- [ ] exports (row 2), registry (row 3), config fields + validation (row 4), renames (row 5)
- [ ] schema: `_dims.yaml` enum + both tables (row 6), `_attention_ca.yaml` (row 7), `_model.yaml` `$ref` (row 8), `use_pe.ca` (row 9)
- [ ] cache: `CACHE_SPECS["ca"]` + `IMPLEMENTED_MIXERS` + docstring counts (row 10); qk-norm set (row 11)
- [ ] `configs/examples/es_arch_ca_adamw.yaml` (row 12); `configs/ca.yaml` (row 13)

**Phase B — tests (rows 14–19)**

- [ ] `tests/test_schema_attention_layers.py`, `tests/test_schema_consistency.py`
- [ ] `tests/test_attention_modules.py` (`ContinuumAttentionTests`)
- [ ] `tests/test_bitnet_coverage.py`; verify `tests/test_kv_cache_mixers.py` picks up `ca`
- [ ] `full_tests/run_e2e.py` `ATTENTIONS` (row 14)

**Phase C — targeted gates**

```bash
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_attention_refactor.py \
  tests/test_attention_modules.py tests/test_bitnet_coverage.py \
  tests/test_kv_cache_mixers.py tests/test_all_mixers_smoke.py \
  -v --tb=short -p no:warnings
```

Expected: all green; `test_cache_specs_match_layer_pattern_enum` now asserts 48;
`test_schema_defaults_match_dataclass` passes with the two new `_DEFAULT_PAIRS`;
`test_mixer_forward_backward_at_harness_dims[ca]` passes; KV parity subTests include
`ca × {rope, nope, alibi}`.

**Phase C2 — full suite**

```bash
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

**Phase D — e2e smoke, incl. targeted real-data training**

```bash
# 1. enumerate the new mixer's combos (build-only, no training)
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --list-combos | grep -F ca
# expected: attn_single_ca + every attn_<a>__ca / attn_ca__<b> pair (47 partners)

# 2. TARGETED REAL-DATA SMOKE — ONLY the new mixer: actual training on the real
#    Borges corpus. --realonall is the DEFAULT; the first invocation downloads
#    the tokenizer/corpus from the HF Hub (adleme94/borges + BETO).
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only ca

# 3. minimal real-data run if the pair sweep is too heavy: the solo combo only
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only attn_single_ca

# 4. offline synthetic fallback (no HF download), same selection
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only ca --no-realonall

# sizing knobs when needed: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800
# GPU smoke (real training box): swap the extra for --extra cu118 --extra train.
```

**Expected evidence**: every selected combo reports `PASS` with a finite training loss
over real corpus windows; the run directory keeps the generated `config.yaml` + metrics. A
NaN/Inf loss, a schema rejection or a timeout is a **blocker**, not a warning. If the HF
Hub is unreachable, record the `--no-realonall` PASS and re-run the real-data pass later.

**Phase E — figures (rows 20–22)**

```bash
( cd docs/paper/figures && ./build-figures.sh both )
```

Precondition: Chromium is available (`/usr/bin/chromium`, verified 2026-11-09) or
`PUPPETEER_EXECUTABLE`. Renders `mixers/ca.pdf` + `mixers/ca.svg`, `mixer-taxonomy.*`,
`dense-map.*`.

**Phase F — param ground truth + JS estimator (after the examples/presets land)**

```bash
uv run --extra cpu --extra train python full_tests/param_count_check.py
# expected: "... configs counted, 0 failures"; param_truth.json diff is intentional

node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01
# expected: exit 0, ALL PASS (<=1% drift on es_arch_ca_adamw.yaml and ca.yaml)
```

**Phase G — docs EN/ES (rows 23–41)**

- [ ] bibliography (row 23)
- [ ] annex-2 subsubsection + dispatch pseudocode (row 24); §04 census (row 25)
- [ ] annex-summary-tables (row 26); annex-schema (row 27); annex-15 (row 28); annex-12 (row 29); annex-16 note (row 30)
- [ ] all ES counterparts (rows 31–38)
- [ ] README.md (row 39); docs/architecture.md (row 40); configs/README.md (row 41)

**Phase H — Sphinx (row 42)**

```bash
bash docs/build_docs.sh
# expected: builds to docs/_build/html without warnings about missing modules;
# api/model/attention/ca.html generated
```

**Phase I — PDF rebuild (only at implementation time)**

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

`pdflatex` and `bibtex` are installed (`/usr/bin`, verified 2026-11-09). Post-build
verification: no `Fatal error`, no `Citation .* undefined` (the new
`\citep{khadangi_continuum_2026}` must resolve via `paper.bbl`), `paper.pdf` /
`paper-es.pdf` newer than the edited `.tex`/`.bbl`, and `git status --short docs/paper
docs/paper-es` shows only intended `.tex`/`.bbl`/`.pdf`/figure files. Fallbacks: if TeX
were unavailable, leave sources updated and note the rebuild must happen on a TeX machine
before the next release (the website sync copies prebuilt PDFs only); if no Chrome were
available for mermaid, commit the `.mmd` source and render before the paper/website sync.

---

## 6. Website sync (planned, not executed)

Follow `.opencode/command/sync-website.md`:

1. Bootstrap/refresh the mirror (`git clone` if `WEBSITE_REPO=MISSING`), then copy per the
   file map: `src/schema.yaml` + `src/schema/**` (rows 6–9), `configs/ca.yaml` +
   `configs/README.md` + `configs/examples/**` (rows 12–13, 41), `docs/paper/**` +
   `docs/paper-es/**` (tex, bbl, pdf, figures incl. `mixers/ca.{mmd,pdf,svg}` and the
   rebuilt `mixer-taxonomy`/`dense-map`), `docs/bibliography/**` (row 23),
   `docs/architecture.md` (row 40).
2. **Site-only edits** (mirror is the source of truth for these):
   - `ft-diagram.js`: `CATS["ca"]` + `SUBSTRUCT["ca"]` template (row 43); no new `STYLES`
     class (Dense family exists).
   - `ft-param-estimator.js`: `mixerParams()` `case 'ca'` (row 44) — bias- and
     knob-aware: `4H² + (learned_gain ? nH : 0)`.
   - `index.html`: grep `standard`/`sigmoid`; the mixer catalogue is schema-driven —
     expected **no** edit; extend only an exhaustive hardcoded list if found (row 45).
3. Paper-tab HTML rebuild: `flatten_tex.py --web` + `pandoc` + `postprocess_html.py`
   (en + es) and top-level `paper/en.pdf` / `paper/es.pdf` copies (sync doc §3).
4. Validation Gate: `param_count_check.py` → `0 failures` (regenerates
   `full_tests/param_truth.json` — intentional diff); estimator `--tol 0.01` → `ALL
   PASS`; JSON round-trip script; CLI parse script; state-script probes (`schema-refs`,
   `example-yaml`, `cli-flags`, `paper-en-copy`, `paper-es-copy`, …) all PASS.
5. The mirror stays uncommitted until the commit phase.

---

## 7. Commit plan (planned, not executed)

Follow `/make-a-commit` conventions (English, conventional tag, structured body; never
stage `*.aux *.log *.out *.toc *.blg`, caches, or mermaid temp files; `param_truth.json`
changes are intentional).

1. **Main repo** — `feat(attention): add Continuum Attention mixer (arXiv:2610.20586)`

   Body: module + registry + QK-norm membership; schema (enum + bilingual tables + knob
   block + `$ref` + use_pe + `_DEFAULT_PAIRS`); cache decision 1 (`KVSlot`,
   `cached_qk_attention` CA normalizer, `IMPLEMENTED_MIXERS`); example + full preset;
   tests (schema, family, bitnet, KV parity auto-param); e2e `ATTENTIONS` + real-data
   smoke result; figures (mmd + rendered pdf/svg); docs (annexes EN/ES incl. summary
   tables, schema annex, annex-12/15/16, section 04, README/architecture/configs README,
   Sphinx rst); rebuilt PDFs.

   Stage: `src/model/attention/ca.py`, `src/model/attention/__init__.py`,
   `src/model/hybrid_layer.py`, `src/model/config.py`, `src/utils/config_flatten.py`,
   `src/schema/_model/_dims.yaml`, `src/schema/_model/_attention_ca.yaml`,
   `src/schema/_model.yaml`, `src/schema/_model/_positional_encoding.yaml`,
   `src/model/cache/capabilities.py`, `src/model/norm/qk.py`,
   `configs/examples/es_arch_ca_adamw.yaml`, `configs/ca.yaml`, `full_tests/run_e2e.py`,
   `full_tests/param_truth.json`, `tests/test_schema_attention_layers.py`,
   `tests/test_schema_consistency.py`, `tests/test_attention_modules.py`,
   `tests/test_bitnet_coverage.py`, `docs/paper/figures/mixers/ca.{mmd,pdf,svg}`,
   `docs/paper/figures/{mixer-taxonomy,dense-map}.{mmd,pdf,svg}`,
   `docs/bibliography/attention_types.bib`,
   `docs/paper/{appendices,sections}/*.tex` (edited ones), `docs/paper/paper.{tex,bbl,pdf}`,
   `docs/paper-es/{appendices,sections}/*.tex` (edited ones),
   `docs/paper-es/paper-es.{tex,bbl,pdf}`, `README.md`, `docs/architecture.md`,
   `configs/README.md`, `docs/source/api/model/attention/{ca.rst,index.rst}`.

   Then `git push origin HEAD:main`.

2. **Mirror** — `git -C erickfmm.github.io …` commit
   `docs(site): sync Continuum Attention mixer (arXiv:2610.20586)`; stage the synced
   schema/configs/paper/figures/bibliography/spec files plus the site-only
   `ft-diagram.js`, `ft-param-estimator.js`, `paper/paper-{en,es}.html`,
   `paper/{en,es}.pdf`, copied SVGs; then
   `git -C erickfmm.github.io push origin HEAD:main`.

---

## 8. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Targeted pytest families | `uv run --extra cpu --extra train python -m pytest tests/test_schema_attention_layers.py tests/test_schema_consistency.py tests/test_yaml_examples.py tests/test_attention_refactor.py tests/test_attention_modules.py tests/test_bitnet_coverage.py tests/test_kv_cache_mixers.py tests/test_all_mixers_smoke.py -v --tb=short -p no:warnings` | all pass; enum count 48; `ca` subTests visible |
| Full pytest | `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` | 0 failures |
| e2e combo enumeration | `uv run --extra cpu --extra train python full_tests/run_e2e.py --category attn --list-combos \| grep -F ca` | `attn_single_ca` + all 47 `attn_<a>__ca` / `attn_ca__<b>` pairs |
| e2e targeted **real-data** training | `uv run --extra cpu --extra train python full_tests/run_e2e.py --category attn --only ca` | every selected combo PASS with finite loss on the real Borges corpus |
| e2e solo-only real-data | `… --category attn --only attn_single_ca` | PASS, finite loss |
| Offline fallback | `… --only ca --no-realonall` | PASS on synthetic data (use only if HF Hub unreachable) |
| Figure render | `( cd docs/paper/figures && ./build-figures.sh both )` | `mixers/ca.{pdf,svg}`, taxonomy + dense-map rebuilt |
| Param count | `uv run --extra cpu --extra train python full_tests/param_count_check.py` | `… configs counted, 0 failures` |
| JS estimator | `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` | exit 0, `ALL PASS` (≤1%) |
| Sphinx | `bash docs/build_docs.sh` | clean build; `api/model/attention/ca.html` present |
| PDF checks | Phase I + `grep -E "Fatal error|Citation .* undefined" <logs>` | no fatal, no undefined citations; PDFs newer than sources |
| Mirror probes | `bash .opencode/command/sync-website-state.sh` | all `[probes]` PASS; drift only in expected new files |
| KV parity | `pytest tests/test_kv_cache_mixers.py -v -k ca` | `ca × {rope, nope, alibi}` parity PASS (decision 1) |

---

## 9. Risks, assumptions and open questions

**Assumptions to verify at implementation time**

* Line numbers above were read on 2026-11-09 (HEAD `f5f922e`); re-anchor every edit by
  content, not line.
* `cached_qk_attention`'s `normalize` callback receives the **causal-masked** score
  tensor: for multi-token prefill chunks masked entries are `-inf`. The CA weight builder
  must sentinel-mask before forming `Δ` (avoid `-inf − -inf = NaN`) and zero masked
  weights afterwards. Unit-test chunked prefill parity explicitly (the KV parity suite
  runs prefill+decode, which covers it — but add a direct chunk test in
  `ContinuumAttentionTests`).
* Moment numerics: use fp32 internally (`S` in fp32; `.to(torch.float32)` for the moment
  math, cast back to `x.dtype`), Taylor fallback for `|Δ| < 1e-3` (Eq. 34, 3 terms:
  `J0 ≈ 1 + Δ/2 + Δ²/6`, `J1 ≈ 1/2 + Δ/3 + Δ²/8`).
* `ca.yaml` scheduler keys (`scheduler_warmup_ratio`, `scheduler_type: cosine`,
  `grad_clip_max_norm: 1.0`) exist in the schema — they appear in other presets; verify
  against `_training.yaml` before committing (the strict preset gate is the backstop).
* The paper's cosine decays to 3e-5; if the repo scheduler lacks a min-lr key, keep the
  standard cosine (document the deviation in the preset comment).
* The ES annex figure path (`figures/mixers/ca.pdf`) must resolve under the `paper-es`
  build — mirror how `annex-2` ES references `figures/mixers/standard.pdf` (same shared
  `figures/` tree).
* `capabilities.py` header docstring currently says "(48 total: 47 mixer classes + the
  mamba linear placeholder)" while holding 47 entries — with `ca` the numbers become
  exactly correct; do not "fix" them backwards.

**Risks**

* **Numerical**: catastrophic cancellation in `J0/J1` near `Δ = 0` → Taylor fallback;
  repeated identical keys give `Δ = 0` exactly (covered by the moments unit test).
  Row-max subtraction keeps `exp` finite over long prefixes (paper §2.6).
* **Masking NaN**: see the `-inf` sentinel note above — the single most likely bug class.
* **Encoder-mode extension** is repo-defined (paper is causal-only): documented deviation
  in module docstring + annexes; the e2e encoder sweep will exercise it broadly.
* **`ca` name length**: extremely short registry key — grep for `"ca"`-shaped collisions
  before landing (none found in the registry or `__all__`; `cca` is distinct).
* **BitNet**: the scalar `log_gain` stays a plain `nn.Parameter` (not BitLinear-quantized),
  matching `kda`'s decay parameters; `test_bitnet_coverage` asserts the projections are
  BitLinear.
* **Estimator coupling**: `ft-param-estimator.js` must read `attention.ca.learned_gain`
  with the same default (`true`) as the schema, or `ca.yaml`/`es_arch_ca_adamw.yaml` will
  exceed the 1% tolerance.
* **HF Hub reachability** for the real-data smoke → `--no-realonall` fallback, re-run later.

**Open questions (none blocking; defaults chosen)**

* `ca_learned_gain` default `true` (flagship config) was **user-approved**; CA-P is
  expressible as `{learned_gain: false, gain_init: 3.3333}`.
* The paper's analytic backward (Eqs. 25–33) and its Metal/CUDA kernels are **not**
  implemented (autograd over the closed-form forward is exact); noted in the annex.
* Chunked/tiled integration without materializing the attention matrix (paper §2.6) is out
  of scope — the repo materializes like every other dense mixer.

---

## 10. Out of scope

No code, schema, preset, test, figure, doc, PDF, mirror or git change was made by this
planning command. This document is the complete plan; every action above is deferred to an
explicitly requested implementation session. No destructive or out-of-scope step was
implied by the paper (no renames, no family changes, no unrelated drift fixes).
