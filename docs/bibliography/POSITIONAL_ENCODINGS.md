# Positional Encoding Bibliography

This document collects the bibliographic references for the 12 positional
encodings supported by the model-wide `positional_encoding` enum. The
BibTeX entries live in `docs/bibliography/other.bib`. Full mathematical
formulations are in
[`docs/paper/appendices/annex-12-positional-encodings.tex`](../paper/appendices/annex-12-positional-encodings.tex).

## Summary Table

| Encoding | BibTeX key | arXiv | Year | Authors |
|---|---|---|---|---|
| `rope` | `su_rope_2024` | 2104.09864 | 2024 | Su et al. (RoFormer) |
| `hope` | `dai_hope_2025` | 2509.05218 | 2025 | Dai et al. |
| `nope` | `kazemnejad_nope_2023` | 2305.19466 | 2023 | Kazemnejad et al. |
| `alibi` | `press_alibi_2022` | 2108.12409 | 2022 | Press, Smith, Lewis |
| `bam` | `bianchessi_bam_2025` | 2505.22842 | 2025 | Bianchessi et al. |
| `pape` / `pape_efficient` / `pape_ri` | `ohrstrom_pape_2026` | 2602.01418 | 2026 | Øhrstrøm et al. |
| `sinusoidal_absolute` / `sinusoidal_rotary` / `learned_absolute` | `vaswani_attention_2017` | 1706.03762 | 2017 | Vaswani et al. (Transformer) |
| `none` | — | — | — | — |

## Key Papers

### RoPE — Rotary Position Embedding
**Su, Jianlin et al.** "RoFormer: Enhanced Transformer with Rotary
Position Embedding." arXiv:2104.09864 (2024).
[arXiv](https://arxiv.org/abs/2104.09864) · `su_rope_2024`

Encodes absolute position with a rotation matrix and naturally
incorporates explicit relative position dependency in the self-attention
formulation. Arbitrary fixed-length text sequences, linear scaling with
sequence length, and relative position degradation from long-term to
short-term.

### HoPE — Hyperbolic Rotary Positional Encoding
**Dai, Chang et al.** "HoPE: Hyperbolic Rotary Positional Encoding for
Stable Long-Range Dependency Modeling in Large Language Models."
arXiv:2509.05218 (2025).
[arXiv](https://arxiv.org/abs/2509.05218) · `dai_hope_2025`

Generalises RoPE to the hyperbolic geometry of the Lorentz model.
Lorentz rotations built from hyperbolic functions enforce a monotonic
decay of attention weights with increasing token distance — a property
RoPE lacks. RoPE is recovered as a special case.

### NoPE — No Positional Encoding
**Kazemnejad, Amirhossein et al.** "The Impact of Positional Encoding on
Length Generalization in Transformers." arXiv:2305.19466 (2023).
[arXiv](https://arxiv.org/abs/2305.19466) · `kazemnejad_nope_2023`

Systematic empirical study comparing length generalisation of
decoder-only Transformers with APE, T5's Relative PE, ALiBi, Rotary,
and NoPE (no positional encoding). NoPE outperforms other explicit PE
methods on reasoning and mathematical tasks while requiring no
additional computation. Theoretically, NoPE can represent both absolute
and relative PEs, but under SGD training it mostly resembles T5's
relative PE attention patterns.

### ALiBi — Attention with Linear Biases
**Press, Ofir; Smith, Noah A.; Lewis, Mike.** "Train Short, Test Long:
Attention with Linear Biases Enables Input Length Extrapolation."
arXiv:2108.12409 (2022).
[arXiv](https://arxiv.org/abs/2108.12409) · `press_alibi_2022`

Does not add positional embeddings to word embeddings; instead biases
query-key attention scores with a penalty proportional to their
distance. A 1.3B-parameter model trained on 1024-token sequences
extrapolates to 2048 tokens, matching the perplexity of a sinusoidal
model trained on 2048 tokens while training 11% faster and using 11%
less memory.

### BAM — Bayesian Attention Mechanism
**Bianchessi, Arthur S.; Aguirre, Yasmin C.; Barros, Rodrigo C.; Kupssinskü,
Lucas S.** "Bayesian Attention Mechanism: A Probabilistic Framework for
Positional Encoding and Context Length Extrapolation."
arXiv:2505.22842 (2025).
[arXiv](https://arxiv.org/abs/2505.22842) · `bianchessi_bam_2025`

Reframes positional encoding as a probabilistic prior over positions and
unifies existing methods: NoPE = Uniform prior, ALiBi = Uniform × Laplace
prior. Introduces a Generalized-Gaussian positional prior with a learnable
per-head shape parameter (β): β=1 recovers ALiBi, β=2 gives a Normal prior,
0<β<1 yields heavier-than-Laplace tails, and β<0 produces long-range
"retrieval heads". Initialised at θ=0 (Uniform prior), the per-head shape
and scale are trained from scratch. Paired with Scalable Softmax (SSMax),
a transversal logit rescale `s·ln(n)` that counteracts attention fading
and is composable with any PE, BAM SSMax achieves accurate information
retrieval at 500× the training context length.

### PaPE — Parabolic Position Encoding
**Øhrstrøm, Christoffer Koo et al.** "Parabolic Position Encoding:
Vision-Centric, Principled, Extrapolatable, General." arXiv:2602.01418
(2026).
[arXiv](https://arxiv.org/abs/2602.01418) · `ohrstrom_pape_2026`

Parabola-based position encoding for vision modalities in
attention-based architectures. Designed from principles distilled from
prior work: translation invariance, rotation invariance (PaPE-RI),
distance decay, directionality, and context awareness. Extrapolation
experiments on ImageNet-1K show up to 10.5% absolute improvement over
the next-best encoding. Three variants are implemented: `pape`
(reference), `pape_efficient` (pure-PyTorch, no Triton), and `pape_ri`
(rotation-invariant).

### Sinusoidal / Learned Absolute — Original Transformer
**Vaswani, Ashish et al.** "Attention Is All You Need."
arXiv:1706.03762 (2017).
[arXiv](https://arxiv.org/abs/1706.03762) · `vaswani_attention_2017`

The original Transformer paper introduced both the fixed sinusoidal
absolute encoding (`sinusoidal_absolute`) and the learned absolute
encoding (`learned_absolute`). The sinusoidal variant uses sine/cosine
pairs of varying frequencies; the learned variant is a trainable
parameter matrix. The `sinusoidal_rotary` variant reuses the sinusoidal
frequencies as a rotation matrix on Q/K (analogous to RoPE).

## BibTeX Location

All entries are in [`docs/bibliography/other.bib`](other.bib):
- `su_rope_2024` — RoPE
- `dai_hope_2025` — HoPE
- `kazemnejad_nope_2023` — NoPE
- `press_alibi_2022` — ALiBi
- `bianchessi_bam_2025` — BAM
- `ohrstrom_pape_2026` — PaPE (all three variants)
- `vaswani_attention_2017` — sinusoidal + learned absolute