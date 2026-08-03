# Plan: Expand Activation-Function Annex 8 (EN + ES)

## Objective

Rewrite `docs/paper/appendices/annex-8-activation-functions.tex` and
`docs/paper-es/appendices/annex-8-funciones-activacion.tex` to document **all 43
activation functions** (40 elementwise + 3 gated-FFN) with per-function formula,
range, monotonicity, differentiability, learnable parameters, and bibliography
citation — aligned 1:1 with the in-code docstrings in
`src/model/activation_function/`. Add ~6 missing `.bib` entries, recompile both
PDFs, regenerate the pandoc HTML copies, optionally rebuild Sphinx, and
commit+push per `make-a-commit.md`.

## Structure per family (subsection each)

| Subsection | Functions (with own formula) |
|---|---|
| Classical / Sigmoid–Tanh (12) | Sigmoid, Tanh, Arctan, Softsign, Elliott, Identity, Softplus, Mish, GELU, GELU-tanh, ReLU, SiLU |
| Rectified (12) | LeakyReLU, ReLU6, HardSwish, PReLU, AbsReLU, NLReLU, BReLU, VReLU, Hexpo, PenalizedTanh, DisReLU, LiSHT |
| Exponential / ELU (12) | ELU, SELU, CELU, PELU, MPELU, FELU, EELU, PDELU, PREU, SoftExp, ELiSH, HardELiSH |
| Learnable / Adaptive (4) | Swish (fixed beta), SwishTrainable, Maxout, RAF (versions A/B/C/D/N + RAFT scaling) |
| Gated FFN (3) | SwiGLU, GEGLU, ReGLU (GatedFFN module, make_gated_ffn) |
| Factory & Configuration | get_activation dispatch, GLU_VARIANTS, ffn_activation_config full key table |

## New bib entries (docs/bibliography/other.bib)

1. `elfwing2018silu` — Elfwing et al., "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning", 2018. arXiv:1702.03118
2. `trottier2017pelu` — Trottier et al., "Parametric Exponential Linear Unit for Deep Learning", ICIP 2017. arXiv:1605.09332
3. `godfrey2015softexp` — Godfrey & Gashler, "Soft Exponential Activation Functions for Deep Neural Networks", ICANN 2015. arXiv:1602.01321
4. `howard2017relu6` — Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", 2017. arXiv:1704.04861
5. `forest2014nlrelu` — Forest, "NLReLU: Natural Logarithm Rectified Linear Unit", 2014. (arXiv or workshop)
6. `bjorck2017absrelu` — Bjorck et al., "Understanding Training and Generalization in Deep Learning by Fourier Analysis", 2018. arXiv:1808.01819

## Build steps

### 1. Recompile PDFs
```bash
for dir in docs/paper docs/paper-es; do
  cd $dir
  base=$(basename $dir | sed 's/-es//')
  rm -f ${base}.pdf ${base}.aux ${base}.bbl ${base}.blg ${base}.log ${base}.toc ${base}.out
  pdflatex -interaction=nonstopmode ${base}.tex
  bibtex ${base}
  pdflatex -interaction=nonstopmode ${base}.tex
  pdflatex -interaction=nonstopmode ${base}.tex
  cd ../..
done
```

### 2. Regenerate HTML (pandoc)
```bash
pandoc --standalone --mathjax \
  -o docs/source/_static/papers/paper-en.html \
  docs/paper/paper.tex

pandoc --standalone --mathjax \
  -o docs/source/_static/papers/paper-es.html \
  docs/paper-es/paper-es.tex
```

### 3. Sphinx build (optional local verification)
```bash
bash docs/build_docs.sh
```

### 4. Commit + push (make-a-commit.md)
Stage: 2 `.tex` annexes, `other.bib`, 2 PDFs, 2 HTML.
Message: `docs(annex-8): expand activation-function annex with full per-function detail and references`

## Verification
- 0 undefined references in pdflatex output
- Every `\citep{key}` in annex matches an entry in `other.bib`
- Enum count (43) matches documented count
- Sphinx build clean (if run)
