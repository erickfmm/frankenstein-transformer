# Documentation

This folder holds the project's docs. There is exactly **one implementation
specification** — [`architecture.md`](architecture.md) — plus the academic
papers and the bibliography. The former `docs/specs/` directory was removed
because it had drifted from the code and duplicated both the paper and the
architecture guide.

## Where to read what

| Document | Contents |
|---|---|
| [`../README.md`](../README.md) | Quickstart, feature overview, web UI link |
| [`architecture.md`](architecture.md) | **Canonical implementation spec** and extension guide (Mermaid diagrams, component tables, recipes) |
| [`../configs/README.md`](../configs/README.md) | Presets and the full optimizer-parameter reference |
| [`paper/`](paper/), [`paper-es/`](paper-es/) | **Source of truth for theory**, formulations and citations |
| [`bibliography/`](bibliography/) | BibTeX databases (`.bib`) used by the papers |
| [`source/`](source/) | Sphinx sources (API autodoc, papers, hosted docs) |
| [`../AGENTS.md`](../AGENTS.md) | Agent-facing rules, where-to-find map, decisions, skills |
| [`dashai-plugin-audit.md`](dashai-plugin-audit.md) | DashAI plugin integration design |
| [`transformers_compatibility.md`](transformers_compatibility.md) | HuggingFace export compatibility |

## Building the documentation locally

The pipeline is **Sphinx + MyST + Read the Docs**.

```bash
# Build HTML (installs Sphinx deps if missing; compiles LaTeX papers when pdflatex is available)
bash docs/build_docs.sh
# Output: docs/_build/html/index.html
```

The Read the Docs build (`.readthedocs.yaml`) reuses the same sources under
`docs/source/`.

## Academic reports (LaTeX)

The LaTeX sources are split by section under `docs/paper/` (English) and
`docs/paper-es/` (Spanish). Each folder references the shared
`docs/bibliography/` via `../bibliography/...`, so compile **from inside the
paper folder**.

English (from `docs/paper/`):

```bash
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

Spanish (from `docs/paper-es/`):

```bash
pdflatex -interaction=nonstopmode paper-es.tex
bibtex paper-es
pdflatex -interaction=nonstopmode paper-es.tex
pdflatex -interaction=nonstopmode paper-es.tex
```

Each main file contains only the preamble, front matter and `\input` links to
`sections/` and `appendices/`. The compiled PDFs are embedded in the Read the
Docs pages **Technical Report (English)** and **Informe Tecnico (Espanol)**.
