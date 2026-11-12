# Paper figures

Mermaid sources for the diagrams in the paper body and the attention annexes
(no TikZ is used). Each `.mmd` file is compiled to:

- a vector **PDF** that LaTeX embeds with `\includegraphics`, and
- an **SVG** used by the HTML render of the paper (the website *Paper* tab).

Every diagram is bilingual. `<name>.mmd` is the English source used by
`docs/paper/`, and `<name>-es.mmd` the Spanish one used by `docs/paper-es/`;
only the human-readable text changes (node labels, edge labels, subgraph
titles, comments). The `%%{init: ...}%%` header, node ids, edges, tensor
shapes, formulas and operator names are copied verbatim, so both languages
share layout, style and size. `docs/paper/` embeds `<name>.pdf` and
`docs/paper-es/` embeds `<name>-es.pdf`; the website mirror indexes them by
stem, so the Spanish Paper tab picks up `<name>-es.svg` automatically.

`mixers/` holds one dataflow diagram per sequence mixer, embedded at the start
of each mixer's section in the attention annexes. Top-level `*.mmd` files are
the paper-body figures, including `audio_frontend.mmd` (audio model class).

`optimizers/` holds one step-flow diagram per optimizer, embedded at the start
of each optimizer's entry in the optimizer annexes (English and `-es`).

`optimizers/tutorial/` holds the nine charts of the optimizer tutorial
(Annex 9.2), in English and `-es`. They are not Mermaid sources: they are
rendered by `optimizers/tutorial/make_tutorial_figures.py` with matplotlib and
seaborn, as PDF (LaTeX) plus SVG (web), in the same light theme as the
experiment charts. Each builder takes `lang="en" | "es"` and routes every
user-visible string through a `tr()` table.
Rebuild them with

```bash
uv run --with matplotlib --with seaborn --with numpy \
    python figures/optimizers/tutorial/make_tutorial_figures.py [NAME ...]
```

`optimizer-tutorial-flow.mmd` is the only tutorial figure that belongs to the
Mermaid build; everything else in that folder is Python output.

`tutorial/` holds the figures of the beginner tutorial (Annex 1). The five
Mermaid diagrams (`tutorial-*.mmd`) follow the same `-es` convention. The
three matplotlib charts (`tut-attention-heatmap`, `tut-softmax-focus`,
`tut-complexity-growth`) are
rendered with their `-es` counterparts by `tutorial/make_tutorial_figures.py`,
same PDF+SVG convention and rebuild command as the optimizer tutorial above.

`experiments/` holds the figures of the empirical-evaluation section and its
appendices. `experiments/round2/*.pdf` and `*.svg` (English) and
`*-es.pdf`/`*-es.svg` (Spanish) are matplotlib charts exported by
`experimentos/2/report/make_paper_figures.py`, which imports the chart builders
from the Experiment 2 web report and re-renders a curated subset in the light
theme at paper size. The builders read their strings through
`make_charts.tr()`, and the exporter flips `make_charts.set_lang()` per
language, so both variants share canvas, theme and scale. They are not Mermaid
sources; only the two top-level `experiment-pipeline.mmd` and
`experiment-reporting.mmd` diagrams belong to the build script, and they use
`mermaid-paper.json` for legible SVG-text labels.

## Rebuild

One diagram at a time (low-resource friendly), through the Mermaid CLI via
`pnpm`:

```bash
./build-figures.sh          # PDFs only (default) → next to the sources
./build-figures.sh svg      # SVGs only         → next to the sources
./build-figures.sh both     # both
```

Both formats must stay in sync with the `.mmd` sources; commit them together
when a diagram changes. The website mirror copies the SVG tree into
`erickfmm.github.io/frankenstein-transformer/paper/figures/` (see
`.opencode/command/sync-website.md`).

SVG builds pass `web-theme.css` to the Mermaid CLI, which thickens the
node/cluster outlines (2px) so the boxes stay legible when the diagram is
scaled down in the Paper tab. PDF builds keep the theme default.

The script auto-detects `google-chrome` / `chromium` / `chromium-browser`. To
pin a specific browser, use:

```bash
PUPPETEER_EXECUTABLE=/path/to/chrome ./build-figures.sh svg
```

### Why two Mermaid CLIs

`build-figures.sh` pins a different Mermaid CLI per format on purpose:

- **PDF (LaTeX): `@mermaid-js/mermaid-cli@12.2.0 --size <width>`.** This is the
  figure style the paper was designed around: node gradients and the drop
  shadow are rasterised at 300 dpi inside the PDF, and the page is fitted to
  the diagram. `11.4.2` with `--width`/`--pdfFit` exports the same diagram as a
  flat vector box with no shadows, which reads as a generic diagram next to the
  rest of the paper.
- **SVG (website Paper tab): `@mermaid-js/mermaid-cli@11.4.2 --width <width>
  -C web-theme.css`.** Keeps the node metrics and thick outlines the published
  render was tuned against.

Both formats must be regenerated together for a figure change
(`./build-figures.sh both`). `ONLY='mixers/kpa.mmd'` limits a run to matching
sources; `ONLY='*-es.mmd'` rebuilds only the Spanish set.

### Translating a diagram

When a diagram changes, edit both sources. Copy the English file and translate
only the human-readable text: the `%%{init: ...}%%` line, node ids, `subgraph`
ids, edges, `direction`, `classDef`/`class` lines and every formula, tensor
shape and operator name (`softmax`, `RoPE`, `SVD`, `[B,n,H]`, …) stay
byte-identical. Keep the wrapping width of the header, so the two renders keep
the same size instead of drifting apart.

`puppeteer-config.json` is kept for manual runs of `mermaid-cli`; the build
script generates its own temporary Puppeteer config so it works on machines
where Chrome is not at `/usr/bin/google-chrome`.
