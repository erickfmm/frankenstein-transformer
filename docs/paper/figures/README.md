# Paper figures

Mermaid sources for the diagrams in the paper body and the attention annexes
(no TikZ is used). Each `.mmd` file is compiled to:

- a vector **PDF** that LaTeX embeds with `\includegraphics` (both
  `docs/paper/` and `docs/paper-es/` reuse the same PDFs), and
- an **SVG** used by the HTML render of the paper (the website *Paper* tab).

`mixers/` holds one dataflow diagram per sequence mixer, embedded at the start
of each mixer's section in the attention annexes. Top-level `*.mmd` files are
the paper-body figures, including `audio_frontend.mmd` (audio model class).

`optimizers/` holds one step-flow diagram per optimizer, embedded at the start
of each optimizer's entry in the optimizer annexes. Sources and labels are in
English; the Spanish paper reuses the same PDF/SVG set.

`experiments/` holds the figures of the empirical-evaluation section and its
appendices. `experiments/round2/*.pdf` and `*.svg` are matplotlib charts
exported by `experimentos/2/report/make_paper_figures.py`, which imports the
chart builders from the Experiment 2 web report and re-renders a curated subset
in the light theme at paper size; the PDFs go to LaTeX and the SVGs to the web
render. They are not Mermaid sources; only the two top-level
`experiment-pipeline.mmd` and `experiment-reporting.mmd` diagrams belong to the
build script, and they use `mermaid-paper.json` for legible SVG-text labels.

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

`puppeteer-config.json` is kept for manual runs of `mermaid-cli`; the build
script generates its own temporary Puppeteer config so it works on machines
where Chrome is not at `/usr/bin/google-chrome`.
