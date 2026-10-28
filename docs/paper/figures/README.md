# Paper figures

Mermaid sources for the diagrams in the paper body and the attention annexes
(no TikZ is used). Each `.mmd` file is compiled to a vector PDF that LaTeX
embeds with `\includegraphics`.

`mixers/` holds one dataflow diagram per sequence mixer, embedded at the start
of each mixer's section in the attention annexes (English sources; the Spanish
paper reuses the same PDFs). Top-level `*.mmd` files are the paper-body figures,
including `audio_frontend.mmd` (audio model class).

`optimizers/` holds one step-flow diagram per optimizer, embedded at the start
of each optimizer's entry in the optimizer annexes. Sources and labels are in
English; the Spanish paper reuses the same PDFs.

Rebuild every diagram (one at a time, Mermaid CLI via `pnpm`):

```bash
./build-figures.sh
```

The script uses `puppeteer-config.json`, which points Puppeteer at the system
Chromium (`/usr/bin/google-chrome`). If Chromium lives elsewhere, override it
without editing the repo config:

```bash
PUPPETEER_CONFIG=/path/to/puppeteer.json ./build-figures.sh
```

Both `docs/paper/` and `docs/paper-es/` embed the same PDFs.
