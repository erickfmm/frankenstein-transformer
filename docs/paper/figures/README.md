# Paper figures

Mermaid sources for the diagrams in the paper body and the attention annexes
(no TikZ is used). Each `.mmd` file is compiled to a vector PDF that LaTeX
embeds with `\includegraphics`.

`mixers/` holds one dataflow diagram per sequence mixer, embedded at the start
of each mixer's section in the attention annexes (English sources; the Spanish
paper reuses the same PDFs).

Rebuild every diagram (one at a time, Mermaid CLI via `pnpm`):

```bash
./build-figures.sh
```

The script uses `puppeteer-config.json`, which points Puppeteer at the system
Chromium (`/usr/bin/google-chrome`). Both `docs/paper/` and `docs/paper-es/`
embed the same PDFs.
