# Paper figures

Mermaid sources for the diagrams in the paper body (the appendices do not use
TikZ). Each `.mmd` file is compiled to a vector PDF that LaTeX embeds with
`\includegraphics`.

Rebuild every diagram (one at a time, Mermaid CLI via `pnpm`):

```bash
./build-figures.sh
```

The script uses `puppeteer-config.json`, which points Puppeteer at the system
Chromium (`/usr/bin/google-chrome`). Both `docs/paper/` and `docs/paper-es/`
embed the same PDFs.
