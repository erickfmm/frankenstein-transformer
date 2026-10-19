#!/usr/bin/env bash
# Rebuild every Mermaid diagram in this directory into a vector PDF for LaTeX.
# Runs the Mermaid CLI through pnpm, one diagram at a time (low-resource friendly).
set -euo pipefail
cd "$(dirname "$0")"

for src in *.mmd; do
  [ "$src" = "test.mmd" ] && continue
  out="${src%.mmd}.pdf"
  width=1600
  if [ "$src" = "mixer-taxonomy.mmd" ]; then width=2600; fi
  if [ "$src" = "latent-map.mmd" ]; then width=1200; fi
  echo "==> $src -> $out (width ${width}px)"
  pnpm dlx @mermaid-js/mermaid-cli \
    -i "$src" -o "$out" \
    -p puppeteer-config.json \
    --size "$width"
done

echo "All diagrams rebuilt."
