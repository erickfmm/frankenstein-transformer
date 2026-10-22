#!/usr/bin/env bash
# Rebuild every Mermaid diagram in this directory into a vector PDF for LaTeX.
# Runs the Mermaid CLI through pnpm, one diagram at a time (low-resource friendly).
set -euo pipefail
cd "$(dirname "$0")"

# Optional override: PUPPETEER_CONFIG=/path/to/config.json ./build-figures.sh
PUPPETEER_CONFIG="${PUPPETEER_CONFIG:-puppeteer-config.json}"

for src in *.mmd; do
  [ "$src" = "test.mmd" ] && continue
  out="${src%.mmd}.pdf"
  width=1600
  if [ "$src" = "mixer-taxonomy.mmd" ]; then width=2600; fi
  if [ "$src" = "latent-map.mmd" ]; then width=1200; fi
  echo "==> $src -> $out (width ${width}px)"
  pnpm dlx @mermaid-js/mermaid-cli \
    -i "$src" -o "$out" \
    -p "$PUPPETEER_CONFIG" \
    --size "$width"
done

# Per-mixer dataflow diagrams (one per sequence mixer) live in mixers/ and are
# embedded by the attention annexes of both papers.
for src in mixers/*.mmd; do
  out="${src%.mmd}.pdf"
  echo "==> $src -> $out (width 1400px)"
  pnpm dlx @mermaid-js/mermaid-cli \
    -i "$src" -o "$out" \
    -p "$PUPPETEER_CONFIG" \
    --size 1400
done

# Per-optimizer step diagrams live in optimizers/ and are embedded by the
# optimizer annexes of both papers.
for src in optimizers/*.mmd; do
  out="${src%.mmd}.pdf"
  echo "==> $src -> $out (width 1400px)"
  pnpm dlx @mermaid-js/mermaid-cli \
    -i "$src" -o "$out" \
    -p "$PUPPETEER_CONFIG" \
    --size 1400
done

echo "All diagrams rebuilt."
