#!/usr/bin/env bash
# Render the README command/pipeline diagram to a transparent PNG.
#
# Usage:
#   ./build-readme-figure.sh
#
# Requires pnpm plus Chrome/Chromium. Set PUPPETEER_EXECUTABLE to pin a browser.
set -euo pipefail
cd "$(dirname "$0")"

SRC="readme-command-flow.mmd"
OUT="readme-command-flow.png"
SIZE="${SIZE:-2400}"

BROWSER="${PUPPETEER_EXECUTABLE:-}"
if [[ -z "$BROWSER" ]]; then
  for candidate in google-chrome google-chrome-stable chromium chromium-browser chrome; do
    if command -v "$candidate" >/dev/null 2>&1; then
      BROWSER="$(command -v "$candidate")"
      break
    fi
  done
fi
if [[ -z "$BROWSER" ]]; then
  echo "ERROR: no Chrome/Chromium found; set PUPPETEER_EXECUTABLE=/path/to/chrome." >&2
  exit 1
fi

TMP_CONFIG="$(mktemp)"
trap 'rm -f "$TMP_CONFIG"' EXIT
printf '{"executablePath": "%s", "args": ["--no-sandbox", "--disable-setuid-sandbox"]}\n' \
  "$BROWSER" > "$TMP_CONFIG"

pnpm dlx @mermaid-js/mermaid-cli \
  -i "$SRC" \
  -o "$OUT" \
  -p "$TMP_CONFIG" \
  -C readme-figure.css \
  --size "$SIZE" \
  --scale 2 \
  --backgroundColor transparent

echo "Wrote $(pwd)/$OUT"
