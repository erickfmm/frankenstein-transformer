#!/usr/bin/env bash
# Rebuild every Mermaid diagram in this directory.
#
# Usage:
#   ./build-figures.sh [pdf|svg|both] [OUTDIR]
#
#   pdf   (default) vector PDFs for LaTeX, written next to the sources.
#   svg   SVGs for the website/paper HTML, written next to the sources unless
#         OUTDIR is given (the sync copies them to the GitHub Pages mirror).
#   both  PDFs and SVGs.
#
# The Mermaid CLI runs through pnpm, one diagram at a time (low-resource
# friendly). Puppeteer needs a Chrome/Chromium binary: the script auto-detects
# google-chrome / chromium / chromium-browser, and PUPPETEER_EXECUTABLE or
# PUPPETEER_CONFIG override the detection.
set -euo pipefail
cd "$(dirname "$0")"

FORMAT="${1:-pdf}"
OUTDIR="${2:-${OUTDIR:-.}}"
case "$FORMAT" in
  pdf|svg|both) ;;
  *) echo "usage: $0 [pdf|svg|both] [OUTDIR]" >&2; exit 2 ;;
esac

# Resolve the browser once and hand Puppeteer a generated config, so the build
# works on machines that only ship Chromium (puppeteer-config.json points at
# google-chrome, which is not always installed).
PUPPETEER_CONFIG="${PUPPETEER_CONFIG:-}"
TMP_CONFIG=""
if [[ -z "$PUPPETEER_CONFIG" ]]; then
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
  printf '{"executablePath": "%s", "args": ["--no-sandbox", "--disable-setuid-sandbox"]}\n' \
    "$BROWSER" > "$TMP_CONFIG"
  PUPPETEER_CONFIG="$TMP_CONFIG"
  trap 'rm -f "$TMP_CONFIG"' EXIT
fi

render_width() { # render_width relative_mmd_path
  case "$1" in
    mixer-taxonomy.mmd) echo 2600 ;;
    latent-map.mmd) echo 1200 ;;
    experiment-*.mmd) echo 1400 ;;
    mixers/*|optimizers/*) echo 1400 ;;
    *) echo 1600 ;;
  esac
}

render() { # render relative_mmd_path pdf|svg
  local src="$1" fmt="$2" out width
  width="$(render_width "$src")"
  if [[ "$fmt" == "pdf" ]]; then
    out="${src%.mmd}.pdf"
  else
    out="$OUTDIR/${src%.mmd}.$fmt"
  fi
  mkdir -p "$(dirname "$out")"
  echo "==> $src -> $out (${fmt}, width ${width}px)"
  local extra=()
  # Thicker outlines for the web build only; the LaTeX PDFs keep the theme default.
  if [[ "$fmt" == "svg" ]]; then
    extra=(-C web-theme.css)
  fi
  # Experiment diagrams use the paper Mermaid config (SVG <text> labels and
  # tighter spacing) so the PDF stays legible at \linewidth.
  case "$src" in
    experiment-*.mmd) extra+=(-c mermaid-paper.json) ;;
  esac
  pnpm dlx @mermaid-js/mermaid-cli \
    -i "$src" -o "$out" \
    -p "$PUPPETEER_CONFIG" \
    --size "$width" \
    --backgroundColor white \
    "${extra[@]}"
}

mapfile -t SOURCES < <(find . -name '*.mmd' -not -name 'test.mmd' | sed 's|^\./||' | sort)
if [[ ${#SOURCES[@]} -eq 0 ]]; then
  echo "No .mmd sources found in $(pwd)" >&2
  exit 1
fi

for src in "${SOURCES[@]}"; do
  case "$FORMAT" in
    pdf) render "$src" pdf ;;
    svg) render "$src" svg ;;
    both)
      render "$src" pdf
      render "$src" svg
      ;;
  esac
done

echo "All ${#SOURCES[@]} diagrams rebuilt (${FORMAT})."
