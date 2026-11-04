#!/usr/bin/env bash
# Publish the Experiment 2 report to the GitHub Pages mirror.
#
# Usage:  bash experimentos/2/report/publish.sh [--no-raw]
#
# Copies the generated report, figures and datasets into
#   erickfmm.github.io/frankenstein-transformer/charts/
# and (unless --no-raw) the 868 per-run CSVs into charts/data/raw/.
#
# It never commits or pushes: review `git -C erickfmm.github.io status` first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)"
HERE="$ROOT/experimentos/2/report"
SITE="$ROOT/erickfmm.github.io/frankenstein-transformer/charts"
RAW_SRC="$ROOT/experimentos/2/training_metrics"

if [[ ! -f "$HERE/out/index.html" ]]; then
  echo "missing $HERE/out/index.html — run build_data.py, make_charts.py and make_report.py first" >&2
  exit 1
fi
if [[ ! -d "$ROOT/erickfmm.github.io" ]]; then
  echo "website mirror not found at $ROOT/erickfmm.github.io" >&2
  exit 1
fi

mkdir -p "$SITE"/{data,img,vendor}

echo "→ report + figures"
cp "$HERE/out/index.html" "$SITE/index.html"
# The mirror lives one level deeper than the source, so fix the favicon path.
sed -i 's#href="\.\./favicon\.ico"#href="../../favicon.ico"#' "$SITE/index.html"
cp "$HERE/out/img/"*.png "$SITE/img/"

echo "→ datasets"
cp "$HERE/out/data/"*.csv "$SITE/data/"

if [[ "${1:-}" != "--no-raw" ]]; then
  echo "→ raw per-run logs, gzip-compressed (868 files, ~160 MB)"
  mkdir -p "$SITE/data/raw"
  rm -f "$SITE/data/raw/"*.csv
  for f in "$RAW_SRC"/*.csv; do
    gzip -9 -c "$f" > "$SITE/data/raw/$(basename "$f").gz"
  done
fi

if [[ ! -f "$SITE/vendor/echarts.min.js" ]]; then
  echo "warning: vendor/echarts.min.js is missing (the chart maker needs it)" >&2
fi

echo
echo "done. next:"
echo "  git -C $ROOT/erickfmm.github.io status --short"
