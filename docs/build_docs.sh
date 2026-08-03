#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# build_docs.sh — Build Sphinx ReadTheDocs-style HTML documentation
# for Frankenstein Transformer
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_SOURCE="$SCRIPT_DIR/source"
DOCS_BUILD="$SCRIPT_DIR/_build/html"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Step 1: Check/install Sphinx dependencies ──────────────────────────────
log "Checking Sphinx dependencies..."

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_CMD="$PYTHON_BIN -m pip"

MISSING_PKGS=()
for pkg in sphinx myst-parser sphinx-rtd-theme linkify-it-py; do
    if ! $PYTHON_BIN -c "import ${pkg//-/_}" 2>/dev/null; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    warn "Missing packages: ${MISSING_PKGS[*]}"
    log "Installing Sphinx dependencies..."
    $PIP_CMD install "${MISSING_PKGS[@]}"
    ok "Sphinx dependencies installed."
else
    ok "All Sphinx dependencies already installed."
fi

# ── Step 2: Ensure LaTeX papers are compiled ────────────────────────────────
log "Checking LaTeX papers..."

compile_latex() {
    local tex_file="$1"
    local tex_dir
    tex_dir="$(dirname "$tex_file")"
    local base_name
    base_name="$(basename "$tex_file" .tex)"

    if [[ -f "$tex_dir/$base_name.pdf" ]]; then
        ok "$base_name.pdf already exists."
        return 0
    fi

    warn "$base_name.pdf not found. Attempting to compile..."
    if command -v pdflatex &>/dev/null; then
        (
            cd "$tex_dir"
            pdflatex -interaction=nonstopmode "$base_name.tex" > /dev/null 2>&1 || true
            if [[ -f "$base_name.aux" ]] && grep -q '\\citation' "$base_name.aux" 2>/dev/null; then
                bibtex "$base_name" > /dev/null 2>&1 || true
                pdflatex -interaction=nonstopmode "$base_name.tex" > /dev/null 2>&1 || true
                pdflatex -interaction=nonstopmode "$base_name.tex" > /dev/null 2>&1 || true
            fi
        )
        if [[ -f "$tex_dir/$base_name.pdf" ]]; then
            ok "Compiled $base_name.pdf successfully."
        else
            warn "pdflatex ran but $base_name.pdf was not produced."
        fi
    else
        warn "pdflatex not found. Skipping LaTeX compilation."
        warn "Install texlive to compile papers: sudo apt install texlive-latex-base texlive-bibtex-extra"
    fi
}

compile_latex "$PROJECT_ROOT/docs/paper.tex"
compile_latex "$PROJECT_ROOT/docs/paper-es.tex"

# ── Step 3: Build Sphinx HTML ──────────────────────────────────────────────
log "Building Sphinx HTML documentation..."

cd "$SCRIPT_DIR"

sphinx-build -b html \
    -d "$SCRIPT_DIR/_build/doctrees" \
    -W --keep-going \
    "$DOCS_SOURCE" \
    "$DOCS_BUILD"

ok "Sphinx HTML build complete."

# ── Step 4: Summary ─────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  Documentation built successfully!                          ║${NC}"
echo -e "${BOLD}${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}${GREEN}║${NC}  HTML output: ${BLUE}$DOCS_BUILD${NC}"
echo -e "${BOLD}${GREEN}║${NC}  Open with:   ${BLUE}xdg-open $DOCS_BUILD/index.html${NC}"
echo -e "${BOLD}${GREEN}║${NC}  Serve with:  ${BLUE}python -m http.server 8080 -d $DOCS_BUILD${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
ok "Done."
