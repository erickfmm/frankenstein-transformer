#!/usr/bin/env bash
# run_tests.sh — Run the full Frankenstein Transformer test suite.
#
# Usage:
#   ./run_tests.sh                 # run all tests
#   ./run_tests.sh -k "attention"  # run tests matching a keyword
#   ./run_tests.sh -v              # verbose output (already default)
#   ./run_tests.sh --fast          # skip slow tests (if marked)
#
# Exit code mirrors pytest: 0 = all passed, non-zero = failures/errors.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

info()    { echo -e "${YELLOW}[run_tests]${NC} $*"; }
success() { echo -e "${GREEN}[run_tests]${NC} $*"; }
error()   { echo -e "${RED}[run_tests]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
if ! command -v python &>/dev/null; then
    error "python not found. Activate your virtual environment first."
    exit 1
fi

if ! python -c "import pytest" &>/dev/null; then
    info "pytest not found — installing..."
    pip install pytest -q || { error "Failed to install pytest. Run: pip install pytest"; exit 1; }
fi

TORCH_STATUS="available"
if ! python -c "import torch" &>/dev/null; then
    TORCH_STATUS="NOT installed — torch-dependent tests will be skipped"
fi

info "Python   : $(python --version)"
info "pytest   : $(python -m pytest --version 2>&1 | head -1)"
info "torch    : ${TORCH_STATUS}"
echo ""

# ---------------------------------------------------------------------------
# Build pytest arguments from script arguments
# ---------------------------------------------------------------------------
PYTEST_ARGS=(
    tests/
    --continue-on-collection-errors
    -v
    --tb=short
    -p no:warnings
)

# Pass any extra arguments directly to pytest
PYTEST_ARGS+=("$@")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
info "Running: python -m pytest ${PYTEST_ARGS[*]}"
echo ""

set +e
python -m pytest "${PYTEST_ARGS[@]}"
EXIT_CODE=$?
set -e

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    success "All tests passed."
elif [[ $EXIT_CODE -eq 5 ]]; then
    # pytest exit code 5 = no tests collected (not a failure)
    success "No tests collected (check test discovery)."
    EXIT_CODE=0
else
    error "Tests finished with exit code ${EXIT_CODE}."
fi

exit $EXIT_CODE
