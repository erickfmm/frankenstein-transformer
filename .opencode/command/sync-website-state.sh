#!/usr/bin/env bash
# sync-website-state.sh — READ-ONLY state report for the /sync-website command.
#
# Prints the current desync state between this repo (source of truth) and the
# GitHub Pages mirror at ./erickfmm.github.io/frankenstein-transformer/, plus
# consistency probes on index.html and tool availability. Performs NO writes,
# NO git operations, NO network access — safe to run at prompt-build time.
#
# Sections: [paths] [website-repo] [drift] [probes] [estimator] [tools] [summary]
# Drift statuses: DIFF | MISSING_SITE | ONLY_SITE (in-sync files are counted, not listed).

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEBSITE="$REPO_ROOT/erickfmm.github.io"
SITE="$WEBSITE/frankenstein-transformer"
CLI_SRC="$REPO_ROOT/src/cli.py"
PYPROJECT="$REPO_ROOT/pyproject.toml"
INDEX="$SITE/index.html"
TRUTH="$REPO_ROOT/full_tests/param_truth.json"
ESTIMATOR="$SITE/ft-param-estimator.js"

# ---------------------------------------------------------------------------
# [paths]
# ---------------------------------------------------------------------------
echo "=== SYNC-WEBSITE STATE ==="
echo "[paths]"
echo "main_repo=$REPO_ROOT"
echo "website=$WEBSITE"
echo "site=$SITE"

# ---------------------------------------------------------------------------
# [website-repo]
# ---------------------------------------------------------------------------
echo ""
echo "[website-repo]"
website_present="no"
if [[ ! -d "$WEBSITE/.git" ]]; then
  echo "WEBSITE_REPO=MISSING"
  echo "clone=git clone https://github.com/erickfmm/erickfmm.github.io.git \"$WEBSITE\""
else
  website_present="yes"
  echo "WEBSITE_REPO=present"
  echo "branch=$(git -C "$WEBSITE" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  dirty=$(git -C "$WEBSITE" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  echo "dirty_files=$dirty"
  if [[ "${dirty:-0}" -gt 0 ]]; then
    git -C "$WEBSITE" status --porcelain 2>/dev/null | head -15 | sed 's/^/  /'
    if [[ "$dirty" -gt 15 ]]; then echo "  ...and $((dirty - 15)) more"; fi
  fi
  ab=$(git -C "$WEBSITE" rev-list --left-right --count "origin/main...HEAD" 2>/dev/null || true)
  if [[ -n "$ab" ]]; then
    echo "behind/ahead_vs_origin_main=$(echo "$ab" | tr '\t' '/')"
  fi
fi

# ---------------------------------------------------------------------------
# [drift] — per-file comparison, main repo -> website mirror
# ---------------------------------------------------------------------------
DIFF_N=0 MISSING_SITE=0 ONLY_SITE=0 OK_N=0
declare -a DRIFT_LINES=()

emit() { # emit STATUS rel_src rel_dst
  case "$1" in
    DIFF) DIFF_N=$((DIFF_N + 1)) ;;
    MISSING_SITE) MISSING_SITE=$((MISSING_SITE + 1)) ;;
    ONLY_SITE) ONLY_SITE=$((ONLY_SITE + 1)) ;;
    *) OK_N=$((OK_N + 1)); return ;;
  esac
  DRIFT_LINES+=("$1 $2 -> $3")
}

pair_file() { # pair_file abs_src abs_dst rel_src rel_dst
  local s="$1" d="$2"
  if [[ -e "$s" && ! -e "$d" ]]; then
    emit MISSING_SITE "$3" "$4"
  elif [[ ! -e "$s" && -e "$d" ]]; then
    emit ONLY_SITE "$3" "$4"
  elif [[ -e "$s" && -e "$d" ]]; then
    if cmp -s "$s" "$d"; then emit OK "$3" "$4"; else emit DIFF "$3" "$4"; fi
  fi
}

pair_tree() { # pair_tree abs_src_dir abs_dst_dir rel_src rel_dst [find args...]
  local s="$1" d="$2" rs="$3" rd="$4"
  shift 4
  [[ -d "$s" || -d "$d" ]] || return 0
  local union rel
  union=$(
    {
      [[ -d "$s" ]] && (cd "$s" && find . "$@" 2>/dev/null)
      [[ -d "$d" ]] && (cd "$d" && find . "$@" 2>/dev/null)
    } | sed 's|^\./||' | sed '/^$/d' | sort -u
  )
  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    pair_file "$s/$rel" "$d/$rel" "$rs/$rel" "$rd/$rel"
  done <<< "$union"
}

echo ""
echo "[drift]"

# schema (single file + directory)
pair_file "$REPO_ROOT/src/schema.yaml" "$SITE/schema.yaml" "src/schema.yaml" "schema.yaml"
pair_tree "$REPO_ROOT/src/schema" "$SITE/schema" "src/schema" "schema" -type f

# configs -> examples (top-level yamls + README only; subdirs via explicit pair below)
pair_tree "$REPO_ROOT/configs" "$SITE/examples" "configs" "examples" -maxdepth 1 -type f -name "*.yaml"
pair_file "$REPO_ROOT/configs/README.md" "$SITE/examples/README.md" "configs/README.md" "examples/README.md"
pair_tree "$REPO_ROOT/configs/examples" "$SITE/examples/examples" "configs/examples" "examples/examples" -type f

# paper (en / es)
P="$REPO_ROOT/docs/paper"
PE="$REPO_ROOT/docs/paper-es"
pair_file "$P/paper.tex" "$SITE/paper/en/paper.tex" "docs/paper/paper.tex" "paper/en/paper.tex"
pair_file "$P/paper.bbl" "$SITE/paper/en/paper.bbl" "docs/paper/paper.bbl" "paper/en/paper.bbl"
pair_file "$P/paper.pdf" "$SITE/paper/en/paper.pdf" "docs/paper/paper.pdf" "paper/en/paper.pdf"
pair_tree "$P/sections" "$SITE/paper/en/sections" "docs/paper/sections" "paper/en/sections" -type f -name "*.tex"
pair_tree "$P/appendices" "$SITE/paper/en/appendices" "docs/paper/appendices" "paper/en/appendices" -type f -name "*.tex"
pair_file "$PE/paper-es.tex" "$SITE/paper/es/paper-es.tex" "docs/paper-es/paper-es.tex" "paper/es/paper-es.tex"
pair_file "$PE/paper-es.bbl" "$SITE/paper/es/paper-es.bbl" "docs/paper-es/paper-es.bbl" "paper/es/paper-es.bbl"
pair_file "$PE/paper-es.pdf" "$SITE/paper/es/paper-es.pdf" "docs/paper-es/paper-es.pdf" "paper/es/paper-es.pdf"
pair_tree "$PE/sections" "$SITE/paper/es/sections" "docs/paper-es/sections" "paper/es/sections" -type f -name "*.tex"
pair_tree "$PE/appendices" "$SITE/paper/es/appendices" "docs/paper-es/appendices" "paper/es/appendices" -type f -name "*.tex"

# bibliography + specs
pair_tree "$REPO_ROOT/docs/bibliography" "$SITE/paper/bibliography" "docs/bibliography" "paper/bibliography" -type f
pair_tree "$REPO_ROOT/docs/specs" "$SITE/paper/specs" "docs/specs" "paper/specs" -type f

# streamlit gui + schema loader
pair_file "$REPO_ROOT/src/streamlit_gui/app.py" "$SITE/streamlit_gui/app.py" "src/streamlit_gui/app.py" "streamlit_gui/app.py"
pair_file "$REPO_ROOT/src/streamlit_gui/__init__.py" "$SITE/streamlit_gui/__init__.py" "src/streamlit_gui/__init__.py" "streamlit_gui/__init__.py"
pair_file "$REPO_ROOT/src/utils/schema_loader.py" "$SITE/utils/schema_loader.py" "src/utils/schema_loader.py" "utils/schema_loader.py"

if [[ ${#DRIFT_LINES[@]} -eq 0 ]]; then
  echo "all pairs in sync"
else
  printf '%s\n' "${DRIFT_LINES[@]}" | head -60
  if [[ ${#DRIFT_LINES[@]} -gt 60 ]]; then
    echo "...and $(( ${#DRIFT_LINES[@]} - 60 )) more"
  fi
fi
echo "counts: diff=$DIFF_N missing_site=$MISSING_SITE only_site=$ONLY_SITE ok=$OK_N"

# ---------------------------------------------------------------------------
# [probes] — consistency of index.html and derived site artifacts
# ---------------------------------------------------------------------------
echo ""
echo "[probes]"
PROBE_FAIL=0
probe() { # probe PASS|FAIL id detail...
  local st="$1" id="$2"
  shift 2
  if [[ "$st" == "FAIL" ]]; then PROBE_FAIL=$((PROBE_FAIL + 1)); fi
  printf '%s %s: %s\n' "$st" "$id" "$*"
}

# Expected CLI entrypoint name from pyproject.toml
pyname=$(sed -n '/^\[project.scripts\]/,/^\[/ { s/^\([A-Za-z0-9_-]*\) *=.*/\1/p; }' "$PYPROJECT" 2>/dev/null | head -1)
pyname="${pyname:-frankenstein-transformer}"

if [[ ! -f "$INDEX" ]]; then
  probe FAIL index-missing "$INDEX not found"
else
  # 1) binary name used by the generated CLI command
  if grep -q "cmd = 'frankestein-transformer" "$INDEX"; then
    probe FAIL cmd-binary-name "index.html generates 'frankestein-transformer'; expected '$pyname' (pyproject [project.scripts])"
  elif grep -q "cmd = '$pyname " "$INDEX"; then
    probe PASS cmd-binary-name "'$pyname'"
  else
    probe FAIL cmd-binary-name "generated command prefix not found (expected \"cmd = '$pyname ...\")"
  fi

  # 2) GitHub repo URL spelling
  remote_url=$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null | sed 's/\.git$//')
  expected_repo="${remote_url##*/}"
  if grep -q "github.com/erickfmm/frankestein-transformer" "$INDEX"; then
    probe FAIL github-url "links to github.com/erickfmm/frankestein-transformer; expected github.com/erickfmm/$expected_repo"
  else
    probe PASS github-url "no misspelled repo link found"
  fi

  # 3) deploy --format choices offered by the page vs src/cli.py
  cli_fmts=$(sed -n 's/.*"--format", type=str, choices=\[\([^]]*\)\].*/\1/p' "$CLI_SRC" 2>/dev/null | head -1 | tr -d '" ' | tr ',' '\n' | sed '/^$/d' | sort -u)
  page_fmts=$(grep -o 'value="--format [a-z]*"' "$INDEX" 2>/dev/null | sed 's/value="--format //; s/"//' | sort -u)
  bad_fmts=""
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    if ! grep -qx "$f" <<< "$cli_fmts"; then bad_fmts="$bad_fmts $f"; fi
  done <<< "$page_fmts"
  if [[ -n "$bad_fmts" ]]; then
    probe FAIL deploy-format "page offers invalid --format:$bad_fmts; cli.py choices: $(echo "$cli_fmts" | tr '\n' ' ')"
  else
    probe PASS deploy-format "page --format options ⊆ cli.py choices ($(echo "$cli_fmts" | tr '\n' ' '))"
  fi

  # 4) subcommands that the page appends --config to vs subcommands that accept it
  js_subs=$(grep -o "\['train'[^]]*\]" "$INDEX" 2>/dev/null | head -1 | tr -d "[]'" | tr ',' '\n' | sed 's/ //g; /^$/d')
  no_cfg=""
  while IFS= read -r sub; do
    [[ -z "$sub" ]] && continue
    var="${sub//-/_}_parser"
    if ! grep -q "$var\.add_argument(\"--config\"" "$CLI_SRC" 2>/dev/null; then
      no_cfg="$no_cfg $sub"
    fi
  done <<< "$js_subs"
  if [[ -n "$no_cfg" ]]; then
    probe FAIL cmd-config-flag "page appends --config for:$no_cfg — these subcommands have no --config in cli.py"
  else
    probe PASS cmd-config-flag "--config appended only for subcommands that accept it"
  fi

  # 5) Command tab subcommands ⊆ cli.py subparsers
  cli_subs=$(grep -o 'add_parser("[a-z-]*"' "$CLI_SRC" 2>/dev/null | sed 's/add_parser("//; s/"//' | sort -u)
  page_cmds=$(sed -n '/id="cmd-select"/,/<\/select>/p' "$INDEX" | grep -o 'value="[a-z-]*"' | sed 's/value="//; s/"//' | sort -u)
  bad_cmds=""
  while IFS= read -r c; do
    [[ -z "$c" ]] && continue
    if ! grep -qx "$c" <<< "$cli_subs"; then bad_cmds="$bad_cmds $c"; fi
  done <<< "$page_cmds"
  if [[ -n "$bad_cmds" ]]; then
    probe FAIL cmd-select "Command tab offers unknown subcommand(s):$bad_cmds"
  else
    probe PASS cmd-select "all Command tab subcommands exist in cli.py ($(echo "$page_cmds" | tr '\n' ' '))"
  fi

  # 6) sbert-infer --mode must not offer an empty default (CLI: required=True)
  # (capture first: `sed | grep -q` under pipefail fails via SIGPIPE when grep exits early)
  mode_block=$(sed -n '/id="opt-sbert-infer-mode"/,/<\/select>/p' "$INDEX")
  if grep -q 'value=""' <<< "$mode_block"; then
    probe FAIL sbert-mode-default "--mode select offers empty '(default)' but sbert-infer requires --mode (choices: similarity|search|cluster|encode)"
  else
    probe PASS sbert-mode-default "--mode has no empty default"
  fi

  # 7) asset paths point at the deployed subfolder
  if grep -q "'/frankenstein-transformer/" "$INDEX"; then
    probe PASS asset-base-url "/frankenstein-transformer/ prefix used for schema, examples, paper"
  else
    probe FAIL asset-base-url "expected '/frankenstein-transformer/...' asset URLs"
  fi
fi

# 8) top-level PDF copies match the compiled ones
if [[ -f "$SITE/paper/en.pdf" && -f "$SITE/paper/en/paper.pdf" ]] && ! cmp -s "$SITE/paper/en.pdf" "$SITE/paper/en/paper.pdf"; then
  probe FAIL paper-en-copy "paper/en.pdf != paper/en/paper.pdf"
else
  probe PASS paper-en-copy "en.pdf in sync with en/paper.pdf"
fi
if [[ -f "$SITE/paper/es.pdf" && -f "$SITE/paper/es/paper-es.pdf" ]] && ! cmp -s "$SITE/paper/es.pdf" "$SITE/paper/es/paper-es.pdf"; then
  probe FAIL paper-es-copy "paper/es.pdf != paper/es/paper-es.pdf"
else
  probe PASS paper-es-copy "es.pdf in sync with es/paper-es.pdf"
fi

# ---------------------------------------------------------------------------
# [python-probes] — schema refs + example YAML validity (frankenstein env)
# ---------------------------------------------------------------------------
echo ""
echo "[python-probes]"
conda_ok="no"
if command -v conda >/dev/null 2>&1; then
  conda_ok="yes"
  out=$(conda run -n frankenstein python -c "
import sys
sys.path.insert(0, r'$REPO_ROOT')
try:
    from src.utils.schema_loader import resolve_schema
    resolve_schema(r'$SITE/schema.yaml')
    print('SCHEMA_REFS=OK')
except Exception as e:
    print('SCHEMA_REFS=FAIL: ' + str(e)[:160])
" 2>/dev/null | grep '^SCHEMA_REFS=')
  case "$out" in
    "SCHEMA_REFS=OK") probe PASS schema-refs "site schema.yaml resolves all \$ref pointers" ;;
    "") probe FAIL schema-refs "could not run schema resolution (conda env 'frankenstein' missing python deps?)" ;;
    *) probe FAIL schema-refs "${out#SCHEMA_REFS=FAIL: }" ;;
  esac

  out=$(conda run -n frankenstein python -c "
import glob, os, sys
sys.path.insert(0, r'$REPO_ROOT')
from src.training.config_loader import load_training_config
ok = bad = 0
msgs = []
for p in sorted(glob.glob(os.path.join(r'$SITE', 'examples', '**', '*.yaml'), recursive=True)):
    try:
        load_training_config(p); ok += 1
    except Exception as e:
        bad += 1
        msgs.append(os.path.relpath(p, r'$SITE') + ': ' + str(e)[:110])
print('YAML_OK=%d' % ok)
print('YAML_BAD=%d' % bad)
for m in msgs[:15]:
    print('BAD ' + m)
" 2>/dev/null | grep -E '^(YAML_OK|YAML_BAD|BAD)=')
  yaml_ok=$(sed -n 's/^YAML_OK=//p' <<< "$out")
  yaml_bad=$(sed -n 's/^YAML_BAD=//p' <<< "$out")
  yaml_ok="${yaml_ok:-0}"; yaml_bad="${yaml_bad:-0}"
  if [[ "$yaml_bad" -gt 0 ]]; then
    probe FAIL example-yaml "$yaml_bad of $((yaml_ok + yaml_bad)) site example YAMLs fail load_training_config:"
    grep '^BAD ' <<< "$out" | sed 's/^/  /'
  elif [[ "$yaml_ok" -gt 0 ]]; then
    probe PASS example-yaml "$yaml_ok site example YAMLs load cleanly"
  else
    probe FAIL example-yaml "no site example YAMLs found"
  fi
else
  probe FAIL conda "conda not found; cannot run schema/YAML probes"
fi

# ---------------------------------------------------------------------------
# [estimator] — JS estimator vs committed PyTorch ground truth (fast, no torch)
# ---------------------------------------------------------------------------
echo ""
echo "[estimator]"
if command -v node >/dev/null 2>&1 && [[ -f "$TRUTH" && -f "$ESTIMATOR" ]]; then
  newer=$(find "$REPO_ROOT/configs" "$REPO_ROOT/src" -name "*.yaml" -newer "$TRUTH" 2>/dev/null | wc -l | tr -d ' ')
  echo "truth_freshness=$([[ "$newer" -gt 0 ]] && echo "STALE ($newer configs newer than param_truth.json — regenerate in validation gate)" || echo fresh)"
  est_out=$(node "$REPO_ROOT/full_tests/param_estimate_check.mjs" "$ESTIMATOR" "$TRUTH" --tol 0.01 2>&1)
  est_rc=$?
  summary=$(grep '^tolerance:' <<< "$est_out")
  echo "check: ${summary:-$est_out}"
  if [[ $est_rc -eq 0 ]]; then
    probe PASS estimator-drift "ft-param-estimator.js within 1% of committed ground truth"
  else
    probe FAIL estimator-drift "drift > 1% vs committed ground truth (regenerate truth, then fix estimator):"
    grep '^FAIL' <<< "$est_out" | head -6 | sed 's/^/  /'
  fi
elif [[ ! -f "$TRUTH" ]]; then
  probe FAIL estimator-truth "full_tests/param_truth.json missing (run full_tests/param_count_check.py)"
else
  probe FAIL estimator-run "node or $ESTIMATOR missing"
fi

# ---------------------------------------------------------------------------
# [tools]
# ---------------------------------------------------------------------------
echo ""
echo "[tools]"
for t in node pandoc pdflatex bibtex rsync conda; do
  if command -v "$t" >/dev/null 2>&1; then echo "$t=yes"; else echo "$t=no"; fi
done
echo "conda_env_frankenstein=$conda_ok"

# ---------------------------------------------------------------------------
# [summary]
# ---------------------------------------------------------------------------
echo ""
echo "[summary]"
echo "website_repo=$([[ "$website_present" == "yes" ]] && echo present || echo MISSING)"
echo "drift: diff=$DIFF_N missing_site=$MISSING_SITE only_site=$ONLY_SITE ok=$OK_N"
echo "probes_failed=$PROBE_FAIL"
exit 0
