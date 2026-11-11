# Installation Guide

This guide covers every supported way to install **frankenstein-transformer**:
CPU and CUDA development installs, editable installs, wheels from PyPI, the
optional capability extras, and third-party plugins.

> **TL;DR** — one package installs the whole toolkit, including the component
> registry and the plugin loader (`src/registry/`, `src/plugins/`). Pick
> **exactly one** torch extra (`cpu` *or* one CUDA flavour) plus the
> capabilities you need (`train`, `sbert`, `web`, `wandb`).

---

## 1. Requirements

| Requirement | Notes |
|---|---|
| Python | **≥ 3.10** (`requires-python = ">=3.10"`). CI matrix: 3.10, 3.11, 3.12. |
| torch | Resolved from PyPI by default; the `cpu`/`cu118`/`cu126`/`cu128` extras pin the matching wheel. |
| CUDA driver | Must match the chosen wheel (see §5). A Tesla P40 works with `cu118`. |
| Disk | A CUDA torch wheel is ~2–3 GB; keep ~8 GB free for a full install with extras. |
| `uv` *or* `pip` | `uv` is recommended for development; `pip` works for users and CI. |

---

## 2. Development install (repository clone)

```bash
git clone https://github.com/erickfmm/frankenstein-transformer.git
cd frankenstein-transformer
```

### 2.1 With `uv` (recommended)

| Target | Command |
|---|---|
| CPU only | `uv sync --extra cpu --extra train` |
| CUDA 11.8 (Tesla P40, safest) | `uv sync --extra cu118 --extra train` |
| CUDA 12.6 (driver ≥ 560) | `uv sync --extra cu126 --extra train` |
| CUDA 12.8 (driver ≥ 570) | `uv sync --extra cu128 --extra train` |

The `dev` dependency group (pytest) is synced by default — no extra flag needed.

> **Always repeat the same extras when you run a command.**
> ```bash
> uv run --extra cpu --extra train frankenstein-transformer --help
> uv run --extra cpu --extra train python -m pytest tests/ -v
> ```
> A bare `uv run …` (or `uv run` with different extras) makes uv re-sync and
> swap the torch wheel, which is slow and can break an in-progress session.

### 2.2 With `pip` (editable)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU-only (recommended on machines without a GPU)
pip install -e ".[cpu,train]"

# CUDA 11.8 (P40) / 12.6 / 12.8
pip install -e ".[cu118,train]"
# pip install -e ".[cu126,train]"
# pip install -e ".[cu128,train]"
```

`pip install -e ".[train]"` (without a torch extra) uses the PyPI `torch`
build; on Linux that build is CUDA-enabled and large. Prefer an explicit torch
extra when you know the target.

---

## 3. Installing from PyPI (end users)

```bash
pip install frankenstein-transformer
```

That installs the library, the four CLI subcommands
(`train`, `deploy`, `infer`, `web-server`) **and** the component registry /
plugin machinery. Add capability extras as needed:

```bash
# CPU-pinned torch + the full training stack
pip install "frankenstein-transformer[cpu,train]"

# CUDA 11.8 + training stack
pip install "frankenstein-transformer[cu118,train]"

# Everything optional (wandb is not part of `train`)
pip install "frankenstein-transformer[cpu,train,wandb]"
```

---

## 4. Extras reference

| Extra | Pulls in | When to use |
|---|---|---|
| `cpu` | `torch==2.7.0+cpu` | CPU-only machines, CI, laptops |
| `cu118` | `torch==2.6.0+cu118` | Tesla P40 / pre-Volta GPUs, older drivers |
| `cu126` | `torch==2.7.0+cu126` | Ampere+ with driver ≥ 560 |
| `cu128` | `torch==2.7.0+cu128` | Blackwell/newest GPUs with driver ≥ 570 |
| `sbert` | `sentence-transformers`, `sentencepiece`, `accelerate` | SBERT training/inference |
| `web` | `streamlit` | `frankenstein-transformer web-server` YAML builder |
| `wandb` | `wandb` | Optional experiment tracking (CSV logging works without it) |
| `train` | `sentence-transformers`, `sentencepiece`, `accelerate`, `streamlit`, `soundfile`, `librosa` | Full local dev / audio / SBERT / web (union of `sbert` + `web` + audio decode) |

Rules:

- **The four torch extras are mutually exclusive.** `uv` enforces this via
  `tool.uv.conflicts`; combining two in one `pip` command is unsupported.
- `train` is **not** a torch extra. `--extra cpu --extra train` is the normal
  CPU combination.
- `wandb` is separate on purpose: enabling W&B without the package only emits a
  one-time warning.

---

## 5. Choosing a CUDA wheel

| GPU / driver | Recommended extra | Notes |
|---|---|---|
| No NVIDIA GPU | `cpu` | Smallest install; all tests run on CPU |
| Tesla P40 (sm_61), driver 450–550 | `cu118` | The project's reference GPU; the safest choice for Pascal |
| Ampere/Ada, driver ≥ 560 | `cu126` | Modern default for RTX 30/40 and A-series |
| Ada/Blackwell, driver ≥ 570 | `cu128` | Newest wheels |
| Unknown | Check `nvidia-smi` (driver version) and PyTorch's compatibility table | The driver must be ≥ the CUDA runtime the wheel was built for |

If you install an incompatible wheel you will see either an import-time
`undefined symbol` error or `torch.cuda.is_available() == False`; re-install
with the matching extra.

---

## 6. Plugins

The **plugin system ships inside the main package** — `src/registry/` and
`src/plugins/` are part of the wheel (`packages = ["src"]`), so the install
commands above are all you need for the core.

There are two ways to add a plugin:

### 6.1 Local plugin file — nothing to install

List the module under the top-level `plugins:` key of your YAML:

```yaml
plugins:
  - ./examples/plugins/window_mixer.py     # file path
  # - my_package.my_plugin                # importable module
  # - my_package.my_plugin:register       # module:callable
```

Relative file paths are resolved against the **process working directory**, so
run commands from the repository root or use an absolute path.

### 6.2 Distributable plugin — a separate package

A third-party plugin is its own distribution. It declares the
`frankenstein.plugins` entry point and is discovered automatically:

```toml
[project]
name = "my-frankenstein-plugin"
dependencies = ["frankenstein-transformer>=1.5"]

[project.entry-points."frankenstein.plugins"]
my_plugin = "my_plugin:register"
```

```bash
pip install -e ./my-frankenstein-plugin
```

See [`HOWTO_MAKE_PLUGIN.md`](HOWTO_MAKE_PLUGIN.md) for the complete authoring
contract, family-by-family examples and testing guidance.

> The repository's `examples/` and `configs/` directories are **excluded from
> the sdist/wheel** (`[tool.hatch.build.targets.sdist] exclude`). The example
> plugin and preset are repo-only; a published plugin must ship its own module
> package.

---

## 7. Verify the installation

```bash
frankenstein-transformer --help
frankenstein-transformer train --list-configs | head

python -c "import src.registry, src.plugins; print('registry + plugins: ok')"

python -c "
from src.registry import ensure_builtins, registry, ComponentFamily
ensure_builtins()
for family in ('MIXER', 'OPTIMIZER', 'ACTIVATION', 'NORM', 'RESIDUAL',
               'POSITIONAL_ENCODING', 'EMBEDDING'):
    print(family.lower(), len(registry.names(getattr(ComponentFamily, family))))
"
```

Expected census: **50 mixers, 36 optimizers, 44 activations, 6 norms,
4 residuals, 11 positional encodings (+ the `none` alias), 3 embedding
variants.**

Plugin smoke test:

```bash
python -c "
from src.plugins.loader import load_plugins
from src.registry import registry, ComponentFamily
load_plugins(['./examples/plugins/window_mixer.py'])
print('local_window' in registry.names(ComponentFamily.MIXER))
"
```

Regenerate the assembled schema snapshot (used by the website mirror and CI):

```bash
python -m src.registry.snapshot --out docs/schema.assembled.yaml
```

---

## 8. Run the test suite

CPU-only, as CI does:

```bash
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

Single file / single test:

```bash
uv run --extra cpu --extra train python -m pytest tests/test_registry_core.py -v
uv run --extra cpu --extra train python -m pytest tests/test_plugin_loader.py::PluginPathTests -v
```

The suite never requires CUDA; GPU-specific code paths are guarded.

---

## 9. Build a wheel or sdist

```bash
uv build                 # or: python -m build
```

- The **wheel** contains the whole `src` package tree, including
  `src/registry/**` and `src/plugins/**`.
- The **sdist** excludes `docs/`, `tests/`, `configs/`, `examples/`,
  `full_tests/`, `.opencode/`, `AGENTS.md`, etc. (see
  `[tool.hatch.build.targets.sdist]` in `pyproject.toml`).
- The version comes from git tags via `hatch-vcs` (fallback `0.0.0` outside a
  repository).

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `uv run` suddenly re-downloads torch | Extras not repeated on the command; always use `uv run --extra cpu --extra train …` |
| `ModuleNotFoundError: pip` in a uv venv | uv venvs omit pip by design; use `uv pip install …` or `uv add …` |
| `ImportError: … undefined symbol` from torch | CUDA wheel/driver mismatch; reinstall with the matching torch extra |
| `torch.cuda.is_available()` is `False` | CPU wheel installed, or driver too old for the wheel |
| Conflict error about multiple torch indexes | Two torch extras selected; keep exactly one |
| `datasets` audio decode fails (`soundfile`, `torchcodec`) | Install the `train` extra (pins `datasets<4` + `soundfile` + `librosa`) |
| `streamlit` missing for `web-server` | Install the `web` or `train` extra |
| SBERT training refuses to start (`accelerate`) | Install the `sbert` or `train` extra |
| Plugin not found (`Unknown mixer … plugin missing`) | The module is not installed/listed; check `plugins:` paths (relative to CWD) or the entry point |
| `__pycache__` committed by mistake | It is ignored in the main repo; the website mirror also ignores it |

---

## 11. Minimal CI recipe

```yaml
- uses: actions/setup-python@v5
  with:
    python-version: "3.12"
- run: pip install torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu
- run: pip install -e . --no-deps
- run: pip install pytest
- run: python -m pytest tests/ --continue-on-collection-errors -q
```
