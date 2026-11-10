# Plan: Plugin-Based Extensibility — Declarative Component Registry

> **Status**: Deep implementation plan, produced in plan mode. No code has been
> changed. Execution checklist in §9; verification matrix in §10.
>
> **User decisions locked in** (Q&A before this plan):
> 1. **Layered plugin loading** — Python entry-points (pip-installable) **and** a
>    `plugins:` key in the YAML config **and** in-repo built-ins registered
>    through the exact same API.
> 2. **Full unification** — every built-in becomes a first-party component
>    registered through the same spec objects. No parallel dispatch paths.
> 3. **Registry-assembled schema** — component specs own their schema
>    fragments; `resolve_schema` merges core YAML + all registered fragments
>    before validation. `additionalProperties: false` is preserved everywhere.
> 4. **Namespaced params** — per-component parameters live in per-family
>    parameter dicts; there are no more per-component flat dataclass fields.
> 5. **Scope: all seven families** — mixers, optimizers, norms, activations,
>    residuals, positional encodings, token embeddings.
> 6. **Registry-derived + extended test gates** — same invariants, sourced from
>    specs; new plugin-contract tests added.
> 7. **No backward compatibility required** — and **the flattener
>    (`src/utils/config_flatten.py`) is deleted**. The nested YAML schema shape
>    becomes the runtime config shape. Legacy flat `config.json` checkpoints are
>    not loadable (optional one-shot migration script included in §9, P6).

---

## 1. Problem statement (why)

Adding **one optimizer** today touches 19 files (see
`.opencode/plan/add-optimizer-2604.09967.md` §4 for the full 19-row table): the
class, `optimizer/__init__.py`, `factory.py` (registry dict +
`OPTIMIZER_ALLOWED_KEYS` + one hand-written branch in a ~370-line `if/elif`
chain), `_optimizer.yaml` (enum + 2 bilingual use-case tables + 2 suffix bullet
lists + `by_optimizer` entry + references), `_training.yaml` (one `if/then`
prefix rule), `_sbert.yaml`, an example YAML, `configs/README.md`,
`full_tests/run_e2e.py`, `tests/test_optimizer_factory.py`, plus bibliography
and LaTeX census passes.

Adding **one mixer** touches ~10 places (AGENTS.md "Extending" table, Mixer row;
`docs/architecture.md` §11.1): implementation + 2 package `__init__`s, the
`mixer_registry` dict **rebuilt inside `HybridLayer.__init__` on every layer
instantiation** (`src/model/hybrid_layer.py:204-253`), `_dims.yaml`
`layer_pattern.items.enum` + 2 bilingual tables, an `_attention_<mixer>.yaml`
fragment + `_model.yaml` `$ref`, `_ATTENTION_MIXER_RENAMES` in
`config_flatten.py:79-209`, flat fields in `model/config.py` (1,913 lines
total), a `CacheSpec` + `IMPLEMENTED_MIXERS` entry, QK-norm support sets,
example YAML, tests.

The same component name is re-enumerated in up to **9 parallel lists**:

| # | List | File |
|---|------|------|
| 1 | runtime registry dict | `hybrid_layer.py` / `factory.py` / `factory.py` (norm) |
| 2 | schema enum | `_dims.yaml` / `_optimizer.yaml` / `_norm.yaml` / `_ffn.yaml` / `_residuals.yaml` / `_positional_encoding.yaml` |
| 3 | bilingual doc tables (EN+ES) | same schema files (scanned by `test_schema_consistency.py`) |
| 4 | optimizer suffix bullet lists (EN+ES) + `by_optimizer` | `_optimizer.yaml:201-…` |
| 5 | `if/then` prefix rules | `_training.yaml` (~36 blocks) |
| 6 | flatten renames | `config_flatten.py` `_ATTENTION_MIXER_RENAMES`, `_flatten_pe_parameters` |
| 7 | flat dataclass fields | `model/config.py` (one or more per component) |
| 8 | defaults parity list | `tests/test_schema_consistency.py::_DEFAULT_PAIRS` (115+ entries) |
| 9 | derived capability sets | `cache/capabilities.py` `CACHE_SPECS` (49) + `IMPLEMENTED_MIXERS` (38), `norm/qk.py` `QK_NORM_MIXERS`/`QK_CLIP_MIXERS`, `HybridLayer.TRAINING_FREE_LAYERS`, Streamlit `prefix_map` (`streamlit_gui/app.py:371-396`), `run_e2e.py` `OPTIMIZERS` list |

Every addition/renaming must keep all 9 lists in sync by hand — this is the
stated source of dispatch bugs and maintainer friction.

**Current census** (must be preserved by the refactor — no component is added
or removed): 49 mixers (48 + `mamba`), 36 optimizers, 6 norms, 43 activations
(40 elementwise + 3 GLU), 4 residual strategies, 12 positional encodings,
3 token-embedding variants (standard/factorized/dynamic).

---

## 2. Target architecture

### 2.1 The declarative unit: `ComponentSpec`

One frozen dataclass per component, colocated with the implementation module.
It is the **single source of truth** for everything the 9 lists used to hold
separately.

```python
# src/registry/spec.py
class ComponentFamily(str, Enum):
    MIXER = "mixer"
    OPTIMIZER = "optimizer"
    NORM = "norm"
    ACTIVATION = "activation"
    RESIDUAL = "residual"
    POSITIONAL_ENCODING = "positional_encoding"
    EMBEDDING = "embedding"

@dataclass(frozen=True)
class ComponentSpec:
    """Declarative registration record for one pluggable component."""
    name: str                          # registry key; snake_case, == optimizer prefix
    family: ComponentFamily
    builder: LazyBuilder               # "module:Attr" string or zero-arg callable; resolved+cached lazily
    description: str                  # EN row text for the schema tables (docs parity gate reads this)
    description_es: str                # ES row text (same gate)
    references: tuple[str, ...] = ()   # arxiv ids -> "(paper: arxiv:XXXX.XXXXX)" lines in schema docs
    params_schema: SchemaFragment | None = None
    # params_schema: inline JSON-Schema dict for the component's params
    # sub-object, OR a ("path.yaml", "#/json/pointer") reference to a fragment
    # file (built-ins keep their existing _attention_*.yaml fragments).

    # --- optional declarative hooks (family-specific subclasses below) ---
    def bind(self, raw: Mapping[str, Any] | None) -> ComponentParams: ...
    def build(self, *args, **kwargs): ...   # resolves builder, injects bound params last


@dataclass(frozen=True)
class MixerSpec(ComponentSpec):
    training_free: bool = False        # old TRAINING_FREE_LAYERS (fasa, sparge)
    needs_pos_encoder: bool = True      # default for positional_encoding_parameters.use_pe.<name>
    cache: CacheSpec | None = None      # moved verbatim from cache/capabilities.py
    cache_implemented: bool = False     # old IMPLEMENTED_MIXERS membership
    qk_norm_supported: bool = False     # old QK_NORM_MIXERS membership
    qk_clip_supported: bool = False     # old QK_CLIP_MIXERS membership
    aliases: tuple[str, ...] = ()      # e.g. ("none",) -> "nope" PE alias


@dataclass(frozen=True)
class OptimizerSpec(ComponentSpec):
    # replaces OPTIMIZER_REGISTRY + OPTIMIZER_ALLOWED_KEYS + the if/elif chain
    param_bindings: Mapping[str, ParamBinding] = field(default_factory=dict)
    # suffix (without the "<name>-" prefix) -> ParamBinding(coerce: to_float|to_int|bool|str, default)
    supports_shared_per_group_suffixes: bool = True   # old by_optimizer flag
    group_policy: str = "adaptive"      # "adaptive" (lr/wd/betas/eps) | "sgd" (lr/wd/momentum only)
    closure_based: bool = False         # eve, ea4llm — consumed by trainer
    sbert_allowed: bool = True
    requires_grad_accum_one: bool = False  # eve, ea4llm — load-time check
    custom_build: Callable | None = None   # escape hatch (adafactor eps tuple, seed passthrough)


@dataclass(frozen=True)
class NormSpec(ComponentSpec):
    supports_ffn_fusion: bool = False  # flash_norm Prop. 2 fusion hook in HybridLayer
    reads_partial_ratio: bool = False  # prms_norm / flash_norm read norm.partial_ratio


@dataclass(frozen=True)
class ActivationSpec(ComponentSpec):
    is_glu: bool = False               # swiglu/geglu/reglu -> handled by GatedFFN, not get_activation


@dataclass(frozen=True)
class ResidualSpec(ComponentSpec): ...          # builder receives the residual sub-config

@dataclass(frozen=True)
class PESpec(ComponentSpec):
    additive_bias: bool = False        # alibi/pape-style additive interface (documentation)

@dataclass(frozen=True)
class EmbeddingSpec(ComponentSpec): ...         # standard/factorized/dynamic token embedding
```

`CacheSpec` moves **verbatim** from `src/model/cache/capabilities.py:40-61`
into `spec.py` (frozen dataclass, same fields: `supported`, `slot_type`,
`slot_types`, `extras`, `notes`).

### 2.2 The registry

```python
# src/registry/registry.py
class ComponentRegistry:
    """Per-family spec maps. Torch-free, import-cheap, process-global."""
    def register(self, spec) -> None:       # DuplicateComponentError on collision
    def get(self, family, name) -> ComponentSpec   # UnknownComponentError
                                           # message lists known names + "is a plugin missing?"
    def names(self, family) -> tuple[str, ...]
    def specs(self, family) -> tuple[ComponentSpec, ...]
    def property_set(self, family, attr) -> frozenset[str]  # derived views, §2.8
    def override(self, spec, *, force=False) -> None  # explicit, logged; plugins may not silently shadow built-ins

registry = ComponentRegistry()   # module-level singleton
register_spec = registry.register  # used by family packages and plugins
```

Rules baked into tests:

- `name` is validated (`^[a-z][a-z0-9_]*$`; optimizers additionally must be
  valid key prefixes — already enforced by the pattern `^[a-z0-9_]+-`).
- Re-registering the **identical** spec is idempotent (module reimports under
  `pytest` / Streamlit reloads); a different spec with the same
  `(family, name)` raises `DuplicateComponentError`.
- The registry module **never imports torch or any model package** (CPU-first;
  schema work and plugin loading must run in torch-free contexts). Builders are
  lazy `"module:Attr"` strings, resolved on first `build()` and cached.

### 2.3 Schema assembly (D1 preserved: schema stays the contract)

`resolve_schema()` (`src/utils/schema_loader.py`) gains one final step:
`assemble_schema(schema)` from `src/registry/schema_assembly.py`. The hand-written
YAML files keep all *structural* content and get small marker keys where the
registry is the source of truth. Assembly replaces the marker with
registry-derived content **before** any validation, so the fully-assembled
document is still `additionalProperties: false` — now uniformly for built-ins
**and** plugins.

| Marker (in YAML) | Registry content injected |
|---|---|
| `x_registry_enum: mixers` on `dims.layer_pattern.items` | `enum` ← `registry.names(MIXER)` + `mamba`; bilingual per-mixer table appended to `description`/`description_es` from spec `description`/`description_es` |
| `x_registry_properties: mixers` on `model.attention` | one `properties.<name>` entry per mixer from its `params_schema` fragment (mixers without knobs get none, exactly as today's comment at `_model.yaml:50-54` documents) |
| `x_registry_enum: optimizers` on `optimizer_class` | enum + bilingual use-case table + references from optimizer specs |
| `x_registry_by_optimizer: true` on `parameters` | the whole `x_parameter_reference.by_optimizer` map generated from optimizer specs (prefix, `supports_shared_per_group_suffixes`, `specific_global_suffixes` = `param_bindings.keys()`); EN/ES suffix bullet lists appended to `description` from spec descriptions |
| `x_registry_if_then: optimizers` (an `allOf` slot in `_training.yaml`) | one generated `if/then` block per optimizer (`const: <name>` → `patternProperties: {^<name>-: …}`) — the ~36 hand-written blocks are deleted |
| `x_registry_enum: norms` on `norm.type` | enum + bilingual bullets from norm specs |
| `x_registry_enum: activations` on `ffn.activation` | enum + GLU handling from activation specs |
| `x_registry_enum: residuals` on `residuals.type` | enum + bullets |
| `x_registry_enum: positional_encodings` on `positional_encoding` | enum + bullets; `positional_encoding_parameters` per-PE sub-objects from PESpec fragments; `use_pe` per-mixer flags generated from `MixerSpec.needs_pos_encoder` defaults |
| `x_registry_enum: embeddings` (new `embedding.type` key, §2.7) | enum + variant fragments |
| `x_registry_mixin: attention_fragments` (plugin fragments only differ in origin — same code path) | — |

Assembly is **idempotent and total**: every marker found must resolve, and the
assembled snapshot must contain exactly the union of core + registry content.
The schema consistency tests (§7) flip from "hand-written YAML is consistent
with itself" to "assembled schema is consistent with the registry" — same
invariants, now *automatic*.

**Schema snapshot export**: new CLI subcommand
`frankenstein-transformer export-schema [--out schema.assembled.yaml]`
(`src/registry/snapshot.py`) writes the fully assembled schema to a file.
Consumers:
- the website mirror (`.opencode/command/sync-website.md` update: copy the
  assembled snapshot instead of raw `src/schema.yaml`, so the public site
  keeps serving a complete closed schema);
- a committed diff artifact for CI (`docs/source/_static/schema.assembled.yaml`)
  reviewed like any generated file;
- plugin authors debugging their fragments.

### 2.4 Runtime config restructure (the flattener dies)

`FrankensteinModelConfig` (`src/model/config.py`, 1,913 lines) becomes a
**nested** dataclass tree mirroring the YAML `model:` block (plus the
top-level `image:`/`audio:` blocks that used to be merged flat by
`flatten_image_dict`/`flatten_audio_dict`):

```python
@dataclass
class FrankensteinModelConfig:
    dims: DimsConfig                     # vocab_size, hidden_size, num_layers, num_loops,
                                        # num_heads, num_kv_heads, dropout, layer_pattern, mode
    norm: NormConfig                     # type, partial_ratio, flashnorm_partial_ratio,
                                        # qk_norm: QKNormConfig
    embedding: EmbeddingBlockConfig     # type (NEW enum, §2.7), factorized, dynamic, conv
    attention: Dict[str, Dict[str, Any]]        # per-mixer raw params; validated by assembled
                                                # schema, bound via MixerSpec at build time
    positional_encoding: str
    positional_encoding_parameters: PpeParamsConfig  # per-PE dicts + use_pe flags
    moe: MoEConfig
    bitnet: BitnetConfig
    mixture_of_depths: ModConfig
    ssmax: SsmaxConfig
    numerics: NumericsConfig
    ffn: FfnConfig                      # hidden_size, activation, activation_config: dict
    classification_head: ClassificationHeadConfig
    mhc: MhcConfig
    residuals: ResidualsConfig
    image: ImageConfig | None = None     # was flatten_image_dict
    audio: AudioConfig | None = None     # was flatten_audio_dict

    @classmethod
    def from_dict(cls, model_block: dict, *, image_block=None, audio_block=None) -> "FrankensteinModelConfig": ...
    def __post_init__(self): ...         # same invariants: hidden_size % num_heads == 0,
                                        # num_kv_heads | num_heads, len(layer_pattern) == num_layers,
                                        # mHC XOR MoD, ffn activation enum, embedding.type checks…
```

Consequences:

- **`src/utils/config_flatten.py` (819 lines) is deleted** in its entirety:
  `_ATTENTION_MIXER_RENAMES`, `_flatten_pe_parameters`, `_is_nested_shape`,
  `flatten_model_dict`, `flatten_image_dict`, `flatten_audio_dict`. The
  tolerance path that accepted *legacy flat* input (old checkpoint
  `config.json`) is removed — no retrocompat per the user's decision.
- **Every per-component flat field in `config.py` is deleted** (~800 lines):
  `mla_latent_rank`, `gqla_*`, `mlra_*`, `tucker_*`, `iha_*`, `gta_*`,
  `mtla_*`, `cca_*`, `ccgqa_*`, `msa_*`, `sparda_*`, `lsa_*`, `engram_*`,
  `gma_*`, `ssog_*`, `swa_*`, `ca_*`, `fpa_*`, `fastconformer_*`,
  `conformer_*`, `falcon_*`, `ode_solver/steps`, `retention_heads`, PE flat
  fields (`rope_base`, `hope_*`, `alibi_num_heads`, `bam_*`, `pape_*`,
  `sinusoidal_*`, `learned_*`), embedding flags
  (`use_factorized_embedding`, `use_dynamic_embedding`, `use_embedding_conv`, …),
  activation knobs (`ffn_activation_config` stays as a **dict**), residual
  flags (`full_attn_*`, `block_attn_*`, `attnres_*`).
- **Components read bound params, not config attributes.** The complete
  old-flat → nested mapping is the inverse of the tables in
  `config_flatten.py:9-63` (docstring) + `docs/architecture.md` §4 "Flattening
  contract" (lines 157-192) + `tests/test_schema_consistency.py::_DEFAULT_PAIRS`
  (lines 153-267). `config.py` keeps: core dims, task/runtime fields consumed
  outside the model (`vocab_size` is in `dims`), and the `__post_init__`
  invariants listed above.
- **Checkpoints / export change shape**: `config.json` stores the nested block
  (`deploy.py:220` `self.config.__dict__.copy()` now serializes nested
  sub-dataclasses — add a small `to_jsonable()` that `asdict`s sub-configs);
  `deploy/transformers_export.py` (which already walks the config,
  `transformers_export.py:532`) maps nested fields. Legacy flat checkpoints are
  not loadable; §9 P6 includes an optional one-shot
  `scripts/migrate_flat_config.py` (the deleted flattener's tables are its
  inverse mapping) for the user's existing artifacts.

### 2.5 Param binding (`src/registry/binding.py`)

```python
@dataclass(frozen=True)
class ComponentParams:
    """Attribute-access view over one component's params (built-ins AND plugins)."""
    name: str
    _values: Dict[str, Any]
    def __getattr__(self, key): ...   # KeyError with the component's allowed keys

def bind_params(spec, raw: Mapping | None) -> ComponentParams:
    """Validate + coerce a raw params dict against spec.params_schema.

    - unknown keys: hard error (the assembled schema already rejected them at
      load time; this is the second line of defense for programmatic use);
    - missing keys: filled from the fragment's `default`;
    - coercion by JSON-Schema type: integer->int, number->float,
      boolean->bool, string/enum->str, null->None (optional ranks keep their
      explicit-null semantics, tests/test_schema_consistency.py::test_optional_ranks_accept_explicit_null).
    """
```

This replaces, uniformly: `getattr(config, "mla_latent_rank", …)` in mixers,
the optimizer `scoped.get(...)` + `to_float/to_int/to_betas` ladder, the
activation `p.get("raf_degrees", …)` reads, PE `getattr(config, "rope_base", …)`
reads, residual `getattr(config, "full_attn_init_query_zero", …)` reads.

### 2.6 Optimizer factory: the if/elif chain becomes data

`build_optimizer` (`src/model/optimizer/factory.py`, 569 lines) shrinks to:

```python
def build_optimizer(optimizer_class, param_groups, parameters) -> Optimizer:
    spec = registry.get(ComponentFamily.OPTIMIZER, str(optimizer_class).strip().lower())
    scoped = extract_prefixed_parameters(spec.name, parameters or {})
    ensure_no_unknown_parameters(spec.name, scoped, spec.allowed_keys())  # shared per-group keys | param_bindings
    groups = build_param_groups(param_groups, scoped, spec)   # group_policy: adaptive adds betas/eps, sgd does not
    kwargs = {k: b.coerce(scoped.get(k), b.default) for k, b in spec.param_bindings.items()}
    if spec.custom_build is not None:
        return spec.custom_build(spec, groups, scoped, kwargs)
    return spec.builder()(groups, **kwargs)
```

Each of the 36 optimizer modules grows a `SPEC = OptimizerSpec(...)` with its
`param_bindings` — a mechanical transcription of its current `if/elif` branch
(e.g. `muon2.py` gets `ns_steps: (to_int, 5)`, `ns_eps: (to_float, 1e-7)`,
`factored: (bool, False)` — exactly the branch at `factory.py:493-502`). The
few genuinely special cases keep a `custom_build`:
- `adafactor` (per-group `eps` becomes a tuple; `factory.py:285-296`),
- `sgd_momentum` (group policy already declarative),
- `eve`/`ea4llm` (`seed` None-passthrough; `factory.py:436-472`) + their
  capabilities `closure_based=True`, `requires_grad_accum_one=True`,
  `sbert_allowed=False` (consumed by `config_loader` + trainer as today).

`OPTIMIZER_REGISTRY` and `OPTIMIZER_ALLOWED_KEYS` dicts are deleted;
module-level compatibility shims **are not kept** (user decision #7 — tests are
rewritten instead). `extract_prefixed_parameters`, `ensure_no_unknown_parameters`,
`parse_group_value`, `to_*` stay in `optimizer/base.py` untouched.

### 2.7 Family dispatch, family by family

| Family | Today | After |
|---|---|---|
| **Mixers** | `mixer_registry` dict rebuilt per layer inside `HybridLayer.__init__` (`hybrid_layer.py:204-263`); `mamba` special-cased; `TRAINING_FREE_LAYERS` set (`:126`) | module-level lookup: `spec = registry.get(MIXER, layer_type)`; `self.mixer = spec.build(config, pos_encoder=pos_encoder, params=spec.bind(config.attention.get(name)))`. `mamba` becomes a real `MixerSpec` with a custom builder (`proj_cls(config.hidden_size, config.hidden_size)`, preserving the BitLinear switch at `:188-189`). `TRAINING_FREE_LAYERS` remains as a **class-level derived property** `frozenset(registry.property_set(MIXER, "training_free"))` for docs/tests. The training-free dense-surrogate path (`_training_free_dense_forward`) stays — it consults the property |
| **Optimizers** | §2.6 | §2.6 |
| **Norms** | if/elif in `norm/factory.py:44-56`; FlashNorm fusion hard-coded in `HybridLayer._build_ffn` (`hybrid_layer.py:279-344`) | `get_norm(config)` → `spec.build(config, params=spec.bind(...))`; `NormSpec.supports_ffn_fusion` replaces the hard-coded `config.norm_type == "flash_norm"` check. `layer_norm` (the default) is a spec whose builder is `nn.LayerNorm` |
| **Activations** | 40-branch if/elif in `activation_function/factory.py:126-231`; `GLU_VARIANTS` set (`:61`); `_validate_ffn_activation_config` in `config.py:37-88` | `ActivationSpec` per activation with `param_bindings` for its `activation_config` keys; `get_activation` becomes registry lookup + `spec.bind(config.ffn.activation_config)`; GLU specs carry `is_glu=True` and `get_activation` keeps raising for them (`GatedFFN` path unchanged). `config.py` validation of `activation_config` is replaced by binding-time validation from the specs' `params_schema` |
| **Residuals** | if/elif in `residuals/factory.py:59-93`; `RESIDUAL_TYPES` tuple | `ResidualSpec` per strategy; `build_residual` = lookup + `spec.build(config, params=…)`. `set_streams` + `use_fixed_softmax` post-processing stays in the factory |
| **Positional encodings** | if/elif in `embeddings/factory.py:48-138`; per-PE flat fields; `use_pe` flags hand-listed in `_positional_encoding.yaml` | `PESpec` per PE (12, including the `none`/`nope` alias pair — handled as one spec with `aliases=("none",)`); `build_pos_encoder` = lookup + bind from `config.positional_encoding_parameters`. The SSMax attach (`:140-149`) stays as transversal post-processing |
| **Token embeddings** | flag-based if/elif in `frankenstein_encoder.py:116-121` (`use_dynamic_embedding` / `use_factorized_embedding` / `nn.Embedding`) | **new `embedding.type` enum** (`standard` \| `factorized` \| `dynamic`) replacing the two booleans (breaking change accepted); `EmbeddingSpec` per variant; the encoder's `emb` dispatch = lookup. `embedding.factorized.*` / `embedding.dynamic.*` params stay in the block, bound via the spec. The conv frontend sub-block stays structural |

Model classes, MoE, mHC/Hyperloop, datasets/tasks, the audio frontend and the
ViT patch embedding stay **outside** the registry (they are structural, with
deep cross-cutting rules — AGENTS.md rules 6/8/11). The design does not
preclude adding families later; it deliberately does not do it now.

### 2.8 Derived views replace the capability sets

All remaining hand-lists become one-line registry views, and the old modules
keep thin derived aliases so existing imports in tests/docs keep working:

| Old list | New source |
|---|---|
| `HybridLayer.TRAINING_FREE_LAYERS` (`hybrid_layer.py:126`) | `registry.property_set(MIXER, "training_free")` |
| `CACHE_SPECS` / `IMPLEMENTED_MIXERS` / `is_cacheable` / `spec_for` (`cache/capabilities.py`) | `MixerSpec.cache` / `.cache_implemented`; `capabilities.py` becomes a thin derived module re-exporting the same names built from the registry (KV code and `config_loader._warn_cache_support` keep their import surface) |
| `QK_NORM_MIXERS` / `QK_CLIP_MIXERS` (`norm/qk.py:46-82`) | `registry.property_set(MIXER, "qk_norm_supported")` / `("qk_clip_supported")`; `config_loader._validate_qk_norm` (`:197-...`) consumes the registry directly |
| `OPTIMIZER_REGISTRY` / `OPTIMIZER_ALLOWED_KEYS` | `registry.names(OPTIMIZER)` / `spec.allowed_keys()` |
| `GLU_VARIANTS` / `ALL_ACTIVATIONS` / `ELEMENTWISE_ACTIVATIONS` (`activation_function/factory.py:61,241-251`) | `registry.property_set(ACTIVATION, "is_glu")` / `registry.names(ACTIVATION)` (minus GLU) — keep the names as derived views |
| `RESIDUAL_TYPES` (`residuals/factory.py:26`) | `registry.names(RESIDUAL)` |
| Streamlit `prefix_map` (`app.py:371-396`) | deleted — prefix == spec.name by construction; `render_optimizer_section` reads specific suffixes from the assembled `by_optimizer` entry |
| `run_e2e.py` `OPTIMIZERS` (`:289`) | `sorted(registry.names(OPTIMIZER))`; `_with_optimizer` drives its specific globals from `spec.param_bindings` (with a small per-optimizer override dict where the e2e needs non-default values) |

### 2.9 Plugin loading (layered)

```python
# src/plugins/loader.py
def load_plugins(config_paths: list[str] | None = None) -> None:
    """Populate the registry from all three layers (idempotent).

    1. Entry-points: every distribution advertising the group
       "frankenstein.plugins" — each entry point is a callable that receives
       the registration API and registers its specs. Load failures are logged
       WARNING + skipped (third-party code must not break core startup).
    2. Config-declared paths: the top-level `plugins:` YAML key (new; list of
       module paths or "pkg.module" import strings). Loaded at
       `load_training_config`/`load_inference_config` time BEFORE schema
       assembly, so plugin keys validate in the same pass. Load failure is a
       HARD error (the config explicitly asked for it).
    3. In-repo built-ins: registered at import of each family package
       (§2.10) — same API, no special casing.
    """
```

- New top-level schema key `plugins: [str]` (`src/schema.yaml` +
  `_training.yaml` cross-rule: plugin modules are loaded before validation of
  the rest).
- Public authoring API: `src/plugins/api.py` re-exports
  `register_mixer / register_optimizer / register_norm / register_activation /
  register_residual / register_posenc / register_embedding` (thin wrappers over
  `registry.register` with the right spec class + family validation), so a
  plugin file is:

```python
# my_plugins/window_mixer.py  — example plugin shipped as examples/plugins/
from src.plugins.api import register_mixer

register_mixer(
    name="local_window",
    builder="my_plugins.window_mixer.LocalWindowAttention",   # lazy
    description="Local windowed attention (example plugin).",
    description_es="Atención por ventana local (plugin de ejemplo).",
    params_schema={
        "type": "object", "additionalProperties": False,
        "properties": {"window_size": {"type": "integer", "minimum": 1, "default": 128}},
    },
    needs_pos_encoder=True,
    cache=None,          # unsupported -> legacy full-recompute path (D5)
)
```

  and a preset uses it via `plugins: [./my_plugins/window_mixer.py]` +
  `layer_pattern: [local_window]` — validating strictly, rendering in the
  Streamlit builder, and failing with a clear `UnknownComponentError` listing
  known names when the plugin is absent.

- `pyproject.toml`: no changes required for consumers (plugin packages declare
  their own `[project.entry-points."frankenstein.plugins"]`); the repo's own
  example plugin is not installed by default.

### 2.10 Built-in registration flow (no central imports file)

Each family package registers on import, colocated with implementations:

```python
# src/model/optimizer/muon2.py  (bottom of module)
SPEC = OptimizerSpec(
    name="muon2",
    family=ComponentFamily.OPTIMIZER,
    builder="src.model.optimizer.muon2:Muon2",
    description="Adam-style second-moment preconditioning of the momentum before Newton-Schulz; …",
    description_es="Precondicionamiento del momentum con segundo momento …",
    references=("2604.09967",),
    param_bindings={"ns_steps": ParamBinding(to_int, 5),
                    "ns_eps": ParamBinding(to_float, 1e-7),
                    "factored": ParamBinding(bool, False)},
)

# src/model/optimizer/__init__.py  (already imports every module)
from ..registry import register_spec
for _mod in (adamw, adafactor, …, muon2, …):
    register_spec(getattr(_mod, "SPEC"))
```

Mixers: each family dir (`attention/gated|latent|sparse`, top level) does the
same in its `__init__.py`; `hybrid_layer.py` stops importing 48 classes
(its import block `:21-83` shrinks to the registry import). Cache specs move
into the same `SPEC` (the `CacheSpec(...)` bodies currently in
`capabilities.py:113-…` are transcribed verbatim into the specs).

---

## 3. What is deleted (net line-count sanity)

| Deleted | Lines (approx) |
|---|---|
| `src/utils/config_flatten.py` (whole file) | 819 |
| per-component flat fields + their `__post_init__` validation in `config.py` | ~800 |
| optimizer `if/elif` dispatch + `OPTIMIZER_REGISTRY` + `OPTIMIZER_ALLOWED_KEYS` (`factory.py`) | ~500 |
| activation dispatch chain (`factory.py:126-231`) | ~110 |
| `get_norm` / `build_residual` / `build_pos_encoder` if/elif bodies | ~200 |
| `hybrid_layer.py` per-layer `mixer_registry` + 60-line import block | ~120 |
| `_training.yaml` per-optimizer `if/then` blocks | ~36 blocks |
| `_optimizer.yaml` hand tables / suffix lists / `by_optimizer` bodies; `_dims.yaml` bilingual tables; `_norm.yaml`/`_ffn.yaml`/`_residuals.yaml`/`_positional_encoding.yaml` enum+bullet bodies | several hundred |
| `test_schema_consistency.py::_DEFAULT_PAIRS` | ~115 |
| Streamlit `prefix_map`; `run_e2e.py` OPTIMIZERS list | ~40 |
| **Added** (`src/registry/` ~700, `src/plugins/` ~250, per-family SPEC blocks ~36+49+6+43+4+12+3 ≈ 150 lines of specs, example plugin ~80, snapshot/migration scripts ~200, new tests ~600) | +~2,000 |

The win is not raw line count — it is **one source of truth per component** and
a core dispatch that stops growing. The `add-optimizer` checklist collapses
from 19 files to **3 code files** (module+SPEC, `__init__` auto, example YAML)
+ optional docs (annex/census stay a *documentation* concern for core research
components, not a registration requirement).

---

## 4. File-by-file plan

### 4.1 New files

| # | File | Contents |
|---|---|---|
| N1 | `src/registry/__init__.py` | public API: `registry`, `ComponentFamily`, spec classes, `bind_params`, `ComponentParams`, errors |
| N2 | `src/registry/spec.py` | `ComponentSpec` + family subclasses + `CacheSpec` (moved from `cache/capabilities.py`) + `ParamBinding` |
| N3 | `src/registry/registry.py` | `ComponentRegistry`, singleton, duplicate/unknown errors, `property_set` |
| N4 | `src/registry/binding.py` | `bind_params` / `ComponentParams` (coercion per §2.5) |
| N5 | `src/registry/schema_assembly.py` | `assemble_schema()` + marker handling + fragment loading (file refs + inline dicts) |
| N6 | `src/registry/snapshot.py` | assembled-schema export (CLI hook in `cli.py`) |
| N7 | `src/plugins/__init__.py` / `api.py` / `loader.py` / `errors.py` | §2.9 |
| N8 | `examples/plugins/window_mixer.py` + `configs/examples/plugin_window_mixer.yaml` | example plugin + preset exercising the `plugins:` key (smoke-gated like every example, D8) |
| N9 | `scripts/migrate_flat_config.py` | optional one-shot legacy flat `config.json` → nested converter (inverse tables of the deleted flattener) |
| N10 | `tests/test_registry_core.py`, `tests/test_registry_binding.py`, `tests/test_schema_assembly.py`, `tests/test_plugin_loader.py` | §7 |

### 4.2 Modified files — core pipeline

| # | File | Change |
|---|---|---|
| M1 | `src/utils/schema_loader.py` | `resolve_schema()` tail-calls `assemble_schema` (§2.3); keep the `$ref` resolver as-is |
| M2 | `src/model/config.py` | restructure per §2.4; delete flat fields; keep invariants; `from_dict`/`to_jsonable` |
| M3 | `src/training/config_loader.py` | drop flattener calls; call `load_plugins(config["plugins"])` before validation; `_validate_qk_norm` + `_warn_cache_support` read the registry; embedding.type checks; keep the unknown-key WARNING mechanism (now against the assembled schema) |
| M4 | `src/engine.py` | construct nested config; no behavioral change otherwise |
| M5 | `src/model/hybrid_layer.py` | §2.7 mixers row; import block shrinks; fusion hook from `NormSpec`; FlashNorm pair logic unchanged |
| M6 | `src/model/optimizer/factory.py` | §2.6 (keep `base.py` helpers; `build_optimizer` generic) |
| M7 | `src/model/optimizer/*.py` (36 modules) + `__init__.py` | add `SPEC` + registration loop |
| M8 | `src/model/norm/factory.py` + `norm/*.py` | registry; `qk.py` sets become derived views |
| M9 | `src/model/activation_function/factory.py` + family modules | registry; `GLU_VARIANTS` derived |
| M10 | `src/model/residuals/factory.py` + modules | registry |
| M11 | `src/model/embeddings/factory.py` + PE modules | registry; token-embedding specs live beside the embedding modules |
| M12 | `src/model/frankenstein_{encoder,decoder,vit,audio}.py` | nested config access + `emb` dispatch via `embedding.type`; sweep all `getattr(config, …)` per the §2.4 mapping |
| M13 | `src/model/attention/**` (48 mixer modules) | read bound params (`spec.bind`) instead of flat config attrs; `__mixer_key__` set by `HybridLayer` as today (`cache/hooks.py:52`); dense_forward contract unchanged |
| M14 | `src/model/cache/capabilities.py` | thin derived module (same public names) |
| M15 | `src/model/{mhc,hyperloop,moe/*}.py`, `norm/qk.py` consumers | nested config access sweep only |
| M16 | `src/deploy/{deploy,inference,transformers_export,inference_config}.py` | nested `config.json`; `to_jsonable`; per-mixer special cases audited (list in P5 checklist) |
| M17 | `src/training/{trainer,main,sbert/*,audio_dataset,rnnt}.py` | nested config access; trainer consumes `closure_based`/`requires_grad_accum_one` capabilities |
| M18 | `src/cli.py` | `export-schema` subcommand; model-class choices unchanged |
| M19 | `src/streamlit_gui/app.py` | delete `prefix_map`; `render_optimizer_section` iterates assembled `by_optimizer` suffixes; everything else is already schema-driven |
| M20 | `full_tests/run_e2e.py` | `OPTIMIZERS` from registry; `_with_optimizer` from `param_bindings` (+ small overrides) |
| M21 | `src/schema.yaml` | `plugins:` top-level key (+ `$ref` to a small `_plugins.yaml`) |
| M22 | `src/schema/_training.yaml` | delete ~36 `if/then` blocks → one `x_registry_if_then: optimizers` slot |
| M23 | `src/schema/_optimizer.yaml` | enum/tables/suffix-lists/`by_optimizer` replaced by markers; prose + shared-suffix docs stay |
| M24 | `src/schema/_model/_dims.yaml` | `layer_pattern` enum + tables → marker; prose stays |
| M25 | `src/schema/_model.yaml` | `attention.properties` hand-`$ref`s → marker |
| M26 | `src/schema/_model/{_norm,_ffn,_residuals,_positional_encoding,_embedding}.yaml` | enums/tables → markers; fragments (`_attention_*.yaml` files, PE param sub-objects) **stay as the spec-referenced fragments** |
| M27 | `tests/**` | §7 |
| M28 | `docs/architecture.md` | §4 flattening contract → "config is native nested"; §7 component reference tables generated from registry where enumerations appear; §11 extension recipes rewritten (one recipe for all families: implement + `SPEC` + example + tests); §12 gates updated |
| M29 | `AGENTS.md` | "Extending" table replaced by the single SPEC recipe + registry map; hard rules 1/7/9/10 reworded (rule 1: "the assembled schema is the contract; specs own fragments"); census numbers unchanged |
| M30 | `.opencode/command/add-mixer.md`, `add-optimizer.md` | rewritten around the SPEC recipe (paper reading, annex/PDF/website steps remain for core research components) |
| M31 | `configs/README.md` + `README.md` | optimizer suffix table generated from specs; census wording unchanged |
| M32 | `.opencode/command/sync-website.md` (+ state script) | sync the assembled schema snapshot; probe update |

### 4.3 Deleted files

- `src/utils/config_flatten.py` (whole file).
- Nothing else is deleted outright; everything else is slimmed per §3.

---

## 5. Config-format changes (breaking, accepted)

1. `model.embedding.{factorized,dynamic}.enabled` → `model.embedding.type:
   standard|factorized|dynamic` (params stay in their sub-objects).
2. Legacy **flat** `model:` blocks (old checkpoint `config.json`) no longer
   load; the nested schema shape is the only accepted shape.
3. Checkpoint `config.json` and HF export store the nested model block.
4. New top-level `plugins:` key (opt-in; absent = built-ins only, zero cost).
5. No YAML preset in `configs/` or `configs/examples/` changes shape (they are
   already nested and schema-valid) **except** presets using the two embedding
   booleans (grep in P4) — mechanically rewritten to `embedding.type`.

Programmatic constructors (`FrankensteinModelConfig(**flat_kwargs)` in tests
and `full_tests/`) are rewritten to `from_dict`/nested constructors; the P5
checklist includes the grep recipe to enumerate them all
(`grep -rn "FrankensteinModelConfig(" tests/ full_tests/ src/`).

---

## 6. Plugin contract (what a plugin author must provide / may rely on)

**Must**: `name` (validated), a lazy `builder`, `description` +
`description_es` (bilingual docs are part of the contract, D8/parity gate),
`params_schema` fragment (strict; `additionalProperties: false` enforced by
assembly), family-specific capability flags it truthfully declares
(`training_free`, `cache`/`cache_implemented`, `qk_norm_supported`,
`needs_pos_encoder`, `is_glu`, `closure_based`, …).

**May rely on**: constructor contracts per family — mixer builder signature
`(config, pos_encoder=None, params: ComponentParams)`; optimizer
`(param_groups, **kwargs)` with shared per-group routing done for it;
norm `(config, params)`; activation `(params, dim)`; residual `(config,
params)`; PE `(config, params)`; embedding `(config, params)`. Plus: schema
injection, Streamlit rendering, preset gate, e2e harness listing, error
messages listing known names — all automatic.

**Never allowed**: shadowing a built-in name (hard error); mutating the
registry of another family; reading undocumented keys (binding rejects).

**Documented limitations for plugins**: no automatic KV-cache (default
`supported=False` → legacy full-recompute, D5 "inference never fails"); no
website param-estimator coverage (`ft-param-estimator.js` stays built-ins-only
— noted in the plugin docs); no LaTeX annex (papers are the theory source of
truth for core components, D11 — a plugin documents via its spec descriptions).

---

## 7. Test gates rewrite (same invariants, now automatic)

| Gate file | Change |
|---|---|
| `tests/test_schema_consistency.py` | `test_all_presets_validate` / `test_schema_examples_validate` unchanged (now against assembled schema — includes the example-plugin preset). `test_layer_pattern_enum_matches_documentation` → iterate `registry.specs(MIXER)`, assert each spec's `description`/`description_es` non-empty and present in the assembled schema. `test_optimizer_prefixes_match_conditional_rules` → assert the assembled `_training.yaml` slot contains exactly one generated rule per optimizer spec and `spec.name` is its own prefix. `_DEFAULT_PAIRS` **replaced** by a generic iterator: for every spec with `params_schema`, every declared `default` must equal what `bind_params({})` returns (schema default ↔ runtime default parity, mechanically enforced for built-ins and plugins). `test_sbert_optimizer_enum_resolves_to_canonical` unchanged (the SBERT `$ref` now resolves to the assembled canonical enum). `test_legacy_flat_feature_keys_are_gone` extended: assert **no** per-component flat keys exist in the schema or config dataclass. `test_optional_ranks_accept_explicit_null` kept, driven from specs |
| `tests/test_optimizer_factory.py` / `test_optimizer_base.py` | `required` set → `set(registry.names(OPTIMIZER))`; per-optimizer build tests stay (they exercise real optimizers); unknown-suffix error tests now also iterate `spec.allowed_keys()` |
| `tests/test_schema_attention_layers.py` | mixer enum + `attention.*` sub-keys sourced from assembled schema ↔ registry |
| `tests/test_common_modules.py`, `test_activation_functions.py`, `test_residuals.py`, `test_positional_encodings.py` | parametrize over `registry.specs(family)` instead of hand lists |
| `tests/test_kv_cache_mixers.py` | parametrizes `registry.property_set(MIXER, "cache_implemented")` (mamba passthrough exclusion kept via spec flag) |
| `tests/test_naming_conventions.py` | add rules: no re-creation of per-component flat fields in `config.py`; no hand-written component enums in `src/schema/**` (markers instead); no new `mixer_registry`-style dicts outside `src/registry/` |
| `tests/test_hybrid_layer.py` | training-free property, mamba builder, flash-norm fusion flag |
| **New** (N10) | duplicate registration raises; unknown name error lists candidates; malformed spec (bad name, missing bilingual description, non-strict fragment) rejected; `bind_params` coercion + defaults parity; entry-point loading (mocked) + `plugins:` path loading (tmp file) + failure semantics (entry-point skip-warning vs config-path hard error); assembled-schema snapshot contains every registered component; registry imports without torch |

---

## 8. Documentation & commands

- `docs/architecture.md`: rewrite §4 (native nested config; delete the
  flattening-contract table), §7 (component reference: each family table now
  says "registry-driven; per-component fragments in specs"; keep the prose
  explanations), §11 (one unified "Add a component" recipe + per-family
  constructor contracts), §12 (gate list updated with the new tests).
- `AGENTS.md`: replace the 10-row "Extending" table with the SPEC recipe;
  update hard rules 1, 7, 9, 10 to the assembled-schema/registry wording;
  add `src/registry/` + `src/plugins/` to the location table; decisions table
  gains D13–D17 (below).
- `.opencode/command/add-mixer.md` / `add-optimizer.md`: core checklist shrinks
  to implement + SPEC + example + family tests; the paper/annex/census/website
  phases remain for core research components and now explicitly note that
  plugin-style local additions skip them.
- `configs/README.md`: optimizer suffix section generated from specs (a small
  doc-build step or a table kept in sync by the naming-conventions test).
- Website sync command: assembled-schema snapshot + estimator note.
- No LaTeX/paper changes are required by this refactor (census unchanged);
  an optional short "extensibility architecture" paragraph for the paper's
  implementation section is offered but not required.

**New decisions to record**:

| # | Decision |
|---|---|
| D13 | Every pluggable component is a declarative `ComponentSpec`; the registry is the single source of truth (name, class, fragment, params, capabilities, bilingual docs) |
| D14 | The schema is assembled at resolve time from core YAML + registry fragments; `additionalProperties: false` continues to hold for built-ins and plugins alike |
| D15 | The runtime config is native nested; the flattener and per-component flat fields are deleted; legacy flat checkpoints are not supported (one-shot migration script provided) |
| D16 | Plugin loading is layered: entry-points (skip-with-warning) + config `plugins:` paths (fail-hard) + in-repo built-ins (same API) |
| D17 | Test gates are registry-derived: adding a component automatically covers docs parity, defaults parity, prefix rules, cache parity and e2e listing |

---

## 9. Phased execution checklist (do NOT run now)

Each phase ends with the full gate suite green; phases are ordered so the
repo is consistent at every boundary.

**P0 — Registry core (no behavior change)**
1. [X] N1–N5 (+ unit tests N10 core/binding parts). No existing file touched.
2. [X] Gates: full suite green (nothing consumes the registry yet).

**P1 — Optimizers (proves the pattern on the most painful dispatch)**
3. [X] M7: `SPEC` in all 36 modules + registration loop; M6 generic
   `build_optimizer` (+ `custom_build` for adafactor/sgd/eve/ea4llm).
4. [X] M22/M23 markers; M20 `run_e2e`; M19 `prefix_map` removal.
5. [X] Rewrite `test_optimizer_factory.py` / `test_optimizer_base.py` /
   optimizer parts of `test_schema_consistency.py` per §7.
6. [X] Gates: `test_optimizer_factory.py test_optimizer_base.py
   test_schema_consistency.py test_yaml_examples.py` + e2e `--category opt
   --limit 3` + full suite.
7. [X] Snapshot: `export-schema` produces an assembled schema diffing only in
   generated optimizer sections.

**P2 — Norms + activations**
8. [X] M8 (+ 6 `NormSpec`s incl. fusion flag), M9 (+ 43 `ActivationSpec`s,
   GLU flags), schema markers M26 (`_norm.yaml`, `_ffn.yaml`).
9. [X] `config.py`: delete activation-config validation (binding owns it).
10. [X] Gates: `test_common_modules.py test_activation_functions.py
    test_qk_norm.py test_schema_consistency.py` + full suite + e2e
    `--category norm` (if present) / `--category opt --limit 3`.

**P3 — Residuals + PEs + token embeddings**
11. [X] M10, M11 (+ 4 + 12 + 3 specs), `embedding.type` enum (M12 emb
    dispatch), preset rewrite for the two embedding booleans (§5.5),
    `_positional_encoding.yaml` / `_residuals.yaml` / `_embedding.yaml`
    markers + `use_pe` generation.
12. [X] Gates: `test_residuals.py test_positional_encodings.py
    test_schema_consistency.py test_yaml_examples.py` + full suite.

**P4 — Config restructure + flattener deletion (the deep rewrite)**
13. [X] M2 (nested dataclasses per §2.4), delete `config_flatten.py`, M3/M4
    loader+engine rewiring, M12/M15/M17 nested-access sweep (grep recipe in
    §5), M16 deploy/export/checkpoints + `to_jsonable`.
14. [X] N9 migration script (optional artifact).
15. [X] Rewrite affected tests (`grep -rn "FrankensteinModelConfig(" tests/
    full_tests/ src/`); update `test_legacy_flat_feature_keys_are_gone`.
16. [X] Gates: full suite + e2e smoke all categories + `param_count_check.py`
    (0 failures — architecture unchanged, param counts must match
    `param_truth.json`) + deploy round-trip (export + reload) smoke.

**P5 — Mixers (biggest family; uses everything P0–P4 built)**
17. [X] M5 HybridLayer registry + mamba spec + training-free property; M13
    param binding sweep across 48 mixer modules (§2.4 mapping); M14
    capabilities derived; M8q qk sets derived; `_dims.yaml`/`_model.yaml`
    markers + bilingual table migration into specs.
18. [X] Gates: `test_schema_attention_layers.py test_kv_cache_mixers.py
    test_hybrid_layer.py test_qk_norm.py test_attention_*.py` + full suite +
    e2e all categories + `param_count_check.py`.

**P6 — Plugin loading + example + docs**
19. [X] N7 loader + N8 example plugin + M21 `plugins:` key; plugin docs section
    in `architecture.md` (§6 contract).
20. [X] M28–M32 docs/commands/README/AGENTS/sync-website updates; census
    wording pass (counts unchanged).
21. [X] Gates: full suite incl. new plugin tests; e2e with the
    `plugin_window_mixer.yaml` preset; Streamlit smoke (schema load +
    optimizer section renders a plugin optimizer spec).
22. [X] Website sync + commit phases (mirror the structure of the existing
    plans' §6/§7: main repo `feat(registry): plugin-based extensibility`,
    then mirror sync commit).

---

## 10. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Targeted per phase | `uv run --extra cpu --extra train python -m pytest tests/ -v --tb=short -p no:warnings` (phase subsets per §9) | All pass |
| Full suite | same, whole `tests/` | No failures vs the P0 baseline |
| e2e smoke | `uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 3` (+ other categories at P4/P5) | Builds + trains 1 step on CPU; optimizer list registry-derived |
| Param counts | `uv run --extra cpu --extra train python full_tests/param_count_check.py` | `0 failures` — proves the config restructure + param binding did not alter any architecture |
| Estimator | `node full_tests/param_estimate_check.mjs <estimator> full_tests/param_truth.json --tol 0.01` | PASS (built-ins unchanged) |
| Schema snapshot | `frankenstein-transformer export-schema --out /tmp/opencode/schema.assembled.yaml` + committed-artifact diff | Diff only in generated sections; snapshot contains every registered component; strict validation of all presets against it |
| Streamlit | `uv run --extra cpu --extra train frankenstein-transformer web-server` smoke: schema loads, mixer/optimizer pickers populated incl. plugin names | Renders; no hand lists |
| Torch-free | `python -c "import src.registry, src.plugins"` in a torch-free env | Imports clean |
| Deploy round-trip | export → `deploy/inference.py` reload of nested `config.json` | Loads and generates |
| Docs | `bash docs/build_docs.sh` | Builds; §4/§7/§11/§12 updated |

---

## 11. Risks & mitigations

1. **Mechanical sweep of `getattr(config, …)` across ~80 model modules**
   (P4/P5). Mitigation: the §2.4 mapping is fully derivable from the deleted
   flattener's tables + `_DEFAULT_PAIRS`; `param_count_check.py` is the
   architectural-equivalence oracle; sweep is done per-family with immediate
   family-test runs.
2. **Schema assembly bugs** (marker typos, fragment mis-merge). Mitigation:
   the assembled snapshot is diffed and committed; the preset gate validates
   every example against the assembled schema; assembly is total (any
   unresolved marker is a hard error, not a silent skip).
3. **Registry import cycles** (family module registers → imports registry;
   registry must never import families). Mitigation: lazy `"module:Attr"`
   builders; a naming-conventions test forbids `src.model` imports inside
   `src/registry`.
4. **Perf regressions in `HybridLayer`**: today the 49-entry dict is rebuilt
   per layer; the module-level registry is strictly cheaper. `spec.bind` runs
   once per layer build (not per forward).
5. **Optimizer behavioral drift** during if/elif → binding transcription
   (36 optimizers). Mitigation: transcription is mechanical (each branch lists
   its own keys/defaults); `test_optimizer_factory.py` keeps every existing
   per-optimizer test unchanged in *intent*; e2e `--category opt` runs all of
   them.
6. **Streamlit regressions**: only `prefix_map` + `render_optimizer_section`
   depend on internals; both are covered by the P6 smoke.
7. **Plugin failure isolation**: entry-point plugins are skip-with-warning;
   config-path plugins fail hard (explicit user intent). The registry itself
   never lets a plugin shadow a built-in.
8. **Website mirror** serving a schema that no longer contains enums on disk.
   Mitigation: mirror serves the assembled snapshot (M32), and the state
   probe verifies it.
9. **Legacy checkpoints** become unreadable (accepted per user decision). The
   optional `scripts/migrate_flat_config.py` converts them mechanically.

**Open questions** (non-blocking, defaults chosen):
- Plugin-provided norm fusion hooks (like FlashNorm Prop. 2) — the
  `supports_ffn_fusion` flag exists but the fusion code path in `HybridLayer`
  only special-cases FlashNorm/BitLinear classes for now; a generic fusion
  interface is future work (documented).
- `override=True` for plugins replacing built-ins — deferred (hard error today).
- Whether `FrankensteinModelConfig` keeps its name (kept — docstrings and
  papers reference it; only its internals change).

---

## 12. Implementation notes (done)

Status: **implemented end to end; 1978 tests pass (1944 pre-existing + 34 new),
1 skipped, 343 subtests.**

What shipped:

- `src/registry/` — `spec.py` (ComponentSpec + seven family specs + CacheSpec),
  `registry.py` (global registry, duplicate detection by stable fingerprint,
  derived capability views), `binding.py` (`bind_params` / `ComponentParams`
  with `value()` None-as-absent semantics), `fragments.py` (YAML fragment
  refs), `coercions.py`, `schema_assembly.py` (registry-driven enums, attention
  fragments incl. the shared `falcon` key, optimizer `by_optimizer` +
  `if/then` + SBERT enum sync, `use_pe` flags, `embedding.type`, `plugins:`),
  `builtin/` (all built-in specs: 49 mixers, 36 optimizers, 6 norms,
  43 activations, 4 residuals, 11 PEs + `none`, 3 embeddings).
- `src/plugins/` — `api.py` (register_* helpers) and `loader.py`
  (entry-points skip-with-warning, config paths fail-hard, built-ins).
- Generic dispatch: `build_optimizer` has no per-optimizer branches;
  `get_norm`, `get_activation`, `build_residual`, `build_pos_encoder` and
  `build_token_embedding` are registry lookups; `HybridLayer` no longer
  imports 48 mixer classes and no longer rebuilds a per-layer registry dict.
- Nested runtime config (`src/model/config.py`): typed structural blocks,
  per-component mappings bound through specs, `from_dict`, `to_jsonable`,
  ported cross-field validation (`src/model/_validation.py`), read/write flat
  aliases and `from_flat` (legacy checkpoints/tests/exporters).
- `src/utils/config_flatten.py` **deleted**; `flatten_inference_dict` moved to
  `src/deploy/inference_config.py`; the validators read nested blocks;
  exporters use `src/deploy/export_compat.py` for their flat metadata view.
- Schema: `resolve_schema()` assembles from the registry; every preset and
  example (including the plugin example) validates strictly.
- Examples: `examples/plugins/window_mixer.py` and
  `configs/examples/plugin_window_mixer.yaml`.
- Tests: registry core/binding/assembly/plugin-loader suites added; the
  schema-consistency, optimizer, KV-parity, deploy and export suites now run
  against the assembled schema and the nested config.
- Docs: `AGENTS.md` (rules, locations, D13–D17), `docs/architecture.md` §4/§11,
  the two plan-mode commands and `configs/README.md`; plus the end-user guides
  `INSTALL.md` (install matrix, extras, plugins, build, troubleshooting) and
  `HOWTO_MAKE_PLUGIN.md` (authoring contract for all seven families,
  packaging, testing, pitfalls), linked from `README.md` and `docs/README.md`.

Deliberate deviations from the plan text:

1. **Built-in specs live in `src/registry/builtin/<family>.py`** rather than
   one `SPEC` per implementation module. This keeps the registry package
   torch-free (lazy builder strings) and avoids a central import cycle; it is
   still declarative registration, and plugins use the same API.
2. **Hand-written bilingual tables were kept**; assembly only appends rows for
   components missing from them (built-ins are already documented). This
   avoided rewriting ~40 KB of reviewed documentation while still exposing
   plugin components automatically.
3. **`from_flat` remains as a migration path** (not a flattener) for legacy
   checkpoints, the exporters and the test suite; per-component flat fields
   and the nested-to-flat pipeline are gone.
4. **`flat_config_view`** in `src/deploy/export_compat.py` is a read-only
   adapter that materialises the historical flat vocabulary the HF/GGUF
   exporters consume; it is not part of the runtime configuration pipeline.

### Final verification (all gates)

| Gate | Result |
|---|---|
| Full test suite | `1978 passed, 1 skipped, 343 subtests passed` (1944 pre-existing + 34 new) |
| Parameter ground truth | `full_tests/param_count_check.py`: **168 configs counted, 0 failures** (includes the plugin preset) |
| JS estimator | `param_estimate_check.mjs … --tol 0.01`: **ALL PASS (168 within 1.0%)** |
| Assembled schema | `docs/schema.assembled.yaml` regenerated; every preset/example validates; mirror `schema.yaml` ships the snapshot |
| Website sync | state script: `diff=0 missing_site=0 only_site=0`; every probe PASS (schema refs, 189 example YAMLs, CLI flags, paper figures, estimator drift) |
| Paper | EN + ES rebuilt (`paper.pdf` 365 pp, `paper-es.pdf` 377 pp); Paper-tab HTML rebuilt; PDFs/HTML synced to the mirror |
| Commits | main `d1f6ad0` (implementation), `80ea6f1` (paper + snapshot); mirror `fe184d5` (sync), `100881d` (pycache cleanup) |
