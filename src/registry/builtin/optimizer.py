"""Built-in optimizer component specs.

One :class:`~src.registry.spec.OptimizerSpec` per optimizer: registry name,
lazy builder, per-suffix constructor bindings (replacing the old
``build_optimizer`` if/elif chain), shared-suffix routing policy and the
closure/SBERT/grad-accumulation capability flags.
"""

from __future__ import annotations

from ..coercions import (
    as_betas,
    as_bool,
    as_float,
    as_int,
    as_lower_str,
    as_optional_int,
)
from ..spec import OptimizerSpec, ParamBinding

__all__ = ["OPTIMIZER_SPECS", "register"]


def _build_adafactor(spec, groups, scoped, kwargs):
    """Adafactor keeps an (eps1, eps2) tuple per group (not a scalar eps)."""
    from ..registry import resolve_builder

    eps = (kwargs["eps1"], kwargs["eps2"])
    for group in groups:
        group["eps"] = eps
    return resolve_builder(spec.builder)(
        groups,
        beta2_decay=kwargs["beta2_decay"],
        clip_threshold=kwargs["clip_threshold"],
        eps=eps,
    )


OPTIMIZER_SPECS = (
    OptimizerSpec(
        name="ada_muon",
        builder="src.model.optimizer.ada_muon:AdaMuon",
        description="Adaptive orthogonal updates — Element-wise second moment on orthogonalized directions + sign stabilization + RMS-aligned scaling (paper: arxiv:2507.11005)",
        description_es="Actualizaciones ortogonales adaptativas — Segundo momento por elemento sobre direcciones ortogonalizadas + estabilización por signo + escalado alineado por RMS (paper: arxiv:2507.11005)",
        references=("2507.11005",),
        param_bindings={
            "momentum": ParamBinding(as_float, 0.95),
            "ns_steps": ParamBinding(as_int, 5),
            "ns_eps": ParamBinding(as_float, 1e-07),
        },
    ),
    OptimizerSpec(
        name="adafactor",
        builder="src.model.optimizer.adafactor:Adafactor",
        description="Large models, low memory — Memory-efficient factored moments",
        description_es="Modelos grandes, baja memoria — Eficiente en memoria",
        param_bindings={
            "beta2_decay": ParamBinding(as_float, 0.8),
            "clip_threshold": ParamBinding(as_float, 1.0),
            "eps1": ParamBinding(as_float, 1e-30),
            "eps2": ParamBinding(as_float, 0.001),
        },
        custom_build=_build_adafactor,
    ),
    OptimizerSpec(
        name="adam_pm",
        builder="src.model.optimizer.adam_pm:AdamPM",
        description="Memory-constrained training — Pooled second moment: group size `s` shrinks the V-state to 1/s, Adam hyperparameters transfer (paper: arxiv:2610.24598)",
        description_es="Entrenamiento con memoria limitada — Segundo momento agrupado: el tamaño de grupo `s` reduce el estado V a 1/s, los hiperparámetros de Adam se transfieren (paper: arxiv:2610.24598)",
        references=("2610.24598",),
        param_bindings={
            "group_size": ParamBinding(as_int, 8),
            "pool_dim": ParamBinding(as_lower_str, "row"),
        },
    ),
    OptimizerSpec(
        name="adamw",
        builder="src.model.optimizer.adamw:AdamWOptimizer",
        description="General purpose **(recommended)** — Decoupled weight decay, stable",
        description_es="Propósito general **(recomendado)** — Weight decay desacoplado, estable",
    ),
    OptimizerSpec(
        name="adan",
        builder="src.model.optimizer.adan:Adan",
        description="Nesterov-accelerated — Adaptive with Nesterov, 3 buffers",
        description_es="Acelerado con Nesterov — Adaptativo con Nesterov",
    ),
    OptimizerSpec(
        name="ademamix",
        builder="src.model.optimizer.ademamix:AdEMAMix",
        description="Dual-EMA mixing — Combines fast/slow momentum for data efficiency",
        description_es="Mezcla de momentos — Combina EMAs de corto y largo plazo",
    ),
    OptimizerSpec(
        name="adopt",
        builder="src.model.optimizer.adopt:ADOPT",
        description="Momentum-based — Decoupled gradient/variance, optimal convergence",
        description_es="Basado en momentum — Momentum adaptativo con convergencia óptima",
    ),
    OptimizerSpec(
        name="af_muon",
        builder="src.model.optimizer.af_muon:AFMuon",
        description="AdamW-free tied-vocab models — Spectral matrices + capped-row LMO table + RMS vectors",
        description_es="Modelos con vocabulario atado sin AdamW — Matrices espectrales + LMO por fila con tope + vectores RMS",
        param_bindings={
            "momentum": ParamBinding(as_float, 0.95),
            "nesterov": ParamBinding(as_bool, True),
            "ns_steps": ParamBinding(as_int, 5),
            "ns_eps": ParamBinding(as_float, 1e-07),
            "vector_lr": ParamBinding(as_float, 0.0003),
            "tied_cap": ParamBinding(as_float, 3.0),
            "tied_scale": ParamBinding(as_float, 0.5),
            "rho_hidden": ParamBinding(as_float, 50.0),
            "rho_output": ParamBinding(as_float, 3000.0),
            "chunk_rows": ParamBinding(as_int, 2048),
            "bisection_steps": ParamBinding(as_int, 32),
            "max_bracket_steps": ParamBinding(as_int, 128),
        },
    ),
    OptimizerSpec(
        name="anon",
        builder="src.model.optimizer.anon:Anon",
        description="Tunable adaptivity — gamma parameter (paper: arxiv:2605.02317)",
        description_es="Adaptividad ajustable — Parámetro gamma (paper: arxiv:2605.02317)",
        references=("2605.02317",),
        param_bindings={
            "gamma": ParamBinding(as_float, 0.0),
        },
    ),
    OptimizerSpec(
        name="apollo",
        builder="src.model.optimizer.apollo:Apollo",
        description="Low-rank scaling — Projected gradient scaling (SGD-like memory)",
        description_es="Escalado de bajo rango — Escalado de gradiente proyectado",
        param_bindings={
            "rank": ParamBinding(as_int, 128),
            "update_proj_gap": ParamBinding(as_int, 200),
            "scale": ParamBinding(as_float, 1.0),
            "scale_type": ParamBinding(as_lower_str, "channel"),
            "proj_type": ParamBinding(as_lower_str, "std"),
            "scale_front": ParamBinding(as_bool, False),
            "disable_nl": ParamBinding(as_bool, False),
        },
    ),
    OptimizerSpec(
        name="apollo_mini",
        builder="src.model.optimizer.apollo_mini:ApolloMini",
        description="Rank-1 APOLLO — Tensor-wise scaled, extreme memory savings",
        description_es="APOLLO rango-1 — Proyección mínima tensorial",
        param_bindings={
            "update_proj_gap": ParamBinding(as_int, 200),
            "scale": ParamBinding(as_float, 128.0),
            "proj_type": ParamBinding(as_lower_str, "std"),
            "scale_front": ParamBinding(as_bool, False),
            "disable_nl": ParamBinding(as_bool, False),
        },
    ),
    OptimizerSpec(
        name="cautious_adamw",
        builder="src.model.optimizer.cautious_adamw:CautiousAdamW",
        description="Gradient masking — Masks conflicted gradient dimensions",
        description_es="Enmascaramiento de gradientes — Enmascara gradientes conflictivos",
        param_bindings={
            "cautious_clip": ParamBinding(as_float, 1.0),
        },
    ),
    OptimizerSpec(
        name="conda",
        builder="src.model.optimizer.conda:Conda",
        description="Spectral + coordinate-wise adaptive pretraining — Adam update inside the SVD subspace of the first moment; second moment of *projected* gradients (column-wise adaptive); lazy SVD every `update_proj_gap` steps; AdamW fallback for non-matrix params (paper: arxiv:2509.24218)",
        description_es="Preentrenamiento espectral + adaptativo por coordenada — Actualización de Adam dentro del subespacio SVD del primer momento; segundo momento de gradientes *proyectadas* (adaptativo por columna); SVD perezoso cada `update_proj_gap` pasos; respaldo AdamW para parámetros no matriciales (paper: arxiv:2509.24218)",
        references=("2509.24218",),
        param_bindings={
            "update_proj_gap": ParamBinding(as_int, 2000),
            "scale": ParamBinding(as_float, 1.0),
            "correct_bias": ParamBinding(as_bool, True),
        },
    ),
    OptimizerSpec(
        name="dion3",
        builder="src.model.optimizer.dion3:Dion3",
        description="Faster orthogonal updates — Fractional Gram-Newton-Schulz Muon with error feedback (paper: arxiv:2608.11612)",
        description_es="Ortogonalización acelerada — Muon fraccional con Gram-Newton-Schulz y retroalimentación de error (paper: arxiv:2608.11612)",
        references=("2608.11612",),
        param_bindings={
            "fraction": ParamBinding(as_float, 0.25),
            "momentum": ParamBinding(as_float, 0.95),
            "ns_steps": ParamBinding(as_int, 5),
            "ns_eps": ParamBinding(as_float, 1e-07),
            "normuon": ParamBinding(as_bool, False),
        },
    ),
    OptimizerSpec(
        name="ea4llm",
        builder="src.model.optimizer.ea4llm:EA4LLM",
        description="Gradient-free training, memory-constrained regimes — Zeroth-order evolution strategies on the token-level log-likelihood fitness; Gaussian perturbations with z-score/rank shaping; no optimizer state, N forward evaluations per step (paper: arxiv:2510.10603)",
        description_es="Entrenamiento sin gradientes, regímenes con memoria limitada — Estrategias evolutivas de orden cero sobre la aptitud de log-verosimilitud por token; perturbaciones gaussianas con z-score/shaping por rangos; sin estado del optimizador, N evaluaciones forward por paso (paper: arxiv:2510.10603)",
        references=("2510.10603",),
        param_bindings={
            "population": ParamBinding(as_int, 30),
            "sigma": ParamBinding(as_float, 0.001),
            "antithetic": ParamBinding(as_bool, False),
            "rank_shaping": ParamBinding(as_bool, False),
            "nes_scaling": ParamBinding(as_bool, False),
            "seed": ParamBinding(as_optional_int, None),
        },
        closure_based=True,
        sbert_allowed=False,
        requires_grad_accum_one=True,
    ),
    OptimizerSpec(
        name="eve",
        builder="src.model.optimizer.eve:Eve",
        description="Fast proxy training, search loops — Population-of-4 differential evolution with targeted Adam fallback; monotone best-so-far; needs its own LR search (paper: arxiv:2609.35614)",
        description_es="Entrenamiento proxy rápido, bucles de búsqueda — Evolución diferencial con población de 4 y respaldo Adam dirigido; mejor-observado monótono; requiere su propia búsqueda de LR (paper: arxiv:2609.35614)",
        references=("2609.35614",),
        param_bindings={
            "de_weight": ParamBinding(as_float, 0.5),
            "de_jitter": ParamBinding(as_float, 0.0001),
            "init_sigma": ParamBinding(as_float, 0.1),
            "adam_steps": ParamBinding(as_int, 3),
            "use_pm": ParamBinding(as_bool, False),
            "pm_eta": ParamBinding(as_float, 20.0),
            "seed": ParamBinding(as_optional_int, None),
        },
        closure_based=True,
        sbert_allowed=False,
        requires_grad_accum_one=True,
    ),
    OptimizerSpec(
        name="galore_adamw",
        builder="src.model.optimizer.galore_adamw:GaLoreAdamW",
        description="Memory-constrained large models — Gradient low-rank projection",
        description_es="Modelos grandes con memoria limitada — Proyección de gradiente de bajo rango",
        param_bindings={
            "rank": ParamBinding(as_int, 128),
            "update_proj_gap": ParamBinding(as_int, 200),
        },
    ),
    OptimizerSpec(
        name="lamb",
        builder="src.model.optimizer.lamb:LAMB",
        description="Large batch training — LARS-style layer-wise adaptive",
        description_es="Entrenamiento con lotes grandes — Escalado por capa estilo LARS",
    ),
    OptimizerSpec(
        name="laprop",
        builder="src.model.optimizer.laprop:LaProp",
        description="Decoupled momentum + adaptivity — Adam-shaped, but the first moment averages the *preconditioned* gradient (RMSProp-with-momentum view), so parameter-space momentum stays additive and is not erased by a single pathological gradient; same state and hyperparameters as Adam (paper: arxiv:2002.04839)",
        description_es="Momentum y adaptividad desacoplados — Con forma de Adam, pero el primer momento promedia el gradiente *precondicionado* (vista de RMSProp con momentum), de modo que el momentum en el espacio de parámetros permanece aditivo y no lo borra un único gradiente patológico; mismo estado e hiperparámetros que Adam (paper: arxiv:2002.04839)",
        references=("2002.04839",),
    ),
    OptimizerSpec(
        name="lion",
        builder="src.model.optimizer.lion:Lion",
        description="Memory-efficient — Sign-based updates, reduced state",
        description_es="Eficiencia de memoria — Actualizaciones basadas en signo",
    ),
    OptimizerSpec(
        name="mars_adamw",
        builder="src.model.optimizer.mars_adamw:MARSAdamW",
        description="Variance reduction — Recursive momentum correction (SVRG-style)",
        description_es="Regularizado con MARS — Reducción de varianza",
    ),
    OptimizerSpec(
        name="muon",
        builder="src.model.optimizer.muon:Muon",
        description="Stable training — Newton-Schulz orthogonalization for hidden matrices; AdamW fallback for embeddings, pos, heads and non-matrix params",
        description_es="Entrenamiento estable — Ortogonalización Newton-Schulz para matrices ocultas; respaldo AdamW para embeddings, pos, heads y parámetros no matriciales",
        param_bindings={
            "momentum": ParamBinding(as_float, 0.95),
            "nesterov": ParamBinding(as_bool, True),
            "ns_steps": ParamBinding(as_int, 5),
            "ns_eps": ParamBinding(as_float, 1e-07),
            "per_head": ParamBinding(as_bool, False),
            "per_head_heads": ParamBinding(as_int, 0),
            "clip_value": ParamBinding(as_float, 0.0),
        },
    ),
    OptimizerSpec(
        name="muon2",
        builder="src.model.optimizer.muon2:Muon2",
        description="Adaptive orthogonal updates — Adam-style second-moment preconditioning of the momentum before Newton-Schulz; 40% fewer NS iterations; optional factored moments (Muon²-F); AdamW fallback for embeddings, pos, heads and non-matrix params (paper: arxiv:2604.09967)",
        description_es="Ortogonalización adaptativa — Precondicionamiento del momentum con segundo momento estilo Adam antes de Newton-Schulz; 40% menos iteraciones NS; momentos factorizados opcionales (Muon²-F); respaldo AdamW para embeddings, pos, heads y parámetros no matriciales (paper: arxiv:2604.09967)",
        references=("2604.09967",),
        param_bindings={
            "ns_steps": ParamBinding(as_int, 5),
            "ns_eps": ParamBinding(as_float, 1e-07),
            "factored": ParamBinding(as_bool, False),
        },
    ),
    OptimizerSpec(
        name="namo",
        builder="src.model.optimizer.namo:NAMO",
        description="Noise-adaptive orthogonal updates — Norm-based Adam-type adaptive stepsize α_t on Muon's orthogonalized momentum: a scalar second-moment EMA of the squared Frobenius gradient norm rescales the whole orthogonal update (zero extra memory); AdamW fallback for embeddings, pos, heads and non-matrix params (paper: arxiv:2602.17080)",
        description_es="Actualizaciones ortogonales adaptativas al ruido — Paso adaptativo estilo Adam basado en normas sobre el momentum ortogonalizado de Muon: una EMA escalar del segundo momento de la norma de Frobenius al cuadrado del gradiente reescala toda la actualización ortogonal (sin memoria extra); respaldo AdamW para embeddings, pos, heads y parámetros no matriciales (paper: arxiv:2602.17080)",
        references=("2602.17080",),
        param_bindings={
            "betas": ParamBinding(as_betas, (0.95, 0.99)),
            "ns_steps": ParamBinding(as_int, 5),
            "ns_eps": ParamBinding(as_float, 1e-07),
        },
    ),
    OptimizerSpec(
        name="namo_d",
        builder="src.model.optimizer.namo_d:NAMO_D",
        description="Noise-adaptive orthogonal updates (diagonal) — Diagonal extension of `namo`: a neuron-wise (column) second-moment EMA gives clamped per-column stepsizes that right-multiply the orthogonalized momentum (one extra ℝⁿ vector per matrix); AdamW fallback for embeddings, pos, heads and non-matrix params (paper: arxiv:2602.17080)",
        description_es="Actualizaciones ortogonales adaptativas al ruido (diagonal) — Extensión diagonal de `namo`: una EMA por neurona (columna) del segundo momento produce pasos por columna acotados que multiplican por la derecha el momentum ortogonalizado (un vector ℝⁿ extra por matriz); respaldo AdamW para embeddings, pos, heads y parámetros no matriciales (paper: arxiv:2602.17080)",
        references=("2602.17080",),
        param_bindings={
            "betas": ParamBinding(as_betas, (0.95, 0.99)),
            "ns_steps": ParamBinding(as_int, 5),
            "ns_eps": ParamBinding(as_float, 1e-07),
            "clamp": ParamBinding(as_float, 0.1),
        },
    ),
    OptimizerSpec(
        name="nor_muon",
        builder="src.model.optimizer.nor_muon:NorMuon",
        description="Balanced neuron updates — Neuron-wise (row) second-moment normalization after Newton-Schulz; Muon memory + m scalars; Adam-RMS-aligned update (paper: arxiv:2510.05491)",
        description_es="Actualizaciones por neurona equilibradas — Normalización por segundo momento de fila (neurona) tras Newton-Schulz; memoria de Muon + m escalares; actualización alineada al RMS de Adam (paper: arxiv:2510.05491)",
        references=("2510.05491",),
        param_bindings={
            "momentum": ParamBinding(as_float, 0.95),
            "ns_steps": ParamBinding(as_int, 5),
            "ns_eps": ParamBinding(as_float, 1e-07),
        },
    ),
    OptimizerSpec(
        name="prodigy",
        builder="src.model.optimizer.prodigy:Prodigy",
        description="Minimal LR tuning — Adaptive, auto-LR",
        description_es="Mínimo ajuste de LR — Adaptativo, LR automático",
        param_bindings={
            "d_coef": ParamBinding(as_float, 0.8),
        },
    ),
    OptimizerSpec(
        name="q_apollo",
        builder="src.model.optimizer.q_apollo:QApollo",
        description="Quantized APOLLO — Quantized optimizer states",
        description_es="APOLLO cuantizado — Estados cuantizados",
        param_bindings={
            "rank": ParamBinding(as_int, 128),
            "update_proj_gap": ParamBinding(as_int, 200),
            "scale": ParamBinding(as_float, 1.0),
            "scale_type": ParamBinding(as_lower_str, "channel"),
            "proj_type": ParamBinding(as_lower_str, "std"),
            "scale_front": ParamBinding(as_bool, False),
            "disable_nl": ParamBinding(as_bool, False),
            "quant_bits": ParamBinding(as_int, 8),
        },
    ),
    OptimizerSpec(
        name="radam",
        builder="src.model.optimizer.radam:RAdamOptimizer",
        description="Adaptive warmup — Rectified Adam, automatic warmup",
        description_es="Warmup adaptativo — Rectified Adam",
    ),
    OptimizerSpec(
        name="rmsprop",
        builder="src.model.optimizer.rmsprop:RMSProp",
        description="Classical adaptive baseline — Divides the gradient by a running average of its recent squared magnitude (slides: MeanSquare = 0.9·MS + 0.1·g²); one state buffer, no bias correction (source: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)",
        description_es="Línea base adaptativa clásica — Divide el gradiente por un promedio móvil de su magnitud cuadrada reciente (diapositivas: MeanSquare = 0.9·MS + 0.1·g²); un búfer de estado, sin corrección de sesgo (fuente: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)",
        param_bindings={
            "alpha": ParamBinding(as_float, 0.9),
        },
    ),
    OptimizerSpec(
        name="root",
        builder="src.model.optimizer.root:ROOT",
        description="Robust orthogonal updates — Dimension-calibrated Newton-Schulz coefficients + soft-threshold outlier suppression on the momentum; AdamW fallback for embeddings, pos, heads and non-matrix params (paper: arxiv:2511.20626)",
        description_es="Ortogonalización robusta — Coeficientes Newton-Schulz calibrados por dimensión + supresión de atípicos por soft-threshold sobre el momentum; respaldo AdamW para embeddings, pos, heads y parámetros no matriciales (paper: arxiv:2511.20626)",
        references=("2511.20626",),
        param_bindings={
            "momentum": ParamBinding(as_float, 0.95),
            "nesterov": ParamBinding(as_bool, True),
            "ns_steps": ParamBinding(as_int, 5),
            "ns_eps": ParamBinding(as_float, 1e-07),
            "quantile": ParamBinding(as_float, 0.9),
            "softthresh": ParamBinding(as_bool, True),
            "ada_newton": ParamBinding(as_bool, True),
        },
    ),
    OptimizerSpec(
        name="schedulefree_adamw",
        builder="src.model.optimizer.schedulefree_adamw:ScheduleFreeAdamW",
        description="No LR schedule needed — Eliminates scheduler tuning",
        description_es="Sin scheduler de LR — Elimina ajuste de scheduler",
    ),
    OptimizerSpec(
        name="sgd",
        builder="src.model.optimizer.sgd:SGDOptimizer",
        description="Pure mini-batch SGD baseline — w ← w − η·∇E; no momentum, zero optimizer state (source: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)",
        description_es="Línea base SGD de mini-lotes puro — w ← w − η·∇E; sin momentum, sin estado del optimizador (fuente: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)",
        group_policy="sgd",
    ),
    OptimizerSpec(
        name="sgd_momentum",
        builder="src.model.optimizer.sgd_momentum:SGDMomentum",
        description="Baselines, classic tasks — Simple, less adaptive",
        description_es="Líneas base, tareas clásicas — Simple, menos adaptativo",
        param_bindings={
            "momentum": ParamBinding(as_float, 0.9),
            "nesterov": ParamBinding(as_bool, False),
        },
        group_policy="sgd",
    ),
    OptimizerSpec(
        name="shampoo",
        builder="src.model.optimizer.shampoo:Shampoo",
        description="Second-order — Kronecker-factored preconditioners",
        description_es="Segundo orden — Gradientes precondicionados",
    ),
    OptimizerSpec(
        name="soap",
        builder="src.model.optimizer.soap:SOAP",
        description="Second-order + Adam — Adam in Shampoo eigenbasis",
        description_es="Optimización dispersa — Segundo orden eficiente en memoria",
    ),
    OptimizerSpec(
        name="sophia",
        builder="src.model.optimizer.sophia:Sophia",
        description="Robust training — Hessian-aware with diagonal preconditioning",
        description_es="Entrenamiento robusto — Consciente del Hessiano",
        param_bindings={
            "rho": ParamBinding(as_float, 0.04),
            "update_k": ParamBinding(as_int, 10),
        },
    ),
    OptimizerSpec(
        name="tuon",
        builder="src.model.optimizer.tuon:TUON",
        description="Cross-layer orthogonal updates — Tensor-aware Muon: stacks same-shape hidden matrices per group, mixes them with a unitary layer transform (DecorQ from the layer Gram matrix, refreshed every `refresh_interval` steps), polarizes each slice with Newton-Schulz, inverts the transform and rescales the stack to RMS 1/√max(m,n); K=1 reduces to `muon`; AdamW fallback for embeddings, pos, heads and non-matrix params (paper: arxiv:2610.35130)",
        description_es="Actualizaciones ortogonales entre capas — Muon consciente de tensores: apila matrices ocultas de la misma forma por grupo, las mezcla con una transformación unitaria de capas (DecorQ desde la matriz de Gram de capas, refrescada cada `refresh_interval` pasos), polariza cada slice con Newton-Schulz, invierte la transformación y reescala la pila a RMS 1/√max(m,n); K=1 reduce a `muon`; respaldo AdamW para embeddings, pos, heads y parámetros no matriciales (paper: arxiv:2610.35130)",
        references=("2610.35130",),
        param_bindings={
            "momentum": ParamBinding(as_float, 0.95),
            "ns_steps": ParamBinding(as_int, 5),
            "ns_eps": ParamBinding(as_float, 1e-07),
            "refresh_interval": ParamBinding(as_int, 200),
            "rms_eps": ParamBinding(as_float, 1e-08),
            "transform": ParamBinding(as_lower_str, "decorq"),
            "stack": ParamBinding(as_lower_str, "full"),
        },
    ),
    OptimizerSpec(
        name="turbo_muon",
        builder="src.model.optimizer.turbo_muon:TurboMuon",
        description="Faster muon — AOL-preconditioned orthogonalization for hidden matrices; AdamW fallback for embeddings, pos, heads and non-matrix params",
        description_es="Muon acelerado — Ortogonalización precondicionada con AOL para matrices ocultas; respaldo AdamW para embeddings, pos, heads y parámetros no matriciales",
        param_bindings={
            "momentum": ParamBinding(as_float, 0.95),
            "nesterov": ParamBinding(as_bool, True),
            "ns_steps": ParamBinding(as_int, 4),
            "ns_eps": ParamBinding(as_float, 1e-07),
        },
    ),
)


def register() -> None:
    """Register every built-in optimizer spec (idempotent)."""
    from src.registry import register_spec

    for spec in OPTIMIZER_SPECS:
        register_spec(spec)
