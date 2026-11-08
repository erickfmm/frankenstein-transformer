"""Optimizer factory: registry, parameter routing, and construction.

Maps optimizer class names (as they appear in YAML training configs) to
concrete ``torch.optim.Optimizer`` subclasses. Handles extraction of
prefixed hyperparameters, per-group routing, and validation of allowed
keys before dispatching to the optimizer constructor.
"""

from __future__ import annotations

from typing import Any, Dict, List

from torch.optim import Optimizer

from .adam_pm import AdamPM
from .adamw import AdamWOptimizer
from .ada_muon import AdaMuon
from .adafactor import Adafactor
from .adan import Adan
from .ademamix import AdEMAMix
from .adopt import ADOPT
from .af_muon import AFMuon
from .anon import Anon
from .apollo import Apollo
from .apollo_mini import ApolloMini
from .base import (
    GROUP_NAMES,
    ensure_no_unknown_parameters,
    extract_prefixed_parameters,
    parse_group_value,
    to_betas,
    to_float,
    to_int,
    with_named_groups,
)
from .cautious_adamw import CautiousAdamW
from .conda import Conda
from .dion3 import Dion3
from .ea4llm import EA4LLM
from .eve import Eve
from .galore_adamw import GaLoreAdamW
from .lamb import LAMB
from .lion import Lion
from .mars_adamw import MARSAdamW
from .muon import Muon
from .muon2 import Muon2
from .nor_muon import NorMuon
from .prodigy import Prodigy
from .q_apollo import QApollo
from .radam import RAdamOptimizer
from .root import ROOT
from .schedulefree_adamw import ScheduleFreeAdamW
from .sgd_momentum import SGDMomentum
from .shampoo import Shampoo
from .sophia import Sophia
from .soap import SOAP
from .tuon import TUON
from .turbo_muon import TurboMuon

OPTIMIZER_REGISTRY = {
    "sgd_momentum": SGDMomentum,
    "adamw": AdamWOptimizer,
    "adam_pm": AdamPM,
    "adafactor": Adafactor,
    "galore_adamw": GaLoreAdamW,
    "prodigy": Prodigy,
    "lion": Lion,
    "sophia": Sophia,
    "muon": Muon,
    "turbo_muon": TurboMuon,
    "af_muon": AFMuon,
    "ada_muon": AdaMuon,
    "radam": RAdamOptimizer,
    "adan": Adan,
    "adopt": ADOPT,
    "ademamix": AdEMAMix,
    "mars_adamw": MARSAdamW,
    "cautious_adamw": CautiousAdamW,
    "lamb": LAMB,
    "schedulefree_adamw": ScheduleFreeAdamW,
    "shampoo": Shampoo,
    "soap": SOAP,
    "anon": Anon,
    "apollo": Apollo,
    "apollo_mini": ApolloMini,
    "q_apollo": QApollo,
    "dion3": Dion3,
    "ea4llm": EA4LLM,
    "eve": Eve,
    "root": ROOT,
    "muon2": Muon2,
    "nor_muon": NorMuon,
    "conda": Conda,
    "tuon": TUON,
}
"""Mapping from YAML optimizer class names to concrete Optimizer subclasses.

Each key is the string value accepted by the ``optimizer_class`` field in
the training configuration schema. The corresponding value is the class
used to instantiate the optimizer via :func:`build_optimizer`.
"""

_COMMON_PER_GROUP_KEYS = {
    f"lr_{group}" for group in GROUP_NAMES
} | {
    f"wd_{group}" for group in GROUP_NAMES
} | {
    f"betas_{group}" for group in GROUP_NAMES
} | {
    f"eps_{group}" for group in GROUP_NAMES
}

OPTIMIZER_ALLOWED_KEYS = {
    "sgd_momentum": _COMMON_PER_GROUP_KEYS | {"momentum", "nesterov"},
    "adamw": _COMMON_PER_GROUP_KEYS,
    "adam_pm": _COMMON_PER_GROUP_KEYS | {"group_size", "pool_dim"},
    "adafactor": _COMMON_PER_GROUP_KEYS | {"beta2_decay", "clip_threshold", "eps1", "eps2"},
    "galore_adamw": _COMMON_PER_GROUP_KEYS | {"rank", "update_proj_gap"},
    "prodigy": _COMMON_PER_GROUP_KEYS | {"d_coef"},
    "lion": _COMMON_PER_GROUP_KEYS,
    "sophia": _COMMON_PER_GROUP_KEYS | {"rho", "update_k"},
    "muon": _COMMON_PER_GROUP_KEYS | {"momentum", "nesterov", "ns_steps", "ns_eps"},
    "turbo_muon": _COMMON_PER_GROUP_KEYS | {"momentum", "nesterov", "ns_steps", "ns_eps"},
    "af_muon": _COMMON_PER_GROUP_KEYS
    | {
        "momentum",
        "nesterov",
        "ns_steps",
        "ns_eps",
        "vector_lr",
        "tied_cap",
        "tied_scale",
        "rho_hidden",
        "rho_output",
        "chunk_rows",
        "bisection_steps",
        "max_bracket_steps",
    },
    "ada_muon": _COMMON_PER_GROUP_KEYS | {"momentum", "ns_steps", "ns_eps"},
    "radam": _COMMON_PER_GROUP_KEYS,
    "adan": _COMMON_PER_GROUP_KEYS,
    "adopt": _COMMON_PER_GROUP_KEYS,
    "ademamix": _COMMON_PER_GROUP_KEYS,
    "mars_adamw": _COMMON_PER_GROUP_KEYS,
    "cautious_adamw": _COMMON_PER_GROUP_KEYS | {"cautious_clip"},
    "lamb": _COMMON_PER_GROUP_KEYS,
    "schedulefree_adamw": _COMMON_PER_GROUP_KEYS,
    "shampoo": _COMMON_PER_GROUP_KEYS,
    "soap": _COMMON_PER_GROUP_KEYS,
    "apollo": _COMMON_PER_GROUP_KEYS | {"rank", "update_proj_gap", "scale", "scale_type", "proj_type", "scale_front", "disable_nl"},
    "apollo_mini": _COMMON_PER_GROUP_KEYS | {"update_proj_gap", "scale", "proj_type", "scale_front", "disable_nl"},
    "anon": _COMMON_PER_GROUP_KEYS | {"gamma"},
    "q_apollo": _COMMON_PER_GROUP_KEYS | {"rank", "update_proj_gap", "scale", "scale_type", "proj_type", "scale_front", "disable_nl", "quant_bits"},
    "dion3": _COMMON_PER_GROUP_KEYS | {"fraction", "momentum", "ns_steps", "ns_eps", "normuon"},
    "ea4llm": _COMMON_PER_GROUP_KEYS
    | {"population", "sigma", "antithetic", "rank_shaping", "nes_scaling", "seed"},
    "eve": _COMMON_PER_GROUP_KEYS
    | {"de_weight", "de_jitter", "init_sigma", "adam_steps", "use_pm", "pm_eta", "seed"},
    "root": _COMMON_PER_GROUP_KEYS
    | {"momentum", "nesterov", "ns_steps", "ns_eps", "quantile", "softthresh", "ada_newton"},
    "muon2": _COMMON_PER_GROUP_KEYS | {"ns_steps", "ns_eps", "factored"},
    "nor_muon": _COMMON_PER_GROUP_KEYS | {"momentum", "ns_steps", "ns_eps"},
    "conda": _COMMON_PER_GROUP_KEYS | {"update_proj_gap", "scale", "correct_bias"},
    "tuon": _COMMON_PER_GROUP_KEYS
    | {
        "momentum",
        "ns_steps",
        "ns_eps",
        "refresh_interval",
        "rms_eps",
        "transform",
        "stack",
    },
}


def _build_param_groups_for_optimizer(
    optimizer_name: str,
    base_param_groups: List[Dict[str, Any]],
    scoped_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert raw model parameter groups into optimizer-ready param groups.

    For each named group, resolves per-group ``lr``, ``weight_decay``,
    ``betas``, and ``eps`` from the scoped configuration, falling back
    to the values already present on the group (or sensible defaults).

    Args:
        optimizer_name: The optimizer class name (e.g. ``"adamw"``).
            Used to decide whether betas/eps are applicable.
        base_param_groups: Raw parameter groups from the model, possibly
            lacking ``"name"`` keys.
        scoped_params: Optimizer-scoped hyperparameters extracted via
            :func:`extract_prefixed_parameters`.

    Returns:
        A list of parameter group dicts suitable for passing directly to
        an ``Optimizer`` constructor. Every group includes ``params``,
        ``lr``, ``weight_decay``, and ``name``; adaptive optimizers also
        include ``betas`` and ``eps``.
    """
    result: List[Dict[str, Any]] = []
    for group in with_named_groups(base_param_groups):
        group_name = str(group["name"])
        converted = {
            "params": group["params"],
            "lr": to_float(parse_group_value(scoped_params, "lr", group_name, group.get("lr", 1e-3)), 1e-3),
            "weight_decay": to_float(
                parse_group_value(scoped_params, "wd", group_name, group.get("weight_decay", 0.0)),
                0.0,
            ),
        }

        if optimizer_name not in {"sgd_momentum"}:
            converted["betas"] = to_betas(parse_group_value(scoped_params, "betas", group_name, group.get("betas", (0.9, 0.95))))
            converted["eps"] = to_float(parse_group_value(scoped_params, "eps", group_name, group.get("eps", 1e-8)), 1e-8)

        converted["name"] = group_name
        result.append(converted)
    return result


def build_optimizer(
    optimizer_class: str,
    param_groups: List[Dict[str, Any]],
    parameters: Dict[str, Any],
) -> Optimizer:
    """Construct an optimizer instance from a configuration-driven spec.

    This is the single entry point used by the training pipeline. It
    validates the optimizer name, extracts and validates scoped
    hyperparameters, builds per-group parameter dictionaries, and
    dispatches to the appropriate constructor with any optimizer-specific
    top-level arguments.

    Args:
        optimizer_class: The optimizer identifier string from the YAML
            config (e.g. ``"adamw"``, ``"sgd_momentum"``).
        param_groups: Raw parameter groups produced by the model's
            ``configure_optimizers`` method.
        parameters: The full flat configuration dictionary (may contain
            keys for multiple optimizers and global settings).

    Returns:
        A configured ``torch.optim.Optimizer`` instance ready for use
        in the training loop.

    Raises:
        ValueError: If `optimizer_class` is not in
            :data:`OPTIMIZER_REGISTRY`, or if unrecognized scoped
            parameters are present.
    """
    optimizer_name = str(optimizer_class or "").strip().lower()
    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer_class '{optimizer_class}'. "
            f"Supported values: {sorted(OPTIMIZER_REGISTRY)}"
        )

    scoped = extract_prefixed_parameters(optimizer_name, parameters or {})
    ensure_no_unknown_parameters(
        optimizer_name,
        scoped,
        OPTIMIZER_ALLOWED_KEYS[optimizer_name],
    )

    converted_groups = _build_param_groups_for_optimizer(optimizer_name, param_groups, scoped)
    optimizer_cls = OPTIMIZER_REGISTRY[optimizer_name]

    if optimizer_name == "sgd_momentum":
        momentum = to_float(scoped.get("momentum"), 0.9)
        nesterov = bool(scoped.get("nesterov", False))
        return optimizer_cls(converted_groups, momentum=momentum, nesterov=nesterov)

    if optimizer_name == "cautious_adamw":
        cautious_clip = to_float(scoped.get("cautious_clip"), 1.0)
        return optimizer_cls(converted_groups, cautious_clip=cautious_clip)

    if optimizer_name == "adafactor":
        beta2_decay = to_float(scoped.get("beta2_decay"), 0.8)
        clip_threshold = to_float(scoped.get("clip_threshold"), 1.0)
        eps1 = to_float(scoped.get("eps1"), 1e-30)
        eps2 = to_float(scoped.get("eps2"), 1e-3)
        # Adafactor keeps a (eps1, eps2) tuple per group, unlike the scalar
        # eps used by Adam-style families.  Overwrite the group-level float
        # injected by _build_param_groups_for_optimizer so step() can unpack
        # it; eps1/eps2 are the documented adafactor-specific suffixes.
        for group in converted_groups:
            group["eps"] = (eps1, eps2)
        return optimizer_cls(converted_groups, beta2_decay=beta2_decay, clip_threshold=clip_threshold, eps=(eps1, eps2))

    if optimizer_name == "galore_adamw":
        rank = int(to_float(scoped.get("rank"), 128))
        update_proj_gap = int(to_float(scoped.get("update_proj_gap"), 200))
        return optimizer_cls(converted_groups, rank=rank, update_proj_gap=update_proj_gap)

    if optimizer_name == "anon":
        gamma = to_float(scoped.get("gamma"), 0.0)
        return optimizer_cls(converted_groups, gamma=gamma)

    if optimizer_name == "adam_pm":
        group_size = to_int(scoped.get("group_size"), 8)
        pool_dim = str(scoped.get("pool_dim", "row")).strip().lower()
        return optimizer_cls(converted_groups, group_size=group_size, pool_dim=pool_dim)

    if optimizer_name == "prodigy":
        d_coef = to_float(scoped.get("d_coef"), 0.8)
        return optimizer_cls(converted_groups, d_coef=d_coef)

    if optimizer_name == "sophia":
        rho = to_float(scoped.get("rho"), 0.04)
        update_k = int(to_float(scoped.get("update_k"), 10))
        return optimizer_cls(converted_groups, rho=rho, update_k=update_k)

    if optimizer_name in {"muon", "turbo_muon"}:
        momentum = to_float(scoped.get("momentum"), 0.95)
        nesterov = bool(scoped.get("nesterov", True))
        ns_steps = int(to_float(scoped.get("ns_steps"), 5 if optimizer_name == "muon" else 4))
        ns_eps = to_float(scoped.get("ns_eps"), 1e-7)
        return optimizer_cls(
            converted_groups,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
        )

    if optimizer_name == "af_muon":
        momentum = to_float(scoped.get("momentum"), 0.95)
        nesterov = bool(scoped.get("nesterov", True))
        ns_steps = to_int(scoped.get("ns_steps"), 5)
        ns_eps = to_float(scoped.get("ns_eps"), 1e-7)
        vector_lr = to_float(scoped.get("vector_lr"), 3e-4)
        tied_cap = to_float(scoped.get("tied_cap"), 3.0)
        tied_scale = to_float(scoped.get("tied_scale"), 0.5)
        rho_hidden = to_float(scoped.get("rho_hidden"), 50.0)
        rho_output = to_float(scoped.get("rho_output"), 3000.0)
        chunk_rows = to_int(scoped.get("chunk_rows"), 2048)
        bisection_steps = to_int(scoped.get("bisection_steps"), 32)
        max_bracket_steps = to_int(scoped.get("max_bracket_steps"), 128)
        return optimizer_cls(
            converted_groups,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            vector_lr=vector_lr,
            tied_cap=tied_cap,
            tied_scale=tied_scale,
            rho_hidden=rho_hidden,
            rho_output=rho_output,
            chunk_rows=chunk_rows,
            bisection_steps=bisection_steps,
            max_bracket_steps=max_bracket_steps,
        )

    if optimizer_name == "ada_muon":
        momentum = to_float(scoped.get("momentum"), 0.95)
        ns_steps = to_int(scoped.get("ns_steps"), 5)
        ns_eps = to_float(scoped.get("ns_eps"), 1e-7)
        return optimizer_cls(
            converted_groups,
            momentum=momentum,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
        )

    if optimizer_name in {"apollo", "q_apollo"}:
        rank = to_int(scoped.get("rank"), 128)
        update_proj_gap = to_int(scoped.get("update_proj_gap"), 200)
        scale = to_float(scoped.get("scale"), 1.0)
        scale_type = str(scoped.get("scale_type", "channel")).strip().lower()
        proj_type = str(scoped.get("proj_type", "std")).strip().lower()
        scale_front = bool(scoped.get("scale_front", False))
        disable_nl = bool(scoped.get("disable_nl", False))
        if optimizer_name == "q_apollo":
            quant_bits = to_int(scoped.get("quant_bits"), 8)
            return optimizer_cls(
                converted_groups,
                rank=rank,
                update_proj_gap=update_proj_gap,
                scale=scale,
                scale_type=scale_type,
                proj_type=proj_type,
                scale_front=scale_front,
                disable_nl=disable_nl,
                quant_bits=quant_bits,
            )
        return optimizer_cls(
            converted_groups,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            scale_type=scale_type,
            proj_type=proj_type,
            scale_front=scale_front,
            disable_nl=disable_nl,
        )

    if optimizer_name == "apollo_mini":
        update_proj_gap = to_int(scoped.get("update_proj_gap"), 200)
        scale = to_float(scoped.get("scale"), 128.0)
        proj_type = str(scoped.get("proj_type", "std")).strip().lower()
        scale_front = bool(scoped.get("scale_front", False))
        disable_nl = bool(scoped.get("disable_nl", False))
        return optimizer_cls(
            converted_groups,
            update_proj_gap=update_proj_gap,
            scale=scale,
            proj_type=proj_type,
            scale_front=scale_front,
            disable_nl=disable_nl,
        )

    if optimizer_name == "dion3":
        fraction = to_float(scoped.get("fraction"), 0.25)
        momentum = to_float(scoped.get("momentum"), 0.95)
        ns_steps = to_int(scoped.get("ns_steps"), 5)
        ns_eps = to_float(scoped.get("ns_eps"), 1e-7)
        normuon = bool(scoped.get("normuon", False))
        return optimizer_cls(
            converted_groups,
            fraction=fraction,
            momentum=momentum,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            normuon=normuon,
        )

    if optimizer_name == "ea4llm":
        population = to_int(scoped.get("population"), 30)
        sigma = to_float(scoped.get("sigma"), 1e-3)
        antithetic = bool(scoped.get("antithetic", False))
        rank_shaping = bool(scoped.get("rank_shaping", False))
        nes_scaling = bool(scoped.get("nes_scaling", False))
        seed_value = scoped.get("seed")
        seed = to_int(seed_value, 0) if seed_value is not None else None
        return optimizer_cls(
            converted_groups,
            population=population,
            sigma=sigma,
            antithetic=antithetic,
            rank_shaping=rank_shaping,
            nes_scaling=nes_scaling,
            seed=seed,
        )

    if optimizer_name == "eve":
        de_weight = to_float(scoped.get("de_weight"), 0.5)
        de_jitter = to_float(scoped.get("de_jitter"), 1e-4)
        init_sigma = to_float(scoped.get("init_sigma"), 0.1)
        adam_steps = to_int(scoped.get("adam_steps"), 3)
        use_pm = bool(scoped.get("use_pm", False))
        pm_eta = to_float(scoped.get("pm_eta"), 20.0)
        seed_value = scoped.get("seed")
        seed = to_int(seed_value, 0) if seed_value is not None else None
        return optimizer_cls(
            converted_groups,
            de_weight=de_weight,
            de_jitter=de_jitter,
            init_sigma=init_sigma,
            adam_steps=adam_steps,
            use_pm=use_pm,
            pm_eta=pm_eta,
            seed=seed,
        )

    if optimizer_name == "root":
        momentum = to_float(scoped.get("momentum"), 0.95)
        nesterov = bool(scoped.get("nesterov", True))
        ns_steps = to_int(scoped.get("ns_steps"), 5)
        ns_eps = to_float(scoped.get("ns_eps"), 1e-7)
        quantile = to_float(scoped.get("quantile"), 0.90)
        softthresh = bool(scoped.get("softthresh", True))
        ada_newton = bool(scoped.get("ada_newton", True))
        return optimizer_cls(
            converted_groups,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            quantile=quantile,
            softthresh=softthresh,
            ada_newton=ada_newton,
        )

    if optimizer_name == "muon2":
        ns_steps = to_int(scoped.get("ns_steps"), 5)
        ns_eps = to_float(scoped.get("ns_eps"), 1e-7)
        factored = bool(scoped.get("factored", False))
        return optimizer_cls(
            converted_groups,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            factored=factored,
        )

    if optimizer_name == "nor_muon":
        momentum = to_float(scoped.get("momentum"), 0.95)
        ns_steps = to_int(scoped.get("ns_steps"), 5)
        ns_eps = to_float(scoped.get("ns_eps"), 1e-7)
        return optimizer_cls(
            converted_groups,
            momentum=momentum,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
        )

    if optimizer_name == "conda":
        update_proj_gap = to_int(scoped.get("update_proj_gap"), 2000)
        scale = to_float(scoped.get("scale"), 1.0)
        correct_bias = bool(scoped.get("correct_bias", True))
        return optimizer_cls(
            converted_groups,
            update_proj_gap=update_proj_gap,
            scale=scale,
            correct_bias=correct_bias,
        )

    if optimizer_name == "tuon":
        momentum = to_float(scoped.get("momentum"), 0.95)
        ns_steps = to_int(scoped.get("ns_steps"), 5)
        ns_eps = to_float(scoped.get("ns_eps"), 1e-7)
        refresh_interval = to_int(scoped.get("refresh_interval"), 200)
        rms_eps = to_float(scoped.get("rms_eps"), 1e-8)
        transform = str(scoped.get("transform", "decorq")).strip().lower()
        stack = str(scoped.get("stack", "full")).strip().lower()
        return optimizer_cls(
            converted_groups,
            momentum=momentum,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            refresh_interval=refresh_interval,
            rms_eps=rms_eps,
            transform=transform,
            stack=stack,
        )

    return optimizer_cls(converted_groups)
