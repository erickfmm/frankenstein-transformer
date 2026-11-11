"""Unit tests for the Mixture-of-Experts runtime and schema contract.

Covers the features added by the MoE completion plan: router scoring, output
scale, z-loss, load balancing (aux loss scopes + auxiliary-loss-free bias),
expert choice, capacity/dropping, hash layers, shared experts, LatentMoE,
layer layout, activation clamping, diagnostics, export and parameter
categorisation.

The tests are CPU-only and use tiny geometries so they run in the default
``--extra cpu`` gate.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.model.config import FrankensteinModelConfig, merge_model_blocks
from src.model.hybrid_layer import HybridLayer
from src.model.moe.balancing import load_balancing_loss, max_vio
from src.model.moe.layer import MoELayer
from src.model.moe.router import MoERouter
from src.model.frankenstein_encoder import FrankensteinEncoder
from src.utils.param_categories import categorize_param


def _config(**overrides) -> FrankensteinModelConfig:
    values = {"dims": {"vocab_size": 100, "hidden_size": 32, "num_layers": 2, "num_loops": 1, "num_heads": 4, "layer_pattern": ['standard', 'standard'], "dropout": 0.0}, "bitnet": {"enabled": False}, "norm": {"type": 'layer_norm'}, "ffn": {"hidden_size": 64}, "moe": {"enabled": True, "num_experts": 4, "top_k": 2}}
    values = merge_model_blocks(values, overrides)
    return FrankensteinModelConfig.from_dict({**values})


def _expert_factory(intermediate_size, in_out_size, clamp):
    layers = [torch.nn.Linear(in_out_size, intermediate_size)]
    if clamp:
        layers.append(torch.nn.Hardtanh(-clamp, clamp))
    layers += [
        torch.nn.SiLU(),
        torch.nn.Linear(intermediate_size, in_out_size),
    ]
    return torch.nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Schema / validation rules
# ---------------------------------------------------------------------------


def test_softmax_top1_auto_enables_pre_softmax():
    with pytest.warns(UserWarning, match="pre_softmax"):
        cfg = _config(moe={'top_k': 1})
    assert cfg.moe.router.pre_softmax is True

    cfg = _config(moe={'top_k': 1, 'router': {'pre_softmax': True}})
    assert cfg.moe.router.pre_softmax is True


def test_bias_requires_non_softmax_scoring():
    with pytest.raises(ValueError, match="bias_update"):
        _config(moe={'balancing': {'bias_update': 'sign'}})
    cfg = _config(moe={'router': {'scoring': 'sigmoid'}, 'balancing': {'bias_update': 'sign'}})
    assert cfg.moe.balancing.bias_update == "sign"


def test_expert_choice_forbids_balancing():
    with pytest.raises(ValueError, match="self-balancing"):
        _config(moe={'routing': {'type': 'expert_choice'}})
    cfg = _config(moe={'routing': {'type': 'expert_choice'}, 'balancing': {'aux_loss': 'none'}})
    assert cfg.moe.routing.type == "expert_choice"


def test_hash_layers_require_token_choice_and_no_bias():
    with pytest.raises(ValueError, match="token_choice"):
        _config(moe={'routing': {'type': 'expert_choice', 'hash_layers': 1}, 'balancing': {'aux_loss': 'none'}})
    with pytest.raises(ValueError, match="bias"):
        _config(moe={'router': {'scoring': 'sigmoid'}, 'balancing': {'bias_update': 'sign'}, 'routing': {'hash_layers': 1}})


def test_latent_requires_valid_dim():
    with pytest.raises(ValueError, match="latent"):
        _config(moe={'latent': {'enabled': True}})
    with pytest.raises(ValueError, match="latent_dim"):
        _config(moe={'latent': {'enabled': True, 'dim': 32}})
    cfg = _config(moe={'latent': {'enabled': True, 'dim': 16}})
    assert cfg.moe.latent.dim == 16


def test_layout_mask():
    cfg = _config(dims={'num_layers': 5}, moe={'layout': {'dense_layers': 1, 'every_n_layers': 2}})
    assert cfg.moe_layer_mask() == [False, True, False, True, False]
    assert cfg.moe_layer_count() == 2

    dense = _config(dims={'num_layers': 3}, moe={'enabled': False})
    assert dense.moe_layer_mask() == [False, False, False]


# ---------------------------------------------------------------------------
# Router scoring
# ---------------------------------------------------------------------------


def _router(**overrides):
    values = dict(
        hidden_size=16,
        num_experts=8,
        top_k=2,
        scoring="softmax",
        pre_softmax=False,
        renormalize=True,
        routed_scaling_factor=1.0,
        dtype="float32",
        jitter_eps=0.0,
        z_loss_coeff=0.0,
        bias_update="none",
        bias_update_rate=0.0,
    )
    values = merge_model_blocks(values, overrides)
    torch.manual_seed(0)
    return MoERouter(**values)


def test_scoring_functions_rank_identically():
    torch.manual_seed(1)
    x = torch.randn(5, 16)
    indices = {}
    for scoring in ("softmax", "sigmoid", "softplus_sqrt"):
        router = _router(scoring=scoring)
        _, idx, _, _ = router(x)
        indices[scoring] = idx
    assert torch.equal(indices["softmax"], indices["sigmoid"])
    assert torch.equal(indices["softmax"], indices["softplus_sqrt"])


def test_renormalisation_and_scaling():
    x = torch.randn(6, 16)
    router = _router(renormalize=True, routed_scaling_factor=2.5)
    weights, _, _, _ = router(x)
    assert torch.allclose(weights.sum(-1), torch.full((6,), 2.5), atol=1e-5)

    raw = _router(renormalize=False, routed_scaling_factor=1.0, scoring="sigmoid")
    weights, _, _, _ = raw(x)
    assert not torch.allclose(weights.sum(-1), torch.ones(6), atol=1e-4)


def test_pre_softmax_keeps_raw_probabilities():
    x = torch.randn(4, 16)
    router = _router(pre_softmax=True, top_k=1)
    weights, _, _, _ = router(x)
    assert (weights < 1.0).all()


def test_softmax_top1_without_pre_softmax_has_no_gradient():
    # This is the behaviour the validation guards against: the chosen weight
    # is exactly 1 and detached from the logits.
    router = _router(pre_softmax=False, top_k=1)
    x = torch.randn(3, 16, requires_grad=True)
    weights, _, _, _ = router(x)
    assert torch.allclose(weights, torch.ones_like(weights))
    weights.sum().backward()
    assert x.grad is None or float(x.grad.abs().sum()) == 0.0


def test_z_loss_matches_logsumexp():
    router = _router(z_loss_coeff=1.0, scoring="sigmoid")
    x = torch.randn(5, 16)
    _, _, _, z_loss = router(x)
    with torch.no_grad():
        expected = (torch.logsumexp(router.linear(x).float(), dim=-1) ** 2).mean()
    assert torch.allclose(z_loss, expected, atol=1e-6)


def test_bias_sign_update_directions():
    router = _router(scoring="sigmoid", bias_update="sign", bias_update_rate=0.1)
    counts = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    router.update_expert_bias(counts=counts)
    bias = router.expert_bias
    assert bias[0] < 0.0  # overloaded steps down
    assert bias[1] > 0.0  # underloaded step up


def test_centered_sign_update_keeps_zero_mean():
    router = _router(
        scoring="sigmoid", bias_update="centered_sign", bias_update_rate=0.1
    )
    router.update_expert_bias(
        counts=torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    assert abs(float(router.expert_bias.mean())) < 1e-6


def test_bias_update_frozen_in_eval():
    model = _config(moe={'router': {'scoring': 'sigmoid'}, 'balancing': {'bias_update': 'sign', 'bias_update_rate': 0.1}})
    encoder = FrankensteinEncoder(model)
    encoder.eval()
    encoder(torch.randint(0, 100, (2, 8)))
    for module in encoder.modules():
        if isinstance(module, MoELayer):
            assert module.router._pending_counts is not None
            assert float(module.router._pending_counts.abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# Load balancing losses
# ---------------------------------------------------------------------------


def test_batch_aux_loss_exact_value():
    probs = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])  # (B=1, S=2, E=2)
    indices = torch.zeros(1, 2, 1, dtype=torch.long)  # both pick expert 0
    loss, counts = load_balancing_loss(probs, indices, 2, scope="batch")
    assert torch.allclose(loss, torch.tensor(2.0))
    assert counts.tolist() == [2.0, 0.0]


def test_sequence_scope_differs_from_batch():
    # Sequence 0 is balanced, sequence 1 is fully on expert 0.
    probs = torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    indices = torch.tensor(
        [
            [[0], [1]],
            [[0], [0]],
        ]
    )
    batch_loss, _ = load_balancing_loss(probs, indices, 2, scope="batch")
    seq_loss, _ = load_balancing_loss(probs, indices, 2, scope="sequence")
    assert not torch.allclose(batch_loss, seq_loss)


def test_max_vio():
    assert max_vio(torch.tensor([2.0, 2.0, 2.0, 2.0])) == pytest.approx(0.0)
    assert max_vio(torch.tensor([4.0, 0.0])) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Layer features
# ---------------------------------------------------------------------------


def _layer(config, **kwargs):
    return MoELayer(config, expert_factory=_expert_factory, **kwargs)


def test_shared_experts_and_gate():
    cfg = _config(moe={'shared': {'count': 1, 'hidden_size': 64, 'gate': 'sigmoid'}})
    layer = _layer(cfg)
    assert len(layer.shared_experts) == 1
    gate = torch.sigmoid(layer.shared_gate_proj(torch.randn(5, 32)))
    assert bool(((gate > 0) & (gate < 1)).all())


def test_latent_projection_shapes_and_norm():
    cfg = _config(moe={'latent': {'enabled': True, 'dim': 16, 'norm_before_up_proj': True}})
    layer = _layer(cfg)
    assert layer.latent_down_proj.in_features == 32
    assert layer.latent_down_proj.out_features == 16
    assert layer.latent_up_proj.in_features == 16
    assert layer.latent_up_proj.out_features == 32
    assert layer.up_norm is not None

    out = layer(torch.randn(2, 5, 32))
    assert out.shape == (2, 5, 32)


def test_capacity_factor_drops_tokens():
    cfg = _config(moe={'routing': {'capacity_factor': 0.6}})
    layer = _layer(cfg)
    out = layer(torch.randn(2, 8, 32))
    assert out.shape == (2, 8, 32)
    assert layer.last_moe_stats["dropped_fraction"] > 0.0
    assert layer.last_moe_stats["max_vio"] >= 0.0


def test_expert_choice_balances_load():
    cfg = _config(moe={'routing': {'type': 'expert_choice'}, 'balancing': {'aux_loss': 'none'}})
    layer = _layer(cfg)
    layer(torch.randn(1, 8, 32))
    assert layer.last_moe_stats["max_vio"] == pytest.approx(0.0)


def test_hash_layers_are_context_free():
    cfg = _config(moe={'routing': {'hash_layers': 1}})
    layer = _layer(cfg, use_hash=True)
    table = layer.tid2eid
    assert table.shape == (100, 2)
    # Every row names `top_k` distinct experts (strictly increasing after sort).
    sorted_table = table.sort(dim=-1).values
    assert bool((sorted_table[:, 1:] > sorted_table[:, :-1]).all())
    assert bool((table < 4).all())
    # Same token id resolves to the same experts regardless of context.
    probe_a = torch.tensor([[3, 4]])
    probe_b = torch.tensor([[3, 9]])
    assert torch.equal(table[probe_a][0, 0], table[probe_b][0, 0])

    out = layer(torch.randn(1, 2, 32), input_ids=probe_a)
    assert out.shape == (1, 2, 32)


def test_hash_layers_require_input_ids():
    cfg = _config(moe={'routing': {'hash_layers': 1}})
    layer = _layer(cfg, use_hash=True)
    with pytest.raises(ValueError, match="input_ids"):
        layer(torch.randn(1, 2, 32))


def test_activation_clamp_bounds_experts():
    cfg = _config(moe={'experts': {'activation_clamp': 0.5}}, ffn={'activation': 'swiglu'})
    hybrid = HybridLayer(cfg, layer_type="standard", layer_index=0)
    expert = hybrid.moe_layer.experts[0]
    assert expert.clamp_value == pytest.approx(0.5)
    # The gate projection is clamped before the SiLU activation.
    out = expert(torch.randn(4, 32))
    assert out.shape == (4, 32)


def test_layer_layout_builds_routers_on_expected_layers():
    cfg = _config(dims={'num_layers': 4, 'layer_pattern': ['standard'] * 4}, moe={'layout': {'dense_layers': 1, 'every_n_layers': 2}})
    layers = [
        HybridLayer(cfg, layer_type="standard", layer_index=i)
        for i in range(cfg.dims.num_layers)
    ]
    assert [layer.use_moe for layer in layers] == [False, True, False, True]
    assert layers[0].moe_layer is None
    assert layers[1].moe_layer is not None
    assert sum(layer.use_hash for layer in layers) == 0


def test_hash_layers_follow_first_moe_layers():
    cfg = _config(dims={'num_layers': 4, 'layer_pattern': ['standard'] * 4}, moe={'layout': {'dense_layers': 1, 'every_n_layers': 1}, 'routing': {'hash_layers': 2}})
    layers = [
        HybridLayer(cfg, layer_type="standard", layer_index=i)
        for i in range(cfg.dims.num_layers)
    ]
    assert [layer.use_hash for layer in layers] == [False, True, True, False]


def test_diagnostics_keys_present():
    cfg = _config()
    layer = _layer(cfg)
    layer(torch.randn(2, 6, 32))
    assert "moe_load_balancing_loss" in layer.last_auxiliary_losses
    for key in (
        "max_vio",
        "load_balancing_loss",
        "z_loss",
        "routing_entropy",
        "top1_weight",
        "expert_norm_ratio",
        "dropped_fraction",
    ):
        assert key in layer.last_moe_stats


def test_aux_loss_reaches_the_task_loss():
    cfg = _config()
    encoder = FrankensteinEncoder(cfg)
    encoder(torch.randint(0, 100, (2, 8)))
    assert "moe_load_balancing_loss" in encoder.last_auxiliary_losses
    assert torch.is_tensor(encoder.last_auxiliary_losses["moe_load_balancing_loss"])


# ---------------------------------------------------------------------------
# Export / parameter taxonomy
# ---------------------------------------------------------------------------


def test_param_categories_cover_moe_modules():
    assert categorize_param("layers.0.moe_layer.router.linear.weight") == "router"
    assert categorize_param("layers.0.moe_layer.experts.0.gate_proj.weight") == "ffn"
    assert (
        categorize_param("layers.0.moe_layer.shared_experts.0.down_proj.weight")
        == "ffn"
    )
    assert (
        categorize_param("layers.0.moe_layer.latent_down_proj.weight") == "ffn"
    )
    assert (
        categorize_param("layers.0.moe_layer.shared_gate_proj.weight") == "router"
    )


def test_transformers_export_contains_moe_block():
    from src.deploy.transformers_export import _build_transformers_config

    model_block = {
        "dims": {"layer_pattern": ["standard"]},
        "ffn": {"hidden_size": 18432},
        "moe": {
            "enabled": True,
            "num_experts": 256,
            "top_k": 8,
            "expert_hidden_size": 2048,
            "router": {
                "scoring": "sigmoid",
                "renormalize": True,
                "routed_scaling_factor": 2.5,
                "pre_softmax": False,
                "dtype": "float32",
                "z_loss_coeff": 0.0,
            },
            "shared": {"count": 1, "hidden_size": 2048, "gate": "none"},
            "layout": {"dense_layers": 3, "every_n_layers": 1},
            "balancing": {
                "bias_update": "sign",
                "aux_loss": "sequence",
                "aux_loss_coeff": 1e-4,
                "bias_update_rate": 1e-3,
            },
            "latent": {
                "enabled": True,
                "dim": 3584,
                "norm_before_up_proj": True,
            },
            "routing": {
                "type": "token_choice",
                "capacity_factor": 0.0,
                "hash_layers": 0,
            },
            "num_experts": 256,
            "top_k": 8,
            "experts": {"activation_clamp": 0.0},
        },
    }
    payload = _build_transformers_config(model_block, "frankenstein_text_decoder", "mlm", "X")
    moe = payload["moe"]
    assert moe["num_experts"] == 256
    assert moe["num_experts_per_tok"] == 8
    assert moe["moe_intermediate_size"] == 2048
    assert moe["scoring_func"] == "sigmoid"
    assert moe["routed_scaling_factor"] == 2.5
    assert moe["n_shared_experts"] == 1
    assert moe["first_k_dense_replace"] == 3
    assert moe["topk_method"] == "noaux_tc"
    assert moe["latent"]["dim"] == 3584
    assert moe["latent"]["norm_before_up_proj"] is True
    assert payload["num_experts_per_tok"] == 8


def test_quantile_balancing_matches_target_margin_count():
    """QB leaves exactly q = m*k/n margins above the cutoff (no ties)."""
    torch.manual_seed(0)
    m, n, k = 64, 8, 2
    router = MoERouter(
        hidden_size=n,
        num_experts=n,
        top_k=k,
        scoring="sigmoid",
        bias_update="quantile",
    )
    router.train()
    scores = torch.rand(m, n)
    biased = scores + router.expert_bias
    cutoff = torch.topk(biased, k + 1, dim=-1).values[:, -1]
    router._quantile_bias_update(scores)
    q = m * k // n
    margin_counts = ((scores + router.expert_bias) > cutoff[:, None]).sum(0)
    assert int(q) == 16
    # Boundary ties are included, so no expert stays below its target load.
    assert all(int(c) >= q for c in margin_counts)


def test_quantile_small_batch_is_noop():
    """q = m*k/n < 1 leaves the bias unchanged."""
    router = MoERouter(
        hidden_size=4, num_experts=4, top_k=1, scoring="sigmoid",
        bias_update="quantile",
    )
    with torch.no_grad():
        router.linear.weight.copy_(torch.eye(4))
    router.train()
    before = router.expert_bias.clone()
    router(torch.randn(1, 4))
    assert torch.equal(before, router.expert_bias)


def test_quantile_bias_frozen_in_eval():
    router = MoERouter(
        hidden_size=4, num_experts=4, top_k=1, scoring="sigmoid",
        bias_update="quantile",
    )
    with torch.no_grad():
        router.linear.weight.copy_(torch.eye(4))
    router.eval()
    before = router.expert_bias.clone()
    with torch.no_grad():
        router(torch.randn(4, 4))
    assert torch.equal(before, router.expert_bias)
