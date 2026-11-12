"""BitNet coverage tests: every gate is BitLinear, routers honour bitnet_routers.

Verifies that when ``use_bitnet=True`` all recurrent-state gate projections
(alpha/beta/forget/erase/write/merge/gk) become ``BitLinear`` instances, while
routing/scoring projections (MoE router, MoD router, sparse block-index,
forecast, top-k score nets) only become ``BitLinear`` when ``bitnet_routers``
is true. Covers encoder and decoder variants across all mixer families.
"""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch.nn as nn
    from src.model.attention.common import BitConv1d, BitLinear
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder
    from src.model.frankenstein_encoder import FrankensteinEncoder


def _make(bitnet_routers: bool, mode: str = "encoder"):
    return FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 64, "hidden_size": 64, "num_layers": 8, "num_heads": 4, "dropout": 0.0, "mode": mode, "layer_pattern": ['standard', 'gla', 'deltanet', 'gated_deltanet', 'fox', 'fpa', 'hgrn2', 'gated_deltanet2', 'kda', 'mtla', 'msa', 'sparda', 'lsa', 'nsa', 'sparsek', 'titan', 'retnet', 'ode', 'gated_softmax', 'sigmoid', 'ca', 'xsa', 'gqa', 'memory', 'mla', 'gqla', 'mlra', 'tucker', 'iha', 'gta', 'gma', 'fastconformer']}, "moe": {"num_experts": 4, "top_k": 2, "enabled": True}, "bitnet": {"enabled": True, "routers": bitnet_routers}, "mixture_of_depths": {"enabled": True}, "ffn": {"hidden_size": 128}})


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestBitNetCoverage(unittest.TestCase):
    """Gate projections must be BitLinear; routers respect bitnet_routers."""

    def _build(self, bitnet_routers: bool, model_cls, mode: str = "encoder"):
        cfg = _make(bitnet_routers, mode=mode)
        if model_cls is FrankensteinDecoder:
            return cfg, FrankensteinDecoder(cfg)
        return cfg, FrankensteinEncoder(cfg)

    def _gate_names(self, model):
        gates = {
            "alpha_proj", "beta_proj", "f_proj", "forget_proj",
            "erase_proj", "write_proj", "merge_proj", "gk_proj",
        }
        found = []
        for name, mod in model.named_modules():
            base = name.split(".")[-1]
            if base in gates:
                for child in mod.children():
                    found.append(child)
                # gk_proj is a Sequential; the projection is itself.
                if isinstance(mod, nn.Sequential):
                    found.extend(list(mod.children()))
                else:
                    found.append(mod)
        # Deduplicate while preserving identity check targets
        return found

    def _router_names(self, model):
        routers = []
        for name, mod in model.named_modules():
            base = name.split(".")[-1]
            if base in {"router", "depth_router", "q_idx_proj", "k_idx_proj",
                        "forecast_proj", "compress_k", "compress_v",
                        "score_net"}:
                if isinstance(mod, nn.Sequential):
                    routers.extend(
                        [c for c in mod.children() if isinstance(c, nn.Linear)]
                    )
                elif isinstance(mod, nn.Linear):
                    routers.append(mod)
        return routers

    def _assert_all_bitlinear(self, modules, expect_bitlinear: bool, msg: str):
        for mod in modules:
            with self.subTest(module=type(mod).__name__):
                self.assertEqual(isinstance(mod, BitLinear), expect_bitlinear, msg)

    def test_gates_are_bitlinear_when_use_bitnet_true(self):
        for model_cls in (FrankensteinEncoder, FrankensteinDecoder):
            with self.subTest(model=model_cls.__name__):
                _, model = self._build(False, model_cls)
                gates = self._gate_names(model)
                self.assertGreater(len(gates), 0)
                self._assert_all_bitlinear(
                    gates, True,
                    "gates must be BitLinear when use_bitnet=True",
                )

    def test_routers_float_when_bitnet_routers_false(self):
        _, model = self._build(False, FrankensteinEncoder)
        routers = self._router_names(model)
        self.assertGreater(len(routers), 0)
        for mod in routers:
            with self.subTest(module=type(mod).__name__):
                self.assertFalse(
                    isinstance(mod, BitLinear),
                    "routers must stay nn.Linear when bitnet_routers=False",
                )

    def test_routers_bitlinear_when_bitnet_routers_true(self):
        _, model = self._build(True, FrankensteinEncoder)
        routers = self._router_names(model)
        self.assertGreater(len(routers), 0)
        for mod in routers:
            with self.subTest(module=type(mod).__name__):
                self.assertTrue(
                    isinstance(mod, BitLinear),
                    "routers must be BitLinear when bitnet_routers=True",
                )

    def test_no_plain_linear_when_bitnet_routers_true(self):
        _, model = self._build(True, FrankensteinEncoder)
        plain = [
            m for m in model.modules()
            if isinstance(m, nn.Linear) and not isinstance(m, BitLinear)
        ]
        # Only norm/norm-alpha parameters (DynamicTanhNorm uses Linear-free
        # parameters) could remain; there should be no nn.Linear weights left.
        self.assertEqual(plain, [], "no plain nn.Linear should remain")

    def test_embedding_conv_is_bitconv_only_when_flag_true(self):
        """BitConv1d is opt-in via use_bitnet_conv; off by default."""
        from src.model.embeddings.factorized_embedding import FactorizedEmbedding

        def make(use_bitnet_conv):
            cfg = FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 64, "hidden_size": 64, "num_layers": 1, "num_heads": 4, "dropout": 0.0, "layer_pattern": ['standard']}, "moe": {"num_experts": 2, "top_k": 1, "enabled": False}, "bitnet": {"enabled": True, "conv": use_bitnet_conv}, "embedding": {"factorized": {"enabled": True, "dim": 32}, "conv": {"enabled": True}}, "ffn": {"hidden_size": 64}})
            return FactorizedEmbedding(cfg)

        off = make(False)
        self.assertIsInstance(off.conv, nn.Conv1d)
        self.assertNotIsInstance(off.conv, BitConv1d)

        on = make(True)
        self.assertIsInstance(on.conv, BitConv1d)

    def test_embedding_conv_unaffected_when_bitnet_false(self):
        """use_bitnet_conv has no effect when use_bitnet is False."""
        from src.model.embeddings.factorized_embedding import FactorizedEmbedding

        cfg = FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 64, "hidden_size": 64, "num_layers": 1, "num_heads": 4, "dropout": 0.0, "layer_pattern": ['standard']}, "moe": {"num_experts": 2, "top_k": 1, "enabled": False}, "bitnet": {"enabled": False, "conv": True}, "embedding": {"factorized": {"enabled": True, "dim": 32}, "conv": {"enabled": True}}, "ffn": {"hidden_size": 64}})
        emb = FactorizedEmbedding(cfg)
        self.assertNotIsInstance(emb.conv, BitConv1d)


if __name__ == "__main__":
    unittest.main()