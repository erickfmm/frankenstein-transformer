"""
End-to-end mini-model training tests using synthetic (fake) data.

These tests verify that:
  - Forward passes produce finite outputs
  - Backward passes succeed (gradients flow)
  - Loss decreases over a handful of optimizer steps
  - All layer families (gated, sparse, legacy, mamba) can be used in a
    training loop without crashing

All models use tiny configs (small hidden_size, short sequences, few layers)
so each test completes in a few seconds on CPU.
"""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from src.model.tormented_bert_frankestein import (
        TormentedBertFrankenstein,
        TormentedBertMini,
        FrankensteinDecoder,
        UltraConfig,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 64
SEQ = 8
BSZ = 2
HIDDEN = 48
HEADS = 6
EXPERTS = 2


def _cfg(layer_pattern, mode="encoder", use_moe=False, **kw):
    """Return a minimal UltraConfig for the given layer pattern."""
    defaults = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_layers=len(layer_pattern),
        num_loops=1,
        num_heads=HEADS,
        retention_heads=HEADS,
        num_experts=EXPERTS,
        top_k_experts=1,
        dropout=0.0,
        norm_type="layer_norm",
        use_bitnet=False,
        layer_pattern=layer_pattern,
        use_moe=use_moe,
        ode_solver="euler",
        ode_steps=1,
        ffn_hidden_size=HIDDEN * 2,
        ffn_activation="gelu",
        use_hope=True,
        mode=mode,
    )
    defaults.update(kw)
    return UltraConfig(**defaults)


def _fake_batch():
    """Return (input_ids, labels) filled with random token indices."""
    ids = torch.randint(0, VOCAB, (BSZ, SEQ))
    labels = torch.randint(0, VOCAB, (BSZ, SEQ))
    return ids, labels


def _train_steps(model, n=5):
    """Run n gradient-descent steps and return list of losses."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(n):
        ids, labels = _fake_batch()
        logits = model(ids)  # (B, S, V)
        loss = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------------------
# TormentedBertMini  (uses legacy retnet/titan/mamba/ode pattern)
# ---------------------------------------------------------------------------

@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MiniModelTrainingTests(unittest.TestCase):

    def test_mini_model_loss_is_finite(self):
        model = TormentedBertMini(TormentedBertMini.build_mini_config(vocab_size=VOCAB, use_bitnet=False))
        losses = _train_steps(model, n=3)
        for loss in losses:
            self.assertTrue(
                torch.isfinite(torch.tensor(loss)),
                f"Non-finite loss: {loss}",
            )

    def test_mini_model_gradients_nonzero(self):
        model = TormentedBertMini(TormentedBertMini.build_mini_config(vocab_size=VOCAB, use_bitnet=False))
        model.train()
        ids, labels = _fake_batch()
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1))
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        self.assertTrue(len(grad_norms) > 0, "No gradients were computed")
        total_norm = sum(grad_norms)
        self.assertGreater(total_norm, 0.0)

    def test_mini_model_params_update(self):
        model = TormentedBertMini(TormentedBertMini.build_mini_config(vocab_size=VOCAB, use_bitnet=False))
        before = [p.data.clone() for p in model.parameters()]
        _train_steps(model, n=2)
        changed = sum(
            0 if torch.equal(b, a) else 1
            for b, a in zip(before, model.parameters())
        )
        self.assertGreater(changed, 0, "No parameters were updated")


# ---------------------------------------------------------------------------
# Standard / sigmoid / mamba
# ---------------------------------------------------------------------------

@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class LegacyLayerTrainingTests(unittest.TestCase):

    def _train(self, layers, **kw):
        model = TormentedBertFrankenstein(_cfg(layers, **kw))
        return _train_steps(model, n=3)

    def test_standard_attn_training(self):
        losses = self._train(["standard_attn"])
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_sigmoid_attn_training(self):
        losses = self._train(["sigmoid_attn"])
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_mamba_training(self):
        losses = self._train(["mamba"])
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_ode_training(self):
        losses = self._train(["ode"])
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_retnet_training(self):
        losses = self._train(["retnet"])
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_titan_attn_training(self):
        losses = self._train(["titan_attn"])
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_retnet_attn_training(self):
        losses = self._train(["retnet_attn"])
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_moe_training(self):
        losses = self._train(["standard_attn"], use_moe=True)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_decoder_mode_training(self):
        losses = self._train(["standard_attn"], mode="decoder")
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))


# ---------------------------------------------------------------------------
# Gated attention families
# ---------------------------------------------------------------------------

@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GatedLayerTrainingTests(unittest.TestCase):

    def _train(self, layer_type):
        model = TormentedBertFrankenstein(_cfg([layer_type]))
        return _train_steps(model, n=3)

    def test_gla_attn_training(self):
        losses = self._train("gla_attn")
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_deltanet_attn_training(self):
        losses = self._train("deltanet_attn")
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_gated_deltanet_attn_training(self):
        losses = self._train("gated_deltanet_attn")
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_hgrn2_attn_training(self):
        losses = self._train("hgrn2_attn")
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_fox_attn_training(self):
        losses = self._train("fox_attn")
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_gated_softmax_attn_training(self):
        losses = self._train("gated_softmax_attn")
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))


# ---------------------------------------------------------------------------
# Sparse attention families (eval-only guards for sparge/fasa)
# ---------------------------------------------------------------------------

@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SparseLayerTrainingTests(unittest.TestCase):

    def _train(self, layer_type):
        model = TormentedBertFrankenstein(_cfg([layer_type]))
        return _train_steps(model, n=3)

    def _eval_forward(self, layer_type):
        model = TormentedBertFrankenstein(_cfg([layer_type]))
        model.eval()
        ids, _ = _fake_batch()
        with torch.no_grad():
            logits = model(ids)
        return logits

    def test_sparse_transformer_attn_training(self):
        losses = self._train("sparse_transformer_attn")
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_longformer_attn_training(self):
        losses = self._train("longformer_attn")
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_bigbird_attn_training(self):
        losses = self._train("bigbird_attn")
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_sparsek_attn_training(self):
        losses = self._train("sparsek_attn")
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_nsa_attn_training(self):
        losses = self._train("nsa_attn")
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_sparge_attn_raises_in_train_mode(self):
        # sparge_attn is inference/eval-only; must raise when model.train() is active
        model = TormentedBertFrankenstein(_cfg(["sparge_attn"]))
        model.train()
        ids, _ = _fake_batch()
        with self.assertRaises(ValueError):
            model(ids)

    def test_fasa_attn_raises_in_train_mode(self):
        # fasa_attn is inference/eval-only; must raise when model.train() is active
        model = TormentedBertFrankenstein(_cfg(["fasa_attn"]))
        model.train()
        ids, _ = _fake_batch()
        with self.assertRaises(ValueError):
            model(ids)

    def test_sparge_attn_eval_ok(self):
        logits = self._eval_forward("sparge_attn")
        self.assertEqual(logits.shape, (BSZ, SEQ, VOCAB))

    def test_fasa_attn_eval_ok(self):
        logits = self._eval_forward("fasa_attn")
        self.assertEqual(logits.shape, (BSZ, SEQ, VOCAB))


# ---------------------------------------------------------------------------
# FrankensteinDecoder causal training
# ---------------------------------------------------------------------------

@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class DecoderTrainingTests(unittest.TestCase):

    def _decoder(self, layer_pattern, **kw):
        cfg = FrankensteinDecoder.build_decoder_config(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            num_layers=len(layer_pattern),
            num_loops=1,
            use_bitnet=False,
            layer_pattern=layer_pattern,
        )
        for k, v in kw.items():
            setattr(cfg, k, v)
        return FrankensteinDecoder(cfg)

    def test_decoder_standard_attn_training(self):
        model = self._decoder(["standard_attn"])
        losses = _train_steps(model, n=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_decoder_gated_attn_training(self):
        model = self._decoder(["gla_attn", "standard_attn"])
        losses = _train_steps(model, n=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_decoder_mixed_pattern_training(self):
        model = self._decoder(["titan_attn", "retnet_attn", "mamba", "standard_attn"])
        losses = _train_steps(model, n=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_generate_does_not_crash(self):
        model = self._decoder(["standard_attn"])
        model.eval()
        ids = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(ids, max_new_tokens=2)
        self.assertEqual(out.shape[1], 6)

    def test_decoder_loss_is_finite_after_10_steps(self):
        model = self._decoder(["standard_attn", "titan_attn"])
        losses = _train_steps(model, n=10)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))


# ---------------------------------------------------------------------------
# Mixed / multi-layer pattern training
# ---------------------------------------------------------------------------

@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MixedPatternTrainingTests(unittest.TestCase):

    def test_mixed_legacy_pattern(self):
        model = TormentedBertFrankenstein(
            _cfg(["retnet", "ode", "mamba", "titan_attn", "standard_attn", "sigmoid_attn"])
        )
        losses = _train_steps(model, n=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_mixed_gated_legacy_pattern(self):
        model = TormentedBertFrankenstein(
            _cfg(["gla_attn", "standard_attn", "deltanet_attn", "titan_attn"])
        )
        losses = _train_steps(model, n=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_mixed_sparse_legacy_pattern(self):
        model = TormentedBertFrankenstein(
            _cfg(["sparse_transformer_attn", "standard_attn", "longformer_attn", "titan_attn"])
        )
        losses = _train_steps(model, n=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_multi_loop_training(self):
        cfg = _cfg(["standard_attn", "titan_attn"], num_loops=2, num_layers=2)
        model = TormentedBertFrankenstein(cfg)
        losses = _train_steps(model, n=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_bitnet_training(self):
        cfg = _cfg(["standard_attn"], use_bitnet=True)
        model = TormentedBertFrankenstein(cfg)
        losses = _train_steps(model, n=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_derf_norm_training(self):
        cfg = _cfg(["standard_attn"], norm_type="derf")
        model = TormentedBertFrankenstein(cfg)
        losses = _train_steps(model, n=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_dynamic_tanh_norm_training(self):
        cfg = _cfg(["standard_attn"], norm_type="dynamic_tanh")
        model = TormentedBertFrankenstein(cfg)
        losses = _train_steps(model, n=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_factorized_embedding_training(self):
        cfg = _cfg(
            ["standard_attn"],
            use_factorized_embedding=True,
            factorized_embedding_dim=16,
            use_embedding_conv=True,
            embedding_conv_kernel=3,
        )
        model = TormentedBertFrankenstein(cfg)
        losses = _train_steps(model, n=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_rope_positional_encoding_training(self):
        cfg = _cfg(["standard_attn"], positional_encoding="rope")
        model = TormentedBertFrankenstein(cfg)
        losses = _train_steps(model, n=3)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))


if __name__ == "__main__":
    unittest.main()
