"""Unit tests for EELBERT-style dynamic token embeddings (arXiv:2310.20144)."""
import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

    from src.deploy.transformers_export import export_transformers_model
    from src.model.config import FrankensteinModelConfig
    from src.model.embeddings import DynamicEmbedding
    from src.model.frankenstein_decoder import FrankensteinDecoder
    from src.model.frankenstein_encoder import FrankensteinEncoder
    from src.utils.tokenizer_vocab import inject_token_vocab, token_strings


# Id 0..4 are special tokens, 5..7 are words used by the morphology test.
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "runner", "running", "banana"]


def _cfg(**overrides):
    base = dict(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_layers=1,
        num_loops=1,
        num_heads=4,
        retention_heads=4,
        num_experts=2,
        top_k_experts=1,
        dropout=0.0,
        norm_type="layer_norm",
        use_bitnet=False,
        use_moe=False,
        layer_pattern=["standard"],
        ode_solver="rk4",
        ode_steps=1,
        ffn_hidden_size=64,
        ffn_activation="gelu",
        positional_encoding="hope",
        use_factorized_embedding=False,
        use_embedding_conv=False,
        embedding_conv_kernel=3,
        use_dynamic_embedding=True,
        token_vocab=VOCAB,
    )
    base.update(overrides)
    return FrankensteinModelConfig.from_flat(**base)


class _HFTokenizerStub:
    """Minimal HuggingFace-like tokenizer exposing ``get_vocab``."""

    def get_vocab(self):
        return {"[PAD]": 0, "hello": 1, "world": 2}

    def __len__(self):
        return 3


class _SPMTokenizerStub:
    """Minimal SentencePiece-like tokenizer exposing ``vocab`` (token → id)."""

    def __init__(self):
        self.vocab = {"[PAD]": 0, "hola": 1, "mundo": 2}


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class DynamicEmbeddingTests(unittest.TestCase):
    def test_output_shape(self):
        emb = DynamicEmbedding(_cfg())
        ids = torch.randint(0, len(VOCAB), (2, 10))
        self.assertEqual(emb(ids).shape, (2, 10, 32))

    def test_no_trainable_parameters(self):
        emb = DynamicEmbedding(_cfg())
        self.assertEqual(sum(p.numel() for p in emb.parameters()), 0)

    def test_state_dict_has_no_embedding_table(self):
        # The memoised table is a non-persistent buffer, so nothing
        # embedding-shaped is saved in checkpoints.
        emb = DynamicEmbedding(_cfg())
        self.assertEqual(len(emb.state_dict()), 0)

    def test_values_are_bounded(self):
        emb = DynamicEmbedding(_cfg())
        ids = torch.randint(0, len(VOCAB), (2, 10))
        self.assertLessEqual(emb(ids).abs().max().item(), 1.0 + 1e-6)

    def test_same_seed_is_deterministic(self):
        ids = torch.randint(0, len(VOCAB), (2, 10))
        left = DynamicEmbedding(_cfg(dynamic_embedding_seed=42))(ids)
        right = DynamicEmbedding(_cfg(dynamic_embedding_seed=42))(ids)
        self.assertTrue(torch.equal(left, right))

    def test_different_seed_changes_embeddings(self):
        ids = torch.randint(0, len(VOCAB), (2, 10))
        left = DynamicEmbedding(_cfg(dynamic_embedding_seed=42))(ids)
        right = DynamicEmbedding(_cfg(dynamic_embedding_seed=43))(ids)
        self.assertFalse(torch.equal(left, right))

    def test_precompute_matches_on_the_fly(self):
        ids = torch.randint(0, len(VOCAB), (3, 7))
        cached = DynamicEmbedding(_cfg(dynamic_embedding_precompute=True))(ids)
        online = DynamicEmbedding(_cfg(dynamic_embedding_precompute=False))(ids)
        # Same deterministic function; the only difference is float32 reduction
        # order in the window mean (~1e-8).
        self.assertTrue(torch.allclose(cached, online, atol=1e-6, rtol=0.0))

    def test_morphology_similarity(self):
        # Paper §5.3: n-gram pooling keeps morphologically related words close.
        table = DynamicEmbedding(_cfg()).embedding_table()
        runner = table[5].unsqueeze(0)
        running = table[6].unsqueeze(0)
        banana = table[7].unsqueeze(0)
        self.assertGreater(
            F.cosine_similarity(runner, running).item(),
            F.cosine_similarity(runner, banana).item(),
        )

    def test_embedding_table_shape(self):
        table = DynamicEmbedding(_cfg()).embedding_table()
        self.assertEqual(tuple(table.shape), (len(VOCAB), 32))

    def test_conv_composition(self):
        emb = DynamicEmbedding(_cfg(use_embedding_conv=True, embedding_conv_kernel=4))
        self.assertIsNotNone(emb.conv)
        ids = torch.randint(0, len(VOCAB), (1, 9))
        self.assertEqual(emb(ids).shape, (1, 9, 32))

    def test_requires_vocab(self):
        with self.assertRaises(ValueError):
            DynamicEmbedding(_cfg(token_vocab=None))

    def test_vocab_length_mismatch(self):
        with self.assertRaises(ValueError):
            DynamicEmbedding(_cfg(token_vocab=VOCAB[:-1]))

    def test_factorized_is_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            _cfg(use_factorized_embedding=True)

    def test_invalid_ngram_max(self):
        with self.assertRaises(ValueError):
            _cfg(dynamic_embedding_ngram_max=0)

    def test_ngram_max_must_fit_hidden_size(self):
        with self.assertRaises(ValueError):
            _cfg(dynamic_embedding_ngram_max=64)

    def test_invalid_bucket(self):
        with self.assertRaises(ValueError):
            _cfg(dynamic_embedding_bucket=1)

    def test_encoder_forward_with_dynamic_embeddings(self):
        model = FrankensteinEncoder(_cfg())
        logits = model(torch.randint(0, len(VOCAB), (1, 5)))
        self.assertEqual(tuple(logits.shape), (1, 5, len(VOCAB)))

    def test_decoder_forward_with_dynamic_embeddings(self):
        model = FrankensteinDecoder(_cfg())
        logits = model(torch.randint(0, len(VOCAB), (1, 5)))
        self.assertEqual(tuple(logits.shape), (1, 5, len(VOCAB)))

    def test_nested_dynamic_block(self):
        config = FrankensteinModelConfig.from_dict(
            {
                "dims": {"hidden_size": 32},
                "embedding": {
                    "dynamic": {
                        "enabled": True,
                        "ngram_max": 2,
                        "bucket": 1000000007,
                        "seed": 7,
                        "precompute": False,
                    }
                },
            }
        )
        self.assertEqual(config.embedding.type, "dynamic")
        self.assertEqual(config.embedding.dynamic.ngram_max, 2)
        self.assertEqual(config.embedding.dynamic.bucket, 1000000007)
        self.assertEqual(config.embedding.dynamic.seed, 7)
        self.assertFalse(config.embedding.dynamic.precompute)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TokenizerVocabTests(unittest.TestCase):
    def test_hf_token_strings(self):
        self.assertEqual(token_strings(_HFTokenizerStub()), ["[PAD]", "hello", "world"])

    def test_spm_token_strings(self):
        self.assertEqual(token_strings(_SPMTokenizerStub()), ["[PAD]", "hola", "mundo"])

    def test_missing_tokenizer(self):
        self.assertIsNone(token_strings(None))

    def test_inject_is_noop_without_dynamic(self):
        cfg = _cfg(use_dynamic_embedding=False, token_vocab=None)
        inject_token_vocab(cfg, _HFTokenizerStub())
        self.assertIsNone(cfg.token_vocab)

    def test_inject_attaches_vocab(self):
        cfg = _cfg(use_dynamic_embedding=False, token_vocab=None)
        cfg.embedding.dynamic.enabled = True
        inject_token_vocab(cfg, _HFTokenizerStub())
        self.assertEqual(cfg.token_vocab, ["[PAD]", "hello", "world"])


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class DynamicEmbeddingExportTests(unittest.TestCase):
    def _write_inputs(self, tmp_path: Path, vocab_available: bool):
        """Create a checkpoint + YAML pair for the export path."""
        cfg = _cfg()
        model = FrankensteinEncoder(cfg)
        checkpoint_cfg = cfg
        if not vocab_available:
            # Simulate a checkpoint that never recorded the vocabulary.
            import dataclasses

            checkpoint_cfg = dataclasses.replace(cfg, token_vocab=None)

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(
            {"model_state_dict": model.state_dict(), "config": checkpoint_cfg},
            checkpoint_path,
        )

        yaml_path = tmp_path / "train.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    "model_class: frankenstein_text_encoder",
                    "model:",
                    "  dims:",
                    f"    vocab_size: {len(VOCAB)}",
                    "    hidden_size: 32",
                    "    num_layers: 1",
                    "    num_heads: 4",
                    "    layer_pattern: [standard]",
                    "    mode: encoder",
                    "  norm:",
                    "    type: layer_norm",
                    "  embedding:",
                    "    dynamic:",
                    "      enabled: true",
                    "      ngram_max: 3",
                    "      bucket: 1000000007",
                    "      seed: 42",
                    "      precompute: true",
                    "    conv:",
                    "      enabled: false",
                    "  moe:",
                    "    enabled: false",
                    "  bitnet:",
                    "    enabled: false",
                    "  ffn:",
                    "    hidden_size: 64",
                    "    activation: gelu",
                    "training:",
                    "  task: mlm",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return checkpoint_path, yaml_path

    def test_export_materializes_frozen_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path, yaml_path = self._write_inputs(tmp_path, vocab_available=True)
            out_dir = tmp_path / "exported"
            result = export_transformers_model(
                str(checkpoint_path), str(yaml_path), str(out_dir)
            )

            self.assertEqual(result["status"], "ok")
            from safetensors.torch import load_file

            state = load_file(str(out_dir / "model.safetensors"))
            self.assertIn("model.emb.weight", state)
            self.assertEqual(tuple(state["model.emb.weight"].shape), (len(VOCAB), 32))

            exported_config = json.loads((out_dir / "config.json").read_text())
            self.assertFalse(
                exported_config["frankenstein_config"]["use_dynamic_embedding"]
            )
            report = json.loads((out_dir / "compatibility_report.json").read_text())
            self.assertTrue(
                any("materialized" in warning for warning in report["warnings"])
            )

    def test_export_without_vocab_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path, yaml_path = self._write_inputs(tmp_path, vocab_available=False)
            with self.assertRaises(ValueError):
                export_transformers_model(
                    str(checkpoint_path), str(yaml_path), str(tmp_path / "exported")
                )


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class DynamicEmbeddingSchemaTests(unittest.TestCase):
    def _validator(self):
        from jsonschema import Draft202012Validator

        from src.utils.schema_loader import resolve_schema

        root = Path(__file__).resolve().parent.parent
        return Draft202012Validator(resolve_schema(str(root / "src" / "schema.yaml")))

    def _payload(self, **embedding):
        return {
            "model_class": "frankenstein_text_encoder",
            "model": {
                "dims": {
                    "hidden_size": 32,
                    "num_layers": 1,
                    "num_heads": 4,
                    "layer_pattern": ["standard"],
                },
                "embedding": embedding,
            },
            "training": {
                "task": "mlm",
                "optimizer": {
                    "optimizer_class": "adamw",
                    "parameters": {"adamw-lr_other": 1e-4},
                },
            },
        }

    def test_dynamic_alone_validates(self):
        errors = list(
            self._validator().iter_errors(self._payload(dynamic={"enabled": True}))
        )
        self.assertEqual(errors, [])

    def test_dynamic_with_factorized_is_rejected(self):
        errors = list(
            self._validator().iter_errors(
                self._payload(
                    dynamic={"enabled": True}, factorized={"enabled": True}
                )
            )
        )
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
