"""Tests for the transformers 4/5 checkpoint-loading compat helpers.

``src.utils.hf_compat`` restores the transformers-4 resolution fallbacks
that transformers 5 dropped: legacy checkpoints whose ``config.json`` lacks
``model_type`` (e.g. ``prajjwal1/bert-tiny``) fail with an opaque backend
ValueError under ``AutoTokenizer``/``AutoModelFor*`` on transformers 5.
These tests exercise the prefix derivation and the explicit-class retry
fully offline, using a local checkpoint directory (no hub access) and
class/module-level patches that survive the repo's dual module identity
(``utils.*`` vs ``src.utils.*``).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from utils import hf_compat  # noqa: E402

transformers = pytest.importorskip("transformers")  # noqa: E402 - helpers are transformers-only


def _active_transformers():
    """The module object ``import transformers`` resolves at call time.

    Other test modules (e.g. the SBERT engine dispatch tests) can leave
    ``sys.modules['transformers']`` pointing at a re-executed copy, so the
    helper's lazy ``import transformers`` may resolve a different object
    than this test module's binding. Patch targets must use the active one.
    """
    return sys.modules["transformers"]


def _local_checkpoint(config: dict) -> str:
    """Write ``config`` as a local checkpoint dir's ``config.json``."""
    directory = tempfile.mkdtemp()
    with open(Path(directory) / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    return directory


class ArchitecturePrefixTests(unittest.TestCase):
    def test_prefix_from_architectures(self):
        cases = {
            "BertForMaskedLM": "Bert",
            "LlamaForCausalLM": "Llama",
            "GPT2LMHeadModel": "GPT2",
            "T5ForConditionalGeneration": "T5",
            "ViTForImageClassification": "ViT",
        }
        for arch, expected in cases.items():
            directory = _local_checkpoint({"architectures": [arch]})
            self.assertEqual(hf_compat._architecture_prefix(directory), expected)

    def test_prefix_from_legacy_bert_signature(self):
        directory = _local_checkpoint(
            {"type_vocab_size": 2, "num_hidden_layers": 2, "hidden_size": 128}
        )
        self.assertEqual(hf_compat._architecture_prefix(directory), "Bert")

    def test_prefix_none_when_unresolvable(self):
        directory = _local_checkpoint({"hidden_size": 128})
        self.assertIsNone(hf_compat._architecture_prefix(directory))

    def test_prefix_ignores_bare_suffix_names(self):
        directory = _local_checkpoint({"architectures": ["Model"]})
        self.assertIsNone(hf_compat._architecture_prefix(directory))


class LoadAutoTokenizerTests(unittest.TestCase):
    def test_fallback_retries_with_explicit_class(self):
        directory = _local_checkpoint({"architectures": ["StubbedForMaskedLM"]})
        boom = ValueError("Couldn't instantiate the backend tokenizer")
        stub = type(
            "StubbedTokenizer",
            (),
            {
                "from_pretrained": classmethod(
                    lambda cls, name, **kwargs: "stub-tokenizer"
                )
            },
        )
        active = _active_transformers()
        with mock.patch.object(
            active.AutoTokenizer, "from_pretrained", side_effect=boom
        ), mock.patch.object(
            active, "StubbedTokenizer", stub, create=True
        ):
            result = hf_compat.load_auto_tokenizer(directory)
        self.assertEqual(result, "stub-tokenizer")

    def test_fallback_re_raises_when_no_class(self):
        directory = _local_checkpoint({"hidden_size": 128})
        boom = ValueError("Couldn't instantiate the backend tokenizer")
        with mock.patch.object(
            transformers.AutoTokenizer, "from_pretrained", side_effect=boom
        ):
            with self.assertRaises(ValueError):
                hf_compat.load_auto_tokenizer(directory)

    def test_success_path_skips_fallback(self):
        sentinel = object()
        with mock.patch.object(
            transformers.AutoTokenizer, "from_pretrained", return_value=sentinel
        ) as auto_call:
            result = hf_compat.load_auto_tokenizer(
                "repo", use_fast=True, trust_remote_code=True
            )
        self.assertIs(result, sentinel)
        auto_call.assert_called_once_with(
            "repo", use_fast=True, trust_remote_code=True
        )


class LoadAutoModelTests(unittest.TestCase):
    def test_fallback_retries_with_explicit_class(self):
        directory = _local_checkpoint({"architectures": ["StubForCausalLM"]})
        boom = ValueError("Unrecognized model in repo")
        stub = type(
            "StubForCausalLM",
            (),
            {
                "from_pretrained": classmethod(
                    lambda cls, name, **kwargs: "stub-model"
                )
            },
        )
        active = _active_transformers()
        with mock.patch.object(
            active.AutoModelForCausalLM, "from_pretrained", side_effect=boom
        ), mock.patch.object(
            active, "StubForCausalLM", stub, create=True
        ):
            result = hf_compat.load_auto_model(directory, task="causal_lm")
        self.assertEqual(result, "stub-model")

    def test_fallback_re_raises_when_no_class(self):
        directory = _local_checkpoint({"hidden_size": 128})
        boom = ValueError("Unrecognized model in repo")
        with mock.patch.object(
            transformers.AutoModelForMaskedLM, "from_pretrained", side_effect=boom
        ):
            with self.assertRaises(ValueError):
                hf_compat.load_auto_model(directory, task="mlm")


if __name__ == "__main__":
    unittest.main()
