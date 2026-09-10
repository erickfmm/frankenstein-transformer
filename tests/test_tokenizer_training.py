"""Tests for src/tokenizer/train_tokenizer.py (train_from_dataset mode)."""
from __future__ import annotations

import os
import unittest
from importlib.util import find_spec

from src.tokenizer.train_tokenizer import (
    DEFAULT_SPECIAL_TOKENS,
    train_tokenizer_from_iterator,
)

TOKENIZERS_AVAILABLE = find_spec("tokenizers") is not None
TRANSFORMERS_AVAILABLE = find_spec("transformers") is not None

TEXTS = [
    "el rápido zorro marrón salta sobre el perro perezoso",
    "la inteligencia artificial aprende patrones complejos",
    "los transformers procesan secuencias con atención",
    "entrenar un modelo requiere datos y paciencia",
    "el zorro marrón salta alto y el perro ladra fuerte",
] * 5


@unittest.skipUnless(TOKENIZERS_AVAILABLE, "tokenizers library not installed")
class TrainTokenizerAlgorithmsTests(unittest.TestCase):
    """Each supported algorithm trains a usable tokenizer."""

    def test_train_bpe(self):
        tok = train_tokenizer_from_iterator(
            TEXTS, algorithm="bpe", vocab_size=200,
            special_tokens=["[PAD]", "[UNK]", "[MASK]"],
        )
        self.assertEqual(tok.mask_token, "[MASK]")
        self.assertEqual(tok.pad_token, "[PAD]")
        self.assertGreaterEqual(len(tok), 3)
        enc = tok("el zorro salta")
        decoded = tok.decode(enc["input_ids"]).strip()
        self.assertIn("zorro", decoded)

    def test_train_wordpiece(self):
        tok = train_tokenizer_from_iterator(
            TEXTS, algorithm="wordpiece", vocab_size=200,
            special_tokens=["[PAD]", "[UNK]", "[MASK]"],
        )
        self.assertGreaterEqual(len(tok), 3)
        self.assertIn("zorro", tok("el zorro")["input_ids"] is not None and tok.get_vocab())

    def test_train_wordlevel(self):
        tok = train_tokenizer_from_iterator(
            TEXTS, algorithm="wordlevel", vocab_size=200,
            special_tokens=["[PAD]", "[UNK]", "[MASK]"],
        )
        self.assertGreaterEqual(len(tok), 3)
        enc = tok("zorro")
        self.assertNotEqual(enc["input_ids"], [])

    def test_train_unigram_without_vocab_size(self):
        tok = train_tokenizer_from_iterator(
            TEXTS, algorithm="unigram", special_tokens=["[PAD]", "[UNK]", "[MASK]"],
        )
        self.assertGreaterEqual(len(tok), 3)
        self.assertEqual(tok.mask_token, "[MASK]")


@unittest.skipUnless(TOKENIZERS_AVAILABLE, "tokenizers library not installed")
class TrainTokenizerOptionsTests(unittest.TestCase):
    """Normalizer / pre-tokenizer / persistence options."""

    def test_lowercase_normalizer(self):
        tok = train_tokenizer_from_iterator(
            ["HOLA MUNDO"] * 20, algorithm="bpe", vocab_size=100, lowercase=True,
        )
        enc = tok("hola mundo")
        self.assertNotIn(tok.unk_token_id, enc["input_ids"])

    def test_strip_accents(self):
        tok = train_tokenizer_from_iterator(
            ["canción acción"] * 20, algorithm="bpe", vocab_size=100, strip_accents=True,
        )
        enc = tok("cancion accion")
        self.assertNotIn(tok.unk_token_id, enc["input_ids"])

    def test_bytelevel_pre_tokenizer(self):
        tok = train_tokenizer_from_iterator(
            TEXTS, algorithm="bpe", vocab_size=256, pre_tokenizer="bytelevel",
        )
        # ByteLevel round-trips any text losslessly.
        enc = tok("héllo wörld 你好")
        self.assertNotIn(tok.unk_token_id, enc["input_ids"])

    def test_default_special_tokens(self):
        tok = train_tokenizer_from_iterator(TEXTS, algorithm="bpe", vocab_size=200)
        for token in DEFAULT_SPECIAL_TOKENS:
            self.assertIsNotNone(getattr(tok, _SPECIAL_ATTRS[token]))

    def test_save_and_reload_roundtrip(self):
        from transformers import AutoTokenizer

        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_tok_save_test"
        )
        self.addCleanup(_rmtree, save_dir)
        tok = train_tokenizer_from_iterator(
            TEXTS, algorithm="bpe", vocab_size=200, save_dir=save_dir,
        )
        reloaded = AutoTokenizer.from_pretrained(save_dir)
        self.assertEqual(len(reloaded), len(tok))
        self.assertEqual(
            reloaded("el zorro")["input_ids"], tok("el zorro")["input_ids"]
        )


@unittest.skipUnless(TOKENIZERS_AVAILABLE, "tokenizers library not installed")
class TrainTokenizerValidationTests(unittest.TestCase):
    """Invalid arguments raise clear errors."""

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError):
            train_tokenizer_from_iterator(TEXTS, algorithm="gpt2", vocab_size=100)

    def test_missing_vocab_size_raises_for_bpe(self):
        with self.assertRaises(ValueError):
            train_tokenizer_from_iterator(TEXTS, algorithm="bpe", vocab_size=0)

    def test_empty_corpus_raises(self):
        with self.assertRaises(ValueError):
            train_tokenizer_from_iterator([], algorithm="bpe", vocab_size=100)

    def test_unknown_pre_tokenizer_raises(self):
        with self.assertRaises(ValueError):
            train_tokenizer_from_iterator(
                TEXTS, algorithm="bpe", vocab_size=100, pre_tokenizer="gpt4"
            )


_SPECIAL_ATTRS = {
    "[PAD]": "pad_token",
    "[UNK]": "unk_token",
    "[CLS]": "cls_token",
    "[SEP]": "sep_token",
    "[MASK]": "mask_token",
}


def _rmtree(path: str) -> None:
    import shutil

    if os.path.isdir(path):
        shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()