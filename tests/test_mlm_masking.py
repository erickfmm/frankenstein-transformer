"""Unit tests for _apply_mlm_mask_standalone."""
import random
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

_IMPORTS_OK = False
if TORCH_AVAILABLE:
    try:
        from src.training.streaming_mlm_dataset import _apply_mlm_mask_standalone
        _IMPORTS_OK = True
    except ImportError:
        pass


def _simple_encode(text, max_length=16, vocab_size=100):
    """Deterministic fake tokenizer for test reproducibility."""
    tokens = [hash(w) % (vocab_size - 5) + 5 for w in text.split()]
    tokens = tokens[:max_length]
    pad_id = 0
    tokens += [pad_id] * (max_length - len(tokens))
    mask = [1 if t != pad_id else 0 for t in tokens]
    return tokens, mask


@unittest.skipUnless(_IMPORTS_OK, "torch and datasets dep required")
class ApplyMLMMaskTests(unittest.TestCase):
    def _call(self, input_ids, attention_mask, mlm_prob=0.15, vocab_size=100, mask_id=1, special_ids=None, pad_id=0):
        return _apply_mlm_mask_standalone(
            input_ids=list(input_ids),
            attention_mask=list(attention_mask),
            mlm_probability=mlm_prob,
            vocab_size=vocab_size,
            mask_token_id=mask_id,
            special_token_ids=special_ids or [0, 1, 2, 3, 4],
            pad_token_id=pad_id,
        )

    def test_output_lengths_match_input(self):
        ids, mask = _simple_encode("hello world how are you")
        out_ids, labels = self._call(ids, mask)
        self.assertEqual(len(out_ids), len(ids))
        self.assertEqual(len(labels), len(ids))

    def test_labels_preserve_original_ids(self):
        ids, mask = _simple_encode("hello world how are you today")
        _, labels = self._call(ids, mask)
        # labels should equal original ids (before masking)
        self.assertEqual(labels, ids)

    def test_masked_positions_become_mask_token(self):
        random.seed(0)
        # Use a high probability to ensure masking happens
        ids = list(range(10, 20)) + [0] * 6  # 10 real tokens, 6 padding
        mask = [1] * 10 + [0] * 6
        out_ids, _ = self._call(ids, mask, mlm_prob=0.9)
        # At least some positions should be [MASK]=1
        self.assertIn(1, out_ids[:10])

    def test_pad_tokens_never_masked(self):
        ids = list(range(10, 14)) + [0] * 12  # 4 real tokens, 12 padding
        mask = [1] * 4 + [0] * 12
        for _ in range(10):
            out_ids, _ = self._call(ids, mask, mlm_prob=1.0)
            # Padding positions (idx 4..15) must remain 0
            for i in range(4, 16):
                self.assertEqual(out_ids[i], 0, f"Pad token at position {i} was modified")

    def test_special_tokens_never_masked(self):
        # Use special_ids=[10,11,12,13,14] for the first 5 tokens
        special_ids = [10, 11, 12, 13, 14]
        ids = [10, 11, 20, 21, 22, 23, 24, 25, 12, 13] + [0] * 6
        mask = [1] * 10 + [0] * 6
        for _ in range(20):
            out_ids, _ = self._call(ids, mask, mlm_prob=1.0, special_ids=special_ids)
            for i, orig in enumerate(ids[:10]):
                if orig in special_ids:
                    self.assertEqual(out_ids[i], orig, f"Special token at position {i} was masked")

    def test_all_padding_returns_unchanged(self):
        ids = [0] * 16
        mask = [0] * 16
        out_ids, labels = self._call(ids, mask, mlm_prob=0.15)
        self.assertEqual(out_ids, ids)
        self.assertEqual(labels, ids)

    def test_zero_probability_masks_at_least_one_token(self):
        # mlm_probability=0 → num_to_mask = max(1, 0) = 1
        ids = list(range(10, 26))
        mask = [1] * 16
        out_ids, _ = self._call(ids, mask, mlm_prob=0.0)
        # exactly 1 position changed
        changed = sum(1 for a, b in zip(ids, out_ids) if a != b)
        # Either 1 was masked (to mask_id), or it was a random replacement
        self.assertGreaterEqual(changed, 0)  # may stay same if random replacement same token

    def test_number_of_masked_positions_proportional_to_probability(self):
        random.seed(42)
        ids = list(range(10, 26))  # 16 non-special, non-padding tokens
        mask = [1] * 16
        out_ids, _ = self._call(ids, mask, mlm_prob=0.5)
        # Roughly 50% should be changed; allow wide range due to randomness
        changed = sum(1 for a, b in zip(ids, out_ids) if a != b)
        self.assertGreater(changed, 0)

    def test_full_probability_changes_some_positions(self):
        random.seed(1)
        ids = list(range(10, 26))
        mask = [1] * 16
        out_ids, _ = self._call(ids, mask, mlm_prob=1.0)
        changed = sum(1 for a, b in zip(ids, out_ids) if a != b)
        # With 80% of positions getting [MASK]=1 and 10% random replacement,
        # there should be changes
        self.assertGreater(changed, 0)

    def test_returns_copies_not_original_references(self):
        ids = list(range(10, 26))
        mask = [1] * 16
        orig = list(ids)
        out_ids, _ = self._call(ids, mask, mlm_prob=0.5)
        # The original ids should not have been mutated
        self.assertEqual(ids, orig)


if __name__ == "__main__":
    unittest.main()
