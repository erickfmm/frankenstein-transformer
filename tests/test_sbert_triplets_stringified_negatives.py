"""Tests for the SBERT triplets negatives coercion.

Datasets exported from CSV/parquet (e.g.
``erickfmm/embedding-data__QQP_triplets__translated_to_spanish``) carry the
negatives column as a stringified Python literal instead of a true list
feature. :func:`_coerce_negatives_list` and
:meth:`SBERTTrainer._dataset_to_examples_triplets` must parse those back so
the rows are not silently skipped.
"""
import unittest
from importlib.util import find_spec

SBERT_TEST_DEPS_AVAILABLE = all(
    find_spec(module_name) is not None
    for module_name in ("torch", "datasets", "sentence_transformers")
)

if SBERT_TEST_DEPS_AVAILABLE:
    from src.sbert.train_sbert import SBERTTrainer, _coerce_negatives_list


@unittest.skipUnless(
    SBERT_TEST_DEPS_AVAILABLE,
    "torch + datasets + sentence_transformers are required for SBERT triplets tests",
)
class CoerceNegativesListTests(unittest.TestCase):
    def test_real_list_passthrough(self):
        self.assertEqual(_coerce_negatives_list(["a", "b"]), ["a", "b"])

    def test_empty_list_passthrough(self):
        # An empty list is still a list; downstream skips the row for having
        # no valid negatives, which is not this helper's concern.
        self.assertEqual(_coerce_negatives_list([]), [])

    def test_stringified_list_is_parsed(self):
        raw = "['Can people stop this debate?', 'Why do politicians fight?']"
        self.assertEqual(
            _coerce_negatives_list(raw),
            ["Can people stop this debate?", "Why do politicians fight?"],
        )

    def test_stringified_list_with_escaped_quotes_is_parsed(self):
        raw = '["Isn\'t that a misrepresentation?", \'Why is discourse hollow?\']'
        self.assertEqual(
            _coerce_negatives_list(raw),
            ["Isn't that a misrepresentation?", "Why is discourse hollow?"],
        )

    def test_malformed_string_returns_none(self):
        self.assertIsNone(_coerce_negatives_list("not-a-list"))
        self.assertIsNone(_coerce_negatives_list("[unclosed,"))

    def test_non_list_scalar_returns_none(self):
        self.assertIsNone(_coerce_negatives_list(3))
        self.assertIsNone(_coerce_negatives_list(None))

    def test_stringified_non_list_literal_returns_none(self):
        self.assertIsNone(_coerce_negatives_list("('a', 'b')"))
        self.assertIsNone(_coerce_negatives_list("42"))


@unittest.skipUnless(
    SBERT_TEST_DEPS_AVAILABLE,
    "torch + datasets + sentence_transformers are required for SBERT triplets tests",
)
class TripletRowsToExamplesTests(unittest.TestCase):
    """End-to-end check of the row conversion (no network, no model)."""

    def test_stringified_negatives_rows_produce_examples(self):
        trainer = SBERTTrainer.__new__(SBERTTrainer)
        trainer.dataset_columns = {
            "query": "Q_traducida",
            "positive": "POS_traducida",
            "negatives": "NEGs_traducidas",
        }
        trainer.query_prefix = ""
        trainer.document_prefix = ""

        rows = [
            {
                "Q_traducida": "q1",
                "POS_traducida": "p1",
                "NEGs_traducidas": "['n1', 'n2']",
            },
            # Real list keeps working unchanged.
            {
                "Q_traducida": "q2",
                "POS_traducida": "p2",
                "NEGs_traducidas": ["n3"],
            },
            # Malformed cell is skipped (counted, not raised).
            {
                "Q_traducida": "q3",
                "POS_traducida": "p3",
                "NEGs_traducidas": "garbage",
            },
        ]

        examples = trainer._dataset_to_examples_triplets(rows, split_name="train")

        texts = [tuple(example.texts) for example in examples]
        self.assertEqual(
            texts,
            [("q1", "p1", "n1"), ("q1", "p1", "n2"), ("q2", "p2", "n3")],
        )


if __name__ == "__main__":
    unittest.main()
