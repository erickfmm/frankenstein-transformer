"""Unit tests for optimizer base utilities."""
import unittest

try:
    from src.model.optimizer.base import (
        to_float,
        to_int,
        to_betas,
        extract_prefixed_parameters,
        ensure_no_unknown_parameters,
        parse_group_value,
        with_named_groups,
        GROUP_NAMES,
    )
    _IMPORTS_OK = True
except ImportError:
    _IMPORTS_OK = False


@unittest.skipUnless(_IMPORTS_OK, "src.model.optimizer.base unavailable")
class ToFloatTests(unittest.TestCase):
    def test_float_from_int(self):
        self.assertAlmostEqual(to_float(5, 1.0), 5.0)

    def test_float_from_string(self):
        self.assertAlmostEqual(to_float("3.14", 0.0), 3.14)

    def test_none_returns_default(self):
        self.assertAlmostEqual(to_float(None, 2.5), 2.5)

    def test_non_convertible_returns_default(self):
        self.assertAlmostEqual(to_float("not_a_number", 9.9), 9.9)

    def test_zero_value(self):
        self.assertAlmostEqual(to_float(0, 1.0), 0.0)

    def test_negative_value(self):
        self.assertAlmostEqual(to_float(-3, 1.0), -3.0)


@unittest.skipUnless(_IMPORTS_OK, "src.model.optimizer.base unavailable")
class ToIntTests(unittest.TestCase):
    def test_int_from_float(self):
        self.assertEqual(to_int(3.9, 0), 3)

    def test_int_from_string(self):
        self.assertEqual(to_int("7", 0), 7)

    def test_none_returns_default(self):
        self.assertEqual(to_int(None, 42), 42)

    def test_non_convertible_returns_default(self):
        self.assertEqual(to_int("abc", 10), 10)


@unittest.skipUnless(_IMPORTS_OK, "src.model.optimizer.base unavailable")
class ToBetasTests(unittest.TestCase):
    def test_tuple_pair(self):
        self.assertEqual(to_betas((0.9, 0.999)), (0.9, 0.999))

    def test_list_pair(self):
        self.assertEqual(to_betas([0.8, 0.95]), (0.8, 0.95))

    def test_none_returns_default(self):
        self.assertEqual(to_betas(None), (0.9, 0.95))

    def test_single_element_returns_default(self):
        self.assertEqual(to_betas([0.9]), (0.9, 0.95))

    def test_wrong_type_returns_default(self):
        self.assertEqual(to_betas("0.9,0.95"), (0.9, 0.95))

    def test_non_numeric_pair_returns_default(self):
        self.assertEqual(to_betas(["a", "b"]), (0.9, 0.95))

    def test_three_elements_returns_default(self):
        self.assertEqual(to_betas([0.9, 0.95, 0.999]), (0.9, 0.95))


@unittest.skipUnless(_IMPORTS_OK, "src.model.optimizer.base unavailable")
class ExtractPrefixedParametersTests(unittest.TestCase):
    def test_extracts_matching_prefix(self):
        params = {"adamw-lr_embeddings": 1e-4, "adamw-wd_other": 0.01, "lion-lr_embeddings": 5e-5}
        result = extract_prefixed_parameters("adamw", params)
        self.assertIn("lr_embeddings", result)
        self.assertIn("wd_other", result)
        self.assertNotIn("lion-lr_embeddings", result)

    def test_empty_parameters(self):
        self.assertEqual(extract_prefixed_parameters("adamw", {}), {})

    def test_no_matching_prefix(self):
        params = {"lion-lr_embeddings": 5e-5}
        self.assertEqual(extract_prefixed_parameters("adamw", params), {})

    def test_prefix_value_correct(self):
        params = {"adamw-lr_embeddings": 2e-5}
        result = extract_prefixed_parameters("adamw", params)
        self.assertAlmostEqual(result["lr_embeddings"], 2e-5)


@unittest.skipUnless(_IMPORTS_OK, "src.model.optimizer.base unavailable")
class EnsureNoUnknownParametersTests(unittest.TestCase):
    def test_all_known_passes(self):
        ensure_no_unknown_parameters("adamw", {"lr_embeddings": 1e-4}, {"lr_embeddings", "wd_embeddings"})

    def test_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ensure_no_unknown_parameters("adamw", {"bad_param": 1.0}, {"lr_embeddings"})
        self.assertIn("bad_param", str(ctx.exception))

    def test_empty_params_passes(self):
        ensure_no_unknown_parameters("adamw", {}, {"lr_embeddings"})

    def test_error_message_includes_optimizer_name(self):
        with self.assertRaises(ValueError) as ctx:
            ensure_no_unknown_parameters("lion", {"unknown": 1.0}, set())
        self.assertIn("lion", str(ctx.exception))


@unittest.skipUnless(_IMPORTS_OK, "src.model.optimizer.base unavailable")
class ParseGroupValueTests(unittest.TestCase):
    def test_group_specific_key_returned(self):
        scoped = {"lr_embeddings": 1e-5, "lr_other": 1e-3}
        self.assertAlmostEqual(parse_group_value(scoped, "lr", "embeddings", 1e-4), 1e-5)

    def test_fallback_to_default_when_missing(self):
        self.assertAlmostEqual(parse_group_value({}, "lr", "embeddings", 9e-4), 9e-4)

    def test_default_from_group(self):
        result = parse_group_value({"wd_norms": 0.001}, "wd", "attention", 0.0)
        self.assertAlmostEqual(result, 0.0)


@unittest.skipUnless(_IMPORTS_OK, "src.model.optimizer.base unavailable")
class WithNamedGroupsTests(unittest.TestCase):
    def test_adds_name_other_when_missing(self):
        groups = [{"params": [1, 2]}]
        result = with_named_groups(groups)
        self.assertEqual(result[0]["name"], "other")

    def test_preserves_existing_name(self):
        groups = [{"params": [], "name": "embeddings"}]
        result = with_named_groups(groups)
        self.assertEqual(result[0]["name"], "embeddings")

    def test_does_not_mutate_original(self):
        groups = [{"params": []}]
        _ = with_named_groups(groups)
        self.assertNotIn("name", groups[0])

    def test_multiple_groups(self):
        groups = [
            {"params": [], "name": "norms"},
            {"params": []},
        ]
        result = with_named_groups(groups)
        self.assertEqual(result[0]["name"], "norms")
        self.assertEqual(result[1]["name"], "other")


@unittest.skipUnless(_IMPORTS_OK, "src.model.optimizer.base unavailable")
class GroupNamesTests(unittest.TestCase):
    def test_group_names_is_tuple(self):
        self.assertIsInstance(GROUP_NAMES, tuple)

    def test_contains_expected_groups(self):
        for name in ("embeddings", "norms", "attention", "other"):
            self.assertIn(name, GROUP_NAMES)


if __name__ == "__main__":
    unittest.main()
