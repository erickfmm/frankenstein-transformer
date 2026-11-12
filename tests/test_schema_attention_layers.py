import pathlib
import unittest

from src.utils.schema_loader import resolve_schema


class SchemaAttentionLayerTests(unittest.TestCase):
    def test_schema_includes_all_attention_layer_names(self):
        schema_path = pathlib.Path(__file__).parent.parent / "src" / "schema.yaml"
        schema = resolve_schema(schema_path)

        enum_values = (
            schema["properties"]["model"]["properties"]["dims"]["properties"][
                "layer_pattern"
            ]["items"]["enum"]
        )

        expected = {
            "retnet",
            "mamba",
            "ode",
            "titan",
            "standard",
            "sigmoid",
            "ca",
            "sparse_transformer",
            "longformer",
            "swa",
            "bigbird",
            "sparsek",
            "nsa",
            "sparge",
            "fasa",
            "gla",
            "deltanet",
            "gated_deltanet",
            "gated_deltanet2",
            "hgrn2",
            "fox",
            "gated_softmax",
            "engram",
            "gqa",
            "memory",
            "msa",
            "sparda",
            "lsa",
            "kda",
            "mla",
            "gqla",
            "mlra",
            "tucker",
            "kpa",
            "tpa",
            "iha",
            "gta",
            "mtla",
            "cca",
            "ccgqa",
            "gma",
            "ssog",
            "conformer",
            "falcon1",
            "falcon2",
            "falcon3",
            "falcon1a",
            "falcon2a",
            "falcon3a",
            "fastconformer",
            "fpa",
            "xsa",
            "mta",
        }

        self.assertTrue(expected.issubset(set(enum_values)))

    def test_schema_includes_mixture_of_depths_fields(self):
        schema_path = pathlib.Path(__file__).parent.parent / "src" / "schema.yaml"
        schema = resolve_schema(schema_path)

        model_properties = schema["properties"]["model"]["properties"]
        self.assertIn("mixture_of_depths", model_properties)

        mod_properties = model_properties["mixture_of_depths"]["properties"]
        for field_name in ["enabled", "capacity_ratio", "router_aux_loss_weight"]:
            self.assertIn(field_name, mod_properties)

    def test_schema_includes_new_latent_sparse_gated_fields(self):
        schema_path = pathlib.Path(__file__).parent.parent / "src" / "schema.yaml"
        schema = resolve_schema(schema_path)

        attention_properties = (
            schema["properties"]["model"]["properties"]["attention"]["properties"]
        )

        # Map of mixer sub-key -> set of expected leaf field names.
        expected_per_mixer = {
            "mla": {"latent_rank"},
            "gqla": {"latent_rank", "num_groups", "decode_path"},
            "mlra": {"latent_rank", "num_latent_heads"},
            "tucker": {"query_rank", "key_rank", "value_rank"},
            "kpa": {
                "h1", "h2", "d1", "d2",
                "query_rank", "key_rank", "value_rank",
            },
            "tpa": {"query_rank", "key_rank", "value_rank"},
            "iha": {"num_pseudo_heads"},
            "gta": {"num_shared_groups", "value_latent_rank"},
            "mtla": {"latent_rank", "merge_factor", "stride"},
            "cca": {
                "latent_rank", "num_conv_layers", "conv_kernel_seq",
                "conv_kernel_ch", "qk_mean", "value_shift",
            },
            "ccgqa": {
                "query_latent_rank", "kv_latent_rank", "num_kv_heads",
                "num_conv_layers", "conv_kernel_seq", "conv_kernel_ch",
                "qk_mean", "value_shift",
            },
            "msa": {"block_size", "topk_blocks", "index_dim", "kl_loss_weight"},
            "sparda": {"block_size", "topk_blocks", "forecast_dim"},
            "lsa": {
                "num_long_range", "num_buckets", "num_sinks", "pool_size",
                "cluster_block", "keys_per_bucket", "max_keys_per_bucket",
            },
            "gma": {
                "num_components", "routing_dim", "epsilon", "sigma_eps",
                "init_mean_std",
            },
            "ssog": {
                "num_atoms", "lookat", "max_offset", "cold_init",
                "sigma_floor", "grid_h", "grid_w",
            },
            "swa": {"window_size", "num_sinks"},
            "ca": {"learned_gain", "gain_init"},
            "mta": {
                "cq", "ck", "ch", "pre_softmax", "post_softmax",
                "gated_group_norm",
            },
            "fastconformer": {
                "window_size", "global_tokens", "global_tokens_spacing",
                "global_attn_separate",
            },
            "conformer": {"kernel_size", "macaron", "conv_norm"},
            "falcon": {
                "chunk_size", "qk_norm", "beta_mode", "lambda_mode",
                "beta", "lambda", "window", "short_conv", "conv_kernel",
                "eps", "eps_gamma",
            },
        }

        for mixer, expected_leaves in expected_per_mixer.items():
            self.assertIn(
                mixer,
                attention_properties,
                f"attention.{mixer} sub-object missing from schema",
            )
            mixer_props = attention_properties[mixer]["properties"]
            actual_leaves = set(mixer_props.keys())
            self.assertTrue(
                expected_leaves.issubset(actual_leaves),
                f"attention.{mixer} missing leaves: "
                f"{expected_leaves - actual_leaves}",
            )

    def test_cache_specs_match_layer_pattern_enum(self):
        """T12: CACHE_SPECS covers every built-in mixer (54).

        The assembled enum may also contain plugin mixers (config-declared
        plugins are loaded by other tests in the same process), so assert
        coverage rather than strict equality.
        """
        from src.model.cache.capabilities import CACHE_SPECS

        schema_path = pathlib.Path(__file__).parent.parent / "src" / "schema.yaml"
        schema = resolve_schema(schema_path)
        enum_values = set(
            schema["properties"]["model"]["properties"]["dims"]["properties"][
                "layer_pattern"
            ]["items"]["enum"]
        )
        self.assertEqual(len(CACHE_SPECS), 54)
        self.assertTrue(set(CACHE_SPECS).issubset(enum_values))

    def test_inference_schema_is_referenced(self):
        """The optional inference: block is wired into the root schema."""
        schema_path = pathlib.Path(__file__).parent.parent / "src" / "schema.yaml"
        schema = resolve_schema(schema_path)
        self.assertIn("inference", schema["properties"])
        inference = schema["properties"]["inference"]
        self.assertFalse(inference.get("additionalProperties", True))
        self.assertIn("kv_cache", inference["properties"])
        self.assertIn("generation", inference["properties"])


if __name__ == "__main__":
    unittest.main()
