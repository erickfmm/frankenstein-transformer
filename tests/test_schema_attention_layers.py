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
            "retnet_attn",
            "mamba",
            "ode",
            "titan_attn",
            "standard_attn",
            "sigmoid_attn",
            "sparse_transformer_attn",
            "longformer_attn",
            "bigbird_attn",
            "sparsek_attn",
            "nsa_attn",
            "sparge_attn",
            "fasa_attn",
            "gla_attn",
            "deltanet_attn",
            "gated_deltanet_attn",
            "gated_deltanet2_attn",
            "hgrn2_attn",
            "fox_attn",
            "gated_softmax_attn",
            "engram_attn",
            "gqa_attn",
            "msa_attn",
            "sparda_attn",
            "kda_attn",
            "mla_attn",
            "gqla_attn",
            "mlra_attn",
            "tucker_attn",
            "iha_attn",
            "gta_attn",
            "mtla_attn",
            "cca_attn",
            "ccgqa_attn",
        }

        self.assertTrue(expected.issubset(set(enum_values)))

    def test_schema_includes_mixture_of_depths_fields(self):
        schema_path = pathlib.Path(__file__).parent.parent / "src" / "schema.yaml"
        schema = resolve_schema(schema_path)

        model_properties = schema["properties"]["model"]["properties"]

        for field_name in [
            "use_mixture_of_depths",
            "mixture_of_depths_capacity_ratio",
            "mixture_of_depths_router_aux_loss_weight",
        ]:
            self.assertIn(field_name, model_properties)

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


if __name__ == "__main__":
    unittest.main()
