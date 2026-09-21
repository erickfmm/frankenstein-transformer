"""Unit tests for the CLI argument parser (no torch required)."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.cli import build_parser, main


class BuildParserStructureTests(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_parser_has_subcommands(self):
        # Verify that the parser accepts known subcommands without error.
        self.assertIsNotNone(self.parser)

    def test_removed_subcommands_exit(self):
        # quantize, sbert-train, sbert-infer, transformers-export and
        # bitnet-gguf were merged into deploy/infer; they must not parse.
        for argv in (
            ["quantize", "--checkpoint", "c.pt", "--output", "out"],
            ["sbert-train"],
            ["sbert-infer", "--model_path", "m", "--mode", "encode"],
            ["transformers-export", "--model", "m", "--yaml", "y", "--output", "o"],
            ["bitnet-gguf", "--model", "m", "--yaml", "y", "--output", "o"],
        ):
            with self.assertRaises(SystemExit):
                self.parser.parse_args(argv)

    def test_train_subcommand_defaults(self):
        args = self.parser.parse_args(["train"])
        self.assertEqual(args.command, "train")
        self.assertEqual(args.device, "auto")
        self.assertIsNone(args.config)
        self.assertEqual(args.config_name, "frankenstein")
        self.assertFalse(args.list_configs)

    def test_train_subcommand_device_choices(self):
        for device in ("auto", "cpu", "cuda", "mps"):
            args = self.parser.parse_args(["train", "--device", device])
            self.assertEqual(args.device, device)

    def test_train_has_no_yaml_overriding_flags(self):
        # Settings that live in the YAML schema must not be CLI overrides.
        train_args = self.parser.parse_args(["train"])
        for removed in (
            "batch_size",
            "model_mode",
            "gpu_temp_guard",
            "gpu_temp_pause_threshold_c",
            "gpu_temp_resume_threshold_c",
            "gpu_temp_critical_threshold_c",
            "gpu_temp_poll_interval_seconds",
            "gpu_temp_checkpoint_grace_seconds",
            "resume_from_checkpoint",
            "switch_on_thermal",
            "transformers_export",
        ):
            self.assertFalse(hasattr(train_args, removed))

    def test_deploy_defaults(self):
        args = self.parser.parse_args([
            "deploy",
            "--checkpoint", "/tmp/ckpt.pt",
            "--output", "/tmp/out",
        ])
        self.assertEqual(args.command, "deploy")
        self.assertEqual(args.checkpoint, "/tmp/ckpt.pt")
        self.assertEqual(args.format, "quantized")
        self.assertIsNone(args.yaml)
        self.assertFalse(args.validate)
        self.assertFalse(args.check)

    def test_deploy_format_choices(self):
        for fmt in ("quantized", "standard", "transformers", "gguf"):
            args = self.parser.parse_args([
                "deploy",
                "--checkpoint", "/tmp/ckpt.pt",
                "--output", "/tmp/out",
                "--format", fmt,
            ])
            self.assertEqual(args.format, fmt)

    def test_infer_defaults(self):
        args = self.parser.parse_args(["infer", "--model", "/tmp/model"])
        self.assertEqual(args.command, "infer")
        self.assertEqual(args.task, "mlm")
        self.assertIsNone(args.mode)
        self.assertIsNone(args.text)
        self.assertIsNone(args.input)
        self.assertEqual(args.batch_size, 8)
        self.assertFalse(args.fp16)
        self.assertFalse(args.benchmark)

    def test_infer_task_choices(self):
        for task in ("mlm", "sbert"):
            args = self.parser.parse_args(["infer", "--model", "/tmp/model", "--task", task])
            self.assertEqual(args.task, task)

    def test_infer_sbert_mode_choices(self):
        for mode in ("similarity", "search", "cluster", "encode"):
            args = self.parser.parse_args([
                "infer", "--model", "/tmp/sbert", "--task", "sbert", "--mode", mode,
            ])
            self.assertEqual(args.task, "sbert")
            self.assertEqual(args.mode, mode)

    def test_infer_sbert_flags_use_hyphenated_names(self):
        args = self.parser.parse_args([
            "infer", "--model", "/tmp/sbert", "--task", "sbert",
            "--mode", "search", "--query", "ml", "--corpus-file", "docs.txt",
            "--top-k", "10", "--n-clusters", "3",
        ])
        self.assertEqual(args.query, "ml")
        self.assertEqual(args.corpus_file, "docs.txt")
        self.assertEqual(args.top_k, 10)
        self.assertEqual(args.n_clusters, 3)

    def test_web_server_defaults(self):
        args = self.parser.parse_args(["web-server"])
        self.assertEqual(args.command, "web-server")
        self.assertEqual(args.server_port, 8501)
        self.assertEqual(args.server_address, "localhost")
        self.assertFalse(args.server_headless)
        self.assertFalse(args.development_mode)

    def test_missing_required_subcommand_exits(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])


class MainDispatchTests(unittest.TestCase):
    def test_train_dispatches_to_training_main(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.training.main": fake_mod}):
            rc = main(["train"])
        self.assertEqual(rc, 0)
        mocked.assert_called_once()
        forwarded = mocked.call_args[0][0]
        self.assertEqual(forwarded, ["--device", "auto", "--config-name", "frankenstein"])

    def test_train_forwards_config_and_device(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.training.main": fake_mod}):
            rc = main(["train", "--config", "my.yaml", "--device", "cuda", "--list-configs"])
        self.assertEqual(rc, 0)
        forwarded = mocked.call_args[0][0]
        self.assertEqual(forwarded, ["--device", "cuda", "--config", "my.yaml", "--list-configs"])

    def test_deploy_standard_dispatches_to_deploy_main(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.deploy": fake_mod}):
            rc = main([
                "deploy", "--checkpoint", "/tmp/ckpt.pt", "--output", "/tmp/out",
                "--format", "standard", "--validate",
            ])
        self.assertEqual(rc, 0)
        forwarded = mocked.call_args[0][0]
        self.assertIn("--format", forwarded)
        self.assertIn("standard", forwarded)
        self.assertIn("--validate", forwarded)

    def test_deploy_default_is_quantized(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.deploy": fake_mod}):
            rc = main(["deploy", "--checkpoint", "/tmp/ckpt.pt", "--output", "/tmp/out"])
        self.assertEqual(rc, 0)
        forwarded = mocked.call_args[0][0]
        self.assertIn("quantized", forwarded)

    def test_deploy_transformers_dispatches_to_export_main(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.transformers_export": fake_mod}):
            with patch(
                "src.cli._validate_transformers_export_compatibility",
                return_value=(True, {"issues": []}),
            ):
                rc = main([
                    "deploy", "--checkpoint", "/tmp/ckpt.pt", "--output", "/tmp/out",
                    "--format", "transformers", "--yaml", "/tmp/train.yaml",
                ])
        self.assertEqual(rc, 0)
        mocked.assert_called_once()
        forwarded = mocked.call_args[0][0]
        self.assertEqual(forwarded[0], "--model")
        self.assertIn("/tmp/train.yaml", forwarded)

    def test_deploy_transformers_requires_yaml(self):
        with patch.dict("sys.modules", {"src.deploy.transformers_export": SimpleNamespace(main=Mock(return_value=0))}):
            rc = main([
                "deploy", "--checkpoint", "/tmp/ckpt.pt", "--output", "/tmp/out",
                "--format", "transformers",
            ])
        self.assertEqual(rc, 2)

    def test_deploy_transformers_stops_on_incompatibility(self):
        export_main = Mock(return_value=0)
        with patch.dict("sys.modules", {"src.deploy.transformers_export": SimpleNamespace(main=export_main)}):
            with patch(
                "src.cli._validate_transformers_export_compatibility",
                return_value=(False, {"issues": ["x"]}),
            ):
                rc = main([
                    "deploy", "--checkpoint", "/tmp/ckpt.pt", "--output", "/tmp/out",
                    "--format", "transformers", "--yaml", "/tmp/train.yaml",
                ])
        self.assertEqual(rc, 1)
        export_main.assert_not_called()

    def test_deploy_gguf_dispatches_to_bitnet_gguf_main(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.bitnet_gguf_export": fake_mod}):
            with patch(
                "src.cli._validate_transformers_export_compatibility",
                return_value=(True, {"issues": []}),
            ):
                rc = main([
                    "deploy", "--checkpoint", "/tmp/ckpt.pt", "--output", "/tmp/out.gguf",
                    "--format", "gguf", "--yaml", "/tmp/train.yaml",
                ])
        self.assertEqual(rc, 0)
        mocked.assert_called_once()
        forwarded = mocked.call_args[0][0]
        self.assertIn("/tmp/ckpt.pt", forwarded)
        self.assertNotIn("--check", forwarded)

    def test_deploy_gguf_check_forwarded(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.bitnet_gguf_export": fake_mod}):
            with patch(
                "src.cli._validate_transformers_export_compatibility",
                return_value=(True, {"issues": []}),
            ):
                rc = main([
                    "deploy", "--checkpoint", "/tmp/ckpt.pt", "--output", "/tmp/out.gguf",
                    "--format", "gguf", "--yaml", "/tmp/train.yaml", "--check",
                ])
        self.assertEqual(rc, 0)
        forwarded = mocked.call_args[0][0]
        self.assertIn("--check", forwarded)

    def test_deploy_check_only_valid_for_gguf(self):
        rc = main([
            "deploy", "--checkpoint", "/tmp/ckpt.pt", "--output", "/tmp/out",
            "--format", "standard", "--check",
        ])
        self.assertEqual(rc, 2)

    def test_infer_dispatches_to_infer_main(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.inference": fake_mod}):
            rc = main(["infer", "--model", "/tmp/model"])
        self.assertEqual(rc, 0)
        mocked.assert_called_once()

    def test_infer_sbert_requires_mode(self):
        infer_main = Mock(return_value=0)
        sbert_main = Mock(return_value=0)
        with patch.dict("sys.modules", {
            "src.deploy.inference": SimpleNamespace(main=infer_main),
            "src.sbert.inference_sbert": SimpleNamespace(main=sbert_main),
        }):
            rc = main(["infer", "--model", "/tmp/sbert", "--task", "sbert"])
        self.assertEqual(rc, 2)
        sbert_main.assert_not_called()
        infer_main.assert_not_called()

    def test_infer_sbert_dispatches_to_sbert_infer_main(self):
        infer_main = Mock(return_value=0)
        sbert_main = Mock(return_value=0)
        with patch.dict("sys.modules", {
            "src.deploy.inference": SimpleNamespace(main=infer_main),
            "src.sbert.inference_sbert": SimpleNamespace(main=sbert_main),
        }):
            rc = main([
                "infer", "--model", "/tmp/sbert", "--task", "sbert",
                "--mode", "similarity", "--sentence1", "a", "--sentence2", "b",
            ])
        self.assertEqual(rc, 0)
        infer_main.assert_not_called()
        sbert_main.assert_called_once()
        forwarded = sbert_main.call_args[0][0]
        self.assertEqual(forwarded[0], "--model_path")
        self.assertIn("/tmp/sbert", forwarded)
        self.assertIn("--mode", forwarded)
        self.assertIn("similarity", forwarded)

    def test_return_code_zero_int(self):
        mocked = Mock(return_value=None)  # returns None, should convert to 0
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.training.main": fake_mod}):
            rc = main(["train"])
        self.assertEqual(rc, 0)

    def test_return_code_nonzero_propagated(self):
        mocked = Mock(return_value=1)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.training.main": fake_mod}):
            rc = main(["train"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
