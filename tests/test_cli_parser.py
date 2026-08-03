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

    def test_train_subcommand_defaults(self):
        args = self.parser.parse_args(["train"])
        self.assertEqual(args.command, "train")
        self.assertEqual(args.device, "auto")
        self.assertIsNone(args.gpu_temp_guard)
        self.assertIsNone(args.batch_size)
        self.assertIsNone(args.model_mode)
        self.assertFalse(args.transformers_export)

    def test_train_subcommand_device_choices(self):
        for device in ("auto", "cpu", "cuda", "mps"):
            args = self.parser.parse_args(["train", "--device", device])
            self.assertEqual(args.device, device)

    def test_train_gpu_temp_guard_flag(self):
        args = self.parser.parse_args(["train", "--gpu-temp-guard"])
        self.assertTrue(args.gpu_temp_guard)

    def test_train_no_gpu_temp_guard_flag(self):
        args = self.parser.parse_args(["train", "--no-gpu-temp-guard"])
        self.assertFalse(args.gpu_temp_guard)

    def test_train_gpu_temp_guard_flags_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["train", "--gpu-temp-guard", "--no-gpu-temp-guard"])

    def test_train_switch_on_thermal_flag(self):
        args = self.parser.parse_args(["train", "--switch-on-thermal"])
        self.assertTrue(args.switch_on_thermal)

    def test_train_no_switch_on_thermal_flag(self):
        args = self.parser.parse_args(["train", "--no-switch-on-thermal"])
        self.assertFalse(args.switch_on_thermal)

    def test_train_switch_on_thermal_flags_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["train", "--switch-on-thermal", "--no-switch-on-thermal"])

    def test_train_gpu_temp_thresholds(self):
        args = self.parser.parse_args([
            "train",
            "--gpu-temp-pause-threshold-c", "91.5",
            "--gpu-temp-resume-threshold-c", "80.0",
            "--gpu-temp-critical-threshold-c", "95.0",
            "--gpu-temp-poll-interval-seconds", "15.0",
        ])
        self.assertAlmostEqual(args.gpu_temp_pause_threshold_c, 91.5)
        self.assertAlmostEqual(args.gpu_temp_resume_threshold_c, 80.0)
        self.assertAlmostEqual(args.gpu_temp_critical_threshold_c, 95.0)
        self.assertAlmostEqual(args.gpu_temp_poll_interval_seconds, 15.0)

    def test_train_model_mode_choices(self):
        for mode in ("frankenstein", "mini", "frankesteindecoder"):
            args = self.parser.parse_args(["train", "--model-mode", mode])
            self.assertEqual(args.model_mode, mode)

    def test_deploy_requires_checkpoint_and_output(self):
        args = self.parser.parse_args([
            "deploy",
            "--checkpoint", "/tmp/ckpt.pt",
            "--output", "/tmp/out",
        ])
        self.assertEqual(args.command, "deploy")
        self.assertEqual(args.checkpoint, "/tmp/ckpt.pt")

    def test_deploy_format_default(self):
        args = self.parser.parse_args([
            "deploy",
            "--checkpoint", "/tmp/ckpt.pt",
            "--output", "/tmp/out",
        ])
        self.assertEqual(args.format, "quantized")
        self.assertFalse(args.transformers_export)
        self.assertIsNone(args.yaml)

    def test_deploy_format_choices(self):
        for fmt in ("quantized", "standard"):
            args = self.parser.parse_args([
                "deploy",
                "--checkpoint", "/tmp/ckpt.pt",
                "--output", "/tmp/out",
                "--format", fmt,
            ])
            self.assertEqual(args.format, fmt)

    def test_quantize_subcommand(self):
        args = self.parser.parse_args([
            "quantize",
            "--checkpoint", "/tmp/ckpt.pt",
            "--output", "/tmp/out",
        ])
        self.assertEqual(args.command, "quantize")
        self.assertFalse(args.transformers_export)
        self.assertIsNone(args.yaml)

    def test_infer_subcommand_defaults(self):
        args = self.parser.parse_args(["infer", "--model", "/tmp/model"])
        self.assertEqual(args.command, "infer")
        self.assertIsNone(args.text)
        self.assertIsNone(args.input)
        self.assertEqual(args.batch_size, 8)
        self.assertFalse(args.fp16)
        self.assertFalse(args.benchmark)
        self.assertFalse(args.transformers_export)

    def test_sbert_train_defaults(self):
        args = self.parser.parse_args(["sbert-train"])
        self.assertEqual(args.command, "sbert-train")
        self.assertIsNone(args.base_model)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.pooling_mode, "mean")
        self.assertFalse(args.no_amp)
        self.assertFalse(args.transformers_export)

    def test_sbert_train_pooling_mode_choices(self):
        for mode in ("mean", "cls", "max"):
            args = self.parser.parse_args(["sbert-train", "--pooling_mode", mode])
            self.assertEqual(args.pooling_mode, mode)

    def test_sbert_infer_requires_model_path_and_mode(self):
        args = self.parser.parse_args([
            "sbert-infer",
            "--model_path", "/tmp/sbert",
            "--mode", "similarity",
        ])
        self.assertEqual(args.command, "sbert-infer")
        self.assertEqual(args.mode, "similarity")

    def test_sbert_infer_mode_choices(self):
        for mode in ("similarity", "search", "cluster", "encode"):
            args = self.parser.parse_args([
                "sbert-infer",
                "--model_path", "/tmp/sbert",
                "--mode", mode,
            ])
            self.assertEqual(args.mode, mode)

    def test_transformers_export_flag_available_in_other_commands(self):
        train = self.parser.parse_args(["train", "--transformers-export"])
        deploy = self.parser.parse_args([
            "deploy",
            "--checkpoint", "/tmp/ckpt.pt",
            "--output", "/tmp/out",
            "--transformers-export",
        ])
        quantize = self.parser.parse_args([
            "quantize",
            "--checkpoint", "/tmp/ckpt.pt",
            "--output", "/tmp/out",
            "--transformers-export",
        ])
        infer = self.parser.parse_args(["infer", "--model", "/tmp/model", "--transformers-export"])
        sbert_train = self.parser.parse_args(["sbert-train", "--transformers-export"])
        sbert_infer = self.parser.parse_args([
            "sbert-infer",
            "--model_path", "/tmp/sbert",
            "--mode", "encode",
            "--transformers-export",
        ])

        self.assertTrue(train.transformers_export)
        self.assertTrue(deploy.transformers_export)
        self.assertTrue(quantize.transformers_export)
        self.assertTrue(infer.transformers_export)
        self.assertTrue(sbert_train.transformers_export)
        self.assertTrue(sbert_infer.transformers_export)

    def test_transformers_export_requires_model_yaml_and_output(self):
        args = self.parser.parse_args([
            "transformers-export",
            "--model", "/tmp/ckpt.pt",
            "--yaml", "/tmp/train.yaml",
            "--output", "/tmp/out",
        ])
        self.assertEqual(args.command, "transformers-export")
        self.assertEqual(args.model, "/tmp/ckpt.pt")
        self.assertEqual(args.yaml, "/tmp/train.yaml")
        self.assertEqual(args.output, "/tmp/out")

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

    def test_deploy_dispatches_to_deploy_main(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.deploy": fake_mod}):
            rc = main(["deploy", "--checkpoint", "/tmp/ckpt.pt", "--output", "/tmp/out"])
        self.assertEqual(rc, 0)
        mocked.assert_called_once()

    def test_quantize_dispatches_to_deploy_main_with_quantized_format(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.deploy": fake_mod}):
            rc = main(["quantize", "--checkpoint", "/tmp/ckpt.pt", "--output", "/tmp/out"])
        self.assertEqual(rc, 0)
        forwarded = mocked.call_args[0][0]
        self.assertIn("quantized", forwarded)

    def test_infer_dispatches_to_infer_main(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.inference": fake_mod}):
            rc = main(["infer", "--model", "/tmp/model"])
        self.assertEqual(rc, 0)
        mocked.assert_called_once()

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

    def test_transformers_export_dispatches_to_export_main(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.transformers_export": fake_mod}):
            rc = main([
                "transformers-export",
                "--model",
                "/tmp/ckpt.pt",
                "--yaml",
                "/tmp/train.yaml",
                "--output",
                "/tmp/out",
            ])
        self.assertEqual(rc, 0)
        mocked.assert_called_once()

    def test_deploy_transformers_export_requires_yaml(self):
        with patch.dict("sys.modules", {"src.deploy.deploy": SimpleNamespace(main=Mock(return_value=0))}):
            rc = main([
                "deploy",
                "--checkpoint",
                "/tmp/ckpt.pt",
                "--output",
                "/tmp/out",
                "--transformers-export",
            ])
        self.assertEqual(rc, 2)

    def test_quantize_transformers_export_requires_yaml(self):
        with patch.dict("sys.modules", {"src.deploy.deploy": SimpleNamespace(main=Mock(return_value=0))}):
            rc = main([
                "quantize",
                "--checkpoint",
                "/tmp/ckpt.pt",
                "--output",
                "/tmp/out",
                "--transformers-export",
            ])
        self.assertEqual(rc, 2)

    def test_deploy_transformers_export_stops_on_incompatibility(self):
        deploy_main = Mock(return_value=0)
        with patch.dict("sys.modules", {"src.deploy.deploy": SimpleNamespace(main=deploy_main)}):
            with patch("src.cli._validate_transformers_export_compatibility", return_value=(False, {"issues": ["x"]})):
                rc = main([
                    "deploy",
                    "--checkpoint",
                    "/tmp/ckpt.pt",
                    "--output",
                    "/tmp/out",
                    "--yaml",
                    "/tmp/train.yaml",
                    "--transformers-export",
                ])
        self.assertEqual(rc, 1)
        deploy_main.assert_not_called()

    def test_deploy_transformers_export_runs_after_successful_deploy(self):
        deploy_main = Mock(return_value=0)
        with patch.dict("sys.modules", {"src.deploy.deploy": SimpleNamespace(main=deploy_main)}):
            with patch("src.cli._validate_transformers_export_compatibility", return_value=(True, {"issues": []})):
                with patch("src.cli._run_transformers_export_to_subfolder", return_value=0) as export_run:
                    rc = main([
                        "deploy",
                        "--checkpoint",
                        "/tmp/ckpt.pt",
                        "--output",
                        "/tmp/out",
                        "--yaml",
                        "/tmp/train.yaml",
                        "--transformers-export",
                    ])
        self.assertEqual(rc, 0)
        deploy_main.assert_called_once()
        export_run.assert_called_once_with(
            model_path="/tmp/ckpt.pt",
            yaml_path="/tmp/train.yaml",
            output_root="/tmp/out",
        )

    def test_quantize_transformers_export_runs_after_successful_quantize(self):
        deploy_main = Mock(return_value=0)
        with patch.dict("sys.modules", {"src.deploy.deploy": SimpleNamespace(main=deploy_main)}):
            with patch("src.cli._validate_transformers_export_compatibility", return_value=(True, {"issues": []})):
                with patch("src.cli._run_transformers_export_to_subfolder", return_value=0) as export_run:
                    rc = main([
                        "quantize",
                        "--checkpoint",
                        "/tmp/ckpt.pt",
                        "--output",
                        "/tmp/out",
                        "--yaml",
                        "/tmp/train.yaml",
                        "--transformers-export",
                    ])
        self.assertEqual(rc, 0)
        deploy_main.assert_called_once()
        forwarded = deploy_main.call_args[0][0]
        self.assertIn("quantized", forwarded)
        export_run.assert_called_once_with(
            model_path="/tmp/ckpt.pt",
            yaml_path="/tmp/train.yaml",
            output_root="/tmp/out",
        )

    def test_train_transformers_export_stops_on_incompatibility(self):
        with patch("src.cli._resolve_train_yaml_path", return_value="/tmp/train.yaml"):
            with patch("src.cli._validate_transformers_export_compatibility", return_value=(False, {"issues": ["x"]})):
                with patch.dict("sys.modules", {"src.training.main": SimpleNamespace(main=Mock(return_value=0))}):
                    rc = main(["train", "--transformers-export"])
        self.assertEqual(rc, 1)

    def test_train_transformers_export_runs_after_success(self):
        train_main = Mock(return_value=0)
        with patch("src.cli._resolve_train_yaml_path", return_value="/tmp/train.yaml"):
            with patch("src.cli._validate_transformers_export_compatibility", return_value=(True, {"issues": []})):
                with patch("src.cli._latest_checkpoint_path", return_value="/tmp/ckpt.pt"):
                    with patch("src.cli._run_transformers_export_to_subfolder", return_value=0) as export_run:
                        with patch.dict("sys.modules", {"src.training.main": SimpleNamespace(main=train_main)}):
                            rc = main(["train", "--transformers-export"])
        self.assertEqual(rc, 0)
        train_main.assert_called_once()
        export_run.assert_called_once_with(
            model_path="/tmp/ckpt.pt",
            yaml_path="/tmp/train.yaml",
            output_root="checkpoints",
        )

    def test_train_transformers_export_fails_without_checkpoint(self):
        train_main = Mock(return_value=0)
        with patch("src.cli._resolve_train_yaml_path", return_value="/tmp/train.yaml"):
            with patch("src.cli._validate_transformers_export_compatibility", return_value=(True, {"issues": []})):
                with patch("src.cli._latest_checkpoint_path", return_value=None):
                    with patch.dict("sys.modules", {"src.training.main": SimpleNamespace(main=train_main)}):
                        rc = main(["train", "--transformers-export"])
        self.assertEqual(rc, 1)

    def test_infer_transformers_export_not_supported(self):
        infer_main = Mock(return_value=0)
        with patch.dict("sys.modules", {"src.deploy.inference": SimpleNamespace(main=infer_main)}):
            rc = main(["infer", "--model", "/tmp/model", "--transformers-export"])
        self.assertEqual(rc, 2)
        infer_main.assert_not_called()

    def test_sbert_train_transformers_export_not_supported(self):
        sbert_train_main = Mock(return_value=0)
        with patch.dict("sys.modules", {"src.sbert.train_sbert": SimpleNamespace(main=sbert_train_main)}):
            rc = main(["sbert-train", "--transformers-export"])
        self.assertEqual(rc, 2)
        sbert_train_main.assert_not_called()

    def test_sbert_infer_transformers_export_not_supported(self):
        sbert_infer_main = Mock(return_value=0)
        with patch.dict("sys.modules", {"src.sbert.inference_sbert": SimpleNamespace(main=sbert_infer_main)}):
            rc = main(["sbert-infer", "--model_path", "/tmp/sbert", "--mode", "encode", "--transformers-export"])
        self.assertEqual(rc, 2)
        sbert_infer_main.assert_not_called()


if __name__ == "__main__":
    unittest.main()
