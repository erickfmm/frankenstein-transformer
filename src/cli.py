"""CLI entrypoint for frankestein-transformer.

Provides the ``frankestein-transformer`` command with subcommands for
training, deployment, quantization, inference, SBERT training/inference,
HuggingFace Transformers export, and a Streamlit web server for
configuration building.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _resolve_train_yaml_path(args: argparse.Namespace) -> str:
    if args.config:
        return str(Path(args.config).expanduser().resolve())

    from .training.config_loader import list_config_paths

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    config_paths = list_config_paths(str(config_dir))
    selected = config_paths.get(args.config_name)
    if not selected:
        raise ValueError(f"Unknown config-name '{args.config_name}'")
    return str(Path(selected).expanduser().resolve())


def _validate_transformers_export_compatibility(yaml_path: str) -> tuple[bool, dict]:
    from .deploy.transformers_export import check_yaml_export_compatibility

    compatibility = check_yaml_export_compatibility(yaml_path)
    is_compatible = bool(compatibility.get("is_compatible", False))
    if not is_compatible:
        print(
            "Transformers export compatibility check failed:\n"
            + json.dumps(compatibility, indent=2)
        )
    return is_compatible, compatibility


def _run_transformers_export_to_subfolder(
    *,
    model_path: str,
    yaml_path: str,
    output_root: str,
) -> int:
    from .deploy.transformers_export import export_transformers_model

    transformers_output = str(Path(output_root).expanduser().resolve() / "transformers-export")
    result = export_transformers_model(
        model_path=model_path,
        yaml_path=yaml_path,
        output_dir=transformers_output,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 1


def _latest_checkpoint_path(path: str = "checkpoints") -> str | None:
    checkpoints_dir = Path(path).expanduser().resolve()
    if not checkpoints_dir.exists():
        return None
    candidates = sorted(
        checkpoints_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return str(candidates[0])


def _run_train(args: argparse.Namespace) -> int:
    from .training.main import main as train_main

    yaml_path = None
    if args.transformers_export:
        try:
            yaml_path = _resolve_train_yaml_path(args)
        except Exception as exc:
            print(f"Unable to resolve training YAML for transformers export: {exc}")
            return 2

        is_compatible, _ = _validate_transformers_export_compatibility(yaml_path)
        if not is_compatible:
            return 1

    argv = [
        "--config-name",
        args.config_name,
        "--device",
        args.device,
    ]
    if args.config:
        argv.extend(["--config", args.config])
    if args.list_configs:
        argv.append("--list-configs")
    if args.batch_size is not None:
        argv.extend(["--batch-size", str(args.batch_size)])
    if args.model_mode:
        argv.extend(["--model-mode", args.model_mode])
    if args.gpu_temp_guard is True:
        argv.append("--gpu-temp-guard")
    elif args.gpu_temp_guard is False:
        argv.append("--no-gpu-temp-guard")
    if args.gpu_temp_pause_threshold_c is not None:
        argv.extend(["--gpu-temp-pause-threshold-c", str(args.gpu_temp_pause_threshold_c)])
    if args.gpu_temp_resume_threshold_c is not None:
        argv.extend(["--gpu-temp-resume-threshold-c", str(args.gpu_temp_resume_threshold_c)])
    if args.gpu_temp_critical_threshold_c is not None:
        argv.extend(["--gpu-temp-critical-threshold-c", str(args.gpu_temp_critical_threshold_c)])
    if args.gpu_temp_poll_interval_seconds is not None:
        argv.extend(["--gpu-temp-poll-interval-seconds", str(args.gpu_temp_poll_interval_seconds)])
    if args.gpu_temp_checkpoint_grace_seconds is not None:
        argv.extend(["--gpu-temp-checkpoint-grace-seconds", str(args.gpu_temp_checkpoint_grace_seconds)])
    if args.resume_from_checkpoint is not None:
        argv.extend(["--resume-from-checkpoint", str(args.resume_from_checkpoint)])
    if args.switch_on_thermal is True:
        argv.append("--switch-on-thermal")
    elif args.switch_on_thermal is False:
        argv.append("--no-switch-on-thermal")
    result = train_main(argv)
    rc = int(result) if isinstance(result, int) else 0
    if rc != 0:
        return rc

    if args.transformers_export:
        checkpoint_path = _latest_checkpoint_path("checkpoints")
        if not checkpoint_path:
            print("No checkpoint found in ./checkpoints for transformers export.")
            return 1
        return _run_transformers_export_to_subfolder(
            model_path=checkpoint_path,
            yaml_path=str(yaml_path),
            output_root="checkpoints",
        )
    return rc


def _run_deploy(args: argparse.Namespace) -> int:
    from .deploy.deploy import main as deploy_main

    if args.transformers_export:
        if not args.yaml:
            print("`--yaml` is required when using `--transformers-export`.")
            return 2
        is_compatible, _ = _validate_transformers_export_compatibility(args.yaml)
        if not is_compatible:
            return 1

    argv = [
        "--checkpoint",
        args.checkpoint,
        "--output",
        args.output,
        "--format",
        args.format,
        "--device",
        args.device,
    ]
    if args.validate:
        argv.append("--validate")
    if args.config:
        argv.extend(["--config", args.config])
    result = deploy_main(argv)
    rc = int(result) if isinstance(result, int) else 0
    if rc != 0:
        return rc

    if args.transformers_export:
        return _run_transformers_export_to_subfolder(
            model_path=args.checkpoint,
            yaml_path=args.yaml,
            output_root=args.output,
        )
    return rc


def _run_quantize(args: argparse.Namespace) -> int:
    from .deploy.deploy import main as deploy_main

    if args.transformers_export:
        if not args.yaml:
            print("`--yaml` is required when using `--transformers-export`.")
            return 2
        is_compatible, _ = _validate_transformers_export_compatibility(args.yaml)
        if not is_compatible:
            return 1

    argv = [
        "--checkpoint",
        args.checkpoint,
        "--output",
        args.output,
        "--format",
        "quantized",
        "--device",
        args.device,
    ]
    if args.validate:
        argv.append("--validate")
    if args.config:
        argv.extend(["--config", args.config])
    result = deploy_main(argv)
    rc = int(result) if isinstance(result, int) else 0
    if rc != 0:
        return rc

    if args.transformers_export:
        return _run_transformers_export_to_subfolder(
            model_path=args.checkpoint,
            yaml_path=args.yaml,
            output_root=args.output,
        )
    return rc


def _run_infer(args: argparse.Namespace) -> int:
    from .deploy.inference import main as infer_main

    if args.transformers_export:
        print("`--transformers-export` is not supported for `infer`.")
        return 2

    argv = [
        "--model",
        args.model,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.text:
        argv.extend(["--text", args.text])
    if args.input:
        argv.extend(["--input", args.input])
    if args.output:
        argv.extend(["--output", args.output])
    if args.fp16:
        argv.append("--fp16")
    if args.benchmark:
        argv.append("--benchmark")
    result = infer_main(argv)
    return int(result) if isinstance(result, int) else 0


def _run_sbert_train(args: argparse.Namespace) -> int:
    from .sbert.train_sbert import main as sbert_train_main

    if args.transformers_export:
        print(
            "`--transformers-export` is not supported for `sbert-train`. "
            "Use the dedicated `transformers-export` command with an MLM/decoder checkpoint + YAML."
        )
        return 2

    argv = [
        "--output_dir",
        args.output_dir,
        "--dataset_name",
        args.dataset_name,
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--warmup_steps",
        str(args.warmup_steps),
        "--evaluation_steps",
        str(args.evaluation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--max_eval_samples",
        str(args.max_eval_samples),
        "--max_seq_length",
        str(args.max_seq_length),
        "--hidden_size",
        str(args.hidden_size),
        "--num_layers",
        str(args.num_layers),
        "--pooling_mode",
        args.pooling_mode,
        "--resample_std",
        str(args.resample_std),
        "--device",
        args.device,
    ]
    if args.base_model:
        argv.extend(["--base-model", args.base_model])
    if args.pretrained:
        argv.extend(["--pretrained", args.pretrained])
    if args.max_train_samples is not None:
        argv.extend(["--max_train_samples", str(args.max_train_samples)])
    if args.no_amp:
        argv.append("--no_amp")
    if args.no_resample:
        argv.append("--no_resample")
    if args.trust_remote_code:
        argv.append("--trust_remote_code")
    if args.switch_on_thermal is True:
        argv.append("--switch-on-thermal")
    elif args.switch_on_thermal is False:
        argv.append("--no-switch-on-thermal")
    result = sbert_train_main(argv)
    return int(result) if isinstance(result, int) else 0


def _run_sbert_infer(args: argparse.Namespace) -> int:
    from .sbert.inference_sbert import main as sbert_infer_main

    if args.transformers_export:
        print("`--transformers-export` is not supported for `sbert-infer`.")
        return 2

    argv = [
        "--model_path",
        args.model_path,
        "--mode",
        args.mode,
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
        "--top_k",
        str(args.top_k),
        "--n_clusters",
        str(args.n_clusters),
    ]
    if args.sentence1:
        argv.extend(["--sentence1", args.sentence1])
    if args.sentence2:
        argv.extend(["--sentence2", args.sentence2])
    if args.query:
        argv.extend(["--query", args.query])
    if args.corpus_file:
        argv.extend(["--corpus_file", args.corpus_file])
    if args.sentences_file:
        argv.extend(["--sentences_file", args.sentences_file])
    if args.input_file:
        argv.extend(["--input_file", args.input_file])
    if args.output_file:
        argv.extend(["--output_file", args.output_file])
    result = sbert_infer_main(argv)
    return int(result) if isinstance(result, int) else 0


def _run_transformers_export(args: argparse.Namespace) -> int:
    from .deploy.transformers_export import main as transformers_export_main

    argv = [
        "--model",
        args.model,
        "--yaml",
        args.yaml,
        "--output",
        args.output,
    ]
    result = transformers_export_main(argv)
    return int(result) if isinstance(result, int) else 0


def _run_bitnet_gguf(args: argparse.Namespace) -> int:
    from .deploy.bitnet_gguf_export import main as bitnet_gguf_main

    argv = ["--model", args.model, "--yaml", args.yaml, "--output", args.output]
    if getattr(args, "check", False):
        argv.append("--check")
    result = bitnet_gguf_main(argv)
    return int(result) if isinstance(result, int) else 0


def _run_web_server(args: argparse.Namespace) -> int:
    """Run the Streamlit web server for building configurations."""
    import subprocess
    import sys

    # Build the streamlit run command
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(__file__).replace("cli.py", "streamlit_gui/app.py"),
    ]
    
    if args.server_port:
        streamlit_cmd.extend(["--server.port", str(args.server_port)])
    
    if args.server_address:
        streamlit_cmd.extend(["--server.address", args.server_address])
    
    if args.server_headless:
        streamlit_cmd.append("--server.headless")
    
    if args.development_mode:
        streamlit_cmd.append("--logger.level=debug")
    
    print(f"Starting Streamlit server: {' '.join(streamlit_cmd)}")
    
    try:
        result = subprocess.run(streamlit_cmd, check=True)
        return int(result.returncode) if result.returncode else 0
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit server: {e}", file=sys.stderr)
        return e.returncode
    except KeyboardInterrupt:
        print("\nStreamlit server stopped by user.")
        return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands.

    Returns:
        An :class:`argparse.ArgumentParser` with subparsers for ``train``,
        ``deploy``, ``quantize``, ``infer``, ``sbert-train``,
        ``sbert-infer``, ``transformers-export``, and ``web-server``.
    """
    parser = argparse.ArgumentParser(
        prog="frankestein-transformer",
        description="Configurable training library and CLI for Transformer Encoder Frankenstein",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run main training")
    train_parser.add_argument("--config", type=str, default=None)
    train_parser.add_argument("--config-name", type=str, default="mini")
    train_parser.add_argument("--list-configs", action="store_true")
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument(
        "--model-mode",
        choices=["frankenstein", "mini", "frankesteindecoder"],
        default=None,
    )
    train_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    gpu_temp_group = train_parser.add_mutually_exclusive_group()
    gpu_temp_group.add_argument("--gpu-temp-guard", dest="gpu_temp_guard", action="store_true")
    gpu_temp_group.add_argument("--no-gpu-temp-guard", dest="gpu_temp_guard", action="store_false")
    train_parser.set_defaults(gpu_temp_guard=None)
    train_parser.add_argument("--gpu-temp-pause-threshold-c", type=float, default=None)
    train_parser.add_argument("--gpu-temp-resume-threshold-c", type=float, default=None)
    train_parser.add_argument("--gpu-temp-critical-threshold-c", type=float, default=None)
    train_parser.add_argument("--gpu-temp-poll-interval-seconds", type=float, default=None)
    train_parser.add_argument("--gpu-temp-checkpoint-grace-seconds", type=float, default=None)
    train_parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Resume training from a checkpoint ('auto' or a path)")
    switch_on_thermal_group = train_parser.add_mutually_exclusive_group()
    switch_on_thermal_group.add_argument("--switch-on-thermal", dest="switch_on_thermal", action="store_true")
    switch_on_thermal_group.add_argument("--no-switch-on-thermal", dest="switch_on_thermal", action="store_false")
    train_parser.set_defaults(switch_on_thermal=None)
    train_parser.add_argument("--transformers-export", action="store_true", default=False)
    train_parser.set_defaults(func=_run_train)

    deploy_parser = subparsers.add_parser("deploy", help="Convert checkpoint to deployment artifacts")
    deploy_parser.add_argument("--checkpoint", type=str, required=True)
    deploy_parser.add_argument("--output", type=str, required=True)
    deploy_parser.add_argument("--format", type=str, choices=["quantized", "standard"], default="quantized")
    deploy_parser.add_argument("--validate", action="store_true")
    deploy_parser.add_argument("--config", type=str, default=None)
    deploy_parser.add_argument("--yaml", type=str, default=None)
    deploy_parser.add_argument("--transformers-export", action="store_true", default=False)
    deploy_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    deploy_parser.set_defaults(func=_run_deploy)

    quantize_parser = subparsers.add_parser("quantize", help="Export checkpoint in quantized deployment format")
    quantize_parser.add_argument("--checkpoint", type=str, required=True)
    quantize_parser.add_argument("--output", type=str, required=True)
    quantize_parser.add_argument("--validate", action="store_true")
    quantize_parser.add_argument("--config", type=str, default=None)
    quantize_parser.add_argument("--yaml", type=str, default=None)
    quantize_parser.add_argument("--transformers-export", action="store_true", default=False)
    quantize_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    quantize_parser.set_defaults(func=_run_quantize)

    infer_parser = subparsers.add_parser("infer", help="Run deployed model inference")
    infer_parser.add_argument("--model", type=str, required=True)
    infer_parser.add_argument("--text", type=str, default=None)
    infer_parser.add_argument("--input", type=str, default=None)
    infer_parser.add_argument("--output", type=str, default=None)
    infer_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    infer_parser.add_argument("--fp16", action="store_true")
    infer_parser.add_argument("--batch-size", type=int, default=8)
    infer_parser.add_argument("--benchmark", action="store_true")
    infer_parser.add_argument("--transformers-export", action="store_true", default=False)
    infer_parser.set_defaults(func=_run_infer)

    sbert_train_parser = subparsers.add_parser("sbert-train", help="Train SBERT model")
    sbert_train_parser.add_argument("--base-model", type=str, default=None)
    sbert_train_parser.add_argument("--pretrained", type=str, default=None)
    sbert_train_parser.add_argument("--output_dir", type=str, default="./output/sbert_tormented_v2")
    sbert_train_parser.add_argument(
        "--dataset_name",
        type=str,
        default="erickfmm/agentlans__multilingual-sentences__paired_10_sts",
    )
    sbert_train_parser.add_argument("--batch_size", type=int, default=16)
    sbert_train_parser.add_argument("--epochs", type=int, default=4)
    sbert_train_parser.add_argument("--warmup_steps", type=int, default=1000)
    sbert_train_parser.add_argument("--evaluation_steps", type=int, default=5000)
    sbert_train_parser.add_argument("--learning_rate", type=float, default=2e-5)
    sbert_train_parser.add_argument("--max_train_samples", type=int, default=None)
    sbert_train_parser.add_argument("--max_eval_samples", type=int, default=10000)
    sbert_train_parser.add_argument("--max_seq_length", type=int, default=512)
    sbert_train_parser.add_argument("--hidden_size", type=int, default=768)
    sbert_train_parser.add_argument("--num_layers", type=int, default=12)
    sbert_train_parser.add_argument("--pooling_mode", choices=["mean", "cls", "max"], default="mean")
    sbert_train_parser.add_argument("--trust_remote_code", action="store_true")
    sbert_train_parser.add_argument("--no_amp", action="store_true")
    sbert_train_parser.add_argument("--no_resample", action="store_true")
    sbert_train_parser.add_argument("--resample_std", type=float, default=0.3)
    sbert_train_parser.add_argument("--transformers-export", action="store_true", default=False)
    sbert_train_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    switch_on_thermal_group = sbert_train_parser.add_mutually_exclusive_group()
    switch_on_thermal_group.add_argument("--switch-on-thermal", dest="switch_on_thermal", action="store_true")
    switch_on_thermal_group.add_argument("--no-switch-on-thermal", dest="switch_on_thermal", action="store_false")
    sbert_train_parser.set_defaults(switch_on_thermal=None)
    sbert_train_parser.set_defaults(func=_run_sbert_train)

    sbert_infer_parser = subparsers.add_parser("sbert-infer", help="Run SBERT inference tasks")
    sbert_infer_parser.add_argument("--model_path", type=str, required=True)
    sbert_infer_parser.add_argument("--mode", choices=["similarity", "search", "cluster", "encode"], required=True)
    sbert_infer_parser.add_argument("--sentence1", type=str, default=None)
    sbert_infer_parser.add_argument("--sentence2", type=str, default=None)
    sbert_infer_parser.add_argument("--query", type=str, default=None)
    sbert_infer_parser.add_argument("--corpus_file", type=str, default=None)
    sbert_infer_parser.add_argument("--top_k", type=int, default=5)
    sbert_infer_parser.add_argument("--sentences_file", type=str, default=None)
    sbert_infer_parser.add_argument("--n_clusters", type=int, default=5)
    sbert_infer_parser.add_argument("--input_file", type=str, default=None)
    sbert_infer_parser.add_argument("--output_file", type=str, default=None)
    sbert_infer_parser.add_argument("--batch_size", type=int, default=32)
    sbert_infer_parser.add_argument("--transformers-export", action="store_true", default=False)
    sbert_infer_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    sbert_infer_parser.set_defaults(func=_run_sbert_infer)

    transformers_export_parser = subparsers.add_parser(
        "transformers-export",
        help="Export checkpoint + YAML into Hugging Face Transformers-compatible folder",
    )
    transformers_export_parser.add_argument("--model", type=str, required=True)
    transformers_export_parser.add_argument("--yaml", type=str, required=True)
    transformers_export_parser.add_argument("--output", type=str, required=True)
    transformers_export_parser.set_defaults(func=_run_transformers_export)

    bitnet_gguf_parser = subparsers.add_parser(
        "bitnet-gguf",
        help="Best-effort GGUF (BitNet i2_s) export for standard_attn-only models",
    )
    bitnet_gguf_parser.add_argument("--model", type=str, required=True, help="Path to training checkpoint (*.pt)")
    bitnet_gguf_parser.add_argument("--yaml", type=str, required=True, help="Path to YAML used during training")
    bitnet_gguf_parser.add_argument("--output", type=str, required=True, help="Output .gguf path")
    bitnet_gguf_parser.add_argument("--check", action="store_true", help="Only run the compatibility check and exit")
    bitnet_gguf_parser.set_defaults(func=_run_bitnet_gguf)

    web_server_parser = subparsers.add_parser("web-server", help="Run Streamlit web server for building configurations")
    web_server_parser.add_argument("--server-port", type=int, default=8501, help="Port to run the Streamlit server on")
    web_server_parser.add_argument("--server-address", type=str, default="localhost", help="Address to bind the Streamlit server to")
    web_server_parser.add_argument("--server-headless", action="store_true", help="Run in headless mode (no browser)")
    web_server_parser.add_argument("--development-mode", action="store_true", help="Enable development mode (debug logging)")
    web_server_parser.set_defaults(func=_run_web_server)

    return parser


def main(argv=None) -> int:
    """Parse CLI arguments and dispatch to the selected subcommand.

    Args:
        argv: Optional list of command-line arguments (defaults to
            ``sys.argv[1:]``).

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
