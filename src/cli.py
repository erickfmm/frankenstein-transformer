"""CLI entrypoint for frankenstein-transformer.

Provides the ``frankenstein-transformer`` command with exactly four
subcommands:

- ``train``: schema-validated training (MLM, SBERT, causal LM, vision, ...)
  driven entirely by a YAML config.
- ``deploy``: convert a checkpoint into deployment artifacts
  (``standard``/``quantized``) or export formats (``transformers``,
  ``gguf``).
- ``infer``: run inference with a deployed artifact (``--task mlm``) or a
  trained SBERT model (``--task sbert``).
- ``web-server``: Streamlit web UI for building YAML configurations.

The YAML config is the single source of truth for model/training settings;
``--device`` is the only training runtime override accepted on the CLI. The
inference subcommand exposes the runtime ``inference:`` block flags with the
documented precedence CLI > YAML > schema defaults, and ``deploy --config``
supplies a JSON model config only when the checkpoint does not embed one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def _run_train(args: argparse.Namespace) -> int:
    from .training.main import main as train_main

    argv = ["--device", args.device]
    if args.config:
        argv.extend(["--config", args.config])
    else:
        argv.extend(["--config-name", args.config_name])
    if args.list_configs:
        argv.append("--list-configs")
    result = train_main(argv)
    return int(result) if isinstance(result, int) else 0


def _run_deploy(args: argparse.Namespace) -> int:
    if args.check and args.format != "gguf":
        print("`--check` is only supported with `--format gguf`.")
        return 2
    if args.format in ("transformers", "gguf"):
        if args.config:
            print("`--config` is only supported with `--format standard|quantized`.")
            return 2
        if not args.yaml:
            print(f"`--yaml` is required when using `--format {args.format}`.")
            return 2
        is_compatible, _ = _validate_transformers_export_compatibility(args.yaml)
        if not is_compatible:
            return 1

    if args.format in ("standard", "quantized"):
        from .deploy.deploy import main as deploy_main

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
        if args.config:
            argv.extend(["--config", args.config])
        if args.validate:
            argv.append("--validate")
        result = deploy_main(argv)
        return int(result) if isinstance(result, int) else 0

    if args.format == "transformers":
        from .deploy.transformers_export import main as transformers_export_main

        argv = [
            "--model",
            args.checkpoint,
            "--yaml",
            args.yaml,
            "--output",
            args.output,
        ]
        result = transformers_export_main(argv)
        return int(result) if isinstance(result, int) else 0

    # args.format == "gguf"
    from .deploy.bitnet_gguf_export import main as bitnet_gguf_main

    argv = [
        "--model",
        args.checkpoint,
        "--yaml",
        args.yaml,
        "--output",
        args.output,
    ]
    if args.check:
        argv.append("--check")
    result = bitnet_gguf_main(argv)
    return int(result) if isinstance(result, int) else 0


def _run_infer(args: argparse.Namespace) -> int:
    if args.task == "sbert":
        if not args.mode:
            print("`--mode` is required when using `--task sbert`.")
            return 2
        from .sbert.inference_sbert import main as sbert_infer_main

        argv = [
            "--model_path",
            args.model,
            "--mode",
            args.mode,
            "--batch_size",
            str(args.batch_size),
            "--device",
            args.device,
            "--top_k",
            str(args.top_k if args.top_k is not None else 5),
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

    from .deploy.inference import main as infer_main

    argv = [
        "--model",
        args.model,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.task == "causal_lm":
        argv.extend(["--task", "causal_lm"])
    if args.config:
        argv.extend(["--config", args.config])
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
    if args.benchmark_generation:
        argv.append("--benchmark-generation")
    # Generation / KV-cache flags: forward only explicitly-passed values so
    # the CLI > YAML > schema-default precedence holds (D2).
    for flag in (
        "prompt",
        "max_new_tokens",
        "temperature",
        "top_p",
        "greedy",
        "kv_cache",
        "kv_quant",
        "kv_group_size",
        "kv_protected_window",
        "kv_compress",
        "kv_num_sinks",
        "kv_window",
    ):
        value = getattr(args, flag, None)
        if value is None:
            continue
        cli_name = flag.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(f"--{cli_name}")
            else:
                argv.append(f"--no-{cli_name}")
        else:
            argv.extend([f"--{cli_name}", str(value)])
    if args.top_k is not None and args.task == "causal_lm":
        argv.extend(["--top-k", str(args.top_k)])
    result = infer_main(argv)
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
        str(Path(__file__).resolve().parent / "streamlit_gui" / "app.py"),
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
        ``deploy``, ``infer``, and ``web-server``.
    """
    parser = argparse.ArgumentParser(
        prog="frankenstein-transformer",
        description="Configurable training library and CLI for Transformer Encoder Frankenstein",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run schema-validated training from a YAML config")
    train_parser.add_argument("--config", type=str, default=None, help="Path to custom YAML config file")
    train_parser.add_argument("--config-name", type=str, default="frankenstein_text_encoder", help="Named preset from configs/")
    train_parser.add_argument("--list-configs", action="store_true", help="List available config presets and exit")
    train_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    train_parser.set_defaults(func=_run_train)

    deploy_parser = subparsers.add_parser("deploy", help="Convert a checkpoint into deployment artifacts or export formats")
    deploy_parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint (.pt)")
    deploy_parser.add_argument("--output", type=str, required=True, help="Output directory for artifacts")
    deploy_parser.add_argument(
        "--format",
        type=str,
        choices=["quantized", "standard", "transformers", "gguf"],
        default="quantized",
        help="quantized/standard: deploy artifacts; transformers: HuggingFace export; gguf: BitNet i2_s GGUF",
    )
    deploy_parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="Training YAML (required for --format transformers|gguf)",
    )
    deploy_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON model config for --format standard|quantized (when not stored in the checkpoint)",
    )
    deploy_parser.add_argument("--validate", action="store_true", help="Validate artifact after creation (standard/quantized)")
    deploy_parser.add_argument("--check", action="store_true", help="Only run the GGUF compatibility check and exit (gguf only)")
    deploy_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    deploy_parser.set_defaults(func=_run_deploy)

    infer_parser = subparsers.add_parser("infer", help="Run inference with a deployed model or a trained SBERT model")
    infer_parser.add_argument("--model", type=str, required=True, help="Path to deployed model dir or SBERT model")
    infer_parser.add_argument(
        "--task",
        choices=["mlm", "sbert", "causal_lm"],
        default="mlm",
        help="mlm: deployed encoder; sbert: sentence embeddings; causal_lm: autoregressive generation (requires a decoder artifact)",
    )
    infer_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML preset carrying an inference: block (causal_lm generation + KV cache)",
    )
    infer_parser.add_argument("--prompt", type=str, default=None, help="Prompt text (causal_lm)")
    infer_parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None, help="Maximum tokens to generate (causal_lm)")
    infer_parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (causal_lm)")
    infer_parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="causal_lm: top-k sampling filter (0 disables); sbert: number of search results (default 5)",
    )
    infer_parser.add_argument("--top-p", dest="top_p", type=float, default=None, help="Nucleus mass (causal_lm; 1.0 disables)")
    infer_parser.add_argument("--greedy", dest="greedy", action="store_true", default=None, help="Greedy argmax decoding (causal_lm)")
    infer_parser.add_argument("--kv-cache", dest="kv_cache", action="store_true", default=None, help="Enable the KV cache (causal_lm)")
    infer_parser.add_argument("--no-kv-cache", dest="kv_cache", action="store_false", help="Disable the KV cache (causal_lm)")
    infer_parser.add_argument("--kv-quant", dest="kv_quant", type=str, choices=["none", "int8", "int4_kivi"], default=None, help="KV quantization codec (causal_lm)")
    infer_parser.add_argument("--kv-group-size", dest="kv_group_size", type=int, default=None, help="Tokens per quantization group, int4_kivi (causal_lm)")
    infer_parser.add_argument("--kv-protected-window", dest="kv_protected_window", type=int, default=None, help="Recent full-precision tokens, int4_kivi (causal_lm)")
    infer_parser.add_argument("--kv-compress", dest="kv_compress", type=str, choices=["none", "window_sinks"], default=None, help="KV eviction policy (causal_lm)")
    infer_parser.add_argument("--kv-num-sinks", dest="kv_num_sinks", type=int, default=None, help="Pinned attention-sink positions (causal_lm)")
    infer_parser.add_argument("--kv-window", dest="kv_window", type=int, default=None, help="Sliding window size, window_sinks (causal_lm)")
    infer_parser.add_argument(
        "--mode",
        type=str,
        choices=["similarity", "search", "cluster", "encode"],
        default=None,
        help="SBERT task mode (required with --task sbert)",
    )
    infer_parser.add_argument("--text", type=str, default=None, help="Single text input (mlm task)")
    infer_parser.add_argument("--input", type=str, default=None, help="Input file, one text per line (mlm task)")
    infer_parser.add_argument("--output", type=str, default=None, help="Output file for results (mlm task)")
    infer_parser.add_argument("--fp16", action="store_true", help="Use FP16 precision (mlm task)")
    infer_parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark (mlm task)")
    infer_parser.add_argument(
        "--benchmark-generation",
        dest="benchmark_generation",
        action="store_true",
        default=False,
        help="Benchmark generation prefill/decode throughput (causal_lm task)",
    )
    infer_parser.add_argument("--sentence1", type=str, default=None, help="First sentence (sbert similarity)")
    infer_parser.add_argument("--sentence2", type=str, default=None, help="Second sentence (sbert similarity)")
    infer_parser.add_argument("--query", type=str, default=None, help="Search query (sbert search)")
    infer_parser.add_argument("--corpus-file", type=str, default=None, help="Corpus file (sbert search)")
    infer_parser.add_argument("--sentences-file", type=str, default=None, help="Sentences file (sbert cluster/encode)")
    infer_parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters (sbert cluster)")
    infer_parser.add_argument("--input-file", type=str, default=None, help="Input sentences file (sbert encode)")
    infer_parser.add_argument("--output-file", type=str, default=None, help="Output file (sbert encode)")
    infer_parser.add_argument("--batch-size", type=int, default=8, help="Batch size (mlm file mode / sbert)")
    infer_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    infer_parser.set_defaults(func=_run_infer)

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
