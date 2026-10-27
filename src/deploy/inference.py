#!/usr/bin/env python3
"""High-performance inference engine for deployed Frankenstein models.

Provides :class:`FrankensteinInference` for loading quantized or standard
deployment artifacts and running MLM predictions, masked token prediction,
batch inference, and performance benchmarking.
"""

import torch
import argparse
import logging
from pathlib import Path
import json
from typing import List, Optional, Union
import time

try:
    from ..model.config import FrankensteinModelConfig
    from ..model.frankenstein_decoder import FrankensteinDecoder
    from ..model.frankenstein_encoder import FrankensteinEncoder
    from ..model.cache import FrankensteinCache
    from .quantization import load_quantized_checkpoint
    from .inference_config import InferenceConfig, apply_cli_overrides
    from ..tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
    from ..utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device
    from ..model.attention.common import BitLinear
    from ..utils.config_flatten import flatten_model_dict
    from ..model.numerics import module_softmax
except ImportError:
    from model.config import FrankensteinModelConfig
    from model.frankenstein_decoder import FrankensteinDecoder
    from model.frankenstein_encoder import FrankensteinEncoder
    from model.cache import FrankensteinCache
    from deploy.quantization import load_quantized_checkpoint
    from deploy.inference_config import InferenceConfig, apply_cli_overrides
    from tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
    from utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device
    from model.attention.common import BitLinear
    from utils.config_flatten import flatten_model_dict
    from model.numerics import module_softmax

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrankensteinInference:
    """Optimized inference engine for Frankenstein deployed models.

    Loads deployment artifacts (config, quantized or standard weights,
    optional tokenizer) and provides methods for single/batch prediction,
    masked token prediction, and performance benchmarking.

    Attributes:
        model_dir: Path to the deployment directory.
        device: Resolved PyTorch device string.
        use_half_precision: Whether FP16 inference is enabled (CUDA only).
        config: Loaded :class:`FrankensteinModelConfig`.
        model: Loaded :class:`FrankensteinEncoder` in eval mode.
        tokenizer: Optional :class:`SpanishSPMTokenizer` instance.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = 'auto',
        use_half_precision: bool = False
    ):
        """Initialize the inference engine.

        Args:
            model_dir: Directory containing deployment artifacts
                (``config.json``, ``model*.pt``, optional
                ``tokenizer.model``).
            device: Device string (``"auto"``, ``"cuda"``, ``"mps"``,
                or ``"cpu"``).
            use_half_precision: If ``True``, use FP16 for faster inference
                (CUDA only).
        """
        self.model_dir = Path(model_dir)
        self.device = resolve_torch_device(device)
        self.use_half_precision = use_half_precision and self.device == 'cuda'
        self.inference_section = None
        
        logger.info(f"Initializing inference engine on {self.device}")
        
        # Load config
        self.config = self._load_config()
        
        # Load model
        self.model = self._load_model()
        
        # Load tokenizer if available
        self.tokenizer = self._load_tokenizer()
        
        logger.info("✅ Inference engine ready")
    
    def _load_config(self) -> FrankensteinModelConfig:
        """Load model configuration."""
        config_path = self.model_dir / "config.json"
        
        if not config_path.exists():
            logger.warning("Config file not found, using default")
            return FrankensteinModelConfig()
        
        with open(config_path) as f:
            config_dict = json.load(f)
        
        # The runtime inference block is NOT part of FrankensteinModelConfig;
        # keep it aside for the generation API (§11.5) and drop it before
        # constructing the architectural config.
        self.inference_section = config_dict.pop("inference", None)
        config = FrankensteinModelConfig(**flatten_model_dict(config_dict))
        logger.info(f"Config loaded: {config.hidden_size}D, {config.num_layers} layers")
        return config

    def load_inference_config(self, yaml_path=None, args=None) -> InferenceConfig:
        """Resolve the runtime :class:`InferenceConfig` for generation.

        Precedence (D2): CLI flags > deployed ``config.json`` "inference"
        section (or a YAML preset via ``yaml_path``) > schema defaults.

        Args:
            yaml_path: Optional YAML preset carrying an ``inference:`` block.
            args: Optional argparse namespace with the generation flags
                (all defaulting to ``None``).

        Returns:
            The fully resolved flat :class:`InferenceConfig`.
        """
        cfg = InferenceConfig()
        if yaml_path:
            try:
                from ..training.config_loader import load_inference_config
            except ImportError:
                from training.config_loader import load_inference_config

            cfg = load_inference_config(yaml_path)
        elif isinstance(getattr(self, "inference_section", None), dict):
            try:
                from ..training.config_loader import load_inference_config
            except ImportError:
                from training.config_loader import load_inference_config

            cfg = load_inference_config({"inference": self.inference_section})
        if args is not None:
            cfg = apply_cli_overrides(cfg, args)
        return cfg

    def generate(
        self,
        prompt: Union[str, List[str], torch.Tensor],
        inference_config: Optional[InferenceConfig] = None,
        max_length: int = 512,
    ):
        """Autoregressive generation from a text prompt or token tensor.

        Builds a :class:`~src.model.cache.FrankensteinCache` from the
        resolved config and runs the decoder's prefill+decode loop. When
        ``kv_cache_enabled`` is false the legacy full-recompute loop runs
        (bit-identical, D4/D12).

        Args:
            prompt: Text string, list of strings (batch), or an int64 token
                tensor ``(B, S)``.
            inference_config: Resolved runtime config; defaults are used
                when ``None``.
            max_length: Truncation length for tokenized prompts.

        Returns:
            Dict with ``input_ids`` ``(B, S+T)`` and, when a tokenizer is
            available, ``texts`` (one decoded string per batch row).
        """
        cfg = inference_config or InferenceConfig()
        model = self.model
        if not isinstance(model, FrankensteinDecoder):
            raise ValueError(
                "Generation requires a decoder model (config.mode == 'decoder'); "
                "deployed artifact is an encoder/ViT."
            )

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(prompt, dict):
            prompt = prompt.get("input_ids", prompt)
        if isinstance(prompt, (list, tuple)) and prompt and isinstance(prompt[0], str):
            if self.tokenizer is None:
                raise ValueError(
                    "A tokenizer is required to generate from text; deploy one "
                    "into the artifact directory or pass an int64 token tensor."
                )
            input_ids = []
            for text in prompt:
                encoded = self.tokenizer.encode(text)
                input_ids.append(encoded["input_ids"][:max_length])
            max_len = max(len(seq) for seq in input_ids)
            input_tensor = torch.tensor(
                [seq + [0] * (max_len - len(seq)) for seq in input_ids], dtype=torch.long
            )
        elif isinstance(prompt, torch.Tensor):
            input_tensor = prompt.to(torch.long)
        else:
            input_tensor = torch.tensor(prompt, dtype=torch.long)

        input_tensor = input_tensor.to(self.device)
        cache = FrankensteinCache.new(model, cfg, device=self.device)
        out_ids = model.generate(
            input_tensor,
            max_new_tokens=int(cfg.max_new_tokens),
            temperature=float(cfg.temperature),
            top_k=int(cfg.top_k),
            top_p=float(cfg.top_p),
            greedy=bool(cfg.greedy),
            past_key_values=cache,
            use_cache=bool(cfg.kv_cache_enabled),
        )
        result = {"input_ids": out_ids, "cache": cache}
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            result["texts"] = [
                self.tokenizer.decode(row.tolist()) for row in out_ids
            ]
        return result

    def benchmark_generation(
        self,
        prompt_length: int = 64,
        max_new_tokens: int = 64,
        inference_config: Optional[InferenceConfig] = None,
        num_runs: int = 3,
    ):
        """Benchmark generation split into prefill and decode phases.

        Logs prefill tokens/sec and decode tokens/sec separately (the two
        regimes have different computational profiles, arXiv:2403.02310).

        Args:
            prompt_length: Prompt length for the dummy input.
            max_new_tokens: Number of tokens generated per run.
            inference_config: Resolved runtime config.
            num_runs: Number of timed runs (best is reported).
        """
        cfg = inference_config or InferenceConfig()
        model = self.model
        if not isinstance(model, FrankensteinDecoder):
            raise ValueError("Generation benchmarking requires a decoder model.")
        was_cached = bool(cfg.kv_cache_enabled)
        best = {"prefill_tps": 0.0, "decode_tps": 0.0}
        for _ in range(num_runs):
            dummy = torch.randint(0, self.config.vocab_size, (1, prompt_length), device=self.device)
            cache = FrankensteinCache.new(model, cfg, device=self.device)
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            logits, cache = model(
                dummy, past_key_values=cache, use_cache=was_cached, return_cache=was_cached
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            prefill_s = time.time() - t0
            prefill_tps = prompt_length / max(prefill_s, 1e-9)

            next_logits = logits[:, -1, :]
            decode_tps = 0.0
            if max_new_tokens > 1:
                t1 = time.time()
                steps = max_new_tokens - 1
                for step in range(steps):
                    token = torch.argmax(next_logits, dim=-1, keepdim=True)
                    if step < steps - 1:
                        logits, cache = model(
                            token,
                            past_key_values=cache if was_cached else None,
                            use_cache=was_cached,
                            cache_position=cache.next_position() if was_cached else None,
                            return_cache=was_cached,
                        )
                        next_logits = logits[:, -1, :]
                if self.device == "cuda":
                    torch.cuda.synchronize()
                decode_s = time.time() - t1
                decode_tps = steps / max(decode_s, 1e-9)
            best["prefill_tps"] = max(best["prefill_tps"], prefill_tps)
            best["decode_tps"] = max(best["decode_tps"], decode_tps)

        logger.info("Generation benchmark (cache=%s):", was_cached)
        logger.info("  Prefill: %.0f tokens/sec", best["prefill_tps"])
        logger.info("  Decode:  %.0f tokens/sec", best["decode_tps"])
        return best
    
    def _load_model(self) -> FrankensteinEncoder:
        """Load and prepare model for inference."""
        # Initialize model (decoder when configured)
        if isinstance(self.config.mode, str) and self.config.mode == "decoder":
            model = FrankensteinDecoder(self.config)
        elif hasattr(self.config, "patch_size") and getattr(self.config, "pos_embedding_type", "") in ("learned_1d", "none"):
            from ..model.frankenstein_vit import FrankensteinViT
            model = FrankensteinViT(self.config)
        else:
            model = FrankensteinEncoder(self.config)
        
        # Find model file
        model_files = list(self.model_dir.glob("model*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No model file found in {self.model_dir}")
        
        model_path = model_files[0]
        logger.info(f"Loading model from {model_path.name}")
        
        # Load weights
        if 'quantized' in model_path.name:
            load_quantized_checkpoint(str(model_path), model)
        else:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            state_dict = (
                checkpoint.get('model_state_dict')
                or checkpoint.get('state_dict')
                or checkpoint
            )
            model.load_state_dict(state_dict, strict=False)
        
        # Prepare for inference
        model.eval()
        model.to(self.device)
        
        # FP16 only when no BitLinear layers are present (BitLinear's STE
        # activation/weight quantization assumes float32 statistics).
        has_bitlinear = any(isinstance(m, BitLinear) for m in model.modules())
        if self.use_half_precision:
            if has_bitlinear:
                logger.warning(
                    "FP16 requested but model contains BitLinear layers; "
                    "keeping FP32 to preserve ternary quantization statistics."
                )
            else:
                model.half()
                logger.info("Using FP16 precision")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded: {total_params / 1e6:.2f}M parameters")
        
        return model
    
    def _load_tokenizer(self):
        """Load the deploy tokenizer (SPM first, then HF fallback).

        Prefers the legacy SentencePiece ``tokenizer.model``. When the
        deployment directory instead carries a HuggingFace ``tokenizer.json``
        (e.g. trained via ``tokenizer.source=train_from_dataset``), an
        ``AutoTokenizer`` wrapper exposing the SPM-like ``encode`` /
        ``decode`` / ``mask_id`` interface is loaded instead so inference
        token IDs match the deployed model's vocabulary.
        """
        tokenizer_path = self.model_dir / "tokenizer.model"
        hf_tokenizer_json = self.model_dir / "tokenizer.json"

        if not tokenizer_path.exists() and hf_tokenizer_json.exists():
            return self._load_hf_tokenizer()

        if not tokenizer_path.exists():
            logger.warning("Tokenizer not found in deployment directory")
            logger.warning("You'll need to provide token IDs directly")
            return None

        tokenizer = SpanishSPMTokenizer(
            vocab_size=self.config.vocab_size,
            model_path=str(tokenizer_path)
        )
        logger.info("Tokenizer loaded")
        return tokenizer

    def _load_hf_tokenizer(self):
        """Wrap a HuggingFace tokenizer with the SPM-like encode/decode API."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            logger.warning("transformers not installed; cannot load HF tokenizer")
            return None
        try:
            hf_tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load HF tokenizer: %s", exc)
            return None
        if len(hf_tok) != int(getattr(self.config, "vocab_size", len(hf_tok))):
            logger.warning(
                "Deployed tokenizer vocab (%d) != config vocab_size (%s); "
                "using the tokenizer's size for the model interface.",
                len(hf_tok), getattr(self.config, "vocab_size", None),
            )
        mask_id = hf_tok.mask_token_id
        logger.info("HF tokenizer loaded (vocab=%d)", len(hf_tok))

        class _HFTokenizerAdapter:
            """Expose ``encode``/``decode``/``mask_id`` like SpanishSPMTokenizer."""

            def __init__(self, hf_tok):
                self._tok = hf_tok
                self.mask_id = mask_id

            def encode(self, text, max_length=512):
                enc = self._tok(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                return {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                }

            def decode(self, ids):
                return self._tok.decode(ids)

        return _HFTokenizerAdapter(hf_tok)
    
    @torch.no_grad()
    def predict(
        self,
        inputs: Union[str, List[str], torch.Tensor],
        max_length: int = 512,
        return_logits: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Run inference on input(s).
        
        Args:
            inputs: Text string(s) or token tensor
            max_length: Maximum sequence length
            return_logits: If True, return raw logits; else return probabilities
            
        Returns:
            Model predictions (logits or probabilities)
        """
        # Convert inputs to tensor
        if isinstance(inputs, str):
            inputs = [inputs]
        
        if isinstance(inputs, list):
            # Tokenize text
            if self.tokenizer is None:
                raise ValueError("Tokenizer not available. Provide token IDs directly.")
            
            input_ids = []
            for text in inputs:
                encoded = self.tokenizer.encode(text)
                tokens = encoded["input_ids"][:max_length]  # Truncate
                input_ids.append(tokens)
            
            # Pad sequences
            max_len = max(len(seq) for seq in input_ids)
            padded_ids = []
            for seq in input_ids:
                padded = seq + [0] * (max_len - len(seq))
                padded_ids.append(padded)
            
            input_tensor = torch.tensor(padded_ids, dtype=torch.long)
        else:
            input_tensor = inputs
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        if self.use_half_precision:
            # Note: Input IDs stay as long, model will handle conversion
            pass
        
        # Run inference
        start_time = time.time()
        outputs = self.model(input_tensor)
        inference_time = time.time() - start_time
        
        # Log performance
        batch_size = input_tensor.size(0)
        seq_len = input_tensor.size(1)
        tokens_per_sec = (batch_size * seq_len) / inference_time
        logger.debug(f"Inference: {inference_time*1000:.2f}ms ({tokens_per_sec:.0f} tokens/sec)")
        
        if return_logits:
            return outputs
        else:
            return module_softmax(self.model, outputs, dim=-1)
    
    def predict_masked(
        self,
        text: str,
        mask_token: str = "[MASK]"
    ) -> List[tuple]:
        """
        Predict masked tokens (MLM task).
        
        Args:
            text: Text with [MASK] tokens
            mask_token: Mask token string
            
        Returns:
            List of (position, top_predictions) tuples
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for masked prediction")
        
        # Tokenize
        encoded = self.tokenizer.encode(text)
        tokens = encoded["input_ids"]
        mask_positions = [i for i, t in enumerate(tokens) if t == self.tokenizer.mask_id]
        
        if not mask_positions:
            logger.warning("No mask tokens found in text")
            return []
        
        # Run prediction
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        probs = self.predict(input_tensor, return_logits=False)[0]
        
        # Extract predictions for masked positions
        results = []
        for pos in mask_positions:
            top_probs, top_ids = torch.topk(probs[pos], k=5)
            top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_ids]
            predictions = list(zip(top_tokens, top_probs.tolist()))
            results.append((pos, predictions))
        
        return results
    
    def batch_predict(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512
    ) -> List[torch.Tensor]:
        """
        Process multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            List of prediction tensors
        """
        all_predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            predictions = self.predict(batch, max_length=max_length)
            all_predictions.extend(predictions)
        
        return all_predictions
    
    def benchmark(self, batch_size: int = 1, seq_length: int = 512, num_runs: int = 10):
        """
        Benchmark inference performance.
        
        Args:
            batch_size: Batch size to test
            seq_length: Sequence length to test
            num_runs: Number of runs for averaging
        """
        logger.info(f"Benchmarking: batch_size={batch_size}, seq_length={seq_length}")
        
        # Create dummy input
        dummy_input = torch.randint(
            0, self.config.vocab_size,
            (batch_size, seq_length),
            device=self.device
        )
        
        # Warmup
        for _ in range(3):
            _ = self.predict(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.predict(dummy_input)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        tokens_per_sec = (batch_size * seq_length) / avg_time
        
        logger.info(f"Results:")
        logger.info(f"  Average time: {avg_time*1000:.2f}ms")
        logger.info(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
        logger.info(f"  Per-token latency: {avg_time/seq_length*1000:.3f}ms")
        
        if self.device == 'cuda':
            memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
            logger.info(f"  Peak GPU memory: {memory_mb:.2f}MB")


def interactive_mode(engine: FrankensteinInference):
    """Interactive inference mode."""
    print("\n" + "="*60)
    print("Frankenstein Interactive Inference")
    print("="*60)
    print("\nCommands:")
    print("  - Enter text to get predictions")
    print("  - 'quit' or 'exit' to exit")
    print("  - 'benchmark' to run performance test")
    print("="*60 + "\n")
    
    while True:
        try:
            text = input(">>> ").strip()
            
            if text.lower() in ['quit', 'exit']:
                break
            
            if text.lower() == 'benchmark':
                engine.benchmark()
                continue
            
            if not text:
                continue
            
            # Run prediction
            start = time.time()
            predictions = engine.predict(text)
            elapsed = time.time() - start
            
            print(f"\nPrediction shape: {predictions.shape}")
            print(f"Time: {elapsed*1000:.2f}ms")
            
            # Show top predictions for first position (example)
            if predictions.dim() == 3:
                first_pos_probs = predictions[0, 0]  # First batch, first position
                top_probs, top_ids = torch.topk(first_pos_probs, k=5)
                print("\nTop 5 predictions (first position):")
                for prob, idx in zip(top_probs, top_ids):
                    print(f"  Token {idx.item()}: {prob.item():.4f}")
            
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Frankenstein Inference Engine'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to deployed model directory'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Text to process (for single prediction)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input file with texts (one per line)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for predictions'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=SUPPORTED_DEVICE_CHOICES,
        default='auto',
        help='Device to run inference on'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use FP16 precision (GPU only)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for batch processing'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark'
    )
    # ---- Generation mode (causal_lm) ----
    parser.add_argument(
        '--task',
        type=str,
        choices=['mlm', 'causal_lm'],
        default=None,
        help='mlm (default): full-forward prediction; causal_lm: autoregressive generation'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='YAML preset carrying an inference: block (generation + KV cache)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Prompt text for generation (--task causal_lm)'
    )
    parser.add_argument(
        '--max-new-tokens',
        dest='max_new_tokens',
        type=int,
        default=None,
        help='Maximum tokens to generate (inference.generation.max_new_tokens)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Sampling temperature (inference.generation.temperature)'
    )
    parser.add_argument(
        '--top-p',
        dest='top_p',
        type=float,
        default=None,
        help='Nucleus mass (inference.generation.top_p; 1.0 disables)'
    )
    parser.add_argument(
        '--greedy',
        dest='greedy',
        action='store_true',
        default=None,
        help='Greedy argmax decoding (overrides sampling)'
    )
    # ---- KV-cache flags (CLI > YAML > default, D2) ----
    parser.add_argument(
        '--kv-cache',
        dest='kv_cache',
        action='store_true',
        default=None,
        help='Enable the KV cache for generation'
    )
    parser.add_argument(
        '--no-kv-cache',
        dest='kv_cache',
        action='store_false',
        help='Disable the KV cache for generation'
    )
    parser.add_argument(
        '--kv-quant',
        dest='kv_quant',
        type=str,
        choices=['none', 'int8', 'int4_kivi'],
        default=None,
        help='KV quantization codec (inference.kv_cache.quantization.method)'
    )
    parser.add_argument(
        '--kv-group-size',
        dest='kv_group_size',
        type=int,
        default=None,
        help='Tokens per quantization group (int4_kivi)'
    )
    parser.add_argument(
        '--kv-protected-window',
        dest='kv_protected_window',
        type=int,
        default=None,
        help='Recent tokens kept in full precision (int4_kivi)'
    )
    parser.add_argument(
        '--kv-compress',
        dest='kv_compress',
        type=str,
        choices=['none', 'window_sinks'],
        default=None,
        help='KV eviction policy (inference.kv_cache.compression.method)'
    )
    parser.add_argument(
        '--kv-num-sinks',
        dest='kv_num_sinks',
        type=int,
        default=None,
        help='Pinned leading attention-sink positions'
    )
    parser.add_argument(
        '--kv-window',
        dest='kv_window',
        type=int,
        default=None,
        help='Sliding window size (window_sinks)'
    )
    parser.add_argument(
        '--benchmark-generation',
        dest='benchmark_generation',
        action='store_true',
        default=None,
        help='Benchmark generation split into prefill/decode throughput'
    )
    
    args = parser.parse_args(argv)
    resolved_device = resolve_torch_device(args.device)
    logger.info(f"Inference device requested='{args.device}', resolved='{resolved_device}'")
    
    # Initialize engine
    engine = FrankensteinInference(
        args.model,
        device=resolved_device,
        use_half_precision=args.fp16
    )

    # ---- Generation mode (causal_lm) ----
    generation_requested = (
        args.task == 'causal_lm'
        or args.prompt is not None
        or bool(args.benchmark_generation)
    )
    if generation_requested:
        if not isinstance(engine.model, FrankensteinDecoder):
            logger.error(
                "Generation requires a decoder model (config.mode == 'decoder'); "
                "the deployed artifact is not a decoder."
            )
            return 2
        cfg = engine.load_inference_config(yaml_path=args.config, args=args)
        if args.benchmark_generation:
            engine.benchmark_generation(
                max_new_tokens=int(cfg.max_new_tokens),
                inference_config=cfg,
            )
            return 0
        prompt = args.prompt
        if prompt is None and args.input:
            input_path = Path(args.input)
            if input_path.suffix == ".pt":
                # Token-id tensor (useful when the artifact ships without a
                # tokenizer): (B, S) int64 or a 1-D sequence.
                prompt = torch.load(input_path, map_location="cpu", weights_only=False)
            else:
                with open(input_path) as handle:
                    prompt = [line.strip() for line in handle if line.strip()]
        if prompt is None:
            logger.error("--task causal_lm requires --prompt (or --input file)")
            return 2
        result = engine.generate(prompt, inference_config=cfg)
        texts = result.get("texts")
        if texts:
            for row in texts:
                print(row)
        else:
            ids = result["input_ids"]
            print(f"Generated ids shape: {tuple(ids.shape)}")
        if args.output:
            torch.save(result["input_ids"].cpu(), args.output)
            logger.info("Generated ids saved to %s", args.output)
        return 0
    
    # Run benchmark if requested
    if args.benchmark:
        engine.benchmark()
        return 0
    
    # Single text prediction
    if args.text:
        predictions = engine.predict(args.text)
        print(f"Prediction shape: {predictions.shape}")
        print(f"Output: {predictions}")
        return 0
    
    # Batch file processing
    if args.input:
        logger.info(f"Processing file: {args.input}")
        
        with open(args.input) as f:
            texts = [line.strip() for line in f if line.strip()]
        
        predictions = engine.batch_predict(texts, batch_size=args.batch_size)
        
        if args.output:
            # Save predictions
            torch.save(predictions, args.output)
            logger.info(f"Predictions saved to {args.output}")
        else:
            print(f"Processed {len(texts)} texts")
            for i, pred in enumerate(predictions[:3]):  # Show first 3
                print(f"Text {i+1} prediction shape: {pred.shape}")
        
        return 0
    
    # Interactive mode
    interactive_mode(engine)
    return 0


if __name__ == "__main__":
    exit(main())
