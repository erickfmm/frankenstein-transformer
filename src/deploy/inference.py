#!/usr/bin/env python3
"""High-performance inference engine for deployed TORMENTED-BERT models.

Provides :class:`TormentedBertInference` for loading quantized or standard
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
    from ..model.tormented_bert_frankestein import (
        FrankensteinDecoder,
        TormentedBertFrankenstein,
        UltraConfig,
    )
    from .quantization import load_quantized_checkpoint
    from ..tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
    from ..utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device
    from ..model.attention.common import BitLinear
    from ..utils.config_flatten import flatten_model_dict
except ImportError:
    from model.tormented_bert_frankestein import (
        FrankensteinDecoder,
        TormentedBertFrankenstein,
        UltraConfig,
    )
    from deploy.quantization import load_quantized_checkpoint
    from tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
    from utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device
    from model.attention.common import BitLinear
    from utils.config_flatten import flatten_model_dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TormentedBertInference:
    """Optimized inference engine for TORMENTED-BERT deployed models.

    Loads deployment artifacts (config, quantized or standard weights,
    optional tokenizer) and provides methods for single/batch prediction,
    masked token prediction, and performance benchmarking.

    Attributes:
        model_dir: Path to the deployment directory.
        device: Resolved PyTorch device string.
        use_half_precision: Whether FP16 inference is enabled (CUDA only).
        config: Loaded :class:`UltraConfig`.
        model: Loaded :class:`TormentedBertFrankenstein` in eval mode.
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
        
        logger.info(f"Initializing inference engine on {self.device}")
        
        # Load config
        self.config = self._load_config()
        
        # Load model
        self.model = self._load_model()
        
        # Load tokenizer if available
        self.tokenizer = self._load_tokenizer()
        
        logger.info("✅ Inference engine ready")
    
    def _load_config(self) -> UltraConfig:
        """Load model configuration."""
        config_path = self.model_dir / "config.json"
        
        if not config_path.exists():
            logger.warning("Config file not found, using default")
            return UltraConfig()
        
        with open(config_path) as f:
            config_dict = json.load(f)
        
        config = UltraConfig(**flatten_model_dict(config_dict))
        logger.info(f"Config loaded: {config.hidden_size}D, {config.num_layers} layers")
        return config
    
    def _load_model(self) -> TormentedBertFrankenstein:
        """Load and prepare model for inference."""
        # Initialize model (decoder when configured)
        if isinstance(self.config.mode, str) and self.config.mode == "decoder":
            model = FrankensteinDecoder(self.config)
        else:
            model = TormentedBertFrankenstein(self.config)
        
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
    
    def _load_tokenizer(self) -> Optional[SpanishSPMTokenizer]:
        """Load tokenizer if available."""
        tokenizer_path = self.model_dir / "tokenizer.model"
        
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
                tokens = self.tokenizer.encode(text)
                tokens = tokens[:max_length]  # Truncate
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
            return torch.softmax(outputs, dim=-1)
    
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
        tokens = self.tokenizer.encode(text)
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


def interactive_mode(engine: TormentedBertInference):
    """Interactive inference mode."""
    print("\n" + "="*60)
    print("TORMENTED-BERT Interactive Inference")
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
        description='TORMENTED-BERT Inference Engine'
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
    
    args = parser.parse_args(argv)
    resolved_device = resolve_torch_device(args.device)
    logger.info(f"Inference device requested='{args.device}', resolved='{resolved_device}'")
    
    # Initialize engine
    engine = TormentedBertInference(
        args.model,
        device=resolved_device,
        use_half_precision=args.fp16
    )
    
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
