#!/usr/bin/env python3
"""
Complete example of the deployment pipeline.
Demonstrates: Model creation -> Quantization -> Deployment -> Inference

Run this to test the full pipeline without a training checkpoint.
"""

import torch
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.tormented_bert_frankestein import TormentedBertFrankenstein, UltraConfig
from deploy.quantization import save_quantized_checkpoint, estimate_model_size
from deploy.deploy import ModelDeployer
from deploy.inference import TormentedBertInference


def main():
    print("="*70)
    print("TORMENTED-BERT v2: Complete Deployment Pipeline Example")
    print("="*70)
    
    # Step 1: Create a model (normally you'd load a trained one)
    print("\n[1/5] Creating model...")
    config = UltraConfig(
        vocab_size=10000,  # Smaller for demo
        hidden_size=768,   # Smaller for demo
        num_layers=4,      # Fewer layers for demo
        num_loops=1,
        ode_steps=2
    )
    
    model = TormentedBertFrankenstein(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params / 1e6:.2f}M parameters")
    
    # Step 2: Estimate sizes
    print("\n[2/5] Estimating model sizes...")
    sizes = estimate_model_size(model)
    for format_name, size_mb in sizes.items():
        print(f"  {format_name}: {size_mb:.2f} MB")
    
    # Step 3: Save as quantized checkpoint
    print("\n[3/5] Saving quantized checkpoint...")
    output_dir = Path("example_deployment")
    output_dir.mkdir(exist_ok=True)
    
    checkpoint_path = output_dir / "model_quantized.pt"
    save_quantized_checkpoint(
        model,
        str(checkpoint_path),
        additional_data={
            'config': config.__dict__,
            'vocab_size': config.vocab_size,
            'hidden_size': config.hidden_size,
        }
    )
    print(f"✓ Checkpoint saved to {checkpoint_path}")
    
    # Step 4: Create deployment package
    print("\n[4/5] Creating deployment package...")
    deployer = ModelDeployer(config)
    deployer.model = model
    deployer.convert_to_deployment(
        str(output_dir),
        save_format='quantized'
    )
    print(f"✓ Deployment package created in {output_dir}")
    
    # Step 5: Test inference
    print("\n[5/5] Testing inference...")
    engine = TormentedBertInference(
        str(output_dir),
        device='cpu'  # Use CPU for demo (change to 'cuda' if available)
    )
    
    # Create dummy input
    dummy_text_ids = torch.randint(0, config.vocab_size, (2, 32))
    
    print("Running inference on dummy input...")
    predictions = engine.predict(dummy_text_ids)
    print(f"✓ Inference successful!")
    print(f"  Input shape: {dummy_text_ids.shape}")
    print(f"  Output shape: {predictions.shape}")
    
    # Benchmark
    print("\n[Bonus] Running quick benchmark...")
    engine.benchmark(batch_size=1, seq_length=64, num_runs=5)
    
    print("\n" + "="*70)
    print("✅ Pipeline test complete!")
    print("="*70)
    print(f"\nDeployment artifacts saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print(f"  1. Check the deployment: ls {output_dir}")
    print(f"  2. Run inference: python inference.py --model {output_dir}")
    print(f"  3. Use in your application with the Python API")
    print("="*70)


if __name__ == "__main__":
    main()
