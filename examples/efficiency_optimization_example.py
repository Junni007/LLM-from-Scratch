"""
Example demonstrating efficiency optimization techniques.

This example shows how to apply quantization, compression, and optimized training
techniques to make LLMs more efficient.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import sys
import os

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.quantization import apply_quantization, compare_model_sizes
from src.utils.compression import apply_pruning, apply_low_rank_compression, setup_knowledge_distillation
from src.train.optimization import create_optimized_trainer, profile_memory_usage

def create_sample_models():
    """Create sample models for demonstration."""
    # Create a small GPT-2 model for demonstration
    config = GPT2Config(
        vocab_size=1000,  # Small vocab for demo
        n_positions=256,
        n_embd=128,
        n_layer=2,
        n_head=2
    )
    
    # Create base model
    base_model = GPT2LMHeadModel(config)
    
    return base_model

def create_sample_inputs(batch_size=2, seq_length=10):
    """Create sample inputs for demonstration."""
    # Create random input IDs and labels
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    labels = torch.randint(0, 1000, (batch_size, seq_length))
    
    return input_ids, labels

def main():
    """Main function demonstrating efficiency optimizations."""
    print("=== Efficiency Optimization Example ===")
    
    # Create sample base model
    print("Creating sample base model...")
    original_model = create_sample_models()
    print(f"Original model parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    
    # Create sample inputs
    print("Creating sample inputs...")
    input_ids, labels = create_sample_inputs(batch_size=2, seq_length=10)
    print(f"Input shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 1. Quantization
    print("\n=== 1. Quantization ===")
    quantized_model = apply_quantization(original_model, bits=8)
    print(f"Quantized model parameters: {sum(p.numel() for p in quantized_model.parameters()):,}")
    compare_model_sizes(original_model, quantized_model)
    
    # 2. Pruning
    print("\n=== 2. Pruning ===")
    pruned_model = apply_pruning(original_model, sparsity_ratio=0.3)
    print(f"Pruned model parameters: {sum(p.numel() for p in pruned_model.parameters()):,}")
    
    # Count zero weights in pruned model
    zero_weights = 0
    total_weights = 0
    for param in pruned_model.parameters():
        zero_weights += (param == 0).sum().item()
        total_weights += param.numel()
    
    print(f"Sparsity ratio: {zero_weights / total_weights:.2%}")
    
    # 3. Low-rank compression
    print("\n=== 3. Low-rank Compression ===")
    compressed_model = apply_low_rank_compression(original_model, rank_ratio=0.5)
    print(f"Compressed model parameters: {sum(p.numel() for p in compressed_model.parameters()):,}")
    
    # 4. Knowledge distillation setup
    print("\n=== 4. Knowledge Distillation ===")
    # Create a smaller student model
    student_config = GPT2Config(
        vocab_size=1000,
        n_positions=256,
        n_embd=64,  # Smaller hidden size
        n_layer=1,  # Fewer layers
        n_head=1    # Fewer attention heads
    )
    student_model = GPT2LMHeadModel(student_config)
    
    print(f"Teacher model parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    print(f"Student model parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    print(f"Compression ratio: {sum(p.numel() for p in student_model.parameters()) / sum(p.numel() for p in original_model.parameters()):.2%}")
    
    # Setup distillation
    distiller = setup_knowledge_distillation(original_model, student_model)
    print("Knowledge distillation setup complete")
    
    # 5. Optimized training
    print("\n=== 5. Optimized Training ===")
    # Create optimized trainer
    trainer = create_optimized_trainer(
        original_model,
        learning_rate=5e-5,
        gradient_accumulation_steps=2,
        mixed_precision=True,
        gradient_clipping=1.0
    )
    
    print("Optimized trainer created with:")
    print("  - Gradient accumulation: 2 steps")
    print("  - Mixed precision training: Enabled")
    print("  - Gradient clipping: 1.0")
    
    # Profile memory usage
    print("\n=== 6. Memory Profiling ===")
    if torch.cuda.is_available():
        print("Profiling memory usage...")
        original_model.cuda()
        input_ids = input_ids.cuda()
        labels = labels.cuda()
        profile_memory_usage(original_model, input_ids, labels)
        # Move back to CPU for compatibility
        original_model.cpu()
        input_ids = input_ids.cpu()
        labels = labels.cpu()
    else:
        print("CUDA not available, skipping memory profiling")
    
    # Demonstrate a training step
    print("\n=== 7. Training Step Example ===")
    loss = trainer.train_step(input_ids, labels)
    print(f"Training step completed with loss: {loss:.4f}")
    
    print("\n=== Efficiency Optimization Example Completed ===")

if __name__ == "__main__":
    main()