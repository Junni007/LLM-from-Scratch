"""
Example demonstrating Supervised Fine-Tuning (SFT) loss functions.
"""

import torch
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sft.loss import (
    CausalLMLoss, 
    InstructionLoss, 
    WeightedInstructionLoss, 
    compute_perplexity
)


def main():
    """Main function demonstrating SFT loss functions."""
    print("Supervised Fine-Tuning (SFT) Loss Functions Example")
    print("=" * 50)
    
    # Model parameters
    batch_size = 4
    seq_len = 16
    vocab_size = 1000
    
    # Create sample logits and labels
    print("1. Creating sample data...")
    torch.manual_seed(42)  # For reproducible results
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Test CausalLMLoss
    print("\n2. Testing CausalLMLoss:")
    causal_loss_fn = CausalLMLoss(ignore_index=-100, label_smoothing=0.1)
    
    # Standard loss
    causal_loss = causal_loss_fn(logits, labels)
    print(f"  Standard causal LM loss: {causal_loss.item():.4f}")
    
    # Loss with some ignored tokens
    labels_with_ignore = labels.clone()
    labels_with_ignore[:, :3] = -100  # Ignore first 3 tokens of each sequence
    causal_loss_ignored = causal_loss_fn(logits, labels_with_ignore)
    print(f"  Causal LM loss (with ignored tokens): {causal_loss_ignored.item():.4f}")
    
    # Test InstructionLoss
    print("\n3. Testing InstructionLoss:")
    instruction_loss_fn = InstructionLoss(
        ignore_index=-100, 
        label_smoothing=0.05,
        instruction_weight=0.1
    )
    
    # Create instruction mask (1 for instruction tokens, 0 for response tokens)
    instruction_mask = torch.zeros(batch_size, seq_len)
    instruction_mask[:, :5] = 1.0  # First 5 tokens are instruction
    
    print(f"  Instruction mask shape: {instruction_mask.shape}")
    print(f"  Instruction tokens per sequence: {instruction_mask.sum(dim=1)[0].item()}")
    print(f"  Response tokens per sequence: {(1.0 - instruction_mask).sum(dim=1)[0].item()}")
    
    # Loss with instruction mask
    instruction_loss = instruction_loss_fn(logits, labels, instruction_mask)
    print(f"  Instruction-aware loss: {instruction_loss.item():.4f}")
    
    # Loss without instruction mask (should be same as causal loss)
    instruction_loss_no_mask = instruction_loss_fn(logits, labels)
    print(f"  Instruction loss (no mask): {instruction_loss_no_mask.item():.4f}")
    
    # Test different instruction weights
    print("\n4. Testing different instruction weights:")
    weights = [0.0, 0.1, 0.5, 1.0]
    for weight in weights:
        loss_fn = InstructionLoss(instruction_weight=weight)
        loss = loss_fn(logits, labels, instruction_mask)
        print(f"  Weight {weight:.1f}: Loss = {loss.item():.4f}")
    
    # Test WeightedInstructionLoss
    print("\n5. Testing WeightedInstructionLoss:")
    weighted_loss_fn = WeightedInstructionLoss(ignore_index=-100, label_smoothing=0.05)
    
    # Create token weights (give more importance to later tokens)
    token_weights = torch.ones(batch_size, seq_len)
    token_weights[:, -4:] = 2.0  # Double weight for last 4 tokens
    
    print(f"  Token weights shape: {token_weights.shape}")
    print(f"  Normal weight tokens: {(token_weights == 1.0).sum().item()}")
    print(f"  Double weight tokens: {(token_weights == 2.0).sum().item()}")
    
    # Loss with token weights
    weighted_loss = weighted_loss_fn(logits, labels, token_weights)
    print(f"  Weighted instruction loss: {weighted_loss.item():.4f}")
    
    # Compare with standard loss
    weighted_loss_no_weights = weighted_loss_fn(logits, labels)
    print(f"  Weighted loss (no weights): {weighted_loss_no_weights.item():.4f}")
    
    # Test with ignored tokens
    labels_with_ignore = labels.clone()
    labels_with_ignore[:, 0] = -100  # Ignore first token
    weighted_loss_ignored = weighted_loss_fn(logits, labels_with_ignore, token_weights)
    print(f"  Weighted loss (with ignored tokens): {weighted_loss_ignored.item():.4f}")
    
    # Test perplexity computation
    print("\n6. Testing perplexity computation:")
    perplexities = []
    losses = [causal_loss, instruction_loss, weighted_loss]
    
    for i, loss in enumerate(losses):
        ppl = compute_perplexity(loss)
        perplexities.append(ppl)
        print(f"  Perplexity {i+1}: {ppl.item():.2f}")
    
    # Show relationship between loss and perplexity
    print("\n7. Loss vs Perplexity relationship:")
    test_losses = [0.5, 1.0, 2.0, 3.0]
    for loss_val in test_losses:
        loss_tensor = torch.tensor(loss_val)
        ppl = compute_perplexity(loss_tensor)
        print(f"  Loss: {loss_val:.1f} -> Perplexity: {ppl.item():.2f}")
    
    # Performance comparison
    print("\n8. Performance comparison:")
    import time
    
    # Time causal loss computation
    start_time = time.time()
    for _ in range(100):
        _ = causal_loss_fn(logits, labels)
    causal_time = time.time() - start_time
    
    # Time instruction loss computation
    start_time = time.time()
    for _ in range(100):
        _ = instruction_loss_fn(logits, labels, instruction_mask)
    instruction_time = time.time() - start_time
    
    # Time weighted loss computation
    start_time = time.time()
    for _ in range(100):
        _ = weighted_loss_fn(logits, labels, token_weights)
    weighted_time = time.time() - start_time
    
    print(f"  CausalLMLoss (100 iterations): {causal_time:.4f}s")
    print(f"  InstructionLoss (100 iterations): {instruction_time:.4f}s")
    print(f"  WeightedInstructionLoss (100 iterations): {weighted_time:.4f}s")
    
    print("\nSFT loss functions example completed!")


if __name__ == "__main__":
    main()