"""
Example demonstrating Mixture of Experts (MoE) layer implementation.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from moe.moe_layer import MoELayer, SwitchTransformerLayer


def main():
    """Main function demonstrating MoE layer."""
    print("Mixture of Experts (MoE) Layer Example")
    print("=" * 40)
    
    # Model parameters
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_experts = 4
    top_k = 2
    
    # Create input tensor
    print("Creating input tensor...")
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    
    # Test MoELayer
    print("\nTesting MoELayer...")
    moe_layer = MoELayer(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        d_ff=128,
        dropout=0.1
    )
    
    # Forward pass
    output = moe_layer(x)
    balance_loss = moe_layer.balance_loss
    
    print(f"Output shape: {output.shape}")
    print(f"Balance loss: {balance_loss.item():.6f}")
    
    # Verify output shape matches input
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
    
    # Show that output is different from input (layer is doing something)
    diff = torch.mean(torch.abs(output - x))
    print(f"Mean absolute difference from input: {diff.item():.6f}")
    
    # Test SwitchTransformerLayer
    print("\nTesting SwitchTransformerLayer...")
    switch_layer = SwitchTransformerLayer(
        d_model=d_model,
        num_experts=num_experts,
        d_ff=128,
        dropout=0.1
    )
    
    # Forward pass
    output = switch_layer(x)
    balance_loss = switch_layer.balance_loss
    
    print(f"Output shape: {output.shape}")
    print(f"Balance loss: {balance_loss.item():.6f}")
    
    # Verify output shape matches input
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
    
    # Compare MoE layer with standard FFN
    print("\nComparing with standard Feed-Forward Network...")
    
    # Standard FFN
    ffn = nn.Sequential(
        nn.Linear(d_model, 128),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(128, d_model),
        nn.Dropout(0.1)
    )
    
    ffn_output = ffn(x)
    print(f"Standard FFN output shape: {ffn_output.shape}")
    
    # Compare parameter counts
    moe_params = sum(p.numel() for p in moe_layer.parameters())
    switch_params = sum(p.numel() for p in switch_layer.parameters())
    ffn_params = sum(p.numel() for p in ffn.parameters())
    
    print(f"\nParameter comparison:")
    print(f"  MoELayer parameters: {moe_params:,}")
    print(f"  SwitchTransformerLayer parameters: {switch_params:,}")
    print(f"  Standard FFN parameters: {ffn_params:,}")
    
    # Show expert utilization
    print(f"\nTesting expert utilization...")
    
    # Create a larger batch to better see expert distribution
    large_batch_size = 16
    large_x = torch.randn(large_batch_size, seq_len, d_model)
    
    # Test with MoELayer
    moe_layer.eval()  # Set to eval mode for consistent routing
    with torch.no_grad():
        moe_output = moe_layer(large_x)
        
        # Access internal routing (this is a simplified approach)
        router = moe_layer.expert_router
        router_logits, top_k_indices, top_k_gates = router(large_x)
        
        # Show expert selection distribution
        top_1_experts = top_k_indices[:, :, 0].flatten()
        expert_counts = torch.bincount(top_1_experts, minlength=num_experts)
        total_tokens = large_batch_size * seq_len
        
        print(f"Expert selection distribution (top-1):")
        for i in range(num_experts):
            fraction = expert_counts[i].item() / total_tokens
            print(f"  Expert {i}: {expert_counts[i].item():3.0f} tokens ({fraction:.2%})")
    
    print("\nMoE layer example completed!")


if __name__ == "__main__":
    main()