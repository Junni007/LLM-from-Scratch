"""
Example demonstrating Hybrid Mixture of Experts (MoE) implementation.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from moe.hybrid_moe import HybridMoEBlock, AdaptiveHybridMoE, SparseMoEWithSharedExperts


def main():
    """Main function demonstrating hybrid MoE implementations."""
    print("Hybrid Mixture of Experts (MoE) Example")
    print("=" * 45)
    
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
    
    # Test HybridMoEBlock with different routing types
    print("\n1. Testing HybridMoEBlock with different routing types:")
    routing_types = ["moe", "dense", "sequential", "parallel"]
    
    for routing_type in routing_types:
        print(f"\n  Testing {routing_type} routing...")
        hybrid_block = HybridMoEBlock(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            routing_type=routing_type,
            moe_ratio=0.7
        )
        
        # Forward pass
        output = hybrid_block(x)
        print(f"    Output shape: {output.shape}")
        
        # Verify output shape matches input
        assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
        
        # Show that output is different from input (layer is doing something)
        diff = torch.mean(torch.abs(output - x))
        print(f"    Mean absolute difference from input: {diff.item():.6f}")
    
    # Test AdaptiveHybridMoE
    print("\n2. Testing AdaptiveHybridMoE:")
    adaptive_moe = AdaptiveHybridMoE(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        initial_moe_ratio=0.5
    )
    
    # Forward pass
    output = adaptive_moe(x)
    print(f"  Output shape: {output.shape}")
    
    # Verify output shape matches input
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
    
    # Show that output is different from input
    diff = torch.mean(torch.abs(output - x))
    print(f"  Mean absolute difference from input: {diff.item():.6f}")
    
    # Test manual ratio adjustment
    print("  Testing manual ratio adjustment...")
    adaptive_moe.set_moe_ratio(0.8)
    output2 = adaptive_moe(x)
    diff2 = torch.mean(torch.abs(output2 - x))
    print(f"  With higher MoE ratio, difference: {diff2.item():.6f}")
    
    # Test SparseMoEWithSharedExperts
    print("\n3. Testing SparseMoEWithSharedExperts:")
    sparse_moe = SparseMoEWithSharedExperts(
        d_model=d_model,
        num_experts=num_experts,
        num_shared_experts=2,
        top_k=top_k
    )
    
    # Forward pass
    output = sparse_moe(x)
    print(f"  Output shape: {output.shape}")
    
    # Verify output shape matches input
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
    
    # Show that output is different from input
    diff = torch.mean(torch.abs(output - x))
    print(f"  Mean absolute difference from input: {diff.item():.6f}")
    
    # Compare parameter counts
    print("\n4. Parameter comparison:")
    
    # Standard dense layer for comparison
    dense_layer = nn.Sequential(
        nn.Linear(d_model, 4 * d_model),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(4 * d_model, d_model),
        nn.Dropout(0.1)
    )
    
    # MoE layer for comparison
    from moe.moe_layer import MoELayer
    moe_layer = MoELayer(d_model, num_experts, top_k)
    
    # Hybrid layers
    hybrid_moe = HybridMoEBlock(d_model, num_experts, top_k, routing_type="moe")
    hybrid_dense = HybridMoEBlock(d_model, num_experts, top_k, routing_type="dense")
    hybrid_sequential = HybridMoEBlock(d_model, num_experts, top_k, routing_type="sequential")
    hybrid_parallel = HybridMoEBlock(d_model, num_experts, top_k, routing_type="parallel")
    adaptive_hybrid = AdaptiveHybridMoE(d_model, num_experts, top_k)
    sparse_hybrid = SparseMoEWithSharedExperts(d_model, num_experts, num_shared_experts=2, top_k=top_k)
    
    # Count parameters
    params = {
        "Standard Dense": sum(p.numel() for p in dense_layer.parameters()),
        "Pure MoE": sum(p.numel() for p in moe_layer.parameters()),
        "Hybrid MoE": sum(p.numel() for p in hybrid_moe.parameters()),
        "Hybrid Dense": sum(p.numel() for p in hybrid_dense.parameters()),
        "Hybrid Sequential": sum(p.numel() for p in hybrid_sequential.parameters()),
        "Hybrid Parallel": sum(p.numel() for p in hybrid_parallel.parameters()),
        "Adaptive Hybrid": sum(p.numel() for p in adaptive_hybrid.parameters()),
        "Sparse Hybrid": sum(p.numel() for p in sparse_hybrid.parameters())
    }
    
    # Print parameter counts
    for name, count in params.items():
        print(f"  {name:18}: {count:8,}")
    
    # Show computational efficiency
    print("\n5. Computational efficiency analysis:")
    print("  - Pure MoE: Sparse computation, good for large models")
    print("  - Hybrid Sequential: Combines MoE and dense benefits")
    print("  - Hybrid Parallel: Dynamic routing based on input")
    print("  - Adaptive Hybrid: Input-dependent routing")
    print("  - Sparse Hybrid: Always-active shared experts + sparse MoE")
    
    print("\nHybrid MoE example completed!")


if __name__ == "__main__":
    main()