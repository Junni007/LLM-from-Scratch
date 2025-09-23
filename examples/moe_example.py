"""
Example demonstrating Mixture of Experts (MoE) theory components.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from moe.moe_theory import MoETheoryComponents


def main():
    """Main function demonstrating MoE theory components."""
    print("Mixture of Experts (MoE) Theory Components Example")
    print("=" * 50)
    
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
    
    # Create MoE theory components
    print("\nCreating MoE theory components...")
    moe_components = MoETheoryComponents(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        gating_type="softmax",
        balance_loss_weight=0.01
    )
    
    # Process input through MoE components
    print("\nProcessing input through MoE components...")
    router_logits, top_k_indices, top_k_gates, balance_loss = moe_components(x)
    
    # Display results
    print(f"\nResults:")
    print(f"  Router logits shape: {router_logits.shape}")
    print(f"  Top-k indices shape: {top_k_indices.shape}")
    print(f"  Top-k gates shape: {top_k_gates.shape}")
    print(f"  Balance loss: {balance_loss.item():.6f}")
    
    # Show example routing for first token
    print(f"\nExample routing for first token (batch 0, position 0):")
    print(f"  Router logits: {router_logits[0, 0].detach().numpy()}")
    print(f"  Top-{top_k} expert indices: {top_k_indices[0, 0].detach().numpy()}")
    print(f"  Top-{top_k} gate values: {top_k_gates[0, 0].detach().numpy()}")
    
    # Show routing distribution
    print(f"\nRouting distribution across all tokens:")
    # Count how many times each expert is selected as top-1
    top_1_experts = top_k_indices[:, :, 0].flatten()
    expert_counts = torch.bincount(top_1_experts, minlength=num_experts)
    print(f"  Top-1 expert selection counts: {expert_counts.detach().numpy()}")
    
    # Show load balancing effect
    print(f"\nLoad balancing analysis:")
    total_tokens = batch_size * seq_len
    expert_fractions = expert_counts.float() / total_tokens
    print(f"  Expert usage fractions: {expert_fractions.detach().numpy()}")
    print(f"  Ideal uniform fraction: {1.0/num_experts:.3f}")
    
    # Test different gating types
    print(f"\nTesting different gating types:")
    for gating_type in ["softmax", "noisy_top_k"]:
        print(f"  {gating_type}:")
        gating_network = moe_components.gating_network
        gating_network.gating_type = gating_type
        gate_logits, gate_values = gating_network(x)
        print(f"    Gate values sum (should be 1.0): {torch.sum(gate_values[0, 0]).item():.6f}")
    
    print("\nMoE theory components example completed!")


if __name__ == "__main__":
    main()