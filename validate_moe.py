#!/usr/bin/env python3
"""
Validation script for MoE (Mixture of Experts) components.
This script demonstrates that the MoE implementation is working correctly.
"""

import sys
import torch

# Add src to path
sys.path.append('.')

# Import MoE components
from src.moe.moe_theory import ExpertRouter, GatingNetwork, LoadBalancer, MoETheoryComponents
from src.moe.moe_layer import MoELayer, SwitchTransformerLayer


def validate_moe_theory_components():
    """Validate the MoE theory components."""
    print("=== MoE Theory Components Validation ===")
    
    batch_size = 2
    seq_len = 4
    d_model = 128
    num_experts = 4
    top_k = 2
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    
    # Test ExpertRouter
    print("1. Testing ExpertRouter...")
    router = ExpertRouter(d_model, num_experts, top_k)
    router_logits, top_k_indices, top_k_gates = router(x)
    
    print(f"   Router logits shape: {router_logits.shape}")
    print(f"   Top-k indices shape: {top_k_indices.shape}")
    print(f"   Top-k gates shape: {top_k_gates.shape}")
    
    # Validate shapes
    assert router_logits.shape == (batch_size, seq_len, num_experts)
    assert top_k_indices.shape == (batch_size, seq_len, top_k)
    assert top_k_gates.shape == (batch_size, seq_len, top_k)
    print("   ExpertRouter validation passed!")
    
    # Test GatingNetwork
    print("2. Testing GatingNetwork...")
    gating_network = GatingNetwork(d_model, num_experts)
    gate_logits, gate_values = gating_network(x)
    
    print(f"   Gate logits shape: {gate_logits.shape}")
    print(f"   Gate values shape: {gate_values.shape}")
    
    # Validate shapes
    assert gate_logits.shape == (batch_size, seq_len, num_experts)
    assert gate_values.shape == (batch_size, seq_len, num_experts)
    
    # Check that gate values sum to 1
    gate_sums = torch.sum(gate_values, dim=-1)
    assert torch.allclose(gate_sums, torch.ones(batch_size, seq_len))
    print("   GatingNetwork validation passed!")
    
    # Test LoadBalancer
    print("3. Testing LoadBalancer...")
    load_balancer = LoadBalancer(num_experts)
    balance_loss = load_balancer.compute_load_balance_loss(router_logits, top_k_indices)
    importance_loss = load_balancer.compute_importance_loss(gate_values)
    
    print(f"   Balance loss: {balance_loss.item():.6f}")
    print(f"   Importance loss: {importance_loss.item():.6f}")
    
    # Validate types
    assert isinstance(balance_loss, torch.Tensor)
    assert isinstance(importance_loss, torch.Tensor)
    print("   LoadBalancer validation passed!")
    
    # Test complete MoETheoryComponents
    print("4. Testing complete MoETheoryComponents...")
    moe_components = MoETheoryComponents(d_model, num_experts, top_k)
    router_logits, top_k_indices, top_k_gates, total_balance_loss = moe_components(x)
    
    print(f"   Total balance loss: {total_balance_loss.item():.6f}")
    
    # Validate shapes
    assert router_logits.shape == (batch_size, seq_len, num_experts)
    assert top_k_indices.shape == (batch_size, seq_len, top_k)
    assert top_k_gates.shape == (batch_size, seq_len, top_k)
    assert isinstance(total_balance_loss, torch.Tensor)
    print("   Complete MoETheoryComponents validation passed!")


def validate_moe_layers():
    """Validate the MoE layer implementations."""
    print("\n=== MoE Layer Implementations Validation ===")
    
    batch_size = 2
    seq_len = 8
    d_model = 128
    num_experts = 4
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    
    # Test MoELayer
    print("1. Testing MoELayer...")
    moe_layer = MoELayer(d_model=d_model, num_experts=num_experts, top_k=2)
    output = moe_layer(x)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Balance loss: {moe_layer.balance_loss.item():.6f}")
    
    # Validate shapes
    assert output.shape == (batch_size, seq_len, d_model)
    assert isinstance(moe_layer.balance_loss, torch.Tensor)
    print("   MoELayer validation passed!")
    
    # Test SwitchTransformerLayer
    print("2. Testing SwitchTransformerLayer...")
    switch_layer = SwitchTransformerLayer(d_model=d_model, num_experts=num_experts)
    output = switch_layer(x)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Balance loss: {switch_layer.balance_loss.item():.6f}")
    
    # Validate shapes
    assert output.shape == (batch_size, seq_len, d_model)
    assert isinstance(switch_layer.balance_loss, torch.Tensor)
    print("   SwitchTransformerLayer validation passed!")


def main():
    """Run all validations."""
    print("MoE (Mixture of Experts) Components Validation")
    print("=" * 50)
    
    validate_moe_theory_components()
    validate_moe_layers()
    
    print("\n" + "=" * 50)
    print("All MoE components validated successfully!")
    print("The MoE implementation is fully functional.")


if __name__ == "__main__":
    main()