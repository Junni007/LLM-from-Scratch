"""
Mixture of Experts (MoE) layer implementation.

This module implements a complete MoE layer that can be used as a drop-in replacement
for standard feed-forward layers in Transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
from .moe_theory import ExpertRouter, GatingNetwork, LoadBalancer


class Expert(nn.Module):
    """
    Individual expert network for MoE.
    
    This is a simple feed-forward network that serves as one expert in the MoE system.
    """
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        """
        Initialize Expert.
        
        Args:
            d_model (int): Model dimension
            d_ff (int, optional): Feed-forward dimension (default: 4 * d_model)
            dropout (float): Dropout probability
        """
        super(Expert, self).__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu  # Using GELU activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (..., d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    
    This layer implements a complete MoE layer that can replace standard feed-forward
    layers in Transformer architectures.
    """
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2,
                 d_ff: int = None, dropout: float = 0.1, 
                 gating_type: str = "softmax", balance_loss_weight: float = 0.01):
        """
        Initialize MoELayer.
        
        Args:
            d_model (int): Model dimension
            num_experts (int): Number of experts
            top_k (int): Number of top experts to route to per token
            d_ff (int, optional): Feed-forward dimension for experts (default: 4 * d_model)
            dropout (float): Dropout probability
            gating_type (str): Type of gating function
            balance_loss_weight (float): Weight for load balancing loss
        """
        super(MoELayer, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_weight = balance_loss_weight
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        
        # Initialize routing components
        self.expert_router = ExpertRouter(d_model, num_experts, top_k)
        self.gating_network = GatingNetwork(d_model, num_experts, gating_type)
        self.load_balancer = LoadBalancer(num_experts, balance_loss_weight)
        
        # For tracking balance loss
        self.balance_loss = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Route tokens to experts
        router_logits, top_k_indices, top_k_gates = self.expert_router(x)
        
        # Compute gating values
        gate_logits, gate_values = self.gating_network(x)
        
        # Reshape inputs for expert processing
        # (batch_size, seq_len, d_model) -> (batch_size * seq_len, d_model)
        x_flat = x.view(-1, d_model)
        
        # Prepare indices for expert selection
        # (batch_size, seq_len, top_k) -> (batch_size * seq_len, top_k)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)
        top_k_gates_flat = top_k_gates.view(-1, self.top_k)
        
        # Initialize output tensor
        output = torch.zeros_like(x_flat)
        
        # Process tokens through selected experts
        for i in range(self.top_k):
            # Get expert indices and gates for this level
            expert_indices = top_k_indices_flat[:, i]  # (batch_size * seq_len,)
            gates = top_k_gates_flat[:, i]  # (batch_size * seq_len,)
            
            # Group tokens by expert
            for expert_idx in range(self.num_experts):
                # Find tokens assigned to this expert
                token_mask = (expert_indices == expert_idx)
                if token_mask.any():
                    # Get tokens for this expert
                    expert_tokens = x_flat[token_mask]  # (num_tokens_for_expert, d_model)
                    
                    # Process through expert
                    expert_output = self.experts[expert_idx](expert_tokens)
                    
                    # Apply gating and add to output
                    gated_output = expert_output * gates[token_mask].unsqueeze(-1)
                    output[token_mask] += gated_output
        
        # Reshape output back to original shape
        output = output.view(batch_size, seq_len, d_model)
        
        # Compute and store balance loss
        balance_loss = self.load_balancer.compute_load_balance_loss(router_logits, top_k_indices)
        importance_loss = self.load_balancer.compute_importance_loss(gate_values)
        self.balance_loss = balance_loss + importance_loss
        
        return output


class SwitchTransformerLayer(nn.Module):
    """
    Switch Transformer layer implementation.
    
    This implements the Switch Transformer variant where each token is routed to
    exactly one expert (top-1 routing).
    """
    
    def __init__(self, d_model: int, num_experts: int, d_ff: int = None, 
                 dropout: float = 0.1, balance_loss_weight: float = 0.01):
        """
        Initialize SwitchTransformerLayer.
        
        Args:
            d_model (int): Model dimension
            num_experts (int): Number of experts
            d_ff (int, optional): Feed-forward dimension for experts (default: 4 * d_model)
            dropout (float): Dropout probability
            balance_loss_weight (float): Weight for load balancing loss
        """
        super(SwitchTransformerLayer, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.balance_loss_weight = balance_loss_weight
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        
        # Initialize router (top-1 routing)
        self.router = nn.Linear(d_model, num_experts)
        
        # Initialize load balancer
        self.load_balancer = LoadBalancer(num_experts, balance_loss_weight)
        
        # For tracking balance loss
        self.balance_loss = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Switch Transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute router logits
        router_logits = self.router(x)  # (batch_size, seq_len, num_experts)
        
        # Get top-1 expert for each token
        expert_indices = torch.argmax(router_logits, dim=-1)  # (batch_size, seq_len)
        
        # Compute softmax probabilities for gating
        router_probs = F.softmax(router_logits, dim=-1)
        expert_gates = torch.gather(router_probs, -1, expert_indices.unsqueeze(-1)).squeeze(-1)
        
        # Reshape inputs for expert processing
        # (batch_size, seq_len, d_model) -> (batch_size * seq_len, d_model)
        x_flat = x.view(-1, d_model)
        expert_indices_flat = expert_indices.view(-1)  # (batch_size * seq_len,)
        expert_gates_flat = expert_gates.view(-1)  # (batch_size * seq_len,)
        
        # Initialize output tensor
        output = torch.zeros_like(x_flat)
        
        # Process tokens through selected experts
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            token_mask = (expert_indices_flat == expert_idx)
            if token_mask.any():
                # Get tokens for this expert
                expert_tokens = x_flat[token_mask]  # (num_tokens_for_expert, d_model)
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_tokens)
                
                # Apply gating and add to output
                gated_output = expert_output * expert_gates_flat[token_mask].unsqueeze(-1)
                output[token_mask] += gated_output
        
        # Reshape output back to original shape
        output = output.view(batch_size, seq_len, d_model)
        
        # Compute and store balance loss
        # For Switch Transformer, we only need load balancing loss
        # Create compatible shapes for load balancer
        router_logits_compatible = router_logits.unsqueeze(-1)  # (batch_size, seq_len, num_experts, 1)
        expert_indices_compatible = expert_indices.unsqueeze(-1)  # (batch_size, seq_len, 1)
        balance_loss = self.load_balancer.compute_load_balance_loss(
            router_logits_compatible, expert_indices_compatible
        )
        self.balance_loss = balance_loss
        
        return output


def test_moe_layer():
    """Test function for MoE layer."""
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_experts = 4
    top_k = 2
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test MoELayer
    moe_layer = MoELayer(d_model, num_experts, top_k)
    output = moe_layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert moe_layer.balance_loss is not None
    print(f"MoELayer balance loss: {moe_layer.balance_loss.item():.6f}")
    
    # Test SwitchTransformerLayer
    switch_layer = SwitchTransformerLayer(d_model, num_experts)
    output = switch_layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert switch_layer.balance_loss is not None
    print(f"SwitchTransformerLayer balance loss: {switch_layer.balance_loss.item():.6f}")
    
    print("MoE layer test passed!")


if __name__ == "__main__":
    test_moe_layer()