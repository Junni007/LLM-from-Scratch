"""
Mixture of Experts (MoE) theory components implementation.

This module implements the core theoretical components of Mixture of Experts:
1. Expert routing mechanisms
2. Gating networks
3. Load balancing techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class ExpertRouter(nn.Module):
    """
    Expert routing mechanism for MoE.
    
    This module routes input tokens to different experts based on a routing function.
    """
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        """
        Initialize ExpertRouter.
        
        Args:
            d_model (int): Model dimension
            num_experts (int): Number of experts
            top_k (int): Number of top experts to route to per token
        """
        super(ExpertRouter, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Routing network (simple linear layer for now)
        self.router = nn.Linear(d_model, num_experts)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - router_logits: Raw router outputs of shape (batch_size, seq_len, num_experts)
                - top_k_indices: Indices of top-k experts of shape (batch_size, seq_len, top_k)
                - top_k_gates: Gating values for top-k experts of shape (batch_size, seq_len, top_k)
        """
        # Compute router logits
        router_logits = self.router(x)  # (batch_size, seq_len, num_experts)
        
        # Get top-k experts for each token
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Compute gating values using softmax
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        return router_logits, top_k_indices, top_k_gates


class GatingNetwork(nn.Module):
    """
    Gating network for MoE.
    
    This module computes gating values that determine how much each expert contributes
    to the final output for each token.
    """
    
    def __init__(self, d_model: int, num_experts: int, gating_type: str = "softmax"):
        """
        Initialize GatingNetwork.
        
        Args:
            d_model (int): Model dimension
            num_experts (int): Number of experts
            gating_type (str): Type of gating function ("softmax", "noisy_top_k")
        """
        super(GatingNetwork, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.gating_type = gating_type
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts)
        
        # For noisy top-k gating
        self.noise_epsilon = 1e-2
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating values.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - gate_logits: Raw gate outputs of shape (batch_size, seq_len, num_experts)
                - gate_values: Normalized gate values of shape (batch_size, seq_len, num_experts)
        """
        # Compute gate logits
        gate_logits = self.gate(x)  # (batch_size, seq_len, num_experts)
        
        if self.gating_type == "softmax":
            # Standard softmax gating
            gate_values = F.softmax(gate_logits, dim=-1)
        elif self.gating_type == "noisy_top_k":
            # Noisy top-k gating
            noise = torch.randn_like(gate_logits) * self.noise_epsilon
            gate_logits = gate_logits + noise
            gate_values = F.softmax(gate_logits, dim=-1)
        else:
            raise ValueError(f"Unknown gating type: {self.gating_type}")
        
        return gate_logits, gate_values


class LoadBalancer(nn.Module):
    """
    Load balancing mechanism for MoE.
    
    This module helps distribute tokens evenly across experts to avoid overloading
    some experts while underutilizing others.
    """
    
    def __init__(self, num_experts: int, balance_loss_weight: float = 0.01):
        """
        Initialize LoadBalancer.
        
        Args:
            num_experts (int): Number of experts
            balance_loss_weight (float): Weight for load balancing loss
        """
        super(LoadBalancer, self).__init__()
        self.num_experts = num_experts
        self.balance_loss_weight = balance_loss_weight
    
    def compute_load_balance_loss(self, gate_logits: torch.Tensor, 
                                 top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss.
        
        Args:
            gate_logits (torch.Tensor): Gate logits of shape (batch_size, seq_len, num_experts) or 
                                       (batch_size, seq_len, num_experts, top_k)
            top_k_indices (torch.Tensor): Top-k expert indices of shape (batch_size, seq_len, top_k) or
                                         (batch_size, seq_len, 1) for top-1
            
        Returns:
            torch.Tensor: Load balancing loss
        """
        # Handle different input shapes
        if gate_logits.dim() == 4:  # (batch_size, seq_len, num_experts, top_k)
            batch_size, seq_len, _, top_k = gate_logits.shape
        else:  # (batch_size, seq_len, num_experts)
            batch_size, seq_len, _ = gate_logits.shape
            top_k = top_k_indices.shape[-1] if top_k_indices.dim() > 2 else 1
        
        total_tokens = batch_size * seq_len
        
        # Count tokens routed to each expert
        expert_counts = torch.zeros(self.num_experts, device=gate_logits.device)
        
        # Handle different top_k_indices shapes
        if top_k_indices.dim() == 4:  # (batch_size, seq_len, 1, top_k)
            indices_to_use = top_k_indices.squeeze(-2)  # (batch_size, seq_len, top_k)
        elif top_k_indices.dim() == 3:  # (batch_size, seq_len, top_k) or (batch_size, seq_len, 1)
            indices_to_use = top_k_indices
        else:  # (batch_size, seq_len) - should not happen but handle just in case
            indices_to_use = top_k_indices.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # If indices_to_use has top_k dimension > 1, we need to handle multiple experts per token
        if indices_to_use.shape[-1] > 1:
            for k in range(indices_to_use.shape[-1]):
                # Flatten indices and count
                flat_indices = indices_to_use[:, :, k].flatten()
                expert_counts += torch.bincount(flat_indices, minlength=self.num_experts)
        else:
            # Flatten indices and count
            flat_indices = indices_to_use.squeeze(-1).flatten()
            expert_counts += torch.bincount(flat_indices, minlength=self.num_experts)
        
        # Normalize to get fraction of tokens per expert
        expert_frac = expert_counts / total_tokens
        
        # Compute load balancing loss (auxiliary loss)
        # This encourages uniform distribution of tokens across experts
        aux_loss = self.num_experts * torch.sum(expert_frac ** 2) - 1.0
        
        return self.balance_loss_weight * aux_loss
    
    def compute_importance_loss(self, gate_values: torch.Tensor) -> torch.Tensor:
        """
        Compute importance loss to encourage uniform importance across experts.
        
        Args:
            gate_values (torch.Tensor): Gate values of shape (batch_size, seq_len, num_experts)
            
        Returns:
            torch.Tensor: Importance loss
        """
        # Compute importance per expert (sum of gate values for each expert)
        importance_per_expert = torch.sum(gate_values, dim=(0, 1))  # (num_experts,)
        
        # Normalize by total importance
        total_importance = torch.sum(importance_per_expert)
        if total_importance > 0:
            importance_frac = importance_per_expert / total_importance
        else:
            importance_frac = torch.zeros_like(importance_per_expert)
        
        # Compute importance loss (encourages uniform distribution)
        uniform_dist = torch.ones_like(importance_frac) / self.num_experts
        importance_loss = torch.sum((importance_frac - uniform_dist) ** 2)
        
        return self.balance_loss_weight * importance_loss


class MoETheoryComponents(nn.Module):
    """
    Complete MoE theory components implementation.
    
    This module combines expert routing, gating networks, and load balancing
    into a single cohesive unit.
    """
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, 
                 gating_type: str = "softmax", balance_loss_weight: float = 0.01):
        """
        Initialize MoETheoryComponents.
        
        Args:
            d_model (int): Model dimension
            num_experts (int): Number of experts
            top_k (int): Number of top experts to route to per token
            gating_type (str): Type of gating function
            balance_loss_weight (float): Weight for load balancing loss
        """
        super(MoETheoryComponents, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Initialize components
        self.expert_router = ExpertRouter(d_model, num_experts, top_k)
        self.gating_network = GatingNetwork(d_model, num_experts, gating_type)
        self.load_balancer = LoadBalancer(num_experts, balance_loss_weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process input through MoE theory components.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - router_logits: Raw router outputs of shape (batch_size, seq_len, num_experts)
                - top_k_indices: Indices of top-k experts of shape (batch_size, seq_len, top_k)
                - top_k_gates: Gating values for top-k experts of shape (batch_size, seq_len, top_k)
                - balance_loss: Load balancing loss
        """
        # Route tokens to experts
        router_logits, top_k_indices, top_k_gates = self.expert_router(x)
        
        # Compute gating values
        gate_logits, gate_values = self.gating_network(x)
        
        # Compute load balancing loss
        balance_loss = self.load_balancer.compute_load_balance_loss(router_logits, top_k_indices)
        importance_loss = self.load_balancer.compute_importance_loss(gate_values)
        
        # Total balance loss
        total_balance_loss = balance_loss + importance_loss
        
        return router_logits, top_k_indices, top_k_gates, total_balance_loss


def test_moe_theory_components():
    """Test function for MoE theory components."""
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_experts = 4
    top_k = 2
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test ExpertRouter
    router = ExpertRouter(d_model, num_experts, top_k)
    router_logits, top_k_indices, top_k_gates = router(x)
    
    assert router_logits.shape == (batch_size, seq_len, num_experts)
    assert top_k_indices.shape == (batch_size, seq_len, top_k)
    assert top_k_gates.shape == (batch_size, seq_len, top_k)
    
    # Test GatingNetwork
    gating_network = GatingNetwork(d_model, num_experts)
    gate_logits, gate_values = gating_network(x)
    
    assert gate_logits.shape == (batch_size, seq_len, num_experts)
    assert gate_values.shape == (batch_size, seq_len, num_experts)
    # Check that gate values sum to 1
    assert torch.allclose(torch.sum(gate_values, dim=-1), torch.ones(batch_size, seq_len))
    
    # Test LoadBalancer
    load_balancer = LoadBalancer(num_experts)
    balance_loss = load_balancer.compute_load_balance_loss(router_logits, top_k_indices)
    importance_loss = load_balancer.compute_importance_loss(gate_values)
    
    assert isinstance(balance_loss, torch.Tensor)
    assert isinstance(importance_loss, torch.Tensor)
    
    # Test complete MoETheoryComponents
    moe_components = MoETheoryComponents(d_model, num_experts, top_k)
    router_logits, top_k_indices, top_k_gates, total_balance_loss = moe_components(x)
    
    assert router_logits.shape == (batch_size, seq_len, num_experts)
    assert top_k_indices.shape == (batch_size, seq_len, top_k)
    assert top_k_gates.shape == (batch_size, seq_len, top_k)
    assert isinstance(total_balance_loss, torch.Tensor)
    
    print("MoE theory components test passed!")


if __name__ == "__main__":
    test_moe_theory_components()