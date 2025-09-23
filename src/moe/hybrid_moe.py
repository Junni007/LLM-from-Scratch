"""
Hybrid Mixture of Experts (MoE) implementation.

This module implements hybrid architectures that combine MoE layers with dense layers,
allowing for more flexible and efficient model designs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
import sys
import os

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from .moe_layer import MoELayer, Expert
from models.mlp import FeedForwardNetwork


class HybridMoEBlock(nn.Module):
    """
    Hybrid MoE block that combines MoE layers with dense layers.
    
    This block allows for flexible mixing of MoE and dense computation within
    a single transformer block.
    """
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2,
                 d_ff: int = None, dropout: float = 0.1, 
                 moe_ratio: float = 0.5, routing_type: str = "moe"):
        """
        Initialize HybridMoEBlock.
        
        Args:
            d_model (int): Model dimension
            num_experts (int): Number of experts for MoE layers
            top_k (int): Number of top experts to route to per token
            d_ff (int, optional): Feed-forward dimension (default: 4 * d_model)
            dropout (float): Dropout probability
            moe_ratio (float): Ratio of MoE computation vs dense computation (0.0 = all dense, 1.0 = all MoE)
            routing_type (str): Type of routing ("moe", "dense", "parallel", "sequential")
        """
        super(HybridMoEBlock, self).__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ff = d_ff
        self.moe_ratio = moe_ratio
        self.routing_type = routing_type
        
        # Initialize components based on routing type
        if routing_type in ["moe", "sequential"]:
            # MoE layer
            self.moe_layer = MoELayer(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k,
                d_ff=d_ff,
                dropout=dropout
            )
        
        if routing_type in ["dense", "sequential"]:
            # Dense layer
            self.dense_layer = FeedForwardNetwork(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout
            )
        
        # For parallel routing, we need both layers and a router
        if routing_type == "parallel":
            # MoE layer
            self.moe_layer = MoELayer(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k,
                d_ff=d_ff,
                dropout=dropout
            )
            
            # Dense layer
            self.dense_layer = FeedForwardNetwork(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout
            )
            
            # Router to decide which path to take
            self.router = nn.Linear(d_model, 2)  # 2 options: MoE or Dense
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid MoE block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        if self.routing_type == "moe":
            # Pure MoE routing
            return self.moe_layer(x)
        elif self.routing_type == "dense":
            # Pure dense routing
            return self.dense_layer(x)
        elif self.routing_type == "sequential":
            # Sequential: MoE followed by dense (or vice versa based on ratio)
            if self.moe_ratio >= 0.5:
                # MoE first, then dense
                x = self.moe_layer(x)
                x = self.dense_layer(x)
            else:
                # Dense first, then MoE
                x = self.dense_layer(x)
                x = self.moe_layer(x)
            return x
        elif self.routing_type == "parallel":
            # Parallel: Route tokens to MoE or Dense based on router decision
            batch_size, seq_len, d_model = x.shape
            
            # Compute routing logits
            router_logits = self.router(x)  # (batch_size, seq_len, 2)
            routing_probs = F.softmax(router_logits, dim=-1)
            
            # Split tokens based on routing probabilities
            moe_weight = routing_probs[:, :, 0:1]  # (batch_size, seq_len, 1)
            dense_weight = routing_probs[:, :, 1:2]  # (batch_size, seq_len, 1)
            
            # Process through both paths
            moe_output = self.moe_layer(x)
            dense_output = self.dense_layer(x)
            
            # Combine outputs based on routing weights
            output = moe_weight * moe_output + dense_weight * dense_output
            return output
        else:
            raise ValueError(f"Unknown routing type: {self.routing_type}")


class AdaptiveHybridMoE(nn.Module):
    """
    Adaptive Hybrid MoE that dynamically adjusts the ratio of MoE to dense computation
    based on input characteristics or training progress.
    """
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2,
                 d_ff: int = None, dropout: float = 0.1, 
                 initial_moe_ratio: float = 0.5):
        """
        Initialize AdaptiveHybridMoE.
        
        Args:
            d_model (int): Model dimension
            num_experts (int): Number of experts for MoE layers
            top_k (int): Number of top experts to route to per token
            d_ff (int, optional): Feed-forward dimension (default: 4 * d_model)
            dropout (float): Dropout probability
            initial_moe_ratio (float): Initial ratio of MoE computation
        """
        super(AdaptiveHybridMoE, self).__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ff = d_ff
        self.moe_ratio = initial_moe_ratio
        
        # Initialize components
        self.moe_layer = MoELayer(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.dense_layer = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Adaptive controller (simple version - could be more sophisticated)
        self.adaptation_network = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def set_moe_ratio(self, ratio: float):
        """
        Set the MoE ratio manually.
        
        Args:
            ratio (float): MoE ratio (0.0 to 1.0)
        """
        self.moe_ratio = max(0.0, min(1.0, ratio))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adaptive hybrid MoE.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute adaptive weights
        # Simple approach: use average of input features to determine routing
        x_avg = torch.mean(x, dim=1)  # (batch_size, d_model)
        adaptation_signal = self.adaptation_network(x_avg)  # (batch_size, 1)
        
        # Expand to match sequence length
        adaptation_weights = adaptation_signal.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, 1)
        
        # Compute dynamic MoE ratio
        dynamic_moe_ratio = self.moe_ratio * adaptation_weights  # (batch_size, seq_len, 1)
        dynamic_dense_ratio = 1.0 - dynamic_moe_ratio  # (batch_size, seq_len, 1)
        
        # Process through both paths
        moe_output = self.moe_layer(x)
        dense_output = self.dense_layer(x)
        
        # Combine outputs based on dynamic weights
        output = dynamic_moe_ratio * moe_output + dynamic_dense_ratio * dense_output
        return output


class SparseMoEWithSharedExperts(nn.Module):
    """
    Sparse MoE with shared experts that are always activated.
    
    This combines the efficiency of sparse MoE with the reliability of shared experts
    that process all tokens.
    """
    
    def __init__(self, d_model: int, num_experts: int, num_shared_experts: int = 1,
                 top_k: int = 2, d_ff: int = None, dropout: float = 0.1):
        """
        Initialize SparseMoEWithSharedExperts.
        
        Args:
            d_model (int): Model dimension
            num_experts (int): Number of routing experts
            num_shared_experts (int): Number of shared experts that always process all tokens
            top_k (int): Number of top experts to route to per token
            d_ff (int, optional): Feed-forward dimension (default: 4 * d_model)
            dropout (float): Dropout probability
        """
        super(SparseMoEWithSharedExperts, self).__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.d_ff = d_ff
        
        # Initialize routing MoE layer
        self.routing_moe = MoELayer(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Initialize shared experts
        self.shared_experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_shared_experts)
        ])
        
        # Optional: residual connection scaling
        self.shared_expert_scaling = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through sparse MoE with shared experts.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Process through routing MoE
        moe_output = self.routing_moe(x)
        
        # Process through shared experts
        shared_output = x
        for expert in self.shared_experts:
            shared_output = shared_output + expert(shared_output)
        
        # Scale shared expert output
        shared_output = shared_output * self.shared_expert_scaling
        
        # Combine outputs (residual connection)
        output = x + moe_output + shared_output
        
        return output


def test_hybrid_moe():
    """Test function for hybrid MoE implementations."""
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_experts = 4
    top_k = 2
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test HybridMoEBlock with different routing types
    routing_types = ["moe", "dense", "sequential", "parallel"]
    for routing_type in routing_types:
        print(f"Testing HybridMoEBlock with {routing_type} routing...")
        hybrid_block = HybridMoEBlock(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            routing_type=routing_type
        )
        output = hybrid_block(x)
        assert output.shape == (batch_size, seq_len, d_model)
        print(f"  Output shape: {output.shape}")
    
    # Test AdaptiveHybridMoE
    print("\nTesting AdaptiveHybridMoE...")
    adaptive_moe = AdaptiveHybridMoE(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k
    )
    output = adaptive_moe(x)
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"  Output shape: {output.shape}")
    
    # Test SparseMoEWithSharedExperts
    print("\nTesting SparseMoEWithSharedExperts...")
    sparse_moe = SparseMoEWithSharedExperts(
        d_model=d_model,
        num_experts=num_experts,
        num_shared_experts=2,
        top_k=top_k
    )
    output = sparse_moe(x)
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"  Output shape: {output.shape}")
    
    print("Hybrid MoE test passed!")


if __name__ == "__main__":
    test_hybrid_moe()