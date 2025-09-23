"""
Implementation of normalization layers for Transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    """
    Layer Normalization implementation.
    
    This implementation follows the original LayerNorm paper:
    Layer Normalization: https://arxiv.org/abs/1607.06450
    
    Args:
        d_model (int): Dimension of input embeddings
        eps (float): Small value to prevent division by zero
    """
    
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        """
        Forward pass through LayerNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Normalized output tensor of shape (batch_size, seq_len, d_model)
        """
        # Calculate mean and variance along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize
        normalized = (x - mean) / (std + self.eps)
        
        # Scale and shift
        output = self.gamma * normalized + self.beta
        
        return output


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization implementation.
    
    This implementation follows the RMSNorm paper:
    https://arxiv.org/abs/1910.07467
    
    RMSNorm is a simpler normalization technique that often performs as well as
    LayerNorm but with reduced computational overhead.
    
    Args:
        d_model (int): Dimension of input embeddings
        eps (float): Small value to prevent division by zero
    """
    
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        """
        Forward pass through RMSNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Normalized output tensor of shape (batch_size, seq_len, d_model)
        """
        # Calculate RMS (Root Mean Square) along the last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        normalized = x / rms
        output = self.gamma * normalized
        
        return output


def test_layer_norm():
    """
    Test function for LayerNorm.
    """
    batch_size = 2
    seq_len = 4
    d_model = 8
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize LayerNorm
    layer_norm = LayerNorm(d_model)
    
    # Compute forward pass
    output = layer_norm(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {output.shape}"
    
    # Check that output has mean ~0 and std ~1 along the last dimension
    mean = output.mean(dim=-1)
    std = output.std(dim=-1)
    
    # Allow for some numerical tolerance
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), "Mean is not close to 0"
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5), "Std is not close to 1"
    
    print("LayerNorm test passed!")


def test_rms_norm():
    """
    Test function for RMSNorm.
    """
    batch_size = 2
    seq_len = 4
    d_model = 8
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize RMSNorm
    rms_norm = RMSNorm(d_model)
    
    # Compute forward pass
    output = rms_norm(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {output.shape}"
    
    # Check that output has RMS ~1 along the last dimension (before scaling)
    normalized = output / rms_norm.gamma  # Remove scaling
    rms = torch.sqrt(torch.mean(normalized ** 2, dim=-1))
    
    # Allow for some numerical tolerance
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5), "RMS is not close to 1"
    
    print("RMSNorm test passed!")


if __name__ == "__main__":
    test_layer_norm()
    test_rms_norm()
