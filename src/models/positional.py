"""
Implementation of positional embeddings for Transformer architecture.
"""

import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation.
    
    This implementation follows the RoPE paper:
    https://arxiv.org/abs/2104.09864
    
    RoPE encodes positional information by rotating the input vectors in a
    way that's compatible with the attention mechanism.
    
    Args:
        d_model (int): Dimension of input embeddings
        max_seq_len (int): Maximum sequence length
    """
    
    def __init__(self, d_model, max_seq_len=512):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create rotation matrices
        cos, sin = self._get_cos_sin(max_seq_len, d_model)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
    
    def _get_cos_sin(self, seq_len, d_model):
        """
        Generate cosine and sine values for rotary embeddings.
        
        Args:
            seq_len (int): Sequence length
            d_model (int): Model dimension
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and sine tensors
        """
        # Create position indices
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        
        # Create inverse frequencies
        # For RoPE, we need d_model//2 dimensions since we rotate pairs
        dim = torch.arange(0, d_model // 2, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (dim / (d_model // 2)))
        
        # Calculate angles
        angles = position * inv_freq.unsqueeze(0)
        
        # Repeat angles to match d_model dimensions
        # Each angle needs to be repeated twice for the pair
        angles = angles.repeat(1, 2)
        
        # Return cosine and sine
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        return cos, sin
    
    def forward(self, x, offset=0):
        """
        Apply rotary positional embeddings to input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            offset (int): Position offset for rotary embeddings
            
        Returns:
            torch.Tensor: Output tensor with rotary embeddings applied
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get cos and sin for current sequence
        cos = self.cos[offset:offset+seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        sin = self.sin[offset:offset+seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply rotation
        x_rot = self._rotate_half(x)
        
        # Apply rotary embeddings
        x_rope = (x * cos) + (x_rot * sin)
        
        return x_rope
    
    def _rotate_half(self, x):
        """
        Rotate half of the dimensions for rotary embeddings.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Rotated tensor
        """
        # Split into two halves
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        
        # Rotate by 90 degrees: (x1, x2) -> (-x2, x1)
        x_rot = torch.cat([-x2, x1], dim=-1)
        
        return x_rot


def test_rotary_positional_embedding():
    """
    Test function for RotaryPositionalEmbedding.
    """
    batch_size = 2
    seq_len = 8
    d_model = 16
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize RoPE
    rope = RotaryPositionalEmbedding(d_model, max_seq_len=seq_len)
    
    # Apply rotary embeddings
    x_rope = rope(x)
    
    # Check output shape
    assert x_rope.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {x_rope.shape}"
    
    # Check that applying RoPE twice doesn't change the result significantly
    # (RoPE should be deterministic)
    x_rope2 = rope(x)
    assert torch.allclose(x_rope, x_rope2), "RoPE is not deterministic"
    
    print("RotaryPositionalEmbedding test passed!")


if __name__ == "__main__":
    test_rotary_positional_embedding()