"""
Implementation of Transformer blocks.
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .mlp import FeedForwardNetwork, SwiGLU
from .normalization import LayerNorm, RMSNorm
from .positional import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    """
    Transformer block implementation.
    
    This implements a single Transformer block with:
    1. Multi-head attention with residual connection and LayerNorm
    2. Feed-forward network with residual connection and LayerNorm
    
    Args:
        d_model (int): Dimension of input embeddings
        num_heads (int): Number of attention heads
        d_ff (int): Dimension of feed-forward hidden layer
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # LayerNorm layers
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Feed-forward network
        self.ff_network = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass through Transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Mask to apply to attention scores
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First sub-layer: Multi-head attention with residual connection
        attention_output = self.attention(x, mask)
        attention_output = self.dropout1(attention_output)
        x = x + attention_output  # Residual connection
        x = self.norm1(x)  # LayerNorm
        
        # Second sub-layer: Feed-forward network with residual connection
        ff_output = self.ff_network(x)
        ff_output = self.dropout2(ff_output)
        x = x + ff_output  # Residual connection
        x = self.norm2(x)  # LayerNorm
        
        return x


class ModernTransformerBlock(nn.Module):
    """
    Modern Transformer block implementation with RMSNorm, RoPE, and SwiGLU.
    
    This implements a modern Transformer block with:
    1. Multi-head attention with RoPE positional embeddings
    2. RMSNorm instead of LayerNorm
    3. SwiGLU activation in feed-forward network
    4. Residual connections
    
    Args:
        d_model (int): Dimension of input embeddings
        num_heads (int): Number of attention heads
        d_ff (int): Dimension of feed-forward hidden layer
        dropout (float): Dropout probability
        max_seq_len (int): Maximum sequence length for positional embeddings
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, max_seq_len=512):
        super(ModernTransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Multi-head attention with RoPE
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.rope = RotaryPositionalEmbedding(d_model, max_seq_len)
        
        # RMSNorm layers
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # SwiGLU feed-forward network
        self.ff_network = SwiGLU(d_model, d_ff, dropout)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass through modern Transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Mask to apply to attention scores
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Apply RoPE to input
        x_rope = self.rope(x)
        
        # First sub-layer: Multi-head attention with residual connection
        attention_output = self.attention(x_rope, mask)
        attention_output = self.dropout1(attention_output)
        x = x + attention_output  # Residual connection
        x = self.norm1(x)  # RMSNorm
        
        # Second sub-layer: SwiGLU feed-forward network with residual connection
        ff_output = self.ff_network(x)
        ff_output = self.dropout2(ff_output)
        x = x + ff_output  # Residual connection
        x = self.norm2(x)  # RMSNorm
        
        return x


def test_modern_transformer_block():
    """
    Test function for ModernTransformerBlock.
    """
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 4
    d_ff = 16
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize modern transformer block
    modern_transformer_block = ModernTransformerBlock(d_model, num_heads, d_ff)
    
    # Compute forward pass
    output = modern_transformer_block(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {output.shape}"
    
    print("ModernTransformerBlock test passed!")


def test_transformer_block():
    """
    Test function for TransformerBlock.
    """
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 4
    d_ff = 16
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize transformer block
    transformer_block = TransformerBlock(d_model, num_heads, d_ff)
    
    # Compute forward pass
    output = transformer_block(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {output.shape}"
    
    print("TransformerBlock test passed!")


if __name__ == "__main__":
    test_transformer_block()
    test_modern_transformer_block()
