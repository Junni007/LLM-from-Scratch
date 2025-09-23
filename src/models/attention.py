"""
Implementation of attention mechanisms for Transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    This is the core attention mechanism used in Transformers.
    Given query, key, and value matrices, it computes:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Args:
        d_k (int): Dimension of key vectors
    """
    
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    
    def forward(self, queries, keys, values, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            queries (torch.Tensor): Query matrix of shape (batch_size, n_heads, seq_len, d_k)
            keys (torch.Tensor): Key matrix of shape (batch_size, n_heads, seq_len, d_k)
            values (torch.Tensor): Value matrix of shape (batch_size, n_heads, seq_len, d_v)
            mask (torch.Tensor, optional): Mask to apply to attention scores
            
        Returns:
            torch.Tensor: Output of attention mechanism
        """
        # Compute attention scores: Q * K^T
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        # Scale by square root of d_k
        scores = scores / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Set masked positions to large negative value so they become near zero after softmax
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, values)
        
        return output, attention_weights


def test_scaled_dot_product_attention():
    """
    Test function for ScaledDotProductAttention.
    """
    batch_size = 2
    n_heads = 3
    seq_len = 4
    d_k = 5
    d_v = 6
    
    # Create random tensors for testing
    queries = torch.randn(batch_size, n_heads, seq_len, d_k)
    keys = torch.randn(batch_size, n_heads, seq_len, d_k)
    values = torch.randn(batch_size, n_heads, seq_len, d_v)
    
    # Initialize attention mechanism
    attention = ScaledDotProductAttention(d_k)
    
    # Compute attention
    output, weights = attention(queries, keys, values)
    
    # Check output shapes
    assert output.shape == (batch_size, n_heads, seq_len, d_v), f"Output shape mismatch: {output.shape}"
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len), f"Weights shape mismatch: {weights.shape}"
    
    print("ScaledDotProductAttention test passed!")


class AttentionHead(nn.Module):
    """
    Single attention head implementation.
    
    This class implements a single attention head that projects input embeddings
    into query, key, and value vectors, then applies scaled dot-product attention.
    
    Args:
        d_model (int): Dimension of input embeddings
        d_k (int): Dimension of key vectors
        d_v (int): Dimension of value vectors
    """
    
    def __init__(self, d_model, d_k, d_v):
        super(AttentionHead, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        
        # Scaled dot-product attention mechanism
        self.attention = ScaledDotProductAttention(d_k)
    
    def forward(self, x, mask=None):
        """
        Forward pass through single attention head.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Mask to apply to attention scores
            
        Returns:
            torch.Tensor: Output of attention head of shape (batch_size, seq_len, d_v)
        """
        # Apply linear projections to get Q, K, V
        queries = self.W_q(x)  # (batch_size, seq_len, d_k)
        keys = self.W_k(x)     # (batch_size, seq_len, d_k)
        values = self.W_v(x)   # (batch_size, seq_len, d_v)
        
        # Reshape for attention mechanism
        # Add dimension for single head: (batch_size, 1, seq_len, d_k/d_v)
        queries = queries.unsqueeze(1)
        keys = keys.unsqueeze(1)
        values = values.unsqueeze(1)
        
        # Apply scaled dot-product attention
        output, attention_weights = self.attention(queries, keys, values, mask)
        
        # Remove head dimension and return output
        output = output.squeeze(1)  # (batch_size, seq_len, d_v)
        
        return output


def test_attention_head():
    """
    Test function for AttentionHead.
    """
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_k = 5
    d_v = 6
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize attention head
    attention_head = AttentionHead(d_model, d_k, d_v)
    
    # Compute attention
    output = attention_head(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_v), f"Output shape mismatch: {output.shape}"
    
    print("AttentionHead test passed!")


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention implementation.
    
    This class implements multi-head attention by creating multiple attention heads
    in parallel and concatenating their outputs.
    
    Args:
        d_model (int): Dimension of input embeddings
        num_heads (int): Number of attention heads
    """
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        
        # Create multiple attention heads
        self.heads = nn.ModuleList([
            AttentionHead(d_model, self.d_k, self.d_v) for _ in range(num_heads)
        ])
        
        # Linear projection for concatenated outputs
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        """
        Forward pass through multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Mask to apply to attention scores
            
        Returns:
            torch.Tensor: Output of multi-head attention of shape (batch_size, seq_len, d_model)
        """
        # Apply each attention head to input
        head_outputs = []
        for head in self.heads:
            head_output = head(x, mask)
            head_outputs.append(head_output)
        
        # Concatenate outputs from all heads
        concatenated = torch.cat(head_outputs, dim=-1)  # (batch_size, seq_len, d_model)
        
        # Apply final linear projection
        output = self.W_o(concatenated)
        
        return output


def test_multi_head_attention():
    """
    Test function for MultiHeadAttention.
    """
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 4
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize multi-head attention
    multi_head_attention = MultiHeadAttention(d_model, num_heads)
    
    # Compute attention
    output = multi_head_attention(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {output.shape}"
    
    print("MultiHeadAttention test passed!")


if __name__ == "__main__":
    test_scaled_dot_product_attention()
    test_attention_head()
    test_multi_head_attention()
