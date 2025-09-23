"""
Implementation of attention mechanisms for Transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


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


class KVCache:
    """
    Key-Value cache for faster inference.
    
    This cache stores previously computed key and value tensors to avoid
    recomputing them for previously processed tokens during autoregressive generation.
    
    Args:
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        max_batch_size (int): Maximum batch size
        max_seq_len (int): Maximum sequence length
    """
    
    def __init__(self, num_heads: int, head_dim: int, max_batch_size: int = 1, max_seq_len: int = 512):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
        # Initialize cache tensors
        self.keys = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim)
        self.values = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim)
        self.current_seq_len = 0
        
        # Track which device the cache is on
        self.device = torch.device('cpu')
    
    def update(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new keys and values.
        
        Args:
            keys (torch.Tensor): New key tensors of shape (batch_size, num_heads, seq_len, head_dim)
            values (torch.Tensor): New value tensors of shape (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated keys and values including cached values
        """
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # Ensure we have enough space
        if self.current_seq_len + seq_len > self.max_seq_len:
            # If we exceed the maximum sequence length, keep only the most recent tokens
            keep_len = self.max_seq_len - seq_len
            self.keys = torch.cat([self.keys[:, :, -keep_len:, :], keys], dim=2)
            self.values = torch.cat([self.values[:, :, -keep_len:, :], values], dim=2)
            self.current_seq_len = self.max_seq_len
        else:
            # Append new keys and values
            self.keys[:, :, self.current_seq_len:self.current_seq_len+seq_len, :] = keys
            self.values[:, :, self.current_seq_len:self.current_seq_len+seq_len, :] = values
            self.current_seq_len += seq_len
        
        # Return the cached keys and values up to current sequence length
        return self.keys[:, :, :self.current_seq_len, :], self.values[:, :, :self.current_seq_len, :]
    
    def clear(self):
        """Clear the cache."""
        self.current_seq_len = 0
    
    def to(self, device: torch.device):
        """Move the cache to the specified device."""
        self.keys = self.keys.to(device)
        self.values = self.values.to(device)
        self.device = device
        return self


class RollingBufferKVCache:
    """
    Rolling buffer Key-Value cache for streaming inference.
    
    This cache maintains a fixed-size buffer that rolls over as new tokens are processed,
    making it suitable for streaming applications where memory is constrained.
    
    Args:
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        max_cache_size (int): Maximum size of the cache buffer
        max_batch_size (int): Maximum batch size
    """
    
    def __init__(self, num_heads: int, head_dim: int, max_cache_size: int = 1024, max_batch_size: int = 1):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_cache_size = max_cache_size
        self.max_batch_size = max_batch_size
        
        # Initialize cache tensors
        self.keys = torch.zeros(max_batch_size, num_heads, max_cache_size, head_dim)
        self.values = torch.zeros(max_batch_size, num_heads, max_cache_size, head_dim)
        self.current_cache_size = 0
        self.cache_start_idx = 0
        
        # Track which device the cache is on
        self.device = torch.device('cpu')
    
    def update(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new keys and values using rolling buffer strategy.
        
        Args:
            keys (torch.Tensor): New key tensors of shape (batch_size, num_heads, seq_len, head_dim)
            values (torch.Tensor): New value tensors of shape (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated keys and values including cached values
        """
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # If new sequence is longer than cache, only keep the most recent tokens
        if seq_len >= self.max_cache_size:
            start_idx = seq_len - self.max_cache_size
            self.keys[:, :, :self.max_cache_size, :] = keys[:, :, start_idx:, :]
            self.values[:, :, :self.max_cache_size, :] = values[:, :, start_idx:, :]
            self.current_cache_size = self.max_cache_size
            self.cache_start_idx = 0
            return self.keys, self.values
        
        # If adding new tokens would exceed cache size, shift the buffer
        if self.current_cache_size + seq_len > self.max_cache_size:
            # Calculate how many existing tokens to keep
            keep_size = self.max_cache_size - seq_len
            
            # Shift existing tokens to the beginning of the buffer
            if keep_size > 0:
                # Copy the most recent keep_size tokens to the beginning
                copy_start = (self.cache_start_idx + self.current_cache_size - keep_size) % self.max_cache_size
                if copy_start + keep_size <= self.max_cache_size:
                    # No wraparound needed
                    self.keys[:, :, :keep_size, :] = self.keys[:, :, copy_start:copy_start+keep_size, :]
                    self.values[:, :, :keep_size, :] = self.values[:, :, copy_start:copy_start+keep_size, :]
                else:
                    # Wraparound needed
                    first_part_size = self.max_cache_size - copy_start
                    self.keys[:, :, :first_part_size, :] = self.keys[:, :, copy_start:, :]
                    self.values[:, :, :first_part_size, :] = self.values[:, :, copy_start:, :]
                    second_part_size = keep_size - first_part_size
                    self.keys[:, :, first_part_size:keep_size, :] = self.keys[:, :, :second_part_size, :]
                    self.values[:, :, first_part_size:keep_size, :] = self.values[:, :, :second_part_size, :]
            
            # Add new tokens
            self.keys[:, :, keep_size:keep_size+seq_len, :] = keys
            self.values[:, :, keep_size:keep_size+seq_len, :] = values
            
            self.current_cache_size = self.max_cache_size
            self.cache_start_idx = 0
        else:
            # Add new tokens without shifting
            insert_idx = (self.cache_start_idx + self.current_cache_size) % self.max_cache_size
            if insert_idx + seq_len <= self.max_cache_size:
                # No wraparound needed
                self.keys[:, :, insert_idx:insert_idx+seq_len, :] = keys
                self.values[:, :, insert_idx:insert_idx+seq_len, :] = values
            else:
                # Wraparound needed
                first_part_size = self.max_cache_size - insert_idx
                self.keys[:, :, insert_idx:, :] = keys[:, :, :first_part_size, :]
                self.values[:, :, insert_idx:, :] = values[:, :, :first_part_size, :]
                second_part_size = seq_len - first_part_size
                self.keys[:, :, :second_part_size, :] = keys[:, :, first_part_size:, :]
                self.values[:, :, :second_part_size, :] = values[:, :, first_part_size:, :]
            
            self.current_cache_size += seq_len
        
        # Return the cached keys and values
        if self.cache_start_idx + self.current_cache_size <= self.max_cache_size:
            # No wraparound in cached data
            return (self.keys[:, :, self.cache_start_idx:self.cache_start_idx+self.current_cache_size, :],
                    self.values[:, :, self.cache_start_idx:self.cache_start_idx+self.current_cache_size, :])
        else:
            # Wraparound in cached data
            first_part_size = self.max_cache_size - self.cache_start_idx
            cached_keys = torch.cat([
                self.keys[:, :, self.cache_start_idx:, :],
                self.keys[:, :, :self.current_cache_size - first_part_size, :]
            ], dim=2)
            cached_values = torch.cat([
                self.values[:, :, self.cache_start_idx:, :],
                self.values[:, :, :self.current_cache_size - first_part_size, :]
            ], dim=2)
            return cached_keys, cached_values
    
    def clear(self):
        """Clear the cache."""
        self.current_cache_size = 0
        self.cache_start_idx = 0
    
    def to(self, device: torch.device):
        """Move the cache to the specified device."""
        self.keys = self.keys.to(device)
        self.values = self.values.to(device)
        self.device = device
        return self


def create_sliding_window_mask(seq_len: int, window_size: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a sliding window attention mask.
    
    Args:
        seq_len (int): Sequence length
        window_size (int): Size of the sliding window
        device (torch.device, optional): Device to create mask on
        
    Returns:
        torch.Tensor: Sliding window mask of shape (seq_len, seq_len)
    """
    mask = torch.ones(seq_len, seq_len, device=device)
    
    # Create sliding window mask
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        end = i + 1
        mask[i, :start] = 0
        mask[i, end:] = 0
    
    return mask


def create_sliding_window_mask_with_sink(seq_len: int, window_size: int, sink_size: int = 1, 
                                       device: torch.device = None) -> torch.Tensor:
    """
    Create a sliding window attention mask with attention sink.
    
    Args:
        seq_len (int): Sequence length
        window_size (int): Size of the sliding window
        sink_size (int): Number of initial tokens that all tokens can attend to (sink tokens)
        device (torch.device, optional): Device to create mask on
        
    Returns:
        torch.Tensor: Sliding window mask with sink of shape (seq_len, seq_len)
    """
    mask = torch.ones(seq_len, seq_len, device=device)
    
    # Create sliding window mask with sink
    for i in range(seq_len):
        # All tokens can attend to sink tokens
        # Sink tokens are the first sink_size tokens
        if i < sink_size:
            # Sink tokens can attend to all previous tokens
            mask[i, i+1:] = 0
        else:
            # Regular tokens can attend to sink tokens and window tokens
            mask[i, sink_size:i - window_size + 1] = 0
            mask[i, i+1:] = 0
    
    return mask


class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention implementation.
    
    This attention mechanism restricts each token to only attend to a local window
    of preceding tokens, which reduces memory and computation requirements.
    
    Args:
        d_k (int): Dimension of key vectors
        window_size (int): Size of the sliding window
        sink_size (int): Number of initial tokens that all tokens can attend to
    """
    
    def __init__(self, d_k: int, window_size: int = 128, sink_size: int = 1):
        super(SlidingWindowAttention, self).__init__()
        self.d_k = d_k
        self.window_size = window_size
        self.sink_size = sink_size
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, 
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sliding window attention.
        
        Args:
            queries (torch.Tensor): Query matrix of shape (batch_size, n_heads, seq_len, d_k)
            keys (torch.Tensor): Key matrix of shape (batch_size, n_heads, seq_len, d_k)
            values (torch.Tensor): Value matrix of shape (batch_size, n_heads, seq_len, d_v)
            mask (torch.Tensor, optional): Additional mask to apply
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output of attention mechanism and attention weights
        """
        batch_size, n_heads, seq_len, d_k = queries.shape
        
        # Create sliding window mask if not provided
        if mask is None:
            mask = create_sliding_window_mask_with_sink(
                seq_len, self.window_size, self.sink_size, device=queries.device
            )
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
            mask = mask.expand(batch_size, n_heads, -1, -1)
        
        # Compute attention scores: Q * K^T
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        # Scale by square root of d_k
        scores = scores / math.sqrt(self.d_k)
        
        # Apply mask
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


def test_sliding_window_attention():
    """
    Test function for SlidingWindowAttention.
    """
    batch_size = 2
    n_heads = 3
    seq_len = 8
    d_k = 5
    d_v = 6
    window_size = 4
    sink_size = 1
    
    # Create random tensors for testing
    queries = torch.randn(batch_size, n_heads, seq_len, d_k)
    keys = torch.randn(batch_size, n_heads, seq_len, d_k)
    values = torch.randn(batch_size, n_heads, seq_len, d_v)
    
    # Initialize sliding window attention
    sliding_attention = SlidingWindowAttention(d_k, window_size, sink_size)
    
    # Compute attention
    output, weights = sliding_attention(queries, keys, values)
    
    # Check output shapes
    assert output.shape == (batch_size, n_heads, seq_len, d_v), f"Output shape mismatch: {output.shape}"
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len), f"Weights shape mismatch: {weights.shape}"
    
    print("SlidingWindowAttention test passed!")


def test_rolling_buffer_kv_cache():
    """
    Test function for RollingBufferKVCache.
    """
    num_heads = 4
    head_dim = 8
    max_cache_size = 10
    max_batch_size = 2
    
    # Initialize rolling buffer KV cache
    rb_kv_cache = RollingBufferKVCache(num_heads, head_dim, max_cache_size, max_batch_size)
    
    # Create test keys and values
    batch_size = 1
    seq_len = 4
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Update cache
    cached_keys, cached_values = rb_kv_cache.update(keys, values)
    
    # Check shapes
    assert cached_keys.shape == (max_batch_size, num_heads, seq_len, head_dim)
    assert cached_values.shape == (max_batch_size, num_heads, seq_len, head_dim)
    
    # Add more keys and values that would exceed cache size
    seq_len2 = 8
    keys2 = torch.randn(batch_size, num_heads, seq_len2, head_dim)
    values2 = torch.randn(batch_size, num_heads, seq_len2, head_dim)
    
    # Update cache again
    cached_keys2, cached_values2 = rb_kv_cache.update(keys2, values2)
    
    # Check that we have at most max_cache_size tokens
    assert cached_keys2.shape[2] <= max_cache_size
    assert cached_values2.shape[2] <= max_cache_size
    
    print("RollingBufferKVCache test passed!")


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
    Multi-head attention implementation with KV cache support.
    
    This class implements multi-head attention by creating multiple attention heads
    in parallel and concatenating their outputs. It also supports KV caching for
    faster inference during autoregressive generation.
    
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
        
        # KV cache
        self.kv_cache: Optional[KVCache] = None
        self.rb_kv_cache: Optional[RollingBufferKVCache] = None
    
    def enable_kv_cache(self, max_batch_size: int = 1, max_seq_len: int = 512):
        """
        Enable KV caching for faster inference.
        
        Args:
            max_batch_size (int): Maximum batch size
            max_seq_len (int): Maximum sequence length
        """
        self.kv_cache = KVCache(self.num_heads, self.d_k, max_batch_size, max_seq_len)
    
    def enable_rolling_buffer_kv_cache(self, max_cache_size: int = 1024, max_batch_size: int = 1):
        """
        Enable rolling buffer KV caching for streaming inference.
        
        Args:
            max_cache_size (int): Maximum size of the cache buffer
            max_batch_size (int): Maximum batch size
        """
        self.rb_kv_cache = RollingBufferKVCache(self.num_heads, self.d_k, max_cache_size, max_batch_size)
    
    def disable_kv_cache(self):
        """Disable KV caching."""
        self.kv_cache = None
        self.rb_kv_cache = None
    
    def forward(self, x, mask=None):
        """
        Forward pass through multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Mask to apply to attention scores
            
        Returns:
            torch.Tensor: Output of multi-head attention of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply each attention head to input
        head_outputs = []
        
        for head in self.heads:
            # Get query, key, and value projections
            queries = head.W_q(x).view(batch_size, seq_len, 1, self.d_k)
            keys = head.W_k(x).view(batch_size, seq_len, 1, self.d_k)
            values = head.W_v(x).view(batch_size, seq_len, 1, self.d_v)
            
            # Reshape for attention mechanism
            queries = queries.transpose(1, 2)  # (batch_size, 1, seq_len, d_k)
            keys = keys.transpose(1, 2)        # (batch_size, 1, seq_len, d_k)
            values = values.transpose(1, 2)    # (batch_size, 1, seq_len, d_v)
            
            # Use KV cache if enabled
            if self.kv_cache is not None:
                # Update cache with new keys and values
                cached_keys, cached_values = self.kv_cache.update(keys, values)
                
                # Use cached keys and values for attention computation
                keys = cached_keys
                values = cached_values
            elif self.rb_kv_cache is not None:
                # Update rolling buffer cache with new keys and values
                cached_keys, cached_values = self.rb_kv_cache.update(keys, values)
                
                # Use cached keys and values for attention computation
                keys = cached_keys
                values = cached_values
            
            # Apply scaled dot-product attention
            attention_output, _ = head.attention(queries, keys, values, mask)
            
            # Reshape output back
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.view(batch_size, seq_len, self.d_v)
            
            head_outputs.append(attention_output)
        
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


def test_kv_cache():
    """
    Test function for KVCache.
    """
    num_heads = 4
    head_dim = 8
    max_batch_size = 2
    max_seq_len = 16
    
    # Initialize KV cache
    kv_cache = KVCache(num_heads, head_dim, max_batch_size, max_seq_len)
    
    # Create test keys and values
    batch_size = 1
    seq_len = 4
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Update cache
    cached_keys, cached_values = kv_cache.update(keys, values)
    
    # Check shapes
    assert cached_keys.shape == (max_batch_size, num_heads, seq_len, head_dim)
    assert cached_values.shape == (max_batch_size, num_heads, seq_len, head_dim)
    
    # Add more keys and values
    seq_len2 = 3
    keys2 = torch.randn(batch_size, num_heads, seq_len2, head_dim)
    values2 = torch.randn(batch_size, num_heads, seq_len2, head_dim)
    
    # Update cache again
    cached_keys2, cached_values2 = kv_cache.update(keys2, values2)
    
    # Check that we now have more tokens
    assert cached_keys2.shape == (max_batch_size, num_heads, seq_len + seq_len2, head_dim)
    assert cached_values2.shape == (max_batch_size, num_heads, seq_len + seq_len2, head_dim)
    
    print("KVCache test passed!")


if __name__ == "__main__":
    test_scaled_dot_product_attention()
    test_sliding_window_attention()
    test_rolling_buffer_kv_cache()
    test_attention_head()
    test_multi_head_attention()
    test_kv_cache()
