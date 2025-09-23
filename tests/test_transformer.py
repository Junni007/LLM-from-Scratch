"""
Unit tests for core transformer components.
"""

import torch
import torch.nn as nn
import sys
import os
import unittest

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.attention import MultiHeadAttention
from src.models.mlp import FeedForwardNetwork as MLP
from src.models.normalization import LayerNorm
from src.models.positional import RotaryPositionalEmbedding as RotaryPositionalEncoding
from src.models.transformer import TransformerBlock

class TestAttention(unittest.TestCase):
    """Test attention mechanisms."""
    
    def test_multi_head_attention(self):
        """Test multi-head attention implementation."""
        # Create attention module
        attention = MultiHeadAttention(d_model=128, num_heads=8)
        
        # Create sample input
        batch_size, seq_length, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Test forward pass
        output = attention(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, d_model))

class TestMLP(unittest.TestCase):
    """Test MLP components."""
    
    def test_mlp(self):
        """Test MLP implementation."""
        # Create MLP module
        mlp = MLP(d_model=128)
        
        # Create sample input
        batch_size, seq_length, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Test forward pass
        output = mlp(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, d_model))

class TestNormalization(unittest.TestCase):
    """Test normalization components."""
    
    def test_layer_norm(self):
        """Test LayerNorm implementation."""
        # Create LayerNorm module
        layer_norm = LayerNorm(d_model=128)
        
        # Create sample input
        batch_size, seq_length, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Test forward pass
        output = layer_norm(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, d_model))
        
        # Check that output is normalized (mean ≈ 0, std ≈ 1)
        # LayerNorm normalizes each position independently across the feature dimension
        mean = output.mean(dim=-1)  # Mean across feature dimension for each position
        std = output.std(dim=-1)    # Std across feature dimension for each position
        
        # Allow for reasonable numerical tolerance
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-3))
        self.assertTrue(torch.allclose(std, torch.ones_like(std), atol=1e-1))

class TestPositional(unittest.TestCase):
    """Test positional encoding components."""
    
    def test_rotary_positional_encoding(self):
        """Test RotaryPositionalEncoding implementation."""
        # Create rotary positional encoding module
        rope = RotaryPositionalEncoding(d_model=128)
        
        # Create sample input
        batch_size, seq_length, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Test forward pass
        output = rope(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, d_model))

class TestTransformerBlock(unittest.TestCase):
    """Test transformer block components."""
    
    def test_transformer_block(self):
        """Test TransformerBlock implementation."""
        # Create transformer block module
        transformer_block = TransformerBlock(d_model=128, num_heads=8, dropout=0.1)
        
        # Create sample input
        batch_size, seq_length, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Test forward pass
        output = transformer_block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, d_model))

if __name__ == '__main__':
    unittest.main()