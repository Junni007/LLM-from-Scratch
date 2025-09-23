"""
Unit tests for MoE (Mixture of Experts) components.
"""

import torch
import torch.nn as nn
import sys
import os
import unittest

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.moe.gating import TopKGating, ExpertGating
from src.moe.expert import Expert, SparseExpert
from src.moe.moe_layer import MoELayer
from src.moe.load_balancing import LoadBalancingLoss

class TestGating(unittest.TestCase):
    """Test gating mechanisms."""
    
    def test_top_k_gating(self):
        """Test TopKGating implementation."""
        # Create gating network
        gating = TopKGating(input_size=128, num_experts=4, k=2)
        
        # Create sample input
        batch_size, seq_length, input_size = 2, 10, 128
        x = torch.randn(batch_size, seq_length, input_size)
        
        # Test forward pass
        gate_logits, gate_weights, expert_indices = gating(x)
        
        # Check output shapes
        self.assertEqual(gate_logits.shape, (batch_size, seq_length, 4))
        self.assertEqual(gate_weights.shape, (batch_size, seq_length, 2))
        self.assertEqual(expert_indices.shape, (batch_size, seq_length, 2))
        
        # Check that expert indices are valid
        self.assertTrue(torch.all(expert_indices >= 0))
        self.assertTrue(torch.all(expert_indices < 4))
        
        # Check that gate weights sum to 1 (approximately)
        weights_sum = gate_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6))
        
    def test_expert_gating(self):
        """Test ExpertGating implementation."""
        # Create gating network
        gating = ExpertGating(input_size=128, num_experts=4)
        
        # Create sample input
        batch_size, seq_length, input_size = 2, 10, 128
        x = torch.randn(batch_size, seq_length, input_size)
        
        # Test forward pass
        router_logits = gating(x)
        
        # Check output shape
        self.assertEqual(router_logits.shape, (batch_size, seq_length, 4))

class TestExpert(unittest.TestCase):
    """Test expert components."""
    
    def test_expert(self):
        """Test Expert implementation."""
        # Create expert
        expert = Expert(input_size=128, hidden_size=256, output_size=128)
        
        # Create sample input
        batch_size, seq_length, input_size = 2, 10, 128
        x = torch.randn(batch_size, seq_length, input_size)
        
        # Test forward pass
        output = expert(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, 128))
        
    def test_sparse_expert(self):
        """Test SparseExpert implementation."""
        # Create sparse expert
        expert = SparseExpert(input_size=128, hidden_size=256, output_size=128)
        
        # Create sample input
        batch_size, seq_length, input_size = 2, 10, 128
        x = torch.randn(batch_size, seq_length, input_size)
        
        # Test forward pass
        output = expert(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, 128))

class TestMoELayer(unittest.TestCase):
    """Test MoE layer components."""
    
    def test_moe_layer(self):
        """Test MoELayer implementation."""
        # Create MoE layer
        moe_layer = MoELayer(
            input_size=128,
            hidden_size=256,
            output_size=128,
            num_experts=4,
            k=2
        )
        
        # Create sample input
        batch_size, seq_length, input_size = 2, 10, 128
        x = torch.randn(batch_size, seq_length, input_size)
        
        # Test forward pass
        output, aux_loss = moe_layer(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, 128))
        
        # Check that auxiliary loss is a scalar
        self.assertIsInstance(aux_loss, torch.Tensor)
        self.assertEqual(aux_loss.dim(), 0)

class TestLoadBalancing(unittest.TestCase):
    """Test load balancing components."""
    
    def test_load_balancing_loss(self):
        """Test LoadBalancingLoss implementation."""
        # Create load balancing loss
        lb_loss = LoadBalancingLoss(num_experts=4)
        
        # Create sample gate logits and expert indices
        batch_size, seq_length, num_experts = 2, 10, 4
        gate_logits = torch.randn(batch_size, seq_length, num_experts)
        expert_indices = torch.randint(0, num_experts, (batch_size, seq_length, 2))
        
        # Test loss computation
        loss = lb_loss(gate_logits, expert_indices)
        
        # Check that loss is a scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)

if __name__ == '__main__':
    unittest.main()