"""
Unit tests for efficiency optimization components.
"""

import torch
import torch.nn as nn
import sys
import os
import unittest

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.quantization import QuantizationConfig, Quantizer, QuantizedLinear, ModelQuantizer
from src.utils.compression import PruningConfig, Pruner, LowRankApproximation, KnowledgeDistillation
from src.train.optimization import TrainingOptimizationConfig, GradientCompressor, OptimizedTrainer

class TestQuantization(unittest.TestCase):
    """Test quantization components."""
    
    def test_quantization_config(self):
        """Test QuantizationConfig data class."""
        config = QuantizationConfig(bits=8, method="linear", symmetric=True, per_channel=False)
        
        # Check values
        self.assertEqual(config.bits, 8)
        self.assertEqual(config.method, "linear")
        self.assertTrue(config.symmetric)
        self.assertFalse(config.per_channel)
        
    def test_quantizer(self):
        """Test Quantizer implementation."""
        # Create quantizer
        config = QuantizationConfig(bits=8)
        quantizer = Quantizer(config)
        
        # Create sample tensor
        tensor = torch.randn(10, 10)
        
        # Test symmetric quantization
        quantized, scale, zero_point = quantizer.quantize(tensor)
        
        # Check output types and shapes
        self.assertIsInstance(quantized, torch.Tensor)
        self.assertIsInstance(scale, torch.Tensor)
        self.assertIsInstance(zero_point, torch.Tensor)
        self.assertEqual(quantized.dtype, torch.int8)
        
        # Test dequantization
        dequantized = quantizer.dequantize(quantized, scale, zero_point)
        
        # Check output shape
        self.assertEqual(dequantized.shape, tensor.shape)
        
        # Check that dequantized values are close to original (with some error due to quantization)
        error = torch.abs(dequantized - tensor).mean()
        self.assertLess(error, 0.1)  # Quantization error should be relatively small
        
    def test_quantized_linear(self):
        """Test QuantizedLinear implementation."""
        # Create quantized linear layer
        linear = QuantizedLinear(in_features=128, out_features=64)
        
        # Create sample input
        batch_size, in_features = 2, 128
        x = torch.randn(batch_size, in_features)
        
        # Test forward pass before quantization
        output_before = linear(x)
        self.assertEqual(output_before.shape, (batch_size, 64))
        
        # Quantize parameters
        linear.quantize_parameters()
        self.assertTrue(linear.quantized)
        
        # Test forward pass after quantization
        output_after = linear(x)
        self.assertEqual(output_after.shape, (batch_size, 64))

class TestCompression(unittest.TestCase):
    """Test compression components."""
    
    def test_pruning_config(self):
        """Test PruningConfig data class."""
        config = PruningConfig(method="magnitude", sparsity_ratio=0.5, structured=False, layer_wise=True)
        
        # Check values
        self.assertEqual(config.method, "magnitude")
        self.assertEqual(config.sparsity_ratio, 0.5)
        self.assertFalse(config.structured)
        self.assertTrue(config.layer_wise)
        
    def test_pruner(self):
        """Test Pruner implementation."""
        # Create pruner
        config = PruningConfig(sparsity_ratio=0.3)
        pruner = Pruner(config)
        
        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(128, 64)
                self.linear2 = nn.Linear(64, 32)
                
            def forward(self, x):
                return self.linear2(self.linear1(x))
        
        model = SimpleModel()
        
        # Test magnitude pruning
        pruned_model = pruner.prune_model(model)
        
        # Check that pruned model has the same structure
        self.assertIsInstance(pruned_model, SimpleModel)
        
        # Check that some weights are zero (pruned)
        linear1_weights = pruned_model.linear1.weight.data
        linear2_weights = pruned_model.linear2.weight.data
        
        # Calculate sparsity
        linear1_sparsity = (linear1_weights == 0).float().mean()
        linear2_sparsity = (linear2_weights == 0).float().mean()
        
        # Sparsity should be close to target (allowing some tolerance)
        self.assertGreater(linear1_sparsity, 0.2)  # At least 20% sparsity
        self.assertGreater(linear2_sparsity, 0.2)  # At least 20% sparsity
        
    def test_low_rank_approximation(self):
        """Test LowRankApproximation implementation."""
        # Create sample weight matrix
        weight = torch.randn(64, 128)
        
        # Test SVD compression
        rank = 32
        US, VT = LowRankApproximation.svd_compress(weight, rank)
        
        # Check output shapes
        self.assertEqual(US.shape, (64, rank))
        self.assertEqual(VT.shape, (rank, 128))
        
        # Check that reconstructed matrix is close to original
        reconstructed = torch.matmul(US, VT)
        error = torch.abs(reconstructed - weight).mean()
        self.assertLess(error, 0.1)  # Reconstruction error should be relatively small

class TestOptimization(unittest.TestCase):
    """Test optimization components."""
    
    def test_training_optimization_config(self):
        """Test TrainingOptimizationConfig data class."""
        config = TrainingOptimizationConfig(
            gradient_accumulation_steps=4,
            mixed_precision=True,
            gradient_clipping=1.0,
            learning_rate=1e-4
        )
        
        # Check values
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertTrue(config.mixed_precision)
        self.assertEqual(config.gradient_clipping, 1.0)
        self.assertEqual(config.learning_rate, 1e-4)
        
    def test_gradient_compressor(self):
        """Test GradientCompressor implementation."""
        # Create gradient compressor
        compressor = GradientCompressor(compression_ratio=0.5)
        
        # Create sample gradients
        gradients = [torch.randn(10, 10), torch.randn(5, 5), torch.randn(20)]
        
        # Test compression
        compressed_gradients, masks = compressor.compress_gradients(gradients)
        
        # Check output structure
        self.assertEqual(len(compressed_gradients), len(gradients))
        self.assertEqual(len(masks), len(gradients))
        
        # Check that compressed gradients have the same shapes
        for i, (orig, comp) in enumerate(zip(gradients, compressed_gradients)):
            if orig is not None and comp is not None:
                self.assertEqual(orig.shape, comp.shape)
                
    def test_optimized_trainer(self):
        """Test OptimizedTrainer implementation."""
        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)
                
            def forward(self, input_ids, labels=None):
                logits = self.linear(input_ids)
                loss = None
                if labels is not None:
                    loss = nn.functional.mse_loss(logits, labels.float())
                class Output:
                    def __init__(self, logits, loss):
                        self.logits = logits
                        self.loss = loss
                return Output(logits, loss)
        
        model = SimpleModel()
        
        # Create trainer config
        config = TrainingOptimizationConfig(
            gradient_accumulation_steps=2,
            mixed_precision=False,  # Disable for CPU testing
            gradient_clipping=1.0,
            learning_rate=1e-3
        )
        
        # Create trainer
        trainer = OptimizedTrainer(model, config)
        
        # Create sample data
        batch_size, seq_length, hidden_size = 2, 10, 128
        input_ids = torch.randn(batch_size, seq_length, hidden_size)
        labels = torch.randn(batch_size, seq_length, hidden_size)
        
        # Test training step
        loss = trainer.train_step(input_ids, labels)
        
        # Check that loss is a float value
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

if __name__ == '__main__':
    unittest.main()