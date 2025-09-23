"""
Unit tests for semantic processing components.
"""

import torch
import torch.nn as nn
import sys
import os
import unittest

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.semantic.processing import (
    SemanticConfig, LargeConceptModel, ContextAwareSemanticProcessor,
    ThoughtUnit, SemanticDecoder, HyperbolicSpace, ByteLatentTokenizer,
    SemanticProcessor
)

class TestSemanticConfig(unittest.TestCase):
    """Test semantic configuration."""
    
    def test_semantic_config(self):
        """Test SemanticConfig data class."""
        config = SemanticConfig()
        
        # Check default values
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.concept_dim, 512)
        self.assertEqual(config.num_concepts, 1024)
        self.assertEqual(config.hyperbolic_dim, 128)
        self.assertEqual(config.thought_unit_size, 64)
        self.assertEqual(config.max_thought_units, 32)
        self.assertEqual(config.byte_latent_dim, 256)

class TestLargeConceptModel(unittest.TestCase):
    """Test Large Concept Model."""
    
    def test_large_concept_model(self):
        """Test LargeConceptModel implementation."""
        # Create configuration
        config = SemanticConfig(
            hidden_size=128,
            concept_dim=64,
            num_concepts=256
        )
        
        # Create LCM
        lcm = LargeConceptModel(config)
        
        # Create sample input
        batch_size, seq_length, hidden_size = 2, 10, 128
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Test forward pass
        concept_embeddings, concept_logits = lcm(hidden_states)
        
        # Check output shapes
        self.assertEqual(concept_embeddings.shape, (batch_size, seq_length, 64))
        self.assertEqual(concept_logits.shape, (batch_size, seq_length, 256))

class TestContextAwareSemanticProcessor(unittest.TestCase):
    """Test context-aware semantic processor."""
    
    def test_context_aware_processor(self):
        """Test ContextAwareSemanticProcessor implementation."""
        # Create configuration
        config = SemanticConfig(concept_dim=64)
        
        # Create processor
        processor = ContextAwareSemanticProcessor(config)
        
        # Create sample inputs
        batch_size, seq_length, concept_dim = 2, 10, 64
        concept_embeddings = torch.randn(batch_size, seq_length, concept_dim)
        context_embeddings = torch.randn(batch_size, 5, concept_dim)  # Different context length
        
        # Test forward pass
        output = processor(concept_embeddings, context_embeddings)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, concept_dim))

class TestThoughtUnit(unittest.TestCase):
    """Test thought unit components."""
    
    def test_thought_unit(self):
        """Test ThoughtUnit implementation."""
        # Create configuration
        config = SemanticConfig(concept_dim=64, thought_unit_size=32)
        
        # Create thought unit
        thought_unit = ThoughtUnit(config)
        
        # Create sample input
        batch_size, concept_dim = 2, 64
        concept_embedding = torch.randn(batch_size, concept_dim)
        
        # Test forward pass without previous thought
        thought = thought_unit(concept_embedding)
        
        # Check output shape
        self.assertEqual(thought.shape, (batch_size, 32))
        
        # Test forward pass with previous thought
        prev_thought = torch.randn(batch_size, 32)
        thought_with_prev = thought_unit(concept_embedding, prev_thought)
        
        # Check output shape
        self.assertEqual(thought_with_prev.shape, (batch_size, 32))

class TestSemanticDecoder(unittest.TestCase):
    """Test semantic decoder."""
    
    def test_semantic_decoder(self):
        """Test SemanticDecoder implementation."""
        # Create configuration
        config = SemanticConfig(concept_dim=64, thought_unit_size=32)
        vocab_size = 1000
        
        # Create decoder
        decoder = SemanticDecoder(config, vocab_size)
        
        # Create sample input
        batch_size, seq_length, concept_dim = 2, 10, 64
        concept_embeddings = torch.randn(batch_size, seq_length, concept_dim)
        
        # Test forward pass
        logits = decoder(concept_embeddings)
        
        # Check output shape
        self.assertEqual(logits.shape, (batch_size, seq_length, vocab_size))

class TestHyperbolicSpace(unittest.TestCase):
    """Test hyperbolic space components."""
    
    def test_hyperbolic_space(self):
        """Test HyperbolicSpace implementation."""
        # Create configuration
        config = SemanticConfig(concept_dim=64, hyperbolic_dim=32)
        
        # Create hyperbolic space mapper
        hyperbolic = HyperbolicSpace(config)
        
        # Create sample input
        batch_size, seq_length, concept_dim = 2, 10, 64
        concept_embeddings = torch.randn(batch_size, seq_length, concept_dim)
        
        # Test forward pass
        hyperbolic_embeddings = hyperbolic(concept_embeddings)
        
        # Check output shape
        self.assertEqual(hyperbolic_embeddings.shape, (batch_size, seq_length, 32))
        
    def test_hyperbolic_distance(self):
        """Test hyperbolic distance computation."""
        # Create configuration
        config = SemanticConfig(hyperbolic_dim=32)
        
        # Create hyperbolic space mapper
        hyperbolic = HyperbolicSpace(config)
        
        # Create sample points
        batch_size, dim = 2, 32
        x = torch.randn(batch_size, dim)
        y = torch.randn(batch_size, dim)
        
        # Normalize to Poincaré ball
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        
        # Test distance computation
        distance = hyperbolic.hyperbolic_distance(x, y)
        
        # Check output shape
        self.assertEqual(distance.shape, (batch_size,))
        
        # Check that distances are non-negative
        self.assertTrue(torch.all(distance >= 0))

class TestByteLatentTokenizer(unittest.TestCase):
    """Test byte-level latent tokenizer."""
    
    def test_byte_latent_tokenizer(self):
        """Test ByteLatentTokenizer implementation."""
        # Create configuration
        config = SemanticConfig(byte_latent_dim=64, hidden_size=128)
        
        # Create tokenizer
        tokenizer = ByteLatentTokenizer(config)
        
        # Create sample input (byte sequence)
        batch_size, seq_length = 2, 20
        text_bytes = torch.randint(0, 256, (batch_size, seq_length))
        
        # Test forward pass
        latent_representations = tokenizer(text_bytes)
        
        # Check output shape
        self.assertEqual(latent_representations.shape, (batch_size, seq_length, 128))
        
    def test_byte_latent_decoding(self):
        """Test byte-level decoding."""
        # Create configuration
        config = SemanticConfig(byte_latent_dim=64, hidden_size=128)
        
        # Create tokenizer
        tokenizer = ByteLatentTokenizer(config)
        
        # Create sample latent representations
        batch_size, seq_length, hidden_size = 2, 20, 128
        latent_representations = torch.randn(batch_size, seq_length, hidden_size)
        
        # Test decoding
        reconstructed_bytes = tokenizer.decode(latent_representations)
        
        # Check output shape
        self.assertEqual(reconstructed_bytes.shape, (batch_size, seq_length))
        
        # Check that output is valid byte values
        self.assertTrue(torch.all(reconstructed_bytes >= 0))
        self.assertTrue(torch.all(reconstructed_bytes < 256))

class TestSemanticProcessor(unittest.TestCase):
    """Test complete semantic processor."""
    
    def test_semantic_processor(self):
        """Test SemanticProcessor implementation."""
        # Create configuration
        config = SemanticConfig(
            hidden_size=128,
            concept_dim=64,
            num_concepts=256,
            hyperbolic_dim=32,
            thought_unit_size=32
        )
        vocab_size = 1000
        
        # Create semantic processor
        processor = SemanticProcessor(config, vocab_size)
        
        # Create sample input
        batch_size, seq_length, hidden_size = 2, 10, 128
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Test forward pass
        outputs = processor(hidden_states)
        
        # Check that outputs is a dictionary
        self.assertIsInstance(outputs, dict)
        
        # Check key outputs
        self.assertIn('concept_embeddings', outputs)
        self.assertIn('concept_logits', outputs)
        self.assertIn('context_aware_embeddings', outputs)
        self.assertIn('hyperbolic_embeddings', outputs)
        self.assertIn('token_logits', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['concept_embeddings'].shape, (batch_size, seq_length, 64))
        self.assertEqual(outputs['concept_logits'].shape, (batch_size, seq_length, 256))
        self.assertEqual(outputs['context_aware_embeddings'].shape, (batch_size, seq_length, 64))
        self.assertEqual(outputs['hyperbolic_embeddings'].shape, (batch_size, seq_length, 32))
        self.assertEqual(outputs['token_logits'].shape, (batch_size, seq_length, 1000))

if __name__ == '__main__':
    unittest.main()