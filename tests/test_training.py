"""
Unit tests for training components.
"""

import torch
import torch.nn as nn
import sys
import os
import unittest

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train.data import ByteLevelTokenizer, create_dataloader
from src.train.loss import cross_entropy_loss
from src.train.sampling import top_k_sampling, top_p_sampling
from src.train.evaluation import calculate_perplexity

class TestTokenizer(unittest.TestCase):
    """Test tokenizer components."""
    
    def test_byte_level_tokenizer(self):
        """Test ByteLevelTokenizer implementation."""
        # Create tokenizer
        tokenizer = ByteLevelTokenizer(vocab_size=256)
        
        # Test encoding
        text = "Hello, world!"
        encoded = tokenizer.encode(text)
        
        # Check that encoded is a list of integers
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(token, int) for token in encoded))
        
        # Test decoding
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)
        
        # Test tensor conversion
        tensor_encoded = tokenizer.encode_to_tensor(text)
        self.assertIsInstance(tensor_encoded, torch.Tensor)
        self.assertEqual(tensor_encoded.dim(), 1)

class TestDataLoading(unittest.TestCase):
    """Test data loading components."""
    
    def test_create_dataloader(self):
        """Test create_dataloader function."""
        # Create sample data
        texts = ["Hello world", "This is a test", "Another example"]
        
        # Create dataloader
        dataloader = create_dataloader(texts, batch_size=2, seq_length=10)
        
        # Check that dataloader is created
        self.assertIsNotNone(dataloader)
        
        # Check that we can iterate over it
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            # Check batch shape
            self.assertEqual(len(batch), 2)  # input_ids and labels
            self.assertEqual(batch[0].dim(), 2)  # (batch_size, seq_length)
            self.assertEqual(batch[1].dim(), 2)  # (batch_size, seq_length)
            break  # Just test first batch
            
        self.assertGreater(batch_count, 0)

class TestLoss(unittest.TestCase):
    """Test loss components."""
    
    def test_cross_entropy_loss(self):
        """Test cross_entropy_loss function."""
        # Create sample logits and labels
        batch_size, seq_length, vocab_size = 2, 5, 100
        logits = torch.randn(batch_size, seq_length, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        # Test loss computation
        loss = cross_entropy_loss(logits, labels)
        
        # Check that loss is a scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)

class TestSampling(unittest.TestCase):
    """Test sampling components."""
    
    def test_top_k_sampling(self):
        """Test top_k_sampling function."""
        # Create sample logits
        batch_size, vocab_size = 2, 100
        logits = torch.randn(batch_size, vocab_size)
        
        # Test top-k sampling
        k = 10
        samples = top_k_sampling(logits, k=k)
        
        # Check output shape
        self.assertEqual(samples.shape, (batch_size,))
        self.assertTrue(torch.all(samples >= 0))
        self.assertTrue(torch.all(samples < vocab_size))
        
    def test_top_p_sampling(self):
        """Test top_p_sampling function."""
        # Create sample logits
        batch_size, vocab_size = 2, 100
        logits = torch.randn(batch_size, vocab_size)
        
        # Test top-p sampling
        p = 0.9
        samples = top_p_sampling(logits, p=p)
        
        # Check output shape
        self.assertEqual(samples.shape, (batch_size,))
        self.assertTrue(torch.all(samples >= 0))
        self.assertTrue(torch.all(samples < vocab_size))

class TestEvaluation(unittest.TestCase):
    """Test evaluation components."""
    
    def test_calculate_perplexity(self):
        """Test calculate_perplexity function."""
        # Create sample logits and labels
        batch_size, seq_length, vocab_size = 2, 5, 100
        logits = torch.randn(batch_size, seq_length, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        # Test perplexity calculation
        perplexity = calculate_perplexity(logits, labels)
        
        # Check that perplexity is a positive scalar
        self.assertIsInstance(perplexity, float)
        self.assertGreater(perplexity, 0)

if __name__ == '__main__':
    unittest.main()