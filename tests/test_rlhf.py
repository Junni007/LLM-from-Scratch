"""
Unit tests for RLHF components.
"""

import torch
import torch.nn as nn
import sys
import os
import unittest

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rlhf.ppo import PolicyValueNetwork, PPOTrainer, PPOConfig
from src.rlhf.grpo import GRPOTrainer, GRPOConfig
from src.reward.model import RewardModel
from src.reward.loss import BradleyTerryLoss, MarginRankingLoss
from src.reward.preference_data import PreferenceExample

class TestRewardModel(unittest.TestCase):
    """Test reward model components."""
    
    def test_reward_model(self):
        """Test RewardModel implementation."""
        # Create a simple base model for testing
        class SimpleBaseModel(nn.Module):
            def __init__(self, hidden_size=128):
                super().__init__()
                self.hidden_size = hidden_size
                self.embedding = nn.Embedding(100, hidden_size)
                self.linear = nn.Linear(hidden_size, hidden_size)
                
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                x = self.linear(x)
                # Mock outputs with required attributes
                class MockOutput:
                    def __init__(self, last_hidden_state, logits):
                        self.last_hidden_state = last_hidden_state
                        self.logits = logits
                return MockOutput(x, x)
        
        base_model = SimpleBaseModel(hidden_size=128)
        reward_model = RewardModel(base_model, hidden_size=128)
        
        # Create sample input
        batch_size, seq_length = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        
        # Test forward pass
        output = reward_model(input_ids)
        
        # Check output shape - should be scalar reward per batch item
        self.assertEqual(output.shape, (batch_size,))

class TestRewardLoss(unittest.TestCase):
    """Test reward loss components."""
    
    def test_bradley_terry_loss(self):
        """Test BradleyTerryLoss implementation."""
        loss_fn = BradleyTerryLoss()
        
        # Create sample rewards
        batch_size = 4
        chosen_rewards = torch.randn(batch_size)
        rejected_rewards = torch.randn(batch_size)
        
        # Test loss computation
        loss = loss_fn(chosen_rewards, rejected_rewards)
        
        # Check that loss is a scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)
        
    def test_margin_ranking_loss(self):
        """Test MarginRankingLoss implementation."""
        margin = 0.1
        loss_fn = MarginRankingLoss(margin=margin)
        
        # Create sample rewards
        batch_size = 4
        chosen_rewards = torch.randn(batch_size)
        rejected_rewards = torch.randn(batch_size)
        
        # Test loss computation
        loss = loss_fn(chosen_rewards, rejected_rewards)
        
        # Check that loss is a scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)

class TestPreferenceData(unittest.TestCase):
    """Test preference data components."""
    
    def test_preference_example(self):
        """Test PreferenceExample data class."""
        # Create preference example
        example = PreferenceExample(
            prompt="What is 2+2?",
            chosen_response="2+2=4",
            rejected_response="2+2=5"
        )
        
        # Check attributes
        self.assertEqual(example.prompt, "What is 2+2?")
        self.assertEqual(example.chosen_response, "2+2=4")
        self.assertEqual(example.rejected_response, "2+2=5")
        self.assertIsNone(example.chosen_score)
        self.assertIsNone(example.rejected_score)
        self.assertIsNone(example.metadata)
        
        # Test with scores and metadata
        example_with_scores = PreferenceExample(
            prompt="What is 2+2?",
            chosen_response="2+2=4",
            rejected_response="2+2=5",
            chosen_score=0.9,
            rejected_score=0.1,
            metadata={"source": "math_dataset"}
        )
        
        self.assertEqual(example_with_scores.chosen_score, 0.9)
        self.assertEqual(example_with_scores.rejected_score, 0.1)
        self.assertEqual(example_with_scores.metadata, {"source": "math_dataset"})

class TestPPO(unittest.TestCase):
    """Test PPO components."""
    
    def test_ppo_config(self):
        """Test PPOConfig data class."""
        config = PPOConfig()
        
        # Check default values
        self.assertEqual(config.ppo_epochs, 4)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.clip_epsilon, 0.2)
        self.assertEqual(config.learning_rate, 1e-5)
        
    def test_policy_value_network(self):
        """Test PolicyValueNetwork implementation."""
        # Create a simple base model for testing
        class SimpleBaseModel(nn.Module):
            def __init__(self, hidden_size=128, vocab_size=100):
                super().__init__()
                self.hidden_size = hidden_size
                self.vocab_size = vocab_size
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.linear = nn.Linear(hidden_size, hidden_size)
                
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                x = self.linear(x)
                logits = torch.randn(input_ids.shape[0], input_ids.shape[1], self.vocab_size)
                # Mock outputs with required attributes
                class MockOutput:
                    def __init__(self, last_hidden_state, logits):
                        self.last_hidden_state = last_hidden_state
                        self.logits = logits
                return MockOutput(x, logits)
        
        base_model = SimpleBaseModel(hidden_size=128, vocab_size=100)
        policy_model = PolicyValueNetwork(base_model, hidden_size=128)
        
        # Create sample input
        batch_size, seq_length = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        
        # Test forward pass
        logits, values = policy_model(input_ids)
        
        # Check output shapes
        self.assertEqual(logits.shape, (batch_size, seq_length, 100))
        self.assertEqual(values.shape, (batch_size, seq_length))

class TestGRPO(unittest.TestCase):
    """Test GRPO components."""
    
    def test_grpo_config(self):
        """Test GRPOConfig data class."""
        config = GRPOConfig()
        
        # Check default values
        self.assertEqual(config.grpo_epochs, 4)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.num_completions_per_prompt, 4)
        self.assertEqual(config.group_size, 4)
        
    # Note: More comprehensive GRPO tests would require more complex setup

if __name__ == '__main__':
    unittest.main()