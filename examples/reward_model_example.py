"""
Example demonstrating Reward Modeling architecture.
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn

from reward.model import (
    RewardModel, 
    PairwiseRewardModel, 
    DebertaRewardModel
)
from reward.preference_data import (
    PreferenceExample, 
    PreferenceDataProcessor,
    generate_sample_preference_data
)


class MockTokenizer:
    """Mock tokenizer for demonstration purposes."""
    
    def __init__(self):
        self.vocab_size = 1000
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
    
    def encode(self, text, add_bos=False, add_eos=False):
        """Simple mock encoding."""
        # Convert text to list of integers (mock tokenization)
        tokens = [ord(c) % self.vocab_size for c in text[:50]]  # Limit to 50 tokens
        
        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token_id]
            
        return tokens


class SimpleBaseModel(nn.Module):
    """Simple base model for testing reward models."""
    
    def __init__(self, vocab_size=1000, hidden_size=128):
        super(SimpleBaseModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=2
        )
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        embedded = self.embedding(input_ids)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to key_padding_mask format
            key_padding_mask = (1 - attention_mask).bool()
        else:
            key_padding_mask = None
        
        # Transformer
        output = self.transformer(embedded, src_key_padding_mask=key_padding_mask)
        return output, None  # Return tuple to match expected format, None  # Return tuple to match expected format


def main():
    """Main function demonstrating reward model architecture."""
    print("Reward Model Architecture Example")
    print("=" * 35)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create base model
    print("1. Creating base language model...")
    base_model = SimpleBaseModel(vocab_size=1000, hidden_size=128)
    print(f"  Base model hidden size: 128")
    print(f"  Base model parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    
    # Test RewardModel
    print("\n2. Testing RewardModel...")
    reward_model = RewardModel(base_model, hidden_size=128)
    print(f"  Reward model parameters: {sum(p.numel() for p in reward_model.parameters()):,}")
    
    # Create sample input
    batch_size, seq_len = 4, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Attention mask shape: {attention_mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        rewards = reward_model(input_ids, attention_mask)
    print(f"  Reward shape: {rewards.shape}")
    print(f"  Sample rewards: {rewards.tolist()}")
    
    # Test PairwiseRewardModel
    print("\n3. Testing PairwiseRewardModel...")
    pairwise_model = PairwiseRewardModel(base_model, hidden_size=128)
    print(f"  Pairwise model parameters: {sum(p.numel() for p in pairwise_model.parameters()):,}")
    
    # Create sample inputs for chosen and rejected responses
    chosen_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    chosen_attention_mask = torch.ones(batch_size, seq_len)
    rejected_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    rejected_attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"  Chosen input shape: {chosen_input_ids.shape}")
    print(f"  Rejected input shape: {rejected_input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        chosen_rewards, rejected_rewards = pairwise_model(
            chosen_input_ids, chosen_attention_mask,
            rejected_input_ids, rejected_attention_mask
        )
    print(f"  Chosen rewards shape: {chosen_rewards.shape}")
    print(f"  Rejected rewards shape: {rejected_rewards.shape}")
    print(f"  Sample chosen rewards: {chosen_rewards.tolist()}")
    print(f"  Sample rejected rewards: {rejected_rewards.tolist()}")
    
    # Test with preference data
    print("\n4. Testing with preference data...")
    
    # Generate sample preference data
    examples = generate_sample_preference_data()
    print(f"  Generated {len(examples)} preference examples")
    
    # Create tokenizer and processor
    tokenizer = MockTokenizer()
    processor = PreferenceDataProcessor(max_length=64, pad_token_id=tokenizer.pad_token_id)
    
    # Process first example
    first_example = examples[0]
    processed = processor.preprocess_example(first_example, tokenizer)
    
    print(f"  Processed example shapes:")
    print(f"    Chosen input_ids: {processed['chosen_input_ids'].shape}")
    print(f"    Rejected input_ids: {processed['rejected_input_ids'].shape}")
    
    # Compute rewards for processed example
    with torch.no_grad():
        chosen_reward = reward_model(
            processed['chosen_input_ids'].unsqueeze(0),  # Add batch dimension
            processed['chosen_attention_mask'].unsqueeze(0)
        )
        rejected_reward = reward_model(
            processed['rejected_input_ids'].unsqueeze(0),  # Add batch dimension
            processed['rejected_attention_mask'].unsqueeze(0)
        )
    
    print(f"  Chosen response reward: {chosen_reward.item():.4f}")
    print(f"  Rejected response reward: {rejected_reward.item():.4f}")
    print(f"  Reward difference (chosen - rejected): {(chosen_reward - rejected_reward).item():.4f}")
    
    # Test DebertaRewardModel
    print("\n5. Testing DebertaRewardModel...")
    deberta_model = DebertaRewardModel(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4)
    print(f"  DeBERTa model parameters: {sum(p.numel() for p in deberta_model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        deberta_rewards = deberta_model(input_ids, attention_mask)
    print(f"  DeBERTa reward shape: {deberta_rewards.shape}")
    print(f"  Sample DeBERTa rewards: {deberta_rewards.tolist()}")
    
    # Compare model sizes
    print("\n6. Model Size Comparison:")
    print(f"  Base model:        {sum(p.numel() for p in base_model.parameters()):>10,} parameters")
    print(f"  Reward model:      {sum(p.numel() for p in reward_model.parameters()):>10,} parameters")
    print(f"  Pairwise model:    {sum(p.numel() for p in pairwise_model.parameters()):>10,} parameters")
    print(f"  DeBERTa model:     {sum(p.numel() for p in deberta_model.parameters()):>10,} parameters")
    
    # Test model training
    print("\n7. Testing model training setup...")
    
    # Create optimizer
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)
    print(f"  Optimizer: Adam with learning rate 1e-4")
    
    # Simple training step
    reward_model.train()
    optimizer.zero_grad()
    
    # Compute rewards with gradient tracking
    chosen_rewards_train, rejected_rewards_train = pairwise_model(
        chosen_input_ids, chosen_attention_mask,
        rejected_input_ids, rejected_attention_mask
    )
    
    # Compute loss (we want chosen rewards to be higher than rejected rewards)
    reward_loss = -torch.log(torch.sigmoid(chosen_rewards_train - rejected_rewards_train)).mean()
    print(f"  Sample pairwise loss: {reward_loss.item():.4f}")
    
    # Backward pass
    reward_loss.backward()
    print(f"  Backward pass completed")
    
    # Update parameters
    optimizer.step()
    print(f"  Parameters updated")
    
    print("\nReward model architecture example completed!")


if __name__ == "__main__":
    main()