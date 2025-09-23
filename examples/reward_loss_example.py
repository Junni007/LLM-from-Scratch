"""
Example demonstrating Reward Modeling loss functions.
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn

from reward.loss import (
    BradleyTerryLoss,
    MarginRankingLoss,
    HingeLoss,
    ContrastiveLoss,
    AdaptiveLoss,
    compute_accuracy,
    compute_reward_gap
)
from reward.model import PairwiseRewardModel
from reward.preference_data import (
    PreferenceExample, 
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
        return output, None  # Return tuple to match expected format


def main():
    """Main function demonstrating reward loss functions."""
    print("Reward Loss Functions Example")
    print("=" * 30)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create sample rewards for demonstration
    batch_size = 4
    
    print("1. Creating sample reward data...")
    
    # Perfect predictions (chosen > rejected)
    chosen_rewards_perfect = torch.tensor([3.0, 2.5, 4.0, 3.5])
    rejected_rewards_perfect = torch.tensor([1.0, 1.5, 2.0, 2.5])
    
    # Imperfect predictions
    chosen_rewards_imperfect = torch.tensor([1.0, 2.0, 1.5, 2.5])
    rejected_rewards_imperfect = torch.tensor([2.0, 1.0, 3.0, 1.5])
    
    # Random predictions
    chosen_rewards_random = torch.randn(batch_size) * 2 + 2  # Mean 2, std 2
    rejected_rewards_random = torch.randn(batch_size) * 2 + 1  # Mean 1, std 2
    
    print(f"  Perfect predictions:")
    print(f"    Chosen:  {chosen_rewards_perfect.tolist()}")
    print(f"    Rejected: {rejected_rewards_perfect.tolist()}")
    
    print(f"  Imperfect predictions:")
    print(f"    Chosen:  {chosen_rewards_imperfect.tolist()}")
    print(f"    Rejected: {rejected_rewards_imperfect.tolist()}")
    
    print(f"  Random predictions:")
    print(f"    Chosen:  {chosen_rewards_random.tolist()}")
    print(f"    Rejected: {rejected_rewards_random.tolist()}")
    
    # Test Bradley-Terry Loss
    print("\n2. Testing Bradley-Terry Loss...")
    bt_loss_fn = BradleyTerryLoss()
    
    bt_loss_perfect = bt_loss_fn(chosen_rewards_perfect, rejected_rewards_perfect)
    bt_loss_imperfect = bt_loss_fn(chosen_rewards_imperfect, rejected_rewards_imperfect)
    bt_loss_random = bt_loss_fn(chosen_rewards_random, rejected_rewards_random)
    
    print(f"  Perfect predictions loss: {bt_loss_perfect.item():.4f}")
    print(f"  Imperfect predictions loss: {bt_loss_imperfect.item():.4f}")
    print(f"  Random predictions loss: {bt_loss_random.item():.4f}")
    
    # Test Margin Ranking Loss
    print("\n3. Testing Margin Ranking Loss...")
    margin_loss_fn = MarginRankingLoss(margin=0.5)
    
    margin_loss_perfect = margin_loss_fn(chosen_rewards_perfect, rejected_rewards_perfect)
    margin_loss_imperfect = margin_loss_fn(chosen_rewards_imperfect, rejected_rewards_imperfect)
    margin_loss_random = margin_loss_fn(chosen_rewards_random, rejected_rewards_random)
    
    print(f"  Perfect predictions loss: {margin_loss_perfect.item():.4f}")
    print(f"  Imperfect predictions loss: {margin_loss_imperfect.item():.4f}")
    print(f"  Random predictions loss: {margin_loss_random.item():.4f}")
    
    # Test Hinge Loss
    print("\n4. Testing Hinge Loss...")
    hinge_loss_fn = HingeLoss(margin=1.0)
    
    hinge_loss_perfect = hinge_loss_fn(chosen_rewards_perfect, rejected_rewards_perfect)
    hinge_loss_imperfect = hinge_loss_fn(chosen_rewards_imperfect, rejected_rewards_imperfect)
    hinge_loss_random = hinge_loss_fn(chosen_rewards_random, rejected_rewards_random)
    
    print(f"  Perfect predictions loss: {hinge_loss_perfect.item():.4f}")
    print(f"  Imperfect predictions loss: {hinge_loss_imperfect.item():.4f}")
    print(f"  Random predictions loss: {hinge_loss_random.item():.4f}")
    
    # Test Contrastive Loss
    print("\n5. Testing Contrastive Loss...")
    contrastive_loss_fn = ContrastiveLoss(margin=1.0)
    
    contrastive_loss_perfect = contrastive_loss_fn(chosen_rewards_perfect, rejected_rewards_perfect)
    contrastive_loss_imperfect = contrastive_loss_fn(chosen_rewards_imperfect, rejected_rewards_imperfect)
    contrastive_loss_random = contrastive_loss_fn(chosen_rewards_random, rejected_rewards_random)
    
    print(f"  Perfect predictions loss: {contrastive_loss_perfect.item():.4f}")
    print(f"  Imperfect predictions loss: {contrastive_loss_imperfect.item():.4f}")
    print(f"  Random predictions loss: {contrastive_loss_random.item():.4f}")
    
    # Test Adaptive Loss
    print("\n6. Testing Adaptive Loss...")
    adaptive_loss_fn = AdaptiveLoss(alpha=0.5, beta=0.5)
    
    adaptive_loss_perfect = adaptive_loss_fn(chosen_rewards_perfect, rejected_rewards_perfect)
    adaptive_loss_imperfect = adaptive_loss_fn(chosen_rewards_imperfect, rejected_rewards_imperfect)
    adaptive_loss_random = adaptive_loss_fn(chosen_rewards_random, rejected_rewards_random)
    
    print(f"  Perfect predictions loss: {adaptive_loss_perfect.item():.4f}")
    print(f"  Imperfect predictions loss: {adaptive_loss_imperfect.item():.4f}")
    print(f"  Random predictions loss: {adaptive_loss_random.item():.4f}")
    
    # Test evaluation metrics
    print("\n7. Testing Evaluation Metrics...")
    
    # Accuracy
    accuracy_perfect = compute_accuracy(chosen_rewards_perfect, rejected_rewards_perfect)
    accuracy_imperfect = compute_accuracy(chosen_rewards_imperfect, rejected_rewards_imperfect)
    accuracy_random = compute_accuracy(chosen_rewards_random, rejected_rewards_random)
    
    print(f"  Perfect predictions accuracy: {accuracy_perfect:.2f}")
    print(f"  Imperfect predictions accuracy: {accuracy_imperfect:.2f}")
    print(f"  Random predictions accuracy: {accuracy_random:.2f}")
    
    # Reward gap
    gap_perfect = compute_reward_gap(chosen_rewards_perfect, rejected_rewards_perfect)
    gap_imperfect = compute_reward_gap(chosen_rewards_imperfect, rejected_rewards_imperfect)
    gap_random = compute_reward_gap(chosen_rewards_random, rejected_rewards_random)
    
    print(f"  Perfect predictions reward gap: {gap_perfect:.4f}")
    print(f"  Imperfect predictions reward gap: {gap_imperfect:.4f}")
    print(f"  Random predictions reward gap: {gap_random:.4f}")
    
    # Test with actual model
    print("\n8. Testing with Actual Reward Model...")
    
    # Create base model and reward model
    base_model = SimpleBaseModel(vocab_size=1000, hidden_size=128)
    reward_model = PairwiseRewardModel(base_model, hidden_size=128)
    
    # Create sample inputs
    batch_size, seq_len = 4, 16
    chosen_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    chosen_attention_mask = torch.ones(batch_size, seq_len)
    rejected_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    rejected_attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"  Input shapes: {chosen_input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        chosen_rewards, rejected_rewards = reward_model(
            chosen_input_ids, chosen_attention_mask,
            rejected_input_ids, rejected_attention_mask
        )
    
    print(f"  Model output shapes: {chosen_rewards.shape}, {rejected_rewards.shape}")
    print(f"  Sample chosen rewards: {chosen_rewards.tolist()}")
    print(f"  Sample rejected rewards: {rejected_rewards.tolist()}")
    
    # Compute losses with model outputs
    bt_loss = bt_loss_fn(chosen_rewards, rejected_rewards)
    margin_loss = margin_loss_fn(chosen_rewards, rejected_rewards)
    
    print(f"  Bradley-Terry loss: {bt_loss.item():.4f}")
    print(f"  Margin loss: {margin_loss.item():.4f}")
    
    # Test training
    print("\n9. Testing Model Training...")
    
    # Create optimizer
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)
    
    # Training step
    reward_model.train()
    optimizer.zero_grad()
    
    # Forward pass with gradient tracking
    chosen_rewards_train, rejected_rewards_train = reward_model(
        chosen_input_ids, chosen_attention_mask,
        rejected_input_ids, rejected_attention_mask
    )
    
    # Compute loss
    train_loss = bt_loss_fn(chosen_rewards_train, rejected_rewards_train)
    print(f"  Training loss: {train_loss.item():.4f}")
    
    # Backward pass
    train_loss.backward()
    print(f"  Backward pass completed")
    
    # Update parameters
    optimizer.step()
    print(f"  Parameters updated")
    
    # Show parameter change
    with torch.no_grad():
        new_chosen_rewards, new_rejected_rewards = reward_model(
            chosen_input_ids, chosen_attention_mask,
            rejected_input_ids, rejected_attention_mask
        )
    
    reward_change = (new_chosen_rewards - chosen_rewards).mean().item()
    print(f"  Average reward change after update: {reward_change:.6f}")
    
    print("\nReward loss functions example completed!")


if __name__ == "__main__":
    main()