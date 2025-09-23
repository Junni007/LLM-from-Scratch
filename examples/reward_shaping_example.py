"""
Example demonstrating Reward Shaping and Sanity Checks.
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn

from reward.shaping import (
    RewardShaping,
    SanityChecker,
    RewardVisualizer
)
from reward.model import PairwiseRewardModel
from reward.loss import BradleyTerryLoss


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
    """Main function demonstrating reward shaping and sanity checks."""
    print("Reward Shaping and Sanity Checks Example")
    print("=" * 42)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create sample rewards for demonstration
    batch_size = 50
    
    print("1. Creating sample reward data...")
    
    # Perfect predictions (chosen > rejected)
    chosen_rewards_perfect = torch.tensor([3.0, 4.0, 5.0, 2.0, 6.0] * 10)  # Repeat 10 times
    rejected_rewards_perfect = torch.tensor([1.0, 2.0, 3.0, 1.0, 4.0] * 10)
    
    # Good predictions
    chosen_rewards_good = torch.randn(batch_size) * 1.5 + 3.0  # Mean 3, std 1.5
    rejected_rewards_good = torch.randn(batch_size) * 1.2 + 1.5  # Mean 1.5, std 1.2
    
    # Poor predictions
    chosen_rewards_poor = torch.randn(batch_size) * 1.0 + 1.0  # Mean 1, std 1
    rejected_rewards_poor = torch.randn(batch_size) * 1.0 + 1.5  # Mean 1.5, std 1
    
    print(f"  Perfect predictions: {batch_size} pairs")
    print(f"  Good predictions: {batch_size} pairs")
    print(f"  Poor predictions: {batch_size} pairs")
    
    # Test reward shaping
    print("\n2. Testing Reward Shaping...")
    shaper = RewardShaping()
    
    # Test different normalization methods
    print("  Normalization methods:")
    methods = ["min_max", "z_score", "unit"]
    for method in methods:
        normalized = shaper.normalize_rewards(chosen_rewards_good, method=method)
        print(f"    {method:8}: Mean={normalized.mean().item():.4f}, Std={normalized.std().item():.4f}")
    
    # Test clipping
    print("  Clipping:")
    clipped = shaper.clip_rewards(chosen_rewards_good, min_val=-2.0, max_val=5.0)
    print(f"    Original range: [{chosen_rewards_good.min().item():.2f}, {chosen_rewards_good.max().item():.2f}]")
    print(f"    Clipped range:  [{clipped.min().item():.2f}, {clipped.max().item():.2f}]")
    
    # Test scaling and bias
    print("  Scaling and bias:")
    scaled = shaper.apply_reward_scaling(chosen_rewards_good, scale_factor=0.5)
    biased = shaper.apply_reward_bias(chosen_rewards_good, bias=2.0)
    print(f"    Original mean: {chosen_rewards_good.mean().item():.2f}")
    print(f"    Scaled mean (0.5x): {scaled.mean().item():.2f}")
    print(f"    Biased mean (+2.0): {biased.mean().item():.2f}")
    
    # Test sanity checks
    print("\n3. Testing Sanity Checks...")
    checker = SanityChecker()
    
    # Perfect predictions
    print("  Perfect predictions:")
    perfect_metrics = checker.check_preference_consistency(chosen_rewards_perfect, rejected_rewards_perfect)
    for metric, value in perfect_metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    # Good predictions
    print("  Good predictions:")
    good_metrics = checker.check_preference_consistency(chosen_rewards_good, rejected_rewards_good)
    for metric, value in good_metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    # Poor predictions
    print("  Poor predictions:")
    poor_metrics = checker.check_preference_consistency(chosen_rewards_poor, rejected_rewards_poor)
    for metric, value in poor_metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    # Test reward distribution analysis
    print("\n4. Reward Distribution Analysis:")
    chosen_dist = checker.check_reward_distribution(chosen_rewards_good)
    rejected_dist = checker.check_reward_distribution(rejected_rewards_good)
    
    print("  Chosen rewards distribution:")
    for metric, value in chosen_dist.items():
        print(f"    {metric}: {value:.4f}")
    
    print("  Rejected rewards distribution:")
    for metric, value in rejected_dist.items():
        print(f"    {metric}: {value:.4f}")
    
    # Test calibration
    print("\n5. Calibration Analysis:")
    calibration_perfect = checker.check_calibration(chosen_rewards_perfect, rejected_rewards_perfect)
    calibration_good = checker.check_calibration(chosen_rewards_good, rejected_rewards_good)
    calibration_poor = checker.check_calibration(chosen_rewards_poor, rejected_rewards_poor)
    
    print("  Perfect predictions calibration:")
    for metric, value in calibration_perfect.items():
        print(f"    {metric}: {value:.4f}")
    
    print("  Good predictions calibration:")
    for metric, value in calibration_good.items():
        print(f"    {metric}: {value:.4f}")
    
    print("  Poor predictions calibration:")
    for metric, value in calibration_poor.items():
        print(f"    {metric}: {value:.4f}")
    
    # Test with actual model
    print("\n6. Testing with Actual Reward Model...")
    
    # Create base model and reward model
    base_model = SimpleBaseModel(vocab_size=1000, hidden_size=128)
    reward_model = PairwiseRewardModel(base_model, hidden_size=128)
    
    # Create sample inputs
    chosen_input_ids = torch.randint(0, 1000, (batch_size, 16))
    chosen_attention_mask = torch.ones(batch_size, 16)
    rejected_input_ids = torch.randint(0, 1000, (batch_size, 16))
    rejected_attention_mask = torch.ones(batch_size, 16)
    
    print(f"  Input shapes: {chosen_input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        chosen_rewards, rejected_rewards = reward_model(
            chosen_input_ids, chosen_attention_mask,
            rejected_input_ids, rejected_attention_mask
        )
    
    print(f"  Model output shapes: {chosen_rewards.shape}, {rejected_rewards.shape}")
    
    # Apply sanity checks to model outputs
    model_metrics = checker.check_preference_consistency(chosen_rewards, rejected_rewards)
    print("  Model output consistency:")
    for metric, value in model_metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    # Test reward distribution
    model_chosen_dist = checker.check_reward_distribution(chosen_rewards)
    model_rejected_dist = checker.check_reward_distribution(rejected_rewards)
    print("  Model reward distributions:")
    print(f"    Chosen - Mean: {model_chosen_dist['mean']:.4f}, Std: {model_chosen_dist['std']:.4f}")
    print(f"    Rejected - Mean: {model_rejected_dist['mean']:.4f}, Std: {model_rejected_dist['std']:.4f}")
    
    # Test training with shaping
    print("\n7. Testing Training with Reward Shaping...")
    
    # Create optimizer
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)
    loss_fn = BradleyTerryLoss()
    
    # Training step without shaping
    reward_model.train()
    optimizer.zero_grad()
    
    chosen_rewards_train, rejected_rewards_train = reward_model(
        chosen_input_ids, chosen_attention_mask,
        rejected_input_ids, rejected_attention_mask
    )
    
    loss_before = loss_fn(chosen_rewards_train, rejected_rewards_train)
    print(f"  Loss before training: {loss_before.item():.4f}")
    
    # Backward pass
    loss_before.backward()
    optimizer.step()
    
    # Forward pass after training
    with torch.no_grad():
        chosen_rewards_after, rejected_rewards_after = reward_model(
            chosen_input_ids, chosen_attention_mask,
            rejected_input_ids, rejected_attention_mask
        )
    
    loss_after = loss_fn(chosen_rewards_after, rejected_rewards_after)
    print(f"  Loss after training: {loss_after.item():.4f}")
    print(f"  Loss improvement: {(loss_before - loss_after).item():.4f}")
    
    # Check consistency improvement
    consistency_before = checker.check_preference_consistency(chosen_rewards_train.detach(), rejected_rewards_train.detach())
    consistency_after = checker.check_preference_consistency(chosen_rewards_after, rejected_rewards_after)
    print(f"  Accuracy improvement: {consistency_after['accuracy'] - consistency_before['accuracy']:.4f}")
    
    # Test visualization (if matplotlib is available)
    print("\n8. Testing Visualization...")
    visualizer = RewardVisualizer()
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        matplotlib_available = True
    except ImportError:
        matplotlib_available = False
    
    if matplotlib_available:
        # Create a simple plot
        fig1 = visualizer.plot_reward_distribution(chosen_rewards_after, "Chosen Rewards Distribution")
        print("  Created reward distribution plot")
        
        fig2 = visualizer.plot_preference_scatter(chosen_rewards_after, rejected_rewards_after, 
                                                "Preference Scatter Plot")
        print("  Created preference scatter plot")
        
        # Create mock training history
        mock_history = [
            {"accuracy": 0.6, "reward_gap": 0.5, "loss": 1.2},
            {"accuracy": 0.7, "reward_gap": 0.8, "loss": 0.9},
            {"accuracy": 0.8, "reward_gap": 1.1, "loss": 0.6},
            {"accuracy": 0.85, "reward_gap": 1.3, "loss": 0.4},
            {"accuracy": 0.9, "reward_gap": 1.5, "loss": 0.3}
        ]
        
        fig3 = visualizer.plot_training_curves(mock_history, "Mock Training Curves")
        print("  Created training curves plot")
        
        print("  Visualization tests completed (plots would be shown with plt.show())")
    else:
        print("  Visualization tests skipped (matplotlib not available)")
    
    print("\nReward shaping and sanity checks example completed!")


if __name__ == "__main__":
    main()