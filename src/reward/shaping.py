"""
Reward shaping and sanity checks for Reward Modeling.

This module implements various techniques for reward shaping and sanity checks
to ensure reward models are learning meaningful preferences.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False


class RewardShaping:
    """
    Reward shaping utilities for reward modeling.
    """
    
    def __init__(self):
        """Initialize RewardShaping."""
        pass
    
    def normalize_rewards(self, rewards: torch.Tensor, 
                         method: str = "min_max") -> torch.Tensor:
        """
        Normalize rewards using various methods.
        
        Args:
            rewards (torch.Tensor): Rewards to normalize
            method (str): Normalization method ("min_max", "z_score", "unit")
            
        Returns:
            torch.Tensor: Normalized rewards
        """
        if method == "min_max":
            # Min-max normalization to [0, 1]
            min_reward = rewards.min()
            max_reward = rewards.max()
            if max_reward > min_reward:
                normalized = (rewards - min_reward) / (max_reward - min_reward)
            else:
                normalized = torch.zeros_like(rewards)
        elif method == "z_score":
            # Z-score normalization
            mean_reward = rewards.mean()
            std_reward = rewards.std()
            if std_reward > 0:
                normalized = (rewards - mean_reward) / std_reward
            else:
                normalized = torch.zeros_like(rewards)
        elif method == "unit":
            # Unit normalization (L2 norm)
            norm = torch.norm(rewards)
            if norm > 0:
                normalized = rewards / norm
            else:
                normalized = torch.zeros_like(rewards)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def clip_rewards(self, rewards: torch.Tensor, 
                    min_val: float = -10.0, max_val: float = 10.0) -> torch.Tensor:
        """
        Clip rewards to prevent extreme values.
        
        Args:
            rewards (torch.Tensor): Rewards to clip
            min_val (float): Minimum value
            max_val (float): Maximum value
            
        Returns:
            torch.Tensor: Clipped rewards
        """
        return torch.clamp(rewards, min_val, max_val)
    
    def apply_reward_scaling(self, rewards: torch.Tensor, 
                           scale_factor: float = 1.0) -> torch.Tensor:
        """
        Scale rewards by a factor.
        
        Args:
            rewards (torch.Tensor): Rewards to scale
            scale_factor (float): Scaling factor
            
        Returns:
            torch.Tensor: Scaled rewards
        """
        return rewards * scale_factor
    
    def apply_reward_bias(self, rewards: torch.Tensor, 
                         bias: float = 0.0) -> torch.Tensor:
        """
        Apply bias to rewards.
        
        Args:
            rewards (torch.Tensor): Rewards to bias
            bias (float): Bias value
            
        Returns:
            torch.Tensor: Biased rewards
        """
        return rewards + bias


class SanityChecker:
    """
    Sanity checks for reward models.
    """
    
    def __init__(self):
        """Initialize SanityChecker."""
        pass
    
    def check_preference_consistency(self, chosen_rewards: torch.Tensor, 
                                   rejected_rewards: torch.Tensor) -> Dict[str, float]:
        """
        Check consistency of preference predictions.
        
        Args:
            chosen_rewards (torch.Tensor): Rewards for chosen responses
            rejected_rewards (torch.Tensor): Rewards for rejected responses
            
        Returns:
            Dict[str, float]: Consistency metrics
        """
        # Accuracy: proportion of correct predictions
        correct_predictions = (chosen_rewards > rejected_rewards).float()
        accuracy = correct_predictions.mean().item()
        
        # Reward gap: average difference between chosen and rejected
        reward_gap = (chosen_rewards - rejected_rewards).mean().item()
        
        # Margin: minimum difference
        min_margin = (chosen_rewards - rejected_rewards).min().item()
        
        # Variance ratio: ratio of within-pair variance to between-pair variance
        pair_means = (chosen_rewards + rejected_rewards) / 2
        between_pair_variance = pair_means.var()
        within_pair_differences = chosen_rewards - rejected_rewards
        within_pair_variance = within_pair_differences.var()
        
        if between_pair_variance > 0:
            variance_ratio = within_pair_variance / between_pair_variance
        else:
            variance_ratio = 0.0
        
        return {
            "accuracy": accuracy,
            "reward_gap": reward_gap,
            "min_margin": min_margin,
            "variance_ratio": variance_ratio
        }
    
    def check_reward_distribution(self, rewards: torch.Tensor) -> Dict[str, float]:
        """
        Check reward distribution statistics.
        
        Args:
            rewards (torch.Tensor): Rewards to analyze
            
        Returns:
            Dict[str, float]: Distribution metrics
        """
        return {
            "mean": rewards.mean().item(),
            "std": rewards.std().item(),
            "min": rewards.min().item(),
            "max": rewards.max().item(),
            "median": torch.median(rewards).item(),
            "skewness": self._compute_skewness(rewards).item()
        }
    
    def _compute_skewness(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute skewness of a tensor.
        
        Args:
            tensor (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Skewness
        """
        mean = tensor.mean()
        std = tensor.std()
        if std > 0:
            normalized = (tensor - mean) / std
            skewness = (normalized ** 3).mean()
        else:
            skewness = torch.tensor(0.0)
        return skewness
    
    def check_overfitting(self, train_rewards: torch.Tensor, 
                         val_rewards: torch.Tensor) -> Dict[str, float]:
        """
        Check for overfitting between training and validation rewards.
        
        Args:
            train_rewards (torch.Tensor): Training rewards
            val_rewards (torch.Tensor): Validation rewards
            
        Returns:
            Dict[str, float]: Overfitting metrics
        """
        train_stats = self.check_reward_distribution(train_rewards)
        val_stats = self.check_reward_distribution(val_rewards)
        
        # Difference in means
        mean_diff = abs(train_stats["mean"] - val_stats["mean"])
        
        # Difference in standard deviations
        std_diff = abs(train_stats["std"] - val_stats["std"])
        
        return {
            "mean_difference": mean_diff,
            "std_difference": std_diff
        }
    
    def check_calibration(self, chosen_rewards: torch.Tensor, 
                         rejected_rewards: torch.Tensor,
                         confidence_threshold: float = 0.9) -> Dict[str, float]:
        """
        Check calibration of reward model confidence.
        
        Args:
            chosen_rewards (torch.Tensor): Rewards for chosen responses
            rejected_rewards (torch.Tensor): Rewards for rejected responses
            confidence_threshold (float): Threshold for high confidence predictions
            
        Returns:
            Dict[str, float]: Calibration metrics
        """
        # Compute probabilities using sigmoid
        reward_diffs = chosen_rewards - rejected_rewards
        probabilities = torch.sigmoid(reward_diffs)
        
        # High confidence predictions
        high_conf_mask = (probabilities > confidence_threshold) | (probabilities < (1 - confidence_threshold))
        high_conf_correct = (chosen_rewards > rejected_rewards)[high_conf_mask].float()
        high_conf_accuracy = high_conf_correct.mean().item() if high_conf_mask.sum() > 0 else 0.0
        
        # Low confidence predictions
        low_conf_mask = ~high_conf_mask
        low_conf_correct = (chosen_rewards > rejected_rewards)[low_conf_mask].float()
        low_conf_accuracy = low_conf_correct.mean().item() if low_conf_mask.sum() > 0 else 0.0
        
        return {
            "high_confidence_accuracy": high_conf_accuracy,
            "low_confidence_accuracy": low_conf_accuracy,
            "high_confidence_ratio": high_conf_mask.float().mean().item(),
            "low_confidence_ratio": low_conf_mask.float().mean().item()
        }


class RewardVisualizer:
    """
    Visualization tools for reward analysis.
    """
    
    def __init__(self):
        """Initialize RewardVisualizer."""
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
    
    def plot_reward_distribution(self, rewards: torch.Tensor, 
                                title: str = "Reward Distribution") -> Any:
        """
        Plot reward distribution histogram.
        
        Args:
            rewards (torch.Tensor): Rewards to plot
            title (str): Plot title
            
        Returns:
            plt.Figure or None: Matplotlib figure or None if matplotlib not available
        """
        if not self.matplotlib_available:
            print("Matplotlib not available, skipping visualization")
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(rewards.cpu().numpy(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig
    
    def plot_preference_scatter(self, chosen_rewards: torch.Tensor, 
                              rejected_rewards: torch.Tensor,
                              title: str = "Preference Scatter Plot") -> Any:
        """
        Plot chosen vs rejected rewards scatter plot.
        
        Args:
            chosen_rewards (torch.Tensor): Chosen rewards
            rejected_rewards (torch.Tensor): Rejected rewards
            title (str): Plot title
            
        Returns:
            plt.Figure or None: Matplotlib figure or None if matplotlib not available
        """
        if not self.matplotlib_available:
            print("Matplotlib not available, skipping visualization")
            return None
            
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(rejected_rewards.cpu().numpy(), chosen_rewards.cpu().numpy(), 
                  alpha=0.6, color='blue')
        
        # Add diagonal line (y = x)
        min_val = min(chosen_rewards.min().item(), rejected_rewards.min().item())
        max_val = max(chosen_rewards.max().item(), rejected_rewards.max().item())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1)
        
        ax.set_xlabel('Rejected Rewards')
        ax.set_ylabel('Chosen Rewards')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig
    
    def plot_training_curves(self, metrics_history: List[Dict[str, float]],
                           title: str = "Training Metrics") -> Any:
        """
        Plot training metrics curves.
        
        Args:
            metrics_history (List[Dict[str, float]]): History of metrics
            title (str): Plot title
            
        Returns:
            plt.Figure or None: Matplotlib figure or None if matplotlib not available
        """
        if not self.matplotlib_available:
            print("Matplotlib not available, skipping visualization")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract metrics
        epochs = list(range(len(metrics_history)))
        if metrics_history:
            metrics = {key: [entry[key] for entry in metrics_history] 
                      for key in metrics_history[0].keys()}
            
            # Plot each metric
            for metric_name, values in metrics.items():
                ax.plot(epochs, values, label=metric_name, marker='o', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig


def demonstrate_reward_shaping():
    """Demonstrate reward shaping and sanity checks."""
    print("Reward Shaping and Sanity Checks Demonstration")
    print("=" * 48)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create sample rewards
    batch_size = 100
    chosen_rewards = torch.randn(batch_size) * 2 + 3  # Mean 3, std 2
    rejected_rewards = torch.randn(batch_size) * 1.5 + 1  # Mean 1, std 1.5
    
    print("1. Sample Rewards:")
    print(f"  Chosen rewards - Mean: {chosen_rewards.mean().item():.2f}, Std: {chosen_rewards.std().item():.2f}")
    print(f"  Rejected rewards - Mean: {rejected_rewards.mean().item():.2f}, Std: {rejected_rewards.std().item():.2f}")
    
    # Test reward shaping
    print("\n2. Reward Shaping:")
    shaper = RewardShaping()
    
    # Normalization
    normalized_chosen = shaper.normalize_rewards(chosen_rewards, method="min_max")
    normalized_rejected = shaper.normalize_rewards(rejected_rewards, method="min_max")
    print(f"  Min-max normalized chosen - Range: [{normalized_chosen.min().item():.2f}, {normalized_chosen.max().item():.2f}]")
    
    # Clipping
    clipped_chosen = shaper.clip_rewards(chosen_rewards, min_val=-5.0, max_val=10.0)
    print(f"  Clipped chosen - Min: {clipped_chosen.min().item():.2f}, Max: {clipped_chosen.max().item():.2f}")
    
    # Scaling and bias
    scaled_chosen = shaper.apply_reward_scaling(chosen_rewards, scale_factor=0.5)
    biased_chosen = shaper.apply_reward_bias(chosen_rewards, bias=1.0)
    print(f"  Scaled chosen (0.5x) - Mean: {scaled_chosen.mean().item():.2f}")
    print(f"  Biased chosen (+1.0) - Mean: {biased_chosen.mean().item():.2f}")
    
    # Test sanity checks
    print("\n3. Sanity Checks:")
    checker = SanityChecker()
    
    # Preference consistency
    consistency_metrics = checker.check_preference_consistency(chosen_rewards, rejected_rewards)
    print("  Preference Consistency:")
    for metric, value in consistency_metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    # Reward distribution
    distribution_metrics = checker.check_reward_distribution(chosen_rewards)
    print("  Reward Distribution (Chosen):")
    for metric, value in distribution_metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    # Calibration
    calibration_metrics = checker.check_calibration(chosen_rewards, rejected_rewards)
    print("  Calibration:")
    for metric, value in calibration_metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    # Test with perfect predictions
    print("\n4. Perfect Predictions Case:")
    perfect_chosen = torch.tensor([3.0, 4.0, 5.0, 2.0, 6.0])
    perfect_rejected = torch.tensor([1.0, 2.0, 3.0, 1.0, 4.0])
    
    perfect_consistency = checker.check_preference_consistency(perfect_chosen, perfect_rejected)
    print(f"  Accuracy: {perfect_consistency['accuracy']:.2f}")
    print(f"  Reward gap: {perfect_consistency['reward_gap']:.2f}")
    
    # Test with random predictions
    print("\n5. Random Predictions Case:")
    random_chosen = torch.randn(50) * 2
    random_rejected = torch.randn(50) * 2
    
    random_consistency = checker.check_preference_consistency(random_chosen, random_rejected)
    print(f"  Accuracy: {random_consistency['accuracy']:.2f}")
    print(f"  Reward gap: {random_consistency['reward_gap']:.4f}")
    
    print("\nReward shaping and sanity checks demonstration completed!")


if __name__ == "__main__":
    demonstrate_reward_shaping()