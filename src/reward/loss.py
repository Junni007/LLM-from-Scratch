"""
Loss functions for Reward Modeling in LLMs.

This module implements various loss functions used for training reward models,
including Bradley-Terry loss and margin ranking loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BradleyTerryLoss(nn.Module):
    """
    Bradley-Terry loss for pairwise preference learning.
    
    This loss function is based on the Bradley-Terry model, which models
    the probability that one item is preferred over another.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize BradleyTerryLoss.
        
        Args:
            epsilon (float): Small value to prevent log(0)
        """
        super(BradleyTerryLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute Bradley-Terry loss.
        
        Args:
            chosen_rewards (torch.Tensor): Rewards for chosen responses of shape (batch_size,)
            rejected_rewards (torch.Tensor): Rewards for rejected responses of shape (batch_size,)
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Compute the probability that chosen is preferred over rejected
        # P(chosen > rejected) = sigmoid(chosen_reward - rejected_reward)
        reward_diff = chosen_rewards - rejected_rewards
        prob_chosen = torch.sigmoid(reward_diff)
        
        # Clamp probabilities to prevent log(0)
        prob_chosen = torch.clamp(prob_chosen, self.epsilon, 1.0 - self.epsilon)
        
        # Compute negative log-likelihood
        # For preferred pairs, we want to maximize P(chosen > rejected)
        # So we minimize -log(P(chosen > rejected))
        loss = -torch.log(prob_chosen)
        
        # Return mean loss
        return loss.mean()


class MarginRankingLoss(nn.Module):
    """
    Margin ranking loss for pairwise preference learning.
    
    This loss function encourages a margin between chosen and rejected rewards.
    """
    
    def __init__(self, margin: float = 0.1):
        """
        Initialize MarginRankingLoss.
        
        Args:
            margin (float): Margin to enforce between chosen and rejected rewards
        """
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
    
    def forward(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute margin ranking loss.
        
        Args:
            chosen_rewards (torch.Tensor): Rewards for chosen responses of shape (batch_size,)
            rejected_rewards (torch.Tensor): Rewards for rejected responses of shape (batch_size,)
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Compute margin loss
        # We want chosen_rewards > rejected_rewards + margin
        # So we minimize max(0, margin - (chosen_rewards - rejected_rewards))
        reward_diff = chosen_rewards - rejected_rewards
        loss = F.relu(self.margin - reward_diff)
        
        # Return mean loss
        return loss.mean()


class HingeLoss(nn.Module):
    """
    Hinge loss for pairwise preference learning.
    
    Similar to margin ranking loss but with a different formulation.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize HingeLoss.
        
        Args:
            margin (float): Margin to enforce between chosen and rejected rewards
        """
        super(HingeLoss, self).__init__()
        self.margin = margin
    
    def forward(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute hinge loss.
        
        Args:
            chosen_rewards (torch.Tensor): Rewards for chosen responses of shape (batch_size,)
            rejected_rewards (torch.Tensor): Rewards for rejected responses of shape (batch_size,)
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Compute hinge loss
        # We want chosen_rewards > rejected_rewards
        # So we minimize max(0, margin - (chosen_rewards - rejected_rewards))
        reward_diff = chosen_rewards - rejected_rewards
        loss = F.relu(self.margin - reward_diff)
        
        # Return mean loss
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for pairwise preference learning.
    
    This loss function combines similarity and dissimilarity terms.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize ContrastiveLoss.
        
        Args:
            margin (float): Margin for negative pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            chosen_rewards (torch.Tensor): Rewards for chosen responses of shape (batch_size,)
            rejected_rewards (torch.Tensor): Rewards for rejected responses of shape (batch_size,)
            
        Returns:
            torch.Tensor: Computed loss
        """
        # For positive pairs (similar), minimize distance squared
        # For negative pairs (dissimilar), enforce margin
        reward_diff = chosen_rewards - rejected_rewards
        
        # Since we only have negative pairs (chosen preferred over rejected),
        # we enforce that the difference is positive with a margin
        loss = F.relu(self.margin - reward_diff) ** 2
        
        # Return mean loss
        return loss.mean()


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that combines multiple loss components.
    
    This loss function adaptively weights different components based on training progress.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """
        Initialize AdaptiveLoss.
        
        Args:
            alpha (float): Weight for Bradley-Terry component
            beta (float): Weight for margin component
        """
        super(AdaptiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bt_loss = BradleyTerryLoss()
        self.margin_loss = MarginRankingLoss()
    
    def forward(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive loss.
        
        Args:
            chosen_rewards (torch.Tensor): Rewards for chosen responses of shape (batch_size,)
            rejected_rewards (torch.Tensor): Rewards for rejected responses of shape (batch_size,)
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Compute component losses
        bt_loss = self.bt_loss(chosen_rewards, rejected_rewards)
        margin_loss = self.margin_loss(chosen_rewards, rejected_rewards)
        
        # Combine losses
        total_loss = self.alpha * bt_loss + self.beta * margin_loss
        
        return total_loss


def compute_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> float:
    """
    Compute accuracy of reward model predictions.
    
    Args:
        chosen_rewards (torch.Tensor): Rewards for chosen responses
        rejected_rewards (torch.Tensor): Rewards for rejected responses
        
    Returns:
        float: Accuracy (proportion of correct predictions)
    """
    # A prediction is correct if chosen reward > rejected reward
    correct_predictions = (chosen_rewards > rejected_rewards).float()
    accuracy = correct_predictions.mean().item()
    
    return accuracy


def compute_reward_gap(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> float:
    """
    Compute average reward gap between chosen and rejected responses.
    
    Args:
        chosen_rewards (torch.Tensor): Rewards for chosen responses
        rejected_rewards (torch.Tensor): Rewards for rejected responses
        
    Returns:
        float: Average reward gap
    """
    reward_gap = (chosen_rewards - rejected_rewards).mean().item()
    
    return reward_gap


def demonstrate_reward_losses():
    """Demonstrate reward loss functions."""
    print("Reward Loss Functions Demonstration")
    print("=" * 35)
    
    # Create sample rewards
    torch.manual_seed(42)
    batch_size = 4
    
    # Perfect predictions (chosen > rejected)
    chosen_rewards_perfect = torch.tensor([2.0, 3.0, 1.5, 2.5])
    rejected_rewards_perfect = torch.tensor([1.0, 2.0, 0.5, 1.5])
    
    # Imperfect predictions
    chosen_rewards_imperfect = torch.tensor([1.0, 2.0, 0.5, 1.5])
    rejected_rewards_imperfect = torch.tensor([2.0, 1.0, 1.5, 2.5])
    
    print("1. Perfect Predictions (chosen > rejected):")
    print(f"  Chosen rewards:  {chosen_rewards_perfect.tolist()}")
    print(f"  Rejected rewards: {rejected_rewards_perfect.tolist()}")
    
    print("\n2. Imperfect Predictions:")
    print(f"  Chosen rewards:  {chosen_rewards_imperfect.tolist()}")
    print(f"  Rejected rewards: {rejected_rewards_imperfect.tolist()}")
    
    # Test Bradley-Terry Loss
    print("\n3. Bradley-Terry Loss:")
    bt_loss_fn = BradleyTerryLoss()
    
    bt_loss_perfect = bt_loss_fn(chosen_rewards_perfect, rejected_rewards_perfect)
    bt_loss_imperfect = bt_loss_fn(chosen_rewards_imperfect, rejected_rewards_imperfect)
    
    print(f"  Perfect predictions loss: {bt_loss_perfect.item():.4f}")
    print(f"  Imperfect predictions loss: {bt_loss_imperfect.item():.4f}")
    
    # Test Margin Ranking Loss
    print("\n4. Margin Ranking Loss:")
    margin_loss_fn = MarginRankingLoss(margin=0.1)
    
    margin_loss_perfect = margin_loss_fn(chosen_rewards_perfect, rejected_rewards_perfect)
    margin_loss_imperfect = margin_loss_fn(chosen_rewards_imperfect, rejected_rewards_imperfect)
    
    print(f"  Perfect predictions loss: {margin_loss_perfect.item():.4f}")
    print(f"  Imperfect predictions loss: {margin_loss_imperfect.item():.4f}")
    
    # Test Hinge Loss
    print("\n5. Hinge Loss:")
    hinge_loss_fn = HingeLoss(margin=1.0)
    
    hinge_loss_perfect = hinge_loss_fn(chosen_rewards_perfect, rejected_rewards_perfect)
    hinge_loss_imperfect = hinge_loss_fn(chosen_rewards_imperfect, rejected_rewards_imperfect)
    
    print(f"  Perfect predictions loss: {hinge_loss_perfect.item():.4f}")
    print(f"  Imperfect predictions loss: {hinge_loss_imperfect.item():.4f}")
    
    # Test Contrastive Loss
    print("\n6. Contrastive Loss:")
    contrastive_loss_fn = ContrastiveLoss(margin=1.0)
    
    contrastive_loss_perfect = contrastive_loss_fn(chosen_rewards_perfect, rejected_rewards_perfect)
    contrastive_loss_imperfect = contrastive_loss_fn(chosen_rewards_imperfect, rejected_rewards_imperfect)
    
    print(f"  Perfect predictions loss: {contrastive_loss_perfect.item():.4f}")
    print(f"  Imperfect predictions loss: {contrastive_loss_imperfect.item():.4f}")
    
    # Test Adaptive Loss
    print("\n7. Adaptive Loss:")
    adaptive_loss_fn = AdaptiveLoss(alpha=0.5, beta=0.5)
    
    adaptive_loss_perfect = adaptive_loss_fn(chosen_rewards_perfect, rejected_rewards_perfect)
    adaptive_loss_imperfect = adaptive_loss_fn(chosen_rewards_imperfect, rejected_rewards_imperfect)
    
    print(f"  Perfect predictions loss: {adaptive_loss_perfect.item():.4f}")
    print(f"  Imperfect predictions loss: {adaptive_loss_imperfect.item():.4f}")
    
    # Test metrics
    print("\n8. Evaluation Metrics:")
    
    # Accuracy
    accuracy_perfect = compute_accuracy(chosen_rewards_perfect, rejected_rewards_perfect)
    accuracy_imperfect = compute_accuracy(chosen_rewards_imperfect, rejected_rewards_imperfect)
    
    print(f"  Perfect predictions accuracy: {accuracy_perfect:.2f}")
    print(f"  Imperfect predictions accuracy: {accuracy_imperfect:.2f}")
    
    # Reward gap
    gap_perfect = compute_reward_gap(chosen_rewards_perfect, rejected_rewards_perfect)
    gap_imperfect = compute_reward_gap(chosen_rewards_imperfect, rejected_rewards_imperfect)
    
    print(f"  Perfect predictions reward gap: {gap_perfect:.4f}")
    print(f"  Imperfect predictions reward gap: {gap_imperfect:.4f}")
    
    print("\nReward loss functions demonstration completed!")


if __name__ == "__main__":
    demonstrate_reward_losses()