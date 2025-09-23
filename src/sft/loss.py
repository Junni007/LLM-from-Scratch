"""
Loss functions for Supervised Fine-Tuning (SFT).

This module implements specialized loss functions for SFT, including
causal language modeling loss with masked labels for instruction tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalLMLoss(nn.Module):
    """
    Causal Language Modeling loss for instruction tuning.
    
    This loss function computes the cross-entropy loss for next-token prediction
    with optional masking of instruction tokens.
    """
    
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        """
        Initialize CausalLMLoss.
        
        Args:
            ignore_index (int): Index to ignore in loss computation (default: -100)
            label_smoothing (float): Label smoothing factor (default: 0.0)
        """
        super(CausalLMLoss, self).__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute causal language modeling loss.
        
        Args:
            logits (torch.Tensor): Model logits of shape (batch_size, seq_len, vocab_size)
            labels (torch.Tensor): Target labels of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Reshape for cross-entropy loss
        # logits: (batch_size * seq_len, vocab_size)
        # labels: (batch_size * seq_len,)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Compute loss
        loss = self.cross_entropy(logits_flat, labels_flat)
        
        return loss


class InstructionLoss(nn.Module):
    """
    Specialized loss for instruction tuning.
    
    This loss function allows for masking of instruction tokens while
    computing loss only on response tokens.
    """
    
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0,
                 instruction_weight: float = 0.0):
        """
        Initialize InstructionLoss.
        
        Args:
            ignore_index (int): Index to ignore in loss computation (default: -100)
            label_smoothing (float): Label smoothing factor (default: 0.0)
            instruction_weight (float): Weight for instruction tokens in loss (default: 0.0)
        """
        super(InstructionLoss, self).__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.instruction_weight = instruction_weight
        
        # Main loss function
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction='none'  # We'll handle reduction manually
        )
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, 
                instruction_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute instruction-aware loss.
        
        Args:
            logits (torch.Tensor): Model logits of shape (batch_size, seq_len, vocab_size)
            labels (torch.Tensor): Target labels of shape (batch_size, seq_len)
            instruction_mask (torch.Tensor, optional): Mask for instruction tokens of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Reshape for cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Compute per-token loss
        per_token_loss = self.cross_entropy(logits_flat, labels_flat)  # (batch_size * seq_len,)
        
        # Reshape back to sequence format
        per_token_loss = per_token_loss.view(batch_size, seq_len)
        
        if instruction_mask is not None:
            # Apply instruction mask
            # instruction_mask: 1 for instruction tokens, 0 for response tokens
            response_mask = 1.0 - instruction_mask  # 0 for instruction, 1 for response
            
            # Apply different weights to instruction and response tokens
            weighted_loss = (
                instruction_mask * self.instruction_weight * per_token_loss +
                response_mask * per_token_loss
            )
            
            # Compute mean over non-ignored tokens
            valid_mask = (labels != self.ignore_index).float()
            total_valid_tokens = valid_mask.sum()
            
            if total_valid_tokens > 0:
                loss = (weighted_loss * valid_mask).sum() / total_valid_tokens
            else:
                loss = torch.tensor(0.0, device=logits.device)
        else:
            # Standard loss computation (ignore_index handles masking)
            valid_mask = (labels != self.ignore_index).float()
            total_valid_tokens = valid_mask.sum()
            
            if total_valid_tokens > 0:
                loss = (per_token_loss * valid_mask).sum() / total_valid_tokens
            else:
                loss = torch.tensor(0.0, device=logits.device)
        
        return loss


class WeightedInstructionLoss(nn.Module):
    """
    Weighted instruction loss that applies different weights to different parts of the sequence.
    
    This allows for fine-grained control over which parts of the sequence contribute more to the loss.
    """
    
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        """
        Initialize WeightedInstructionLoss.
        
        Args:
            ignore_index (int): Index to ignore in loss computation (default: -100)
            label_smoothing (float): Label smoothing factor (default: 0.0)
        """
        super(WeightedInstructionLoss, self).__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction='none'
        )
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, 
                token_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted instruction loss.
        
        Args:
            logits (torch.Tensor): Model logits of shape (batch_size, seq_len, vocab_size)
            labels (torch.Tensor): Target labels of shape (batch_size, seq_len)
            token_weights (torch.Tensor, optional): Weights for each token of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Reshape for cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Compute per-token loss
        per_token_loss = self.cross_entropy(logits_flat, labels_flat)  # (batch_size * seq_len,)
        
        # Reshape back to sequence format
        per_token_loss = per_token_loss.view(batch_size, seq_len)
        
        if token_weights is not None:
            # Apply token weights
            weighted_loss = per_token_loss * token_weights
            
            # Compute mean over non-ignored tokens
            valid_mask = (labels != self.ignore_index).float()
            total_valid_tokens = valid_mask.sum()
            
            if total_valid_tokens > 0:
                loss = (weighted_loss * valid_mask).sum() / total_valid_tokens
            else:
                loss = torch.tensor(0.0, device=logits.device)
        else:
            # Standard loss computation
            valid_mask = (labels != self.ignore_index).float()
            total_valid_tokens = valid_mask.sum()
            
            if total_valid_tokens > 0:
                loss = (per_token_loss * valid_mask).sum() / total_valid_tokens
            else:
                loss = torch.tensor(0.0, device=logits.device)
        
        return loss


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from loss.
    
    Args:
        loss (torch.Tensor): Computed loss
        
    Returns:
        torch.Tensor: Perplexity
    """
    return torch.exp(loss)


def test_sft_losses():
    """Test function for SFT loss functions."""
    batch_size = 2
    seq_len = 8
    vocab_size = 100
    
    # Create dummy logits and labels
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test CausalLMLoss
    print("Testing CausalLMLoss...")
    causal_loss_fn = CausalLMLoss()
    causal_loss = causal_loss_fn(logits, labels)
    print(f"  Causal LM Loss: {causal_loss.item():.4f}")
    
    # Test with ignored tokens
    labels_with_ignore = labels.clone()
    labels_with_ignore[:, :2] = -100  # Ignore first 2 tokens
    causal_loss_ignored = causal_loss_fn(logits, labels_with_ignore)
    print(f"  Causal LM Loss (with ignored tokens): {causal_loss_ignored.item():.4f}")
    
    # Test InstructionLoss
    print("\nTesting InstructionLoss...")
    instruction_loss_fn = InstructionLoss(instruction_weight=0.1)
    
    # Create instruction mask (1 for instruction tokens, 0 for response tokens)
    instruction_mask = torch.zeros(batch_size, seq_len)
    instruction_mask[:, :3] = 1.0  # First 3 tokens are instruction
    
    instruction_loss = instruction_loss_fn(logits, labels, instruction_mask)
    print(f"  Instruction Loss: {instruction_loss.item():.4f}")
    
    # Test without instruction mask
    instruction_loss_no_mask = instruction_loss_fn(logits, labels)
    print(f"  Instruction Loss (no mask): {instruction_loss_no_mask.item():.4f}")
    
    # Test WeightedInstructionLoss
    print("\nTesting WeightedInstructionLoss...")
    weighted_loss_fn = WeightedInstructionLoss()
    
    # Create token weights
    token_weights = torch.ones(batch_size, seq_len)
    token_weights[:, -2:] = 2.0  # Give more weight to last 2 tokens
    
    weighted_loss = weighted_loss_fn(logits, labels, token_weights)
    print(f"  Weighted Instruction Loss: {weighted_loss.item():.4f}")
    
    # Test perplexity computation
    print("\nTesting perplexity computation...")
    perplexity = compute_perplexity(causal_loss)
    print(f"  Perplexity: {perplexity.item():.4f}")
    
    print("SFT loss functions test passed!")


if __name__ == "__main__":
    test_sft_losses()