"""
Loss functions for LLM training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for language modeling.
    
    This implementation handles the specific requirements of language modeling,
    including optional label shifting and masking.
    """
    
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        """
        Initialize CrossEntropyLoss.
        
        Args:
            ignore_index (int): Index to ignore in loss computation (default: -100)
            label_smoothing (float): Label smoothing factor (default: 0.0)
        """
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, 
                                          label_smoothing=label_smoothing)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits (torch.Tensor): Model logits of shape (batch_size, seq_len, vocab_size)
            targets (torch.Tensor): Target tokens of shape (batch_size, seq_len)
            mask (torch.Tensor, optional): Mask to apply to loss computation
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Reshape logits and targets for cross-entropy loss
        # logits: (batch_size * seq_len, vocab_size)
        # targets: (batch_size * seq_len)
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(batch_size * seq_len, vocab_size)
        targets = targets.view(batch_size * seq_len)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.view(batch_size * seq_len)
            # Set targets to ignore_index where mask is False
            targets = targets.masked_fill(~mask, self.ignore_index)
        
        # Compute loss
        loss = self.loss_fn(logits, targets)
        return loss


def shift_labels(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Shift labels for next-token prediction.
    
    In language modeling, we predict the next token, so we shift labels by one position.
    The last token in each sequence has no target, so it's marked with ignore_index.
    
    Args:
        labels (torch.Tensor): Input labels of shape (batch_size, seq_len)
        ignore_index (int): Index to use for positions with no target
        
    Returns:
        torch.Tensor: Shifted labels of shape (batch_size, seq_len)
    """
    # Shift labels to the left by one position
    shifted_labels = labels.new_zeros(labels.shape)
    shifted_labels[:, :-1] = labels[:, 1:]  # Shift left
    shifted_labels[:, -1] = ignore_index    # Mark last position as ignored
    
    return shifted_labels


def test_cross_entropy_loss():
    """
    Test function for CrossEntropyLoss.
    """
    batch_size = 2
    seq_len = 4
    vocab_size = 10
    
    # Create random logits and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Initialize loss function
    loss_fn = CrossEntropyLoss()
    
    # Compute loss
    loss = loss_fn(logits, targets)
    
    # Check that loss is a scalar
    assert loss.dim() == 0, f"Loss should be a scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    print("CrossEntropyLoss test passed!")


def test_shift_labels():
    """
    Test function for shift_labels.
    """
    batch_size = 2
    seq_len = 4
    ignore_index = -100
    
    # Create sample labels
    labels = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ], dtype=torch.long)
    
    # Shift labels
    shifted_labels = shift_labels(labels, ignore_index)
    
    # Check shifted labels
    expected_shifted = torch.tensor([
        [2, 3, 4, -100],
        [6, 7, 8, -100]
    ], dtype=torch.long)
    
    assert torch.equal(shifted_labels, expected_shifted), \
        f"Label shifting failed: {shifted_labels} != {expected_shifted}"
    
    print("shift_labels test passed!")


if __name__ == "__main__":
    test_cross_entropy_loss()
    test_shift_labels()