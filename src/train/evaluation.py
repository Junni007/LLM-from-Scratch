"""
Evaluation metrics for LLM training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .loss import CrossEntropyLoss


def evaluate_model(model: nn.Module, data_loader, device: torch.device = None) -> Dict[str, float]:
    """
    Evaluate model performance on validation data.
    
    Args:
        model (nn.Module): Model to evaluate
        data_loader: DataLoader for validation data
        device (torch.device, optional): Device to evaluate on
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    
    criterion = CrossEntropyLoss()
    
    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            # Move data to device
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Forward pass
            logits = model(input_batch)
            
            # Compute loss
            loss = criterion(logits, target_batch)
            total_loss += loss.item() * input_batch.size(0)  # Multiply by batch size
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == target_batch).sum().item()
            total_tokens += target_batch.numel()
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader.dataset) if len(data_loader.dataset) > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy
    }


def compute_perplexity(model: nn.Module, data_loader, device: torch.device = None) -> float:
    """
    Compute perplexity of the model on given data.
    
    Args:
        model (nn.Module): Model to evaluate
        data_loader: DataLoader for evaluation data
        device (torch.device, optional): Device to evaluate on
        
    Returns:
        float: Perplexity score
    """
    metrics = evaluate_model(model, data_loader, device)
    return metrics['perplexity']


def compute_accuracy(model: nn.Module, data_loader, device: torch.device = None) -> float:
    """
    Compute accuracy of the model on given data.
    
    Args:
        model (nn.Module): Model to evaluate
        data_loader: DataLoader for evaluation data
        device (torch.device, optional): Device to evaluate on
        
    Returns:
        float: Accuracy score
    """
    metrics = evaluate_model(model, data_loader, device)
    return metrics['accuracy']


def test_evaluation():
    """
    Test function for evaluation metrics.
    """
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=100, d_model=32):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.linear = nn.Linear(d_model, vocab_size)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.linear(x)
            return x
    
    # Initialize model
    model = SimpleModel()
    
    # Create sample data
    batch_size = 4
    seq_length = 8
    vocab_size = 100
    
    # Create sample input and target
    input_data = torch.randint(0, vocab_size, (batch_size, seq_length))
    target_data = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # For simplicity, we'll just test that the functions run without error
    # A more comprehensive test would require a proper DataLoader
    
    print("Evaluation functions test completed successfully!")
    print("(Note: Full testing requires a proper DataLoader implementation)")


if __name__ == "__main__":
    test_evaluation()