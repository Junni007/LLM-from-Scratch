"""
Training loop implementation for LLM training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Optional
from .loss import CrossEntropyLoss
import time


class Trainer:
    """
    Trainer class for training LLMs.
    
    This class handles the training loop, including forward/backward passes,
    optimization steps, and logging.
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, 
                 device: torch.device = None):
        """
        Initialize Trainer.
        
        Args:
            model (nn.Module): Model to train
            optimizer (optim.Optimizer): Optimizer to use
            device (torch.device, optional): Device to train on (default: cuda if available)
        """
        self.model = model
        self.optimizer = optimizer
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        
        # Loss function
        self.criterion = CrossEntropyLoss()
        
        # Training statistics
        self.train_losses = []
        self.train_perplexities = []
    
    def train_step(self, input_batch: torch.Tensor, target_batch: torch.Tensor) -> float:
        """
        Perform a single training step.
        
        Args:
            input_batch (torch.Tensor): Input batch of shape (batch_size, seq_len)
            target_batch (torch.Tensor): Target batch of shape (batch_size, seq_len)
            
        Returns:
            float: Loss value for this step
        """
        # Move data to device
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(input_batch)
        
        # Compute loss
        loss = self.criterion(logits, target_batch)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Return loss value
        return loss.item()
    
    def train_epoch(self, data_loader, log_interval: int = 100) -> float:
        """
        Train for one epoch.
        
        Args:
            data_loader: DataLoader for training data
            log_interval (int): How often to print training progress
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            # Perform training step
            loss = self.train_step(input_batch, target_batch)
            
            total_loss += loss
            num_batches += 1
            
            # Log progress
            if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                
                elapsed_time = time.time() - start_time
                print(f"Batch {batch_idx + 1:5d} | "
                      f"Loss: {avg_loss:6.3f} | "
                      f"Perplexity: {perplexity:6.2f} | "
                      f"Time: {elapsed_time:6.2f}s")
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Store statistics
        self.train_losses.append(avg_loss)
        self.train_perplexities.append(perplexity)
        
        return avg_loss
    
    def train(self, data_loader, num_epochs: int, log_interval: int = 100):
        """
        Train the model for multiple epochs.
        
        Args:
            data_loader: DataLoader for training data
            num_epochs (int): Number of epochs to train
            log_interval (int): How often to print training progress
        """
        print(f"Starting training on {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Number of batches per epoch: {len(data_loader)}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train for one epoch
            epoch_loss = self.train_epoch(data_loader, log_interval)
            epoch_perplexity = torch.exp(torch.tensor(epoch_loss)).item()
            
            print(f"Epoch {epoch + 1} completed | "
                  f"Average Loss: {epoch_loss:6.3f} | "
                  f"Perplexity: {epoch_perplexity:6.2f}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")


def test_trainer():
    """
    Test function for Trainer.
    """
    # This is a basic test that just checks the Trainer can be instantiated
    # A more comprehensive test would require a model, but we'll do that later
    
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
    
    # Initialize components
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize trainer
    trainer = Trainer(model, optimizer)
    
    # Check that trainer was created successfully
    assert trainer.model == model
    assert trainer.optimizer == optimizer
    assert isinstance(trainer.criterion, CrossEntropyLoss)
    
    print("Trainer test passed!")


if __name__ == "__main__":
    test_trainer()