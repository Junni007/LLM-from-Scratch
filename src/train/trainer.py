"""
Training loop implementation for LLM training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Optional
from .loss import CrossEntropyLoss
import time
import os
import sys


class Trainer:
    """
    Trainer class for training LLMs.
    
    This class handles the training loop, including forward/backward passes,
    optimization steps, and logging.
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, 
                 device: torch.device = None, experiment_name: str = "llm_experiment"):
        """
        Initialize Trainer.
        
        Args:
            model (nn.Module): Model to train
            optimizer (optim.Optimizer): Optimizer to use
            device (torch.device, optional): Device to train on (default: cuda if available)
            experiment_name (str): Name of the experiment for logging
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
        
        # Learning rate scheduler
        self.scheduler = None
        
        # Checkpoint information
        self.current_epoch = 0
        self.total_steps = 0
        
        # Logger
        self.experiment_name = experiment_name
        self.logger = None
    
    def set_scheduler(self, scheduler):
        """
        Set learning rate scheduler.
        
        Args:
            scheduler: Learning rate scheduler to use
        """
        self.scheduler = scheduler
    
    def set_logger(self, logger):
        """
        Set training logger.
        
        Args:
            logger: Training logger to use
        """
        self.logger = logger
    
    def save_checkpoint(self, filepath: str, epoch: int = None, **kwargs):
        """
        Save training checkpoint.
        
        Args:
            filepath (str): Path to save checkpoint
            epoch (int, optional): Current epoch number
            **kwargs: Additional information to save
        """
        checkpoint = {
            'epoch': epoch if epoch is not None else self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_perplexities': self.train_perplexities,
            'total_steps': self.total_steps,
            **kwargs
        }
        
        # Add scheduler state if it exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        if self.logger:
            self.logger.logger.info(f"Checkpoint saved to {filepath}")
        else:
            print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load training checkpoint.
        
        Args:
            filepath (str): Path to load checkpoint from
            
        Returns:
            dict: Additional information saved in checkpoint
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_perplexities = checkpoint.get('train_perplexities', [])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        
        # Load scheduler state if it exists
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.logger:
            self.logger.logger.info(f"Checkpoint loaded from {filepath}")
            self.logger.logger.info(f"Resuming from epoch {self.current_epoch}")
        else:
            print(f"Checkpoint loaded from {filepath}")
            print(f"Resuming from epoch {self.current_epoch}")
        
        # Return additional information
        additional_info = {k: v for k, v in checkpoint.items() 
                          if k not in ['model_state_dict', 'optimizer_state_dict', 
                                      'train_losses', 'train_perplexities', 
                                      'epoch', 'total_steps', 'scheduler_state_dict']}
        return additional_info
    
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
        
        # Update learning rate if scheduler is set
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Increment step counter
        self.total_steps += 1
        
        # Return loss value
        return loss.item()
    
    def train_step_with_gradient_accumulation(self, data_loader, accumulation_steps: int = 1) -> float:
        """
        Perform a training step with gradient accumulation.
        
        Args:
            data_loader: DataLoader that yields batches
            accumulation_steps (int): Number of steps to accumulate gradients over
            
        Returns:
            float: Average loss value for this step
        """
        # Zero gradients at the start
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        
        # Accumulate gradients over multiple batches
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= accumulation_steps:
                break
                
            # Move data to device
            input_batch = input_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            
            # Forward pass
            logits = self.model(input_batch)
            
            # Compute loss (normalized by accumulation steps)
            loss = self.criterion(logits, target_batch) / accumulation_steps
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Update learning rate if scheduler is set
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Increment step counter
        self.total_steps += 1
        
        return total_loss
    
    def train_step_mixed_precision(self, input_batch: torch.Tensor, target_batch: torch.Tensor,
                                  scaler: torch.cuda.amp.GradScaler = None) -> float:
        """
        Perform a single training step with mixed precision.
        
        Args:
            input_batch (torch.Tensor): Input batch of shape (batch_size, seq_len)
            target_batch (torch.Tensor): Target batch of shape (batch_size, seq_len)
            scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision
            
        Returns:
            float: Loss value for this step
        """
        # Move data to device
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        if scaler is not None:
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                logits = self.model(input_batch)
                loss = self.criterion(logits, target_batch)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            
            # Update parameters with unscaling
            scaler.step(self.optimizer)
            scaler.update()
        else:
            # Standard training step
            logits = self.model(input_batch)
            loss = self.criterion(logits, target_batch)
            loss.backward()
            self.optimizer.step()
        
        # Update learning rate if scheduler is set
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Increment step counter
        self.total_steps += 1
        
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
                
                # Get current learning rate
                lr = None
                if self.scheduler is not None:
                    lr = self.scheduler.get_last_lr()[0]
                elif self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                    lr = self.optimizer.param_groups[0]['lr']
                
                # Log training step
                if self.logger:
                    self.logger.log_training_step(
                        self.current_epoch, batch_idx + 1, avg_loss, perplexity, lr
                    )
                else:
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
    
    def train_epoch_with_gradient_accumulation(self, data_loader, accumulation_steps: int = 4,
                                             log_interval: int = 100) -> float:
        """
        Train for one epoch with gradient accumulation.
        
        Args:
            data_loader: DataLoader for training data
            accumulation_steps (int): Number of steps to accumulate gradients over
            log_interval (int): How often to print training progress
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        # Create iterator from data loader
        data_iter = iter(data_loader)
        
        for batch_idx in range(0, len(data_loader), accumulation_steps):
            try:
                # Perform training step with gradient accumulation
                loss = self.train_step_with_gradient_accumulation(
                    data_iter, accumulation_steps)
                
                total_loss += loss
                num_batches += 1
                
                # Log progress
                if log_interval > 0 and (num_batches) % log_interval == 0:
                    avg_loss = total_loss / num_batches
                    perplexity = torch.exp(torch.tensor(avg_loss)).item()
                    
                    # Get current learning rate
                    lr = None
                    if self.scheduler is not None:
                        lr = self.scheduler.get_last_lr()[0]
                    elif self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                        lr = self.optimizer.param_groups[0]['lr']
                    
                    # Log training step
                    if self.logger:
                        self.logger.log_training_step(
                            self.current_epoch, batch_idx + 1, avg_loss, perplexity, lr
                        )
                    else:
                        elapsed_time = time.time() - start_time
                        print(f"Batch {batch_idx + 1:5d} | "
                              f"Loss: {avg_loss:6.3f} | "
                              f"Perplexity: {perplexity:6.2f} | "
                              f"Time: {elapsed_time:6.2f}s")
            except StopIteration:
                break
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Store statistics
        self.train_losses.append(avg_loss)
        self.train_perplexities.append(perplexity)
        
        return avg_loss
    
    def train_epoch_mixed_precision(self, data_loader, scaler: torch.cuda.amp.GradScaler = None,
                                  log_interval: int = 100) -> float:
        """
        Train for one epoch with mixed precision.
        
        Args:
            data_loader: DataLoader for training data
            scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision
            log_interval (int): How often to print training progress
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            # Perform training step with mixed precision
            loss = self.train_step_mixed_precision(input_batch, target_batch, scaler)
            
            total_loss += loss
            num_batches += 1
            
            # Log progress
            if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                
                # Get current learning rate
                lr = None
                if self.scheduler is not None:
                    lr = self.scheduler.get_last_lr()[0]
                elif self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                    lr = self.optimizer.param_groups[0]['lr']
                
                # Log training step
                if self.logger:
                    self.logger.log_training_step(
                        self.current_epoch, batch_idx + 1, avg_loss, perplexity, lr
                    )
                else:
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
    
    def train(self, data_loader, num_epochs: int, log_interval: int = 100, 
              checkpoint_interval: int = 0, checkpoint_dir: str = "checkpoints"):
        """
        Train the model for multiple epochs.
        
        Args:
            data_loader: DataLoader for training data
            num_epochs (int): Number of epochs to train
            log_interval (int): How often to print training progress
            checkpoint_interval (int): Save checkpoint every N epochs (0 to disable)
            checkpoint_dir (str): Directory to save checkpoints
        """
        if self.logger:
            self.logger.logger.info(f"Starting training on {self.device}")
            self.logger.logger.info(f"Number of epochs: {num_epochs}")
            self.logger.logger.info(f"Number of batches per epoch: {len(data_loader)}")
        else:
            print(f"Starting training on {self.device}")
            print(f"Number of epochs: {num_epochs}")
            print(f"Number of batches per epoch: {len(data_loader)}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.current_epoch = epoch
            if self.logger:
                self.logger.logger.info(f"\nEpoch {epoch + 1}/{self.current_epoch + num_epochs}")
                self.logger.logger.info("-" * 50)
            else:
                print(f"\nEpoch {epoch + 1}/{self.current_epoch + num_epochs}")
                print("-" * 50)
            
            # Train for one epoch
            epoch_loss = self.train_epoch(data_loader, log_interval)
            epoch_perplexity = torch.exp(torch.tensor(epoch_loss)).item()
            
            # Get current learning rate
            lr = None
            if self.scheduler is not None:
                lr = self.scheduler.get_last_lr()[0]
            elif self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch summary
            if self.logger:
                self.logger.log_epoch(
                    epoch + 1, epoch_loss, epoch_perplexity, learning_rate=lr
                )
            else:
                print(f"Epoch {epoch + 1} completed | "
                      f"Average Loss: {epoch_loss:6.3f} | "
                      f"Perplexity: {epoch_perplexity:6.2f}")
            
            # Save checkpoint if interval is set
            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                self.save_checkpoint(checkpoint_path, epoch=epoch + 1)
        
        total_time = time.time() - start_time
        if self.logger:
            self.logger.logger.info(f"\nTraining completed in {total_time:.2f} seconds")
        else:
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
    
    # Test scheduler functionality
    try:
        from .lr_scheduler import LinearWarmupScheduler
        scheduler = LinearWarmupScheduler(optimizer, warmup_steps=10)
        trainer.set_scheduler(scheduler)
        assert trainer.scheduler == scheduler
    except ImportError:
        # If lr_scheduler is not available, skip this test
        pass
    
    print("Trainer test passed!")


if __name__ == "__main__":
    test_trainer()