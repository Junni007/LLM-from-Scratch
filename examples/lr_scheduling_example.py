"""
Example demonstrating learning rate scheduling for LLM training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.transformer import TransformerBlock
from train.trainer import Trainer
from train.lr_scheduler import LinearWarmupScheduler, CosineDecayScheduler, WarmupDecayScheduler


class TinyLLM(nn.Module):
    """
    A tiny LLM implementation using our Transformer blocks.
    """
    
    def __init__(self, vocab_size=256, d_model=64, num_heads=4, num_layers=2, d_ff=128):
        super(TinyLLM, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Apply embedding
        x = self.embedding(x)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Apply output projection
        logits = self.output_projection(x)
        
        return logits


def create_sample_data(vocab_size=1000, batch_size=32, seq_len=64, num_samples=1000):
    """Create sample data for training."""
    # Create random data
    data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))
    
    # Split into input and target
    input_data = data[:, :-1]  # All tokens except the last
    target_data = data[:, 1:]  # All tokens except the first (shifted by one)
    
    # Create dataset and dataloader
    dataset = TensorDataset(input_data, target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def main():
    """Main function demonstrating learning rate scheduling."""
    print("Learning Rate Scheduling Example")
    print("=" * 40)
    
    # Model parameters
    vocab_size = 1000
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 512
    max_seq_length = 64
    dropout = 0.1
    
    # Training parameters
    batch_size = 32
    num_epochs = 3
    learning_rate = 3e-4
    
    # Create model
    print("Creating model...")
    model = TinyLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=nhead,
        num_layers=num_layers,
        d_ff=dim_feedforward
    )
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create sample data
    print("Creating sample data...")
    data_loader = create_sample_data(
        vocab_size=vocab_size,
        batch_size=batch_size,
        seq_len=max_seq_length,
        num_samples=1000
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(model, optimizer)
    
    # Example 1: Linear warmup scheduler
    print("\nExample 1: Linear Warmup Scheduler")
    print("-" * 30)
    
    # Reset model and optimizer for fair comparison
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer)
    
    # Create linear warmup scheduler
    warmup_steps = 50
    scheduler = LinearWarmupScheduler(optimizer, warmup_steps=warmup_steps)
    trainer.set_scheduler(scheduler)
    
    print(f"Warmup steps: {warmup_steps}")
    print("Initial learning rate:", optimizer.param_groups[0]['lr'])
    
    # Train for a few steps to see the effect
    for i in range(10):
        # Get a batch of data
        input_batch, target_batch = next(iter(data_loader))
        
        # Perform training step
        loss = trainer.train_step(input_batch, target_batch)
        
        print(f"Step {i+1:2d}: Loss = {loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
    
    # Example 2: Cosine decay scheduler
    print("\nExample 2: Cosine Decay Scheduler")
    print("-" * 30)
    
    # Reset model and optimizer
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer)
    
    # Create cosine decay scheduler
    total_steps = len(data_loader) * num_epochs
    scheduler = CosineDecayScheduler(optimizer, total_steps=total_steps, min_lr=1e-5)
    trainer.set_scheduler(scheduler)
    
    print(f"Total steps: {total_steps}")
    print("Initial learning rate:", optimizer.param_groups[0]['lr'])
    
    # Train for a few steps to see the effect
    for i in range(10):
        # Get a batch of data
        input_batch, target_batch = next(iter(data_loader))
        
        # Perform training step
        loss = trainer.train_step(input_batch, target_batch)
        
        print(f"Step {i+1:2d}: Loss = {loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
    
    # Example 3: Warmup + decay scheduler
    print("\nExample 3: Warmup + Cosine Decay Scheduler")
    print("-" * 30)
    
    # Reset model and optimizer
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer)
    
    # Create warmup + decay scheduler
    warmup_steps = 50
    total_steps = len(data_loader) * num_epochs
    scheduler = WarmupDecayScheduler(
        optimizer, 
        warmup_steps=warmup_steps, 
        total_steps=total_steps, 
        decay_type='cosine',
        min_lr=1e-5
    )
    trainer.set_scheduler(scheduler)
    
    print(f"Warmup steps: {warmup_steps}")
    print(f"Total steps: {total_steps}")
    print("Initial learning rate:", optimizer.param_groups[0]['lr'])
    
    # Train for a few steps to see the effect
    for i in range(15):
        # Get a batch of data
        input_batch, target_batch = next(iter(data_loader))
        
        # Perform training step
        loss = trainer.train_step(input_batch, target_batch)
        
        print(f"Step {i+1:2d}: Loss = {loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
    
    # Example 4: Training with scheduler for full epochs
    print("\nExample 4: Full Training with Warmup + Decay Scheduler")
    print("-" * 50)
    
    # Reset model and optimizer
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer)
    
    # Create warmup + decay scheduler
    warmup_steps = 50
    total_steps = len(data_loader) * num_epochs
    scheduler = WarmupDecayScheduler(
        optimizer, 
        warmup_steps=warmup_steps, 
        total_steps=total_steps, 
        decay_type='cosine',
        min_lr=1e-5
    )
    trainer.set_scheduler(scheduler)
    
    # Train for full epochs
    trainer.train(data_loader, num_epochs=num_epochs, log_interval=50)
    
    print("\nLearning rate scheduling example completed!")


if __name__ == "__main__":
    main()