"""
Example demonstrating checkpointing and resuming for LLM training.
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
    """Main function demonstrating checkpointing and resuming."""
    print("Checkpointing and Resuming Example")
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
    num_epochs = 5
    learning_rate = 3e-4
    checkpoint_interval = 2  # Save checkpoint every 2 epochs
    
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
    
    # Example 1: Training with checkpointing
    print("\nExample 1: Training with Checkpointing")
    print("-" * 30)
    
    # Train with checkpointing
    print("Training model with checkpointing...")
    trainer.train(
        data_loader, 
        num_epochs=num_epochs, 
        log_interval=10,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir="checkpoints"
    )
    
    # Show what was saved
    print("\nCheckpoints saved:")
    if os.path.exists("checkpoints"):
        for file in os.listdir("checkpoints"):
            print(f"  - {file}")
    
    # Example 2: Resuming from checkpoint
    print("\nExample 2: Resuming from Checkpoint")
    print("-" * 30)
    
    # Create a new model and trainer to simulate starting fresh
    print("Creating new model and trainer...")
    new_model = TinyLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=nhead,
        num_layers=num_layers,
        d_ff=dim_feedforward
    )
    new_optimizer = optim.AdamW(new_model.parameters(), lr=learning_rate)
    new_trainer = Trainer(new_model, new_optimizer)
    
    # Show training stats before loading checkpoint
    print(f"Before loading checkpoint:")
    print(f"  Current epoch: {new_trainer.current_epoch}")
    print(f"  Total steps: {new_trainer.total_steps}")
    print(f"  Train losses: {len(new_trainer.train_losses)}")
    
    # Load the latest checkpoint
    checkpoint_path = "checkpoints/checkpoint_epoch_4.pt"  # We saved at epoch 2 and 4
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Show training stats after loading checkpoint
        print(f"After loading checkpoint:")
        print(f"  Current epoch: {new_trainer.current_epoch}")
        print(f"  Total steps: {new_trainer.total_steps}")
        print(f"  Train losses: {len(new_trainer.train_losses)}")
        if new_trainer.train_losses:
            print(f"  Last loss: {new_trainer.train_losses[-1]:.4f}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
    
    # Continue training from where we left off
    print("\nContinuing training from checkpoint...")
    remaining_epochs = num_epochs - new_trainer.current_epoch
    if remaining_epochs > 0:
        new_trainer.train(
            data_loader,
            num_epochs=remaining_epochs,
            log_interval=10
        )
    
    print("\nCheckpointing and resuming example completed!")


if __name__ == "__main__":
    main()