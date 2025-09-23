"""
Example demonstrating logging and visualization for LLM training.
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
from utils.logger import TrainingLogger


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
    """Main function demonstrating logging and visualization."""
    print("Logging and Visualization Example")
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
    
    # Create logger
    print("Creating logger...")
    logger = TrainingLogger(log_dir="logs", experiment_name="logging_example")
    
    # Create trainer with logger
    print("Creating trainer with logger...")
    trainer = Trainer(model, optimizer, experiment_name="logging_example")
    trainer.set_logger(logger)
    
    # Example 1: Training with logging
    print("\nExample 1: Training with Logging")
    print("-" * 30)
    
    # Train with logging
    print("Training model with logging...")
    trainer.train(
        data_loader, 
        num_epochs=num_epochs, 
        log_interval=10
    )
    
    # Plot training curves
    print("\nGenerating training curves...")
    figures = logger.plot_training_curves()
    
    # Show what was saved
    print("\nLogs and figures saved:")
    print(f"  Log file: {logger.log_path}")
    print(f"  Metrics file: {logger.metrics_path}")
    print(f"  Figures directory: {logger.figures_dir}")
    
    if os.path.exists(logger.log_path):
        print("  Log file created successfully")
    
    if os.path.exists(logger.metrics_path):
        print("  Metrics file created successfully")
    
    if os.path.exists(logger.figures_dir):
        figures_saved = os.listdir(logger.figures_dir)
        if figures_saved:
            for figure in figures_saved:
                print(f"    - {figure}")
    
    # Close logger
    logger.close()
    
    print("\nLogging and visualization example completed!")


if __name__ == "__main__":
    main()