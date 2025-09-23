"""
Logging utilities for LLM training.
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, Union
import json
import torch

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class TrainingLogger:
    """
    Logger for training metrics and visualization.
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "experiment"):
        """
        Initialize the logger.
        
        Args:
            log_dir (str): Directory to save logs
            experiment_name (str): Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_path = os.path.join(log_dir, f"{experiment_name}.log")
        self.metrics_path = os.path.join(log_dir, f"{experiment_name}_metrics.json")
        self.figures_dir = os.path.join(log_dir, "figures")
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(f"{experiment_name}_logger")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize metrics storage
        self.metrics = {
            'train_losses': [],
            'train_perplexities': [],
            'val_losses': [],
            'val_perplexities': [],
            'learning_rates': [],
            'timestamps': []
        }
    
    def log_training_step(self, epoch: int, batch: int, loss: float, 
                         perplexity: float, learning_rate: float = None):
        """
        Log training step metrics.
        
        Args:
            epoch (int): Current epoch
            batch (int): Current batch
            loss (float): Training loss
            perplexity (float): Training perplexity
            learning_rate (float, optional): Current learning rate
        """
        timestamp = datetime.now().isoformat()
        message = (f"Epoch {epoch}, Batch {batch} - "
                  f"Loss: {loss:.4f}, Perplexity: {perplexity:.4f}")
        
        if learning_rate is not None:
            message += f", LR: {learning_rate:.6f}"
            
        self.logger.info(message)
        
        # Store metrics
        self.metrics['train_losses'].append(loss)
        self.metrics['train_perplexities'].append(perplexity)
        self.metrics['timestamps'].append(timestamp)
        if learning_rate is not None:
            self.metrics['learning_rates'].append(learning_rate)
    
    def log_validation_step(self, epoch: int, val_loss: float, val_perplexity: float):
        """
        Log validation metrics.
        
        Args:
            epoch (int): Current epoch
            val_loss (float): Validation loss
            val_perplexity (float): Validation perplexity
        """
        message = (f"Epoch {epoch} Validation - "
                  f"Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}")
        self.logger.info(message)
        
        # Store metrics
        self.metrics['val_losses'].append(val_loss)
        self.metrics['val_perplexities'].append(val_perplexity)
    
    def log_epoch(self, epoch: int, avg_train_loss: float, avg_train_perplexity: float,
                  avg_val_loss: float = None, avg_val_perplexity: float = None,
                  learning_rate: float = None):
        """
        Log epoch metrics.
        
        Args:
            epoch (int): Current epoch
            avg_train_loss (float): Average training loss
            avg_train_perplexity (float): Average training perplexity
            avg_val_loss (float, optional): Average validation loss
            avg_val_perplexity (float, optional): Average validation perplexity
            learning_rate (float, optional): Current learning rate
        """
        message = (f"Epoch {epoch} Summary - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Perplexity: {avg_train_perplexity:.4f}")
        
        if avg_val_loss is not None and avg_val_perplexity is not None:
            message += f", Val Loss: {avg_val_loss:.4f}, Val Perplexity: {avg_val_perplexity:.4f}"
            
        if learning_rate is not None:
            message += f", LR: {learning_rate:.6f}"
            
        self.logger.info(message)
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {self.metrics_path}")
    
    def plot_training_curves(self, save_fig: bool = True) -> Dict[str, Any]:
        """
        Plot training curves.
        
        Args:
            save_fig (bool): Whether to save figures to files
            
        Returns:
            Dict[str, Any]: Dictionary of figures (empty if matplotlib not available)
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping plotting")
            return {}
        
        figures = {}
        
        # Plot training loss
        if self.metrics['train_losses']:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.metrics['train_losses'], label='Training Loss')
            if self.metrics['val_losses']:
                # Plot validation loss if available
                val_epochs = [i * len(self.metrics['train_losses']) // len(self.metrics['val_losses']) 
                             for i in range(len(self.metrics['val_losses']))]
                ax.plot(val_epochs, self.metrics['val_losses'], label='Validation Loss')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True)
            figures['loss'] = fig
            
            if save_fig:
                fig_path = os.path.join(self.figures_dir, f"{self.experiment_name}_loss.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Loss curve saved to {fig_path}")
        
        # Plot training perplexity
        if self.metrics['train_perplexities']:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.metrics['train_perplexities'], label='Training Perplexity')
            if self.metrics['val_perplexities']:
                # Plot validation perplexity if available
                val_epochs = [i * len(self.metrics['train_perplexities']) // len(self.metrics['val_perplexities']) 
                             for i in range(len(self.metrics['val_perplexities']))]
                ax.plot(val_epochs, self.metrics['val_perplexities'], label='Validation Perplexity')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Perplexity')
            ax.set_title('Training and Validation Perplexity')
            ax.legend()
            ax.grid(True)
            figures['perplexity'] = fig
            
            if save_fig:
                fig_path = os.path.join(self.figures_dir, f"{self.experiment_name}_perplexity.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Perplexity curve saved to {fig_path}")
        
        # Plot learning rate
        if self.metrics['learning_rates']:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.metrics['learning_rates'])
            ax.set_xlabel('Steps')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True)
            figures['lr'] = fig
            
            if save_fig:
                fig_path = os.path.join(self.figures_dir, f"{self.experiment_name}_lr.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Learning rate curve saved to {fig_path}")
        
        return figures
    
    def close(self):
        """Close the logger and save metrics."""
        self.save_metrics()
        logging.shutdown()


def test_logger():
    """Test function for TrainingLogger."""
    # Create logger
    logger = TrainingLogger(log_dir="test_logs", experiment_name="test_experiment")
    
    # Log some training steps
    for epoch in range(3):
        for batch in range(5):
            loss = 2.0 - (epoch * 0.1 + batch * 0.01)
            perplexity = torch.exp(torch.tensor(loss)).item()
            lr = 0.001 * (0.95 ** (epoch * 5 + batch))
            logger.log_training_step(epoch, batch, loss, perplexity, lr)
        
        # Log epoch summary
        avg_loss = 2.0 - epoch * 0.1
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        logger.log_epoch(epoch, avg_loss, avg_perplexity, learning_rate=lr)
    
    # Plot curves (will be skipped if matplotlib is not available)
    figures = logger.plot_training_curves(save_fig=False)
    
    # Close logger
    logger.close()
    
    print("TrainingLogger test passed!")


if __name__ == "__main__":
    test_logger()