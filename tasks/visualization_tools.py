"""
Visualization tools for LLM analysis.

This module provides tools for visualizing attention patterns, training curves,
and other important metrics for understanding LLM behavior.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os

# Conditional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualization functions will be disabled.")

class VisualizationTools:
    """Tools for visualizing LLM components and training metrics."""
    
    def __init__(self):
        if VISUALIZATION_AVAILABLE:
            # Set style for better-looking plots
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
    
    def visualize_attention_weights(self, 
                                 attention_weights: torch.Tensor,
                                 tokens: Optional[List[str]] = None,
                                 head_idx: int = 0,
                                 layer_idx: int = 0,
                                 save_path: Optional[str] = None,
                                 show: bool = True):
        """
        Visualize attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weights tensor of shape 
                             (batch_size, num_heads, seq_length, seq_length)
            tokens: List of token strings for labeling
            head_idx: Which attention head to visualize
            layer_idx: Which layer to visualize (if multiple layers)
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization not available - matplotlib/seaborn not installed")
            return
            
        # Select attention weights for specific head
        if attention_weights.dim() == 4:
            # (batch_size, num_heads, seq_length, seq_length)
            attn = attention_weights[0, head_idx].detach().cpu().numpy()
        elif attention_weights.dim() == 3:
            # (num_heads, seq_length, seq_length)
            attn = attention_weights[head_idx].detach().cpu().numpy()
        else:
            # (seq_length, seq_length)
            attn = attention_weights.detach().cpu().numpy()
        
        seq_length = attn.shape[0]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn, 
                   xticklabels=tokens or range(seq_length),
                   yticklabels=tokens or range(seq_length),
                   cmap='viridis',
                   cbar=True,
                   square=True)
        
        plt.title(f'Attention Weights - Head {head_idx}, Layer {layer_idx}')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        
        if tokens:
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_training_curves(self,
                                metrics: Dict[str, List[float]],
                                save_path: Optional[str] = None,
                                show: bool = True):
        """
        Visualize training metrics over time.
        
        Args:
            metrics: Dictionary of metric names to lists of values
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization not available - matplotlib/seaborn not installed")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot each metric
        for metric_name, values in metrics.items():
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, marker='o', label=metric_name, linewidth=2)
        
        plt.title('Training Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_token_importance(self,
                                 attention_weights: torch.Tensor,
                                 tokens: List[str],
                                 head_idx: int = 0,
                                 save_path: Optional[str] = None,
                                 show: bool = True):
        """
        Visualize token importance based on attention weights.
        
        Args:
            attention_weights: Attention weights tensor
            tokens: List of token strings
            head_idx: Which attention head to analyze
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization not available - matplotlib/seaborn not installed")
            return
            
        # Select attention weights for specific head
        if attention_weights.dim() == 4:
            attn = attention_weights[0, head_idx].detach().cpu().numpy()
        elif attention_weights.dim() == 3:
            attn = attention_weights[head_idx].detach().cpu().numpy()
        else:
            attn = attention_weights.detach().cpu().numpy()
        
        # Compute importance as max attention weight for each token
        importance = np.max(attn, axis=0)  # Max over query positions
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(tokens)), importance)
        
        # Color bars based on importance
        colors = plt.cm.viridis(importance / np.max(importance))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title(f'Token Importance - Attention Head {head_idx}')
        plt.xlabel('Token Position')
        plt.ylabel('Importance Score')
        
        # Add token labels
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_model_architecture(self,
                                   layer_sizes: List[int],
                                   save_path: Optional[str] = None,
                                   show: bool = True):
        """
        Visualize model architecture as a diagram.
        
        Args:
            layer_sizes: List of layer sizes
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization not available - matplotlib/seaborn not installed")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Draw layers as rectangles
        for i, size in enumerate(layer_sizes):
            # Layer rectangle
            rect = plt.Rectangle((i, 0), 0.8, size/100, 
                               facecolor='lightblue', edgecolor='black')
            plt.gca().add_patch(rect)
            
            # Layer label
            plt.text(i + 0.4, size/200, f'Layer {i}\n{size} units',
                    ha='center', va='center', fontsize=10)
        
        plt.xlim(-0.5, len(layer_sizes))
        plt.ylim(0, max(layer_sizes)/100 * 1.2)
        plt.title('Model Architecture')
        plt.xlabel('Layer')
        plt.ylabel('Units (scaled)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_embedding_space(self,
                                embeddings: torch.Tensor,
                                labels: Optional[List[str]] = None,
                                method: str = 'pca',
                                save_path: Optional[str] = None,
                                show: bool = True):
        """
        Visualize embedding space using dimensionality reduction.
        
        Args:
            embeddings: Embedding tensor of shape (num_samples, embedding_dim)
            labels: Optional labels for coloring points
            method: Dimensionality reduction method ('pca' or 'tsne')
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization not available - matplotlib/seaborn not installed")
            return
            
        try:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            
            # Convert to numpy
            embeddings_np = embeddings.detach().cpu().numpy()
            
            # Apply dimensionality reduction
            if method == 'pca':
                reducer = PCA(n_components=2)
            elif method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            reduced_embeddings = reducer.fit_transform(embeddings_np)
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            
            if labels is not None:
                # Color by labels
                unique_labels = list(set(labels))
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = np.array(labels) == label
                    plt.scatter(reduced_embeddings[mask, 0], 
                              reduced_embeddings[mask, 1],
                              c=[colors[i]], label=label, alpha=0.7)
                plt.legend()
            else:
                # Single color
                plt.scatter(reduced_embeddings[:, 0], 
                          reduced_embeddings[:, 1],
                          alpha=0.7)
            
            plt.title(f'Embedding Space Visualization ({method.upper()})')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            if show:
                plt.show()
            else:
                plt.close()
                
        except ImportError:
            print("Warning: scikit-learn not available. Skipping embedding visualization.")
    
    def visualize_loss_landscape(self,
                               model: nn.Module,
                               loss_fn: callable,
                               data_loader: torch.utils.data.DataLoader,
                               save_path: Optional[str] = None,
                               show: bool = True):
        """
        Visualize loss landscape around current model parameters.
        
        Args:
            model: Model to analyze
            loss_fn: Loss function
            data_loader: Data loader for computing loss
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization not available - matplotlib/seaborn not installed")
            return
            
        # This is a simplified version - full implementation would be more complex
        print("Loss landscape visualization is a complex operation.")
        print("This function provides a basic framework.")
        
        # Store original parameters
        original_params = [p.clone() for p in model.parameters()]
        
        # Create parameter vectors
        param_shapes = [p.shape for p in model.parameters()]
        param_sizes = [p.numel() for p in model.parameters()]
        total_params = sum(param_sizes)
        
        print(f"Model has {total_params:,} parameters")
        print("Full loss landscape visualization requires advanced techniques.")
        
        # Restore original parameters
        for p_orig, p_curr in zip(original_params, model.parameters()):
            p_curr.data.copy_(p_orig.data)

# Example usage functions
def create_sample_attention_weights(seq_length: int = 8, num_heads: int = 4) -> torch.Tensor:
    """Create sample attention weights for visualization."""
    # Create random attention weights
    attention_weights = torch.randn(1, num_heads, seq_length, seq_length)
    # Apply softmax to make them valid attention weights
    attention_weights = torch.softmax(attention_weights, dim=-1)
    return attention_weights

def create_sample_training_metrics() -> Dict[str, List[float]]:
    """Create sample training metrics for visualization."""
    epochs = 20
    return {
        'Training Loss': [1.0 - 0.05 * i + 0.1 * np.random.random() for i in range(epochs)],
        'Validation Loss': [1.1 - 0.04 * i + 0.15 * np.random.random() for i in range(epochs)],
        'Accuracy': [0.5 + 0.02 * i + 0.05 * np.random.random() for i in range(epochs)]
    }

def demo_visualizations():
    """Demonstrate visualization tools."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available - matplotlib/seaborn not installed")
        return
        
    print("Demonstrating Visualization Tools")
    print("=" * 40)
    
    # Create visualization tools
    viz = VisualizationTools()
    
    # Demo 1: Attention weights visualization
    print("1. Attention Weights Visualization")
    attention_weights = create_sample_attention_weights(seq_length=8, num_heads=4)
    tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy']
    viz.visualize_attention_weights(attention_weights, tokens, head_idx=0, show=False)
    print("Attention weights visualization created")
    
    # Demo 2: Training curves visualization
    print("2. Training Curves Visualization")
    metrics = create_sample_training_metrics()
    viz.visualize_training_curves(metrics, show=False)
    print("Training curves visualization created")
    
    # Demo 3: Token importance visualization
    print("3. Token Importance Visualization")
    viz.visualize_token_importance(attention_weights, tokens, head_idx=0, show=False)
    print("Token importance visualization created")
    
    print("Visualization demos completed!")

if __name__ == "__main__":
    demo_visualizations()