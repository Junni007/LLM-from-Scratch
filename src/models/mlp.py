"""
Implementation of feed-forward networks (MLP layers) for Transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    
    SwiGLU(x) = Swish(x * W1 + b1) * (x * W2 + b2)
    
    Args:
        d_model (int): Dimension of input and output embeddings
        d_ff (int): Dimension of hidden layer (typically 4 * d_model)
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super(SwiGLU, self).__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # Default to 4 times the model dimension
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_model, d_ff)
        self.linear3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through SwiGLU activation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear transformation
        x1 = self.linear1(x)
        
        # Second linear transformation
        x2 = self.linear2(x)
        
        # Apply Swish activation to first part
        x1_swish = F.silu(x1)  # Swish activation
        
        # Element-wise multiplication
        x_mult = x1_swish * x2
        
        # Apply dropout
        x_mult = self.dropout(x_mult)
        
        # Final linear transformation
        output = self.linear3(x_mult)
        
        return output


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network (MLP) used in Transformer blocks.
    
    This implementation follows the original Transformer paper:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Args:
        d_model (int): Dimension of input and output embeddings
        d_ff (int): Dimension of hidden layer (typically 4 * d_model)
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # Default to 4 times the model dimension
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear transformation + ReLU activation
        x = F.relu(self.linear1(x))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Second linear transformation
        x = self.linear2(x)
        
        return x


def test_swiglu():
    """
    Test function for SwiGLU.
    """
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_ff = 16
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize SwiGLU
    swiglu = SwiGLU(d_model, d_ff)
    
    # Compute forward pass
    output = swiglu(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {output.shape}"
    
    print("SwiGLU test passed!")


def test_feed_forward_network():
    """
    Test function for FeedForwardNetwork.
    """
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_ff = 16
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize feed-forward network
    ff_network = FeedForwardNetwork(d_model, d_ff)
    
    # Compute forward pass
    output = ff_network(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {output.shape}"
    
    print("FeedForwardNetwork test passed!")


if __name__ == "__main__":
    test_swiglu()
    test_feed_forward_network()