"""
Sampling techniques for LLM generation.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature sampling to logits.
    
    Args:
        logits (torch.Tensor): Logits from model output
        temperature (float): Temperature value (higher = more random, lower = more deterministic)
        
    Returns:
        torch.Tensor: Sampled tokens
    """
    if temperature == 0:
        # Greedy sampling - select the token with highest probability
        return torch.argmax(logits, dim=-1)
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Apply softmax to get probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Sample from the probability distribution
    samples = torch.multinomial(probs, 1)
    
    return samples.squeeze(-1)


def top_k_sampling(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    """
    Apply top-k sampling to logits.
    
    Args:
        logits (torch.Tensor): Logits from model output
        k (int): Number of top tokens to consider
        
    Returns:
        torch.Tensor: Sampled tokens
    """
    # Get the top k values and indices
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # Create a mask for values not in top-k
    mask = logits < top_k_values[..., -1, None]
    
    # Set non-top-k values to negative infinity
    masked_logits = logits.masked_fill(mask, float('-inf'))
    
    # Apply softmax to get probabilities
    probs = F.softmax(masked_logits, dim=-1)
    
    # Sample from the probability distribution
    samples = torch.multinomial(probs, 1)
    
    return samples.squeeze(-1)


def top_p_sampling(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to logits.
    
    Args:
        logits (torch.Tensor): Logits from model output
        p (float): Cumulative probability threshold
        
    Returns:
        torch.Tensor: Sampled tokens
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    
    # Shift the indices to the right to keep the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter sorted indices to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    
    # Set tokens to remove to negative infinity
    masked_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    # Apply softmax to get probabilities
    probs = F.softmax(masked_logits, dim=-1)
    
    # Sample from the probability distribution
    samples = torch.multinomial(probs, 1)
    
    return samples.squeeze(-1)


def sample_tokens(logits: torch.Tensor, method: str = 'greedy', 
                  temperature: float = 1.0, k: int = 50, p: float = 0.9) -> torch.Tensor:
    """
    Sample tokens using various sampling methods.
    
    Args:
        logits (torch.Tensor): Logits from model output
        method (str): Sampling method ('greedy', 'temperature', 'top-k', 'top-p')
        temperature (float): Temperature for temperature sampling
        k (int): Top-k value for top-k sampling
        p (float): Probability threshold for top-p sampling
        
    Returns:
        torch.Tensor: Sampled tokens
    """
    if method == 'greedy':
        return torch.argmax(logits, dim=-1)
    elif method == 'temperature':
        return temperature_sampling(logits, temperature)
    elif method == 'top-k':
        return top_k_sampling(logits, k)
    elif method == 'top-p':
        return top_p_sampling(logits, p)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def test_sampling():
    """
    Test function for sampling techniques.
    """
    # Create sample logits
    batch_size = 2
    vocab_size = 10
    logits = torch.randn(batch_size, vocab_size)
    
    # Test greedy sampling
    greedy_samples = sample_tokens(logits, method='greedy')
    assert greedy_samples.shape == (batch_size,), f"Greedy sampling shape mismatch: {greedy_samples.shape}"
    
    # Test temperature sampling
    temp_samples = sample_tokens(logits, method='temperature', temperature=0.8)
    assert temp_samples.shape == (batch_size,), f"Temperature sampling shape mismatch: {temp_samples.shape}"
    
    # Test top-k sampling
    topk_samples = sample_tokens(logits, method='top-k', k=5)
    assert topk_samples.shape == (batch_size,), f"Top-k sampling shape mismatch: {topk_samples.shape}"
    
    # Test top-p sampling
    topp_samples = sample_tokens(logits, method='top-p', p=0.9)
    assert topp_samples.shape == (batch_size,), f"Top-p sampling shape mismatch: {topp_samples.shape}"
    
    print("Sampling techniques test passed!")


if __name__ == "__main__":
    test_sampling()