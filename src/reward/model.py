"""
Reward model architecture for LLM alignment.

This module implements reward models for training with preference data,
including the standard reward model architecture used in RLHF.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class RewardModel(nn.Module):
    """
    Reward model for LLM alignment.
    
    This model takes tokenized text as input and outputs a scalar reward score.
    It's typically based on a pretrained language model with a reward head.
    """
    
    def __init__(self, base_model: nn.Module, hidden_size: int = 768, dropout: float = 0.1):
        """
        Initialize RewardModel.
        
        Args:
            base_model (nn.Module): Base language model (e.g., Transformer)
            hidden_size (int): Hidden size of the base model
            dropout (float): Dropout rate
        """
        super(RewardModel, self).__init__()
        self.base_model = base_model
        
        # Reward head - maps hidden states to scalar rewards
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize reward head weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize reward head weights."""
        for module in self.reward_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the reward model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Reward scores of shape (batch_size,)
        """
        # Get hidden states from base model
        # Note: This assumes the base model returns hidden states as the first element
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]  # First element is usually hidden states
        else:
            hidden_states = outputs
        
        # Get the last hidden state (typically the final layer)
        # Shape: (batch_size, seq_len, hidden_size)
        last_hidden_states = hidden_states
        
        # Apply reward head to get rewards
        # We typically use the last token representation (e.g., for EOS token)
        # or pool over the sequence
        
        # Option 1: Use last token representation
        batch_size, seq_len, hidden_size = last_hidden_states.shape
        if attention_mask is not None:
            # Find the last non-padding token for each sequence
            # attention_mask: 1 for real tokens, 0 for padding
            last_token_indices = attention_mask.sum(dim=1) - 1  # Subtract 1 for 0-based indexing
            last_token_indices = last_token_indices.long()  # Convert to int64 for gather
            last_token_indices = last_token_indices.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
            last_token_indices = last_token_indices.expand(batch_size, 1, hidden_size)  # (batch_size, 1, hidden_size)
            last_token_hidden = torch.gather(last_hidden_states, 1, last_token_indices).squeeze(1)  # (batch_size, hidden_size)
        else:
            # Use the last token for each sequence
            last_token_hidden = last_hidden_states[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply reward head
        rewards = self.reward_head(last_token_hidden).squeeze(-1)  # (batch_size,)
        
        return rewards


class PairwiseRewardModel(nn.Module):
    """
    Reward model for pairwise preference learning.
    
    This model computes rewards for both chosen and rejected responses
    and can be used with pairwise loss functions.
    """
    
    def __init__(self, base_model: nn.Module, hidden_size: int = 768, dropout: float = 0.1):
        """
        Initialize PairwiseRewardModel.
        
        Args:
            base_model (nn.Module): Base language model
            hidden_size (int): Hidden size of the base model
            dropout (float): Dropout rate
        """
        super(PairwiseRewardModel, self).__init__()
        self.base_model = base_model
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize reward head weights."""
        for module in self.reward_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, chosen_input_ids: torch.Tensor, chosen_attention_mask: torch.Tensor,
                rejected_input_ids: torch.Tensor, rejected_attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both chosen and rejected responses.
        
        Args:
            chosen_input_ids (torch.Tensor): Chosen response input IDs
            chosen_attention_mask (torch.Tensor): Chosen response attention mask
            rejected_input_ids (torch.Tensor): Rejected response input IDs
            rejected_attention_mask (torch.Tensor): Rejected response attention mask
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Chosen and rejected rewards
        """
        # Compute rewards for chosen responses
        chosen_rewards = self.compute_reward(chosen_input_ids, chosen_attention_mask)
        
        # Compute rewards for rejected responses
        rejected_rewards = self.compute_reward(rejected_input_ids, rejected_attention_mask)
        
        return chosen_rewards, rejected_rewards
    
    def compute_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for a single input.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Reward scores
        """
        # Get hidden states from base model
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        
        # Get last token representation
        batch_size, seq_len, hidden_size = hidden_states.shape
        if attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=1) - 1
            last_token_indices = last_token_indices.long()  # Convert to int64 for gather
            last_token_indices = last_token_indices.unsqueeze(-1).unsqueeze(-1)
            last_token_indices = last_token_indices.expand(batch_size, 1, hidden_size)
            last_token_hidden = torch.gather(hidden_states, 1, last_token_indices).squeeze(1)
        else:
            last_token_hidden = hidden_states[:, -1, :]
        
        # Apply reward head
        rewards = self.reward_head(last_token_hidden).squeeze(-1)
        
        return rewards


class DebertaRewardModel(nn.Module):
    """
    Specialized reward model based on DeBERTa architecture.
    
    This is a simplified version that demonstrates the concept.
    In practice, you would use a full DeBERTa implementation.
    """
    
    def __init__(self, vocab_size: int = 30522, hidden_size: int = 768, 
                 num_layers: int = 12, num_heads: int = 12, dropout: float = 0.1):
        """
        Initialize DebertaRewardModel.
        
        Args:
            vocab_size (int): Vocabulary size
            hidden_size (int): Hidden size
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super(DebertaRewardModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Position embeddings
        self.pos_embedding = nn.Embedding(512, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            DebertaLayer(hidden_size, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        
        for module in self.reward_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Reward scores
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden_states = self.embedding(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        pos_embeddings = self.pos_embedding(position_ids)
        hidden_states = hidden_states + pos_embeddings
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Get last token representation
        if attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=1) - 1
            last_token_indices = last_token_indices.long()  # Convert to int64 for gather
            last_token_indices = last_token_indices.unsqueeze(-1).unsqueeze(-1)
            last_token_indices = last_token_indices.expand(batch_size, 1, self.hidden_size)
            last_token_hidden = torch.gather(hidden_states, 1, last_token_indices).squeeze(1)
        else:
            last_token_hidden = hidden_states[:, -1, :]
        
        # Apply reward head
        rewards = self.reward_head(last_token_hidden).squeeze(-1)
        
        return rewards


class DebertaLayer(nn.Module):
    """
    Simplified DeBERTa layer for demonstration.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize DebertaLayer.
        
        Args:
            hidden_size (int): Hidden size
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super(DebertaLayer, self).__init__()
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states (torch.Tensor): Hidden states
            attention_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Updated hidden states
        """
        # Self-attention
        attn_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=(1 - attention_mask) if attention_mask is not None else None,
            need_weights=False
        )
        
        # Residual connection and layer normalization
        hidden_states = self.layer_norm1(hidden_states + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(hidden_states)
        
        # Residual connection and layer normalization
        hidden_states = self.layer_norm2(hidden_states + ffn_output)
        
        return hidden_states


def demonstrate_reward_models():
    """Demonstrate reward model functionality."""
    print("Reward Model Architecture Demonstration")
    print("=" * 40)
    
    # Test with a simple base model
    class SimpleBaseModel(nn.Module):
        """Simple base model for testing."""
        
        def __init__(self, vocab_size=1000, hidden_size=128):
            super(SimpleBaseModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
                num_layers=2
            )
        
        def forward(self, input_ids, attention_mask=None):
            # Embedding
            embedded = self.embedding(input_ids)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert attention mask to key_padding_mask format
                key_padding_mask = (1 - attention_mask).bool()
            else:
                key_padding_mask = None
            
            # Transformer
            output = self.transformer(embedded, src_key_padding_mask=key_padding_mask)
            return output
    
    # Create base model
    base_model = SimpleBaseModel(vocab_size=1000, hidden_size=128)
    
    # Test RewardModel
    print("1. Testing RewardModel...")
    reward_model = RewardModel(base_model, hidden_size=128)
    
    # Create sample input
    batch_size, seq_len = 4, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    rewards = reward_model(input_ids, attention_mask)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Reward shape: {rewards.shape}")
    print(f"  Sample rewards: {rewards.tolist()}")
    
    # Test PairwiseRewardModel
    print("\n2. Testing PairwiseRewardModel...")
    pairwise_model = PairwiseRewardModel(base_model, hidden_size=128)
    
    # Create sample inputs for chosen and rejected responses
    chosen_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    chosen_attention_mask = torch.ones(batch_size, seq_len)
    rejected_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    rejected_attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    chosen_rewards, rejected_rewards = pairwise_model(
        chosen_input_ids, chosen_attention_mask,
        rejected_input_ids, rejected_attention_mask
    )
    print(f"  Chosen rewards shape: {chosen_rewards.shape}")
    print(f"  Rejected rewards shape: {rejected_rewards.shape}")
    print(f"  Sample chosen rewards: {chosen_rewards.tolist()}")
    print(f"  Sample rejected rewards: {rejected_rewards.tolist()}")
    
    # Test DebertaRewardModel
    print("\n3. Testing DebertaRewardModel...")
    deberta_model = DebertaRewardModel(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4)
    
    # Forward pass
    deberta_rewards = deberta_model(input_ids, attention_mask)
    print(f"  DeBERTa reward shape: {deberta_rewards.shape}")
    print(f"  Sample DeBERTa rewards: {deberta_rewards.tolist()}")
    
    print("\nReward model architecture demonstration completed!")


if __name__ == "__main__":
    demonstrate_reward_models()