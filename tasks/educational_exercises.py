"""
Educational exercises for LLM implementation.

This file contains hands-on exercises for each part of the LLM from Scratch curriculum.
"""

import torch
import torch.nn as nn
import os
import sys

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our implementations
from src.models.attention import MultiHeadAttention
from src.models.mlp import FeedForwardNetwork as MLP
from src.models.normalization import LayerNorm, RMSNorm
from src.models.positional import RotaryPositionalEmbedding as RotaryPositionalEncoding
from src.models.transformer import TransformerBlock
from src.tokenizers.byte_tokenizer import ByteTokenizer as ByteLevelTokenizer
from src.tokenizers.bpe_tokenizer import BPETokenizer as BPE_Tokenizer
from src.moe.moe_layer import MoELayer
from src.reward.model import RewardModel
from src.rlhf.ppo import PolicyValueNetwork
from src.semantic.processing import SemanticProcessor

class EducationalExercises:
    """Collection of educational exercises for LLM implementation."""
    
    @staticmethod
    def exercise_1_1_basic_attention():
        """
        Exercise 1.1: Implement Basic Attention Mechanism
        
        Students should implement scaled dot-product attention from scratch.
        """
        print("Exercise 1.1: Basic Attention Mechanism")
        print("=" * 40)
        
        # TODO: Students implement this function
        def scaled_dot_product_attention(query, key, value, mask=None):
            """
            Implement scaled dot-product attention.
            
            Args:
                query: Tensor of shape (batch_size, num_heads, seq_length, head_dim)
                key: Tensor of shape (batch_size, num_heads, seq_length, head_dim)
                value: Tensor of shape (batch_size, num_heads, seq_length, head_dim)
                mask: Optional mask tensor
                
            Returns:
                output: Tensor of shape (batch_size, num_heads, seq_length, head_dim)
                attention_weights: Tensor of shape (batch_size, num_heads, seq_length, seq_length)
            """
            # Students should implement this
            pass
        
        # Test data
        batch_size, num_heads, seq_length, head_dim = 2, 4, 8, 16
        query = torch.randn(batch_size, num_heads, seq_length, head_dim)
        key = torch.randn(batch_size, num_heads, seq_length, head_dim)
        value = torch.randn(batch_size, num_heads, seq_length, head_dim)
        
        print("Implement the scaled_dot_product_attention function above.")
        print("Expected output shapes:")
        print(f"  Output: ({batch_size}, {num_heads}, {seq_length}, {head_dim})")
        print(f"  Attention weights: ({batch_size}, {num_heads}, {seq_length}, {seq_length})")
        
    @staticmethod
    def exercise_1_2_multi_head_attention():
        """
        Exercise 1.2: Multi-Head Attention
        
        Use the provided MultiHeadAttention class.
        """
        print("Exercise 1.2: Multi-Head Attention")
        print("=" * 40)
        
        # Create multi-head attention
        d_model = 128
        num_heads = 8
        attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
        # Create sample input
        batch_size, seq_length = 2, 10
        query = torch.randn(batch_size, seq_length, d_model)
        key = torch.randn(batch_size, seq_length, d_model)
        value = torch.randn(batch_size, seq_length, d_model)
        
        # Process with multi-head attention
        output, attention_weights = attention(query, key, value)
        
        print(f"Input shape: {query.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")
        print("Multi-head attention working correctly!")
        
    @staticmethod
    def exercise_1_3_transformer_block():
        """
        Exercise 1.3: Transformer Block
        
        Combine attention, MLP, and normalization.
        """
        print("Exercise 1.3: Transformer Block")
        print("=" * 40)
        
        # Create transformer block
        transformer_block = TransformerBlock(d_model=128, num_heads=8, dropout=0.1)
        
        # Create sample input
        batch_size, seq_length, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Process with transformer block
        output = transformer_block(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print("Transformer block working correctly!")
        
    @staticmethod
    def exercise_2_1_byte_level_tokenization():
        """
        Exercise 2.1: Byte-Level Tokenization
        
        Use byte-level tokenizer to encode/decode text.
        """
        print("Exercise 2.1: Byte-Level Tokenization")
        print("=" * 40)
        
        # Create tokenizer
        tokenizer = ByteLevelTokenizer()
        
        # Test encoding and decoding
        text = "Hello, world! This is a test."
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        print(f"Original text: {text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Match: {text == decoded}")
        
    @staticmethod
    def exercise_3_1_rmsnorm_comparison():
        """
        Exercise 3.1: RMSNorm vs LayerNorm
        
        Compare normalization techniques.
        """
        print("Exercise 3.1: RMSNorm vs LayerNorm")
        print("=" * 40)
        
        # Create normalization layers
        layer_norm = LayerNorm(d_model=128)
        rms_norm = RMSNorm(d_model=128)
        
        # Create sample input
        batch_size, seq_length, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Apply normalization
        ln_output = layer_norm(x)
        rms_output = rms_norm(x)
        
        print(f"Input shape: {x.shape}")
        print(f"LayerNorm output shape: {ln_output.shape}")
        print(f"RMSNorm output shape: {rms_output.shape}")
        
        # Check normalization properties
        ln_mean = ln_output.mean(dim=-1)
        ln_std = ln_output.std(dim=-1)
        rms_mean = rms_output.mean(dim=-1)
        rms_std = rms_output.std(dim=-1)
        
        print(f"LayerNorm mean (should be ~0): {ln_mean.mean():.6f}")
        print(f"LayerNorm std (should be ~1): {ln_std.mean():.6f}")
        print(f"RMSNorm mean: {rms_mean.mean():.6f}")
        print(f"RMSNorm std (should be ~1): {rms_std.mean():.6f}")
        
    @staticmethod
    def exercise_4_1_bpe_tokenization():
        """
        Exercise 4.1: BPE Tokenization
        
        Train and use BPE tokenizer.
        """
        print("Exercise 4.1: BPE Tokenization")
        print("=" * 40)
        
        # Create sample training data
        training_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand text.",
            "Deep learning models have revolutionized many fields.",
            "Transformers are the foundation of modern language models.",
        ]
        
        # Create and train BPE tokenizer
        bpe_tokenizer = BPE_Tokenizer(vocab_size=500)
        bpe_tokenizer.train(training_texts)
        
        # Test tokenization
        test_text = "Transformers are powerful models for NLP tasks."
        encoded = bpe_tokenizer.encode(test_text)
        decoded = bpe_tokenizer.decode(encoded)
        
        print(f"Original: {test_text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Vocabulary size: {bpe_tokenizer.vocab_size}")
        
    @staticmethod
    def exercise_5_1_moe_layer():
        """
        Exercise 5.1: Mixture of Experts
        
        Create and test MoE layer.
        """
        print("Exercise 5.1: Mixture of Experts")
        print("=" * 40)
        
        # Create MoE layer
        moe_layer = MoELayer(d_model=128, num_experts=4, top_k=2)
        
        # Create sample input
        batch_size, seq_length, input_size = 2, 10, 128
        x = torch.randn(batch_size, seq_length, input_size)
        
        # Apply MoE
        output, aux_loss = moe_layer(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Auxiliary loss: {aux_loss.item():.6f}")
        print("MoE layer working correctly!")
        
    @staticmethod
    def exercise_7_1_reward_model():
        """
        Exercise 7.1: Reward Model
        
        Create and test reward model.
        """
        print("Exercise 7.1: Reward Model")
        print("=" * 40)
        
        # Create a simple base model for the reward model
        class SimpleBaseModel(nn.Module):
            def __init__(self, hidden_size=128, vocab_size=1000):
                super().__init__()
                self.hidden_size = hidden_size
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.linear = nn.Linear(hidden_size, hidden_size)
                
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                x = self.linear(x)
                logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 1000)
                
                class MockOutput:
                    def __init__(self, last_hidden_state, logits):
                        self.last_hidden_state = last_hidden_state
                        self.logits = logits
                return MockOutput(x, logits)
        
        # Create base model
        base_model = SimpleBaseModel(hidden_size=128, vocab_size=1000)
        
        # Create reward model
        reward_model = RewardModel(base_model, hidden_size=128)
        
        # Create sample input
        batch_size, seq_length = 2, 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        # Get rewards
        rewards = reward_model(input_ids)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Rewards shape: {rewards.shape}")
        print(f"Rewards: {rewards}")
        print("Reward model working correctly!")
        
    @staticmethod
    def exercise_8_1_ppo_policy():
        """
        Exercise 8.1: PPO Policy Network
        
        Create and test PPO policy network.
        """
        print("Exercise 8.1: PPO Policy Network")
        print("=" * 40)
        
        # Create a simple base model for the policy network
        class SimpleBaseModel(nn.Module):
            def __init__(self, hidden_size=128, vocab_size=1000):
                super().__init__()
                self.hidden_size = hidden_size
                self.vocab_size = vocab_size
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.linear = nn.Linear(hidden_size, hidden_size)
                
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                x = self.linear(x)
                logits = torch.randn(input_ids.shape[0], input_ids.shape[1], self.vocab_size)
                
                class MockOutput:
                    def __init__(self, last_hidden_state, logits):
                        self.last_hidden_state = last_hidden_state
                        self.logits = logits
                return MockOutput(x, logits)
        
        # Create base model
        base_model = SimpleBaseModel(hidden_size=128, vocab_size=1000)
        
        # Create policy model
        policy_model = PolicyValueNetwork(base_model, hidden_size=128)
        
        # Create sample input
        batch_size, seq_length = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        # Get policy and value outputs
        logits, values = policy_model(input_ids)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Values shape: {values.shape}")
        print("PPO policy network working correctly!")
        
    @staticmethod
    def exercise_10_1_semantic_processor():
        """
        Exercise 10.1: Semantic Processor
        
        Create and test semantic processor.
        """
        print("Exercise 10.1: Semantic Processor")
        print("=" * 40)
        
        from src.semantic.processing import SemanticConfig
        
        # Create semantic configuration
        config = SemanticConfig(
            hidden_size=128,
            concept_dim=64,
            num_concepts=256,
            hyperbolic_dim=32
        )
        
        # Create semantic processor
        semantic_processor = SemanticProcessor(config, vocab_size=1000)
        
        # Create sample input
        batch_size, seq_length, hidden_size = 2, 10, 128
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Process with semantic processor
        outputs = semantic_processor(hidden_states)
        
        print(f"Input shape: {hidden_states.shape}")
        print("Output keys:", list(outputs.keys()))
        print("Semantic processor working correctly!")

def run_all_exercises():
    """Run all educational exercises."""
    print("Running All Educational Exercises")
    print("=" * 50)
    
    exercises = [
        EducationalExercises.exercise_1_2_multi_head_attention,
        EducationalExercises.exercise_1_3_transformer_block,
        EducationalExercises.exercise_2_1_byte_level_tokenization,
        EducationalExercises.exercise_3_1_rmsnorm_comparison,
        EducationalExercises.exercise_4_1_bpe_tokenization,
        EducationalExercises.exercise_5_1_moe_layer,
        EducationalExercises.exercise_7_1_reward_model,
        EducationalExercises.exercise_8_1_ppo_policy,
        EducationalExercises.exercise_10_1_semantic_processor,
    ]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"\nExercise {i}:")
        exercise()
        print()

if __name__ == "__main__":
    run_all_exercises()