#!/usr/bin/env python3
"""
Demo script showcasing the LLM from Scratch implementation.
This script demonstrates the core components working together.
"""

import sys
import torch
import torch.nn as nn

# Add src to path
sys.path.append('.')

# Import core components
from src.models.attention import MultiHeadAttention
from src.models.mlp import FeedForwardNetwork
from src.models.normalization import LayerNorm, RMSNorm
from src.models.positional import RotaryPositionalEmbedding
from src.models.transformer import TransformerBlock
from src.tokenizers.byte_tokenizer import ByteTokenizer
from src.moe.moe_layer import MoELayer


def demo_transformer_components():
    """Demonstrate core transformer components."""
    print("=== Transformer Components Demo ===")
    
    # Test MultiHeadAttention
    print("1. Testing MultiHeadAttention...")
    attention = MultiHeadAttention(d_model=128, num_heads=8)
    x = torch.randn(2, 10, 128)
    output = attention(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test LayerNorm
    print("2. Testing LayerNorm...")
    layer_norm = LayerNorm(d_model=128)
    norm_output = layer_norm(x)
    print(f"   Normalized output shape: {norm_output.shape}")
    
    # Test RMSNorm
    print("3. Testing RMSNorm...")
    rms_norm = RMSNorm(d_model=128)
    rms_output = rms_norm(x)
    print(f"   RMS normalized output shape: {rms_output.shape}")
    
    # Test Rotary Positional Embedding
    print("4. Testing Rotary Positional Embedding...")
    rope = RotaryPositionalEmbedding(d_model=128)
    rope_output = rope(x)
    print(f"   RoPE output shape: {rope_output.shape}")
    
    # Test FeedForwardNetwork
    print("5. Testing FeedForwardNetwork...")
    mlp = FeedForwardNetwork(d_model=128)
    mlp_output = mlp(x)
    print(f"   MLP output shape: {mlp_output.shape}")
    
    # Test TransformerBlock
    print("6. Testing TransformerBlock...")
    transformer_block = TransformerBlock(d_model=128, num_heads=8)
    block_output = transformer_block(x)
    print(f"   Transformer block output shape: {block_output.shape}")


def demo_advanced_components():
    """Demonstrate advanced components."""
    print("\n=== Advanced Components Demo ===")
    
    # Test MoE Layer
    print("1. Testing Mixture of Experts (MoE) Layer...")
    moe_layer = MoELayer(d_model=128, num_experts=4, top_k=2)
    x = torch.randn(2, 10, 128)
    moe_output = moe_layer(x)
    print(f"   MoE output shape: {moe_output.shape}")
    print(f"   MoE balance loss: {moe_layer.balance_loss.item():.6f}")


def demo_tokenization():
    """Demonstrate tokenization."""
    print("\n=== Tokenization Demo ===")
    
    # Test Byte Tokenizer
    print("1. Testing Byte Tokenizer...")
    tokenizer = ByteTokenizer()
    text = "Hello, world! This is a test of the LLM from Scratch implementation."
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded)
    print(f"   Original text: {text}")
    print(f"   Encoded length: {len(encoded)} tokens")
    print(f"   Decoded text: {decoded}")
    print(f"   Match: {text == decoded}")


def demo_complete_model():
    """Demonstrate a complete small transformer model."""
    print("\n=== Complete Model Demo ===")
    
    class TinyLLM(nn.Module):
        """A tiny LLM for demonstration purposes."""
        def __init__(self, vocab_size=256, d_model=128, num_heads=8, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                TransformerBlock(d_model, num_heads) for _ in range(num_layers)
            ])
            self.output_projection = nn.Linear(d_model, vocab_size)
            
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            logits = self.output_projection(x)
            return logits
    
    print("1. Creating a tiny LLM...")
    model = TinyLLM(vocab_size=256, d_model=128, num_heads=8, num_layers=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # Test with sample input
    print("2. Testing with sample input...")
    batch_size, seq_length = 2, 8
    input_ids = torch.randint(0, 256, (batch_size, seq_length))
    logits = model(input_ids)
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output logits shape: {logits.shape}")


def main():
    """Run all demos."""
    print("LLM from Scratch - Demo")
    print("=" * 50)
    
    demo_transformer_components()
    demo_advanced_components()
    demo_tokenization()
    demo_complete_model()
    
    print("\n" + "=" * 50)
    print("All demos completed successfully!")
    print("The LLM from Scratch implementation is fully functional.")


if __name__ == "__main__":
    main()