"""
Example demonstrating advanced semantic processing components.

This example shows how to use Large Concept Models (LCMs), context-aware semantic processing,
semantic decoding with thought units, hyperbolic space representations, and
alternative tokenization methods.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import sys
import os

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.semantic.processing import SemanticProcessor, SemanticConfig

def create_sample_models():
    """Create sample models for demonstration."""
    # Create a small GPT-2 model for demonstration
    config = GPT2Config(
        vocab_size=1000,  # Small vocab for demo
        n_positions=256,
        n_embd=128,
        n_layer=2,
        n_head=2
    )
    
    # Create base model
    base_model = GPT2LMHeadModel(config)
    
    return base_model

def create_sample_inputs(batch_size=2, seq_length=10):
    """Create sample inputs for demonstration."""
    # Create random hidden states
    hidden_states = torch.randn(batch_size, seq_length, 128)
    
    # Create attention mask
    attention_mask = torch.ones(batch_size, seq_length)
    
    return hidden_states, attention_mask

def main():
    """Main function demonstrating semantic processing."""
    print("=== Advanced Semantic Processing Example ===")
    
    # Create sample base model
    print("Creating sample base model...")
    base_model = create_sample_models()
    
    # Create semantic configuration
    config = SemanticConfig(
        hidden_size=128,
        concept_dim=64,
        num_concepts=256,
        hyperbolic_dim=32,
        thought_unit_size=32
    )
    
    # Create semantic processor
    print("Creating semantic processor...")
    semantic_processor = SemanticProcessor(config, vocab_size=1000)
    
    # Create sample inputs
    print("Creating sample inputs...")
    hidden_states, attention_mask = create_sample_inputs(batch_size=2, seq_length=10)
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Process with semantic processor
    print("Processing with semantic processor...")
    outputs = semantic_processor(hidden_states, attention_mask)
    
    print("Semantic processing results:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Demonstrate individual components
    print("\n=== Individual Component Demos ===")
    
    # Large Concept Model
    print("1. Large Concept Model (LCM):")
    concept_embeddings, concept_logits = semantic_processor.lcm(hidden_states, attention_mask)
    print(f"   Concept embeddings shape: {concept_embeddings.shape}")
    print(f"   Concept logits shape: {concept_logits.shape}")
    
    # Context-aware semantic processor
    print("2. Context-aware semantic processor:")
    context_aware_embeddings = semantic_processor.context_processor(concept_embeddings, concept_embeddings)
    print(f"   Context-aware embeddings shape: {context_aware_embeddings.shape}")
    
    # Hyperbolic space mapping
    print("3. Hyperbolic space mapping:")
    hyperbolic_embeddings = semantic_processor.hyperbolic_space(context_aware_embeddings)
    print(f"   Hyperbolic embeddings shape: {hyperbolic_embeddings.shape}")
    
    # Compute hyperbolic distance
    batch_size, seq_length, hyperbolic_dim = hyperbolic_embeddings.shape
    if seq_length >= 2:
        x = hyperbolic_embeddings[:, 0, :]  # First position
        y = hyperbolic_embeddings[:, 1, :]  # Second position
        distance = semantic_processor.hyperbolic_space.hyperbolic_distance(x, y)
        print(f"   Hyperbolic distance between first two positions: {distance.mean().item():.4f}")
    
    # Semantic decoder
    print("4. Semantic decoder:")
    token_logits = semantic_processor.decoder(context_aware_embeddings, attention_mask)
    print(f"   Token logits shape: {token_logits.shape}")
    
    # Byte-level latent tokenizer
    print("5. Byte-level latent tokenizer:")
    # Create sample byte sequence
    sample_bytes = torch.randint(0, 256, (2, 20))  # 2 samples, 20 bytes each
    latent_representations = semantic_processor.byte_tokenizer(sample_bytes)
    print(f"   Input bytes shape: {sample_bytes.shape}")
    print(f"   Latent representations shape: {latent_representations.shape}")
    
    # Decode back to bytes
    reconstructed_bytes = semantic_processor.byte_tokenizer.decode(latent_representations)
    print(f"   Reconstructed bytes shape: {reconstructed_bytes.shape}")
    
    # Compute reconstruction accuracy
    accuracy = (sample_bytes == reconstructed_bytes).float().mean()
    print(f"   Reconstruction accuracy: {accuracy.item():.4f}")
    
    print("\n=== Advanced Semantic Processing Example Completed ===")

if __name__ == "__main__":
    main()