#!/usr/bin/env python3
"""
Inference script for the trained Tiny LLM.
This script demonstrates how to load and use a trained model for text generation.
"""

import sys
import torch
import torch.nn as nn

# Add src to path
sys.path.append('.')

# Import our components
from src.tokenizers.byte_tokenizer import ByteTokenizer
from train_tiny_llm import TinyLLM, generate_text


def load_model_and_generate():
    """Load a trained model and generate text."""
    print("LLM Inference Demo")
    print("=" * 30)
    
    # Initialize tokenizer and model
    tokenizer = ByteTokenizer()
    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=4
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    try:
        model.load_state_dict(torch.load('tiny_llm_checkpoint.pth', map_location=device))
        print("Model checkpoint loaded successfully!")
    except FileNotFoundError:
        print("No checkpoint found. Using untrained model.")
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Generate text with different prompts and settings
    prompts = [
        "The future of",
        "Artificial intelligence",
        "Deep learning",
        "Machine learning",
        "Natural language"
    ]
    
    temperatures = [0.5, 0.8, 1.0]
    
    print("\nGenerating text with different temperatures:")
    print("-" * 50)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        for temp in temperatures:
            generated_text = generate_text(
                model, tokenizer, prompt,
                max_length=40, temperature=temp, device=device
            )
            print(f"  Temperature {temp}: '{generated_text}'")
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("Interactive Mode (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\nEnter a prompt: ").strip()
            if prompt.lower() == 'quit':
                break
            
            if not prompt:
                continue
                
            # Generate with different temperatures
            print(f"\nGenerating text for: '{prompt}'")
            
            for temp in [0.7, 0.9, 1.1]:
                generated_text = generate_text(
                    model, tokenizer, prompt,
                    max_length=50, temperature=temp, device=device
                )
                print(f"  Temperature {temp}: '{generated_text}'")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except EOFError:
            print("\n\nExiting...")
            break


def demonstrate_model_info():
    """Demonstrate model information."""
    print("Model Information")
    print("=" * 30)
    
    # Initialize tokenizer and model
    tokenizer = ByteTokenizer()
    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=4
    )
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model dimensions: 128")
    print(f"Number of attention heads: 8")
    print(f"Number of transformer layers: 4")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Show model architecture
    print("\nModel architecture:")
    print("-" * 20)
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}")
        
        # Show more details for some modules
        if name == 'layers':
            print(f"  └── {len(list(module))} Transformer blocks")
        elif name == 'embedding':
            print(f"  └── Embedding dimensions: {module.num_embeddings} x {module.embedding_dim}")


def main():
    """Main function."""
    print("LLM from Scratch - Inference Demo")
    print("=" * 40)
    
    # Show model information
    demonstrate_model_info()
    
    # Load model and generate text
    load_model_and_generate()
    
    print("\n" + "=" * 40)
    print("Inference demo completed!")
    print("You can now use this script to generate text with your trained LLM.")


if __name__ == "__main__":
    main()