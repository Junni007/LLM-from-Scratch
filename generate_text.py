#!/usr/bin/env python3
"""
Script to generate text using a trained LLM model.
"""

import os
import sys
import torch
import argparse

# Add src to path
sys.path.append('.')

# Import our components
from src.tokenizers.byte_tokenizer import ByteTokenizer
from create_llm import LLM, generate_text

def load_model(model_path="./outputs/llm_model.pth"):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the model checkpoint
        
    Returns:
        tuple: (model, tokenizer, device)
    """
    print(f"Loading model from {model_path}...")
    
    # Initialize tokenizer and model
    tokenizer = ByteTokenizer()
    model = LLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        max_seq_length=512
    )
    
    # Load checkpoint
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model checkpoint loaded successfully!")
    else:
        print(f"Warning: No checkpoint found at {model_path}. Using untrained model.")
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def main():
    """Main function to generate text with a trained model."""
    parser = argparse.ArgumentParser(description="Generate text using a trained LLM")
    parser.add_argument("--model", type=str, default="./outputs/llm_model.pth", 
                        help="Path to the trained model checkpoint")
    parser.add_argument("--prompt", type=str, default="The future of", 
                        help="Text prompt to start generation")
    parser.add_argument("--length", type=int, default=100, 
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8, 
                        help="Sampling temperature (lower = more deterministic)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.model)
    
    # Generate text
    print(f"Generating text with prompt: '{args.prompt}'")
    print("-" * 50)
    
    generated_text = generate_text(
        model, tokenizer, args.prompt,
        max_length=args.length, 
        temperature=args.temperature, 
        device=device
    )
    
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()