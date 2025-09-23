#!/usr/bin/env python3
"""
Main pipeline runner for LLM training and inference.
This script orchestrates the entire process from dataset loading to model training to output generation.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.append('.')

# Import our components
from src.tokenizers.byte_tokenizer import ByteTokenizer
from create_llm import LLM, generate_text, TextDataset
from torch.utils.data import DataLoader
from src.utils.dataset_loader import load_all_datasets

def train_model(training_texts, output_dir="./outputs"):
    """
    Train the LLM model with the provided texts.
    
    Args:
        training_texts (list): List of text strings for training
        output_dir (str): Directory to save outputs
        
    Returns:
        tuple: (model, tokenizer) trained model and tokenizer
    """
    print("Training LLM model...")
    print("=" * 50)
    
    if not training_texts:
        print("No training data found. Using sample data for demonstration.")
        training_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand text.",
            "Deep learning models have revolutionized many fields.",
            "Transformers are the foundation of modern language models.",
            "Large language models can generate human-like text.",
        ]
    
    # Initialize tokenizer
    tokenizer = ByteTokenizer()
    print(f"Tokenizer initialized with vocab size: {tokenizer.vocab_size}")
    
    # Create dataset
    seq_length = 64
    dataset = TextDataset(training_texts, tokenizer, seq_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Initialize model
    model = LLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        max_seq_length=512
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Training loop
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Training on device: {device}")
    print("Starting training...")
    
    num_epochs = 2
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            # Move to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, loss = model(input_ids, target_ids)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Print epoch summary
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "llm_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model, tokenizer, device

def generate_outputs(model, tokenizer, device, output_dir="./outputs"):
    """
    Generate sample outputs using the trained model.
    
    Args:
        model (LLM): Trained model
        tokenizer (ByteTokenizer): Tokenizer
        device (torch.device): Device to run generation on
        output_dir (str): Directory to save outputs
    """
    print("\nGenerating sample outputs...")
    
    # Generate text
    model.eval()
    
    prompts = [
        "The future of artificial intelligence",
        "Machine learning has revolutionized",
        "Natural language processing enables",
        "Deep learning models can"
    ]
    
    output_file = os.path.join(output_dir, "generated_samples.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Generated Text Samples\n")
        f.write("=" * 50 + "\n\n")
        
        for prompt in prompts:
            generated_text = generate_text(
                model, tokenizer, prompt, 
                max_length=50, temperature=0.8, device=device
            )
            
            print(f"\nPrompt: '{prompt}'")
            print(f"Generated: '{generated_text}'")
            
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {generated_text}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\nGenerated samples saved to {output_file}")

def main():
    """Main function to run the complete LLM pipeline."""
    print("LLM Pipeline Runner")
    print("=" * 50)
    
    # Load datasets
    training_texts = load_all_datasets("./datasets")
    
    # Train model
    model, tokenizer, device = train_model(training_texts, "./outputs")
    
    # Generate outputs
    generate_outputs(model, tokenizer, device, "./outputs")
    
    print("\n" + "=" * 50)
    print("LLM pipeline completed successfully!")
    print("Check the 'outputs' directory for results.")

if __name__ == "__main__":
    main()