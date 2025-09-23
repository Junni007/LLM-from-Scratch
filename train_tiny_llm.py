#!/usr/bin/env python3
"""
Train a tiny LLM from scratch using our implementation.
This script demonstrates the complete training pipeline.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import os

# Add src to path
sys.path.append('.')

# Import our components
from src.models.transformer import TransformerBlock
from src.tokenizers.byte_tokenizer import ByteTokenizer
from src.train.loss import CrossEntropyLoss
from src.train.sampling import sample_tokens
from src.train.lr_scheduler import WarmupDecayScheduler


class TinyLLM(nn.Module):
    """A tiny LLM for demonstration purposes."""
    
    def __init__(self, vocab_size=256, d_model=128, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = self.create_positional_encoding(1024, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def create_positional_encoding(self, max_seq_length, d_model):
        """Create positional encoding matrix."""
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_seq_length, d_model)
    
    def forward(self, input_ids, targets=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_length)
            targets (torch.Tensor, optional): Target token IDs for loss computation
            
        Returns:
            logits (torch.Tensor): Output logits of shape (batch_size, seq_length, vocab_size)
            loss (torch.Tensor, optional): Computed loss if targets provided
        """
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        x = self.embedding(input_ids)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :seq_length, :].to(x.device)
        x = x + pos_encoding
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Output projection
        logits = self.output_projection(x)  # (batch_size, seq_length, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits, targets)
        
        return logits, loss


class TextDataset(Dataset):
    """Dataset for text training data."""
    
    def __init__(self, texts, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.samples = []
        
        # Process texts into training samples
        for text in texts:
            # Encode text
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            
            # Create samples by sliding window
            for i in range(0, len(tokens) - seq_length):
                if i + seq_length + 1 <= len(tokens):
                    input_seq = tokens[i:i + seq_length]
                    target_seq = tokens[i + 1:i + seq_length + 1]
                    self.samples.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, device='cpu'):
    """
    Generate text using the trained model.
    
    Args:
        model (nn.Module): Trained model
        tokenizer (ByteTokenizer): Tokenizer
        prompt (str): Initial text prompt
        max_length (int): Maximum length of generated text
        temperature (float): Sampling temperature
        device (str): Device to run generation on
        
    Returns:
        str: Generated text
    """
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_bos=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Generate text
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits for last token
            logits, _ = model(input_tensor)
            next_token_logits = logits[0, -1, :]  # Get logits for last position
            
            # Sample next token
            next_token = sample_tokens(
                next_token_logits.unsqueeze(0), 
                method='temperature', 
                temperature=temperature
            )
            
            # Add to sequence
            next_token = next_token.unsqueeze(0).to(device)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
            # Stop if EOS token generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_tokens = input_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def train_model():
    """Train a tiny LLM from scratch."""
    print("Training Tiny LLM from Scratch")
    print("=" * 40)
    
    # Sample training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models have revolutionized many fields.",
        "Transformers are the foundation of modern language models.",
        "Large language models can generate human-like text.",
        "Attention mechanisms help models focus on relevant information.",
        "Neural networks learn patterns from data.",
        "PyTorch provides flexible tools for deep learning research.",
        "Open-source software accelerates scientific progress.",
        "The weather is sunny today with clear blue skies.",
        "Mathematics is the language of science and engineering.",
        "Programming requires logical thinking and problem-solving skills.",
        "Data science combines statistics, programming, and domain expertise.",
        "Artificial intelligence will transform many industries in the future.",
        "The internet has revolutionized how we access information.",
        "Climate change is one of the biggest challenges of our time.",
        "Renewable energy sources are becoming more cost-effective.",
        "Space exploration continues to expand our understanding of the universe.",
        "Education is the key to personal and societal development."
    ]
    
    # Initialize tokenizer
    tokenizer = ByteTokenizer()
    print(f"Tokenizer initialized with vocab size: {tokenizer.vocab_size}")
    
    # Create dataset
    seq_length = 32
    dataset = TextDataset(training_texts, tokenizer, seq_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Initialize model
    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = WarmupDecayScheduler(
        optimizer, 
        warmup_steps=100, 
        total_steps=1000, 
        decay_type='cosine',
        min_lr=1e-5
    )
    
    # Training loop
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Training on device: {device}")
    print("Starting training...")
    
    num_epochs = 5
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
            scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # Print epoch summary
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    
    # Save model
    torch.save(model.state_dict(), 'tiny_llm_checkpoint.pth')
    print("Model checkpoint saved as 'tiny_llm_checkpoint.pth'")
    
    # Generate sample text
    print("\nGenerating sample text...")
    prompts = [
        "The quick brown",
        "Machine learning",
        "Natural language"
    ]
    
    for prompt in prompts:
        generated_text = generate_text(
            model, tokenizer, prompt, 
            max_length=50, temperature=0.8, device=device
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")
    
    return model, tokenizer


def load_and_generate():
    """Load a trained model and generate text."""
    print("\nLoading trained model and generating text...")
    
    # Initialize tokenizer and model
    tokenizer = ByteTokenizer()
    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=4
    )
    
    # Load checkpoint
    if os.path.exists('tiny_llm_checkpoint.pth'):
        model.load_state_dict(torch.load('tiny_llm_checkpoint.pth'))
        print("Model checkpoint loaded successfully!")
    else:
        print("No checkpoint found. Using untrained model.")
    
    # Generate text
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    prompts = [
        "The future of",
        "Artificial intelligence",
        "Deep learning"
    ]
    
    for prompt in prompts:
        generated_text = generate_text(
            model, tokenizer, prompt,
            max_length=30, temperature=0.7, device=device
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")


def main():
    """Main function."""
    print("LLM from Scratch - Training Demo")
    print("=" * 40)
    
    # Train model
    model, tokenizer = train_model()
    
    # Generate with trained model
    load_and_generate()
    
    print("\n" + "=" * 40)
    print("Training demo completed successfully!")
    print("You can now use the trained model for text generation.")


if __name__ == "__main__":
    main()