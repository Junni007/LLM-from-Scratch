"""
Simple example script demonstrating the tiny LLM implementation.
"""

import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from src.models.transformer import TransformerBlock, ModernTransformerBlock
from src.tokenizers.byte_tokenizer import ByteTokenizer
from src.train.data import create_data_loader
from src.train.trainer import Trainer
from src.train.evaluation import evaluate_model


class TinyLLM(nn.Module):
    """
    A tiny LLM implementation using our Transformer blocks.
    """
    
    def __init__(self, vocab_size=256, d_model=64, num_heads=4, num_layers=2, d_ff=128):
        super(TinyLLM, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Apply embedding
        x = self.embedding(x)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Apply output projection
        logits = self.output_projection(x)
        
        return logits


class ModernTinyLLM(nn.Module):
    """
    A modern tiny LLM implementation using our modern Transformer blocks.
    """
    
    def __init__(self, vocab_size=256, d_model=64, num_heads=4, num_layers=2, d_ff=128, max_seq_len=512):
        super(ModernTinyLLM, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Modern Transformer blocks
        self.layers = nn.ModuleList([
            ModernTransformerBlock(d_model, num_heads, d_ff, max_seq_len=max_seq_len) 
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Apply embedding
        x = self.embedding(x)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Apply output projection
        logits = self.output_projection(x)
        
        return logits


def main():
    print("Tiny LLM Example")
    print("=" * 50)
    
    # Initialize tokenizer
    tokenizer = ByteTokenizer()
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    # Sample training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models have revolutionized many fields.",
        "Transformers are a powerful architecture for sequence modeling.",
        "Attention mechanisms allow models to focus on relevant parts of input.",
        "Large language models can generate human-like text.",
        "Tokenization is the process of converting text into tokens.",
        "Neural networks learn patterns from data.",
        "PyTorch is a popular deep learning framework."
    ]
    
    # Encode texts
    print("Encoding training data...")
    tokenized_texts = tokenizer.encode_batch(training_texts, add_bos=True, add_eos=True)
    
    # Create data loader
    seq_length = 16
    batch_size = 4
    data_loader = create_data_loader(tokenized_texts, seq_length, batch_size)
    
    print(f"Number of training samples: {len(data_loader.dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    
    # Initialize models
    print("\nInitializing models...")
    classic_model = TinyLLM(vocab_size=tokenizer.vocab_size, d_model=64, num_heads=4, num_layers=2)
    modern_model = ModernTinyLLM(vocab_size=tokenizer.vocab_size, d_model=64, num_heads=4, num_layers=2)
    
    # Initialize optimizers
    classic_optimizer = optim.Adam(classic_model.parameters(), lr=0.001)
    modern_optimizer = optim.Adam(modern_model.parameters(), lr=0.001)
    
    # Initialize trainers
    classic_trainer = Trainer(classic_model, classic_optimizer)
    modern_trainer = Trainer(modern_model, modern_optimizer)
    
    print(f"Classic model parameters: {sum(p.numel() for p in classic_model.parameters()):,}")
    print(f"Modern model parameters: {sum(p.numel() for p in modern_model.parameters()):,}")
    
    # Train classic model
    print("\nTraining classic model...")
    classic_trainer.train(data_loader, 2, log_interval=5)
    
    # Train modern model
    print("\nTraining modern model...")
    modern_trainer.train(data_loader, 2, log_interval=5)
    
    # Evaluate models
    print("\nEvaluating classic model...")
    classic_metrics = evaluate_model(classic_model, data_loader)
    print(f"Classic Model Metrics:")
    print(f"  Loss: {classic_metrics['loss']:.4f}")
    print(f"  Perplexity: {classic_metrics['perplexity']:.4f}")
    print(f"  Accuracy: {classic_metrics['accuracy']:.4f}")
    
    print("\nEvaluating modern model...")
    modern_metrics = evaluate_model(modern_model, data_loader)
    print(f"Modern Model Metrics:")
    print(f"  Loss: {modern_metrics['loss']:.4f}")
    print(f"  Perplexity: {modern_metrics['perplexity']:.4f}")
    print(f"  Accuracy: {modern_metrics['accuracy']:.4f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()