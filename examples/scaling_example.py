"""
Example demonstrating gradient accumulation and mixed precision training.
"""

import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from src.models.transformer import ModernTransformerBlock
from src.tokenizers.bpe_tokenizer import BPETokenizer
from src.train.data import create_data_loader
from src.train.trainer import Trainer
from src.train.evaluation import evaluate_model


class ScalingLLM(nn.Module):
    """
    A language model for demonstrating scaling techniques.
    """
    
    def __init__(self, vocab_size=1000, d_model=128, num_heads=4, num_layers=4, d_ff=512, max_seq_len=512):
        super(ScalingLLM, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            ModernTransformerBlock(d_model, num_heads, d_ff, max_seq_len=max_seq_len) 
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Apply embedding
        x = self.embedding(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Apply output projection
        logits = self.output_projection(x)
        
        return logits


def main():
    print("Scaling Techniques Example")
    print("=" * 50)
    
    # Create sample training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog. This is a sample sentence for training.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Natural language processing enables computers to understand and generate human language.",
        "Deep learning models have revolutionized many fields including computer vision and NLP.",
        "Transformers are a powerful architecture for sequence modeling tasks.",
        "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
        "Large language models can generate human-like text given a prompt.",
        "Tokenization is the process of converting text into tokens for model processing.",
        "Neural networks learn patterns from data through backpropagation and gradient descent.",
        "PyTorch is a popular deep learning framework for research and production."
    ] * 2  # Reduce repetition to avoid memory issues
    
    # Initialize BPE tokenizer
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(training_texts)
    print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")
    
    # Encode texts
    print("Encoding training data...")
    tokenized_texts = [tokenizer.encode(text, add_bos=True, add_eos=True) for text in training_texts]
    
    # Filter out empty or very short sequences
    tokenized_texts = [text for text in tokenized_texts if len(text) > 10]
    
    print(f"Number of tokenized texts: {len(tokenized_texts)}")
    
    # Create data loader
    seq_length = 32
    batch_size = 2
    
    # Check if we have enough data
    if len(tokenized_texts) == 0:
        print("No valid tokenized texts found. Using fallback data.")
        # Fallback data
        tokenized_texts = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] * 4
    
    data_loader = create_data_loader(tokenized_texts, seq_length, batch_size)
    
    print(f"Number of training samples: {len(data_loader.dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    
    # Initialize model
    print("\nInitializing model...")
    model = ScalingLLM(vocab_size=max(len(tokenizer.vocab), 100), d_model=128, num_heads=4, num_layers=4)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize trainer
    trainer = Trainer(model, optimizer)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train with gradient accumulation
    print("\nTraining with gradient accumulation...")
    try:
        trainer.train_epoch_with_gradient_accumulation(data_loader, accumulation_steps=2, log_interval=2)
    except Exception as e:
        print(f"Gradient accumulation training failed: {e}")
        print("Skipping gradient accumulation training.")
    
    # If CUDA is available, demonstrate mixed precision training
    if torch.cuda.is_available():
        print("\nTraining with mixed precision...")
        try:
            scaler = torch.cuda.amp.GradScaler()
            trainer.train_epoch_mixed_precision(data_loader, scaler=scaler, log_interval=2)
        except Exception as e:
            print(f"Mixed precision training failed: {e}")
            print("Skipping mixed precision training.")
    else:
        print("\nCUDA not available, skipping mixed precision training.")
    
    # Evaluate model
    print("\nEvaluating model...")
    try:
        metrics = evaluate_model(model, data_loader)
        print(f"Model Metrics:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Skipping evaluation.")
    
    print("\nScaling techniques example completed successfully!")


if __name__ == "__main__":
    main()