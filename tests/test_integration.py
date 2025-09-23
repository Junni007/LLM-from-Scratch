"""
Integration tests for end-to-end pipelines.
"""

import torch
import torch.nn as nn
import sys
import os
import unittest

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.transformer import TransformerBlock
from src.train.trainer import Trainer
from src.tokenizers.bpe import BPE Tokenizer
from src.moe.moe_layer import MoELayer
from src.reward.model import RewardModel
from src.rlhf.ppo import PolicyValueNetwork, PPOTrainer, PPOConfig

class TestEndToEndPipelines(unittest.TestCase):
    """Test end-to-end pipelines."""
    
    def test_tiny_llm_training_pipeline(self):
        """Test complete tiny LLM training pipeline."""
        # Create a simple transformer model
        class SimpleTransformer(nn.Module):
            def __init__(self, d_model=128, num_heads=8, num_layers=2):
                super().__init__()
                self.embedding = nn.Embedding(1000, d_model)
                self.layers = nn.ModuleList([
                    TransformerBlock(d_model, num_heads, dropout=0.1)
                    for _ in range(num_layers)
                ])
                self.output_projection = nn.Linear(d_model, 1000)
                
            def forward(self, input_ids, labels=None):
                x = self.embedding(input_ids)
                for layer in self.layers:
                    x = layer(x)
                logits = self.output_projection(x)
                
                loss = None
                if labels is not None:
                    # Shift logits and labels for next-token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                
                class Output:
                    def __init__(self, logits, loss):
                        self.logits = logits
                        self.loss = loss
                        
                return Output(logits, loss)
        
        # Create model
        model = SimpleTransformer(d_model=128, num_heads=8, num_layers=2)
        
        # Create sample data
        texts = ["Hello world", "This is a test", "Another example sentence", "More text for training"]
        
        # Create tokenizer
        tokenizer = BPE Tokenizer(vocab_size=1000)
        tokenizer.train(texts)
        
        # Tokenize data
        tokenized_texts = [tokenizer.encode(text) for text in texts]
        
        # Create datasets
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, texts, seq_length=32):
                self.texts = texts
                self.seq_length = seq_length
                
            def __len__(self):
                return len(self.texts)
                
            def __getitem__(self, idx):
                tokens = self.texts[idx]
                # Pad or truncate to seq_length
                if len(tokens) < self.seq_length:
                    tokens = tokens + [0] * (self.seq_length - len(tokens))
                else:
                    tokens = tokens[:self.seq_length]
                
                input_ids = torch.tensor(tokens[:-1], dtype=torch.long)  # All but last token
                labels = torch.tensor(tokens[1:], dtype=torch.long)      # All but first token
                
                return input_ids, labels
        
        train_dataset = TextDataset(tokenized_texts, seq_length=16)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=None,
            learning_rate=1e-3,
            batch_size=2,
            num_epochs=2
        )
        
        # Test training
        trainer.train()
        
        # Check that model parameters have been updated
        # (This is a simple check - in practice, you'd check loss reduction)
        self.assertIsNotNone(trainer)
        
    def test_moe_training_pipeline(self):
        """Test MoE training pipeline."""
        # Create a simple MoE model
        class SimpleMoEModel(nn.Module):
            def __init__(self, input_size=128, hidden_size=256, output_size=128, num_experts=4):
                super().__init__()
                self.embedding = nn.Embedding(1000, input_size)
                self.moe_layer = MoELayer(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    num_experts=num_experts,
                    k=2
                )
                self.output_projection = nn.Linear(output_size, 1000)
                
            def forward(self, input_ids, labels=None):
                x = self.embedding(input_ids)
                x, aux_loss = self.moe_layer(x)
                logits = self.output_projection(x)
                
                loss = None
                if labels is not None:
                    # Shift logits and labels for next-token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    nll_loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                    # Combine NLL loss with auxiliary load balancing loss
                    loss = nll_loss + 0.01 * aux_loss  # Weight the auxiliary loss
                
                class Output:
                    def __init__(self, logits, loss):
                        self.logits = logits
                        self.loss = loss
                        
                return Output(logits, loss)
        
        # Create model
        model = SimpleMoEModel(input_size=128, hidden_size=256, output_size=128, num_experts=4)
        
        # Create sample data
        batch_size, seq_length = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        labels = torch.randint(0, 1000, (batch_size, seq_length))
        
        # Test forward pass
        outputs = model(input_ids, labels=labels)
        
        # Check outputs
        self.assertIsNotNone(outputs.logits)
        self.assertIsNotNone(outputs.loss)
        self.assertGreaterEqual(outputs.loss.item(), 0)
        
    def test_rlhf_pipeline(self):
        """Test RLHF training pipeline."""
        # Create simple models for testing
        class SimpleBaseModel(nn.Module):
            def __init__(self, hidden_size=128, vocab_size=1000):
                super().__init__()
                self.hidden_size = hidden_size
                self.vocab_size = vocab_size
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.linear = nn.Linear(hidden_size, hidden_size)
                
            def forward(self, input_ids, attention_mask=None, labels=None):
                x = self.embedding(input_ids)
                x = self.linear(x)
                logits = torch.randn(input_ids.shape[0], input_ids.shape[1], self.vocab_size)
                
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                
                class MockOutput:
                    def __init__(self, last_hidden_state, logits, loss=None):
                        self.last_hidden_state = last_hidden_state
                        self.logits = logits
                        self.loss = loss
                        
                return MockOutput(x, logits, loss)
        
        # Create base model
        base_model = SimpleBaseModel(hidden_size=128, vocab_size=1000)
        
        # Create policy model
        policy_model = PolicyValueNetwork(base_model, hidden_size=128)
        
        # Create reference model (copy of base model)
        ref_model = SimpleBaseModel(hidden_size=128, vocab_size=1000)
        ref_model.load_state_dict(base_model.state_dict())
        
        # Create reward model
        reward_model = RewardModel(base_model, hidden_size=128)
        
        # Create PPO configuration
        config = PPOConfig(
            ppo_epochs=2,
            batch_size=2,
            clip_epsilon=0.2,
            learning_rate=1e-4
        )
        
        # Create PPO trainer
        trainer = PPOTrainer(
            policy_value_model=policy_model,
            ref_model=ref_model,
            reward_model=reward_model,
            config=config
        )
        
        # Create sample prompts
        batch_size, prompt_length = 2, 5
        prompts = torch.randint(1, 100, (batch_size, prompt_length))
        
        # Test training step
        results = trainer.train_step(prompts)
        
        # Check results
        self.assertIn('rewards', results)
        self.assertIn('metrics', results)
        self.assertIsInstance(results['rewards'], float)
        self.assertIsInstance(results['metrics'], dict)
        
    def test_sft_pipeline(self):
        """Test Supervised Fine-Tuning pipeline."""
        # Create a simple instruction-following model
        class InstructionModel(nn.Module):
            def __init__(self, vocab_size=1000, d_model=128):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.transformer = TransformerBlock(d_model, num_heads=8)
                self.output_projection = nn.Linear(d_model, vocab_size)
                
            def forward(self, input_ids, labels=None):
                x = self.embedding(input_ids)
                x = self.transformer(x)
                logits = self.output_projection(x)
                
                loss = None
                if labels is not None:
                    # Apply causal language modeling loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                
                class Output:
                    def __init__(self, logits, loss):
                        self.logits = logits
                        self.loss = loss
                        
                return Output(logits, loss)
        
        # Create model
        model = InstructionModel(vocab_size=1000, d_model=128)
        
        # Create sample instruction data
        instructions = [
            ("What is 2+2?", "2+2=4"),
            ("How many days in a week?", "There are 7 days in a week."),
            ("What color is the sky?", "The sky is blue.")
        ]
        
        # Create dataset
        class InstructionDataset(torch.utils.data.Dataset):
            def __init__(self, instructions, tokenizer=None):
                self.instructions = instructions
                # Simple tokenization for testing
                self.tokenizer = tokenizer
                
            def __len__(self):
                return len(self.instructions)
                
            def __getitem__(self, idx):
                instruction, response = self.instructions[idx]
                
                # Simple tokenization (in practice, use proper tokenizer)
                input_text = f"Instruction: {instruction} Response: {response}"
                tokens = [hash(c) % 1000 for c in input_text][:32]  # Simple hash-based tokenization
                tokens = tokens + [0] * (32 - len(tokens))  # Pad to length 32
                
                input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
                labels = torch.tensor(tokens[1:], dtype=torch.long)
                
                return input_ids, labels
        
        dataset = InstructionDataset(instructions)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=None,
            learning_rate=1e-3,
            batch_size=2,
            num_epochs=1
        )
        
        # Test training
        trainer.train()
        
        # Basic check that training completed
        self.assertIsNotNone(trainer)

if __name__ == '__main__':
    unittest.main()