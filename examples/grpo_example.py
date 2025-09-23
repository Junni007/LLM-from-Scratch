"""
Example demonstrating GRPO (Group-Relative Policy Optimization) for LLM alignment.

This example shows how to use GRPO with group-relative baselines for training language models
using reinforcement learning from human feedback.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import sys
import os

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rlhf.grpo import GRPOTrainer, GRPOConfig
from src.reward.model import RewardModel

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
    
    # Create reference model (copy of base model)
    ref_model = GPT2LMHeadModel(config)
    ref_model.load_state_dict(base_model.state_dict())
    
    # Create reward model
    reward_model = RewardModel(base_model, hidden_size=128)
    
    return base_model, ref_model, reward_model

def create_sample_prompts(batch_size=2, prompt_length=10):
    """Create sample prompts for demonstration."""
    # Create random token IDs for prompts
    prompts = torch.randint(1, 100, (batch_size, prompt_length))
    return prompts

class SimplePromptDataset(torch.utils.data.Dataset):
    """Simple dataset of prompts for demonstration."""
    
    def __init__(self, num_prompts=10, prompt_length=10):
        self.prompts = torch.randint(1, 100, (num_prompts, prompt_length))
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]

def main():
    """Main function demonstrating GRPO training."""
    print("=== GRPO Example for LLM Alignment ===")
    
    # Create sample models
    print("Creating sample models...")
    policy_model, ref_model, reward_model = create_sample_models()
    
    # Create GRPO configuration
    config = GRPOConfig(
        grpo_epochs=2,
        batch_size=2,
        num_completions_per_prompt=2,
        learning_rate=1e-4
    )
    
    # Create GRPO trainer
    trainer = GRPOTrainer(
        policy_value_model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        config=config
    )
    
    # Create sample prompt dataset
    print("Creating sample prompt dataset...")
    prompt_dataset = SimplePromptDataset(num_prompts=5, prompt_length=10)
    print(f"Dataset size: {len(prompt_dataset)}")
    
    # Perform a training run
    print("Performing GRPO training...")
    history = trainer.train(prompt_dataset, num_groups=3)
    
    print(f"Training completed with {len(history['groups'])} groups")
    print(f"Final average reward: {history['rewards'][-1]:.4f}")
    print(f"Final relative reward: {history['relative_rewards'][-1]:.4f}")
    
    print("\n=== GRPO Example Completed ===")

if __name__ == "__main__":
    main()