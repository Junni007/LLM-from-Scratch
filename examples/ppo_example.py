"""
Example demonstrating PPO (Proximal Policy Optimization) for LLM alignment.

This example shows how to use PPO with KL penalty for training language models
using reinforcement learning from human feedback.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import sys
import os

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rlhf.ppo import PolicyValueNetwork, PPOTrainer, PPOConfig
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
    
    # Create policy-value network
    policy_model = PolicyValueNetwork(base_model, hidden_size=128)
    
    # Create reference model (copy of base model)
    ref_model = GPT2LMHeadModel(config)
    ref_model.load_state_dict(base_model.state_dict())
    
    # Create reward model
    reward_model = RewardModel(base_model, hidden_size=128)
    
    return policy_model, ref_model, reward_model

def create_sample_prompts(batch_size=2, prompt_length=10):
    """Create sample prompts for demonstration."""
    # Create random token IDs for prompts
    prompts = torch.randint(1, 100, (batch_size, prompt_length))
    return prompts

def main():
    """Main function demonstrating PPO training."""
    print("=== PPO Example for LLM Alignment ===")
    
    # Create sample models
    print("Creating sample models...")
    policy_model, ref_model, reward_model = create_sample_models()
    
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
    print("Creating sample prompts...")
    prompts = create_sample_prompts(batch_size=2, prompt_length=10)
    print(f"Sample prompts shape: {prompts.shape}")
    
    # Perform a training step
    print("Performing PPO training step...")
    results = trainer.train_step(prompts)
    
    print(f"Average reward: {results['rewards']:.4f}")
    print("Training metrics:")
    for key, value in results['metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n=== PPO Example Completed ===")

if __name__ == "__main__":
    main()