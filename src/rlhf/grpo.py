"""
GRPO (Group-Relative Policy Optimization) implementation for LLM alignment.

This module implements GRPO, an alternative to PPO that uses group-relative baselines
for more stable training in reinforcement learning from human feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import math

@dataclass
class GRPOConfig:
    """
    Configuration for GRPO training.
    """
    # GRPO hyperparameters
    grpo_epochs: int = 4
    batch_size: int = 8
    mini_batch_size: int = 1
    clip_epsilon: float = 0.2
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95    # GAE lambda
    
    # KL regularization parameters
    kl_coef: float = 0.1
    kl_target: float = 0.01
    
    # Optimization parameters
    learning_rate: float = 1e-5
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Training parameters
    max_seq_length: int = 512
    generation_max_length: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    
    # Group parameters
    num_completions_per_prompt: int = 4  # Number of completions per prompt for group-relative baseline
    group_size: int = 4  # Size of groups for computing relative rewards


class GroupRelativeBuffer:
    """
    Buffer for storing GRPO experiences with group-relative rewards.
    """
    
    def __init__(self, config: GRPOConfig):
        """
        Initialize GroupRelativeBuffer.
        
        Args:
            config (GRPOConfig): GRPO configuration
        """
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset the buffer."""
        self.prompts = []
        self.responses = []
        self.ref_logprobs = []
        self.values = []
        self.rewards = []
        self.relative_rewards = []
        self.advantages = []
        self.returns = []
        self.groups = []  # Group indices for each sample
        
    def add(self, prompt: torch.Tensor, response: torch.Tensor, 
            ref_logprob: torch.Tensor, value: torch.Tensor, 
            reward: float, group_id: int):
        """
        Add an experience to the buffer.
        
        Args:
            prompt (torch.Tensor): Prompt tokens
            response (torch.Tensor): Response tokens
            ref_logprob (torch.Tensor): Reference log probabilities
            value (torch.Tensor): Value estimates
            reward (float): Raw reward for the response
            group_id (int): Group ID for group-relative baseline
        """
        self.prompts.append(prompt)
        self.responses.append(response)
        self.ref_logprobs.append(ref_logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.groups.append(group_id)
        
    def compute_relative_rewards(self):
        """
        Compute group-relative rewards.
        
        For each group, compute the relative reward as the difference
        between the raw reward and the group mean reward.
        """
        if not self.rewards:
            return
            
        # Convert to tensors
        rewards = torch.tensor(self.rewards)
        groups = torch.tensor(self.groups)
        
        # Compute group means
        unique_groups = torch.unique(groups)
        group_means = {}
        
        for group_id in unique_groups:
            group_mask = (groups == group_id)
            group_rewards = rewards[group_mask]
            group_means[group_id.item()] = group_rewards.mean().item()
        
        # Compute relative rewards
        relative_rewards = []
        for i, (reward, group_id) in enumerate(zip(self.rewards, self.groups)):
            relative_reward = reward - group_means[group_id]
            relative_rewards.append(relative_reward)
            
        self.relative_rewards = relative_rewards
        
    def compute_advantages_returns(self):
        """
        Compute advantages and returns using GAE with relative rewards.
        """
        if not self.values or not self.relative_rewards:
            return
            
        # Convert to tensors
        values = torch.stack(self.values)  # (batch_size, seq_length)
        relative_rewards = torch.tensor(self.relative_rewards, device=values.device)  # (batch_size,)
        
        # Compute advantages using GAE
        batch_size, seq_length = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        # For simplicity, we'll use a simplified GAE computation
        # In practice, this would be more complex
        for i in range(batch_size):
            # Simple advantage computation (could be improved)
            last_value = values[i, -1]
            # Using relative reward for the entire sequence
            advantages[i] = relative_rewards[i] - last_value
            returns[i] = relative_rewards[i]
            
        self.advantages = advantages
        self.returns = returns
        
    def get_batch(self, batch_size: int = None) -> Dict[str, torch.Tensor]:
        """
        Get a batch of experiences.
        
        Args:
            batch_size (int, optional): Batch size. If None, uses config.batch_size.
            
        Returns:
            Dict[str, torch.Tensor]: Batch of experiences
        """
        if batch_size is None:
            batch_size = self.config.batch_size
            
        # For simplicity, return all experiences
        # In practice, this would sample a batch
        return {
            'prompts': torch.stack(self.prompts),
            'responses': torch.stack(self.responses),
            'ref_logprobs': torch.stack(self.ref_logprobs),
            'values': torch.stack(self.values),
            'rewards': torch.tensor(self.rewards),
            'relative_rewards': torch.tensor(self.relative_rewards),
            'advantages': self.advantages,
            'returns': self.returns,
            'groups': torch.tensor(self.groups)
        }


class GRPOTrainer:
    """
    GRPO trainer for LLM alignment using group-relative baselines.
    """
    
    def __init__(self, policy_value_model: nn.Module, 
                 ref_model: nn.Module, 
                 reward_model: nn.Module,
                 config: GRPOConfig):
        """
        Initialize GRPOTrainer.
        
        Args:
            policy_value_model (nn.Module): Policy network with value head
            ref_model (nn.Module): Reference model for KL penalty
            reward_model (nn.Module): Reward model
            config (GRPOConfig): GRPO configuration
        """
        self.policy_value_model = policy_value_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            policy_value_model.parameters(), 
            lr=config.learning_rate,
            eps=config.adam_epsilon
        )
        
        # Buffer
        self.buffer = GroupRelativeBuffer(config)
        
        # Training statistics
        self.training_stats = {
            'episode': 0,
            'total_steps': 0,
            'avg_reward': 0.0,
            'avg_relative_reward': 0.0
        }
        
        # Logging
        self.log_history = []
        
    def compute_rewards(self, prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards using the reward model.
        
        Args:
            prompts (torch.Tensor): Prompt tokens of shape (batch_size, prompt_length)
            responses (torch.Tensor): Response tokens of shape (batch_size, response_length)
            
        Returns:
            torch.Tensor: Rewards of shape (batch_size,)
        """
        batch_size = prompts.shape[0]
        rewards = []
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Concatenate prompt and response
            prompt_resp = torch.cat([prompts[i], responses[i]], dim=0)
            
            # Get reward from reward model
            with torch.no_grad():
                # Ensure the reward model returns a scalar
                reward_output = self.reward_model(prompt_resp.unsqueeze(0))
                if isinstance(reward_output, tuple):
                    reward = reward_output[0]  # Take first element if tuple
                else:
                    reward = reward_output
                    
                # Ensure we get a scalar value
                if reward.dim() > 0:
                    reward = reward.squeeze()
                    
                rewards.append(reward.item())
                
        return torch.tensor(rewards, device=prompts.device)
    
    def compute_logprobs(self, model: nn.Module, input_ids: torch.Tensor, 
                        labels: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities for a model.
        
        Args:
            model (nn.Module): Model to compute log probabilities for
            input_ids (torch.Tensor): Input token IDs
            labels (torch.Tensor): Labels for computing log probabilities
            
        Returns:
            torch.Tensor: Log probabilities
        """
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))
            
            # Reshape and negate to get log probabilities
            logprobs = -loss.view(shift_labels.shape)
            
        return logprobs
    
    def compute_kl_penalty(self, logprob: torch.Tensor, ref_logprob: torch.Tensor) -> torch.Tensor:
        """
        Compute KL penalty between current policy and reference policy.
        
        Args:
            logprob (torch.Tensor): Log probabilities from current policy
            ref_logprob (torch.Tensor): Log probabilities from reference policy
            
        Returns:
            torch.Tensor: KL penalty
        """
        return logprob - ref_logprob
    
    def grpo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single GRPO training step.
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch of experiences
            
        Returns:
            Dict[str, float]: Training metrics
        """
        prompts = batch['prompts']
        responses = batch['responses']
        ref_logprobs = batch['ref_logprobs']
        old_values = batch['values']
        relative_advantages = batch['advantages']
        returns = batch['returns']
        
        # Compute current log probabilities
        # Concatenate prompts and responses for input
        inputs = torch.cat([prompts, responses], dim=1)
        labels = inputs.clone()
        
        # Get current policy log probabilities
        current_outputs = self.policy_value_model.base_model(inputs)
        current_logits = current_outputs.logits
        
        # Shift for next-token prediction
        shift_logits = current_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log probabilities
        logprobs = -F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        ).view(shift_labels.shape)
        
        # Compute ratio (importance sampling)
        ratio = torch.exp(logprobs - ref_logprobs)
        
        # Compute GRPO loss using relative advantages
        surr1 = ratio * relative_advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 
                           1.0 + self.config.clip_epsilon) * relative_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute entropy bonus for stability
        probs = F.softmax(shift_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
        
        # Compute value loss (clipped value loss)
        current_values = self.policy_value_model.value_head(
            current_outputs.last_hidden_state
        ).squeeze(-1)
        
        value_clipped = old_values + torch.clamp(
            current_values - old_values,
            -self.config.clip_epsilon,
            self.config.clip_epsilon
        )
        
        value_loss1 = (current_values - returns) ** 2
        value_loss2 = (value_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        
        # Compute KL penalty
        kl_penalty = self.compute_kl_penalty(logprobs, ref_logprobs).mean()
        
        # Total loss with entropy bonus
        total_loss = (
            policy_loss + 
            0.5 * value_loss - 
            0.01 * entropy +
            self.config.kl_coef * kl_penalty
        )
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_value_model.parameters(), 
                                self.config.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
    
    def train_group(self, prompts: torch.Tensor) -> Dict[str, Any]:
        """
        Perform a complete GRPO training group step.
        
        Args:
            prompts (torch.Tensor): Prompt tokens of shape (batch_size, prompt_length)
            
        Returns:
            Dict[str, Any]: Training results for the group
        """
        batch_size = prompts.shape[0]
        
        # Generate multiple completions for each prompt
        all_responses = []
        all_ref_logprobs = []
        all_values = []
        all_prompts = []
        all_group_ids = []
        
        with torch.no_grad():
            for group_id in range(batch_size):
                prompt = prompts[group_id]
                
                # Generate multiple completions for this prompt
                for completion_id in range(self.config.num_completions_per_prompt):
                    # Generate response
                    response = self.policy_value_model.generate(
                        prompt.unsqueeze(0),
                        max_length=self.config.generation_max_length,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p
                    )
                    
                    # Remove prompt from response
                    response_only = response[:, prompt.shape[0]:]
                    all_responses.append(response_only)
                    all_prompts.append(prompt)
                    all_group_ids.append(group_id)
                    
                    # Get reference log probabilities
                    prompt_resp = torch.cat([prompt, response_only.squeeze(0)], dim=0)
                    ref_output = self.ref_model(prompt_resp.unsqueeze(0))
                    ref_logits = ref_output.logits
                    
                    # Compute reference log probabilities
                    ref_labels = prompt_resp.clone()
                    ref_logprob = -F.cross_entropy(
                        ref_logits[..., :-1, :].contiguous().view(-1, ref_logits.size(-1)),
                        ref_labels[..., 1:].contiguous().view(-1),
                        reduction='none'
                    ).view(ref_labels[..., 1:].shape)
                    all_ref_logprobs.append(ref_logprob)
                    
                    # Get value estimates
                    _, value = self.policy_value_model(prompt_resp.unsqueeze(0))
                    all_values.append(value.squeeze(0))
        
        # Pad responses to same length
        max_resp_len = max([resp.shape[0] for resp in all_responses])
        padded_responses = []
        padded_ref_logprobs = []
        padded_values = []
        padded_prompts = []
        
        for resp, ref_logprob, value, prompt in zip(all_responses, all_ref_logprobs, all_values, all_prompts):
            # Pad response
            pad_len = max_resp_len - resp.shape[0]
            if pad_len > 0:
                resp_padded = F.pad(resp, (0, pad_len), value=0)
                ref_logprob_padded = F.pad(ref_logprob, (0, pad_len), value=0)
                value_padded = F.pad(value, (0, pad_len), value=0)
                prompt_padded = prompt  # Prompt doesn't need padding
            else:
                resp_padded = resp
                ref_logprob_padded = ref_logprob
                value_padded = value
                prompt_padded = prompt
                
            padded_responses.append(resp_padded)
            padded_ref_logprobs.append(ref_logprob_padded)
            padded_values.append(value_padded)
            padded_prompts.append(prompt_padded)
        
        # Stack tensors
        responses_tensor = torch.stack(padded_responses)
        ref_logprobs_tensor = torch.stack(padded_ref_logprobs)
        values_tensor = torch.stack(padded_values)
        prompts_tensor = torch.stack(padded_prompts)
        
        # Compute rewards
        rewards = self.compute_rewards(prompts_tensor, responses_tensor)
        
        # Add to buffer
        for i in range(len(all_responses)):
            self.buffer.add(
                padded_prompts[i],
                padded_responses[i],
                padded_ref_logprobs[i],
                padded_values[i],
                rewards[i].item(),
                all_group_ids[i]
            )
        
        # Compute relative rewards
        self.buffer.compute_relative_rewards()
        
        # Compute advantages and returns
        self.buffer.compute_advantages_returns()
        
        # Perform GRPO updates
        metrics = []
        for epoch in range(self.config.grpo_epochs):
            batch = self.buffer.get_batch()
            epoch_metrics = self.grpo_step(batch)
            metrics.append(epoch_metrics)
        
        # Reset buffer
        self.buffer.reset()
        
        # Update training statistics
        self.training_stats['episode'] += 1
        self.training_stats['total_steps'] += len(metrics)
        self.training_stats['avg_reward'] = (
            self.training_stats['avg_reward'] * (self.training_stats['episode'] - 1) + 
            rewards.mean().item()
        ) / self.training_stats['episode']
        
        # Compute average metrics
        avg_metrics = {}
        for key in metrics[0].keys():
            avg_metrics[key] = sum([m[key] for m in metrics]) / len(metrics)
        
        # Log training data
        log_entry = {
            'episode': self.training_stats['episode'],
            'rewards': rewards.mean().item(),
            'relative_rewards': torch.tensor(self.buffer.relative_rewards).mean().item() if self.buffer.relative_rewards else 0.0,
            'metrics': avg_metrics
        }
        self.log_history.append(log_entry)
        
        return {
            'rewards': rewards.mean().item(),
            'relative_rewards': torch.tensor(self.buffer.relative_rewards).mean().item() if self.buffer.relative_rewards else 0.0,
            'metrics': avg_metrics,
            'training_stats': self.training_stats.copy()
        }
    
    def train(self, prompt_dataset: torch.utils.data.Dataset, 
              num_groups: int = 100) -> Dict[str, Any]:
        """
        Complete GRPO training loop.
        
        Args:
            prompt_dataset (torch.utils.data.Dataset): Dataset of prompts
            num_groups (int): Number of training groups
            
        Returns:
            Dict[str, Any]: Training results
        """
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            prompt_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # Training history
        history = {
            'groups': [],
            'rewards': [],
            'relative_rewards': [],
            'metrics': []
        }
        
        # Training loop
        for group in range(num_groups):
            # Get a batch of prompts
            try:
                prompts_batch = next(iter(dataloader))
            except StopIteration:
                # Reset dataloader if exhausted
                dataloader = torch.utils.data.DataLoader(
                    prompt_dataset, 
                    batch_size=self.config.batch_size, 
                    shuffle=True
                )
                prompts_batch = next(iter(dataloader))
            
            # Train on group
            group_results = self.train_group(prompts_batch)
            
            # Record results
            history['groups'].append(group)
            history['rewards'].append(group_results['rewards'])
            history['relative_rewards'].append(group_results['relative_rewards'])
            history['metrics'].append(group_results['metrics'])
            
            # Print progress
            if group % 10 == 0:
                print(f"Group {group}: "
                      f"Average Reward = {group_results['rewards']:.4f}, "
                      f"Relative Reward = {group_results['relative_rewards']:.4f}, "
                      f"Policy Loss = {group_results['metrics']['policy_loss']:.4f}")
        
        return history
    
    def save_checkpoint(self, filepath: str):
        """
        Save GRPO trainer checkpoint.
        
        Args:
            filepath (str): Path to save checkpoint
        """
        checkpoint = {
            'policy_model_state_dict': self.policy_value_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats,
            'log_history': self.log_history
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load GRPO trainer checkpoint.
        
        Args:
            filepath (str): Path to load checkpoint from
        """
        checkpoint = torch.load(filepath)
        self.policy_value_model.load_state_dict(checkpoint['policy_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.training_stats = checkpoint['training_stats']
        self.log_history = checkpoint['log_history']
        print(f"Checkpoint loaded from {filepath}")