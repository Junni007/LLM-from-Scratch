"""
PPO (Proximal Policy Optimization) implementation for LLM alignment.

This module implements PPO with KL penalty for training language models
using reinforcement learning from human feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import math

@dataclass
class PPOConfig:
    """
    Configuration for PPO training.
    """
    # PPO hyperparameters
    ppo_epochs: int = 4
    batch_size: int = 8
    mini_batch_size: int = 1
    clip_epsilon: float = 0.2
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95    # GAE lambda
    
    # KL penalty parameters
    kl_coef: float = 0.2
    kl_target: float = 0.02
    kl_horizon: int = 10000
    
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
    
    # Stability parameters
    clip_value_loss: bool = True
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    early_stopping_kl: float = 0.1


class PolicyValueNetwork(nn.Module):
    """
    Policy network with value head for PPO.
    
    This network extends a base language model with a value head
    for estimating state values in the PPO algorithm.
    """
    
    def __init__(self, base_model: nn.Module, hidden_size: int = 768):
        """
        Initialize PolicyValueNetwork.
        
        Args:
            base_model (nn.Module): Base language model (e.g., Transformer)
            hidden_size (int): Hidden size of the base model
        """
        super(PolicyValueNetwork, self).__init__()
        self.base_model = base_model
        
        # Value head - maps hidden states to scalar values
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy-value network.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_length)
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_length)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - logits: Language model logits of shape (batch_size, seq_length, vocab_size)
                - values: Value estimates of shape (batch_size, seq_length)
        """
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        
        # Get logits from base model
        logits = outputs.logits  # (batch_size, seq_length, vocab_size)
        
        # Compute values from hidden states
        values = self.value_head(hidden_states).squeeze(-1)  # (batch_size, seq_length)
        
        return logits, values
    
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                 max_length: int = 128, temperature: float = 1.0, 
                 top_k: int = 50, top_p: float = 0.95) -> torch.Tensor:
        """
        Generate text using the policy network.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_length)
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_length)
            max_length (int): Maximum length of generated sequence
            temperature (float): Temperature for sampling
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p (nucleus) sampling parameter
            
        Returns:
            torch.Tensor: Generated token IDs of shape (batch_size, generated_length)
        """
        batch_size, seq_length = input_ids.shape
        generated = input_ids.clone()
        
        # Initialize attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Get logits for the last token
            logits, _ = self.forward(generated, attention_mask)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k and top-p filtering
            next_token_logits = self._top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones(batch_size, 1, device=attention_mask.device)
            ], dim=-1)
            
            # Check for end of sequence token (assuming 0 is EOS)
            if (next_token == 0).all():
                break
                
        return generated
    
    def _top_k_top_p_filtering(self, logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
        """
        Filter logits using top-k and top-p (nucleus) sampling.
        
        Args:
            logits (torch.Tensor): Logits to filter
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p (nucleus) sampling parameter
            
        Returns:
            torch.Tensor: Filtered logits
        """
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            # Remove all tokens with probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        return logits


class PPOBuffer:
    """
    Buffer for storing PPO experiences.
    """
    
    def __init__(self, config: PPOConfig):
        """
        Initialize PPOBuffer.
        
        Args:
            config (PPOConfig): PPO configuration
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
        self.advantages = []
        self.returns = []
        
    def add(self, prompt: torch.Tensor, response: torch.Tensor, 
            ref_logprob: torch.Tensor, value: torch.Tensor, 
            reward: float):
        """
        Add an experience to the buffer.
        
        Args:
            prompt (torch.Tensor): Prompt tokens
            response (torch.Tensor): Response tokens
            ref_logprob (torch.Tensor): Reference log probabilities
            value (torch.Tensor): Value estimates
            reward (float): Reward for the response
        """
        self.prompts.append(prompt)
        self.responses.append(response)
        self.ref_logprobs.append(ref_logprob)
        self.values.append(value)
        self.rewards.append(reward)
        
    def compute_advantages_returns(self):
        """
        Compute advantages and returns using GAE.
        """
        # Convert to tensors
        values = torch.stack(self.values)  # (batch_size, seq_length)
        rewards = torch.tensor(self.rewards, device=values.device)  # (batch_size,)
        
        # Compute advantages using GAE
        batch_size, seq_length = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        # For simplicity, we'll use a simplified GAE computation
        # In practice, this would be more complex
        for i in range(batch_size):
            # Simple advantage computation (could be improved)
            last_value = values[i, -1]
            # Assuming reward is for the entire sequence
            advantages[i] = rewards[i] - last_value
            returns[i] = rewards[i]
            
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
            'advantages': self.advantages,
            'returns': self.returns
        }


class PPOTrainer:
    """
    PPO trainer for LLM alignment.
    """
    
    def __init__(self, policy_value_model: PolicyValueNetwork, 
                 ref_model: nn.Module, 
                 reward_model: nn.Module,
                 config: PPOConfig):
        """
        Initialize PPOTrainer.
        
        Args:
            policy_value_model (PolicyValueNetwork): Policy network with value head
            ref_model (nn.Module): Reference model for KL penalty
            reward_model (nn.Module): Reward model
            config (PPOConfig): PPO configuration
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
        self.buffer = PPOBuffer(config)
        
        # Training statistics
        self.training_stats = {
            'episode': 0,
            'total_steps': 0,
            'avg_reward': 0.0,
            'avg_kl_div': 0.0
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
    
    def compute_ppo_objective(self, logprob: torch.Tensor, ref_logprob: torch.Tensor,
                             advantage: torch.Tensor, old_logprob: torch.Tensor) -> torch.Tensor:
        """
        Compute PPO objective with KL penalty.
        
        Args:
            logprob (torch.Tensor): Current policy log probabilities
            ref_logprob (torch.Tensor): Reference policy log probabilities
            advantage (torch.Tensor): Advantage estimates
            old_logprob (torch.Tensor): Old policy log probabilities
            
        Returns:
            torch.Tensor: PPO objective with KL penalty
        """
        # Compute ratio (importance sampling)
        ratio = torch.exp(logprob - old_logprob)
        
        # Compute clipped PPO objective
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 
                           1.0 + self.config.clip_epsilon) * advantage
        ppo_objective = torch.min(surr1, surr2)
        
        # Compute KL penalty
        kl_penalty = self.compute_kl_penalty(logprob, ref_logprob)
        
        # Combine PPO objective with KL penalty
        objective = ppo_objective - self.config.kl_coef * kl_penalty
        
        return objective.mean()
    
    def ppo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single PPO training step.
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch of experiences
            
        Returns:
            Dict[str, float]: Training metrics
        """
        prompts = batch['prompts']
        responses = batch['responses']
        ref_logprobs = batch['ref_logprobs']
        old_values = batch['values']
        advantages = batch['advantages']
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
        
        # Compute PPO loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 
                           1.0 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute entropy bonus for stability
        probs = F.softmax(shift_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
        
        # Compute value loss (clipped value loss)
        current_values = self.policy_value_model.value_head(
            current_outputs.last_hidden_state
        ).squeeze(-1)
        
        if self.config.clip_value_loss:
            value_clipped = old_values + torch.clamp(
                current_values - old_values,
                -self.config.clip_epsilon,
                self.config.clip_epsilon
            )
            
            value_loss1 = (current_values - returns) ** 2
            value_loss2 = (value_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = 0.5 * ((current_values - returns) ** 2).mean()
        
        # Compute KL penalty
        kl_penalty = self.compute_kl_penalty(logprobs, ref_logprobs).mean()
        
        # Total loss with entropy bonus
        total_loss = (
            policy_loss + 
            self.config.value_loss_coef * value_loss - 
            self.config.entropy_coef * entropy +
            self.config.kl_coef * kl_penalty
        )
        
        # Early stopping based on KL divergence
        if kl_penalty > self.config.early_stopping_kl:
            print(f"Early stopping due to high KL divergence: {kl_penalty:.4f}")
            return {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'kl_penalty': kl_penalty.item(),
                'entropy': entropy.item(),
                'total_loss': total_loss.item(),
                'early_stopped': True
            }
        
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
            'total_loss': total_loss.item(),
            'early_stopped': False
        }
    
    def train_episode(self, prompts: torch.Tensor) -> Dict[str, Any]:
        """
        Perform a complete PPO training episode.
        
        Args:
            prompts (torch.Tensor): Prompt tokens of shape (batch_size, prompt_length)
            
        Returns:
            Dict[str, Any]: Training results for the episode
        """
        batch_size = prompts.shape[0]
        
        # Generate responses using current policy
        responses = []
        ref_logprobs = []
        values = []
        
        with torch.no_grad():
            for i in range(batch_size):
                # Generate response
                response = self.policy_value_model.generate(
                    prompts[i].unsqueeze(0),
                    max_length=self.config.generation_max_length,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p
                )
                
                # Remove prompt from response
                response_only = response[:, prompts[i].shape[0]:]
                responses.append(response_only)
                
                # Get reference log probabilities
                prompt_resp = torch.cat([prompts[i], response_only.squeeze(0)], dim=0)
                ref_output = self.ref_model(prompt_resp.unsqueeze(0))
                ref_logits = ref_output.logits
                
                # Compute reference log probabilities
                ref_labels = prompt_resp.clone()
                ref_logprob = -F.cross_entropy(
                    ref_logits[..., :-1, :].contiguous().view(-1, ref_logits.size(-1)),
                    ref_labels[..., 1:].contiguous().view(-1),
                    reduction='none'
                ).view(ref_labels[..., 1:].shape)
                ref_logprobs.append(ref_logprob)
                
                # Get value estimates
                _, value = self.policy_value_model(prompt_resp.unsqueeze(0))
                values.append(value.squeeze(0))
        
        # Pad responses to same length
        max_resp_len = max([resp.shape[0] for resp in responses])
        padded_responses = []
        padded_ref_logprobs = []
        padded_values = []
        
        for resp, ref_logprob, value in zip(responses, ref_logprobs, values):
            # Pad response
            pad_len = max_resp_len - resp.shape[0]
            if pad_len > 0:
                resp_padded = F.pad(resp, (0, pad_len), value=0)
                ref_logprob_padded = F.pad(ref_logprob, (0, pad_len), value=0)
                value_padded = F.pad(value, (0, pad_len), value=0)
            else:
                resp_padded = resp
                ref_logprob_padded = ref_logprob
                value_padded = value
                
            padded_responses.append(resp_padded)
            padded_ref_logprobs.append(ref_logprob_padded)
            padded_values.append(value_padded)
        
        # Stack tensors
        responses_tensor = torch.stack(padded_responses)
        ref_logprobs_tensor = torch.stack(padded_ref_logprobs)
        values_tensor = torch.stack(padded_values)
        
        # Compute rewards
        rewards = self.compute_rewards(prompts, responses_tensor)
        
        # Add to buffer
        for i in range(batch_size):
            self.buffer.add(
                prompts[i],
                responses_tensor[i],
                ref_logprobs_tensor[i],
                values_tensor[i],
                rewards[i].item()
            )
        
        # Compute advantages and returns
        self.buffer.compute_advantages_returns()
        
        # Perform PPO updates
        metrics = []
        early_stopped = False
        for epoch in range(self.config.ppo_epochs):
            batch = self.buffer.get_batch()
            epoch_metrics = self.ppo_step(batch)
            metrics.append(epoch_metrics)
            
            # Check for early stopping
            if epoch_metrics.get('early_stopped', False):
                early_stopped = True
                break
        
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
            if key != 'early_stopped':
                avg_metrics[key] = sum([m[key] for m in metrics]) / len(metrics)
        
        # Log training data
        log_entry = {
            'episode': self.training_stats['episode'],
            'rewards': rewards.mean().item(),
            'metrics': avg_metrics,
            'early_stopped': early_stopped
        }
        self.log_history.append(log_entry)
        
        return {
            'rewards': rewards.mean().item(),
            'metrics': avg_metrics,
            'training_stats': self.training_stats.copy(),
            'early_stopped': early_stopped
        }
    
    def train(self, prompt_dataset: torch.utils.data.Dataset, 
              num_episodes: int = 100) -> Dict[str, Any]:
        """
        Complete PPO training loop.
        
        Args:
            prompt_dataset (torch.utils.data.Dataset): Dataset of prompts
            num_episodes (int): Number of training episodes
            
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
            'episodes': [],
            'rewards': [],
            'metrics': []
        }
        
        # Training loop
        for episode in range(num_episodes):
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
            
            # Train on episode
            episode_results = self.train_episode(prompts_batch)
            
            # Record results
            history['episodes'].append(episode)
            history['rewards'].append(episode_results['rewards'])
            history['metrics'].append(episode_results['metrics'])
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}: "
                      f"Average Reward = {episode_results['rewards']:.4f}, "
                      f"Policy Loss = {episode_results['metrics']['policy_loss']:.4f}, "
                      f"KL Penalty = {episode_results['metrics']['kl_penalty']:.4f}")
                
                # Print early stopping info
                if episode_results.get('early_stopped', False):
                    print(f"  Early stopped due to high KL divergence")
        
        return history
    
    def save_checkpoint(self, filepath: str):
        """
        Save PPO trainer checkpoint.
        
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
        Load PPO trainer checkpoint.
        
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
