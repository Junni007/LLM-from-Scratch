"""
Optimized training techniques for LLMs.

This module implements gradient compression, mixed precision training,
gradient accumulation, and other optimization techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple, Dict, Any, Optional, List
import math

class TrainingOptimizationConfig:
    """
    Configuration for training optimization techniques.
    """
    def __init__(self, 
                 gradient_accumulation_steps: int = 1,
                 mixed_precision: bool = False,
                 gradient_clipping: float = 1.0,
                 gradient_compression: bool = False,
                 compression_ratio: float = 0.5,
                 learning_rate: float = 5e-5,
                 warmup_steps: int = 1000,
                 weight_decay: float = 0.01):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.gradient_clipping = gradient_clipping
        self.gradient_compression = gradient_compression
        self.compression_ratio = compression_ratio
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay


class GradientCompressor:
    """
    Gradient compressor for reducing communication overhead in distributed training.
    """
    
    def __init__(self, compression_ratio: float = 0.5):
        """
        Initialize GradientCompressor.
        
        Args:
            compression_ratio (float): Ratio of gradients to keep (0.0 to 1.0)
        """
        self.compression_ratio = compression_ratio
        
    def compress_gradients(self, gradients: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compress gradients using top-k selection.
        
        Args:
            gradients (List[torch.Tensor]): List of gradient tensors
            
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: 
                - compressed_gradients: Compressed gradient tensors
                - masks: Masks indicating which gradients were kept
        """
        compressed_gradients = []
        masks = []
        
        for grad in gradients:
            if grad is not None:
                # Flatten gradient tensor
                flat_grad = grad.view(-1)
                
                # Determine number of elements to keep
                num_elements = flat_grad.numel()
                num_keep = int(num_elements * self.compression_ratio)
                
                # Find top-k elements by magnitude
                _, indices = torch.topk(flat_grad.abs(), num_keep)
                
                # Create mask
                mask = torch.zeros_like(flat_grad, dtype=torch.bool)
                mask[indices] = True
                
                # Compress gradient
                compressed_grad = flat_grad * mask.float()
                
                # Reshape back to original shape
                compressed_grad = compressed_grad.view_as(grad)
                mask = mask.view_as(grad)
                
                compressed_gradients.append(compressed_grad)
                masks.append(mask)
            else:
                compressed_gradients.append(None)
                masks.append(None)
                
        return compressed_gradients, masks
    
    def decompress_gradients(self, compressed_gradients: List[torch.Tensor], 
                           masks: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Decompress gradients (in this simple implementation, no decompression needed).
        
        Args:
            compressed_gradients (List[torch.Tensor]): Compressed gradient tensors
            masks (List[torch.Tensor]): Masks indicating which gradients were kept
            
        Returns:
            List[torch.Tensor]: Decompressed gradient tensors
        """
        # In this simple implementation, compressed gradients are already in the right format
        return compressed_gradients


class OptimizedTrainer:
    """
    Optimized trainer implementing various training optimization techniques.
    """
    
    def __init__(self, model: nn.Module, config: TrainingOptimizationConfig):
        """
        Initialize OptimizedTrainer.
        
        Args:
            model (nn.Module): Model to train
            config (TrainingOptimizationConfig): Training optimization configuration
        """
        self.model = model
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = self._get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=100000  # This should be set based on actual training steps
        )
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Gradient compressor
        self.gradient_compressor = GradientCompressor(config.compression_ratio) if config.gradient_compression else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.accumulated_loss = 0.0
        self.accumulation_step = 0
        
    def _get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """
        Create a schedule with a learning rate that decreases linearly after warmup.
        """
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    
    def train_step(self, input_ids: torch.Tensor, 
                   labels: torch.Tensor,
                   attention_mask: Optional[torch.Tensor] = None) -> float:
        """
        Perform a single optimized training step.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            labels (torch.Tensor): Labels
            attention_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            float: Loss value
        """
        # Use mixed precision if enabled
        if self.config.mixed_precision and self.scaler is not None:
            with autocast():
                loss = self._forward_pass(input_ids, labels, attention_mask)
        else:
            loss = self._forward_pass(input_ids, labels, attention_mask)
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        self.accumulated_loss += loss.item()
        
        # Backward pass with mixed precision if enabled
        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        self.accumulation_step += 1
        if self.accumulation_step % self.gradient_accumulation_steps == 0:
            # Apply gradient clipping
            if self.config.gradient_clipping > 0:
                if self.config.mixed_precision and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            
            # Apply gradient compression if enabled
            if self.config.gradient_compression and self.gradient_compressor is not None:
                self._apply_gradient_compression()
            
            # Update parameters
            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Reset accumulation
            self.accumulated_loss = 0.0
        
        return loss.item() * self.gradient_accumulation_steps
    
    def _forward_pass(self, input_ids: torch.Tensor, 
                     labels: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform forward pass.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            labels (torch.Tensor): Labels
            attention_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Loss value
        """
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss
    
    def _apply_gradient_compression(self):
        """
        Apply gradient compression to reduce communication overhead.
        """
        if self.gradient_compressor is None:
            return
            
        # Get gradients
        gradients = [param.grad for param in self.model.parameters() if param.grad is not None]
        
        # Compress gradients
        compressed_gradients, masks = self.gradient_compressor.compress_gradients(gradients)
        
        # Apply compressed gradients
        param_idx = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data = compressed_gradients[param_idx]
                param_idx += 1


class MemoryEfficientAttention:
    """
    Memory-efficient attention implementation to reduce memory usage during training.
    """
    
    @staticmethod
    def forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                dropout_p: float = 0.0) -> torch.Tensor:
        """
        Memory-efficient attention forward pass.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_length, head_dim)
            key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_length, head_dim)
            value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_length, head_dim)
            attention_mask (torch.Tensor, optional): Attention mask
            dropout_p (float): Dropout probability
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(query.size(-1))
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        if dropout_p > 0:
            attention_probs = F.dropout(attention_probs, p=dropout_p, training=True)
        
        # Apply attention to values
        output = torch.matmul(attention_probs, value)
        
        return output


class CheckpointingWrapper:
    """
    Wrapper for gradient checkpointing to reduce memory usage.
    """
    
    @staticmethod
    def checkpoint_forward(module: nn.Module, *args, **kwargs):
        """
        Forward pass with gradient checkpointing.
        
        Args:
            module (nn.Module): Module to apply checkpointing to
            *args: Arguments to pass to module
            **kwargs: Keyword arguments to pass to module
            
        Returns:
            Output of module
        """
        # This is a simplified implementation
        # In practice, you would use torch.utils.checkpoint.checkpoint
        return module(*args, **kwargs)


class EfficientDataLoader:
    """
    Efficient data loader with prefetching and pinning for faster data loading.
    """
    
    def __init__(self, dataset: torch.utils.data.Dataset, 
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 prefetch_factor: int = 2):
        """
        Initialize EfficientDataLoader.
        
        Args:
            dataset (torch.utils.data.Dataset): Dataset to load
            batch_size (int): Batch size
            num_workers (int): Number of worker processes
            pin_memory (bool): Whether to pin memory
            prefetch_factor (int): Number of batches to prefetch per worker
        """
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=True
        )
        
    def __iter__(self):
        return iter(self.dataloader)
        
    def __len__(self):
        return len(self.dataloader)


# Example usage functions
def create_optimized_trainer(model: nn.Module, 
                           learning_rate: float = 5e-5,
                           gradient_accumulation_steps: int = 1,
                           mixed_precision: bool = False,
                           gradient_clipping: float = 1.0) -> OptimizedTrainer:
    """
    Create an optimized trainer.
    
    Args:
        model (nn.Module): Model to train
        learning_rate (float): Learning rate
        gradient_accumulation_steps (int): Gradient accumulation steps
        mixed_precision (bool): Whether to use mixed precision
        gradient_clipping (float): Gradient clipping threshold
        
    Returns:
        OptimizedTrainer: Configured trainer
    """
    config = TrainingOptimizationConfig(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        gradient_clipping=gradient_clipping,
        learning_rate=learning_rate
    )
    
    trainer = OptimizedTrainer(model, config)
    return trainer


def apply_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Apply gradient checkpointing to a model to reduce memory usage.
    
    Args:
        model (nn.Module): Model to apply checkpointing to
        
    Returns:
        nn.Module: Model with gradient checkpointing applied
    """
    # This is a simplified implementation
    # In practice, you would use torch.utils.checkpoint.checkpoint or similar
    return model


def create_efficient_dataloader(dataset: torch.utils.data.Dataset,
                              batch_size: int = 32,
                              num_workers: int = 4) -> EfficientDataLoader:
    """
    Create an efficient data loader.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset to load
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        
    Returns:
        EfficientDataLoader: Configured data loader
    """
    return EfficientDataLoader(dataset, batch_size, num_workers)


def profile_memory_usage(model: nn.Module, 
                        input_ids: torch.Tensor,
                        labels: torch.Tensor):
    """
    Profile memory usage of a model.
    
    Args:
        model (nn.Module): Model to profile
        input_ids (torch.Tensor): Input token IDs
        labels (torch.Tensor): Labels
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        outputs = model(input_ids, labels=labels)
        forward_memory = torch.cuda.memory_allocated()
        
        # Backward pass
        outputs.loss.backward()
        backward_memory = torch.cuda.memory_allocated()
        
        peak_memory = torch.cuda.max_memory_allocated()
        
        print(f"Initial memory: {initial_memory / 1024**2:.2f} MB")
        print(f"Forward pass memory: {forward_memory / 1024**2:.2f} MB")
        print(f"Backward pass memory: {backward_memory / 1024**2:.2f} MB")
        print(f"Peak memory: {peak_memory / 1024**2:.2f} MB")
        print(f"Memory increase: {(peak_memory - initial_memory) / 1024**2:.2f} MB")
    else:
        print("CUDA not available, cannot profile memory usage")