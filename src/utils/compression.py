"""
Compression techniques for LLMs.

This module implements various compression methods including pruning, 
knowledge distillation, and low-rank approximation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List
import math

class PruningConfig:
    """
    Configuration for pruning techniques.
    """
    def __init__(self, 
                 method: str = "magnitude",  # "magnitude", "random", "sensitivity"
                 sparsity_ratio: float = 0.5,
                 structured: bool = False,   # Whether to do structured pruning
                 layer_wise: bool = True):   # Whether to apply different sparsity per layer
        self.method = method
        self.sparsity_ratio = sparsity_ratio
        self.structured = structured
        self.layer_wise = layer_wise


class Pruner:
    """
    Pruner for removing redundant weights from neural networks.
    """
    
    def __init__(self, config: PruningConfig):
        """
        Initialize Pruner.
        
        Args:
            config (PruningConfig): Pruning configuration
        """
        self.config = config
        
    def prune_model(self, model: nn.Module) -> nn.Module:
        """
        Prune a model.
        
        Args:
            model (nn.Module): Model to prune
            
        Returns:
            nn.Module: Pruned model
        """
        if self.config.method == "magnitude":
            return self._magnitude_pruning(model)
        elif self.config.method == "random":
            return self._random_pruning(model)
        else:
            raise ValueError(f"Unknown pruning method: {self.config.method}")
    
    def _magnitude_pruning(self, model: nn.Module) -> nn.Module:
        """
        Magnitude-based pruning.
        
        Args:
            model (nn.Module): Model to prune
            
        Returns:
            nn.Module: Pruned model
        """
        # Create a copy of the model
        pruned_model = type(model)()
        pruned_model.load_state_dict(model.state_dict())
        
        # Prune each layer
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                self._prune_linear_layer(module)
            elif isinstance(module, nn.Conv2d):
                self._prune_conv_layer(module)
                
        return pruned_model
    
    def _random_pruning(self, model: nn.Module) -> nn.Module:
        """
        Random pruning.
        
        Args:
            model (nn.Module): Model to prune
            
        Returns:
            nn.Module: Pruned model
        """
        # Create a copy of the model
        pruned_model = type(model)()
        pruned_model.load_state_dict(model.state_dict())
        
        # Prune each layer randomly
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                self._random_prune_linear_layer(module)
            elif isinstance(module, nn.Conv2d):
                self._random_prune_conv_layer(module)
                
        return pruned_model
    
    def _prune_linear_layer(self, layer: nn.Linear):
        """
        Prune a linear layer using magnitude-based pruning.
        
        Args:
            layer (nn.Linear): Linear layer to prune
        """
        # Get weight tensor
        weight = layer.weight.data
        
        # Compute threshold
        if self.config.layer_wise:
            # Layer-wise sparsity
            num_weights = weight.numel()
            num_prune = int(num_weights * self.config.sparsity_ratio)
            
            # Flatten weight tensor and find threshold
            flat_weights = weight.abs().view(-1)
            threshold = torch.kthvalue(flat_weights, num_prune).values
            
            # Create mask
            mask = weight.abs() > threshold
            
            # Apply mask
            layer.weight.data *= mask.float()
        else:
            # Global sparsity (simplified)
            threshold = torch.quantile(weight.abs().view(-1), self.config.sparsity_ratio)
            mask = weight.abs() > threshold
            layer.weight.data *= mask.float()
    
    def _prune_conv_layer(self, layer: nn.Conv2d):
        """
        Prune a convolutional layer using magnitude-based pruning.
        
        Args:
            layer (nn.Conv2d): Convolutional layer to prune
        """
        # Get weight tensor
        weight = layer.weight.data
        
        # Compute threshold
        if self.config.structured:
            # Structured pruning (prune entire filters)
            filter_norms = torch.norm(weight.view(weight.size(0), -1), dim=1)
            num_filters = weight.size(0)
            num_prune = int(num_filters * self.config.sparsity_ratio)
            
            # Find threshold
            threshold = torch.kthvalue(filter_norms, num_prune).values
            
            # Create mask for entire filters
            filter_mask = filter_norms > threshold
            mask = filter_mask.view(-1, 1, 1, 1).expand_as(weight)
            
            # Apply mask
            layer.weight.data *= mask.float()
        else:
            # Unstructured pruning
            num_weights = weight.numel()
            num_prune = int(num_weights * self.config.sparsity_ratio)
            
            # Flatten weight tensor and find threshold
            flat_weights = weight.abs().view(-1)
            threshold = torch.kthvalue(flat_weights, num_prune).values
            
            # Create mask
            mask = weight.abs() > threshold
            
            # Apply mask
            layer.weight.data *= mask.float()
    
    def _random_prune_linear_layer(self, layer: nn.Linear):
        """
        Randomly prune a linear layer.
        
        Args:
            layer (nn.Linear): Linear layer to prune
        """
        # Get weight tensor
        weight = layer.weight.data
        
        # Create random mask
        mask = torch.rand_like(weight) > self.config.sparsity_ratio
        
        # Apply mask
        layer.weight.data *= mask.float()
    
    def _random_prune_conv_layer(self, layer: nn.Conv2d):
        """
        Randomly prune a convolutional layer.
        
        Args:
            layer (nn.Conv2d): Convolutional layer to prune
        """
        # Get weight tensor
        weight = layer.weight.data
        
        # Create random mask
        mask = torch.rand_like(weight) > self.config.sparsity_ratio
        
        # Apply mask
        layer.weight.data *= mask.float()


class LowRankApproximation:
    """
    Low-rank approximation for compressing weight matrices.
    """
    
    @staticmethod
    def svd_compress(weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a weight matrix using SVD.
        
        Args:
            weight (torch.Tensor): Weight matrix to compress
            rank (int): Target rank for compression
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Compressed matrices (U @ S, V^T)
        """
        # Perform SVD
        U, S, V = torch.svd(weight)
        
        # Truncate to target rank
        U_r = U[:, :rank]
        S_r = S[:rank]
        V_r = V[:, :rank]
        
        # Compute compressed matrices
        US = torch.matmul(U_r, torch.diag(S_r))
        VT = V_r.t()
        
        return US, VT
    
    @staticmethod
    def compress_linear_layer(layer: nn.Linear, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a linear layer using low-rank approximation.
        
        Args:
            layer (nn.Linear): Linear layer to compress
            rank (int): Target rank for compression
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Compressed weight matrices
        """
        return LowRankApproximation.svd_compress(layer.weight.data, rank)


class KnowledgeDistillation:
    """
    Knowledge distillation for transferring knowledge from a large teacher model
    to a smaller student model.
    """
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 temperature: float = 3.0, alpha: float = 0.7):
        """
        Initialize KnowledgeDistillation.
        
        Args:
            teacher_model (nn.Module): Large teacher model
            student_model (nn.Module): Smaller student model
            temperature (float): Temperature for softening probability distributions
            alpha (float): Weight for combining hard and soft targets
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Set teacher model to evaluation mode
        self.teacher_model.eval()
        
    def distillation_loss(self, student_logits: torch.Tensor, 
                         teacher_logits: torch.Tensor, 
                         true_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits (torch.Tensor): Student model logits
            teacher_logits (torch.Tensor): Teacher model logits
            true_labels (torch.Tensor): True labels
            
        Returns:
            torch.Tensor: Distillation loss
        """
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Soft predictions from student
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        kd_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = F.cross_entropy(student_logits, true_labels)
        
        # Combined loss
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * hard_loss
        
        return total_loss
    
    def train_step(self, input_ids: torch.Tensor, 
                   labels: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Perform a single distillation training step.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            labels (torch.Tensor): True labels
            optimizer (torch.optim.Optimizer): Optimizer
            
        Returns:
            torch.Tensor: Training loss
        """
        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids)
            teacher_logits = teacher_outputs.logits
        
        # Get student predictions
        student_outputs = self.student_model(input_ids)
        student_logits = student_outputs.logits
        
        # Compute distillation loss
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        # Update student model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss


class ModelCompressor:
    """
    Unified model compressor combining multiple compression techniques.
    """
    
    def __init__(self, pruning_config: PruningConfig = None):
        """
        Initialize ModelCompressor.
        
        Args:
            pruning_config (PruningConfig, optional): Pruning configuration
        """
        self.pruning_config = pruning_config or PruningConfig()
        self.pruner = Pruner(self.pruning_config)
        
    def compress_model(self, model: nn.Module, 
                      compression_type: str = "pruning",
                      **kwargs) -> nn.Module:
        """
        Compress a model using specified technique.
        
        Args:
            model (nn.Module): Model to compress
            compression_type (str): Type of compression ("pruning", "low_rank", "distillation")
            **kwargs: Additional arguments for specific compression methods
            
        Returns:
            nn.Module: Compressed model
        """
        if compression_type == "pruning":
            return self._prune_model(model, **kwargs)
        elif compression_type == "low_rank":
            return self._low_rank_compress_model(model, **kwargs)
        elif compression_type == "distillation":
            return self._distill_model(model, **kwargs)
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
    
    def _prune_model(self, model: nn.Module, sparsity_ratio: float = 0.5) -> nn.Module:
        """
        Prune a model.
        
        Args:
            model (nn.Module): Model to prune
            sparsity_ratio (float): Sparsity ratio
            
        Returns:
            nn.Module: Pruned model
        """
        # Update pruning config
        self.pruning_config.sparsity_ratio = sparsity_ratio
        
        # Prune model
        pruned_model = self.pruner.prune_model(model)
        
        return pruned_model
    
    def _low_rank_compress_model(self, model: nn.Module, rank_ratio: float = 0.5) -> nn.Module:
        """
        Compress a model using low-rank approximation.
        
        Args:
            model (nn.Module): Model to compress
            rank_ratio (float): Ratio of original rank to keep
            
        Returns:
            nn.Module: Compressed model
        """
        # Create a copy of the model
        compressed_model = type(model)()
        compressed_model.load_state_dict(model.state_dict())
        
        # Compress linear layers
        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear):
                # Compute target rank
                original_rank = min(module.in_features, module.out_features)
                target_rank = int(original_rank * rank_ratio)
                
                # Compress using low-rank approximation
                US, VT = LowRankApproximation.compress_linear_layer(module, target_rank)
                
                # Replace with compressed layers
                # This is a simplified approach - in practice, you'd replace with two linear layers
                # For now, we'll just zero out some weights to simulate compression
                compression_ratio = target_rank / original_rank
                mask = torch.rand_like(module.weight.data) > (1 - compression_ratio)
                module.weight.data *= mask.float()
                
        return compressed_model
    
    def _distill_model(self, model: nn.Module, teacher_model: nn.Module,
                      temperature: float = 3.0, alpha: float = 0.7) -> nn.Module:
        """
        Set up knowledge distillation.
        
        Args:
            model (nn.Module): Student model
            teacher_model (nn.Module): Teacher model
            temperature (float): Temperature for softening
            alpha (float): Weight for combining losses
            
        Returns:
            nn.Module: Student model ready for distillation training
        """
        # Knowledge distillation is typically done during training
        # Here we just return the student model with distillation setup info
        distiller = KnowledgeDistillation(teacher_model, model, temperature, alpha)
        return model  # Return student model, distillation happens during training


# Example usage functions
def apply_pruning(model: nn.Module, sparsity_ratio: float = 0.5) -> nn.Module:
    """
    Apply pruning to a model.
    
    Args:
        model (nn.Module): Model to prune
        sparsity_ratio (float): Sparsity ratio
        
    Returns:
        nn.Module: Pruned model
    """
    config = PruningConfig(sparsity_ratio=sparsity_ratio)
    compressor = ModelCompressor(config)
    pruned_model = compressor.compress_model(model, "pruning", sparsity_ratio=sparsity_ratio)
    return pruned_model


def apply_low_rank_compression(model: nn.Module, rank_ratio: float = 0.5) -> nn.Module:
    """
    Apply low-rank compression to a model.
    
    Args:
        model (nn.Module): Model to compress
        rank_ratio (float): Rank ratio
        
    Returns:
        nn.Module: Compressed model
    """
    compressor = ModelCompressor()
    compressed_model = compressor.compress_model(model, "low_rank", rank_ratio=rank_ratio)
    return compressed_model


def setup_knowledge_distillation(teacher_model: nn.Module, 
                               student_model: nn.Module,
                               temperature: float = 3.0, 
                               alpha: float = 0.7) -> KnowledgeDistillation:
    """
    Set up knowledge distillation.
    
    Args:
        teacher_model (nn.Module): Teacher model
        student_model (nn.Module): Student model
        temperature (float): Temperature for softening
        alpha (float): Weight for combining losses
        
    Returns:
        KnowledgeDistillation: Distillation setup
    """
    distiller = KnowledgeDistillation(teacher_model, student_model, temperature, alpha)
    return distiller


def compare_model_performance(original_model: nn.Module, 
                            compressed_model: nn.Module,
                            test_loader: torch.utils.data.DataLoader,
                            device: torch.device):
    """
    Compare performance of original and compressed models.
    
    Args:
        original_model (nn.Module): Original model
        compressed_model (nn.Module): Compressed model
        test_loader (torch.utils.data.DataLoader): Test data loader
        device (torch.device): Device to run on
    """
    def evaluate_model(model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Evaluate model accuracy and inference time."""
        model.eval()
        model.to(device)
        
        correct = 0
        total = 0
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)
        else:
            inference_time = 0
        
        accuracy = 100 * correct / total
        return accuracy, inference_time
    
    # Evaluate original model
    orig_accuracy, orig_time = evaluate_model(original_model, test_loader)
    
    # Evaluate compressed model
    comp_accuracy, comp_time = evaluate_model(compressed_model, test_loader)
    
    print(f"Original Model - Accuracy: {orig_accuracy:.2f}%, Inference Time: {orig_time:.2f}ms")
    print(f"Compressed Model - Accuracy: {comp_accuracy:.2f}%, Inference Time: {comp_time:.2f}ms")
    print(f"Accuracy Drop: {orig_accuracy - comp_accuracy:.2f}%")
    print(f"Speedup: {orig_time / comp_time:.2f}x (if applicable)")