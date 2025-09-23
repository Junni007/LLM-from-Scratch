"""
Efficiency optimizations for LLMs.

This module implements quantization techniques, compression methods, and
optimized training techniques to make LLMs more efficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List
import math

class QuantizationConfig:
    """
    Configuration for quantization techniques.
    """
    def __init__(self, 
                 bits: int = 8,
                 method: str = "linear",  # "linear", "log", "norm"
                 symmetric: bool = True,
                 per_channel: bool = False):
        self.bits = bits
        self.method = method
        self.symmetric = symmetric
        self.per_channel = per_channel


class Quantizer:
    """
    Quantizer for converting floating-point tensors to quantized representations.
    """
    
    def __init__(self, config: QuantizationConfig):
        """
        Initialize Quantizer.
        
        Args:
            config (QuantizationConfig): Quantization configuration
        """
        self.config = config
        self.num_levels = 2 ** config.bits
        
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor.
        
        Args:
            tensor (torch.Tensor): Input tensor to quantize
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - quantized_tensor: Quantized tensor
                - scale: Scaling factor
                - zero_point: Zero point for asymmetric quantization
        """
        if self.config.symmetric:
            return self._symmetric_quantize(tensor)
        else:
            return self._asymmetric_quantize(tensor)
    
    def _symmetric_quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Symmetric quantization.
        
        Args:
            tensor (torch.Tensor): Input tensor to quantize
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - quantized_tensor: Quantized tensor
                - scale: Scaling factor
                - zero_point: Zero point (always 0 for symmetric)
        """
        # Compute scale factor
        if self.config.per_channel and tensor.dim() > 1:
            # Per-channel quantization
            max_vals = torch.amax(torch.abs(tensor), dim=list(range(1, tensor.dim())), keepdim=True)
            scale = max_vals / (self.num_levels / 2 - 1)
        else:
            # Per-tensor quantization
            max_val = torch.max(torch.abs(tensor))
            scale = max_val / (self.num_levels / 2 - 1)
        
        # Quantize
        quantized = torch.round(tensor / scale)
        
        # Clamp to valid range
        qmin = -self.num_levels / 2
        qmax = self.num_levels / 2 - 1
        quantized = torch.clamp(quantized, qmin, qmax)
        
        return quantized.to(torch.int8), scale, torch.tensor(0.0, device=tensor.device)
    
    def _asymmetric_quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Asymmetric quantization.
        
        Args:
            tensor (torch.Tensor): Input tensor to quantize
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - quantized_tensor: Quantized tensor
                - scale: Scaling factor
                - zero_point: Zero point
        """
        # Compute min and max values
        if self.config.per_channel and tensor.dim() > 1:
            # Per-channel quantization
            min_vals = torch.amin(tensor, dim=list(range(1, tensor.dim())), keepdim=True)
            max_vals = torch.amax(tensor, dim=list(range(1, tensor.dim())), keepdim=True)
        else:
            # Per-tensor quantization
            min_vals = torch.min(tensor)
            max_vals = torch.max(tensor)
        
        # Compute scale and zero point
        scale = (max_vals - min_vals) / (self.num_levels - 1)
        zero_point = torch.round(-min_vals / scale)
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        
        # Clamp to valid range
        qmin = 0
        qmax = self.num_levels - 1
        quantized = torch.clamp(quantized, qmin, qmax)
        
        return quantized.to(torch.uint8), scale, zero_point
    
    def dequantize(self, quantized_tensor: torch.Tensor, 
                   scale: torch.Tensor, 
                   zero_point: torch.Tensor) -> torch.Tensor:
        """
        Dequantize a tensor.
        
        Args:
            quantized_tensor (torch.Tensor): Quantized tensor
            scale (torch.Tensor): Scaling factor
            zero_point (torch.Tensor): Zero point
            
        Returns:
            torch.Tensor: Dequantized tensor
        """
        if self.config.symmetric:
            return self._symmetric_dequantize(quantized_tensor, scale)
        else:
            return self._asymmetric_dequantize(quantized_tensor, scale, zero_point)
    
    def _symmetric_dequantize(self, quantized_tensor: torch.Tensor, 
                             scale: torch.Tensor) -> torch.Tensor:
        """
        Symmetric dequantization.
        
        Args:
            quantized_tensor (torch.Tensor): Quantized tensor
            scale (torch.Tensor): Scaling factor
            
        Returns:
            torch.Tensor: Dequantized tensor
        """
        return quantized_tensor.float() * scale
    
    def _asymmetric_dequantize(self, quantized_tensor: torch.Tensor, 
                              scale: torch.Tensor, 
                              zero_point: torch.Tensor) -> torch.Tensor:
        """
        Asymmetric dequantization.
        
        Args:
            quantized_tensor (torch.Tensor): Quantized tensor
            scale (torch.Tensor): Scaling factor
            zero_point (torch.Tensor): Zero point
            
        Returns:
            torch.Tensor: Dequantized tensor
        """
        return (quantized_tensor.float() - zero_point) * scale


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer that performs computation in quantized space.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 config: QuantizationConfig = None):
        """
        Initialize QuantizedLinear.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            config (QuantizationConfig, optional): Quantization configuration
        """
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Create standard linear layer
        self.linear = nn.Linear(in_features, out_features)
        
        # Quantization configuration
        self.config = config or QuantizationConfig()
        self.quantizer = Quantizer(self.config)
        
        # Quantized parameters
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features if config.per_channel else 1))
        self.register_buffer('weight_zero_point', torch.zeros(out_features if config.per_channel else 1))
        
        self.register_buffer('bias_quantized', torch.zeros(out_features, dtype=torch.int8))
        self.register_buffer('bias_scale', torch.ones(1))
        self.register_buffer('bias_zero_point', torch.zeros(1))
        
        # Flag to indicate if parameters are quantized
        self.quantized = False
        
    def quantize_parameters(self):
        """
        Quantize the layer parameters.
        """
        # Quantize weight
        weight_q, weight_s, weight_z = self.quantizer.quantize(self.linear.weight)
        self.weight_quantized = weight_q
        self.weight_scale = weight_s
        self.weight_zero_point = weight_z
        
        # Quantize bias if it exists
        if self.linear.bias is not None:
            bias_q, bias_s, bias_z = self.quantizer.quantize(self.linear.bias)
            self.bias_quantized = bias_q
            self.bias_scale = bias_s
            self.bias_zero_point = bias_z
        else:
            self.bias_quantized = torch.zeros(self.out_features, dtype=torch.int8)
            self.bias_scale = torch.ones(1)
            self.bias_zero_point = torch.zeros(1)
        
        self.quantized = True
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantized linear layer.
        
        Args:
            input (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        if not self.quantized:
            # Use standard linear layer if not quantized
            return self.linear(input)
        
        # Dequantize weight
        weight = self.quantizer.dequantize(
            self.weight_quantized, 
            self.weight_scale, 
            self.weight_zero_point
        )
        
        # Dequantize bias
        bias = self.quantizer.dequantize(
            self.bias_quantized, 
            self.bias_scale, 
            self.bias_zero_point
        )
        
        # Perform linear operation
        output = F.linear(input, weight, bias)
        
        return output


class QuantizedAttention(nn.Module):
    """
    Quantized attention mechanism for efficient computation.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, 
                 config: QuantizationConfig = None):
        """
        Initialize QuantizedAttention.
        
        Args:
            hidden_size (int): Hidden size
            num_heads (int): Number of attention heads
            config (QuantizationConfig, optional): Quantization configuration
        """
        super(QuantizedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Create quantized linear layers
        self.q_proj = QuantizedLinear(hidden_size, hidden_size, config)
        self.k_proj = QuantizedLinear(hidden_size, hidden_size, config)
        self.v_proj = QuantizedLinear(hidden_size, hidden_size, config)
        self.o_proj = QuantizedLinear(hidden_size, hidden_size, config)
        
        # Quantize parameters
        self.q_proj.quantize_parameters()
        self.k_proj.quantize_parameters()
        self.v_proj.quantize_parameters()
        self.o_proj.quantize_parameters()
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through quantized attention.
        
        Args:
            hidden_states (torch.Tensor): Hidden states of shape (batch_size, seq_length, hidden_size)
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Output of shape (batch_size, seq_length, hidden_size)
        """
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Project to query, key, value
        query = self.q_proj(hidden_states)  # (batch_size, seq_length, hidden_size)
        key = self.k_proj(hidden_states)    # (batch_size, seq_length, hidden_size)
        value = self.v_proj(hidden_states)  # (batch_size, seq_length, hidden_size)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, hidden_size)
        output = self.o_proj(context)
        
        return output


class ModelQuantizer:
    """
    Utility class for quantizing entire models.
    """
    
    def __init__(self, config: QuantizationConfig = None):
        """
        Initialize ModelQuantizer.
        
        Args:
            config (QuantizationConfig, optional): Quantization configuration
        """
        self.config = config or QuantizationConfig()
        
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Quantize a model by replacing linear layers with quantized versions.
        
        Args:
            model (nn.Module): Model to quantize
            
        Returns:
            nn.Module: Quantized model
        """
        # Create a copy of the model
        quantized_model = type(model)()  # Create new instance of same type
        
        # Copy state dict
        quantized_model.load_state_dict(model.state_dict())
        
        # Recursively quantize modules
        self._quantize_modules(quantized_model)
        
        return quantized_model
    
    def _quantize_modules(self, module: nn.Module):
        """
        Recursively quantize modules in a model.
        
        Args:
            module (nn.Module): Module to quantize
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace with quantized linear layer
                quantized_linear = QuantizedLinear(
                    child.in_features, 
                    child.out_features, 
                    self.config
                )
                
                # Copy weights and bias
                quantized_linear.linear.weight.data = child.weight.data
                if child.bias is not None:
                    quantized_linear.linear.bias.data = child.bias.data
                
                # Quantize parameters
                quantized_linear.quantize_parameters()
                
                # Replace in parent module
                setattr(module, name, quantized_linear)
            else:
                # Recursively process child modules
                self._quantize_modules(child)


# Example usage functions
def apply_quantization(model: nn.Module, bits: int = 8) -> nn.Module:
    """
    Apply quantization to a model.
    
    Args:
        model (nn.Module): Model to quantize
        bits (int): Number of bits for quantization
        
    Returns:
        nn.Module: Quantized model
    """
    config = QuantizationConfig(bits=bits)
    quantizer = ModelQuantizer(config)
    quantized_model = quantizer.quantize_model(model)
    return quantized_model


def compare_model_sizes(original_model: nn.Module, quantized_model: nn.Module):
    """
    Compare the sizes of original and quantized models.
    
    Args:
        original_model (nn.Module): Original model
        quantized_model (nn.Module): Quantized model
    """
    def get_model_size(model: nn.Module) -> int:
        """Get the size of a model in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size
    
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    print(f"Original model size: {original_size / (1024**2):.2f} MB")
    print(f"Quantized model size: {quantized_size / (1024**2):.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
    print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.2f}%")