"""
Performance benchmarking tools for LLM components.
"""

import torch
import torch.nn as nn
import time
import gc
from typing import Callable, Any, Dict
import psutil
import os

class PerformanceBenchmark:
    """Performance benchmarking utility for LLM components."""
    
    def __init__(self):
        self.results = {}
        
    def benchmark_forward_pass(self, model: nn.Module, 
                             input_generator: Callable[[], tuple],
                             num_iterations: int = 100,
                             warmup_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark forward pass performance.
        
        Args:
            model (nn.Module): Model to benchmark
            input_generator (Callable): Function that generates input tensors
            num_iterations (int): Number of iterations to benchmark
            warmup_iterations (int): Number of warmup iterations
            
        Returns:
            Dict[str, float]: Benchmark results
        """
        model.eval()
        
        # Warmup
        for _ in range(warmup_iterations):
            inputs = input_generator()
            with torch.no_grad():
                _ = model(*inputs)
                
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                inputs = input_generator()
                _ = model(*inputs)
                
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        return {
            'total_time': total_time,
            'avg_time_per_iteration': avg_time,
            'throughput_iterations_per_second': throughput
        }
    
    def benchmark_memory_usage(self, model: nn.Module,
                             input_generator: Callable[[], tuple]) -> Dict[str, float]:
        """
        Benchmark memory usage.
        
        Args:
            model (nn.Module): Model to benchmark
            input_generator (Callable): Function that generates input tensors
            
        Returns:
            Dict[str, float]: Memory usage results
        """
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        # Get baseline memory
        if torch.cuda.is_available():
            baseline_memory = torch.cuda.memory_allocated()
        else:
            baseline_memory = psutil.Process(os.getpid()).memory_info().rss
        
        # Run forward pass
        inputs = input_generator()
        with torch.no_grad():
            outputs = model(*inputs)
            
        # Get peak memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.memory_allocated()
            memory_increase = peak_memory - baseline_memory
            memory_mb = memory_increase / (1024 ** 2)
        else:
            current_memory = psutil.Process(os.getpid()).memory_info().rss
            memory_increase = current_memory - baseline_memory
            memory_mb = memory_increase / (1024 ** 2)
            
        return {
            'memory_usage_mb': memory_mb,
            'peak_memory_mb': peak_memory / (1024 ** 2) if torch.cuda.is_available() else current_memory / (1024 ** 2)
        }
    
    def benchmark_training_step(self, model: nn.Module,
                               input_generator: Callable[[], tuple],
                               optimizer: torch.optim.Optimizer,
                               num_iterations: int = 100,
                               warmup_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark training step performance.
        
        Args:
            model (nn.Module): Model to benchmark
            input_generator (Callable): Function that generates input tensors
            optimizer (torch.optim.Optimizer): Optimizer to use
            num_iterations (int): Number of iterations to benchmark
            warmup_iterations (int): Number of warmup iterations
            
        Returns:
            Dict[str, float]: Benchmark results
        """
        model.train()
        
        # Warmup
        for _ in range(warmup_iterations):
            inputs = input_generator()
            optimizer.zero_grad()
            outputs = model(*inputs)
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                # Create a simple loss for testing
                loss = outputs[0].sum() if isinstance(outputs, tuple) else outputs.sum()
            loss.backward()
            optimizer.step()
                
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(num_iterations):
            inputs = input_generator()
            optimizer.zero_grad()
            outputs = model(*inputs)
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = outputs[0].sum() if isinstance(outputs, tuple) else outputs.sum()
            loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        return {
            'total_time': total_time,
            'avg_time_per_iteration': avg_time,
            'throughput_iterations_per_second': throughput
        }
    
    def benchmark_component(self, name: str,
                          model: nn.Module,
                          input_generator: Callable[[], tuple],
                          optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
        """
        Comprehensive benchmark of a component.
        
        Args:
            name (str): Name of the component
            model (nn.Module): Model to benchmark
            input_generator (Callable): Function that generates input tensors
            optimizer (torch.optim.Optimizer, optional): Optimizer for training benchmark
            
        Returns:
            Dict[str, Any]: Comprehensive benchmark results
        """
        print(f"Benchmarking {name}...")
        
        results = {
            'component_name': name,
            'forward_pass': {},
            'memory_usage': {},
            'training_step': {} if optimizer else None
        }
        
        # Forward pass benchmark
        try:
            forward_results = self.benchmark_forward_pass(model, input_generator)
            results['forward_pass'] = forward_results
            print(f"  Forward pass: {forward_results['avg_time_per_iteration']*1000:.2f}ms per iteration")
        except Exception as e:
            print(f"  Forward pass benchmark failed: {e}")
            results['forward_pass'] = {'error': str(e)}
            
        # Memory usage benchmark
        try:
            memory_results = self.benchmark_memory_usage(model, input_generator)
            results['memory_usage'] = memory_results
            print(f"  Memory usage: {memory_results['memory_usage_mb']:.2f}MB")
        except Exception as e:
            print(f"  Memory usage benchmark failed: {e}")
            results['memory_usage'] = {'error': str(e)}
            
        # Training step benchmark (if optimizer provided)
        if optimizer:
            try:
                training_results = self.benchmark_training_step(model, input_generator, optimizer)
                results['training_step'] = training_results
                print(f"  Training step: {training_results['avg_time_per_iteration']*1000:.2f}ms per iteration")
            except Exception as e:
                print(f"  Training step benchmark failed: {e}")
                results['training_step'] = {'error': str(e)}
                
        self.results[name] = results
        return results
    
    def compare_models(self, model_pairs: list) -> Dict[str, Dict[str, Any]]:
        """
        Compare performance of multiple model pairs.
        
        Args:
            model_pairs (list): List of tuples (name, model, input_generator, optimizer)
            
        Returns:
            Dict[str, Dict[str, Any]]: Comparison results
        """
        comparison_results = {}
        
        for name, model, input_generator, optimizer in model_pairs:
            results = self.benchmark_component(name, model, input_generator, optimizer)
            comparison_results[name] = results
            
        return comparison_results
    
    def print_summary(self):
        """Print summary of all benchmark results."""
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*50)
        
        for name, results in self.results.items():
            print(f"\n{name}:")
            
            # Forward pass results
            if 'forward_pass' in results and results['forward_pass']:
                if 'error' not in results['forward_pass']:
                    fp = results['forward_pass']
                    print(f"  Forward Pass: {fp['avg_time_per_iteration']*1000:.2f}ms "
                          f"({fp['throughput_iterations_per_second']:.1f} it/s)")
                else:
                    print(f"  Forward Pass: ERROR - {results['forward_pass']['error']}")
            
            # Memory usage results
            if 'memory_usage' in results and results['memory_usage']:
                if 'error' not in results['memory_usage']:
                    mem = results['memory_usage']
                    print(f"  Memory Usage: {mem['memory_usage_mb']:.2f}MB")
                else:
                    print(f"  Memory Usage: ERROR - {results['memory_usage']['error']}")
            
            # Training step results
            if results['training_step']:
                if 'error' not in results['training_step']:
                    ts = results['training_step']
                    print(f"  Training Step: {ts['avg_time_per_iteration']*1000:.2f}ms "
                          f"({ts['throughput_iterations_per_second']:.1f} it/s)")
                else:
                    print(f"  Training Step: ERROR - {results['training_step']['error']}")


# Example usage functions
def create_sample_transformer_input_generator(batch_size: int = 2, 
                                            seq_length: int = 64, 
                                            vocab_size: int = 1000,
                                            d_model: int = 128):
    """Create a sample input generator for transformer models."""
    def input_generator():
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        return (input_ids,)
    return input_generator

def create_sample_moe_input_generator(batch_size: int = 2, 
                                    seq_length: int = 64, 
                                    input_size: int = 128):
    """Create a sample input generator for MoE models."""
    def input_generator():
        x = torch.randn(batch_size, seq_length, input_size)
        return (x,)
    return input_generator

def run_comprehensive_benchmark():
    """Run comprehensive benchmark of various components."""
    benchmark = PerformanceBenchmark()
    
    # Test with a simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_size=128, hidden_size=256, output_size=128):
            super().__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, output_size)
            self.activation = nn.ReLU()
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x
    
    # Create model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create input generator
    def input_generator():
        x = torch.randn(2, 10, 128)
        return (x,)
    
    # Benchmark the model
    results = benchmark.benchmark_component(
        "SimpleModel", 
        model, 
        input_generator, 
        optimizer
    )
    
    # Print summary
    benchmark.print_summary()
    
    return results


if __name__ == "__main__":
    # Run example benchmark
    print("Running performance benchmark example...")
    results = run_comprehensive_benchmark()
    print("Benchmark completed!")