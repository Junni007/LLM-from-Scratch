"""
Learning rate scheduler implementations for LLM training.
"""

import torch
import math
from typing import Optional


class LinearWarmupScheduler:
    """
    Linear warmup learning rate scheduler.
    
    Linearly increases the learning rate from 0 to the initial learning rate
    over the warmup period, then keeps it constant.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, 
                 last_step: int = -1):
        """
        Initialize the scheduler.
        
        Args:
            optimizer (torch.optim.Optimizer): Optimizer to schedule
            warmup_steps (int): Number of warmup steps
            last_step (int): The index of last step. Default: -1
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.last_step = last_step
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate for the next step."""
        self.last_step += 1
        lr_factor = self._get_lr_factor()
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * lr_factor
    
    def _get_lr_factor(self) -> float:
        """Get the learning rate factor for the current step."""
        if self.warmup_steps == 0:
            return 1.0
        
        if self.last_step < self.warmup_steps:
            # Linear warmup
            return float(self.last_step) / float(self.warmup_steps)
        else:
            # Constant after warmup
            return 1.0


class CosineDecayScheduler:
    """
    Cosine decay learning rate scheduler.
    
    Decreases the learning rate following a cosine curve from the initial
    learning rate to a minimum value.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int,
                 min_lr: float = 0.0, last_step: int = -1):
        """
        Initialize the scheduler.
        
        Args:
            optimizer (torch.optim.Optimizer): Optimizer to schedule
            total_steps (int): Total number of training steps
            min_lr (float): Minimum learning rate
            last_step (int): The index of last step. Default: -1
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.last_step = last_step
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate for the next step."""
        self.last_step += 1
        lr_factor = self._get_lr_factor()
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * lr_factor
    
    def _get_lr_factor(self) -> float:
        """Get the learning rate factor for the current step."""
        if self.total_steps == 0:
            return 1.0
        
        progress = float(self.last_step) / float(self.total_steps)
        # Clamp progress to [0, 1]
        progress = max(0.0, min(1.0, progress))
        # Cosine decay
        return 0.5 * (1.0 + math.cos(math.pi * progress))


class LinearDecayScheduler:
    """
    Linear decay learning rate scheduler.
    
    Linearly decreases the learning rate from the initial learning rate
    to a minimum value.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int,
                 min_lr: float = 0.0, last_step: int = -1):
        """
        Initialize the scheduler.
        
        Args:
            optimizer (torch.optim.Optimizer): Optimizer to schedule
            total_steps (int): Total number of training steps
            min_lr (float): Minimum learning rate
            last_step (int): The index of last step. Default: -1
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.last_step = last_step
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate for the next step."""
        self.last_step += 1
        lr_factor = self._get_lr_factor()
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * lr_factor
    
    def _get_lr_factor(self) -> float:
        """Get the learning rate factor for the current step."""
        if self.total_steps == 0:
            return 1.0
        
        progress = float(self.last_step) / float(self.total_steps)
        # Clamp progress to [0, 1]
        progress = max(0.0, min(1.0, progress))
        # Linear decay
        return 1.0 - progress


class WarmupDecayScheduler:
    """
    Combined warmup and decay scheduler.
    
    Linearly increases the learning rate during warmup, then decays it
    according to a specified decay schedule.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 total_steps: int, decay_type: str = 'cosine', 
                 min_lr: float = 0.0, last_step: int = -1):
        """
        Initialize the scheduler.
        
        Args:
            optimizer (torch.optim.Optimizer): Optimizer to schedule
            warmup_steps (int): Number of warmup steps
            total_steps (int): Total number of training steps
            decay_type (str): Type of decay ('cosine' or 'linear')
            min_lr (float): Minimum learning rate
            last_step (int): The index of last step. Default: -1
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_type = decay_type
        self.min_lr = min_lr
        self.last_step = last_step
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate for the next step."""
        self.last_step += 1
        lr_factor = self._get_lr_factor()
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * lr_factor
    
    def _get_lr_factor(self) -> float:
        """Get the learning rate factor for the current step."""
        if self.last_step < self.warmup_steps:
            # Linear warmup
            if self.warmup_steps == 0:
                return 1.0
            return float(self.last_step) / float(self.warmup_steps)
        else:
            # Decay phase
            if self.total_steps <= self.warmup_steps:
                return 1.0
            
            decay_steps = self.total_steps - self.warmup_steps
            progress = float(self.last_step - self.warmup_steps) / float(decay_steps)
            # Clamp progress to [0, 1]
            progress = max(0.0, min(1.0, progress))
            
            if self.decay_type == 'cosine':
                # Cosine decay
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                # Linear decay
                return 1.0 - progress


def test_schedulers():
    """Test function for learning rate schedulers."""
    # Create a simple optimizer for testing
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test LinearWarmupScheduler
    scheduler = LinearWarmupScheduler(optimizer, warmup_steps=10)
    lrs = []
    for _ in range(15):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    
    # Check that learning rate increases linearly during warmup
    for i in range(10):
        expected_lr = 0.001 * (i + 1) / 10
        assert abs(lrs[i] - expected_lr) < 1e-6, f"Expected {expected_lr}, got {lrs[i]}"
    
    # Check that learning rate stays constant after warmup
    for i in range(10, 15):
        assert abs(lrs[i] - 0.001) < 1e-6, f"Expected 0.001, got {lrs[i]}"
    
    print("LinearWarmupScheduler test passed!")
    
    # Test CosineDecayScheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineDecayScheduler(optimizer, total_steps=10, min_lr=0.0001)
    lrs = []
    for _ in range(10):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    
    # Check that learning rate decreases from initial to minimum
    assert abs(lrs[0] - 0.001) < 1e-6, f"Expected 0.001, got {lrs[0]}"
    assert abs(lrs[-1] - 0.0001) < 1e-6, f"Expected 0.0001, got {lrs[-1]}"
    
    print("CosineDecayScheduler test passed!")
    
    # Test LinearDecayScheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = LinearDecayScheduler(optimizer, total_steps=10, min_lr=0.0001)
    lrs = []
    for _ in range(10):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    
    # Check that learning rate decreases linearly from initial to minimum
    assert abs(lrs[0] - 0.001) < 1e-6, f"Expected 0.001, got {lrs[0]}"
    assert abs(lrs[-1] - 0.0001) < 1e-6, f"Expected 0.0001, got {lrs[-1]}"
    
    print("LinearDecayScheduler test passed!")
    
    # Test WarmupDecayScheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = WarmupDecayScheduler(optimizer, warmup_steps=5, total_steps=15, 
                                   decay_type='cosine', min_lr=0.0001)
    lrs = []
    for _ in range(15):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    
    # Check warmup phase
    for i in range(5):
        expected_lr = 0.0001 + (0.001 - 0.0001) * (i + 1) / 5
        assert abs(lrs[i] - expected_lr) < 1e-6, f"Expected {expected_lr}, got {lrs[i]}"
    
    # Check decay phase (cosine)
    # At step 5, should be at full learning rate
    assert abs(lrs[5] - 0.001) < 1e-6, f"Expected 0.001, got {lrs[5]}"
    
    print("WarmupDecayScheduler test passed!")
    
    print("All scheduler tests passed!")


if __name__ == "__main__":
    test_schedulers()