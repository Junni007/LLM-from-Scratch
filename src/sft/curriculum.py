"""
Curriculum learning for Supervised Fine-Tuning (SFT) instruction datasets.

This module implements curriculum learning strategies for instruction datasets,
where training examples are ordered by difficulty or complexity to improve
training stability and performance.
"""

import torch
import random
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class CurriculumStrategy(Enum):
    """Strategies for curriculum learning."""
    SEQUENCE_LENGTH = "sequence_length"  # Sort by sequence length
    COMPLEXITY_SCORE = "complexity_score"  # Sort by complexity score
    INSTRUCTION_LENGTH = "instruction_length"  # Sort by instruction length
    OUTPUT_LENGTH = "output_length"  # Sort by output length
    MIXED = "mixed"  # Mix of different strategies
    RANDOM = "random"  # Random ordering (no curriculum)


@dataclass
class CurriculumExample:
    """Wrapper for instruction examples with curriculum metadata."""
    example: Any  # Original example (InstructionExample or similar)
    sequence_length: int = 0
    instruction_length: int = 0
    output_length: int = 0
    complexity_score: float = 0.0
    metadata: Dict[str, Any] = None


class CurriculumScheduler:
    """
    Scheduler for curriculum learning that controls how difficulty progresses.
    """
    
    def __init__(self, strategy: str = "linear"):
        """
        Initialize CurriculumScheduler.
        
        Args:
            strategy (str): Strategy for difficulty progression ('linear', 'exponential', 'step')
        """
        self.strategy = strategy
        self.current_difficulty = 0.0
        self.step = 0
    
    def update_difficulty(self, epoch: int, total_epochs: int):
        """
        Update difficulty level based on training progress.
        
        Args:
            epoch (int): Current epoch
            total_epochs (int): Total number of epochs
            
        Returns:
            float: Current difficulty level (0.0 to 1.0)
        """
        self.step = epoch
        progress = epoch / max(total_epochs, 1)
        
        if self.strategy == "linear":
            self.current_difficulty = progress
        elif self.strategy == "exponential":
            self.current_difficulty = 1.0 - np.exp(-3 * progress)
        elif self.strategy == "step":
            # Step-wise difficulty increase
            if progress < 0.25:
                self.current_difficulty = 0.0
            elif progress < 0.5:
                self.current_difficulty = 0.25
            elif progress < 0.75:
                self.current_difficulty = 0.5
            else:
                self.current_difficulty = 1.0
        else:
            self.current_difficulty = progress
            
        return self.current_difficulty
    
    def get_current_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.current_difficulty


class CurriculumDataset:
    """
    Dataset wrapper that implements curriculum learning for instruction data.
    """
    
    def __init__(self, examples: List[CurriculumExample], 
                 strategy: CurriculumStrategy = CurriculumStrategy.SEQUENCE_LENGTH,
                 scheduler: CurriculumScheduler = None,
                 reverse: bool = False):
        """
        Initialize CurriculumDataset.
        
        Args:
            examples (List[CurriculumExample]): Examples with curriculum metadata
            strategy (CurriculumStrategy): Strategy for ordering examples
            scheduler (CurriculumScheduler): Scheduler for difficulty progression
            reverse (bool): Whether to reverse the ordering (easiest last)
        """
        self.original_examples = examples
        self.strategy = strategy
        self.reverse = reverse
        self.scheduler = scheduler or CurriculumScheduler()
        
        # Sort examples based on strategy
        self.sorted_examples = self._sort_examples(examples)
        self.current_examples = self.sorted_examples.copy()
        
    def _sort_examples(self, examples: List[CurriculumExample]) -> List[CurriculumExample]:
        """
        Sort examples based on curriculum strategy.
        
        Args:
            examples (List[CurriculumExample]): Examples to sort
            
        Returns:
            List[CurriculumExample]: Sorted examples
        """
        if self.strategy == CurriculumStrategy.RANDOM:
            # For random strategy, shuffle but keep original for reference
            shuffled = examples.copy()
            random.shuffle(shuffled)
            return shuffled
        
        # Sort based on strategy
        if self.strategy == CurriculumStrategy.SEQUENCE_LENGTH:
            sorted_examples = sorted(examples, key=lambda x: x.sequence_length)
        elif self.strategy == CurriculumStrategy.COMPLEXITY_SCORE:
            sorted_examples = sorted(examples, key=lambda x: x.complexity_score)
        elif self.strategy == CurriculumStrategy.INSTRUCTION_LENGTH:
            sorted_examples = sorted(examples, key=lambda x: x.instruction_length)
        elif self.strategy == CurriculumStrategy.OUTPUT_LENGTH:
            sorted_examples = sorted(examples, key=lambda x: x.output_length)
        elif self.strategy == CurriculumStrategy.MIXED:
            # Mixed strategy: combine multiple metrics
            sorted_examples = sorted(examples, key=lambda x: (
                x.sequence_length + 
                x.instruction_length * 0.5 + 
                x.output_length * 0.5 +
                x.complexity_score * 100
            ))
        else:
            # Default to sequence length
            sorted_examples = sorted(examples, key=lambda x: x.sequence_length)
        
        if self.reverse:
            sorted_examples = sorted_examples[::-1]
            
        return sorted_examples
    
    def update_curriculum(self, epoch: int, total_epochs: int):
        """
        Update curriculum based on training progress.
        
        Args:
            epoch (int): Current epoch
            total_epochs (int): Total number of epochs
        """
        if self.strategy == CurriculumStrategy.RANDOM:
            # Reshuffle for random curriculum
            self.current_examples = self.original_examples.copy()
            random.shuffle(self.current_examples)
            return
        
        # Get current difficulty level
        difficulty = self.scheduler.update_difficulty(epoch, total_epochs)
        
        # Select subset of examples based on difficulty
        total_examples = len(self.sorted_examples)
        num_selected = int(total_examples * min(difficulty + 0.1, 1.0))  # At least 10% of examples
        num_selected = max(num_selected, 1)  # At least 1 example
        
        # Select examples from the sorted list
        if self.reverse:
            # For reverse curriculum, start with harder examples
            self.current_examples = self.sorted_examples[-num_selected:]
        else:
            # For normal curriculum, start with easier examples
            self.current_examples = self.sorted_examples[:num_selected]
    
    def get_current_examples(self) -> List[CurriculumExample]:
        """Get current examples based on curriculum."""
        return self.current_examples
    
    def __len__(self) -> int:
        """Get number of current examples."""
        return len(self.current_examples)
    
    def __getitem__(self, idx: int) -> CurriculumExample:
        """Get example by index."""
        return self.current_examples[idx]


def compute_complexity_score(example: Any) -> float:
    """
    Compute complexity score for an instruction example.
    
    Args:
        example (Any): Instruction example
        
    Returns:
        float: Complexity score (higher = more complex)
    """
    # This is a simple heuristic - in practice, you might use:
    # - Number of entities/concepts
    # - Syntactic complexity
    # - Semantic depth
    # - Task type complexity
    
    instruction = getattr(example, 'instruction', '')
    input_text = getattr(example, 'input', '')
    output_text = getattr(example, 'output', '')
    
    # Simple complexity heuristics
    instruction_words = len(instruction.split())
    input_words = len(input_text.split())
    output_words = len(output_text.split())
    
    # Weighted combination
    complexity = (
        instruction_words * 0.3 +
        input_words * 0.2 +
        output_words * 0.5 +
        len(instruction) * 0.01 +  # Character count bonus
        len(input_text) * 0.005 +
        len(output_text) * 0.01
    )
    
    return complexity


def create_curriculum_examples(examples: List[Any]) -> List[CurriculumExample]:
    """
    Create curriculum examples with metadata from regular examples.
    
    Args:
        examples (List[Any]): Regular instruction examples
        
    Returns:
        List[CurriculumExample]: Curriculum examples with metadata
    """
    curriculum_examples = []
    
    for example in examples:
        instruction = getattr(example, 'instruction', '')
        input_text = getattr(example, 'input', '')
        output_text = getattr(example, 'output', '')
        
        # Compute metrics
        instruction_length = len(instruction)
        input_length = len(input_text)
        output_length = len(output_text)
        sequence_length = instruction_length + input_length + output_length
        complexity_score = compute_complexity_score(example)
        
        curriculum_example = CurriculumExample(
            example=example,
            sequence_length=sequence_length,
            instruction_length=instruction_length,
            output_length=output_length,
            complexity_score=complexity_score,
            metadata={
                'instruction_words': len(instruction.split()),
                'input_words': len(input_text.split()),
                'output_words': len(output_text.split())
            }
        )
        
        curriculum_examples.append(curriculum_example)
    
    return curriculum_examples


class CurriculumTrainerMixin:
    """
    Mixin class to add curriculum learning capabilities to trainers.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_dataset = None
        self.current_epoch = 0
        self.total_epochs = 0
    
    def set_curriculum_dataset(self, curriculum_dataset: CurriculumDataset):
        """
        Set curriculum dataset for training.
        
        Args:
            curriculum_dataset (CurriculumDataset): Curriculum dataset
        """
        self.curriculum_dataset = curriculum_dataset
    
    def train_with_curriculum(self, data_loader, num_epochs: int, 
                            log_interval: int = 100, **kwargs):
        """
        Train with curriculum learning.
        
        Args:
            data_loader: DataLoader for training data
            num_epochs (int): Number of epochs to train
            log_interval (int): How often to print training progress
            **kwargs: Additional arguments for training
        """
        self.total_epochs = num_epochs
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Update curriculum
            if self.curriculum_dataset is not None:
                self.curriculum_dataset.update_curriculum(epoch, num_epochs)
                print(f"Epoch {epoch + 1}/{num_epochs} - "
                      f"Curriculum difficulty: {self.curriculum_dataset.scheduler.get_current_difficulty():.2f} - "
                      f"Examples: {len(self.curriculum_dataset)}")
            
            # Train for one epoch
            epoch_loss = self.train_epoch(data_loader, log_interval, **kwargs)
            epoch_perplexity = torch.exp(torch.tensor(epoch_loss)).item()
            
            print(f"Epoch {epoch + 1} completed | "
                  f"Average Loss: {epoch_loss:6.3f} | "
                  f"Perplexity: {epoch_perplexity:6.2f}")


# Example usage functions
def demonstrate_curriculum_strategies():
    """Demonstrate different curriculum strategies."""
    print("Curriculum Learning Strategies Demonstration")
    print("=" * 45)
    
    # Create mock examples with varying complexity
    mock_examples = []
    for i in range(10):
        # Create examples with increasing complexity
        instruction = "instruction " * (i + 1)
        input_text = "input " * (i // 2)
        output_text = "output " * (i + 2)
        
        # Mock example object
        class MockExample:
            def __init__(self, instruction, input_text, output_text):
                self.instruction = instruction
                self.input = input_text
                self.output = output_text
        
        example = MockExample(instruction, input_text, output_text)
        mock_examples.append(example)
    
    # Create curriculum examples
    curriculum_examples = create_curriculum_examples(mock_examples)
    
    print(f"Created {len(curriculum_examples)} curriculum examples")
    
    # Demonstrate different strategies
    strategies = [
        CurriculumStrategy.SEQUENCE_LENGTH,
        CurriculumStrategy.COMPLEXITY_SCORE,
        CurriculumStrategy.INSTRUCTION_LENGTH,
        CurriculumStrategy.OUTPUT_LENGTH,
        CurriculumStrategy.RANDOM
    ]
    
    for strategy in strategies:
        print(f"\n{strategy.value.upper()} Strategy:")
        dataset = CurriculumDataset(curriculum_examples, strategy=strategy)
        
        # Show first 3 examples in order
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            print(f"  Example {i+1}: SeqLen={example.sequence_length}, "
                  f"Complexity={example.complexity_score:.1f}")
    
    # Demonstrate curriculum scheduler
    print("\nCurriculum Scheduler Demonstration:")
    scheduler = CurriculumScheduler(strategy="linear")
    
    for epoch in range(5):
        difficulty = scheduler.update_difficulty(epoch, 10)
        print(f"  Epoch {epoch + 1}: Difficulty = {difficulty:.2f}")


if __name__ == "__main__":
    demonstrate_curriculum_strategies()