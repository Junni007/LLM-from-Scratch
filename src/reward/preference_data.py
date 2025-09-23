"""
Preference datasets for Reward Modeling in LLMs.

This module implements preference datasets for training reward models,
including pairwise ranking datasets and utilities for processing preference data.
"""

import torch
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


@dataclass
class PreferenceExample:
    """
    Data class for preference examples.
    
    Represents a single preference comparison between two responses to the same prompt.
    """
    prompt: str
    chosen_response: str
    rejected_response: str
    chosen_score: Optional[float] = None  # Optional score for the chosen response
    rejected_score: Optional[float] = None  # Optional score for the rejected response
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "chosen_response": self.chosen_response,
            "rejected_response": self.rejected_response,
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreferenceExample':
        """Create from dictionary."""
        return cls(
            prompt=data["prompt"],
            chosen_response=data["chosen_response"],
            rejected_response=data["rejected_response"],
            chosen_score=data.get("chosen_score"),
            rejected_score=data.get("rejected_score"),
            metadata=data.get("metadata")
        )


class PreferenceDataset(Dataset):
    """
    Dataset class for preference data.
    
    This class handles preference examples for training reward models.
    Each example contains a prompt and two responses, one preferred over the other.
    """
    
    def __init__(self, examples: List[PreferenceExample]):
        """
        Initialize PreferenceDataset.
        
        Args:
            examples (List[PreferenceExample]): List of preference examples
        """
        self.examples = examples
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> PreferenceExample:
        """
        Get a single preference example.
        
        Args:
            idx (int): Index of the example to retrieve
            
        Returns:
            PreferenceExample: The preference example
        """
        return self.examples[idx]
    
    def save_to_json(self, file_path: str):
        """
        Save dataset to JSON file.
        
        Args:
            file_path (str): Path to save the JSON file
        """
        data = [example.to_dict() for example in self.examples]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, file_path: str) -> 'PreferenceDataset':
        """
        Load dataset from JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            PreferenceDataset: Loaded dataset
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = [PreferenceExample.from_dict(item) for item in data]
        return cls(examples)


class PreferenceDataProcessor:
    """
    Processor for preference data.
    
    Handles tokenization and preprocessing of preference examples for training.
    """
    
    def __init__(self, max_length: int = 512, pad_token_id: int = 0):
        """
        Initialize PreferenceDataProcessor.
        
        Args:
            max_length (int): Maximum sequence length
            pad_token_id (int): Padding token ID
        """
        self.max_length = max_length
        self.pad_token_id = pad_token_id
    
    def preprocess_example(self, example: PreferenceExample, tokenizer) -> Dict[str, torch.Tensor]:
        """
        Preprocess a single preference example.
        
        Args:
            example (PreferenceExample): Preference example to preprocess
            tokenizer: Tokenizer to use
            
        Returns:
            Dict[str, torch.Tensor]: Preprocessed tensors for chosen and rejected responses
        """
        # Tokenize prompt
        prompt_tokens = tokenizer.encode(example.prompt, add_bos=True)
        
        # Tokenize chosen response
        chosen_tokens = tokenizer.encode(example.chosen_response, add_eos=True)
        
        # Tokenize rejected response
        rejected_tokens = tokenizer.encode(example.rejected_response, add_eos=True)
        
        # Combine prompt with responses
        chosen_input_ids = prompt_tokens + chosen_tokens
        rejected_input_ids = prompt_tokens + rejected_tokens
        
        # Truncate or pad sequences
        chosen_input_ids = self._pad_or_truncate(chosen_input_ids)
        rejected_input_ids = self._pad_or_truncate(rejected_input_ids)
        
        # Create attention masks
        chosen_attention_mask = self._create_attention_mask(chosen_input_ids)
        rejected_attention_mask = self._create_attention_mask(rejected_input_ids)
        
        return {
            "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_attention_mask, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_attention_mask, dtype=torch.long),
            "prompt_length": len(prompt_tokens)
        }
    
    def _pad_or_truncate(self, tokens: List[int]) -> List[int]:
        """
        Pad or truncate tokens to max_length.
        
        Args:
            tokens (List[int]): List of token IDs
            
        Returns:
            List[int]: Padded or truncated tokens
        """
        if len(tokens) > self.max_length:
            # Truncate
            return tokens[:self.max_length]
        else:
            # Pad
            padding_length = self.max_length - len(tokens)
            return tokens + [self.pad_token_id] * padding_length
    
    def _create_attention_mask(self, tokens: List[int]) -> List[int]:
        """
        Create attention mask for tokens.
        
        Args:
            tokens (List[int]): List of token IDs
            
        Returns:
            List[int]: Attention mask (1 for real tokens, 0 for padding)
        """
        return [1 if token != self.pad_token_id else 0 for token in tokens]
    
    def create_dataset(self, examples: List[PreferenceExample], 
                      tokenizer) -> List[Dict[str, torch.Tensor]]:
        """
        Create processed dataset from preference examples.
        
        Args:
            examples (List[PreferenceExample]): Preference examples
            tokenizer: Tokenizer to use
            
        Returns:
            List[Dict[str, torch.Tensor]]: Processed dataset
        """
        processed_examples = []
        for example in examples:
            processed = self.preprocess_example(example, tokenizer)
            processed_examples.append(processed)
        
        return processed_examples


def create_preference_data_loader(dataset: List[Dict[str, torch.Tensor]], 
                                batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for preference data.
    
    Args:
        dataset (List[Dict[str, torch.Tensor]]): Processed preference dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: DataLoader for the preference data
    """
    # Custom collate function for preference data
    def collate_fn(batch):
        # Stack tensors for each field
        chosen_input_ids = torch.stack([item["chosen_input_ids"] for item in batch])
        chosen_attention_mask = torch.stack([item["chosen_attention_mask"] for item in batch])
        rejected_input_ids = torch.stack([item["rejected_input_ids"] for item in batch])
        rejected_attention_mask = torch.stack([item["rejected_attention_mask"] for item in batch])
        prompt_lengths = [item["prompt_length"] for item in batch]
        
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "prompt_lengths": prompt_lengths
        }
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def generate_sample_preference_data() -> List[PreferenceExample]:
    """
    Generate sample preference data for testing and demonstration.
    
    Returns:
        List[PreferenceExample]: Sample preference examples
    """
    examples = [
        PreferenceExample(
            prompt="Explain the concept of photosynthesis.",
            chosen_response="Photosynthesis is the process by which plants convert light energy into chemical energy. During photosynthesis, plants use sunlight, carbon dioxide from the air, and water from the soil to produce glucose and oxygen. This process occurs in the chloroplasts of plant cells, specifically in the chlorophyll pigments that give plants their green color.",
            rejected_response="Plants make food using sunlight."
        ),
        PreferenceExample(
            prompt="What is the capital of France?",
            chosen_response="The capital of France is Paris.",
            rejected_response="I don't know."
        ),
        PreferenceExample(
            prompt="Write a Python function to calculate factorial.",
            chosen_response="def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n\n# Example usage\nprint(factorial(5))  # Output: 120",
            rejected_response="def fact(n):\n    return n * fact(n-1)"
        ),
        PreferenceExample(
            prompt="Translate 'Hello, how are you?' to French.",
            chosen_response="Bonjour, comment allez-vous?",
            rejected_response="Hello in French is Bonjour."
        ),
        PreferenceExample(
            prompt="What are the benefits of exercise?",
            chosen_response="Regular exercise provides numerous benefits including improved cardiovascular health, increased muscle strength, better mental health, enhanced sleep quality, and reduced risk of chronic diseases such as diabetes and heart disease.",
            rejected_response="Exercise is good for you."
        )
    ]
    
    return examples


def demonstrate_preference_data():
    """Demonstrate preference data functionality."""
    print("Preference Dataset Demonstration")
    print("=" * 35)
    
    # Generate sample data
    print("1. Generating sample preference data...")
    examples = generate_sample_preference_data()
    print(f"  Created {len(examples)} preference examples")
    
    # Show first example
    first_example = examples[0]
    print(f"\n  First example:")
    print(f"    Prompt: {first_example.prompt}")
    print(f"    Chosen: {first_example.chosen_response[:50]}...")
    print(f"    Rejected: {first_example.rejected_response}")
    
    # Create dataset
    print("\n2. Creating PreferenceDataset...")
    dataset = PreferenceDataset(examples)
    print(f"  Dataset size: {len(dataset)}")
    
    # Test saving and loading
    print("\n3. Testing save/load functionality...")
    test_file = "test_preference_data.json"
    dataset.save_to_json(test_file)
    print(f"  Saved to {test_file}")
    
    loaded_dataset = PreferenceDataset.load_from_json(test_file)
    print(f"  Loaded dataset size: {len(loaded_dataset)}")
    
    # Verify data integrity
    original_example = dataset[0]
    loaded_example = loaded_dataset[0]
    assert original_example.prompt == loaded_example.prompt
    assert original_example.chosen_response == loaded_example.chosen_response
    assert original_example.rejected_response == loaded_example.rejected_response
    print("  Data integrity check passed")
    
    # Clean up test file
    import os
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"  Cleaned up test file {test_file}")
    
    print("\nPreference dataset demonstration completed!")


if __name__ == "__main__":
    demonstrate_preference_data()