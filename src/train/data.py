"""
Data loading and batching utilities for LLM training.
"""

import torch
from typing import List, Tuple, Iterator
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """
    Dataset class for text data.
    
    This class handles tokenized text data and provides indexing for DataLoader.
    """
    
    def __init__(self, tokenized_texts: List[List[int]], seq_length: int):
        """
        Initialize TextDataset.
        
        Args:
            tokenized_texts (List[List[int]]): List of tokenized texts
            seq_length (int): Sequence length for each sample
        """
        self.seq_length = seq_length
        self.samples = []
        
        # Process each tokenized text into samples of fixed length
        for tokens in tokenized_texts:
            # Create samples by sliding a window of seq_length over the tokens
            for i in range(0, len(tokens) - seq_length):
                sample = tokens[i:i + seq_length + 1]  # +1 for target
                if len(sample) == seq_length + 1:
                    self.samples.append(sample)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input sequence and target sequence
        """
        sample = self.samples[idx]
        # Split into input (first seq_length tokens) and target (last seq_length tokens)
        input_seq = torch.tensor(sample[:-1], dtype=torch.long)
        target_seq = torch.tensor(sample[1:], dtype=torch.long)
        return input_seq, target_seq


def create_data_loader(tokenized_texts: List[List[int]], seq_length: int, 
                      batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for text data.
    
    Args:
        tokenized_texts (List[List[int]]): List of tokenized texts
        seq_length (int): Sequence length for each sample
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: DataLoader for the text data
    """
    dataset = TextDataset(tokenized_texts, seq_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def test_text_dataset():
    """
    Test function for TextDataset.
    """
    # Create sample tokenized texts
    tokenized_texts = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Sample 1
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # Sample 2
    ]
    
    seq_length = 4
    dataset = TextDataset(tokenized_texts, seq_length)
    
    # Check dataset length
    expected_length = (10 - seq_length) + (10 - seq_length)  # 12 samples total
    assert len(dataset) == expected_length, f"Dataset length mismatch: {len(dataset)} != {expected_length}"
    
    # Check first sample
    input_seq, target_seq = dataset[0]
    expected_input = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    expected_target = torch.tensor([2, 3, 4, 5], dtype=torch.long)
    
    assert torch.equal(input_seq, expected_input), f"Input sequence mismatch: {input_seq} != {expected_input}"
    assert torch.equal(target_seq, expected_target), f"Target sequence mismatch: {target_seq} != {expected_target}"
    
    print("TextDataset test passed!")


def test_data_loader():
    """
    Test function for data loader creation.
    """
    # Create sample tokenized texts
    tokenized_texts = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ]
    
    seq_length = 4
    batch_size = 3
    
    data_loader = create_data_loader(tokenized_texts, seq_length, batch_size)
    
    # Check that we can iterate through the data loader
    batch_count = 0
    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        batch_count += 1
        # Check batch shapes
        assert input_batch.shape[1] == seq_length, f"Input sequence length mismatch: {input_batch.shape[1]} != {seq_length}"
        assert target_batch.shape[1] == seq_length, f"Target sequence length mismatch: {target_batch.shape[1]} != {seq_length}"
        assert input_batch.shape[0] <= batch_size, f"Batch size mismatch: {input_batch.shape[0]} > {batch_size}"
        
        # Check that input and target have the same batch size
        assert input_batch.shape[0] == target_batch.shape[0], f"Batch size mismatch between input and target"
        
        # Check shifting: target should be input shifted by one position
        if batch_idx == 0 and input_batch.shape[0] == batch_size:
            # Check first batch if it's full
            for i in range(batch_size):
                assert torch.equal(input_batch[i, 1:], target_batch[i, :-1]), \
                    f"Shifting mismatch in batch {batch_idx}, sample {i}"
    
    assert batch_count > 0, "No batches were generated"
    
    print("DataLoader test passed!")


if __name__ == "__main__":
    test_text_dataset()
    test_data_loader()