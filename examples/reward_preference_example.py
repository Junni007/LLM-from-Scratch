"""
Example demonstrating Reward Modeling preference datasets.
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from reward.preference_data import (
    PreferenceExample, 
    PreferenceDataset, 
    PreferenceDataProcessor,
    generate_sample_preference_data,
    create_preference_data_loader
)


class MockTokenizer:
    """Mock tokenizer for demonstration purposes."""
    
    def __init__(self):
        self.vocab_size = 1000
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
    
    def encode(self, text, add_bos=False, add_eos=False):
        """Simple mock encoding."""
        # Convert text to list of integers (mock tokenization)
        tokens = [ord(c) % self.vocab_size for c in text[:50]]  # Limit to 50 tokens
        
        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token_id]
            
        return tokens


def main():
    """Main function demonstrating reward modeling preference datasets."""
    print("Reward Modeling Preference Dataset Example")
    print("=" * 45)
    
    # Generate sample preference data
    print("1. Generating sample preference data...")
    examples = generate_sample_preference_data()
    print(f"  Created {len(examples)} preference examples")
    
    # Show examples
    print("\n2. Sample preference examples:")
    for i, example in enumerate(examples[:3]):  # Show first 3 examples
        print(f"\n  Example {i+1}:")
        print(f"    Prompt: {example.prompt}")
        print(f"    Chosen Response: {example.chosen_response}")
        print(f"    Rejected Response: {example.rejected_response}")
    
    # Create PreferenceDataset
    print("\n3. Creating PreferenceDataset...")
    dataset = PreferenceDataset(examples)
    print(f"  Dataset size: {len(dataset)}")
    
    # Test indexing
    print("\n4. Testing dataset indexing...")
    first_example = dataset[0]
    print(f"  First example prompt: {first_example.prompt[:50]}...")
    
    # Test saving and loading
    print("\n5. Testing save/load functionality...")
    save_file = "preference_examples.json"
    dataset.save_to_json(save_file)
    print(f"  Saved to {save_file}")
    
    loaded_dataset = PreferenceDataset.load_from_json(save_file)
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
    if os.path.exists(save_file):
        os.remove(save_file)
        print(f"  Cleaned up test file {save_file}")
    
    # Test data processing
    print("\n6. Testing data processing...")
    tokenizer = MockTokenizer()
    processor = PreferenceDataProcessor(max_length=128, pad_token_id=tokenizer.pad_token_id)
    
    # Process first example
    processed = processor.preprocess_example(first_example, tokenizer)
    print(f"  Processed example shapes:")
    print(f"    Chosen input_ids: {processed['chosen_input_ids'].shape}")
    print(f"    Chosen attention_mask: {processed['chosen_attention_mask'].shape}")
    print(f"    Rejected input_ids: {processed['rejected_input_ids'].shape}")
    print(f"    Rejected attention_mask: {processed['rejected_attention_mask'].shape}")
    print(f"    Prompt length: {processed['prompt_length']}")
    
    # Process entire dataset
    print("\n7. Processing entire dataset...")
    processed_dataset = processor.create_dataset(examples, tokenizer)
    print(f"  Processed dataset size: {len(processed_dataset)}")
    
    # Create DataLoader
    print("\n8. Creating DataLoader...")
    data_loader = create_preference_data_loader(processed_dataset, batch_size=2, shuffle=True)
    print(f"  DataLoader batch size: 2")
    print(f"  Number of batches: {len(data_loader)}")
    
    # Show first batch
    print("\n9. First batch details:")
    for batch_idx, batch in enumerate(data_loader):
        print(f"  Batch {batch_idx + 1}:")
        print(f"    Chosen input_ids shape: {batch['chosen_input_ids'].shape}")
        print(f"    Chosen attention_mask shape: {batch['chosen_attention_mask'].shape}")
        print(f"    Rejected input_ids shape: {batch['rejected_input_ids'].shape}")
        print(f"    Rejected attention_mask shape: {batch['rejected_attention_mask'].shape}")
        print(f"    Prompt lengths: {batch['prompt_lengths']}")
        
        # Show sample tokens
        print(f"    Sample chosen tokens: {batch['chosen_input_ids'][0][:10].tolist()}")
        print(f"    Sample rejected tokens: {batch['rejected_input_ids'][0][:10].tolist()}")
        break  # Just show first batch
    
    # Test with custom examples
    print("\n10. Testing with custom examples...")
    custom_examples = [
        PreferenceExample(
            prompt="What is 2+2?",
            chosen_response="2+2 equals 4.",
            rejected_response="I don't know math.",
            metadata={"difficulty": "easy", "domain": "math"}
        ),
        PreferenceExample(
            prompt="Explain quantum computing.",
            chosen_response="Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform operations on data.",
            rejected_response="It's about small computers.",
            metadata={"difficulty": "hard", "domain": "physics"}
        )
    ]
    
    custom_dataset = PreferenceDataset(custom_examples)
    print(f"  Created custom dataset with {len(custom_dataset)} examples")
    
    # Show metadata
    for i, example in enumerate(custom_dataset):
        print(f"    Example {i+1} metadata: {example.metadata}")
    
    print("\nReward modeling preference dataset example completed!")


if __name__ == "__main__":
    main()