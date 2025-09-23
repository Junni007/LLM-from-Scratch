"""
Example demonstrating Supervised Fine-Tuning (SFT) instruction dataset formatting.
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sft.instruction_formatting import (
    InstructionExample, 
    InstructionFormatter, 
    PromptTemplate, 
    DatasetPreprocessor,
    load_instruction_dataset,
    save_instruction_dataset
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
    """Main function demonstrating SFT instruction formatting."""
    print("Supervised Fine-Tuning (SFT) Instruction Formatting Example")
    print("=" * 60)
    
    # Create sample instruction examples
    print("1. Creating sample instruction examples...")
    examples = [
        InstructionExample(
            instruction="Explain the concept of photosynthesis.",
            input="",
            output="Photosynthesis is the process by which plants convert light energy into chemical energy. During photosynthesis, plants use sunlight, carbon dioxide from the air, and water from the soil to produce glucose and oxygen. This process occurs in the chloroplasts of plant cells, specifically in the chlorophyll pigments that give plants their green color."
        ),
        InstructionExample(
            instruction="Translate the following English text to French.",
            input="Hello, how are you today?",
            output="Bonjour, comment allez-vous aujourd'hui?"
        ),
        InstructionExample(
            instruction="Write a Python function to calculate the factorial of a number.",
            input="",
            output="def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n\n# Example usage\nprint(factorial(5))  # Output: 120"
        ),
        InstructionExample(
            instruction="Summarize the main points of the given text.",
            input="Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn from and make predictions or decisions based on data. It has applications in various fields including image recognition, natural language processing, and recommendation systems.",
            output="Machine learning is a branch of AI that develops algorithms to learn from data and make predictions. It's used in image recognition, natural language processing, and recommendation systems."
        )
    ]
    
    print(f"  Created {len(examples)} instruction examples")
    
    # Test different prompt templates
    print("\n2. Testing different prompt templates:")
    templates = [
        PromptTemplate.ALPACA,
        PromptTemplate.VICUNA,
        PromptTemplate.LLAMA2,
        PromptTemplate.OPENASSISTANT
    ]
    
    for template in templates:
        print(f"\n  {template.value.upper()} Template:")
        formatter = InstructionFormatter(template)
        
        # Format first example with this template
        formatted = formatter.format_example(examples[0])
        print(f"    Formatted example (first 150 chars): {formatted[:150]}...")
        
        # Show chat template example
        conversation = [
            {"role": "user", "content": examples[0].instruction},
            {"role": "assistant", "content": examples[0].output}
        ]
        chat_formatted = formatter.create_chat_template(conversation)
        print(f"    Chat format (first 100 chars): {chat_formatted[:100]}...")
    
    # Test batch formatting
    print("\n3. Testing batch formatting:")
    formatter = InstructionFormatter(PromptTemplate.ALPACA)
    batch_formatted = formatter.format_batch(examples)
    print(f"  Formatted {len(batch_formatted)} examples in batch")
    
    # Test data loading and saving
    print("\n4. Testing data loading and saving:")
    
    # Save examples to file
    save_file = "sft_examples.json"
    save_instruction_dataset(examples, save_file)
    print(f"  Saved examples to {save_file}")
    
    # Load examples from file
    loaded_examples = load_instruction_dataset(save_file)
    print(f"  Loaded {len(loaded_examples)} examples from file")
    
    # Verify loaded data
    assert len(loaded_examples) == len(examples)
    print("  Data integrity check passed")
    
    # Clean up test file
    import os
    if os.path.exists(save_file):
        os.remove(save_file)
        print(f"  Cleaned up test file {save_file}")
    
    # Test dataset preprocessing
    print("\n5. Testing dataset preprocessing:")
    
    # Create mock tokenizer
    tokenizer = MockTokenizer()
    preprocessor = DatasetPreprocessor(max_length=128, pad_token_id=tokenizer.pad_token_id)
    
    # Preprocess a formatted example
    formatted_text = formatter.format_example(examples[0])
    processed = preprocessor.preprocess_text(formatted_text, tokenizer)
    
    print(f"  Original text length: {len(formatted_text)} characters")
    print(f"  Tokenized input_ids shape: {processed['input_ids'].shape}")
    print(f"  Attention mask shape: {processed['attention_mask'].shape}")
    print(f"  Sample tokens: {processed['input_ids'][:10].tolist()}")
    
    # Test instruction dataset creation
    print("\n6. Testing instruction dataset creation:")
    instruction_dataset = preprocessor.create_instruction_dataset(
        examples, tokenizer, formatter
    )
    
    print(f"  Created dataset with {len(instruction_dataset)} examples")
    print(f"  First example input_ids shape: {instruction_dataset[0]['input_ids'].shape}")
    
    # Test label masking
    print("\n7. Testing label masking:")
    sample_input_ids = processed['input_ids']
    # Use a mock instruction end token (in practice, this would be a specific token ID)
    instruction_end_token = 2  # Using EOS token as example
    masked_labels = preprocessor.mask_instruction_labels(
        sample_input_ids, instruction_end_token
    )
    
    print(f"  Original input_ids: {sample_input_ids[:10].tolist()}")
    print(f"  Masked labels: {masked_labels[:10].tolist()}")
    print(f"  Number of masked tokens: {(masked_labels == -100).sum().item()}")
    
    # Show parameter counts for different templates
    print("\n8. Template comparison:")
    template_info = {
        "Alpaca": "Standard template with ### headers",
        "Vicuna": "Chat-based with USER/ASSISTANT format",
        "LLaMA2": "System prompt support with [INST] tags",
        "OpenAssistant": "Prompter/assistant format with <|tokens|>"
    }
    
    for name, description in template_info.items():
        print(f"  {name:12}: {description}")
    
    print("\nSFT instruction formatting example completed!")


if __name__ == "__main__":
    main()