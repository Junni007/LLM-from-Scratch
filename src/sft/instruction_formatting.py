"""
Instruction dataset formatting for Supervised Fine-Tuning (SFT).

This module implements utilities for formatting instruction datasets for SFT,
including prompt templates, response formatting, and dataset preprocessing.
"""

import json
import torch
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class PromptTemplate(Enum):
    """Standard prompt templates for instruction tuning."""
    ALPACA = "alpaca"
    VICUNA = "vicuna"
    LLAMA2 = "llama2"
    OPENASSISTANT = "openassistant"
    CUSTOM = "custom"


@dataclass
class InstructionExample:
    """Data class for instruction examples."""
    instruction: str
    input: str = ""
    output: str = ""
    system_prompt: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "system_prompt": self.system_prompt
        }


class InstructionFormatter:
    """Formatter for instruction datasets."""
    
    def __init__(self, template: PromptTemplate = PromptTemplate.ALPACA):
        """
        Initialize InstructionFormatter.
        
        Args:
            template (PromptTemplate): Template to use for formatting
        """
        self.template = template
        self.templates = {
            PromptTemplate.ALPACA: {
                "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
                "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
                "response": "{output}"
            },
            PromptTemplate.VICUNA: {
                "prompt": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction}\n{input} ASSISTANT:",
                "prompt_no_input": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:",
                "response": "{output}"
            },
            PromptTemplate.LLAMA2: {
                "prompt": "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction}\n{input} [/INST]",
                "prompt_no_input": "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST]",
                "response": "{output}"
            },
            PromptTemplate.OPENASSISTANT: {
                "prompt": "<|prompter|>{instruction}\n{input}<|endoftext|><|assistant|>",
                "prompt_no_input": "<|prompter|>{instruction}<|endoftext|><|assistant|>",
                "response": "{output}<|endoftext|>"
            }
        }
    
    def format_example(self, example: InstructionExample) -> str:
        """
        Format a single instruction example.
        
        Args:
            example (InstructionExample): Example to format
            
        Returns:
            str: Formatted example
        """
        template_dict = self.templates.get(self.template, self.templates[PromptTemplate.ALPACA])
        
        # Choose appropriate template based on whether input is provided
        if example.input.strip():
            prompt_template = template_dict["prompt"]
        else:
            prompt_template = template_dict["prompt_no_input"]
        
        # Format prompt
        prompt = prompt_template.format(
            instruction=example.instruction,
            input=example.input,
            system_prompt=example.system_prompt
        )
        
        # Format response
        response = template_dict["response"].format(output=example.output)
        
        return prompt + response
    
    def format_batch(self, examples: List[InstructionExample]) -> List[str]:
        """
        Format a batch of instruction examples.
        
        Args:
            examples (List[InstructionExample]): Examples to format
            
        Returns:
            List[str]: Formatted examples
        """
        return [self.format_example(example) for example in examples]
    
    def create_chat_template(self, conversation: List[Dict[str, str]]) -> str:
        """
        Create a chat template from a conversation history.
        
        Args:
            conversation (List[Dict[str, str]]): List of {"role": "user"/"assistant", "content": "..."}
            
        Returns:
            str: Formatted conversation
        """
        if self.template == PromptTemplate.VICUNA:
            formatted = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            for turn in conversation:
                if turn["role"] == "user":
                    formatted += f" USER: {turn['content']}"
                elif turn["role"] == "assistant":
                    formatted += f" ASSISTANT: {turn['content']}"
            return formatted
        elif self.template == PromptTemplate.LLAMA2:
            formatted = ""
            for turn in conversation:
                if turn["role"] == "user":
                    formatted += f"[INST] {turn['content']} [/INST]"
                elif turn["role"] == "assistant":
                    formatted += f" {turn['content']}"
            return formatted
        else:
            # Default to Alpaca-style formatting
            formatted = ""
            for turn in conversation:
                if turn["role"] == "user":
                    formatted += f"### Instruction:\n{turn['content']}\n\n"
                elif turn["role"] == "assistant":
                    formatted += f"### Response:\n{turn['content']}\n\n"
            return formatted


class DatasetPreprocessor:
    """Preprocessor for instruction datasets."""
    
    def __init__(self, max_length: int = 512, pad_token_id: int = 0):
        """
        Initialize DatasetPreprocessor.
        
        Args:
            max_length (int): Maximum sequence length
            pad_token_id (int): Padding token ID
        """
        self.max_length = max_length
        self.pad_token_id = pad_token_id
    
    def preprocess_text(self, text: str, tokenizer) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for training.
        
        Args:
            text (str): Text to preprocess
            tokenizer: Tokenizer to use
            
        Returns:
            Dict[str, torch.Tensor]: Preprocessed tensors
        """
        # Tokenize text
        encoding = tokenizer.encode(text, add_bos=True, add_eos=True)
        
        # Convert to tensor
        input_ids = torch.tensor(encoding, dtype=torch.long)
        
        # Truncate or pad to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            padding_length = self.max_length - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            ])
        
        # Create attention mask
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def create_instruction_dataset(self, examples: List[InstructionExample], 
                                 tokenizer, formatter: InstructionFormatter = None) -> List[Dict[str, torch.Tensor]]:
        """
        Create instruction dataset for training.
        
        Args:
            examples (List[InstructionExample]): Instruction examples
            tokenizer: Tokenizer to use
            formatter (InstructionFormatter, optional): Formatter to use
            
        Returns:
            List[Dict[str, torch.Tensor]]: Processed dataset
        """
        if formatter is None:
            formatter = InstructionFormatter()
        
        dataset = []
        for example in examples:
            # Format example
            formatted_text = formatter.format_example(example)
            
            # Preprocess text
            processed = self.preprocess_text(formatted_text, tokenizer)
            dataset.append(processed)
        
        return dataset
    
    def mask_instruction_labels(self, input_ids: torch.Tensor, 
                              instruction_end_token: int) -> torch.Tensor:
        """
        Create labels with instruction part masked out.
        
        Args:
            input_ids (torch.Tensor): Input IDs
            instruction_end_token (int): Token ID that marks end of instruction
            
        Returns:
            torch.Tensor: Labels with instruction part masked
        """
        labels = input_ids.clone()
        
        # Find where instruction ends
        instruction_end_pos = (input_ids == instruction_end_token).nonzero(as_tuple=True)[0]
        if len(instruction_end_pos) > 0:
            # Mask everything up to and including the instruction end token
            end_pos = instruction_end_pos[0].item() + 1
            labels[:end_pos] = -100  # Standard mask token for cross-entropy loss
        
        return labels


def load_instruction_dataset(file_path: str) -> List[InstructionExample]:
    """
    Load instruction dataset from JSON file.
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        List[InstructionExample]: Loaded examples
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        example = InstructionExample(
            instruction=item.get("instruction", ""),
            input=item.get("input", ""),
            output=item.get("output", ""),
            system_prompt=item.get("system_prompt", "")
        )
        examples.append(example)
    
    return examples


def save_instruction_dataset(examples: List[InstructionExample], file_path: str):
    """
    Save instruction dataset to JSON file.
    
    Args:
        examples (List[InstructionExample]): Examples to save
        file_path (str): Path to JSON file
    """
    data = [example.to_dict() for example in examples]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def test_instruction_formatting():
    """Test function for instruction formatting."""
    # Create sample examples
    examples = [
        InstructionExample(
            instruction="Explain the concept of photosynthesis.",
            input="",
            output="Photosynthesis is the process by which plants convert light energy into chemical energy..."
        ),
        InstructionExample(
            instruction="Translate the following English text to French.",
            input="Hello, how are you?",
            output="Bonjour, comment allez-vous?"
        )
    ]
    
    # Test different templates
    for template in [PromptTemplate.ALPACA, PromptTemplate.VICUNA]:
        print(f"\nTesting {template.value} template:")
        formatter = InstructionFormatter(template)
        formatted = formatter.format_batch(examples)
        for i, text in enumerate(formatted):
            print(f"  Example {i+1}: {text[:100]}...")
    
    # Test chat template
    conversation = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is the population of Paris?"},
    ]
    
    print(f"\nTesting chat template:")
    formatter = InstructionFormatter(PromptTemplate.VICUNA)
    chat_text = formatter.create_chat_template(conversation)
    print(f"  Chat: {chat_text}")
    
    # Test data loading and saving
    print(f"\nTesting data loading and saving:")
    save_instruction_dataset(examples, "test_dataset.json")
    loaded_examples = load_instruction_dataset("test_dataset.json")
    assert len(loaded_examples) == len(examples)
    print(f"  Saved and loaded {len(loaded_examples)} examples")
    
    # Clean up test file
    import os
    if os.path.exists("test_dataset.json"):
        os.remove("test_dataset.json")
    
    print("Instruction formatting test passed!")


if __name__ == "__main__":
    test_instruction_formatting()