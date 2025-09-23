"""
Implementation of byte-level tokenizer for LLM training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Union


class ByteTokenizer:
    """
    Byte-level tokenizer that converts text to bytes and back.
    
    This tokenizer maps each byte value (0-255) to a token ID, providing a simple
    way to tokenize text without requiring a vocabulary file.
    """
    
    def __init__(self):
        # Byte values range from 0 to 255
        self.vocab_size = 256
        self.pad_token_id = 0
        self.bos_token_id = 1  # Beginning of sequence
        self.eos_token_id = 2  # End of sequence
        
        # Special tokens
        self.special_tokens = {
            'pad': self.pad_token_id,
            'bos': self.bos_token_id,
            'eos': self.eos_token_id
        }
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text into byte-level token IDs.
        
        Args:
            text (str): Input text to encode
            add_bos (bool): Whether to add beginning-of-sequence token
            add_eos (bool): Whether to add end-of-sequence token
            
        Returns:
            List[int]: List of token IDs
        """
        # Convert text to bytes
        byte_values = list(text.encode('utf-8'))
        
        # Add special tokens if requested
        if add_bos:
            byte_values = [self.bos_token_id] + byte_values
        if add_eos:
            byte_values = byte_values + [self.eos_token_id]
            
        return byte_values
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids (List[int]): List of token IDs to decode
            
        Returns:
            str: Decoded text
        """
        # Filter out special tokens
        byte_values = [token_id for token_id in token_ids 
                      if token_id not in self.special_tokens.values()]
        
        # Convert byte values back to text
        try:
            byte_array = bytes(byte_values)
            text = byte_array.decode('utf-8')
        except UnicodeDecodeError:
            # Handle decoding errors by ignoring invalid bytes
            byte_array = bytes([b for b in byte_values if b < 128])
            text = byte_array.decode('utf-8', errors='ignore')
            
        return text
    
    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts (List[str]): List of texts to encode
            add_bos (bool): Whether to add beginning-of-sequence token
            add_eos (bool): Whether to add end-of-sequence token
            
        Returns:
            List[List[int]]: List of token ID sequences
        """
        return [self.encode(text, add_bos, add_eos) for text in texts]
    
    def decode_batch(self, batch_token_ids: List[List[int]]) -> List[str]:
        """
        Decode a batch of token ID sequences.
        
        Args:
            batch_token_ids (List[List[int]]): List of token ID sequences
            
        Returns:
            List[str]: List of decoded texts
        """
        return [self.decode(token_ids) for token_ids in batch_token_ids]


def test_byte_tokenizer():
    """
    Test function for ByteTokenizer.
    """
    # Initialize tokenizer
    tokenizer = ByteTokenizer()
    
    # Test basic encoding/decoding
    text = "Hello, world!"
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    decoded_text = tokenizer.decode(token_ids)
    
    print(f"Original text: {text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded text: {decoded_text}")
    
    # Verify that decoding works correctly (ignoring special tokens)
    # The decoded text should match the original (special tokens are not included in decoded output)
    assert decoded_text == text, f"Decoding failed: {decoded_text} != {text}"
    
    # Test batch encoding/decoding
    texts = ["Hello", "World", "Byte-level tokenization"]
    batch_token_ids = tokenizer.encode_batch(texts, add_bos=True, add_eos=True)
    decoded_texts = tokenizer.decode_batch(batch_token_ids)
    
    print(f"\nBatch encoding/decoding test:")
    for i, (original, decoded) in enumerate(zip(texts, decoded_texts)):
        print(f"  {i}: '{original}' -> '{decoded}'")
        assert original == decoded, f"Batch decoding failed: {original} != {decoded}"
    
    # Test vocabulary size
    assert tokenizer.vocab_size == 256, f"Vocabulary size mismatch: {tokenizer.vocab_size}"
    
    print("ByteTokenizer test passed!")


if __name__ == "__main__":
    test_byte_tokenizer()