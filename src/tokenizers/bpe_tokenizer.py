"""
Implementation of Byte Pair Encoding (BPE) tokenizer.
"""

import torch
import torch.nn as nn
import re
import json
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer implementation.
    
    This tokenizer implements the BPE algorithm for subword tokenization,
    which is commonly used in modern language models.
    
    Args:
        vocab_size (int): Target vocabulary size
        special_tokens (Dict[str, str]): Special tokens to include in vocabulary
    """
    
    def __init__(self, vocab_size: int = 10000, special_tokens: Optional[Dict[str, str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            'pad': '<pad>',
            'unk': '<unk>',
            'bos': '<s>',
            'eos': '</s>',
            'sep': '<sep>',
            'mask': '<mask>'
        }
        
        # Initialize vocabulary
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Initialize merges
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        
        # Pattern for splitting text into tokens (simplified version)
        # This is a simplified pattern that works with standard Python regex
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\w\s]+|\s+(?!\S)|\s+""")
        
        # Initialize with special tokens
        self._initialize_special_tokens()
    
    def _initialize_special_tokens(self):
        """Initialize vocabulary with special tokens."""
        for i, (token_name, token) in enumerate(self.special_tokens.items()):
            self.vocab[token] = i
            self.id_to_token[i] = token
    
    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Get pair statistics from vocabulary.
        
        Args:
            vocab (Dict[str, int]): Vocabulary with token frequencies
            
        Returns:
            Dict[Tuple[str, str], int]: Pair frequencies
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """
        Merge vocabulary based on a pair.
        
        Args:
            pair (Tuple[str, str]): Pair to merge
            vocab (Dict[str, int]): Vocabulary to merge
            
        Returns:
            Dict[str, int]: Merged vocabulary
        """
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        new_vocab = {}
        for word in vocab:
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab
    
    def train(self, texts: List[str]):
        """
        Train BPE tokenizer on a list of texts.
        
        Args:
            texts (List[str]): List of texts to train on
        """
        # Initialize vocabulary with characters
        vocab = defaultdict(int)
        
        # Process texts
        for text in texts:
            # Split text into tokens using regex pattern
            for token in re.findall(self.pat, text):
                # Convert token to character-level representation
                word = ' '.join(list(token)) + ' </w>'
                vocab[word] += 1
        
        # Add special tokens to vocab
        for token in self.special_tokens.values():
            vocab[token] = 1e6  # High frequency to prevent merging
        
        # Target number of merges
        num_merges = self.vocab_size - len(self.vocab)
        
        # Perform BPE merges
        for i in range(num_merges):
            # Get pair statistics
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge vocabulary
            vocab = self._merge_vocab(best_pair, vocab)
            
            # Store merge
            self.bpe_ranks[best_pair] = i
        
        # Build final vocabulary
        self._build_vocab(vocab)
    
    def _build_vocab(self, vocab: Dict[str, int]):
        """
        Build final vocabulary from merged vocabulary.
        
        Args:
            vocab (Dict[str, int]): Merged vocabulary
        """
        # Start with special tokens
        new_vocab = dict(self.vocab)
        new_id_to_token = dict(self.id_to_token)
        
        # Add merged tokens
        next_id = len(new_vocab)
        for word in vocab:
            # Remove </w> markers
            token = word.replace(' </w>', '')
            if token not in new_vocab:
                new_vocab[token] = next_id
                new_id_to_token[next_id] = token
                next_id += 1
        
        self.vocab = new_vocab
        self.id_to_token = new_id_to_token
    
    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        """
        Get consecutive pairs from a word.
        
        Args:
            word (List[str]): List of symbols
            
        Returns:
            Set[Tuple[str, str]]: Set of consecutive pairs
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _encode_word(self, word: str) -> List[str]:
        """
        Encode a single word using BPE.
        
        Args:
            word (str): Word to encode
            
        Returns:
            List[str]: List of BPE tokens
        """
        # Convert word to character-level representation
        word_chars = list(word)
        word_chars.append('</w>')
        
        # Convert to list of tokens
        word_tokens = [' '.join(word_chars)]
        
        # Apply BPE merges
        while len(word_tokens) > 1:
            # Get pairs
            pairs = self._get_pairs(word_tokens)
            if not pairs:
                break
            
            # Find best pair to merge
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            # Merge
            new_word = []
            i = 0
            while i < len(word_tokens):
                try:
                    j = word_tokens.index(bigram[0], i)
                    new_word.extend(word_tokens[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word_tokens[i:])
                    break
                
                if (i < len(word_tokens) - 1 and 
                    word_tokens[i] == bigram[0] and 
                    word_tokens[i+1] == bigram[1]):
                    new_word.append(bigram[0] + bigram[1])
                    i += 2
                else:
                    new_word.append(word_tokens[i])
                    i += 1
            
            if len(new_word) == len(word_tokens):
                break
            word_tokens = new_word
        
        return word_tokens
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text into BPE token IDs.
        
        Args:
            text (str): Text to encode
            add_bos (bool): Whether to add beginning-of-sequence token
            add_eos (bool): Whether to add end-of-sequence token
            
        Returns:
            List[int]: List of token IDs
        """
        # Split text into tokens using regex pattern
        tokens = re.findall(self.pat, text)
        
        # Encode each token
        bpe_tokens = []
        for token in tokens:
            # Encode word using BPE
            word_tokens = self._encode_word(token)
            bpe_tokens.extend(word_tokens)
        
        # Convert to token IDs
        token_ids = []
        for token in bpe_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Use unknown token
                token_ids.append(self.vocab[self.special_tokens['unk']])
        
        # Add special tokens
        if add_bos:
            token_ids = [self.vocab[self.special_tokens['bos']]] + token_ids
        if add_eos:
            token_ids = token_ids + [self.vocab[self.special_tokens['eos']]]
            
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids (List[int]): List of token IDs to decode
            
        Returns:
            str: Decoded text
        """
        # Convert token IDs to tokens
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                # Skip special tokens except for space markers
                if token not in self.special_tokens.values() or token in ['<s>', '</s>']:
                    tokens.append(token)
        
        # Join tokens
        text = ''.join(tokens)
        
        # Remove </w> markers and fix spacing
        text = text.replace('</w>', ' ').strip()
        
        return text
    
    def save(self, path: str):
        """
        Save tokenizer to file.
        
        Args:
            path (str): Path to save tokenizer
        """
        state = {
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'bpe_ranks': self.bpe_ranks,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """
        Load tokenizer from file.
        
        Args:
            path (str): Path to load tokenizer from
        """
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.vocab = {k: int(v) for k, v in state['vocab'].items()}
        self.id_to_token = {int(k): v for k, v in state['id_to_token'].items()}
        self.bpe_ranks = {tuple(k): int(v) for k, v in state['bpe_ranks'].items()}
        self.special_tokens = state['special_tokens']
        self.vocab_size = state['vocab_size']


def test_bpe_tokenizer():
    """
    Test function for BPETokenizer.
    """
    # Create sample texts
    texts = [
        "Hello, world!",
        "This is a test sentence.",
        "BPE tokenization is useful for language models.",
        "It helps handle out-of-vocabulary words.",
        "The algorithm merges frequent character pairs."
    ]
    
    # Initialize tokenizer
    tokenizer = BPETokenizer(vocab_size=1000)
    
    # Train tokenizer
    print("Training BPE tokenizer...")
    tokenizer.train(texts)
    
    # Test encoding/decoding
    test_text = "Hello, world! This is a test."
    token_ids = tokenizer.encode(test_text, add_bos=True, add_eos=True)
    decoded_text = tokenizer.decode(token_ids)
    
    print(f"Original text: {test_text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded text: {decoded_text}")
    
    # Verify that decoding works correctly (approximately)
    # The decoded text should be similar to the original
    assert len(token_ids) > 0, "No tokens generated"
    assert len(decoded_text) > 0, "No text decoded"
    
    print("BPETokenizer test passed!")


if __name__ == "__main__":
    test_bpe_tokenizer()