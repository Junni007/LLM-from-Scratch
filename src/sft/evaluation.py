"""
Evaluation metrics for Supervised Fine-Tuning (SFT) instruction datasets.

This module implements evaluation metrics for comparing model-generated responses
against gold standard responses in instruction tuning scenarios.
"""

import torch
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import math


class ResponseEvaluator:
    """
    Evaluator for instruction-following model responses.
    """
    
    def __init__(self):
        """Initialize ResponseEvaluator."""
        pass
    
    def exact_match(self, prediction: str, reference: str) -> bool:
        """
        Check if prediction exactly matches reference (after normalization).
        
        Args:
            prediction (str): Model-generated response
            reference (str): Gold standard response
            
        Returns:
            bool: True if responses match exactly
        """
        return self._normalize_text(prediction) == self._normalize_text(reference)
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def rouge_l(self, prediction: str, reference: str) -> float:
        """
        Compute ROUGE-L (Longest Common Subsequence) score.
        
        Args:
            prediction (str): Model-generated response
            reference (str): Gold standard response
            
        Returns:
            float: ROUGE-L score (0.0 to 1.0)
        """
        pred_tokens = self._normalize_text(prediction).split()
        ref_tokens = self._normalize_text(reference).split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Compute LCS length
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        # Compute precision and recall
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
        
        # Compute F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Compute length of longest common subsequence.
        
        Args:
            seq1 (List[str]): First sequence
            seq2 (List[str]): Second sequence
            
        Returns:
            int: Length of LCS
        """
        m, n = len(seq1), len(seq2)
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def bleu_score(self, prediction: str, references: List[str], max_n: int = 4) -> float:
        """
        Compute BLEU score (simplified version).
        
        Args:
            prediction (str): Model-generated response
            references (List[str]): List of gold standard responses
            max_n (int): Maximum n-gram order
            
        Returns:
            float: BLEU score (0.0 to 1.0)
        """
        pred_tokens = self._normalize_text(prediction).split()
        
        if not pred_tokens:
            return 0.0
        
        # Compute n-gram precision scores
        precisions = []
        for n in range(1, min(max_n, len(pred_tokens)) + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            pred_count = sum(pred_ngrams.values())
            
            if pred_count == 0:
                precisions.append(0.0)
                continue
            
            # Find maximum count in any reference
            max_clip_count = 0
            for ref in references:
                ref_tokens = self._normalize_text(ref).split()
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                
                # Clip prediction n-grams by reference counts
                clip_count = 0
                for ngram, count in pred_ngrams.items():
                    clip_count += min(count, ref_ngrams.get(ngram, 0))
                
                max_clip_count = max(max_clip_count, clip_count)
            
            precision = max_clip_count / pred_count if pred_count > 0 else 0.0
            precisions.append(precision)
        
        # Compute geometric mean of precisions
        if not precisions or any(p == 0 for p in precisions):
            geometric_mean = 0.0
        else:
            log_sum = sum(math.log(p) for p in precisions)
            geometric_mean = math.exp(log_sum / len(precisions))
        
        # Compute brevity penalty
        ref_lengths = [len(self._normalize_text(ref).split()) for ref in references]
        pred_length = len(pred_tokens)
        
        if not ref_lengths:
            brevity_penalty = 1.0
        else:
            # Find closest reference length
            closest_ref_len = min(ref_lengths, key=lambda x: abs(x - pred_length))
            
            if pred_length > closest_ref_len:
                brevity_penalty = 1.0
            else:
                brevity_penalty = math.exp(1 - closest_ref_len / pred_length) if pred_length > 0 else 0.0
        
        return brevity_penalty * geometric_mean
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """
        Get n-grams from tokens.
        
        Args:
            tokens (List[str]): List of tokens
            n (int): N-gram order
            
        Returns:
            Dict[Tuple[str, ...], int]: N-gram counts
        """
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def word_accuracy(self, prediction: str, reference: str) -> float:
        """
        Compute word-level accuracy.
        
        Args:
            prediction (str): Model-generated response
            reference (str): Gold standard response
            
        Returns:
            float: Word accuracy (0.0 to 1.0)
        """
        pred_tokens = self._normalize_text(prediction).split()
        ref_tokens = self._normalize_text(reference).split()
        
        if not ref_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        # Count correct words
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        
        correct_count = 0
        for word, count in ref_counter.items():
            correct_count += min(count, pred_counter.get(word, 0))
        
        return correct_count / len(ref_tokens)
    
    def evaluate_response(self, prediction: str, reference: str, 
                         references: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate a single response using multiple metrics.
        
        Args:
            prediction (str): Model-generated response
            reference (str): Gold standard response
            references (List[str], optional): Additional reference responses
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        if references is None:
            references = [reference]
        
        exact_match_score = float(self.exact_match(prediction, reference))
        rouge_l_score = self.rouge_l(prediction, reference)
        bleu_score_value = self.bleu_score(prediction, references)
        word_acc = self.word_accuracy(prediction, reference)
        
        return {
            'exact_match': exact_match_score,
            'rouge_l': rouge_l_score,
            'bleu': bleu_score_value,
            'word_accuracy': word_acc
        }
    
    def evaluate_batch(self, predictions: List[str], references: List[str],
                      additional_references: Optional[List[List[str]]] = None) -> Dict[str, float]:
        """
        Evaluate a batch of responses.
        
        Args:
            predictions (List[str]): Model-generated responses
            references (List[str]): Gold standard responses
            additional_references (List[List[str]], optional): Additional reference responses for each example
            
        Returns:
            Dict[str, float]: Average metrics across batch
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        if additional_references is not None and len(additional_references) != len(predictions):
            raise ValueError("Additional references must match predictions length")
        
        # Evaluate each example
        batch_metrics = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            refs = [ref]
            if additional_references is not None:
                refs.extend(additional_references[i])
            
            metrics = self.evaluate_response(pred, ref, refs)
            batch_metrics.append(metrics)
        
        # Compute averages
        avg_metrics = {}
        if batch_metrics:
            for key in batch_metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in batch_metrics) / len(batch_metrics)
        
        return avg_metrics


def demonstrate_evaluation():
    """Demonstrate response evaluation metrics."""
    print("SFT Response Evaluation Metrics")
    print("=" * 35)
    
    # Create evaluator
    evaluator = ResponseEvaluator()
    
    # Test examples
    examples = [
        {
            "prediction": "The capital of France is Paris.",
            "reference": "The capital of France is Paris."
        },
        {
            "prediction": "Paris is the capital city of France in Europe.",
            "reference": "The capital of France is Paris."
        },
        {
            "prediction": "Machine learning is a subset of artificial intelligence.",
            "reference": "Machine learning is a branch of artificial intelligence."
        },
        {
            "prediction": "I don't know the answer to that question.",
            "reference": "The answer is 42."
        }
    ]
    
    print("1. Individual Example Evaluations:")
    for i, example in enumerate(examples):
        print(f"\n  Example {i+1}:")
        print(f"    Prediction: {example['prediction']}")
        print(f"    Reference:  {example['reference']}")
        
        metrics = evaluator.evaluate_response(example['prediction'], example['reference'])
        for metric, score in metrics.items():
            print(f"    {metric}: {score:.4f}")
    
    # Batch evaluation
    print("\n2. Batch Evaluation:")
    predictions = [ex["prediction"] for ex in examples]
    references = [ex["reference"] for ex in examples]
    
    batch_metrics = evaluator.evaluate_batch(predictions, references)
    print("  Average metrics across batch:")
    for metric, score in batch_metrics.items():
        print(f"    {metric}: {score:.4f}")
    
    # Test BLEU with multiple references
    print("\n3. BLEU with Multiple References:")
    multi_ref_example = {
        "prediction": "The cat sat on the mat.",
        "references": [
            "The cat sat on the mat.",
            "A cat was sitting on the mat.",
            "There was a cat on the mat."
        ]
    }
    
    bleu_score = evaluator.bleu_score(
        multi_ref_example["prediction"], 
        multi_ref_example["references"]
    )
    print(f"  Prediction: {multi_ref_example['prediction']}")
    print(f"  References: {multi_ref_example['references']}")
    print(f"  BLEU score: {bleu_score:.4f}")
    
    print("\nResponse evaluation demonstration completed!")


if __name__ == "__main__":
    demonstrate_evaluation()