"""
Example demonstrating Supervised Fine-Tuning (SFT) response evaluation.
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sft.evaluation import ResponseEvaluator


def main():
    """Main function demonstrating SFT response evaluation."""
    print("Supervised Fine-Tuning (SFT) Response Evaluation Example")
    print("=" * 55)
    
    # Create evaluator
    evaluator = ResponseEvaluator()
    
    # Test examples with varying degrees of similarity
    test_cases = [
        # Exact match
        {
            "prediction": "The capital of France is Paris.",
            "reference": "The capital of France is Paris.",
            "description": "Exact match"
        },
        # Minor variation
        {
            "prediction": "Paris is the capital city of France.",
            "reference": "The capital of France is Paris.",
            "description": "Minor variation"
        },
        # Partial match
        {
            "prediction": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "reference": "Machine learning is a branch of artificial intelligence.",
            "description": "Partial match"
        },
        # Completely different
        {
            "prediction": "The weather is sunny today.",
            "reference": "What is the capital of France?",
            "description": "Completely different"
        },
        # With punctuation and case differences
        {
            "prediction": "Hello, World! How are you?",
            "reference": "hello world how are you",
            "description": "Punctuation and case differences"
        }
    ]
    
    print("1. Individual Response Evaluations:")
    print("-" * 40)
    
    for i, case in enumerate(test_cases):
        print(f"\nExample {i+1}: {case['description']}")
        print(f"  Prediction: {case['prediction']}")
        print(f"  Reference:  {case['reference']}")
        
        # Evaluate response
        metrics = evaluator.evaluate_response(case['prediction'], case['reference'])
        
        print("  Metrics:")
        print(f"    Exact Match:     {metrics['exact_match']:.4f}")
        print(f"    ROUGE-L:         {metrics['rouge_l']:.4f}")
        print(f"    BLEU:            {metrics['bleu']:.4f}")
        print(f"    Word Accuracy:   {metrics['word_accuracy']:.4f}")
    
    # Batch evaluation example
    print("\n2. Batch Evaluation:")
    print("-" * 20)
    
    predictions = [case["prediction"] for case in test_cases]
    references = [case["reference"] for case in test_cases]
    
    batch_metrics = evaluator.evaluate_batch(predictions, references)
    
    print("Average metrics across all examples:")
    print(f"  Exact Match:     {batch_metrics['exact_match']:.4f}")
    print(f"  ROUGE-L:         {batch_metrics['rouge_l']:.4f}")
    print(f"  BLEU:            {batch_metrics['bleu']:.4f}")
    print(f"  Word Accuracy:   {batch_metrics['word_accuracy']:.4f}")
    
    # Test with multiple references
    print("\n3. Evaluation with Multiple References:")
    print("-" * 40)
    
    multi_ref_cases = [
        {
            "prediction": "The cat is sitting on the mat.",
            "references": [
                "The cat sat on the mat.",
                "A cat was sitting on the mat.",
                "There was a cat on the mat."
            ],
            "description": "Multiple valid references"
        },
        {
            "prediction": "Photosynthesis converts light to chemical energy.",
            "references": [
                "Photosynthesis is the process by which plants convert light energy into chemical energy.",
                "Plants use photosynthesis to transform light into chemical energy.",
                "Photosynthesis changes light energy into chemical energy in plants."
            ],
            "description": "Technical concept with variations"
        }
    ]
    
    for i, case in enumerate(multi_ref_cases):
        print(f"\nMulti-reference Example {i+1}: {case['description']}")
        print(f"  Prediction:  {case['prediction']}")
        print(f"  References:  {case['references']}")
        
        # Evaluate with multiple references
        metrics = evaluator.evaluate_response(
            case['prediction'], 
            case['references'][0],  # Primary reference
            case['references']       # All references
        )
        
        print("  Metrics:")
        print(f"    Exact Match:     {metrics['exact_match']:.4f}")
        print(f"    ROUGE-L:         {metrics['rouge_l']:.4f}")
        print(f"    BLEU:            {metrics['bleu']:.4f}")
        print(f"    Word Accuracy:   {metrics['word_accuracy']:.4f}")
    
    # Test edge cases
    print("\n4. Edge Cases:")
    print("-" * 15)
    
    edge_cases = [
        {
            "prediction": "",
            "reference": "Non-empty reference",
            "description": "Empty prediction"
        },
        {
            "prediction": "Non-empty prediction",
            "reference": "",
            "description": "Empty reference"
        },
        {
            "prediction": "",
            "reference": "",
            "description": "Both empty"
        }
    ]
    
    for i, case in enumerate(edge_cases):
        print(f"\nEdge Case {i+1}: {case['description']}")
        print(f"  Prediction: '{case['prediction']}'")
        print(f"  Reference:  '{case['reference']}'")
        
        try:
            metrics = evaluator.evaluate_response(case['prediction'], case['reference'])
            print("  Metrics:")
            print(f"    Exact Match:     {metrics['exact_match']:.4f}")
            print(f"    ROUGE-L:         {metrics['rouge_l']:.4f}")
            print(f"    BLEU:            {metrics['bleu']:.4f}")
            print(f"    Word Accuracy:   {metrics['word_accuracy']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Performance comparison
    print("\n5. Performance Characteristics:")
    print("-" * 30)
    
    import time
    
    # Create longer texts for performance testing
    long_prediction = " ".join(["word"] * 100) + " end"
    long_reference = " ".join(["word"] * 90) + " end"
    
    # Time the evaluation
    start_time = time.time()
    for _ in range(100):
        metrics = evaluator.evaluate_response(long_prediction, long_reference)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"  Average evaluation time: {avg_time*1000:.2f} ms")
    print(f"  Evaluations per second: {1/avg_time:.0f}")
    
    print("\nSFT response evaluation example completed!")


if __name__ == "__main__":
    main()