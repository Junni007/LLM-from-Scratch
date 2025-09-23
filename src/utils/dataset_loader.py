"""
Dataset loader utility for handling different types of datasets.
"""

import os
import json
import csv
import glob
from pathlib import Path
from typing import List, Dict, Any

def load_text_files(data_dir: str) -> List[str]:
    """
    Load all text files from a directory.
    
    Args:
        data_dir (str): Path to the directory containing text files
        
    Returns:
        List[str]: List of text contents from all files
    """
    texts = []
    for file_path in Path(data_dir).glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    texts.append(content)
        except Exception as e:
            print(f"Warning: Could not read {file_path.name}: {e}")
    return texts

def load_json_files(data_dir: str) -> List[str]:
    """
    Load text data from JSON files.
    Assumes JSON files contain a list of text strings or objects with a 'text' field.
    
    Args:
        data_dir (str): Path to the directory containing JSON files
        
    Returns:
        List[str]: List of text contents from all files
    """
    texts = []
    for file_path in Path(data_dir).glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # If it's a list of strings
                    for item in data:
                        if isinstance(item, str):
                            texts.append(item)
                        elif isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'])
                elif isinstance(data, dict):
                    # If it's a single object with text field
                    if 'text' in data:
                        texts.append(data['text'])
                    # If it's a dict with multiple text fields
                    for key, value in data.items():
                        if isinstance(value, str) and len(value) > 50:  # Assume long strings are text
                            texts.append(value)
        except Exception as e:
            print(f"Warning: Could not read {file_path.name}: {e}")
    return texts

def load_csv_files(data_dir: str) -> List[str]:
    """
    Load text data from CSV files.
    Assumes CSV files have a 'text' column or the first column contains text.
    
    Args:
        data_dir (str): Path to the directory containing CSV files
        
    Returns:
        List[str]: List of text contents from all files
    """
    texts = []
    for file_path in Path(data_dir).glob("*.csv"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Read header if exists
                
                # Determine text column
                text_column = 0
                if header:
                    # Look for a column named 'text' or similar
                    for i, col_name in enumerate(header):
                        if col_name.lower() in ['text', 'content', 'sentence', 'paragraph']:
                            text_column = i
                            break
                
                # Read all rows
                for row in reader:
                    if len(row) > text_column:
                        text = row[text_column]
                        if text.strip():
                            texts.append(text)
        except Exception as e:
            print(f"Warning: Could not read {file_path.name}: {e}")
    return texts

def load_custom_files(data_dir: str) -> List[str]:
    """
    Load text data from custom format files.
    This is a placeholder for custom parsing logic.
    
    Args:
        data_dir (str): Path to the directory containing custom files
        
    Returns:
        List[str]: List of text contents from all files
    """
    texts = []
    # Add custom parsing logic here based on your specific format
    # For now, we'll just load text files in the custom directory
    for file_path in Path(data_dir).glob("*"):
        if file_path.is_file() and file_path.suffix not in ['.json', '.csv']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        texts.append(content)
            except Exception as e:
                print(f"Warning: Could not read {file_path.name}: {e}")
    return texts

def load_all_datasets(datasets_dir: str = "./datasets") -> List[str]:
    """
    Load all datasets from the datasets directory.
    
    Args:
        datasets_dir (str): Path to the datasets directory
        
    Returns:
        List[str]: List of text contents from all dataset files
    """
    print(f"Loading datasets from {datasets_dir}...")
    
    if not os.path.exists(datasets_dir):
        print(f"Warning: {datasets_dir} directory not found.")
        return []
    
    all_texts = []
    
    # Load text files
    text_dir = os.path.join(datasets_dir, "text")
    if os.path.exists(text_dir):
        texts = load_text_files(text_dir)
        all_texts.extend(texts)
        print(f"  Loaded {len(texts)} text files from {text_dir}")
    
    # Load JSON files
    json_dir = os.path.join(datasets_dir, "json")
    if os.path.exists(json_dir):
        texts = load_json_files(json_dir)
        all_texts.extend(texts)
        print(f"  Loaded {len(texts)} JSON files from {json_dir}")
    
    # Load CSV files
    csv_dir = os.path.join(datasets_dir, "csv")
    if os.path.exists(csv_dir):
        texts = load_csv_files(csv_dir)
        all_texts.extend(texts)
        print(f"  Loaded {len(texts)} CSV files from {csv_dir}")
    
    # Load custom files
    custom_dir = os.path.join(datasets_dir, "custom")
    if os.path.exists(custom_dir):
        texts = load_custom_files(custom_dir)
        all_texts.extend(texts)
        print(f"  Loaded {len(texts)} custom files from {custom_dir}")
    
    # Load any files directly in the datasets directory
    direct_texts = load_text_files(datasets_dir)
    all_texts.extend(direct_texts)
    if direct_texts:
        print(f"  Loaded {len(direct_texts)} files directly from {datasets_dir}")
    
    print(f"Total: Loaded {len(all_texts)} text samples from all datasets")
    return all_texts

def create_sample_datasets():
    """
    Create sample dataset files for demonstration.
    """
    # Create sample text file
    text_content = """The quick brown fox jumps over the lazy dog. This sentence contains all letters of the English alphabet.
Machine learning is a method of data analysis that automates analytical model building.
Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.
Deep learning is part of a broader family of machine learning methods based on artificial neural networks."""
    
    with open("./datasets/text/sample.txt", "w", encoding="utf-8") as f:
        f.write(text_content)
    
    # Create sample JSON file
    json_content = [
        "Transformers are deep learning models that adopt the mechanism of attention.",
        "Large language models are trained on massive datasets and can generate human-like text.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Neural networks consist of layers of interconnected nodes that process information."
    ]
    
    with open("./datasets/json/sample.json", "w", encoding="utf-8") as f:
        json.dump(json_content, f, indent=2)
    
    # Create sample CSV file
    csv_content = [
        ["text"],
        ["PyTorch is an open-source machine learning library for Python."],
        ["TensorFlow is an open-source platform for machine learning."],
        ["The weather is pleasant today with clear skies."],
        ["Mathematics is the study of quantity, structure, space, and change."]
    ]
    
    with open("./datasets/csv/sample.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_content)
    
    print("Sample datasets created in datasets/text, datasets/json, and datasets/csv")

if __name__ == "__main__":
    # Create sample datasets for demonstration
    create_sample_datasets()
    
    # Load and display datasets
    texts = load_all_datasets()
    print(f"\nFirst 3 text samples:")
    for i, text in enumerate(texts[:3]):
        print(f"{i+1}. {text[:100]}...")