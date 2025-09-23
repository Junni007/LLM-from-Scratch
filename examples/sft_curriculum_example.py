"""
Example demonstrating Supervised Fine-Tuning (SFT) curriculum learning.
"""

import sys
import os
import torch
from torch.utils.data import DataLoader, Dataset

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sft.instruction_formatting import (
    InstructionExample, 
    InstructionFormatter, 
    PromptTemplate, 
    DatasetPreprocessor
)
from sft.curriculum import (
    CurriculumDataset, 
    CurriculumExample, 
    CurriculumStrategy, 
    CurriculumScheduler,
    create_curriculum_examples
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
        tokens = [ord(c) % self.vocab_size for c in text[:100]]  # Limit to 100 tokens
        
        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token_id]
            
        return tokens


class MockInstructionDataset(Dataset):
    """Mock dataset for instruction examples."""
    
    def __init__(self, curriculum_examples, tokenizer, formatter, preprocessor):
        self.curriculum_examples = curriculum_examples
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.curriculum_examples)
    
    def __getitem__(self, idx):
        curriculum_example = self.curriculum_examples[idx]
        example = curriculum_example.example
        
        # Format example
        formatted_text = self.formatter.format_example(example)
        
        # Preprocess text
        processed = self.preprocessor.preprocess_text(formatted_text, self.tokenizer)
        
        return {
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
            "sequence_length": curriculum_example.sequence_length,
            "complexity_score": curriculum_example.complexity_score
        }


def create_sample_examples():
    """Create sample instruction examples with varying complexity."""
    examples = [
        # Simple example
        InstructionExample(
            instruction="What is 2+2?",
            input="",
            output="2+2 equals 4."
        ),
        # Medium example
        InstructionExample(
            instruction="Explain photosynthesis.",
            input="",
            output="Photosynthesis is the process by which plants convert light energy into chemical energy. During photosynthesis, plants use sunlight, carbon dioxide from the air, and water from the soil to produce glucose and oxygen. This process occurs in the chloroplasts of plant cells, specifically in the chlorophyll pigments that give plants their green color."
        ),
        # Complex example
        InstructionExample(
            instruction="Write a Python function to implement a binary search tree with insertion, deletion, and search operations. Include proper error handling and documentation.",
            input="",
            output="""class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
        else:
            self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        if val < node.val:
            if node.left is None:
                node.left = TreeNode(val)
            else:
                self._insert_recursive(node.left, val)
        elif val > node.val:
            if node.right is None:
                node.right = TreeNode(val)
            else:
                self._insert_recursive(node.right, val)
        # If val == node.val, we don't insert duplicates
    
    def search(self, val):
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node, val):
        if node is None or node.val == val:
            return node
        if val < node.val:
            return self._search_recursive(node.left, val)
        return self._search_recursive(node.right, val)
    
    def delete(self, val):
        self.root = self._delete_recursive(self.root, val)
    
    def _delete_recursive(self, node, val):
        if node is None:
            return node
        
        if val < node.val:
            node.left = self._delete_recursive(node.left, val)
        elif val > node.val:
            node.right = self._delete_recursive(node.right, val)
        else:
            # Node to be deleted found
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            
            # Node with two children: Get inorder successor
            successor = self._min_value_node(node.right)
            node.val = successor.val
            node.right = self._delete_recursive(node.right, successor.val)
        
        return node
    
    def _min_value_node(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current"""
        ),
        # Another medium example
        InstructionExample(
            instruction="Translate to French.",
            input="Hello, how are you today?",
            output="Bonjour, comment allez-vous aujourd'hui?"
        ),
        # Simple with input
        InstructionExample(
            instruction="Calculate the area.",
            input="Rectangle with length 5 and width 3",
            output="Area = length × width = 5 × 3 = 15 square units."
        ),
        # Very complex example
        InstructionExample(
            instruction="Design a REST API for a social media platform with user authentication, post creation, commenting, and friend management features. Include database schema design and API endpoint specifications.",
            input="",
            output="""# Social Media Platform REST API Design

## Database Schema

### Users Table
- user_id (Primary Key)
- username (Unique)
- email (Unique)
- password_hash
- created_at
- updated_at

### Posts Table
- post_id (Primary Key)
- user_id (Foreign Key)
- content
- created_at
- updated_at

### Comments Table
- comment_id (Primary Key)
- post_id (Foreign Key)
- user_id (Foreign Key)
- content
- created_at
- updated_at

### Friends Table
- user_id (Foreign Key)
- friend_id (Foreign Key)
- status (pending, accepted, blocked)
- created_at

## API Endpoints

### Authentication
- POST /api/auth/register - User registration
- POST /api/auth/login - User login
- POST /api/auth/logout - User logout

### Users
- GET /api/users/{user_id} - Get user profile
- PUT /api/users/{user_id} - Update user profile
- DELETE /api/users/{user_id} - Delete user account

### Posts
- POST /api/posts - Create new post
- GET /api/posts/{post_id} - Get specific post
- PUT /api/posts/{post_id} - Update post
- DELETE /api/posts/{post_id} - Delete post
- GET /api/users/{user_id}/posts - Get user's posts

### Comments
- POST /api/posts/{post_id}/comments - Add comment to post
- GET /api/posts/{post_id}/comments - Get post comments
- PUT /api/comments/{comment_id} - Update comment
- DELETE /api/comments/{comment_id} - Delete comment

### Friends
- POST /api/users/{user_id}/friends - Send friend request
- GET /api/users/{user_id}/friends - Get friend list
- PUT /api/users/{user_id}/friends/{friend_id} - Accept/reject friend request
- DELETE /api/users/{user_id}/friends/{friend_id} - Remove friend"""
        )
    ]
    
    return examples


def main():
    """Main function demonstrating SFT curriculum learning."""
    print("Supervised Fine-Tuning (SFT) Curriculum Learning Example")
    print("=" * 55)
    
    # Create sample instruction examples
    print("1. Creating sample instruction examples...")
    examples = create_sample_examples()
    print(f"  Created {len(examples)} instruction examples with varying complexity")
    
    # Create curriculum examples with metadata
    print("\n2. Creating curriculum examples with metadata...")
    curriculum_examples = create_curriculum_examples(examples)
    
    # Show example metadata
    for i, curriculum_example in enumerate(curriculum_examples):
        example = curriculum_example.example
        print(f"  Example {i+1}:")
        print(f"    Instruction: {example.instruction[:50]}...")
        print(f"    Sequence Length: {curriculum_example.sequence_length}")
        print(f"    Complexity Score: {curriculum_example.complexity_score:.2f}")
        print(f"    Instruction Length: {curriculum_example.instruction_length}")
        print(f"    Output Length: {curriculum_example.output_length}")
    
    # Test different curriculum strategies
    print("\n3. Testing different curriculum strategies:")
    
    # Create mock tokenizer and components
    tokenizer = MockTokenizer()
    formatter = InstructionFormatter(PromptTemplate.ALPACA)
    preprocessor = DatasetPreprocessor(max_length=256, pad_token_id=tokenizer.pad_token_id)
    
    strategies = [
        CurriculumStrategy.SEQUENCE_LENGTH,
        CurriculumStrategy.COMPLEXITY_SCORE,
        CurriculumStrategy.INSTRUCTION_LENGTH,
        CurriculumStrategy.OUTPUT_LENGTH,
        CurriculumStrategy.RANDOM
    ]
    
    for strategy in strategies:
        print(f"\n  {strategy.value.upper()} Strategy:")
        curriculum_dataset = CurriculumDataset(
            curriculum_examples, 
            strategy=strategy,
            reverse=False
        )
        
        # Show ordering
        print("    Ordering (first 3 examples):")
        for i in range(min(3, len(curriculum_dataset))):
            example = curriculum_dataset[i]
            print(f"      {i+1}. SeqLen={example.sequence_length}, "
                  f"Complexity={example.complexity_score:.1f}")
    
    # Test curriculum progression
    print("\n4. Testing curriculum progression:")
    
    # Create curriculum dataset with scheduler
    scheduler = CurriculumScheduler(strategy="linear")
    curriculum_dataset = CurriculumDataset(
        curriculum_examples,
        strategy=CurriculumStrategy.COMPLEXITY_SCORE,
        scheduler=scheduler
    )
    
    # Simulate training epochs
    total_epochs = 5
    for epoch in range(total_epochs):
        # Update curriculum
        curriculum_dataset.update_curriculum(epoch, total_epochs)
        difficulty = curriculum_dataset.scheduler.get_current_difficulty()
        
        print(f"  Epoch {epoch + 1}/{total_epochs}: "
              f"Difficulty={difficulty:.2f}, "
              f"Examples={len(curriculum_dataset)}")
        
        # Show which examples are being used
        if len(curriculum_dataset) > 0:
            example = curriculum_dataset[0]  # Easiest example
            print(f"    Easiest example: SeqLen={example.sequence_length}, "
                  f"Complexity={example.complexity_score:.1f}")
            
            if len(curriculum_dataset) > 1:
                example = curriculum_dataset[-1]  # Hardest example in current curriculum
                print(f"    Hardest example: SeqLen={example.sequence_length}, "
                      f"Complexity={example.complexity_score:.1f}")
    
    # Test reverse curriculum (starting with hard examples)
    print("\n5. Testing reverse curriculum (starting with hard examples):")
    
    reverse_curriculum = CurriculumDataset(
        curriculum_examples,
        strategy=CurriculumStrategy.COMPLEXITY_SCORE,
        scheduler=CurriculumScheduler(strategy="linear"),
        reverse=True
    )
    
    for epoch in range(3):
        reverse_curriculum.update_curriculum(epoch, 5)
        difficulty = reverse_curriculum.scheduler.get_current_difficulty()
        
        print(f"  Epoch {epoch + 1}/3: "
              f"Difficulty={difficulty:.2f}, "
              f"Examples={len(reverse_curriculum)}")
    
    # Test with DataLoader
    print("\n6. Testing with PyTorch DataLoader:")
    
    # Create mock dataset
    mock_dataset = MockInstructionDataset(
        curriculum_examples, tokenizer, formatter, preprocessor
    )
    
    # Create DataLoader
    data_loader = DataLoader(mock_dataset, batch_size=2, shuffle=False)
    
    print(f"  Created DataLoader with {len(mock_dataset)} examples")
    print(f"  Batch size: 2")
    print(f"  Number of batches: {len(data_loader)}")
    
    # Show first batch
    for batch_idx, batch in enumerate(data_loader):
        print(f"  Batch {batch_idx + 1}:")
        print(f"    Input IDs shape: {batch['input_ids'].shape}")
        print(f"    Attention mask shape: {batch['attention_mask'].shape}")
        print(f"    Sequence lengths: {batch['sequence_length'].tolist()}")
        print(f"    Complexity scores: {[f'{x:.2f}' for x in batch['complexity_score'].tolist()]}")
        break  # Just show first batch
    
    print("\nSFT curriculum learning example completed!")


if __name__ == "__main__":
    main()