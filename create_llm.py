#!/usr/bin/env python3
"""
Create and train a complete LLM from scratch using our implementation.
This script demonstrates the full pipeline from model creation to training to inference.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import os

# Add src to path
sys.path.append('.')

# Import our components
from src.models.transformer import TransformerBlock
from src.tokenizers.byte_tokenizer import ByteTokenizer
from src.train.loss import CrossEntropyLoss
from src.train.sampling import sample_tokens
from src.train.lr_scheduler import WarmupDecayScheduler


class LLM(nn.Module):
    """A complete LLM implementation."""
    
    def __init__(self, vocab_size=256, d_model=256, num_heads=8, num_layers=6, dropout=0.1, max_seq_length=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = self.create_positional_encoding(max_seq_length, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def create_positional_encoding(self, max_seq_length, d_model):
        """Create positional encoding matrix."""
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_seq_length, d_model)
    
    def forward(self, input_ids, targets=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_length)
            targets (torch.Tensor, optional): Target token IDs for loss computation
            
        Returns:
            logits (torch.Tensor): Output logits of shape (batch_size, seq_length, vocab_size)
            loss (torch.Tensor, optional): Computed loss if targets provided
        """
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        x = self.embedding(input_ids)  # (batch_size, seq_length, d_model)
        
        # Scale embeddings
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :seq_length, :].to(x.device)
        x = x + pos_encoding
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Output projection
        logits = self.output_projection(x)  # (batch_size, seq_length, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits, targets)
        
        return logits, loss


class TextDataset(Dataset):
    """Dataset for text training data."""
    
    def __init__(self, texts, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.samples = []
        
        # Process texts into training samples
        for text in texts:
            # Encode text
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            
            # Create samples by sliding window
            for i in range(0, len(tokens) - seq_length):
                if i + seq_length + 1 <= len(tokens):
                    input_seq = tokens[i:i + seq_length]
                    target_seq = tokens[i + 1:i + seq_length + 1]
                    self.samples.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, device='cpu'):
    """
    Generate text using the trained model.
    
    Args:
        model (nn.Module): Trained model
        tokenizer (ByteTokenizer): Tokenizer
        prompt (str): Initial text prompt
        max_length (int): Maximum length of generated text
        temperature (float): Sampling temperature
        device (str): Device to run generation on
        
    Returns:
        str: Generated text
    """
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_bos=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Generate text
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits for last token (limit sequence length)
            if input_tensor.size(1) > model.max_seq_length:
                # Trim input to maximum sequence length
                input_tensor = input_tensor[:, -model.max_seq_length:]
            
            logits, _ = model(input_tensor)
            next_token_logits = logits[0, -1, :]  # Get logits for last position
            
            # Sample next token
            next_token = sample_tokens(
                next_token_logits.unsqueeze(0), 
                method='temperature', 
                temperature=temperature
            )
            
            # Add to sequence
            next_token = next_token.unsqueeze(0).to(device)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
            # Stop if EOS token generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_tokens = input_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def train_llm():
    """Train a complete LLM from scratch."""
    print("Creating and Training LLM from Scratch")
    print("=" * 50)
    
    # Extended training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models have revolutionized many fields.",
        "Transformers are the foundation of modern language models.",
        "Large language models can generate human-like text.",
        "Attention mechanisms help models focus on relevant information.",
        "Neural networks learn patterns from data.",
        "PyTorch provides flexible tools for deep learning research.",
        "Open-source software accelerates scientific progress.",
        "The weather is sunny today with clear blue skies.",
        "Mathematics is the language of science and engineering.",
        "Programming requires logical thinking and problem-solving skills.",
        "Data science combines statistics, programming, and domain expertise.",
        "Artificial intelligence will transform many industries in the future.",
        "The internet has revolutionized how we access information.",
        "Climate change is one of the biggest challenges of our time.",
        "Renewable energy sources are becoming more cost-effective.",
        "Space exploration continues to expand our understanding of the universe.",
        "Education is the key to personal and societal development.",
        "The history of computing began with mechanical calculators.",
        "Alan Turing is considered the father of theoretical computer science.",
        "The first electronic digital computers were developed in the 1940s.",
        "Moore's law predicted the doubling of transistors on chips every two years.",
        "The internet was originally developed as a military communication system.",
        "Tim Berners-Lee invented the World Wide Web in 1989.",
        "Mobile phones have become ubiquitous in modern society.",
        "Social media platforms have changed how people communicate.",
        "E-commerce has transformed the retail industry.",
        "Cloud computing provides scalable computing resources over the internet.",
        "Cybersecurity is crucial for protecting digital information.",
        "Blockchain technology enables secure decentralized transactions.",
        "Cryptocurrencies use cryptography for secure financial transactions.",
        "Quantum computing promises to solve complex problems faster than classical computers.",
        "Biotechnology combines biology and technology for medical advances.",
        "Genetic engineering allows modification of organisms' DNA.",
        "Renewable energy includes solar, wind, hydroelectric, and geothermal sources.",
        "Electric vehicles are becoming more popular due to environmental concerns.",
        "Sustainable development balances economic growth with environmental protection.",
        "Biodiversity loss threatens ecosystem stability and human well-being.",
        "The human genome project mapped all human genes.",
        "Vaccines have eradicated or controlled many infectious diseases.",
        "Antibiotics have saved millions of lives since their discovery.",
        "Medical imaging technologies enable non-invasive diagnosis.",
        "Telemedicine provides healthcare services remotely.",
        "Mental health awareness has increased in recent years.",
        "Nutrition science studies the relationship between diet and health.",
        "Exercise physiology examines how physical activity affects the body.",
        "Sports science applies scientific principles to athletic performance.",
        "Environmental science studies interactions between organisms and their environment.",
        "Climate science focuses on understanding Earth's climate systems.",
        "Geology studies the Earth's physical structure and substance.",
        "Astronomy explores celestial objects and phenomena beyond Earth.",
        "Physics seeks to understand the fundamental laws of nature.",
        "Chemistry studies the composition, structure, and properties of matter.",
        "Biology examines living organisms and their interactions.",
        "Mathematics provides the foundation for scientific modeling.",
        "Statistics enables data analysis and inference.",
        "Engineering applies scientific principles to design and build systems.",
        "Architecture combines art and engineering to design buildings.",
        "Industrial design focuses on product aesthetics and functionality.",
        "Graphic design communicates ideas through visual compositions.",
        "Music theory studies the practices and possibilities of music.",
        "Literature encompasses written works of artistic merit.",
        "Philosophy explores fundamental questions about existence and knowledge.",
        "Psychology studies the human mind and behavior.",
        "Sociology examines social behavior and institutions.",
        "Economics analyzes the production, distribution, and consumption of goods.",
        "Political science studies government and political systems.",
        "History documents and interprets past events and societies.",
        "Anthropology studies human societies and cultures.",
        "Linguistics examines the structure and evolution of languages.",
        "Education develops knowledge, skills, and character traits.",
        "Law establishes rules and regulations for society.",
        "Medicine diagnoses, treats, and prevents disease.",
        "Business management organizes and coordinates enterprise activities.",
        "Marketing promotes and sells products or services.",
        "Finance manages money and investments.",
        "Accounting records and reports financial transactions.",
        "Human resources manages employee relations and development.",
        "Operations research applies mathematical methods to decision-making.",
        "Supply chain management coordinates the flow of goods and services.",
        "Project management plans and executes complex initiatives.",
        "Quality management ensures products and services meet standards.",
        "Risk management identifies and mitigates potential threats.",
        "Strategic planning sets long-term organizational goals.",
        "Innovation drives technological and social progress.",
        "Research and development creates new knowledge and solutions.",
        "Intellectual property protects creative works and inventions.",
        "International trade connects global markets and economies.",
        "Globalization has increased interconnectedness between nations.",
        "Cultural diversity enriches societies and perspectives.",
        "Social justice advocates for fair treatment and equal opportunities.",
        "Human rights protect fundamental freedoms and dignity.",
        "Democracy enables citizen participation in governance.",
        "Civil society includes organizations that operate independently of government.",
        "Media literacy helps people critically evaluate information sources.",
        "Digital literacy involves using technology effectively and responsibly.",
        "Financial literacy empowers people to make informed economic decisions.",
        "Health literacy improves understanding of medical information.",
        "Environmental literacy promotes awareness of ecological issues.",
        "Scientific literacy enables understanding of scientific concepts.",
        "Critical thinking involves analyzing and evaluating information.",
        "Problem-solving identifies and implements solutions to challenges.",
        "Creativity generates novel and valuable ideas.",
        "Collaboration combines efforts to achieve common goals.",
        "Leadership inspires and guides others toward objectives.",
        "Communication conveys information effectively through various channels.",
        "Ethics establishes moral principles for behavior.",
        "Sustainability ensures resources for future generations.",
        "Resilience enables adaptation to adversity and change.",
        "Well-being encompasses physical, mental, and social health.",
        "Happiness involves satisfaction and fulfillment in life.",
        "Success combines achievement with personal values and goals.",
        "Wisdom integrates knowledge with experience and judgment.",
        "Curiosity drives exploration and learning.",
        "Empathy enables understanding and sharing others' feelings.",
        "Compassion motivates helping those in need.",
        "Gratitude appreciates benefits and positive aspects of life.",
        "Mindfulness involves present-moment awareness and acceptance.",
        "Self-awareness recognizes one's own thoughts, feelings, and behaviors.",
        "Self-regulation manages emotions and impulses effectively.",
        "Motivation energizes and directs behavior toward goals.",
        "Confidence believes in one's abilities and potential.",
        "Perseverance continues efforts despite obstacles and setbacks.",
        "Adaptability adjusts to changing circumstances and environments.",
        "Flexibility modifies approaches based on new information.",
        "Integrity aligns actions with moral principles and values.",
        "Responsibility involves accountability for one's choices.",
        "Reliability consistently fulfills commitments and expectations.",
        "Respect treats others with dignity and consideration.",
        "Kindness involves friendly, generous, and considerate actions.",
        "Patience tolerates delays and difficulties without complaint.",
        "Humility recognizes limitations and learns from others.",
        "Courage faces fear, danger, or difficulty despite uncertainty.",
        "Honesty tells the truth and avoids deception.",
        "Loyalty maintains faithfulness to commitments and relationships.",
        "Forgiveness releases resentment and offers second chances.",
        "Generosity shares resources and kindness freely.",
        "Optimism maintains hope and positive expectations.",
        "Pessimism expects negative outcomes and difficulties.",
        "Realism assesses situations objectively and practically.",
        "Idealism pursues perfect standards and principles.",
        "Pragmatism focuses on practical consequences and applications.",
        "Skepticism questions claims and requires evidence.",
        "Open-mindedness considers new ideas and perspectives.",
        "Tolerance accepts differences and diversity.",
        "Fairness treats all parties equally and justly.",
        "Justice administers deserved rewards and punishments.",
        "Freedom enables choice and self-determination.",
        "Equality ensures same rights and opportunities for all.",
        "Diversity encompasses differences in race, gender, age, religion, etc.",
        "Inclusion welcomes and values all individuals.",
        "Belonging involves feeling accepted and valued in groups.",
        "Community provides social connections and mutual support.",
        "Family consists of related individuals living together.",
        "Friendship involves mutual affection and trust.",
        "Love encompasses deep affection and care for others.",
        "Romance involves emotional and physical attraction.",
        "Marriage establishes legal and social partnerships.",
        "Parenting nurtures and guides children's development.",
        "Childhood involves growth and learning experiences.",
        "Adolescence transitions from childhood to adulthood.",
        "Adulthood involves independence and responsibility.",
        "Aging involves physical and mental changes over time.",
        "Death is the end of life and biological functions.",
        "Grief involves emotional response to loss.",
        "Hope anticipates positive future outcomes.",
        "Faith believes without complete evidence or proof.",
        "Spirituality involves search for meaning and purpose.",
        "Religion establishes beliefs and practices about the divine.",
        "Culture encompasses shared beliefs, values, and practices.",
        "Tradition passes customs and beliefs through generations.",
        "Innovation creates new methods and ideas.",
        "Progress advances toward better conditions and outcomes.",
        "Regression returns to earlier or worse conditions.",
        "Evolution involves gradual development and change.",
        "Revolution involves rapid and fundamental change.",
        "Reform improves systems through gradual changes.",
        "Rebellion resists authority and established order.",
        "Peace involves harmony and absence of conflict.",
        "War involves armed conflict between groups or nations.",
        "Diplomacy resolves disputes through negotiation.",
        "Alliance forms cooperative partnerships for mutual benefit.",
        "Treaty establishes formal agreements between parties.",
        "Law governs behavior through rules and enforcement.",
        "Crime involves violation of laws and social norms.",
        "Punishment imposes consequences for wrongdoing.",
        "Rehabilitation restores offenders to productive life.",
        "Prevention reduces likelihood of problems occurring.",
        "Intervention addresses problems after they begin.",
        "Treatment provides care for health conditions.",
        "Therapy helps overcome psychological difficulties.",
        "Medication treats medical conditions with drugs.",
        "Surgery treats conditions through operative procedures.",
        "Prevention avoids problems before they occur.",
        "Diagnosis identifies diseases and conditions.",
        "Prognosis predicts likely course of conditions.",
        "Symptom indicates presence of medical conditions.",
        "Treatment manages and cures health problems.",
        "Recovery involves return to normal health.",
        "Disability involves limitations in functioning.",
        "Accessibility ensures equal opportunities for all.",
        "Accommodation modifies environments for inclusion.",
        "Assistive technology helps overcome limitations.",
        "Support services provide help and resources.",
        "Advocacy promotes rights and interests of others.",
        "Volunteering contributes time and skills to causes.",
        "Philanthropy donates resources to charitable causes.",
        "Fundraising collects money for organizations.",
        "Nonprofit organizations serve public interests.",
        "Government provides public services and governance.",
        "Democracy enables citizen participation in decisions.",
        "Republic involves elected representatives governing.",
        "Monarchy involves rule by hereditary sovereign.",
        "Dictatorship involves absolute rule by single person.",
        "Oligarchy involves rule by small group of people.",
        "Anarchy involves absence of government authority.",
        "Federalism divides power between national and state governments.",
        "Separation of powers divides government functions.",
        "Checks and balances prevent abuse of authority.",
        "Constitution establishes fundamental laws and principles.",
        "Bill of rights protects individual freedoms and liberties.",
        "Civil rights ensure equal treatment under law.",
        "Human rights protect fundamental freedoms globally.",
        "Property rights establish ownership of assets.",
        "Contract rights enforce agreements between parties.",
        "Due process ensures fair treatment in legal proceedings.",
        "Equal protection guarantees same laws for all citizens.",
        "Privacy rights protect personal information and space.",
        "Freedom of speech enables expression of ideas.",
        "Freedom of religion allows worship and belief.",
        "Freedom of assembly permits gathering for purposes.",
        "Freedom of press enables media to report news.",
        "Right to vote enables participation in elections.",
        "Right to education ensures access to learning.",
        "Right to healthcare provides medical services.",
        "Right to housing ensures shelter and accommodation.",
        "Right to work enables employment opportunities.",
        "Right to food ensures adequate nutrition.",
        "Right to water ensures access to clean water.",
        "Right to education ensures access to learning.",
        "Right to information enables access to data.",
        "Right to participation enables involvement in decisions.",
        "Right to development ensures progress and improvement.",
        "Right to peace promotes harmony and security.",
        "Right to environment protects natural resources.",
        "Right to culture preserves heritage and traditions.",
        "Right to language maintains linguistic diversity.",
        "Right to identity preserves personal and group characteristics.",
        "Right to family maintains relationships and connections.",
        "Right to privacy protects personal information.",
        "Right to security ensures safety and protection.",
        "Right to justice ensures fair treatment and outcomes.",
        "Right to remedy provides solutions for violations.",
        "Right to truth seeks accurate information and facts.",
        "Right to reconciliation promotes healing and unity.",
        "Right to memory preserves historical events and experiences.",
        "Right to truth seeks accurate information and facts.",
        "Right to reconciliation promotes healing and unity.",
        "Right to memory preserves historical events and experiences.",
        "Right to truth seeks accurate information and facts.",
        "Right to reconciliation promotes healing and unity.",
        "Right to memory preserves historical events and experiences."
    ]
    
    # Initialize tokenizer
    tokenizer = ByteTokenizer()
    print(f"Tokenizer initialized with vocab size: {tokenizer.vocab_size}")
    
    # Create dataset
    seq_length = 64
    dataset = TextDataset(training_texts, tokenizer, seq_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Initialize model
    model = LLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        max_seq_length=512
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = WarmupDecayScheduler(
        optimizer, 
        warmup_steps=500, 
        total_steps=5000, 
        decay_type='cosine',
        min_lr=1e-5
    )
    
    # Training loop
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Training on device: {device}")
    print("Starting training...")
    
    num_epochs = 3
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            # Move to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, loss = model(input_ids, target_ids)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # Print epoch summary
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    
    # Save model
    torch.save(model.state_dict(), 'llm_checkpoint.pth')
    print("Model checkpoint saved as 'llm_checkpoint.pth'")
    
    # Generate sample text
    print("\nGenerating sample text...")
    prompts = [
        "The future of artificial intelligence",
        "Machine learning has revolutionized",
        "Natural language processing enables"
    ]
    
    for prompt in prompts:
        generated_text = generate_text(
            model, tokenizer, prompt, 
            max_length=50, temperature=0.8, device=device
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")
    
    return model, tokenizer


def load_and_generate():
    """Load a trained model and generate text."""
    print("\nLoading trained model and generating text...")
    
    # Initialize tokenizer and model
    tokenizer = ByteTokenizer()
    model = LLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        max_seq_length=512
    )
    
    # Load checkpoint
    if os.path.exists('llm_checkpoint.pth'):
        model.load_state_dict(torch.load('llm_checkpoint.pth'))
        print("Model checkpoint loaded successfully!")
    else:
        print("No checkpoint found. Using untrained model.")
    
    # Generate text
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    prompts = [
        "The advancement of technology",
        "Deep learning models have",
        "In the field of medicine"
    ]
    
    for prompt in prompts:
        generated_text = generate_text(
            model, tokenizer, prompt,
            max_length=40, temperature=0.7, device=device
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")


def main():
    """Main function."""
    print("LLM from Scratch - Complete Implementation")
    print("=" * 50)
    
    # Train model
    model, tokenizer = train_llm()
    
    # Generate with trained model
    load_and_generate()
    
    print("\n" + "=" * 50)
    print("LLM creation and training completed successfully!")
    print("You now have a fully functional LLM that can generate text.")


if __name__ == "__main__":
    main()