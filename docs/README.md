# LLM from Scratch - Documentation

## Project Overview

This project implements a Large Language Model from scratch using PyTorch, following a comprehensive 10-part curriculum that builds from basic Transformer architecture to advanced techniques like Mixture-of-Experts and Reinforcement Learning with Human Feedback.

## Curriculum Structure

1. **Part 0** - Foundations & Mindset
2. **Part 1** - Core Transformer Architecture
3. **Part 2** - Training a Tiny LLM
4. **Part 3** - Modernizing the Architecture
5. **Part 4** - Scaling Up
6. **Part 5** - Mixture-of-Experts (MoE)
7. **Part 6** - Supervised Fine-Tuning (SFT)
8. **Part 7** - Reward Modeling
9. **Part 8** - RLHF with PPO
10. **Part 9** - RLHF with GRPO
11. **Part 10** - Advanced Semantic Processing

## Key Features

- Core Transformer architecture implementation
- Modern architectural improvements (RMSNorm, RoPE, SwiGLU)
- Multiple tokenization approaches (byte-level, BPE, BoundlessBPE)
- Training optimization techniques (mixed precision, gradient accumulation)
- Advanced training paradigms (SFT, RLHF)
- Large Concept Models (LCMs) for semantic processing
- Hyperbolic space representations for hierarchical structures
- Efficiency optimizations (quantization, compression)

## Directory Structure

```
.
├── notebooks/          # Jupyter notebooks for interactive experimentation
├── src/                # Source code
│   ├── models/         # Model architectures
│   ├── train/          # Training pipelines
│   ├── tokenizers/     # Tokenization systems
│   ├── utils/          # Utility functions
│   ├── semantic/       # Semantic processing components
│   ├── hyperbolic/     # Hyperbolic space implementations
│   ├── moe/            # Mixture-of-Experts components
│   └── rlhf/           # RLHF implementations
├── tests/              # Test suite
├── docs/               # Documentation (this directory)
├── tasks/              # Task tracking
└── README.md           # Project overview
```

## Getting Started

Instructions for setting up the environment and running the code will be added here as the implementation progresses.

## Contributing

This project follows strict workflow guidelines to ensure code quality. Please read [dev.md](../dev.md) before contributing.