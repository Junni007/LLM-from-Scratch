# Examples Directory

This directory contains example scripts and notebooks demonstrating the usage of the LLM from Scratch implementation.

## Examples

### Tiny LLM Example
- [tiny_llm_example.py](tiny_llm_example.py): A complete example of training a tiny LLM using our implemented components, including both classic and modern architectures
- [notebooks/tiny_llm_example.ipynb](../notebooks/tiny_llm_example.ipynb): Jupyter notebook version of the tiny LLM example

### Scaling Techniques Example
- [scaling_example.py](scaling_example.py): Example demonstrating gradient accumulation and mixed precision training
- [lr_scheduling_example.py](lr_scheduling_example.py): Example demonstrating learning rate scheduling with warmup and decay
- [checkpointing_example.py](checkpointing_example.py): Example demonstrating checkpointing and resuming training
- [logging_example.py](logging_example.py): Example demonstrating logging and visualization of training metrics

### Mixture of Experts Example
- [moe_example.py](moe_example.py): Example demonstrating Mixture of Experts (MoE) theory components
- [moe_layer_example.py](moe_layer_example.py): Example demonstrating Mixture of Experts (MoE) layer implementation
- [hybrid_moe_example.py](hybrid_moe_example.py): Example demonstrating Hybrid Mixture of Experts implementations

## Running Examples

To run the Python examples:

```bash
python examples/tiny_llm_example.py
python examples/scaling_example.py
python examples/lr_scheduling_example.py
python examples/checkpointing_example.py
python examples/logging_example.py
python examples/moe_example.py
python examples/moe_layer_example.py
python examples/hybrid_moe_example.py
```

To run the Jupyter notebooks:

```bash
jupyter notebook notebooks/tiny_llm_example.ipynb
```

## Example Progress

- [x] Tiny LLM example
- [x] Modern architecture example
- [x] Scaling example
- [x] Learning rate scheduling example
- [x] Checkpointing example
- [x] Logging and visualization example
- [x] MoE example
- [x] MoE layer example
- [x] Hybrid MoE example
- [ ] RLHF example