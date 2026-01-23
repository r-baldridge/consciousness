# Tiny Recursive Model (TRM)

Implementation of the **Tiny Recursive Model** from Samsung SAIT Montreal.

> **"Less is More: Recursive Reasoning with Tiny Networks"**
>
> A 7M parameter model that outperforms billion-parameter LLMs on reasoning tasks
> through recursive refinement rather than depth.

## Paper

- **Title**: Less is More: Recursive Reasoning with Tiny Networks
- **Authors**: Alexia Jolicoeur-Martineau (Samsung SAIT Montreal)
- **arXiv**: [https://arxiv.org/abs/2510.04871](https://arxiv.org/abs/2510.04871)
- **GitHub**: [https://github.com/SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- **License**: MIT

## Key Innovation

**Recursion substitutes for depth.** A single 2-layer network applied recursively achieves
42+ layers of effective depth, enabling tiny models to match or exceed much larger networks
on reasoning tasks.

## Architecture

```
Input (x) -> Initialize y, z
                  |
                  v
          +----------------+
          | Deep Recursion |<----+
          | z <- net(x,y,z)|     |
          | (n times)      |     | T cycles
          | y <- net(y,z)  |     |
          +----------------+-----+
                  |
                  v
          Supervision Step
                  |
                  v
          Early Stop if q_hat > threshold
                  |
                  v
          Repeat up to N_sup times
```

**Effective Depth**: `T × (n+1) × n_layers = 3 × 7 × 2 = 42 layers`

### Dual Semantic States

- **y (solution)**: Embedded current solution, recoverable via output head
- **z (reasoning)**: Latent reasoning feature, analogous to chain-of-thought

### Deep Supervision

Unlike HRM's fixed-point gradient approximation, TRM backpropagates through ALL
recursive steps. This is essential for generalization.

## Benchmarks

| Task | TRM | DeepSeek-R1 | o3-mini | Gemini 2.5 Pro |
|------|-----|-------------|---------|----------------|
| ARC-AGI-1 | **45%** (7M) | - | ~30% | ~25% |
| ARC-AGI-2 | **8%** (7M) | - | - | - |
| Sudoku-Extreme | **87.4%** (5M) | 0.0% | - | - |
| Maze-Hard | **85.3%** (7M) | - | - | - |

## Installation

```bash
# From the ml_research directory
pip install -e .

# Or install dependencies directly
pip install torch pyyaml tqdm
```

## Quick Start

```python
from consciousness.ml_research.modern_dev.trm import TRM, TRMConfig

# Create model for Sudoku
config = TRMConfig.for_sudoku()
model = TRM(config)

print(f"Parameters: {model.num_parameters():,}")
print(f"Effective depth: {config.effective_depth}")

# Training
output = model.train_step(puzzle, solution, max_steps=16)
print(f"Loss: {output['loss']:.4f}, Accuracy: {output['accuracy']:.2%}")

# Inference
result = model.solve(puzzle, max_steps=16)
print(f"Solution found in {result['steps']} steps")
print(f"Confidence: {result['confidence']:.2%}")
```

## Configuration

### Default Hyperparameters

| Parameter | Value | Note |
|-----------|-------|------|
| n_layers | 2 | More layers DECREASE generalization |
| embed_dim | 512 | |
| n_heads | 8 | |
| mlp_ratio | 4 | |
| T_cycles | 3 | High-level cycles |
| n_cycles | 6 | Low-level cycles per T |
| max_supervision_steps | 16 | Training supervision steps |

### Task Presets

```python
# Sudoku (9x9, MLP-only)
config = TRMConfig.for_sudoku()

# Maze solving (30x30, with attention)
config = TRMConfig.for_maze(grid_size=30)

# ARC-AGI tasks (30x30, with attention)
config = TRMConfig.for_arc_agi()
```

## Training

### Command Line

```bash
# Train on Sudoku
python -m trm.cli.train --task sudoku --epochs 100

# Train with custom config
python -m trm.cli.train --config configs/sudoku.yaml

# Specify GPU
python -m trm.cli.train --task sudoku --device cuda:0
```

### Python API

```python
from trm import TRM, TRMConfig
import torch

# Create model
config = TRMConfig.for_sudoku()
model = TRM(config)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    for puzzle, solution in dataloader:
        optimizer.zero_grad()
        output = model.train_step(puzzle, solution)
        output["loss"].backward()
        optimizer.step()
```

## Inference

### Command Line

```bash
# Interactive mode
python -m trm.cli.infer --model checkpoints/best.pt --interactive

# Batch inference
python -m trm.cli.infer --model checkpoints/best.pt --input puzzle.txt
```

### Python API

```python
# Load trained model
model = TRM.from_pretrained("checkpoints/best.pt")
model.eval()

# Solve puzzle
result = model.solve(puzzle, max_steps=16, return_trajectory=True)

print(f"Solution: {result['solution']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Steps: {result['steps']}")

# Examine trajectory
for step in result["trajectory"]:
    print(f"Step {step['step']}: q_hat={step['q_hat']:.3f}")
```

## Mathematical Formulation

### Deep Recursion

For each supervision step:
```
For t in 1..T:
    For i in 1..n:
        z <- net(x, y, z)    # Latent update
    y <- net(y, z)           # Solution update
```

### Loss Function

```
L = CrossEntropy(ŷ, y_true) + BCE(q̂, correct)

where:
    ŷ = OutputHead(y)
    q̂ = QHead(z)
    correct = (ŷ == y_true)
```

### Halting Mechanism

```
if q̂ > 0: early_stop()
else: detach(y, z) and continue
```

## File Structure

```
trm/
├── __init__.py           # Package exports and documentation
├── README.md             # This file
├── src/
│   ├── __init__.py       # Source module exports
│   ├── model.py          # TRM, TRMConfig
│   └── layers.py         # TRMBlock, DeepRecursion, QHead, etc.
├── configs/
│   ├── default.yaml      # Default configuration
│   ├── sudoku.yaml       # Sudoku task config
│   └── arc_agi.yaml      # ARC-AGI task config
├── cli/
│   ├── __init__.py
│   ├── train.py          # Training CLI
│   └── infer.py          # Inference CLI
├── tests/
│   ├── __init__.py
│   └── test_trm.py       # Unit tests
├── docs/                 # Additional documentation
└── models/               # Pretrained model checkpoints
```

## Key Insights from Paper

1. **Depth vs Recursion**: Adding layers hurts generalization. Recursion is the key.

2. **Deep Supervision**: Full backprop through recursion is essential. Fixed-point
   approximations (like HRM) don't work as well.

3. **Simple > Complex**: One tiny network beats two larger networks (as in HRM).

4. **Data Augmentation**: Heavy augmentation (1000x for puzzles) is critical.

5. **MLP for Small Grids**: Attention isn't needed for small grids like 9x9 Sudoku.

## Citation

```bibtex
@article{jolicoeur2025less,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:2510.04871},
  year={2025}
}
```

## License

MIT License - see the original repository for details.
