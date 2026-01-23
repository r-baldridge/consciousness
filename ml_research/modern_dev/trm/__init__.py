"""
Tiny Recursive Model (TRM) - Samsung SAIT Montreal, 2025

A simpler recursive reasoning approach that achieves significantly higher
generalization than HRM (Hierarchical Reasoning Model), using a single tiny
network with only 2 layers and 7M parameters.

Paper: "Less is More: Recursive Reasoning with Tiny Networks"
arXiv: https://arxiv.org/abs/2510.04871
GitHub: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
Author: Alexia Jolicoeur-Martineau (Samsung SAIT Montreal)
License: MIT

=============================================================================
KEY INNOVATIONS
=============================================================================

1. RECURSION SUBSTITUTES FOR DEPTH
   - Single 2-layer network recursively refines predictions
   - Effectively simulates 42+ layer depth via iteration
   - 7M parameters outperforms models 10,000x larger on reasoning

2. SIMPLIFIED ARCHITECTURE
   - One network instead of HRM's two
   - No hierarchical structure or fixed-point theorems
   - Simple halting mechanism (single forward pass)

3. DEEP SUPERVISION WITH FULL BACKPROP
   - Backpropagates through ALL recursive steps
   - Unlike HRM's fixed-point gradient approximation
   - Essential for generalization

4. DUAL SEMANTIC STATES
   - y: Embedded current solution (recoverable via output head)
   - z: Latent reasoning feature (chain-of-thought analog)

=============================================================================
ARCHITECTURE
=============================================================================

Standard Deep Network:
    Input -> [Layer 1] -> [Layer 2] -> ... -> [Layer N] -> Output
    (Fixed depth, single forward pass)

Tiny Recursive Model:
    Input (x) -> Initialize y, z
                      |
                      v
              +----------------+
              | Deep Recursion |<----+
              | z <- net(x,y,z)|     |
              | (n times)      |     | T cycles
              | y <- net(y,z)  |     | (without grad for T-1)
              +----------------+-----+
                      |
                      v
              Supervision Step (loss + backward)
                      |
                      v
              Early Stop if q_hat > threshold
                      |
                      v
              Repeat up to N_sup times

Effective Depth: T × (n+1) × n_layers = 3 × 7 × 2 = 42 layers

=============================================================================
BENCHMARKS
=============================================================================

ARC-AGI-1: 45% (7M params vs LLMs with billions)
ARC-AGI-2: 8%
Sudoku-Extreme: 87.4% (5M params)
Maze-Hard: 85.3% (7M params)

Comparison:
    - DeepSeek-R1 on Sudoku-Extreme: 0.0%
    - o3-mini on ARC-AGI-1: ~30%
    - Gemini 2.5 Pro: ~25%

=============================================================================
HYPERPARAMETERS (Optimal from paper)
=============================================================================

Model:
    - n_layers: 2 (more layers DECREASE generalization)
    - embed_dim: 512
    - n_heads: 8
    - mlp_ratio: 4

Recursion:
    - T (H_cycles): 3
    - n (L_cycles): 6
    - N_sup (max supervision steps): 16

Training:
    - Learning rate: 1e-4 with warmup
    - EMA decay: 0.999
    - Heavy augmentation (1000x for puzzles)

=============================================================================
USAGE
=============================================================================

from consciousness.ml_research.modern_dev.trm import TRM, TRMConfig

# Create model
config = TRMConfig(
    grid_size=9,      # For Sudoku
    vocab_size=10,    # 0-9 digits
    embed_dim=512,
    n_layers=2,
    T_cycles=3,
    n_cycles=6,
)
model = TRM(config)

# Train on puzzle
loss = model.train_step(x_puzzle, y_solution)

# Inference with recursive refinement
solution, confidence = model.solve(x_puzzle, max_steps=16)
"""

__version__ = "0.1.0"
__status__ = "implementation"

# Architecture metadata
ARCHITECTURE_INFO = {
    "name": "Tiny Recursive Model",
    "abbreviation": "TRM",
    "year": 2025,
    "organization": "Samsung SAIT Montreal",
    "paper_url": "https://arxiv.org/abs/2510.04871",
    "github_url": "https://github.com/SamsungSAILMontreal/TinyRecursiveModels",
    "authors": ["Alexia Jolicoeur-Martineau"],
    "license": "MIT",
    "key_innovation": "Recursion substitutes for depth - 7M params beats billion-param LLMs",
}

# Mathematical formulation
FORMULATION = """
Deep Recursion (per supervision step):
    For t in 1..T:
        For i in 1..n:
            z <- net(x, y, z)    # Latent update
        y <- net(y, z)           # Solution update

    Final cycle backpropagates through all n+1 evaluations.

Loss Function:
    L = CrossEntropy(y_hat, y_true) + BCE(q_hat, correct)

    where:
        y_hat = OutputHead(y)
        q_hat = QHead(z)  # Halting probability
        correct = (y_hat == y_true)

Halting:
    if q_hat > 0: early_stop()
    else: detach(y, z) and continue

Effective Depth:
    T × (n + 1) × n_layers = 3 × 7 × 2 = 42 layers equivalent
"""

# Default configuration
DEFAULT_CONFIG = {
    # Model architecture
    "embed_dim": 512,
    "n_layers": 2,
    "n_heads": 8,
    "mlp_ratio": 4,
    "dropout": 0.0,

    # Recursion parameters
    "T_cycles": 3,           # High-level cycles
    "n_cycles": 6,           # Low-level cycles per T
    "max_supervision_steps": 16,

    # Task-specific (can be overridden)
    "grid_size": 9,          # For Sudoku
    "vocab_size": 10,        # Output vocabulary
    "max_seq_len": 81,       # 9x9 grid

    # Training
    "learning_rate": 1e-4,
    "warmup_steps": 1000,
    "ema_decay": 0.999,
    "use_stable_max": True,

    # Architecture variants
    "use_attention": True,   # False for MLP-only variant
    "use_rotary": True,
    "use_swiglu": True,
}

# Task-specific presets
TASK_PRESETS = {
    "sudoku": {
        "grid_size": 9,
        "vocab_size": 10,
        "max_seq_len": 81,
        "T_cycles": 3,
        "n_cycles": 6,
        "use_attention": False,  # MLP better for small grids
    },
    "maze": {
        "grid_size": 30,
        "vocab_size": 4,  # empty, wall, start, end, path
        "max_seq_len": 900,
        "T_cycles": 3,
        "n_cycles": 4,
        "use_attention": True,
    },
    "arc_agi": {
        "grid_size": 30,
        "vocab_size": 11,  # 0-9 colors + background
        "max_seq_len": 900,
        "T_cycles": 3,
        "n_cycles": 6,
        "use_attention": True,
    },
}

# Core model imports
from .src.model import TRM, TRMConfig
from .src.layers import (
    TRMBlock,
    DeepRecursion,
    QHead,
    OutputHead,
    GridEmbedding,
    MLPSequence,
)

# CLI imports (optional - may fail if dependencies missing)
try:
    from .cli.train import main as train
    from .cli.infer import main as infer
    from .cli.agent import TRMAgent, create_agent
except ImportError:
    train = None
    infer = None
    TRMAgent = None
    create_agent = None

__all__ = [
    # Core
    "TRM",
    "TRMConfig",
    # Layers
    "TRMBlock",
    "DeepRecursion",
    "QHead",
    "OutputHead",
    "GridEmbedding",
    "MLPSequence",
    # CLI
    "train",
    "infer",
    "TRMAgent",
    "create_agent",
    # Metadata
    "ARCHITECTURE_INFO",
    "FORMULATION",
    "DEFAULT_CONFIG",
    "TASK_PRESETS",
]
