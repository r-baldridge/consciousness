# TRM Implementation Guide

## Overview

This document describes the implementation of the **Tiny Recursive Model (TRM)** from Samsung SAIT Montreal and how it can be applied to CLI agent applications.

> **Paper**: "Less is More: Recursive Reasoning with Tiny Networks"
> **arXiv**: https://arxiv.org/abs/2510.04871
> **Author**: Alexia Jolicoeur-Martineau

---

## What Was Implemented

### 1. Core Architecture (`src/model.py`, `src/layers.py`)

**TRM Model Class**
- Full implementation of the Tiny Recursive Model
- Configurable via `TRMConfig` dataclass
- Task presets for Sudoku, Maze, and ARC-AGI
- Training with deep supervision through all recursive steps
- Inference with trajectory tracking and early stopping

**Layer Components**
| Component | Purpose |
|-----------|---------|
| `TRMBlock` | Transformer-style block (pre-norm, attention optional, SwiGLU FFN) |
| `DeepRecursion` | Core T×(n+1) recursive refinement loop |
| `QHead` | Halting probability prediction (determines when to stop) |
| `OutputHead` | Solution prediction with stable-max |
| `GridEmbedding` | 2D positional encoding for grid-based tasks |
| `MLPSequence` | MLP-only variant for small grids |
| `RMSNorm` | Root Mean Square Layer Normalization |
| `SwiGLU` | Gated activation function |
| `RotaryEmbedding` | Rotary Position Embedding (RoPE) |
| `MultiHeadAttention` | Multi-head self-attention with optional RoPE |

### 2. Configuration System (`configs/`)

**Default Configuration** (`default.yaml`)
- Model architecture parameters (embed_dim, n_layers, etc.)
- Recursion parameters (T_cycles, n_cycles, max_supervision_steps)
- Training hyperparameters (lr, warmup, EMA decay)
- Inference settings (q_threshold)

**Task Presets**
- `sudoku.yaml`: 9×9 Sudoku (MLP-only, 87.4% on Sudoku-Extreme)
- `arc_agi.yaml`: ARC-AGI tasks (30×30 grid, attention enabled)

### 3. Command-Line Interface (`cli/`)

**Training CLI** (`train.py`)
```bash
# Train on Sudoku
python -m trm.cli.train --task sudoku --epochs 100

# Train with custom config
python -m trm.cli.train --config configs/sudoku.yaml --batch-size 64

# Specify device and preset
python -m trm.cli.train --task arc_agi --device cuda:0 --lr 1e-4
```

Features:
- Exponential Moving Average (EMA) for stable training
- Learning rate warmup and scheduling
- Gradient clipping
- Checkpointing best model

**Inference CLI** (`infer.py`)
```bash
# Interactive puzzle solving
python -m trm.cli.infer --model checkpoints/best.pt --interactive

# Batch inference
python -m trm.cli.infer --model checkpoints/best.pt --input puzzle.txt --output result.json
```

Features:
- Interactive mode for manual puzzle entry
- Batch processing from files
- Trajectory visualization
- JSON output for integration

### 4. Test Suite (`tests/test_trm.py`)

Comprehensive tests for:
- Configuration presets and effective depth calculation
- All layer components (attention, FFN, embeddings)
- DeepRecursion forward pass
- TRM model forward, training, and inference
- Gradient flow through recursion
- Model save/load functionality

### 5. Integration with ML Research Module

**Orchestrator Integration**
- Added to `ARCHITECTURE_CAPABILITIES` (reasoning, planning tasks)
- Registered in `ARCHITECTURE_MODULE_PATHS`
- Model class registered in `ARCHITECTURE_CLASS_NAMES`
- Config class registered in `ARCHITECTURE_CONFIG_NAMES`

**ML Techniques Registry**
- Added `tiny_recursive_model` technique for recursive reasoning
- Documented composability with other techniques

---

## How to Use TRM

### Basic Usage

```python
from consciousness.ml_research.modern_dev.trm import TRM, TRMConfig

# Create model with preset
config = TRMConfig.for_sudoku()
model = TRM(config)

print(f"Parameters: {model.num_parameters():,}")  # ~5-7M
print(f"Effective depth: {config.effective_depth}")  # 42 layers

# Training step
output = model.train_step(puzzle_batch, solution_batch)
print(f"Loss: {output['loss']:.4f}")
print(f"Accuracy: {output['accuracy']:.2%}")
print(f"Steps used: {output['steps']}")

# Inference
result = model.solve(puzzle_batch, return_trajectory=True)
print(f"Solution: {result['solution']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Converged in {result['steps']} steps")
```

### Via Orchestrator

```python
from consciousness.ml_research.modern_dev.orchestrator import Orchestrator, run_task

# Automatic architecture selection for reasoning
result = run_task(
    task_type="reasoning",
    input_data={"puzzle": puzzle_grid},
    architecture="trm",
    preset="small"
)

# Or use orchestrator directly
orch = Orchestrator()
result = orch.run(
    task_type="reasoning",
    input_data={"input_tensor": puzzle_tensor},
    architecture="trm"
)
```

---

## CLI Agent Applications

TRM's recursive reasoning capability makes it ideal for CLI agent applications where:
- Problems require iterative refinement
- Solutions can be verified/scored
- The agent should "think" before answering

### Potential CLI Agent Use Cases

#### 1. **Puzzle Solver Agent**
```bash
# A CLI tool that solves various puzzles
puzzle-solver solve --type sudoku --input puzzle.txt
puzzle-solver solve --type maze --input maze.png
puzzle-solver solve --type arc --input task.json
```

#### 2. **Code Reasoning Agent**
TRM's recursive refinement mirrors how developers debug:
- Initial attempt → test → refine → test → ...
- Could be adapted for code repair tasks

#### 3. **Planning Agent**
The dual-state architecture (y=plan, z=reasoning) fits planning:
- y: Current plan state
- z: Reasoning about plan validity
- Recursively refine until plan is valid

### Implementing a CLI Agent with TRM

Here's a blueprint for a TRM-powered CLI agent:

```python
# cli_agent/agent.py
"""
TRM-Powered CLI Agent for Reasoning Tasks
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from consciousness.ml_research.modern_dev.trm import TRM, TRMConfig


class TRMAgent:
    """CLI Agent powered by Tiny Recursive Model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        task: str = "reasoning",
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.device = torch.device(device)
        self.verbose = verbose
        self.task = task

        # Load or create model
        if model_path and Path(model_path).exists():
            self.model = TRM.from_pretrained(model_path).to(self.device)
        else:
            config = self._get_config_for_task(task)
            self.model = TRM(config).to(self.device)

        self.model.eval()

    def _get_config_for_task(self, task: str) -> TRMConfig:
        """Get appropriate config for task type."""
        if task == "sudoku":
            return TRMConfig.for_sudoku()
        elif task == "maze":
            return TRMConfig.for_maze()
        elif task == "arc" or task == "reasoning":
            return TRMConfig.for_arc_agi()
        else:
            return TRMConfig()

    def solve(
        self,
        input_data: torch.Tensor,
        max_steps: int = 16,
        show_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """
        Solve a reasoning task with recursive refinement.

        Returns solution, confidence, steps, and optionally trajectory.
        """
        with torch.no_grad():
            result = self.model.solve(
                input_data.to(self.device),
                max_steps=max_steps,
                return_trajectory=show_trajectory,
            )

        output = {
            "solution": result["solution"].cpu(),
            "confidence": result["confidence"].item(),
            "steps": result["steps"],
            "converged": result["confidence"].item() > 0.5,
        }

        if show_trajectory and self.verbose:
            self._print_trajectory(result["trajectory"])

        return output

    def _print_trajectory(self, trajectory):
        """Print reasoning trajectory for debugging."""
        print("\n=== Reasoning Trajectory ===")
        for step in trajectory:
            print(f"Step {step['step'] + 1}: q_hat={step['q_hat'].item():.3f}")
        print("=" * 30)

    def interactive_mode(self):
        """Run agent in interactive mode."""
        print(f"\n🧠 TRM Agent - Interactive Mode ({self.task})")
        print("Type 'help' for commands, 'quit' to exit\n")

        while True:
            try:
                cmd = input("agent> ").strip()

                if cmd == "quit" or cmd == "exit":
                    break
                elif cmd == "help":
                    self._print_help()
                elif cmd.startswith("solve "):
                    self._handle_solve(cmd[6:])
                elif cmd == "info":
                    self._print_info()
                else:
                    print("Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def _print_help(self):
        print("""
Available Commands:
  solve <file>  - Solve puzzle from file
  info          - Show model information
  quit          - Exit agent
        """)

    def _print_info(self):
        print(f"""
Model Information:
  Task: {self.task}
  Parameters: {self.model.num_parameters():,}
  Effective Depth: {self.model.config.effective_depth}
  Grid Size: {self.model.config.grid_size}
  T Cycles: {self.model.config.T_cycles}
  n Cycles: {self.model.config.n_cycles}
        """)


def main():
    parser = argparse.ArgumentParser(
        description="TRM-Powered CLI Agent for Reasoning Tasks"
    )
    parser.add_argument(
        "--task",
        choices=["sudoku", "maze", "arc", "reasoning"],
        default="reasoning",
        help="Task type"
    )
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--input", type=str, help="Input file for batch mode")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    agent = TRMAgent(
        model_path=args.model,
        task=args.task,
        device=args.device,
        verbose=args.verbose,
    )

    if args.interactive:
        agent.interactive_mode()
    elif args.input:
        # Load and solve from file
        with open(args.input) as f:
            data = json.load(f)
        puzzle = torch.tensor(data["puzzle"])
        result = agent.solve(puzzle, show_trajectory=args.verbose)
        print(json.dumps({
            "solution": result["solution"].tolist(),
            "confidence": result["confidence"],
            "steps": result["steps"],
        }, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

### Integration with Existing CLI Tools

To integrate TRM reasoning into an existing CLI agent framework:

```python
# Example: Adding TRM reasoning to a tool-calling agent

class ReasoningTool:
    """Tool that uses TRM for complex reasoning steps."""

    name = "recursive_reasoning"
    description = "Use recursive neural reasoning for complex puzzles/planning"

    def __init__(self, model_path: str):
        self.agent = TRMAgent(model_path=model_path)

    def __call__(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute recursive reasoning on a problem.

        Input: {"puzzle": [[...]], "type": "sudoku"}
        Output: {"solution": [[...]], "confidence": 0.95, "steps": 5}
        """
        puzzle = torch.tensor(problem["puzzle"])
        return self.agent.solve(puzzle)


# Usage in a larger agent system
def agent_loop(task: str, tools: list):
    """Example agent loop with TRM reasoning."""
    reasoning_tool = next(t for t in tools if t.name == "recursive_reasoning")

    if needs_complex_reasoning(task):
        # Use TRM for iterative refinement
        result = reasoning_tool({"puzzle": parse_puzzle(task)})
        if result["converged"]:
            return format_solution(result["solution"])

    # Fall back to other tools
    ...
```

---

## Future Extensions

### 1. Multi-Task Training
Train a single TRM model on multiple task types:
```python
config = TRMConfig(
    grid_size=30,  # Max size
    vocab_size=16,  # Union of all task vocabularies
    task_embedding=True,  # Add task type embedding
)
```

### 2. Streaming/Progressive Output
Yield intermediate solutions during inference:
```python
def solve_streaming(self, puzzle, callback):
    for step, (y, z) in enumerate(self.recursive_steps(puzzle)):
        partial_solution = self.output_head(y)
        confidence = torch.sigmoid(self.q_head(z))
        callback(step, partial_solution, confidence)
        if confidence > threshold:
            break
```

### 3. Reward Model Integration
Use Q-head as a learned reward signal for RL fine-tuning:
```python
# Q-head predicts correctness - can be used as reward
reward = model.q_head(z).sigmoid()
```

### 4. Tool-Augmented TRM
Allow TRM to call external tools during reasoning:
```python
class ToolAugmentedTRM(TRM):
    def _update_z(self, x, y, z, tools):
        # Standard z update
        z_new = super()._update_z(x, y, z)

        # Tool consultation gate
        tool_gate = self.tool_gate(z_new)
        if tool_gate > 0.5:
            tool_result = self.call_tool(y, tools)
            z_new = z_new + self.tool_embed(tool_result)

        return z_new
```

---

## Benchmarks Reference

| Task | TRM | DeepSeek-R1 | o3-mini | Gemini 2.5 Pro |
|------|-----|-------------|---------|----------------|
| ARC-AGI-1 | **45%** (7M) | - | ~30% | ~25% |
| ARC-AGI-2 | **8%** (7M) | - | - | - |
| Sudoku-Extreme | **87.4%** (5M) | 0.0% | - | - |
| Maze-Hard | **85.3%** (7M) | - | - | - |

Key insight: **Recursion beats scale**. A 7M parameter model with iterative refinement outperforms billion-parameter models on structured reasoning tasks.

---

## File Manifest

```
trm/
├── __init__.py                    # Module metadata, presets, formulation
├── README.md                      # Quick start guide
├── src/
│   ├── __init__.py               # Exports: TRM, TRMConfig, layers
│   ├── model.py                  # TRM class, TRMConfig, train_step, solve
│   └── layers.py                 # TRMBlock, DeepRecursion, QHead, etc.
├── configs/
│   ├── default.yaml              # Default hyperparameters
│   ├── sudoku.yaml               # Sudoku task config
│   └── arc_agi.yaml              # ARC-AGI task config
├── cli/
│   ├── __init__.py               # CLI exports
│   ├── train.py                  # Training script
│   └── infer.py                  # Inference script
├── tests/
│   ├── __init__.py
│   └── test_trm.py               # Unit tests
├── docs/
│   └── IMPLEMENTATION.md         # This file
└── models/                        # Checkpoint storage (empty)
```

---

## References

1. **TRM Paper**: Jolicoeur-Martineau, A. (2025). "Less is More: Recursive Reasoning with Tiny Networks." arXiv:2510.04871

2. **Original Implementation**: https://github.com/SamsungSAILMontreal/TinyRecursiveModels

3. **Related Work**:
   - HRM (Hierarchical Reasoning Model) - Fixed-point iteration
   - Neural Turing Machines - External memory
   - Adaptive Computation Time - Variable computation

---

*Implementation by the ml_research module team. See main README for project overview.*
