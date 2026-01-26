# Subagent Orchestration Guide

This document provides ready-to-execute prompts for spawning subagents to implement the ML Research pipeline.

---

## Quick Start: Phase 1 Parallel Launch

Execute these 4 Task tool calls in a single message to start Phase 1 with maximum parallelism:

```
Launch all 4 Phase 1 agents in parallel:
1. Agent-A: TRM Forward Pass
2. Agent-B: RLM Variable Extractor
3. Agent-C: Mamba Discrete SSM
4. Agent-D: Shared Backbone Library
```

---

## Phase 1 Agent Prompts

### Agent-A: TRM Forward Pass (TRM-001, TRM-002)

**Subagent Type**: `general-purpose`
**Estimated Time**: 20-32 hours of work
**Run in Background**: Yes

```markdown
# Task: Implement TRM Forward Pass

You are implementing the forward pass for TRM (Tiny Recursive Model), a 7M parameter
recursive reasoning architecture for code repair.

## Files to Create/Modify

1. `/consciousness/ml_research/modern_dev/trm/src/model.py`
2. `/consciousness/ml_research/modern_dev/trm/src/layers.py`

## Current State

Read the existing files first:
- `modern_dev/trm/src/model.py` - Has class structure, needs forward implementation
- `modern_dev/trm/src/layers.py` - Has layer definitions, needs completion
- `modern_dev/trm/src/config.py` - Configuration (reference only)
- `modern_dev/shared/data/loaders/trm.py` - Data format (reference)

## Requirements

### 1. DeepRecursion Class
Implement the core recursive processing:

```python
class DeepRecursion(nn.Module):
    def __init__(self, config: TRMConfig):
        # Initialize recursive blocks
        # Each block: LayerNorm → Attention → FFN → Residual

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # x: [batch, height, width, embed_dim]
        # Returns: output, iteration_info

        for iteration in range(self.max_iterations):
            # Apply recursive block
            x = self.blocks[iteration % len(self.blocks)](x, mask)

            # Check early stopping condition
            if self.should_stop(x, iteration):
                break

        return x, {"iterations": iteration + 1, "confidence": confidence}
```

### 2. RecursiveBlock Class
```python
class RecursiveBlock(nn.Module):
    def __init__(self, config: TRMConfig):
        self.norm1 = RMSNorm(config.embed_dim)
        self.attn = GridAttention(config)  # 2D attention over grid
        self.norm2 = RMSNorm(config.embed_dim)
        self.ffn = SwiGLU(config.embed_dim, config.ffn_dim)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

### 3. GridAttention Class
2D attention that operates over the 64×48 grid:
```python
class GridAttention(nn.Module):
    # Attention over flattened grid positions
    # Use relative position encoding for grid structure
```

### 4. IterationController
```python
class IterationController:
    def __init__(self, q_threshold: float = 0.95):
        self.q_threshold = q_threshold

    def should_stop(self, hidden: torch.Tensor, iteration: int) -> bool:
        # Compute confidence from hidden state
        # Stop if confidence > threshold
```

## Acceptance Criteria

1. Forward pass runs without error:
```python
model = TRM(config)
x = torch.randint(0, 32768, (2, 64, 48))  # [batch, height, width]
mask = torch.ones(2, 64, 48)
output, info = model(x, mask)
assert output.shape == (2, 64, 48, 32768)  # [batch, H, W, vocab]
```

2. Early stopping works:
```python
# Model stops between 1-8 iterations based on confidence
assert 1 <= info["iterations"] <= 8
```

3. Memory usage reasonable:
```python
# Should fit in 8GB VRAM for batch_size=32
```

4. Gradient flow verified:
```python
loss = output.sum()
loss.backward()
assert model.blocks[0].attn.qkv.weight.grad is not None
```

## Implementation Notes

- Use gradient checkpointing for memory efficiency
- Support both training and inference modes
- Add hooks for visualization of attention patterns
- Log iteration counts for analysis

## Deliverables

1. Working `DeepRecursion` class with forward pass
2. `RecursiveBlock` with attention and FFN
3. `GridAttention` for 2D grid processing
4. `IterationController` for early stopping
5. Unit tests in `modern_dev/trm/tests/test_model.py`

When complete, report:
- Files created/modified
- Test results
- Memory usage measurements
- Any design decisions made
```

---

### Agent-B: RLM Variable Extractor (RLM-001, RLM-002)

**Subagent Type**: `general-purpose`
**Estimated Time**: 20-36 hours of work
**Run in Background**: Yes

```markdown
# Task: Implement RLM Variable Extractor

You are implementing the variable extraction component of RLM (Recursive Language Model),
a technique for code synthesis through problem decomposition.

## Files to Create/Modify

1. `/consciousness/ml_research/ml_techniques/code_synthesis/variable_extractor.py` (new)
2. `/consciousness/ml_research/ml_techniques/code_synthesis/constraints.py` (new)
3. `/consciousness/ml_research/ml_techniques/code_synthesis/rlm.py` (extend)

## Current State

Read existing files:
- `ml_techniques/__init__.py` - Technique base classes
- `ml_techniques/code_synthesis/` - May have stubs

## Requirements

### 1. Variable Extractor

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any

class VariableType(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    OBJECT = "object"
    FUNCTION = "function"
    CLASS = "class"
    UNKNOWN = "unknown"

@dataclass
class ExtractedVariable:
    name: str
    var_type: VariableType
    description: str
    constraints: List[str]
    dependencies: List[str]
    source_span: Optional[Tuple[int, int]] = None

class VariableExtractor:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def extract(self, specification: str) -> List[ExtractedVariable]:
        """
        Extract variables from a natural language specification.

        Example:
            Input: "Create a function that takes a list of integers and
                    returns the sum of all even numbers."
            Output: [
                ExtractedVariable(
                    name="input_list",
                    var_type=VariableType.LIST,
                    description="List of integers to process",
                    constraints=["elements are integers"],
                    dependencies=[]
                ),
                ExtractedVariable(
                    name="result",
                    var_type=VariableType.INTEGER,
                    description="Sum of even numbers",
                    constraints=["non-negative if input non-negative"],
                    dependencies=["input_list"]
                )
            ]
        """
        # 1. Parse specification into sentences
        # 2. Identify entities (nouns, noun phrases)
        # 3. Infer types from context
        # 4. Extract relationships and constraints
        # 5. Build dependency graph

    def _parse_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text."""

    def _infer_type(self, entity: str, context: str) -> VariableType:
        """Infer variable type from surrounding context."""

    def _extract_constraints(self, text: str, variable: str) -> List[str]:
        """Extract constraints mentioned for a variable."""
```

### 2. Constraint Analyzer

```python
@dataclass
class Constraint:
    variable: str
    constraint_type: str  # "range", "type", "relationship", "format"
    expression: str
    is_hard: bool = True  # Hard constraint vs preference

class ConstraintAnalyzer:
    def __init__(self):
        self.constraints: List[Constraint] = []

    def analyze(self, variables: List[ExtractedVariable]) -> List[Constraint]:
        """
        Analyze variables and their constraints.

        Returns formal constraints that can be checked.
        """

    def build_dependency_graph(self, variables: List[ExtractedVariable]) -> Dict:
        """
        Build a DAG of variable dependencies.

        Returns: {variable_name: [dependent_variables]}
        """

    def topological_sort(self, graph: Dict) -> List[str]:
        """Return variables in order of computation."""

    def validate(self, variables: Dict[str, Any], constraints: List[Constraint]) -> bool:
        """Check if variable assignments satisfy constraints."""
```

### 3. RLM Core Extension

```python
class RLM(TechniqueBase):
    TECHNIQUE_ID = "rlm"
    CATEGORY = TechniqueCategory.CODE_SYNTHESIS

    def __init__(self, config: Optional[TechniqueConfig] = None):
        super().__init__(config)
        self.extractor = VariableExtractor()
        self.analyzer = ConstraintAnalyzer()

    def decompose(self, specification: str) -> Dict:
        """
        Decompose a specification into structured components.

        Returns:
            {
                "variables": List[ExtractedVariable],
                "constraints": List[Constraint],
                "dependency_order": List[str],
                "subproblems": List[str]
            }
        """
        variables = self.extractor.extract(specification)
        constraints = self.analyzer.analyze(variables)
        order = self.analyzer.topological_sort(
            self.analyzer.build_dependency_graph(variables)
        )

        return {
            "variables": variables,
            "constraints": constraints,
            "dependency_order": order,
            "subproblems": self._identify_subproblems(variables, constraints)
        }

    def _identify_subproblems(self, variables, constraints) -> List[str]:
        """Break down into subproblems based on dependencies."""
```

## Test Cases

```python
def test_simple_extraction():
    extractor = VariableExtractor(use_llm=False)

    spec = "Write a function that takes two numbers and returns their sum."
    variables = extractor.extract(spec)

    assert len(variables) == 3  # num1, num2, result
    assert any(v.name in ["num1", "a", "first"] for v in variables)
    assert any(v.var_type == VariableType.INTEGER for v in variables)

def test_complex_extraction():
    extractor = VariableExtractor(use_llm=False)

    spec = """
    Create a function that:
    1. Takes a list of strings
    2. Filters out strings shorter than 3 characters
    3. Converts remaining strings to uppercase
    4. Returns the sorted list
    """
    variables = extractor.extract(spec)

    assert any(v.var_type == VariableType.LIST for v in variables)
    assert any("uppercase" in c.lower() for v in variables for c in v.constraints)

def test_constraint_analysis():
    analyzer = ConstraintAnalyzer()
    variables = [
        ExtractedVariable("x", VariableType.INTEGER, "Input", ["x > 0"], []),
        ExtractedVariable("y", VariableType.INTEGER, "Output", ["y = x * 2"], ["x"]),
    ]

    constraints = analyzer.analyze(variables)
    assert len(constraints) >= 2

    graph = analyzer.build_dependency_graph(variables)
    assert "x" in graph["y"] or graph["y"] == ["x"]
```

## Acceptance Criteria

1. Extracts variables from 10 diverse specifications correctly
2. Identifies types with >80% accuracy
3. Builds correct dependency graphs
4. Handles edge cases (empty input, ambiguous specs)
5. Works with and without LLM backend

## Deliverables

1. `variable_extractor.py` with full implementation
2. `constraints.py` with constraint analysis
3. Extended `rlm.py` with decompose() method
4. Unit tests with >80% coverage
5. 10 example specifications with expected outputs

When complete, report:
- Files created/modified
- Test results
- Accuracy on test cases
- Any design decisions made
```

---

### Agent-C: Mamba Discrete SSM (MAMBA-001, MAMBA-002)

**Subagent Type**: `general-purpose`
**Estimated Time**: 24-36 hours of work
**Run in Background**: Yes

```markdown
# Task: Implement Mamba Discrete SSM Core

You are implementing the core State Space Model component of Mamba,
a linear-time sequence model with selective state spaces.

## Files to Create/Modify

1. `/consciousness/ml_research/modern_dev/mamba_impl/src/ssm.py` (new/extend)
2. `/consciousness/ml_research/modern_dev/mamba_impl/src/parameterization.py` (new)

## Background: State Space Models

State Space Models (SSMs) model sequences through a hidden state:

```
h'(t) = A h(t) + B x(t)    # State equation
y(t) = C h(t) + D x(t)     # Output equation
```

For discrete sequences, we discretize using zero-order hold:
```
h[k] = Ā h[k-1] + B̄ x[k]
y[k] = C h[k]

where:
Ā = exp(Δ A)
B̄ = (Δ A)^{-1} (exp(Δ A) - I) Δ B
```

## Requirements

### 1. S4D Layer (Diagonal State Space)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class S4DKernel(nn.Module):
    """
    S4D: Diagonal State Space layer.

    Uses diagonal A matrix for efficiency while maintaining expressiveness.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A: diagonal matrix (stored as vector)
        # Initialize with HiPPO-LegS
        A = self._hippo_initializer(d_state)
        self.register_buffer("A", A)

        # Learnable parameters
        self.log_dt = nn.Parameter(torch.rand(d_model) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min))

        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

    def _hippo_initializer(self, N: int) -> torch.Tensor:
        """
        HiPPO-LegS initialization for A matrix.

        A[n] = -(n + 1/2) for n = 0, 1, ..., N-1
        """
        return -torch.arange(N, dtype=torch.float32) - 0.5

    def discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize continuous parameters to discrete.

        Returns:
            A_bar: [d_model, d_state] discretized A
            B_bar: [d_model, d_state] discretized B
        """
        dt = torch.exp(self.log_dt)  # [d_model]

        # A_bar = exp(dt * A)
        # For diagonal A, this is element-wise
        A_bar = torch.exp(dt.unsqueeze(-1) * self.A.unsqueeze(0))  # [d_model, d_state]

        # B_bar = (dt * A)^{-1} (A_bar - I) * dt * B
        # Simplified for diagonal: B_bar = (A_bar - 1) / A * B
        B_bar = (A_bar - 1) / self.A.unsqueeze(0) * self.B  # [d_model, d_state]

        return A_bar, B_bar

    def forward_recurrent(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent mode: O(L) sequential, O(1) memory per step.

        Args:
            x: [batch, d_model] single time step
            h: [batch, d_model, d_state] hidden state

        Returns:
            y: [batch, d_model] output
            h_new: [batch, d_model, d_state] new hidden state
        """
        batch = x.shape[0]

        if h is None:
            h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)

        A_bar, B_bar = self.discretize()

        # h_new = A_bar * h + B_bar * x
        h_new = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x.unsqueeze(-1)

        # y = C @ h + D * x
        y = torch.einsum('bds,ds->bd', h_new, self.C) + self.D * x

        return y, h_new

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolutional mode: O(L log L) parallel via FFT.

        Args:
            x: [batch, length, d_model]

        Returns:
            y: [batch, length, d_model]
        """
        batch, length, d_model = x.shape

        # Compute convolution kernel
        kernel = self._compute_kernel(length)  # [d_model, length]

        # FFT convolution
        x_f = torch.fft.rfft(x.transpose(1, 2), n=2*length)  # [batch, d_model, length+1]
        k_f = torch.fft.rfft(kernel, n=2*length)  # [d_model, length+1]

        y_f = x_f * k_f.unsqueeze(0)
        y = torch.fft.irfft(y_f, n=2*length)[..., :length]  # [batch, d_model, length]

        # Add skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * x.transpose(1, 2)

        return y.transpose(1, 2)

    def _compute_kernel(self, length: int) -> torch.Tensor:
        """
        Compute convolution kernel K.

        K[k] = C @ A_bar^k @ B_bar for k = 0, 1, ..., L-1
        """
        A_bar, B_bar = self.discretize()

        # Power series: A_bar^0, A_bar^1, ..., A_bar^{L-1}
        powers = torch.zeros(self.d_model, self.d_state, length, device=A_bar.device)
        powers[..., 0] = 1.0

        for k in range(1, length):
            powers[..., k] = powers[..., k-1] * A_bar

        # K = C @ powers @ B_bar
        # [d_model, d_state] @ [d_model, d_state, length] @ [d_model, d_state]
        kernel = torch.einsum('ds,dsk,ds->dk', self.C, powers, B_bar)

        return kernel

    def forward(self, x: torch.Tensor, mode: str = "conv") -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, length, d_model]
            mode: "conv" for training, "recurrent" for inference
        """
        if mode == "conv":
            return self.forward_conv(x)
        else:
            # Sequential recurrent for inference
            outputs = []
            h = None
            for t in range(x.shape[1]):
                y, h = self.forward_recurrent(x[:, t], h)
                outputs.append(y)
            return torch.stack(outputs, dim=1)
```

### 2. Parameterization Module

```python
class HiPPOInitializer:
    """
    HiPPO (High-order Polynomial Projection Operators) initialization.

    Provides principled initialization for SSM matrices that enables
    long-range dependency modeling.
    """

    @staticmethod
    def legs(N: int) -> torch.Tensor:
        """
        HiPPO-LegS: Legendre polynomials with scaling.

        Best for general sequence modeling.
        """
        A = torch.zeros(N, N)
        for n in range(N):
            for k in range(N):
                if n > k:
                    A[n, k] = (2*n + 1)**0.5 * (2*k + 1)**0.5
                elif n == k:
                    A[n, k] = n + 1
        return -A

    @staticmethod
    def legs_diagonal(N: int) -> torch.Tensor:
        """
        Diagonal approximation of HiPPO-LegS.

        A[n] = -(n + 1/2)
        """
        return -torch.arange(N, dtype=torch.float32) - 0.5

class SSMNormalization(nn.Module):
    """
    Normalization for SSM outputs to prevent explosion.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms
```

## Acceptance Criteria

1. Forward pass produces correct shapes:
```python
layer = S4DKernel(d_model=256, d_state=64)
x = torch.randn(2, 1000, 256)
y = layer(x, mode="conv")
assert y.shape == (2, 1000, 256)
```

2. Recurrent and conv modes produce same output:
```python
y_conv = layer(x, mode="conv")
y_rec = layer(x, mode="recurrent")
assert torch.allclose(y_conv, y_rec, atol=1e-4)
```

3. Gradients flow correctly:
```python
y = layer(x, mode="conv")
loss = y.sum()
loss.backward()
assert layer.B.grad is not None
```

4. Memory efficient for long sequences:
```python
x_long = torch.randn(1, 100000, 256)  # 100K tokens
y = layer(x_long, mode="conv")  # Should not OOM on 8GB GPU
```

## Deliverables

1. `ssm.py` with S4DKernel implementation
2. `parameterization.py` with HiPPO and normalization
3. Unit tests verifying correctness
4. Benchmark comparing conv vs recurrent modes

When complete, report:
- Files created/modified
- Test results
- Performance benchmarks (speed, memory)
- Any numerical stability issues encountered
```

---

### Agent-D: Shared Backbone Library (INFRA-001)

**Subagent Type**: `general-purpose`
**Estimated Time**: 24-36 hours of work
**Run in Background**: Yes

```markdown
# Task: Create Shared Backbone Library

You are creating a library of shared neural network components that will be
used across TRM, Mamba, CTM, and other architectures.

## Files to Create

```
modern_dev/shared/blocks/
├── __init__.py
├── attention.py      # Attention mechanisms
├── normalization.py  # Normalization layers
├── activations.py    # Activation functions
├── embeddings.py     # Positional embeddings
├── feedforward.py    # FFN variants
└── conv.py           # Convolution layers
```

## Requirements

### 1. Attention Mechanisms (`attention.py`)

```python
class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention with optional features.

    Features:
    - Scaled dot-product attention
    - Optional flash attention (if available)
    - Causal masking
    - Relative position bias
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
        use_flash: bool = True,
    ):
        ...

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...

class GridAttention(nn.Module):
    """
    2D attention for grid-structured data (e.g., code grids).

    Attends over both row and column dimensions.
    """

    def __init__(self, d_model: int, n_heads: int, grid_size: Tuple[int, int]):
        ...
```

### 2. Normalization Layers (`normalization.py`)

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.

    More stable than LayerNorm, used in LLaMA, etc.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms

class LayerNorm(nn.Module):
    """Standard Layer Normalization with optional bias."""

class GroupNorm(nn.Module):
    """Group Normalization for conv-like architectures."""
```

### 3. Activation Functions (`activations.py`)

```python
class SwiGLU(nn.Module):
    """
    SwiGLU activation: Swish-Gated Linear Unit.

    SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
    """

    def __init__(self, d_model: int, d_ffn: int, bias: bool = False):
        super().__init__()
        self.w = nn.Linear(d_model, d_ffn, bias=bias)
        self.v = nn.Linear(d_model, d_ffn, bias=bias)
        self.out = nn.Linear(d_ffn, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.w(x)) * self.v(x))

class GeGLU(nn.Module):
    """GELU-Gated Linear Unit."""

class ReGLU(nn.Module):
    """ReLU-Gated Linear Unit."""
```

### 4. Positional Embeddings (`embeddings.py`)

```python
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Used in LLaMA, applies rotation to q/k based on position.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

class ALiBi(nn.Module):
    """
    Attention with Linear Biases.

    Adds linear bias based on distance, no learned parameters.
    """

class SinusoidalEmbedding(nn.Module):
    """Classic sinusoidal position embedding."""
```

### 5. Feedforward Networks (`feedforward.py`)

```python
class FeedForward(nn.Module):
    """
    Standard feedforward network with configurable activation.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        activation: str = "swiglu",  # "swiglu", "geglu", "relu", "gelu"
        dropout: float = 0.0,
        bias: bool = True,
    ):
        ...

class MoEFeedForward(nn.Module):
    """
    Mixture of Experts feedforward.

    Routes tokens to top-k experts.
    """
```

### 6. Convolution Layers (`conv.py`)

```python
class Conv1D(nn.Module):
    """
    1D convolution with various initializations.

    Used in Mamba, Hyena, etc.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        bias: bool = True,
        init: str = "default",  # "default", "kaiming", "xavier"
    ):
        ...

class DepthwiseConv1D(nn.Module):
    """Depthwise separable 1D convolution."""
```

## Package Init (`__init__.py`)

```python
from .attention import MultiHeadAttention, GridAttention
from .normalization import RMSNorm, LayerNorm, GroupNorm
from .activations import SwiGLU, GeGLU, ReGLU
from .embeddings import RotaryEmbedding, ALiBi, SinusoidalEmbedding
from .feedforward import FeedForward, MoEFeedForward
from .conv import Conv1D, DepthwiseConv1D

__all__ = [
    # Attention
    "MultiHeadAttention", "GridAttention",
    # Normalization
    "RMSNorm", "LayerNorm", "GroupNorm",
    # Activations
    "SwiGLU", "GeGLU", "ReGLU",
    # Embeddings
    "RotaryEmbedding", "ALiBi", "SinusoidalEmbedding",
    # FFN
    "FeedForward", "MoEFeedForward",
    # Conv
    "Conv1D", "DepthwiseConv1D",
]
```

## Acceptance Criteria

1. All components have consistent API
2. Unit tests for each component
3. All components work with mixed precision (FP16/BF16)
4. Memory-efficient implementations
5. Documented public interfaces

## Deliverables

1. All files listed above with full implementations
2. Unit tests in `modern_dev/shared/tests/test_blocks.py`
3. Performance benchmarks for key components
4. Usage examples in docstrings

When complete, report:
- Files created
- Test coverage
- Performance benchmarks
- Any design decisions
```

---

## Phase 2-4 Agent Prompts

The Phase 2-4 prompts follow the same structure but reference dependencies from earlier phases. See `IMPLEMENTATION_PLAN.md` for the complete task specifications.

---

## Execution Commands

### Launch Phase 1 (Copy-paste ready)

```python
# In Claude Code, execute this to launch all Phase 1 agents in parallel:

# This spawns 4 parallel agents for Phase 1
# Use run_in_background=True for each
```

### Monitor Agent Progress

```bash
# Check status of all running agents
# Use /tasks command in Claude Code

# Read agent output files for progress updates
```

### Validate Phase Completion

After Phase 1 completes, run validation:

```bash
# Run Phase 1 tests
cd ./ml_research
python -m pytest modern_dev/tests/phase1/ -v

# Verify imports work
python -c "from modern_dev.trm.src.model import DeepRecursion; print('TRM OK')"
python -c "from ml_techniques.code_synthesis.rlm import RLM; print('RLM OK')"
python -c "from modern_dev.mamba_impl.src.ssm import S4DKernel; print('Mamba OK')"
python -c "from modern_dev.shared.blocks import SwiGLU; print('Blocks OK')"
```

---

## Coordination Notes

### Handling Dependencies

When an agent completes a task that other agents depend on:

1. Agent reports completion with file list
2. Coordinator verifies deliverables
3. Coordinator notifies dependent agents
4. Dependent agents can proceed

### Handling Blockers

If an agent encounters a blocker:

1. Agent reports blocker with details
2. Coordinator evaluates options:
   - Provide guidance to unblock
   - Reassign to different agent
   - Modify plan to work around
3. Update IMPLEMENTATION_PLAN.md with changes

### Code Review Checkpoints

After each phase:

1. Review all new code for consistency
2. Run full test suite
3. Check for integration issues
4. Update documentation

---

## Summary

This orchestration guide enables parallel execution of the ML Research implementation plan with:

- **4 Phase 1 agents** working simultaneously
- **5 Phase 2 agents** after Phase 1 dependencies resolve
- **6 Phase 3 agents** for integration
- **All agents** converging for Phase 4 finalization

Total estimated time with parallel execution: **10 weeks** (vs 24+ weeks sequential)
