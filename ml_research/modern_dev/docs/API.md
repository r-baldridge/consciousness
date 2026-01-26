# API Reference

Complete API documentation for the ML Research Code Repair Pipeline.

---

## Table of Contents

1. [MLOrchestrator](#mlorchestrator)
2. [Request/Response](#requestresponse)
3. [TRM (Tiny Recursive Model)](#trm-tiny-recursive-model)
4. [Mamba (Selective State Spaces)](#mamba-selective-state-spaces)
5. [Architecture Registry](#architectureregistry)
6. [Task Router](#taskrouter)
7. [Configurations](#configurations)

---

## MLOrchestrator

Main entry point for the ML Research pipeline. Coordinates routing between TRM, Mamba, and RLM architectures.

### Constructor

```python
MLOrchestrator(config_path: Optional[str] = None)
```

**Parameters:**
- `config_path`: Optional path to configuration file

**Example:**
```python
from modern_dev.orchestrator.router import MLOrchestrator

orchestrator = MLOrchestrator()
```

### Methods

#### process()

```python
def process(self, request: Request) -> Response:
    """
    Process a request through the optimal pipeline.

    Args:
        request: Request object containing task specification

    Returns:
        Response object with results
    """
```

**Example:**
```python
from modern_dev.orchestrator.router import MLOrchestrator, Request

orch = MLOrchestrator()
response = orch.process(Request(
    task_type="code_repair",
    input_data={"buggy_code": "def foo(): retrun 1"},
))
```

#### get_status()

```python
def get_status(self) -> Dict[str, Any]:
    """
    Get orchestrator status.

    Returns:
        Dictionary with:
            - initialized: Whether orchestrator is ready
            - architectures: List of registered architectures
            - techniques: List of available techniques
            - routing_stats: Routing statistics
            - failure_stats: Failure handling statistics
    """
```

#### get_architecture_info()

```python
def get_architecture_info(self, name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed info about an architecture.

    Args:
        name: Architecture identifier ("trm", "mamba", "rlm")

    Returns:
        Dictionary with architecture details or None if not found
    """
```

#### list_capabilities()

```python
def list_capabilities(self) -> Dict[str, Dict[str, Any]]:
    """
    List all architecture capabilities.

    Returns:
        Dictionary mapping architecture names to their capabilities
    """
```

---

## Request/Response

### Request

```python
@dataclass
class Request:
    """
    A request to the MLOrchestrator.

    Attributes:
        task_type: Type of task to perform
            - "code_repair": Fix buggy code
            - "code_generation": Generate new code
            - "code_analysis": Analyze existing code
            - "iterative_refinement": Iteratively improve code
            - "long_context": Process long sequences
            - "streaming": Stream output tokens
            - "reasoning": Logical reasoning tasks
        input_data: Input data for processing
        constraints: Optional resource constraints
        technique: Optional specific technique to use
        preferred_architecture: Optional architecture preference
        config: Optional configuration overrides
    """
    task_type: str
    input_data: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    technique: Optional[str] = None
    preferred_architecture: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
```

**Input Data Fields:**

For `code_repair`:
```python
{
    "buggy_code": str,           # Required: Code to repair
    "error_message": str,        # Optional: Error context
    "context": str,              # Optional: Surrounding code
    "test_cases": List[Dict],    # Optional: Validation tests
}
```

For `code_generation`:
```python
{
    "specification": str,        # Required: What to generate
    "language": str,             # Optional: Target language
    "constraints": List[str],    # Optional: Requirements
}
```

**Constraints:**
```python
{
    "max_latency_ms": int,       # Maximum latency
    "max_memory": int,           # Maximum memory (GB)
    "context_length": int,       # Required context length
    "streaming": bool,           # Enable streaming
    "priority": int,             # Task priority (1-10)
}
```

### Response

```python
@dataclass
class Response:
    """
    Response from the MLOrchestrator.

    Attributes:
        success: Whether processing succeeded
        output: Processing output
        architecture_used: Architecture that handled the request
        technique_used: Technique applied (if any)
        execution_time_ms: Total execution time
        routing_decision: How the request was routed
        metadata: Additional response metadata
    """
    success: bool
    output: Any
    architecture_used: str
    technique_used: Optional[str] = None
    execution_time_ms: float = 0.0
    routing_decision: Optional[RoutingDecision] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## TRM (Tiny Recursive Model)

Recursive reasoning architecture for code repair tasks.

### CodeRepairTRM

```python
class CodeRepairTRM(nn.Module):
    """
    Tiny Recursive Model for Code Repair.

    Input: 64 rows x 48 tokens grid (representing code)
    Output: Repaired code grid

    Key features:
    - 8 recursive iterations with weight sharing
    - 42 effective layers of depth
    - Early stopping based on confidence threshold
    """
```

#### Constructor

```python
CodeRepairTRM(config: CodeRepairConfig)
```

#### forward()

```python
def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Forward pass.

    Args:
        x: Input token IDs (batch, height, width) - [batch, 64, 48]
        mask: Optional attention mask (batch, height, width)
        labels: Optional target token IDs for loss computation

    Returns:
        Tuple of (logits, info):
            - logits: Output logits (batch, height, width, vocab_size)
            - info: Dictionary with iteration info and optional loss
                - iterations: Number of iterations performed
                - confidence: Model confidence
                - q_hat: Raw confidence logits
                - loss: Cross-entropy loss (if labels provided)
    """
```

#### generate()

```python
@torch.no_grad()
def generate(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate repaired code.

    Args:
        x: Input token IDs (batch, height, width)
        mask: Optional attention mask
        max_iterations: Override max iterations

    Returns:
        Dictionary with:
            - output: Predicted token IDs (batch, height, width)
            - logits: Output logits
            - iterations: Number of iterations performed
            - confidence: Model confidence
    """
```

#### Class Methods

```python
@classmethod
def from_config(cls, config: CodeRepairConfig) -> "CodeRepairTRM":
    """Create model from configuration."""

@classmethod
def from_pretrained(cls, path: str) -> "CodeRepairTRM":
    """Load pretrained model from checkpoint."""
```

#### Instance Methods

```python
def save_pretrained(self, path: str):
    """Save model checkpoint."""

def num_parameters(self, trainable_only: bool = True) -> int:
    """Count model parameters."""
```

### TRM (Base Model)

For Sudoku, maze, and ARC-AGI tasks.

```python
class TRM(nn.Module):
    """
    Tiny Recursive Model.

    A small recursive network that achieves strong generalization
    on reasoning tasks through iterative refinement.
    """
```

#### solve()

```python
@torch.no_grad()
def solve(
    self,
    input_ids: torch.Tensor,
    max_steps: Optional[int] = None,
    return_trajectory: bool = False,
) -> Dict[str, Any]:
    """
    Inference with recursive refinement.

    Args:
        input_ids: Input grid
        max_steps: Maximum inference steps
        return_trajectory: Whether to return intermediate predictions

    Returns:
        Dictionary containing:
            - solution: Final predicted grid (batch, seq_len)
            - confidence: Final q_hat value (batch,)
            - steps: Number of steps taken
            - trajectory: List of intermediate predictions (if requested)
    """
```

#### train_step()

```python
def train_step(
    self,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    max_steps: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Training step with deep supervision.

    Performs multiple supervision steps, halting early if q_hat > 0.

    Args:
        input_ids: Input grid
        labels: Target grid
        max_steps: Maximum supervision steps

    Returns:
        Dictionary with accumulated losses and metrics
    """
```

---

## Mamba (Selective State Spaces)

Linear-time sequence modeling with selective state spaces.

### MambaLM

```python
class MambaLM(nn.Module):
    """
    Full Mamba Language Model.

    Features:
    - O(1) per-token inference with caching
    - Linear complexity O(N) training
    - Efficient long sequence processing
    """
```

#### Constructor

```python
MambaLM(config: MambaLMConfig)
```

#### forward()

```python
def forward(
    self,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    cache: Optional[MambaCache] = None,
    return_hidden_states: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Forward pass with optional loss computation.

    Args:
        input_ids: Token IDs of shape [batch, seq_len]
        labels: Optional target token IDs for loss computation
        cache: Optional MambaCache for efficient inference
        return_hidden_states: Whether to return all layer hidden states

    Returns:
        If labels is None:
            logits: Tensor of shape [batch, seq_len, vocab_size]
        If labels is provided:
            Tuple of (logits, loss) where loss is a scalar
        If return_hidden_states:
            Dict with 'logits', 'loss' (if labels), and 'hidden_states'
    """
```

#### generate()

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    repetition_penalty: float = 1.0,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Autoregressive generation with O(1) per-token inference.

    Args:
        input_ids: Initial token IDs of shape [batch, seq_len]
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top tokens to consider
        top_p: Nucleus sampling probability threshold
        do_sample: Whether to sample or use greedy decoding
        eos_token_id: Token ID to stop generation
        pad_token_id: Token ID for padding
        repetition_penalty: Penalty for repeating tokens
        use_cache: Whether to use caching for efficiency

    Returns:
        Generated token IDs of shape [batch, seq_len + num_generated]
    """
```

#### allocate_inference_cache()

```python
def allocate_inference_cache(
    self,
    batch_size: int,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> MambaCache:
    """
    Allocate an empty inference cache.

    Args:
        batch_size: Batch size for inference
        device: Device to allocate cache on
        dtype: Data type for cache

    Returns:
        Empty MambaCache instance
    """
```

### MambaCache

```python
@dataclass
class MambaCache:
    """
    Cache for efficient autoregressive generation.

    Stores convolution and SSM states for each layer, enabling
    O(1) per-token inference instead of O(n) recomputation.

    Attributes:
        conv_states: List of convolution state tensors
        ssm_states: List of SSM hidden state tensors
        seqlen_offset: Current position in the sequence
    """
    conv_states: List[torch.Tensor]
    ssm_states: List[torch.Tensor]
    seqlen_offset: int = 0
```

#### Class Methods

```python
@classmethod
def empty(
    cls,
    config: MambaLMConfig,
    batch_size: int,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> "MambaCache":
    """Create an empty cache for a given configuration."""
```

#### Instance Methods

```python
def update(self, layer_idx: int, conv_state: torch.Tensor, ssm_state: torch.Tensor):
    """Update cache for a specific layer."""

def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get cache tensors for a specific layer."""

def reset(self):
    """Reset all cache states to zero."""

def increment_offset(self, n: int = 1):
    """Increment the sequence length offset."""
```

---

## ArchitectureRegistry

Central registry for model architectures and their capabilities.

```python
class ArchitectureRegistry:
    """
    Registry of available architectures and their capabilities.
    """
```

### Methods

#### register()

```python
def register(
    self,
    name: str,
    capability: ArchitectureCapability,
    model_factory: Callable[[], Any],
) -> None:
    """
    Register an architecture with its capabilities and factory.

    Args:
        name: Unique identifier for the architecture
        capability: ArchitectureCapability describing what it can do
        model_factory: Callable that creates/loads the model instance
    """
```

#### get_capable()

```python
def get_capable(
    self,
    task: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Get architectures capable of handling a task.

    Args:
        task: Task type to check
        constraints: Optional constraints to filter by

    Returns:
        List of architecture names that can handle the task
    """
```

#### get_ranked()

```python
def get_ranked(
    self,
    task: str,
    constraints: Optional[Dict[str, Any]] = None,
    strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH,
) -> List[Tuple[str, float]]:
    """
    Get architectures ranked by suitability for a task.

    Args:
        task: Task type
        constraints: Optional constraints
        strategy: Ranking strategy to use

    Returns:
        List of (architecture_name, score) tuples, sorted descending
    """
```

#### record_performance()

```python
def record_performance(
    self,
    name: str,
    task: str,
    success: bool,
    latency_ms: float,
    quality_score: Optional[float] = None,
) -> None:
    """Record performance data for future routing decisions."""
```

---

## TaskRouter

Intelligent task routing based on capabilities and constraints.

```python
class TaskRouter:
    """
    Routes tasks to optimal architecture based on multiple factors.
    """
```

### Methods

#### route()

```python
def route(
    self,
    task: Task,
    strategy: Optional[RoutingStrategy] = None,
) -> RoutingDecision:
    """
    Determine best architecture for a task.

    Args:
        task: Task to route
        strategy: Optional override for routing strategy

    Returns:
        RoutingDecision with primary, fallback, and reasoning
    """
```

### RoutingStrategy

```python
class RoutingStrategy(Enum):
    CAPABILITY_MATCH = "capability_match"    # Match based on capabilities
    PERFORMANCE_BASED = "performance_based"  # Use historical performance
    RESOURCE_AWARE = "resource_aware"        # Optimize for resources
    LATENCY_OPTIMIZED = "latency_optimized"  # Minimize latency
    QUALITY_OPTIMIZED = "quality_optimized"  # Maximize quality
```

---

## Configurations

### CodeRepairConfig

```python
@dataclass
class CodeRepairConfig:
    """Configuration for Code Repair TRM model."""

    # Grid configuration
    grid_height: int = 64
    grid_width: int = 48
    vocab_size: int = 32768

    # Model architecture
    embed_dim: int = 256
    n_heads: int = 8
    ffn_dim: int = 1024
    n_blocks: int = 6

    # Recursion parameters
    max_iterations: int = 8
    min_iterations: int = 2

    # Training
    dropout: float = 0.0
    q_threshold: float = 0.95

    # Memory optimization
    use_gradient_checkpointing: bool = False
```

**Presets:**
```python
CodeRepairConfig.for_code_repair_tiny()   # ~1M params
CodeRepairConfig.for_code_repair_small()  # ~9M params
CodeRepairConfig.for_code_repair_base()   # ~12M params
CodeRepairConfig.for_code_repair_large()  # ~23M params
CodeRepairConfig.for_arc_agi_7m()         # 7M params for ARC
```

### MambaLMConfig

```python
@dataclass
class MambaLMConfig:
    """Configuration for Mamba Language Model."""

    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 24
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
    bias: bool = False
    conv_bias: bool = True
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    residual_in_fp32: bool = True
    use_fast_path: bool = True
    dropout: float = 0.0
```

### TRMConfig

```python
@dataclass
class TRMConfig:
    """Configuration for Tiny Recursive Model."""

    # Task configuration
    grid_size: int = 9
    vocab_size: int = 10
    max_seq_len: int = 81

    # Model architecture
    embed_dim: int = 512
    n_layers: int = 2
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Recursion parameters
    T_cycles: int = 3
    n_cycles: int = 6
    max_supervision_steps: int = 16

    # Architecture options
    use_attention: bool = True
    use_rotary: bool = True
    use_swiglu: bool = True
    use_stable_max: bool = True

    # Inference
    q_threshold: float = 0.0
```

**Presets:**
```python
TRMConfig.for_sudoku()          # 9x9 grid, MLP
TRMConfig.for_maze(grid_size)   # Variable size, attention
TRMConfig.for_arc_agi(30)       # 30x30 grid for ARC
```

---

## Utility Functions

### load_pretrained()

```python
def load_pretrained(
    model_name_or_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> MambaLM:
    """
    Load a pretrained Mamba model.

    Args:
        model_name_or_path: Path to checkpoint directory
        device: Device to load model on
        dtype: Data type for model parameters

    Returns:
        Loaded MambaLM model
    """
```

### save_checkpoint()

```python
def save_checkpoint(
    model: MambaLM,
    optimizer: Optional[torch.optim.Optimizer],
    path: Union[str, Path],
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    **kwargs,
) -> None:
    """Save training checkpoint."""
```

### load_checkpoint()

```python
def load_checkpoint(
    path: Union[str, Path],
    model: Optional[MambaLM] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load training checkpoint."""
```
