"""
Memory Benchmarks for ML Research Models (INTEG-003)

This module provides comprehensive memory profiling for:
- Model memory footprint
- Forward pass memory usage
- Backward pass memory usage
- Memory bottleneck detection
- Batch size estimation

Key Components:
    - MemoryBenchmark: Main memory profiling class
    - MemoryProfile: Detailed memory breakdown
    - ForwardPassProfile: Forward pass analysis
    - BackwardPassProfile: Backward pass analysis
    - BottleneckInfo: Memory bottleneck identification

Usage:
    from consciousness.ml_research.modern_dev.benchmarks import MemoryBenchmark

    benchmark = MemoryBenchmark()
    profile = benchmark.profile_model(model, (2, 64, 48))
    print(f"Total memory: {profile.total_memory_mb:.1f} MB")

    bottlenecks = benchmark.find_memory_bottleneck(model, (2, 64, 48))
    for b in bottlenecks:
        print(f"{b.layer_name}: {b.memory_mb:.1f} MB")
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LayerMemory:
    """Memory usage for a single layer.

    Attributes:
        name: Layer name (from named_modules).
        param_memory_mb: Memory for parameters.
        grad_memory_mb: Memory for gradients (during training).
        activation_memory_mb: Memory for activations.
        total_memory_mb: Total memory for this layer.
        num_params: Number of parameters.
        dtype: Data type of parameters.
    """
    name: str
    param_memory_mb: float
    grad_memory_mb: float = 0.0
    activation_memory_mb: float = 0.0
    total_memory_mb: float = 0.0
    num_params: int = 0
    dtype: str = "float32"

    def __post_init__(self):
        """Compute total if not set."""
        if self.total_memory_mb == 0.0:
            self.total_memory_mb = (
                self.param_memory_mb +
                self.grad_memory_mb +
                self.activation_memory_mb
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MemoryProfile:
    """Complete memory profile for a model.

    Attributes:
        model_name: Name of the profiled model.
        total_memory_mb: Total memory usage.
        param_memory_mb: Memory for parameters.
        grad_memory_mb: Memory for gradients.
        activation_memory_mb: Memory for activations.
        optimizer_memory_mb: Estimated optimizer memory.
        peak_memory_mb: Peak memory during profiling.
        layer_breakdown: Memory breakdown by layer.
        input_shape: Shape of input used for profiling.
        batch_size: Batch size used.
        device: Device used for profiling.
        dtype: Data type used.
        metadata: Additional metadata.
    """
    model_name: str
    total_memory_mb: float
    param_memory_mb: float
    grad_memory_mb: float = 0.0
    activation_memory_mb: float = 0.0
    optimizer_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    layer_breakdown: List[LayerMemory] = field(default_factory=list)
    input_shape: Tuple[int, ...] = ()
    batch_size: int = 1
    device: str = "cpu"
    dtype: str = "float32"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "total_memory_mb": self.total_memory_mb,
            "param_memory_mb": self.param_memory_mb,
            "grad_memory_mb": self.grad_memory_mb,
            "activation_memory_mb": self.activation_memory_mb,
            "optimizer_memory_mb": self.optimizer_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "layer_breakdown": [l.to_dict() for l in self.layer_breakdown],
            "input_shape": list(self.input_shape),
            "batch_size": self.batch_size,
            "device": self.device,
            "dtype": self.dtype,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 60,
            f"Memory Profile: {self.model_name}",
            "=" * 60,
            f"Input Shape: {self.input_shape}",
            f"Batch Size: {self.batch_size}",
            f"Device: {self.device}",
            f"Data Type: {self.dtype}",
            "",
            "Memory Breakdown:",
            f"  Parameters:  {self.param_memory_mb:8.2f} MB",
            f"  Gradients:   {self.grad_memory_mb:8.2f} MB",
            f"  Activations: {self.activation_memory_mb:8.2f} MB",
            f"  Optimizer:   {self.optimizer_memory_mb:8.2f} MB",
            "-" * 40,
            f"  Total:       {self.total_memory_mb:8.2f} MB",
            f"  Peak:        {self.peak_memory_mb:8.2f} MB",
        ]

        if self.layer_breakdown:
            lines.append("")
            lines.append("Top 5 Layers by Memory:")
            sorted_layers = sorted(
                self.layer_breakdown,
                key=lambda x: x.total_memory_mb,
                reverse=True
            )[:5]
            for layer in sorted_layers:
                lines.append(f"  {layer.name}: {layer.total_memory_mb:.2f} MB")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path: Union[str, Path]) -> None:
        """Save profile to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ForwardPassProfile:
    """Memory profile for forward pass.

    Attributes:
        input_memory_mb: Memory for input tensors.
        output_memory_mb: Memory for output tensors.
        intermediate_memory_mb: Memory for intermediate activations.
        peak_memory_mb: Peak memory during forward pass.
        memory_timeline: Memory at each step (if tracked).
    """
    input_memory_mb: float
    output_memory_mb: float
    intermediate_memory_mb: float
    peak_memory_mb: float
    memory_timeline: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BackwardPassProfile:
    """Memory profile for backward pass.

    Attributes:
        grad_memory_mb: Memory for gradients.
        retained_graph_mb: Memory for computation graph.
        peak_memory_mb: Peak memory during backward pass.
        memory_before_mb: Memory before backward.
        memory_after_mb: Memory after backward.
    """
    grad_memory_mb: float
    retained_graph_mb: float
    peak_memory_mb: float
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BottleneckInfo:
    """Information about a memory bottleneck.

    Attributes:
        layer_name: Name of the bottleneck layer.
        layer_type: Type of the layer (e.g., "Linear", "Conv1d").
        memory_mb: Memory usage of this layer.
        percentage: Percentage of total memory.
        suggestion: Optimization suggestion.
        params: Number of parameters in layer.
    """
    layer_name: str
    layer_type: str
    memory_mb: float
    percentage: float
    suggestion: str = ""
    params: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Memory Benchmark
# =============================================================================


class MemoryBenchmark:
    """Benchmark suite for memory usage.

    Profiles memory usage of neural network models including:
    - Parameter memory
    - Activation memory
    - Gradient memory
    - Peak memory during training

    Example:
        >>> benchmark = MemoryBenchmark()
        >>> profile = benchmark.profile_model(model, (2, 64, 48))
        >>> print(profile.summary())
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float32",
        verbose: bool = True,
    ):
        """Initialize memory benchmark.

        Args:
            device: Device to profile on ("cuda" or "cpu").
            dtype: Data type for profiling.
            verbose: Whether to print progress.
        """
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            self.device = "cpu"

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float32)

    def _get_device(self) -> torch.device:
        """Get torch device."""
        return torch.device(self.device)

    def _get_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    def _get_peak_memory_mb(self) -> float:
        """Get peak GPU memory usage in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

    def _reset_memory_stats(self):
        """Reset memory statistics."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()

    def _tensor_memory_mb(self, tensor: torch.Tensor) -> float:
        """Calculate memory of a tensor in MB."""
        return tensor.numel() * tensor.element_size() / (1024 * 1024)

    def profile_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        include_backward: bool = True,
        optimizer_type: str = "adam",
    ) -> MemoryProfile:
        """Profile memory usage of a model.

        Args:
            model: The model to profile.
            input_shape: Shape of input tensor (including batch).
            include_backward: Whether to profile backward pass.
            optimizer_type: Type of optimizer for memory estimation.

        Returns:
            MemoryProfile with detailed breakdown.
        """
        device = self._get_device()
        model = model.to(device)

        if self.dtype != "float32":
            model = model.to(self.torch_dtype)

        self._reset_memory_stats()

        # Profile parameters
        layer_breakdown = self._profile_parameters(model)
        param_memory_mb = sum(l.param_memory_mb for l in layer_breakdown)

        # Generate input
        input_tensor = torch.randn(input_shape, device=device, dtype=self.torch_dtype)
        input_memory_mb = self._tensor_memory_mb(input_tensor)

        # Profile forward pass
        self._reset_memory_stats()
        memory_before = self._get_memory_mb()

        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]

        memory_after_forward = self._get_memory_mb()
        activation_memory_mb = memory_after_forward - memory_before - param_memory_mb

        # Profile backward pass
        grad_memory_mb = 0.0
        if include_backward:
            self._reset_memory_stats()
            model.train()

            input_tensor = torch.randn(
                input_shape, device=device, dtype=self.torch_dtype, requires_grad=True
            )

            memory_before_backward = self._get_memory_mb()
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]

            # Create dummy loss
            loss = output.sum()
            loss.backward()

            memory_after_backward = self._get_memory_mb()
            grad_memory_mb = memory_after_backward - memory_before_backward - param_memory_mb

        # Estimate optimizer memory
        optimizer_memory_mb = self._estimate_optimizer_memory(
            model, optimizer_type
        )

        # Total and peak
        total_memory_mb = (
            param_memory_mb +
            activation_memory_mb +
            grad_memory_mb +
            optimizer_memory_mb
        )
        peak_memory_mb = self._get_peak_memory_mb()

        return MemoryProfile(
            model_name=model.__class__.__name__,
            total_memory_mb=total_memory_mb,
            param_memory_mb=param_memory_mb,
            grad_memory_mb=grad_memory_mb,
            activation_memory_mb=activation_memory_mb,
            optimizer_memory_mb=optimizer_memory_mb,
            peak_memory_mb=peak_memory_mb,
            layer_breakdown=layer_breakdown,
            input_shape=input_shape,
            batch_size=input_shape[0],
            device=self.device,
            dtype=self.dtype,
        )

    def _profile_parameters(self, model: nn.Module) -> List[LayerMemory]:
        """Profile parameter memory for each layer."""
        layer_memories = []

        for name, module in model.named_modules():
            params = list(module.parameters(recurse=False))
            if not params:
                continue

            param_memory = sum(
                p.numel() * p.element_size() for p in params
            ) / (1024 * 1024)

            num_params = sum(p.numel() for p in params)
            dtype = str(params[0].dtype) if params else "unknown"

            layer_memories.append(LayerMemory(
                name=name or "root",
                param_memory_mb=param_memory,
                num_params=num_params,
                dtype=dtype,
            ))

        return layer_memories

    def _estimate_optimizer_memory(
        self,
        model: nn.Module,
        optimizer_type: str,
    ) -> float:
        """Estimate optimizer state memory.

        Args:
            model: The model.
            optimizer_type: Type of optimizer.

        Returns:
            Estimated memory in MB.
        """
        param_count = sum(p.numel() for p in model.parameters())
        bytes_per_param = 4  # Assuming float32

        # Optimizer state multipliers
        multipliers = {
            "sgd": 0,           # No state
            "sgd_momentum": 1,  # One momentum buffer
            "adam": 2,          # Two moment buffers
            "adamw": 2,
            "rmsprop": 1,
            "adagrad": 1,
        }

        multiplier = multipliers.get(optimizer_type.lower(), 2)
        return param_count * bytes_per_param * multiplier / (1024 * 1024)

    def profile_forward_pass(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
    ) -> ForwardPassProfile:
        """Profile memory during forward pass.

        Args:
            model: The model.
            input_tensor: Input tensor.

        Returns:
            ForwardPassProfile with memory details.
        """
        device = self._get_device()
        model = model.to(device)
        model.eval()

        input_tensor = input_tensor.to(device)
        if self.dtype != "float32":
            input_tensor = input_tensor.to(self.torch_dtype)
            model = model.to(self.torch_dtype)

        self._reset_memory_stats()

        input_memory = self._tensor_memory_mb(input_tensor)
        memory_before = self._get_memory_mb()

        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]

        memory_after = self._get_memory_mb()
        peak_memory = self._get_peak_memory_mb()

        output_memory = self._tensor_memory_mb(output)
        intermediate_memory = memory_after - memory_before - output_memory

        return ForwardPassProfile(
            input_memory_mb=input_memory,
            output_memory_mb=output_memory,
            intermediate_memory_mb=max(0, intermediate_memory),
            peak_memory_mb=peak_memory,
        )

    def profile_backward_pass(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        loss_fn: Optional[Callable] = None,
    ) -> BackwardPassProfile:
        """Profile memory during backward pass.

        Args:
            model: The model.
            input_tensor: Input tensor.
            loss_fn: Loss function. Uses sum() if None.

        Returns:
            BackwardPassProfile with memory details.
        """
        device = self._get_device()
        model = model.to(device)
        model.train()

        input_tensor = input_tensor.to(device).requires_grad_(True)
        if self.dtype != "float32":
            input_tensor = input_tensor.to(self.torch_dtype)
            model = model.to(self.torch_dtype)

        self._reset_memory_stats()

        # Forward pass
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        if loss_fn is None:
            loss = output.sum()
        else:
            loss = loss_fn(output)

        memory_before = self._get_memory_mb()

        # Backward pass
        loss.backward()

        memory_after = self._get_memory_mb()
        peak_memory = self._get_peak_memory_mb()

        # Estimate gradient memory
        grad_memory = sum(
            p.grad.numel() * p.grad.element_size()
            for p in model.parameters()
            if p.grad is not None
        ) / (1024 * 1024)

        retained_graph = memory_after - memory_before - grad_memory

        return BackwardPassProfile(
            grad_memory_mb=grad_memory,
            retained_graph_mb=max(0, retained_graph),
            peak_memory_mb=peak_memory,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
        )

    def find_memory_bottleneck(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        top_k: int = 5,
    ) -> List[BottleneckInfo]:
        """Find memory bottlenecks in model.

        Identifies layers that use the most memory and provides
        optimization suggestions.

        Args:
            model: The model to analyze.
            input_shape: Shape of input tensor.
            top_k: Number of top bottlenecks to return.

        Returns:
            List of BottleneckInfo for top memory consumers.
        """
        profile = self.profile_model(model, input_shape, include_backward=False)

        # Sort layers by memory
        sorted_layers = sorted(
            profile.layer_breakdown,
            key=lambda x: x.total_memory_mb,
            reverse=True
        )[:top_k]

        total_memory = profile.param_memory_mb

        bottlenecks = []
        for layer in sorted_layers:
            # Determine layer type
            layer_type = "Unknown"
            for name, module in model.named_modules():
                if name == layer.name:
                    layer_type = module.__class__.__name__
                    break

            # Generate suggestion
            suggestion = self._generate_suggestion(layer_type, layer)

            bottlenecks.append(BottleneckInfo(
                layer_name=layer.name,
                layer_type=layer_type,
                memory_mb=layer.total_memory_mb,
                percentage=100 * layer.total_memory_mb / max(total_memory, 1e-6),
                suggestion=suggestion,
                params=layer.num_params,
            ))

        return bottlenecks

    def _generate_suggestion(
        self,
        layer_type: str,
        layer: LayerMemory,
    ) -> str:
        """Generate optimization suggestion for a layer."""
        suggestions = {
            "Embedding": "Consider using lower embedding dimension or weight tying",
            "Linear": "Consider using LoRA or weight sharing",
            "Conv1d": "Consider reducing kernel size or channels",
            "Conv2d": "Consider depthwise separable convolutions",
            "MultiheadAttention": "Consider using flash attention or linear attention",
            "LayerNorm": "Consider using RMSNorm for efficiency",
            "BatchNorm1d": "Consider fusing with adjacent linear layers",
            "BatchNorm2d": "Consider fusing with adjacent conv layers",
        }

        default = "Consider gradient checkpointing or mixed precision"
        return suggestions.get(layer_type, default)

    def estimate_batch_size(
        self,
        model: nn.Module,
        seq_length: int,
        available_memory_gb: float,
        safety_factor: float = 0.8,
    ) -> int:
        """Estimate maximum batch size for given memory.

        Args:
            model: The model.
            seq_length: Sequence length.
            available_memory_gb: Available GPU memory in GB.
            safety_factor: Fraction of memory to use (default 0.8).

        Returns:
            Estimated maximum batch size.
        """
        available_mb = available_memory_gb * 1024 * safety_factor

        # Profile with batch size 1
        model_config = getattr(model, "config", None)
        if model_config is not None:
            # Grid-based model
            height = getattr(model_config, "grid_height", seq_length)
            width = getattr(model_config, "grid_width", 48)
            input_shape = (1, height, width)
        else:
            input_shape = (1, seq_length)

        try:
            profile = self.profile_model(
                model, input_shape, include_backward=True
            )
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not profile model: {e}")
            return 1

        # Memory per sample (excluding fixed costs)
        memory_per_sample = (
            profile.activation_memory_mb +
            profile.grad_memory_mb / 1  # Grad scales with batch
        )

        # Fixed memory (parameters + optimizer)
        fixed_memory = profile.param_memory_mb + profile.optimizer_memory_mb

        # Estimate batch size
        usable_memory = available_mb - fixed_memory
        if usable_memory <= 0:
            return 1

        estimated_batch = int(usable_memory / max(memory_per_sample, 1))
        return max(1, estimated_batch)

    def compare_memory_usage(
        self,
        models: Dict[str, nn.Module],
        input_shape: Tuple[int, ...],
    ) -> Dict[str, MemoryProfile]:
        """Compare memory usage of multiple models.

        Args:
            models: Dictionary of model name -> model.
            input_shape: Input shape for profiling.

        Returns:
            Dictionary of model name -> MemoryProfile.
        """
        profiles = {}

        for name, model in models.items():
            if self.verbose:
                print(f"Profiling {name}...")

            try:
                profile = self.profile_model(model, input_shape)
                profiles[name] = profile
            except Exception as e:
                if self.verbose:
                    print(f"  Error profiling {name}: {e}")

        return profiles

    def generate_report(
        self,
        profiles: Dict[str, MemoryProfile],
        output_path: Union[str, Path],
    ) -> None:
        """Generate a memory usage report.

        Args:
            profiles: Dictionary of model name -> profile.
            output_path: Path to save report.
        """
        output_path = Path(output_path)

        # Generate markdown report
        lines = [
            "# Memory Usage Report",
            "",
            "## Summary",
            "",
            "| Model | Params (MB) | Activations (MB) | Gradients (MB) | Total (MB) |",
            "|-------|-------------|------------------|----------------|------------|",
        ]

        for name, profile in profiles.items():
            lines.append(
                f"| {name} | "
                f"{profile.param_memory_mb:.1f} | "
                f"{profile.activation_memory_mb:.1f} | "
                f"{profile.grad_memory_mb:.1f} | "
                f"{profile.total_memory_mb:.1f} |"
            )

        lines.append("")
        lines.append("## Detailed Profiles")
        lines.append("")

        for name, profile in profiles.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append("```")
            lines.append(profile.summary())
            lines.append("```")
            lines.append("")

        with open(output_path.with_suffix(".md"), "w") as f:
            f.write("\n".join(lines))

        # Save raw data as JSON
        data = {
            name: profile.to_dict()
            for name, profile in profiles.items()
        }
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Gradient Checkpointing Helpers
# =============================================================================


def estimate_checkpoint_savings(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_checkpoints: int,
) -> Dict[str, float]:
    """Estimate memory savings from gradient checkpointing.

    Args:
        model: The model.
        input_shape: Input shape.
        num_checkpoints: Number of checkpoint segments.

    Returns:
        Dictionary with memory estimates.
    """
    benchmark = MemoryBenchmark(verbose=False)

    # Profile without checkpointing
    try:
        profile_no_ckpt = benchmark.profile_model(
            model, input_shape, include_backward=True
        )
    except Exception:
        return {"error": "Could not profile model"}

    # Estimate with checkpointing
    # Activation memory roughly scales as 1/num_checkpoints
    # But recomputation adds overhead
    activation_with_ckpt = (
        profile_no_ckpt.activation_memory_mb / num_checkpoints
    )

    # Extra forward pass time overhead (not memory, but relevant)
    recompute_overhead_factor = 1.0 + (num_checkpoints - 1) / num_checkpoints

    savings = profile_no_ckpt.activation_memory_mb - activation_with_ckpt

    return {
        "original_activation_mb": profile_no_ckpt.activation_memory_mb,
        "estimated_activation_mb": activation_with_ckpt,
        "savings_mb": savings,
        "savings_percent": 100 * savings / max(profile_no_ckpt.activation_memory_mb, 1),
        "recompute_overhead_factor": recompute_overhead_factor,
    }


# =============================================================================
# CLI and Testing
# =============================================================================


def run_sample_benchmark():
    """Run a sample memory benchmark for testing."""
    print("=" * 60)
    print("Memory Benchmark - Sample Run")
    print("=" * 60)

    # Create a simple mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)
            self.layers = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1000),
            )

        def forward(self, x):
            x = self.embedding(x)
            x = x.mean(dim=1)
            return self.layers(x)

    benchmark = MemoryBenchmark(device="cpu", verbose=True)
    model = MockModel()

    print("\nProfiling model...")
    profile = benchmark.profile_model(model, (4, 64))
    print(profile.summary())

    print("\nFinding bottlenecks...")
    bottlenecks = benchmark.find_memory_bottleneck(model, (4, 64))
    for b in bottlenecks:
        print(f"  {b.layer_name} ({b.layer_type}): {b.memory_mb:.2f} MB ({b.percentage:.1f}%)")
        print(f"    Suggestion: {b.suggestion}")

    print("\nEstimating max batch size for 8GB GPU...")
    max_batch = benchmark.estimate_batch_size(model, 64, 8.0)
    print(f"  Estimated max batch size: {max_batch}")


if __name__ == "__main__":
    run_sample_benchmark()
