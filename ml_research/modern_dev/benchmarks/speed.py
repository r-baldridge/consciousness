"""
Speed Benchmarks for ML Research Models (INTEG-003)

This module provides comprehensive speed benchmarks for measuring:
- Inference latency
- Throughput (tokens per second)
- Scaling behavior across batch sizes and sequence lengths
- Model comparison

Key Components:
    - SpeedBenchmark: Main benchmark runner
    - SpeedBenchmarkConfig: Configuration for benchmark parameters
    - ComparisonReport: Compare multiple models

Usage:
    from consciousness.ml_research.modern_dev.benchmarks import SpeedBenchmark

    benchmark = SpeedBenchmark()
    results = benchmark.benchmark_pipeline(pipeline)
    benchmark.plot_results(results, "speed_analysis.png")
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SpeedBenchmarkConfig:
    """Configuration for speed benchmarks.

    Attributes:
        batch_sizes: List of batch sizes to test.
        sequence_lengths: List of sequence lengths to test.
        warmup_iterations: Number of warmup iterations before timing.
        benchmark_iterations: Number of timed iterations.
        device: Device to run benchmarks on ("cuda" or "cpu").
        dtype: Data type for model (float32, float16, bfloat16).
        use_amp: Whether to use automatic mixed precision.
        sync_cuda: Whether to synchronize CUDA before timing.
        gc_collect: Whether to run garbage collection between tests.
    """
    batch_sizes: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32]
    )
    sequence_lengths: List[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 1024]
    )
    warmup_iterations: int = 5
    benchmark_iterations: int = 20
    device: str = "cuda"
    dtype: str = "float32"
    use_amp: bool = False
    sync_cuda: bool = True
    gc_collect: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.benchmark_iterations < 1:
            raise ValueError("benchmark_iterations must be at least 1")

    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass
class SpeedResult:
    """Speed benchmark result for a single configuration.

    Attributes:
        model_name: Name of the benchmarked model.
        batch_size: Batch size used.
        seq_length: Sequence length used.
        latency_ms: Average latency in milliseconds.
        latency_std_ms: Standard deviation of latency.
        throughput_tokens_per_sec: Tokens processed per second.
        throughput_samples_per_sec: Samples processed per second.
        memory_mb: Peak GPU memory usage in MB.
        num_iterations: Number of benchmark iterations.
        warmup_iterations: Number of warmup iterations.
        device: Device used.
        dtype: Data type used.
        metadata: Additional metadata.
    """
    model_name: str
    batch_size: int
    seq_length: int
    latency_ms: float
    latency_std_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    throughput_samples_per_sec: float = 0.0
    memory_mb: float = 0.0
    num_iterations: int = 20
    warmup_iterations: int = 5
    device: str = "cuda"
    dtype: str = "float32"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived metrics."""
        if self.throughput_tokens_per_sec == 0.0 and self.latency_ms > 0:
            tokens_per_sample = self.batch_size * self.seq_length
            samples_per_sec = 1000.0 / self.latency_ms
            self.throughput_samples_per_sec = samples_per_sec
            self.throughput_tokens_per_sec = tokens_per_sample * samples_per_sec

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeedResult":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ComparisonReport:
    """Report comparing multiple models.

    Attributes:
        model_names: List of model names compared.
        results: Dictionary of model name -> list of results.
        summary: Summary statistics.
        winner_by_latency: Model with best latency.
        winner_by_throughput: Model with best throughput.
        winner_by_memory: Model with lowest memory.
        timestamp: Report generation timestamp.
    """
    model_names: List[str]
    results: Dict[str, List[SpeedResult]]
    summary: Dict[str, Dict[str, float]]
    winner_by_latency: str = ""
    winner_by_throughput: str = ""
    winner_by_memory: str = ""
    timestamp: str = ""

    def __post_init__(self):
        """Set timestamp and determine winners."""
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()

        if self.results:
            self._determine_winners()

    def _determine_winners(self):
        """Determine winners in each category."""
        if not self.results:
            return

        avg_latencies = {}
        avg_throughputs = {}
        avg_memories = {}

        for name, results in self.results.items():
            if results:
                avg_latencies[name] = sum(r.latency_ms for r in results) / len(results)
                avg_throughputs[name] = sum(r.throughput_tokens_per_sec for r in results) / len(results)
                avg_memories[name] = sum(r.memory_mb for r in results) / len(results)

        if avg_latencies:
            self.winner_by_latency = min(avg_latencies, key=avg_latencies.get)
        if avg_throughputs:
            self.winner_by_throughput = max(avg_throughputs, key=avg_throughputs.get)
        if avg_memories:
            self.winner_by_memory = min(avg_memories, key=avg_memories.get)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Speed Benchmark Comparison Report",
            "",
            f"Generated: {self.timestamp}",
            "",
            "## Summary",
            "",
            f"- **Best Latency**: {self.winner_by_latency}",
            f"- **Best Throughput**: {self.winner_by_throughput}",
            f"- **Lowest Memory**: {self.winner_by_memory}",
            "",
            "## Detailed Results",
            "",
        ]

        for name in self.model_names:
            lines.append(f"### {name}")
            lines.append("")
            if name in self.summary:
                s = self.summary[name]
                lines.append(f"- Avg Latency: {s.get('avg_latency_ms', 0):.2f} ms")
                lines.append(f"- Avg Throughput: {s.get('avg_throughput_tps', 0):.0f} tokens/sec")
                lines.append(f"- Avg Memory: {s.get('avg_memory_mb', 0):.1f} MB")
            lines.append("")

        return "\n".join(lines)

    def save(self, path: Union[str, Path], format: str = "json") -> None:
        """Save report to file.

        Args:
            path: Output file path.
            format: Output format ("json" or "markdown").
        """
        path = Path(path)

        if format == "json":
            data = {
                "model_names": self.model_names,
                "results": {
                    name: [r.to_dict() for r in results]
                    for name, results in self.results.items()
                },
                "summary": self.summary,
                "winner_by_latency": self.winner_by_latency,
                "winner_by_throughput": self.winner_by_throughput,
                "winner_by_memory": self.winner_by_memory,
                "timestamp": self.timestamp,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif format == "markdown":
            with open(path, "w") as f:
                f.write(self.to_markdown())
        else:
            raise ValueError(f"Unknown format: {format}")


# =============================================================================
# Speed Benchmark
# =============================================================================


class SpeedBenchmark:
    """Benchmark suite for inference speed.

    Measures latency, throughput, and memory usage across various
    batch sizes and sequence lengths.

    Attributes:
        config: Benchmark configuration.

    Example:
        >>> benchmark = SpeedBenchmark()
        >>> results = benchmark.benchmark_model(model, "my_model")
        >>> for r in results:
        ...     print(f"BS={r.batch_size}, Seq={r.seq_length}: {r.latency_ms:.2f}ms")
    """

    def __init__(self, config: Optional[SpeedBenchmarkConfig] = None):
        """Initialize benchmark.

        Args:
            config: Benchmark configuration. Uses defaults if None.
        """
        self.config = config or SpeedBenchmarkConfig()
        self._check_device()

    def _check_device(self):
        """Check if configured device is available."""
        if self.config.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.config.device = "cpu"

    def _get_device(self) -> torch.device:
        """Get torch device."""
        return torch.device(self.config.device)

    def _sync_device(self):
        """Synchronize device if needed."""
        if self.config.sync_cuda and self.config.device == "cuda":
            torch.cuda.synchronize()

    def _get_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

    def _reset_memory_stats(self):
        """Reset memory statistics."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def benchmark_model(
        self,
        model: nn.Module,
        name: str,
        input_generator: Optional[Callable[[int, int], torch.Tensor]] = None,
        batch_sizes: Optional[List[int]] = None,
        sequence_lengths: Optional[List[int]] = None,
    ) -> List[SpeedResult]:
        """Benchmark a single model across configurations.

        Args:
            model: The model to benchmark.
            name: Name for the model.
            input_generator: Optional function to generate inputs.
                             Signature: (batch_size, seq_length) -> tensor.
            batch_sizes: Override batch sizes from config.
            sequence_lengths: Override sequence lengths from config.

        Returns:
            List of SpeedResult for each configuration.
        """
        device = self._get_device()
        model = model.to(device)
        model.eval()

        if self.config.dtype != "float32":
            model = model.to(self.config.torch_dtype)

        batch_sizes = batch_sizes or self.config.batch_sizes
        sequence_lengths = sequence_lengths or self.config.sequence_lengths

        # Default input generator
        if input_generator is None:
            vocab_size = getattr(model, "config", None)
            if vocab_size is not None:
                vocab_size = getattr(vocab_size, "vocab_size", 1000)
            else:
                vocab_size = 1000

            def input_generator(bs: int, seq_len: int) -> torch.Tensor:
                return torch.randint(
                    0, vocab_size, (bs, seq_len), device=device
                )

        results = []

        for batch_size in batch_sizes:
            for seq_length in sequence_lengths:
                try:
                    result = self._benchmark_single_config(
                        model=model,
                        name=name,
                        batch_size=batch_size,
                        seq_length=seq_length,
                        input_generator=input_generator,
                    )
                    results.append(result)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  OOM at BS={batch_size}, Seq={seq_length}")
                        # Clear memory and continue
                        if self.config.device == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    raise

                if self.config.gc_collect:
                    gc.collect()
                    if self.config.device == "cuda":
                        torch.cuda.empty_cache()

        return results

    def _benchmark_single_config(
        self,
        model: nn.Module,
        name: str,
        batch_size: int,
        seq_length: int,
        input_generator: Callable[[int, int], torch.Tensor],
    ) -> SpeedResult:
        """Benchmark a single configuration."""
        device = self._get_device()

        # Generate input
        input_tensor = input_generator(batch_size, seq_length)
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor, device=device)
        if input_tensor.device != device:
            input_tensor = input_tensor.to(device)

        self._reset_memory_stats()

        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                if self.config.use_amp and self.config.device == "cuda":
                    with torch.cuda.amp.autocast():
                        _ = model(input_tensor)
                else:
                    _ = model(input_tensor)
                self._sync_device()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(self.config.benchmark_iterations):
                self._sync_device()
                start = time.perf_counter()

                if self.config.use_amp and self.config.device == "cuda":
                    with torch.cuda.amp.autocast():
                        _ = model(input_tensor)
                else:
                    _ = model(input_tensor)

                self._sync_device()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        # Compute statistics
        avg_latency = sum(latencies) / len(latencies)
        std_latency = (
            sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)
        ) ** 0.5
        memory_mb = self._get_memory_mb()

        return SpeedResult(
            model_name=name,
            batch_size=batch_size,
            seq_length=seq_length,
            latency_ms=avg_latency,
            latency_std_ms=std_latency,
            memory_mb=memory_mb,
            num_iterations=self.config.benchmark_iterations,
            warmup_iterations=self.config.warmup_iterations,
            device=self.config.device,
            dtype=self.config.dtype,
        )

    def benchmark_trm(
        self,
        config: Optional[Any] = None,
        name: str = "TRM",
    ) -> List[SpeedResult]:
        """Benchmark TRM model.

        Args:
            config: TRM configuration. Uses default if None.
            name: Name for the model.

        Returns:
            List of SpeedResult.
        """
        try:
            from consciousness.ml_research.modern_dev.trm.src.model import (
                CodeRepairTRM,
                CodeRepairConfig,
            )
        except ImportError:
            raise ImportError("TRM model not found. Install trm package.")

        if config is None:
            config = CodeRepairConfig.for_code_repair_tiny()

        model = CodeRepairTRM(config)

        # TRM uses 2D grid input
        def input_generator(batch_size: int, seq_length: int) -> torch.Tensor:
            # Approximate grid dimensions
            height = min(seq_length, config.grid_height)
            width = min(seq_length // height + 1, config.grid_width)
            return torch.randint(
                0, config.vocab_size,
                (batch_size, height, width),
                device=self._get_device(),
            )

        # Use grid-appropriate dimensions
        grid_sizes = [16, 32, 64]  # Equivalent sequence lengths
        results = self.benchmark_model(
            model,
            name,
            input_generator=input_generator,
            sequence_lengths=grid_sizes,
        )

        return results

    def benchmark_mamba(
        self,
        config: Optional[Any] = None,
        name: str = "Mamba",
    ) -> List[SpeedResult]:
        """Benchmark Mamba model.

        Args:
            config: Mamba configuration. Uses default if None.
            name: Name for the model.

        Returns:
            List of SpeedResult.
        """
        try:
            from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
                MambaLM,
                MambaLMConfig,
            )
        except ImportError:
            raise ImportError("Mamba model not found. Install mamba package.")

        if config is None:
            config = MambaLMConfig(
                vocab_size=1000,
                d_model=64,
                n_layers=2,
            )

        model = MambaLM(config)

        def input_generator(batch_size: int, seq_length: int) -> torch.Tensor:
            return torch.randint(
                0, config.vocab_size,
                (batch_size, seq_length),
                device=self._get_device(),
            )

        return self.benchmark_model(model, name, input_generator=input_generator)

    def benchmark_pipeline(
        self,
        pipeline: Any,
        name: str = "Pipeline",
        sample_inputs: Optional[List[str]] = None,
    ) -> List[SpeedResult]:
        """Benchmark a code repair pipeline.

        Args:
            pipeline: Pipeline with repair() method.
            name: Name for the pipeline.
            sample_inputs: Sample code strings to use as inputs.

        Returns:
            List of SpeedResult (one per sample input size).
        """
        if sample_inputs is None:
            # Generate sample inputs of varying sizes
            sample_inputs = [
                "def f(x): return x",
                "def add(a, b):\n    return a + b\n",
                "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n",
                "def binary_search(arr, target):\n" + "    " * 10 + "pass\n" * 10,
            ]

        results = []
        device = self._get_device()

        for i, sample in enumerate(sample_inputs):
            sample_len = len(sample)

            # Warmup
            for _ in range(self.config.warmup_iterations):
                try:
                    _ = pipeline.repair(sample)
                except Exception:
                    pass

            # Benchmark
            latencies = []
            for _ in range(self.config.benchmark_iterations):
                self._sync_device()
                start = time.perf_counter()

                try:
                    _ = pipeline.repair(sample)
                except Exception:
                    pass

                self._sync_device()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)

            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            std_latency = (
                sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)
            ) ** 0.5 if latencies else 0

            results.append(SpeedResult(
                model_name=name,
                batch_size=1,
                seq_length=sample_len,
                latency_ms=avg_latency,
                latency_std_ms=std_latency,
                memory_mb=self._get_memory_mb(),
                num_iterations=self.config.benchmark_iterations,
                warmup_iterations=self.config.warmup_iterations,
                device=self.config.device,
                dtype=self.config.dtype,
                metadata={"sample_index": i},
            ))

        return results

    def compare_models(
        self,
        results: Dict[str, List[SpeedResult]],
    ) -> ComparisonReport:
        """Compare multiple models.

        Args:
            results: Dictionary of model name -> list of results.

        Returns:
            ComparisonReport with comparison analysis.
        """
        model_names = list(results.keys())

        # Compute summary statistics for each model
        summary = {}
        for name, model_results in results.items():
            if model_results:
                summary[name] = {
                    "avg_latency_ms": sum(r.latency_ms for r in model_results) / len(model_results),
                    "min_latency_ms": min(r.latency_ms for r in model_results),
                    "max_latency_ms": max(r.latency_ms for r in model_results),
                    "avg_throughput_tps": sum(r.throughput_tokens_per_sec for r in model_results) / len(model_results),
                    "avg_memory_mb": sum(r.memory_mb for r in model_results) / len(model_results),
                    "num_configs": len(model_results),
                }

        return ComparisonReport(
            model_names=model_names,
            results=results,
            summary=summary,
        )

    def plot_results(
        self,
        results: List[SpeedResult],
        output_path: Union[str, Path],
        plot_type: str = "latency",
    ) -> None:
        """Generate performance plots.

        Args:
            results: List of benchmark results.
            output_path: Path to save the plot.
            plot_type: Type of plot ("latency", "throughput", "memory", "scaling").
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available. Saving results as JSON instead.")
            json_path = Path(output_path).with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump([r.to_dict() for r in results], f, indent=2)
            return

        output_path = Path(output_path)

        if plot_type == "latency":
            self._plot_latency(results, output_path)
        elif plot_type == "throughput":
            self._plot_throughput(results, output_path)
        elif plot_type == "memory":
            self._plot_memory(results, output_path)
        elif plot_type == "scaling":
            self._plot_scaling(results, output_path)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def _plot_latency(
        self,
        results: List[SpeedResult],
        output_path: Path,
    ) -> None:
        """Plot latency vs sequence length."""
        import matplotlib.pyplot as plt

        # Group by batch size
        by_batch = {}
        for r in results:
            if r.batch_size not in by_batch:
                by_batch[r.batch_size] = []
            by_batch[r.batch_size].append(r)

        plt.figure(figsize=(10, 6))
        for batch_size, batch_results in sorted(by_batch.items()):
            seq_lengths = [r.seq_length for r in batch_results]
            latencies = [r.latency_ms for r in batch_results]
            plt.plot(seq_lengths, latencies, 'o-', label=f'BS={batch_size}')

        plt.xlabel('Sequence Length')
        plt.ylabel('Latency (ms)')
        plt.title('Inference Latency vs Sequence Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_throughput(
        self,
        results: List[SpeedResult],
        output_path: Path,
    ) -> None:
        """Plot throughput vs sequence length."""
        import matplotlib.pyplot as plt

        by_batch = {}
        for r in results:
            if r.batch_size not in by_batch:
                by_batch[r.batch_size] = []
            by_batch[r.batch_size].append(r)

        plt.figure(figsize=(10, 6))
        for batch_size, batch_results in sorted(by_batch.items()):
            seq_lengths = [r.seq_length for r in batch_results]
            throughputs = [r.throughput_tokens_per_sec for r in batch_results]
            plt.plot(seq_lengths, throughputs, 'o-', label=f'BS={batch_size}')

        plt.xlabel('Sequence Length')
        plt.ylabel('Throughput (tokens/sec)')
        plt.title('Throughput vs Sequence Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_memory(
        self,
        results: List[SpeedResult],
        output_path: Path,
    ) -> None:
        """Plot memory usage."""
        import matplotlib.pyplot as plt

        by_batch = {}
        for r in results:
            if r.batch_size not in by_batch:
                by_batch[r.batch_size] = []
            by_batch[r.batch_size].append(r)

        plt.figure(figsize=(10, 6))
        for batch_size, batch_results in sorted(by_batch.items()):
            seq_lengths = [r.seq_length for r in batch_results]
            memories = [r.memory_mb for r in batch_results]
            plt.plot(seq_lengths, memories, 'o-', label=f'BS={batch_size}')

        plt.xlabel('Sequence Length')
        plt.ylabel('Peak Memory (MB)')
        plt.title('Memory Usage vs Sequence Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_scaling(
        self,
        results: List[SpeedResult],
        output_path: Path,
    ) -> None:
        """Plot scaling behavior (latency vs batch size)."""
        import matplotlib.pyplot as plt

        by_seq = {}
        for r in results:
            if r.seq_length not in by_seq:
                by_seq[r.seq_length] = []
            by_seq[r.seq_length].append(r)

        plt.figure(figsize=(10, 6))
        for seq_length, seq_results in sorted(by_seq.items()):
            batch_sizes = [r.batch_size for r in seq_results]
            latencies = [r.latency_ms for r in seq_results]
            plt.plot(batch_sizes, latencies, 'o-', label=f'Seq={seq_length}')

        plt.xlabel('Batch Size')
        plt.ylabel('Latency (ms)')
        plt.title('Batch Scaling: Latency vs Batch Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_report(
        self,
        results: Dict[str, List[SpeedResult]],
        output_dir: Union[str, Path],
    ) -> None:
        """Generate a complete benchmark report with plots.

        Args:
            results: Dictionary of model name -> results.
            output_dir: Directory to save report files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate comparison report
        comparison = self.compare_models(results)
        comparison.save(output_dir / "comparison.json", format="json")
        comparison.save(output_dir / "comparison.md", format="markdown")

        # Generate plots for each model
        for name, model_results in results.items():
            if model_results:
                safe_name = name.replace(" ", "_").lower()
                self.plot_results(
                    model_results,
                    output_dir / f"{safe_name}_latency.png",
                    plot_type="latency"
                )
                self.plot_results(
                    model_results,
                    output_dir / f"{safe_name}_throughput.png",
                    plot_type="throughput"
                )
                self.plot_results(
                    model_results,
                    output_dir / f"{safe_name}_scaling.png",
                    plot_type="scaling"
                )

        # Save raw results
        all_results = {
            name: [r.to_dict() for r in rs]
            for name, rs in results.items()
        }
        with open(output_dir / "raw_results.json", "w") as f:
            json.dump(all_results, f, indent=2)


# =============================================================================
# CLI and Testing
# =============================================================================


def run_sample_benchmark():
    """Run a sample speed benchmark for testing."""
    print("=" * 60)
    print("Speed Benchmark - Sample Run")
    print("=" * 60)

    # Create a simple mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 64)
            self.linear = nn.Linear(64, 1000)

        def forward(self, x):
            x = self.embedding(x)
            x = x.mean(dim=1)
            return self.linear(x)

    config = SpeedBenchmarkConfig(
        batch_sizes=[1, 2, 4],
        sequence_lengths=[32, 64, 128],
        warmup_iterations=2,
        benchmark_iterations=5,
        device="cpu",
    )

    benchmark = SpeedBenchmark(config)
    model = MockModel()

    print("\nBenchmarking model...")
    results = benchmark.benchmark_model(model, "MockModel")

    print("\nResults:")
    print("-" * 60)
    for r in results:
        print(f"BS={r.batch_size:2d}, Seq={r.seq_length:4d}: "
              f"{r.latency_ms:8.2f}ms, "
              f"{r.throughput_tokens_per_sec:10.0f} tok/s")

    # Test comparison
    print("\nModel Comparison:")
    comparison = benchmark.compare_models({"MockModel": results})
    print(f"Best Latency: {comparison.winner_by_latency}")
    print(f"Best Throughput: {comparison.winner_by_throughput}")


if __name__ == "__main__":
    run_sample_benchmark()
