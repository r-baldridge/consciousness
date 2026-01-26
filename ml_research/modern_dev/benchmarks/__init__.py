"""
Benchmarking Suite for ML Research Code Repair Models (INTEG-003)

This module provides comprehensive benchmarks for:
- Code repair accuracy evaluation
- Inference speed profiling
- Memory usage analysis
- Baseline comparisons

Key Components:
    - CodeRepairBenchmark: Accuracy benchmarks for code repair models
    - SpeedBenchmark: Latency and throughput measurements
    - MemoryBenchmark: Memory profiling and bottleneck detection
    - BaselineComparison: Compare against naive baselines

Usage:
    from consciousness.ml_research.modern_dev.benchmarks import (
        CodeRepairBenchmark,
        SpeedBenchmark,
        MemoryBenchmark,
        BaselineComparison,
    )

    # Run accuracy benchmark
    benchmark = CodeRepairBenchmark(pipeline)
    results = benchmark.run(max_samples=100)
    print(f"Accuracy: {results.accuracy:.2%}")

    # Run speed benchmark
    speed = SpeedBenchmark()
    speed_results = speed.benchmark_pipeline(pipeline)
    speed.plot_results(speed_results, "speed_comparison.png")

    # Run memory benchmark
    memory = MemoryBenchmark()
    profile = memory.profile_model(model, input_shape=(2, 64, 48))
    print(f"Peak memory: {profile.peak_memory_mb:.1f} MB")
"""

from __future__ import annotations

from .code_repair import (
    BenchmarkDataset,
    BenchmarkResult,
    CategoryResult,
    CodeRepairBenchmark,
    CodeRepairSample,
    SampleResult,
    BaselineComparison,
    ComparisonResult,
)

from .speed import (
    SpeedBenchmarkConfig,
    SpeedResult,
    SpeedBenchmark,
    ComparisonReport,
)

from .memory import (
    MemoryBenchmark,
    MemoryProfile,
    ForwardPassProfile,
    BackwardPassProfile,
    BottleneckInfo,
)

__all__ = [
    # Code repair benchmarks
    "BenchmarkDataset",
    "BenchmarkResult",
    "CategoryResult",
    "CodeRepairBenchmark",
    "CodeRepairSample",
    "SampleResult",
    "BaselineComparison",
    "ComparisonResult",
    # Speed benchmarks
    "SpeedBenchmarkConfig",
    "SpeedResult",
    "SpeedBenchmark",
    "ComparisonReport",
    # Memory benchmarks
    "MemoryBenchmark",
    "MemoryProfile",
    "ForwardPassProfile",
    "BackwardPassProfile",
    "BottleneckInfo",
]

__version__ = "0.1.0"
