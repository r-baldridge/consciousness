"""
Shared pytest fixtures for ml_research modern_dev test suite.

This module provides reusable fixtures for testing TRM, Mamba, and integration components.
Designed to support >80% coverage target with efficient test execution.

Run tests with: pytest tests/ -v --cov=modern_dev --cov-report=html
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import pytest
import torch
import torch.nn as nn


# =============================================================================
# Device Fixtures
# =============================================================================


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing.

    Uses CUDA if available, otherwise falls back to CPU.
    Tests should work on both CPU and GPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device() -> torch.device:
    """Force CPU device for tests that need consistent behavior."""
    return torch.device("cpu")


# =============================================================================
# Code Repair Data Fixtures
# =============================================================================


@pytest.fixture
def sample_code_repair_data() -> List[Dict[str, str]]:
    """Sample buggy/fixed code pairs for testing.

    Provides a variety of common bug types:
    - Variable name typos
    - Off-by-one errors
    - Missing returns
    - Type errors
    - Logic errors
    """
    return [
        {
            "buggy": "def add(a, b): return a + c",
            "fixed": "def add(a, b): return a + b",
            "bug_type": "variable_typo",
            "difficulty": 0.2,
        },
        {
            "buggy": "def get_last(lst): return lst[len(lst)]",
            "fixed": "def get_last(lst): return lst[len(lst) - 1]",
            "bug_type": "off_by_one",
            "difficulty": 0.4,
        },
        {
            "buggy": "def factorial(n):\n    if n <= 1:\n        return 1\n    factorial(n - 1) * n",
            "fixed": "def factorial(n):\n    if n <= 1:\n        return 1\n    return factorial(n - 1) * n",
            "bug_type": "missing_return",
            "difficulty": 0.3,
        },
        {
            "buggy": "def greet(name):\n    return 'Hello, ' + name + 1",
            "fixed": "def greet(name):\n    return 'Hello, ' + name + '!'",
            "bug_type": "type_error",
            "difficulty": 0.3,
        },
        {
            "buggy": "def is_even(n): return n % 2 == 1",
            "fixed": "def is_even(n): return n % 2 == 0",
            "bug_type": "logic_error",
            "difficulty": 0.2,
        },
        {
            "buggy": "def safe_divide(a, b):\n    return a / b",
            "fixed": "def safe_divide(a, b):\n    if b == 0:\n        return 0\n    return a / b",
            "bug_type": "missing_guard",
            "difficulty": 0.5,
        },
        {
            "buggy": "def find_max(lst):\n    max_val = 0\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val",
            "fixed": "def find_max(lst):\n    if not lst:\n        return None\n    max_val = lst[0]\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val",
            "bug_type": "edge_case",
            "difficulty": 0.6,
        },
        {
            "buggy": "def count_vowels(s):\n    vowels = 'aeiou'\n    count = 0\n    for c in s:\n        if c in vowels:\n            count += 1\n        return count",  # Wrong indentation
            "fixed": "def count_vowels(s):\n    vowels = 'aeiou'\n    count = 0\n    for c in s:\n        if c in vowels:\n            count += 1\n    return count",
            "bug_type": "indentation",
            "difficulty": 0.3,
        },
    ]


@pytest.fixture
def sample_code_tokens() -> Dict[str, torch.Tensor]:
    """Sample tokenized code for model testing.

    Returns token tensors shaped for the 64x48 grid model.
    """
    return {
        "input_ids": torch.randint(0, 1000, (2, 64, 48)),
        "attention_mask": torch.ones(2, 64, 48),
        "labels": torch.randint(0, 1000, (2, 64, 48)),
    }


# =============================================================================
# TRM Model Fixtures
# =============================================================================


@pytest.fixture
def trm_tiny_config():
    """Tiny TRM config for fast testing."""
    from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairConfig

    return CodeRepairConfig(
        grid_height=8,
        grid_width=8,
        vocab_size=256,
        embed_dim=32,
        n_heads=2,
        ffn_dim=64,
        n_blocks=2,
        max_iterations=2,
        min_iterations=1,
        dropout=0.0,
    )


@pytest.fixture
def trm_small_config():
    """Small TRM config for unit tests."""
    from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairConfig

    return CodeRepairConfig(
        grid_height=16,
        grid_width=12,
        vocab_size=1000,
        embed_dim=64,
        n_heads=4,
        ffn_dim=256,
        n_blocks=2,
        max_iterations=4,
        min_iterations=1,
    )


@pytest.fixture
def trm_model(trm_small_config, cpu_device):
    """Small TRM model for testing.

    Uses CPU to ensure consistent behavior across test environments.
    """
    from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairTRM

    model = CodeRepairTRM(trm_small_config)
    model = model.to(cpu_device)
    model.eval()
    return model


@pytest.fixture
def trm_model_training(trm_small_config, cpu_device):
    """TRM model in training mode."""
    from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairTRM

    model = CodeRepairTRM(trm_small_config)
    model = model.to(cpu_device)
    model.train()
    return model


# =============================================================================
# Mamba Model Fixtures
# =============================================================================


@pytest.fixture
def mamba_tiny_config():
    """Tiny Mamba config for fast testing."""
    from consciousness.ml_research.modern_dev.mamba_impl.src.model import MambaConfig

    return MambaConfig(
        d_model=32,
        n_layer=1,
        vocab_size=256,
        d_state=4,
        d_conv=2,
        expand=2,
    )


@pytest.fixture
def mamba_small_config():
    """Small Mamba config for unit tests."""
    from consciousness.ml_research.modern_dev.mamba_impl.src.model import MambaConfig

    return MambaConfig(
        d_model=64,
        n_layer=2,
        vocab_size=1000,
        d_state=8,
        d_conv=2,
        expand=2,
    )


@pytest.fixture
def mamba_model(mamba_small_config, cpu_device):
    """Small Mamba model for testing.

    Uses CPU to ensure consistent behavior across test environments.
    """
    from consciousness.ml_research.modern_dev.mamba_impl.src.model import Mamba

    model = Mamba(mamba_small_config)
    model = model.to(cpu_device)
    model.eval()
    return model


@pytest.fixture
def mamba_model_training(mamba_small_config, cpu_device):
    """Mamba model in training mode."""
    from consciousness.ml_research.modern_dev.mamba_impl.src.model import Mamba

    model = Mamba(mamba_small_config)
    model = model.to(cpu_device)
    model.train()
    return model


# =============================================================================
# Orchestrator Fixtures
# =============================================================================


@pytest.fixture
def orchestrator():
    """Basic orchestrator for testing."""
    from consciousness.ml_research.modern_dev.orchestrator import Orchestrator

    return Orchestrator(device="cpu", default_preset="tiny")


@pytest.fixture
def architecture_loader():
    """Architecture loader for dynamic model loading tests."""
    from consciousness.ml_research.modern_dev.orchestrator import ArchitectureLoader

    return ArchitectureLoader()


# =============================================================================
# Benchmark Fixtures
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    duration_ms: float
    iterations: int
    memory_mb: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def avg_duration_ms(self) -> float:
        """Average duration per iteration."""
        return self.duration_ms / self.iterations if self.iterations > 0 else 0.0


class BenchmarkTimer:
    """Context manager for benchmarking code blocks.

    Usage:
        with benchmark_timer("my_operation") as timer:
            # code to benchmark
            pass
        print(f"Duration: {timer.result.duration_ms}ms")
    """

    def __init__(
        self,
        name: str = "benchmark",
        iterations: int = 1,
        warmup_iterations: int = 0,
    ):
        self.name = name
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.result: Optional[BenchmarkResult] = None
        self._start_time: Optional[float] = None
        self._start_memory: Optional[float] = None

    def __enter__(self) -> "BenchmarkTimer":
        # Record starting memory if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._start_memory = torch.cuda.memory_allocated() / (1024 * 1024)

        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        duration_ms = (time.perf_counter() - self._start_time) * 1000

        memory_mb = None
        if torch.cuda.is_available() and self._start_memory is not None:
            end_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_mb = end_memory - self._start_memory

        self.result = BenchmarkResult(
            name=self.name,
            duration_ms=duration_ms,
            iterations=self.iterations,
            memory_mb=memory_mb,
        )


@pytest.fixture
def benchmark_timer():
    """Factory fixture for creating benchmark timers.

    Usage in tests:
        def test_inference_speed(benchmark_timer):
            with benchmark_timer("inference", iterations=10) as timer:
                for _ in range(10):
                    model(input_ids)

            assert timer.result.avg_duration_ms < 100  # 100ms threshold
    """
    def _create_timer(
        name: str = "benchmark",
        iterations: int = 1,
        warmup_iterations: int = 0,
    ) -> BenchmarkTimer:
        return BenchmarkTimer(name, iterations, warmup_iterations)

    return _create_timer


@contextmanager
def timed_block(name: str = "operation") -> Generator[Dict[str, float], None, None]:
    """Simple context manager for timing blocks of code.

    Yields a dict that will contain 'duration_ms' after the block completes.

    Usage:
        with timed_block("my_operation") as timing:
            # code to time
            pass
        print(f"Took {timing['duration_ms']}ms")
    """
    result: Dict[str, float] = {}
    start = time.perf_counter()

    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["duration_ms"] = (time.perf_counter() - start) * 1000
        result["name"] = name


# =============================================================================
# Test Data Generators
# =============================================================================


@pytest.fixture
def random_input_generator():
    """Factory for generating random model inputs.

    Returns a callable that generates random inputs with specified shapes.
    """
    def _generate(
        batch_size: int = 2,
        seq_len: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        vocab_size: int = 1000,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Generate random input tensors.

        For sequence models (Mamba): provide seq_len
        For grid models (TRM): provide height and width
        """
        if seq_len is not None:
            # Sequence model input
            input_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len), device=device
            )
            attention_mask = torch.ones(batch_size, seq_len, device=device)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        elif height is not None and width is not None:
            # Grid model input
            input_ids = torch.randint(
                0, vocab_size, (batch_size, height, width), device=device
            )
            mask = torch.ones(batch_size, height, width, device=device)
            return {
                "input_ids": input_ids,
                "mask": mask,
            }
        else:
            raise ValueError("Must provide either seq_len or (height, width)")

    return _generate


# =============================================================================
# Test Utilities
# =============================================================================


def assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = "",
) -> None:
    """Assert two tensors are close within tolerance."""
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = (actual - expected).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        raise AssertionError(
            f"{msg}\nTensors not close: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
        )


def assert_no_nan(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert tensor contains no NaN values."""
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        raise AssertionError(f"{name} contains {nan_count} NaN values")


def assert_no_inf(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert tensor contains no Inf values."""
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        raise AssertionError(f"{name} contains {inf_count} Inf values")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# Pytest Markers and Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: marks test as a performance benchmark"
    )
    config.addinivalue_line(
        "markers", "slow: marks test as slow (deselect with '-m not slow')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks test as an integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and environment."""
    # Skip GPU tests if CUDA not available
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
