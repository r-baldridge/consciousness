"""
Shared pytest fixtures for ml_research integration tests.

This module provides:
- Mock backends for testing without PyTorch/GPU
- Small model configurations for fast tests
- Sample inputs/outputs for testing pipelines
- Markers for slow/GPU tests
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

# Handle pytest import - pytest may not be installed in all environments
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create a mock pytest module for basic compatibility
    class MockPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

        class mark:
            @staticmethod
            def slow(func):
                return func
            @staticmethod
            def pytorch(func):
                return func
            @staticmethod
            def gpu(func):
                return func
            @staticmethod
            def skip(reason=""):
                def decorator(func):
                    return func
                return decorator

    pytest = MockPytest()


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

if PYTEST_AVAILABLE:
    def pytest_addoption(parser):
        """Add custom command line options."""
        parser.addoption(
            "--run-slow",
            action="store_true",
            default=False,
            help="Run slow tests that require GPU or large models",
        )
        parser.addoption(
            "--run-pytorch",
            action="store_true",
            default=False,
            help="Run tests that require PyTorch",
        )

    def pytest_configure(config):
        """Register custom markers."""
        config.addinivalue_line(
            "markers", "slow: mark test as slow (requires --run-slow to run)"
        )
        config.addinivalue_line(
            "markers", "pytorch: mark test as requiring PyTorch (requires --run-pytorch to run)"
        )
        config.addinivalue_line(
            "markers", "gpu: mark test as requiring GPU"
        )

    def pytest_collection_modifyitems(config, items):
        """Skip tests based on markers and command line options."""
        # Handle slow tests
        if not config.getoption("--run-slow"):
            skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
            for item in items:
                if "slow" in item.keywords:
                    item.add_marker(skip_slow)

        # Handle PyTorch tests
        if not config.getoption("--run-pytorch"):
            skip_pytorch = pytest.mark.skip(reason="need --run-pytorch option to run")
            for item in items:
                if "pytorch" in item.keywords:
                    item.add_marker(skip_pytorch)

        # Handle GPU tests
        skip_gpu = pytest.mark.skip(reason="GPU tests require --run-slow and CUDA")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# =============================================================================
# MOCK BACKEND
# =============================================================================

@dataclass
class MockModelConfig:
    """Configuration for mock model."""
    d_model: int = 64
    n_layer: int = 2
    vocab_size: int = 1000
    d_state: int = 8
    d_conv: int = 2
    expand: int = 2
    d_inner: int = field(init=False)

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model


class MockTensor:
    """Mock tensor for testing without PyTorch."""

    def __init__(
        self,
        data: Any = None,
        shape: Optional[tuple] = None,
        dtype: str = "float32",
    ):
        self._shape = shape or (1,)
        self._dtype = dtype
        self._data = data

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    def __repr__(self) -> str:
        return f"MockTensor(shape={self._shape}, dtype={self._dtype})"

    def tolist(self) -> List:
        """Convert to list."""
        import random

        def make_nested(shape):
            if len(shape) == 0:
                return random.random()
            return [make_nested(shape[1:]) for _ in range(shape[0])]

        return make_nested(self._shape)

    def numpy(self) -> Any:
        """Convert to numpy array (returns mock)."""
        return self

    def item(self) -> float:
        """Return scalar value."""
        import random
        return random.random()

    @classmethod
    def randn(cls, *shape) -> "MockTensor":
        """Create random tensor."""
        return cls(shape=shape)

    @classmethod
    def zeros(cls, *shape, dtype: str = "float32") -> "MockTensor":
        """Create zeros tensor."""
        return cls(shape=shape, dtype=dtype)

    @classmethod
    def ones(cls, *shape, dtype: str = "float32") -> "MockTensor":
        """Create ones tensor."""
        return cls(shape=shape, dtype=dtype)

    def __getitem__(self, key) -> "MockTensor":
        """Support indexing."""
        return MockTensor(shape=(1,))

    def __mul__(self, other) -> "MockTensor":
        return self

    def __add__(self, other) -> "MockTensor":
        return self

    def __sub__(self, other) -> "MockTensor":
        return self


class MockBackend:
    """
    Mock ML backend for testing without PyTorch/GPU.

    Provides mock implementations for common operations:
    - Model inference
    - Tensor operations
    - Architecture components
    """

    def __init__(self, config: Optional[MockModelConfig] = None):
        self.config = config or MockModelConfig()
        self._call_count = 0
        self._last_input = None
        self._last_output = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs,
    ) -> str:
        """Mock text generation."""
        self._call_count += 1
        self._last_input = prompt

        # Generate deterministic-ish response based on input
        response = f"[Mock response to: {prompt[:50]}...] " * min(max_tokens // 20, 5)
        self._last_output = response
        return response.strip()

    def embed(self, text: str) -> List[float]:
        """Mock text embedding."""
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]

    def forward(
        self,
        input_ids: Any,
        **kwargs,
    ) -> MockTensor:
        """Mock forward pass."""
        self._call_count += 1

        # Determine output shape
        if hasattr(input_ids, 'shape'):
            batch_size, seq_len = input_ids.shape[:2]
        else:
            batch_size, seq_len = 1, 10

        return MockTensor(shape=(batch_size, seq_len, self.config.vocab_size))

    def get_hidden_states(
        self,
        input_ids: Any,
        layer_idx: int = -1,
    ) -> MockTensor:
        """Get hidden states from a layer."""
        if hasattr(input_ids, 'shape'):
            batch_size, seq_len = input_ids.shape[:2]
        else:
            batch_size, seq_len = 1, 10

        return MockTensor(shape=(batch_size, seq_len, self.config.d_model))

    def reset(self):
        """Reset call tracking."""
        self._call_count = 0
        self._last_input = None
        self._last_output = None


class MockOrchestrator:
    """
    Mock orchestrator for testing architecture selection and loading.

    Simulates the behavior of an ML orchestrator that can:
    - Select architectures based on task
    - Load and initialize models
    - Route inputs to appropriate backends
    """

    AVAILABLE_ARCHITECTURES = [
        "transformer",
        "mamba",
        "rwkv",
        "griffin",
        "hyena",
    ]

    def __init__(self):
        self._current_architecture = None
        self._loaded_model = None
        self._backends: Dict[str, MockBackend] = {}
        self._routing_history: List[Dict] = []

    def select_architecture(self, task: str) -> str:
        """Select architecture based on task characteristics."""
        # Simple heuristic for testing
        task_lower = task.lower()

        if "long" in task_lower or "context" in task_lower:
            return "mamba"
        elif "vision" in task_lower or "image" in task_lower:
            return "transformer"
        elif "efficient" in task_lower or "fast" in task_lower:
            return "rwkv"
        else:
            return "transformer"  # Default

    def load_architecture(
        self,
        architecture: str,
        config: Optional[MockModelConfig] = None,
    ) -> MockBackend:
        """Load and initialize an architecture."""
        if architecture not in self.AVAILABLE_ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")

        self._current_architecture = architecture

        # Create or reuse backend
        if architecture not in self._backends:
            self._backends[architecture] = MockBackend(config)

        self._loaded_model = self._backends[architecture]
        return self._loaded_model

    def run_inference(
        self,
        input_data: Any,
        architecture: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run inference through the loaded architecture."""
        if architecture:
            self.load_architecture(architecture)

        if self._loaded_model is None:
            raise RuntimeError("No architecture loaded")

        # Track routing
        self._routing_history.append({
            "architecture": self._current_architecture,
            "input_type": type(input_data).__name__,
        })

        # Run through backend
        if isinstance(input_data, str):
            output = self._loaded_model.generate(input_data, **kwargs)
        else:
            output = self._loaded_model.forward(input_data, **kwargs)

        return {
            "architecture": self._current_architecture,
            "output": output,
            "metadata": {
                "call_count": self._loaded_model._call_count,
            },
        }

    def switch_architecture(self, new_architecture: str) -> MockBackend:
        """Switch to a different architecture."""
        return self.load_architecture(new_architecture)

    def get_current_architecture(self) -> Optional[str]:
        """Get the currently loaded architecture."""
        return self._current_architecture

    def get_routing_history(self) -> List[Dict]:
        """Get history of routing decisions."""
        return self._routing_history.copy()

    def reset(self):
        """Reset orchestrator state."""
        self._current_architecture = None
        self._loaded_model = None
        self._routing_history.clear()
        for backend in self._backends.values():
            backend.reset()


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_config() -> MockModelConfig:
    """Provide a small model config for fast tests."""
    return MockModelConfig(
        d_model=64,
        n_layer=2,
        vocab_size=1000,
        d_state=8,
        d_conv=2,
        expand=2,
    )


@pytest.fixture
def mock_backend(mock_config) -> MockBackend:
    """Provide a mock backend for testing."""
    return MockBackend(mock_config)


@pytest.fixture
def mock_orchestrator() -> MockOrchestrator:
    """Provide a mock orchestrator for testing."""
    return MockOrchestrator()


@pytest.fixture
def sample_documents() -> List[Dict[str, str]]:
    """Provide sample documents for RAG testing."""
    return [
        {
            "id": "doc1",
            "content": "The capital of France is Paris. Paris is known for the Eiffel Tower.",
            "metadata": {"topic": "geography", "country": "France"},
        },
        {
            "id": "doc2",
            "content": "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks.",
            "metadata": {"topic": "technology", "field": "AI"},
        },
        {
            "id": "doc3",
            "content": "Python is a popular programming language. It is widely used in data science.",
            "metadata": {"topic": "programming", "language": "Python"},
        },
        {
            "id": "doc4",
            "content": "The transformer architecture was introduced in 2017. It uses self-attention mechanisms.",
            "metadata": {"topic": "ML", "architecture": "transformer"},
        },
        {
            "id": "doc5",
            "content": "Mamba is a state space model. It provides linear-time sequence modeling.",
            "metadata": {"topic": "ML", "architecture": "mamba"},
        },
    ]


@pytest.fixture
def sample_questions() -> List[str]:
    """Provide sample questions for testing."""
    return [
        "What is the capital of France?",
        "Explain machine learning in simple terms.",
        "What programming language is popular for data science?",
        "How does the transformer architecture work?",
        "What is Mamba and why is it efficient?",
    ]


@pytest.fixture
def sample_cot_examples():
    """Provide chain-of-thought examples for testing."""
    return [
        {
            "question": "What is 25 * 4?",
            "reasoning": "Step 1: 25 * 4 can be broken down as 25 * 2 * 2.\n"
                        "Step 2: 25 * 2 = 50.\n"
                        "Step 3: 50 * 2 = 100.",
            "answer": "100",
        },
        {
            "question": "If a train travels 60 mph for 2 hours, how far does it go?",
            "reasoning": "Step 1: Distance = Speed * Time.\n"
                        "Step 2: Speed = 60 mph.\n"
                        "Step 3: Time = 2 hours.\n"
                        "Step 4: Distance = 60 * 2 = 120 miles.",
            "answer": "120 miles",
        },
    ]


@pytest.fixture
def mock_torch():
    """Provide mock torch module for testing without PyTorch."""
    mock = MagicMock()

    # Mock tensor creation
    mock.tensor = lambda data, **kwargs: MockTensor(data=data)
    mock.zeros = MockTensor.zeros
    mock.ones = MockTensor.ones
    mock.randn = MockTensor.randn
    mock.randint = lambda low, high, size: MockTensor(shape=size)

    # Mock nn module
    mock.nn = MagicMock()
    mock.nn.Module = type("MockModule", (), {
        "__init__": lambda self: None,
        "forward": lambda self, x: x,
        "parameters": lambda self: iter([MockTensor()]),
        "eval": lambda self: self,
        "train": lambda self, mode=True: self,
    })
    mock.nn.Linear = MagicMock(return_value=MagicMock(
        forward=lambda x: x,
        weight=MockTensor(shape=(10, 10)),
    ))
    mock.nn.Embedding = MagicMock(return_value=MagicMock(
        forward=lambda x: MockTensor(shape=(x.shape[0], x.shape[1], 64)),
        weight=MockTensor(shape=(1000, 64)),
    ))
    mock.nn.LayerNorm = MagicMock(return_value=MagicMock(forward=lambda x: x))
    mock.nn.ModuleList = list

    # Mock functional
    mock.nn.functional = MagicMock()
    mock.nn.functional.silu = lambda x: x
    mock.nn.functional.softmax = lambda x, dim=-1: x

    # Mock no_grad context
    mock.no_grad = MagicMock(return_value=MagicMock(
        __enter__=lambda self: None,
        __exit__=lambda self, *args: None,
    ))

    # Mock device
    mock.device = lambda x: x
    mock.cuda = MagicMock()
    mock.cuda.is_available = lambda: False

    return mock


@pytest.fixture
def patch_torch(mock_torch):
    """Patch torch module for testing."""
    with patch.dict(sys.modules, {"torch": mock_torch, "torch.nn": mock_torch.nn}):
        yield mock_torch


# =============================================================================
# HELPER FIXTURES
# =============================================================================

@pytest.fixture
def hook_tracker():
    """Provide a hook tracker for testing hooks system."""
    class HookTracker:
        def __init__(self):
            self.calls: List[Dict] = []

        def make_hook(self, name: str) -> Callable:
            def hook(*args, **kwargs):
                self.calls.append({
                    "name": name,
                    "args": args,
                    "kwargs": kwargs,
                })
                return kwargs.get("output_data")  # Pass through for transform hooks
            return hook

        def get_calls(self, name: Optional[str] = None) -> List[Dict]:
            if name:
                return [c for c in self.calls if c["name"] == name]
            return self.calls

        def reset(self):
            self.calls.clear()

    return HookTracker()


@pytest.fixture
def technique_factory(mock_backend):
    """Factory for creating technique instances with mock backend."""
    def factory(technique_class, **kwargs):
        return technique_class(model=mock_backend, **kwargs)
    return factory


# =============================================================================
# ARCHITECTURE FIXTURES
# =============================================================================

@pytest.fixture
def mamba_config() -> Dict[str, Any]:
    """Provide Mamba-specific configuration."""
    return {
        "d_model": 64,
        "n_layer": 2,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "vocab_size": 1000,
    }


@pytest.fixture
def transformer_config() -> Dict[str, Any]:
    """Provide Transformer-specific configuration."""
    return {
        "d_model": 64,
        "n_layer": 2,
        "n_head": 4,
        "d_ff": 256,
        "vocab_size": 1000,
    }


# =============================================================================
# EXPORTS FOR DIRECT IMPORT
# =============================================================================

__all__ = [
    "MockModelConfig",
    "MockTensor",
    "MockBackend",
    "MockOrchestrator",
    "PYTEST_AVAILABLE",
]
