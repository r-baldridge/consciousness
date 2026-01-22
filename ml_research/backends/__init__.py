"""
LLM Backend Interface

A pluggable backend system for integrating various LLM providers and local models
with the ml_techniques module. This allows techniques like ChainOfThought, RAG,
ReAct, etc. to work with any compatible backend.

=============================================================================
ARCHITECTURE
=============================================================================

                    ┌─────────────────────────────────────┐
                    │         ML Techniques               │
                    │  (ChainOfThought, RAG, ReAct, ...)  │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │           LLMBackend                │
                    │         (Abstract Base)             │
                    └─────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │  LocalModel   │      │   Anthropic   │      │    OpenAI     │
    │   Backend     │      │    Backend    │      │   Backend     │
    │  (Mamba, etc) │      │   (Claude)    │      │   (GPT-4)     │
    └───────────────┘      └───────────────┘      └───────────────┘

=============================================================================
USAGE
=============================================================================

# Register and use backends
from ml_research.backends import BackendRegistry, MockBackend

# Register a backend
registry = BackendRegistry()
registry.register("mock", MockBackend())

# Get and use backend
backend = registry.get("mock")
response = backend.generate("What is 2+2?", max_tokens=100)

# Use with techniques
from ml_research.ml_techniques.prompting import ChainOfThought
cot = ChainOfThought(backend=backend)
result = cot.run("Solve this math problem...")

# Using the global registry
from ml_research.backends import get_backend, register_backend
register_backend("my_backend", MyCustomBackend())
backend = get_backend("my_backend")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import random
import time


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class BackendType(Enum):
    """Types of LLM backends."""
    LOCAL = "local"          # Local PyTorch/JAX models
    ANTHROPIC = "anthropic"  # Anthropic API (Claude)
    OPENAI = "openai"        # OpenAI API (GPT)
    MOCK = "mock"            # Mock backend for testing
    CUSTOM = "custom"        # Custom implementations


class ModelCapability(Enum):
    """Capabilities that backends may support."""
    TEXT_GENERATION = "text_generation"
    EMBEDDING = "embedding"
    TOKENIZATION = "tokenization"
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO = "audio"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop_sequences": self.stop_sequences,
            "seed": self.seed,
        }


@dataclass
class GenerationResult:
    """Result from text generation."""
    text: str
    finish_reason: str = "stop"  # stop, length, error
    tokens_used: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.finish_reason != "error"


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embedding: List[float]
    dimensions: int
    model: str = ""
    latency_ms: float = 0.0


@dataclass
class TokenizationResult:
    """Result from tokenization."""
    tokens: List[int]
    token_count: int
    text_length: int


@dataclass
class BackendInfo:
    """Information about a backend."""
    name: str
    backend_type: BackendType
    model_name: str
    capabilities: List[ModelCapability]
    max_context_length: int = 4096
    embedding_dimensions: Optional[int] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class LLMBackend(ABC):
    """
    Abstract base class for LLM backends.

    All backends must implement at minimum the generate() method.
    Other methods have default implementations that may be overridden.
    """

    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self._capabilities: List[ModelCapability] = []

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt/text to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (optional)
            stop_sequences: List of sequences that stop generation
            **kwargs: Additional backend-specific parameters

        Returns:
            Generated text string
        """
        pass

    def generate_with_config(
        self,
        prompt: str,
        config: GenerationConfig,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate text using a GenerationConfig object.

        Args:
            prompt: The input prompt
            config: Generation configuration
            **kwargs: Additional parameters

        Returns:
            GenerationResult with text and metadata
        """
        start = time.time()
        try:
            text = self.generate(
                prompt=prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                stop_sequences=config.stop_sequences,
                **kwargs,
            )
            latency = (time.time() - start) * 1000

            return GenerationResult(
                text=text,
                finish_reason="stop",
                tokens_used=len(text.split()),  # Approximate
                latency_ms=latency,
            )
        except Exception as e:
            return GenerationResult(
                text="",
                finish_reason="error",
                latency_ms=(time.time() - start) * 1000,
                metadata={"error": str(e)},
            )

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings"
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts]

    def tokenize(self, text: str) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Text to tokenize

        Returns:
            List of token IDs
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tokenization"
        )

    def detokenize(self, tokens: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text string
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support detokenization"
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text without full tokenization.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        try:
            return len(self.tokenize(text))
        except NotImplementedError:
            # Fallback: approximate with words * 1.3
            return int(len(text.split()) * 1.3)

    def get_info(self) -> BackendInfo:
        """Get information about this backend."""
        return BackendInfo(
            name=self.__class__.__name__,
            backend_type=BackendType.CUSTOM,
            model_name=self.model_name,
            capabilities=self._capabilities,
        )

    @property
    def capabilities(self) -> List[ModelCapability]:
        """Get list of supported capabilities."""
        return self._capabilities

    def supports(self, capability: ModelCapability) -> bool:
        """Check if backend supports a capability."""
        return capability in self._capabilities


# =============================================================================
# MOCK BACKEND (For Testing)
# =============================================================================

class MockBackend(LLMBackend):
    """
    Mock backend for testing without external dependencies.

    Provides deterministic, configurable responses for testing
    ML techniques without needing actual LLM API calls.

    Configuration:
        default_response: Static response for all prompts
        response_map: Dict mapping prompt patterns to responses
        delay_ms: Simulated latency
        simulate_errors: Whether to occasionally return errors

    Usage:
        mock = MockBackend(
            response_map={
                "math": "Let me solve this step by step...",
                "code": "```python\nprint('hello')\n```",
            }
        )
        result = mock.generate("Solve this math problem")
    """

    def __init__(
        self,
        model_name: str = "mock-model-v1",
        default_response: str = "This is a mock response.",
        response_map: Optional[Dict[str, str]] = None,
        delay_ms: float = 0.0,
        simulate_errors: bool = False,
        error_rate: float = 0.1,
        embedding_dim: int = 384,
        vocab_size: int = 50000,
    ):
        super().__init__(model_name)
        self.default_response = default_response
        self.response_map = response_map or {}
        self.delay_ms = delay_ms
        self.simulate_errors = simulate_errors
        self.error_rate = error_rate
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # All capabilities supported for testing
        self._capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.EMBEDDING,
            ModelCapability.TOKENIZATION,
        ]

        # Tracking for tests
        self.call_count = 0
        self.last_prompt: Optional[str] = None
        self.last_config: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate mock response based on prompt patterns."""
        # Track call
        self.call_count += 1
        self.last_prompt = prompt
        self.last_config = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop_sequences": stop_sequences,
            **kwargs,
        }
        self.history.append({
            "prompt": prompt,
            "config": self.last_config,
            "timestamp": time.time(),
        })

        # Simulate delay
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000)

        # Simulate errors
        if self.simulate_errors and random.random() < self.error_rate:
            raise RuntimeError("Simulated backend error")

        # Find matching response
        prompt_lower = prompt.lower()
        for pattern, response in self.response_map.items():
            if pattern.lower() in prompt_lower:
                return self._apply_stop_sequences(response, stop_sequences)

        # Generate context-aware mock response
        response = self._generate_contextual_response(prompt, temperature)
        return self._apply_stop_sequences(response, stop_sequences)

    def _generate_contextual_response(self, prompt: str, temperature: float) -> str:
        """Generate a context-aware mock response."""
        prompt_lower = prompt.lower()

        # Chain of thought patterns
        if "step by step" in prompt_lower or "think" in prompt_lower:
            return self._generate_cot_response(prompt)

        # Math patterns
        if any(op in prompt_lower for op in ["+", "-", "*", "/", "calculate", "solve"]):
            return self._generate_math_response(prompt)

        # Code patterns
        if any(kw in prompt_lower for kw in ["code", "function", "program", "python"]):
            return self._generate_code_response(prompt)

        # Question patterns
        if prompt.strip().endswith("?"):
            return self._generate_answer_response(prompt)

        # Default response with some variation based on temperature
        if temperature > 0.5:
            return f"{self.default_response} [Temperature: {temperature:.2f}]"
        return self.default_response

    def _generate_cot_response(self, prompt: str) -> str:
        """Generate a chain-of-thought style response."""
        return (
            "Let me think through this step by step.\n\n"
            "Step 1: First, I need to understand the problem.\n"
            "The question asks about the key aspects of the input.\n\n"
            "Step 2: Next, I'll analyze the relevant information.\n"
            "Looking at the context, I can identify several important points.\n\n"
            "Step 3: Now I'll synthesize my analysis.\n"
            "Combining these observations leads to a clear conclusion.\n\n"
            "Therefore, the answer is: [mock answer based on reasoning]"
        )

    def _generate_math_response(self, prompt: str) -> str:
        """Generate a math-style response."""
        return (
            "Let me solve this mathematical problem.\n\n"
            "Given the expression, I'll work through it:\n"
            "Step 1: Identify the operations\n"
            "Step 2: Apply order of operations (PEMDAS)\n"
            "Step 3: Calculate the result\n\n"
            "Therefore, the answer is 42."
        )

    def _generate_code_response(self, prompt: str) -> str:
        """Generate a code-style response."""
        return (
            "Here's a solution:\n\n"
            "```python\n"
            "def solve(input_data):\n"
            "    # Process the input\n"
            "    result = process(input_data)\n"
            "    return result\n"
            "\n"
            "# Example usage\n"
            "output = solve(data)\n"
            "print(output)\n"
            "```\n\n"
            "This code handles the requested functionality."
        )

    def _generate_answer_response(self, prompt: str) -> str:
        """Generate an answer to a question."""
        return (
            "Based on the question, here is my response:\n\n"
            "The answer depends on several factors that need to be considered. "
            "Taking into account the context provided, the most accurate answer "
            "would be: [mock answer]\n\n"
            "This conclusion is based on analysis of the available information."
        )

    def _apply_stop_sequences(
        self,
        text: str,
        stop_sequences: Optional[List[str]],
    ) -> str:
        """Apply stop sequences to truncate response."""
        if not stop_sequences:
            return text

        for seq in stop_sequences:
            if seq in text:
                text = text[:text.index(seq)]

        return text

    def embed(self, text: str) -> List[float]:
        """Generate deterministic mock embedding based on text hash."""
        # Use MD5 hash for deterministic embeddings
        h = hashlib.md5(text.encode()).hexdigest()

        # Generate embedding from hash
        embedding = []
        for i in range(0, min(len(h), self.embedding_dim * 2), 2):
            if len(embedding) >= self.embedding_dim:
                break
            val = int(h[i:i+2], 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
            embedding.append(val)

        # Pad if needed
        while len(embedding) < self.embedding_dim:
            embedding.append(0.0)

        # Normalize to unit length
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def tokenize(self, text: str) -> List[int]:
        """Simple tokenization: split on whitespace and punctuation."""
        # Simple word-level tokenization for testing
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text)

        # Convert to pseudo token IDs
        token_ids = []
        for token in tokens:
            # Hash token to get consistent ID
            h = hashlib.md5(token.lower().encode()).hexdigest()
            token_id = int(h[:8], 16) % self.vocab_size
            token_ids.append(token_id)

        return token_ids

    def detokenize(self, tokens: List[int]) -> str:
        """
        Mock detokenization.

        Note: Since we don't have a real vocabulary, this returns a placeholder.
        """
        return f"[Detokenized: {len(tokens)} tokens]"

    def count_tokens(self, text: str) -> int:
        """Count tokens using mock tokenization."""
        return len(self.tokenize(text))

    def get_info(self) -> BackendInfo:
        """Get mock backend info."""
        return BackendInfo(
            name="MockBackend",
            backend_type=BackendType.MOCK,
            model_name=self.model_name,
            capabilities=self._capabilities,
            max_context_length=8192,
            embedding_dimensions=self.embedding_dim,
            description="Mock backend for testing ML techniques",
            metadata={
                "vocab_size": self.vocab_size,
                "call_count": self.call_count,
            },
        )

    def reset_tracking(self):
        """Reset call tracking for tests."""
        self.call_count = 0
        self.last_prompt = None
        self.last_config = None
        self.history = []


# =============================================================================
# LOCAL MODEL BACKEND (PyTorch)
# =============================================================================

class LocalModelBackend(LLMBackend):
    """
    Backend for local PyTorch models (Mamba, custom transformers, etc.).

    This backend wraps local models to provide a consistent interface.
    Requires torch and the model to be loaded.

    Configuration:
        model: The PyTorch model instance
        tokenizer: Tokenizer for the model
        device: Device to run on ('cpu', 'cuda', 'mps')
        dtype: Data type for inference

    Usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        backend = LocalModelBackend(
            model=model,
            tokenizer=tokenizer,
            model_name="gpt2",
        )
        response = backend.generate("Hello, world!")
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        model_name: str = "local-model",
        device: str = "cpu",
        dtype: Optional[str] = None,
        max_context_length: int = 2048,
    ):
        super().__init__(model_name)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_context_length = max_context_length

        # Check for torch availability
        self._torch_available = False
        try:
            import torch
            self._torch_available = True
            self._torch = torch
        except ImportError:
            pass

        # Set capabilities based on what's available
        self._capabilities = [ModelCapability.TEXT_GENERATION]
        if tokenizer is not None:
            self._capabilities.append(ModelCapability.TOKENIZATION)

    def _ensure_torch(self):
        """Ensure torch is available."""
        if not self._torch_available:
            raise ImportError(
                "PyTorch is required for LocalModelBackend. "
                "Install with: pip install torch"
            )

    def _ensure_model(self):
        """Ensure model is loaded."""
        if self.model is None:
            raise ValueError(
                "Model not loaded. Initialize LocalModelBackend with a model."
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate text using the local model."""
        self._ensure_torch()
        self._ensure_model()

        # Tokenize input
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for generation")

        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 0.01),  # Avoid zero
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        if top_k is not None:
            gen_kwargs["top_k"] = top_k

        # Generate
        with self._torch.no_grad():
            outputs = self.model.generate(input_ids, **gen_kwargs)

        # Decode
        generated_ids = outputs[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Apply stop sequences
        if stop_sequences:
            for seq in stop_sequences:
                if seq in text:
                    text = text[:text.index(seq)]

        return text

    def embed(self, text: str) -> List[float]:
        """Generate embeddings using the model's hidden states."""
        self._ensure_torch()
        self._ensure_model()

        if self.tokenizer is None:
            raise ValueError("Tokenizer required for embedding")

        # Check if model has embedding layer
        if not hasattr(self.model, "get_input_embeddings"):
            raise NotImplementedError(
                "Model does not support embedding extraction"
            )

        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        with self._torch.no_grad():
            # Get embeddings from the model
            if hasattr(self.model, "model"):
                # For models with nested structure
                embeddings = self.model.model.embed_tokens(input_ids)
            else:
                embed_layer = self.model.get_input_embeddings()
                embeddings = embed_layer(input_ids)

            # Mean pooling
            embedding = embeddings.mean(dim=1).squeeze().cpu().numpy().tolist()

        return embedding

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the model's tokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available")

        return self.tokenizer.encode(text)

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize tokens using the model's tokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available")

        return self.tokenizer.decode(tokens)

    def get_info(self) -> BackendInfo:
        """Get local model backend info."""
        return BackendInfo(
            name="LocalModelBackend",
            backend_type=BackendType.LOCAL,
            model_name=self.model_name,
            capabilities=self._capabilities,
            max_context_length=self.max_context_length,
            description="Local PyTorch model backend",
            metadata={
                "device": self.device,
                "dtype": self.dtype,
                "torch_available": self._torch_available,
            },
        )


# =============================================================================
# ANTHROPIC BACKEND (Claude)
# =============================================================================

class AnthropicBackend(LLMBackend):
    """
    Backend for Anthropic's Claude API.

    Requires the anthropic package and a valid API key.

    Configuration:
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        model: Model name (default: "claude-sonnet-4-20250514")
        max_retries: Number of retries on failure

    Usage:
        backend = AnthropicBackend(
            api_key="your-api-key",
            model="claude-sonnet-4-20250514",
        )
        response = backend.generate("Explain quantum computing")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        super().__init__(model)
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout

        # Check for anthropic package
        self._client = None
        self._anthropic_available = False
        try:
            import anthropic
            self._anthropic_available = True
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
            else:
                # Will use ANTHROPIC_API_KEY env var
                self._client = anthropic.Anthropic()
        except ImportError:
            pass
        except Exception:
            # API key not set, will fail on use
            pass

        self._capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.STREAMING,
        ]

    def _ensure_client(self):
        """Ensure Anthropic client is available."""
        if not self._anthropic_available:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )
        if self._client is None:
            raise ValueError(
                "Anthropic API key not configured. "
                "Set ANTHROPIC_API_KEY or pass api_key parameter."
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate text using Claude API."""
        self._ensure_client()

        messages = [{"role": "user", "content": prompt}]

        api_kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if temperature != 0.7:  # Only set if non-default
            api_kwargs["temperature"] = temperature

        if top_p != 1.0:
            api_kwargs["top_p"] = top_p

        if top_k is not None:
            api_kwargs["top_k"] = top_k

        if stop_sequences:
            api_kwargs["stop_sequences"] = stop_sequences

        if system:
            api_kwargs["system"] = system

        # Make API call with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(**api_kwargs)
                return response.content[0].text
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        raise last_error

    def get_info(self) -> BackendInfo:
        """Get Anthropic backend info."""
        return BackendInfo(
            name="AnthropicBackend",
            backend_type=BackendType.ANTHROPIC,
            model_name=self.model_name,
            capabilities=self._capabilities,
            max_context_length=200000,  # Claude 3 context
            description="Anthropic Claude API backend",
            metadata={
                "api_available": self._anthropic_available,
                "client_configured": self._client is not None,
            },
        )


# =============================================================================
# OPENAI BACKEND (GPT)
# =============================================================================

class OpenAIBackend(LLMBackend):
    """
    Backend for OpenAI's GPT API.

    Requires the openai package and a valid API key.

    Configuration:
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        model: Model name (default: "gpt-4o")
        organization: Optional organization ID
        max_retries: Number of retries on failure

    Usage:
        backend = OpenAIBackend(
            api_key="your-api-key",
            model="gpt-4o",
        )
        response = backend.generate("Explain machine learning")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        organization: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        super().__init__(model)
        self.api_key = api_key
        self.organization = organization
        self.max_retries = max_retries
        self.timeout = timeout

        # Check for openai package
        self._client = None
        self._openai_available = False
        try:
            import openai
            self._openai_available = True
            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if organization:
                client_kwargs["organization"] = organization
            self._client = openai.OpenAI(**client_kwargs) if client_kwargs else openai.OpenAI()
        except ImportError:
            pass
        except Exception:
            pass

        self._capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.EMBEDDING,
            ModelCapability.STREAMING,
            ModelCapability.FUNCTION_CALLING,
        ]

    def _ensure_client(self):
        """Ensure OpenAI client is available."""
        if not self._openai_available:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )
        if self._client is None:
            raise ValueError(
                "OpenAI API key not configured. "
                "Set OPENAI_API_KEY or pass api_key parameter."
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: Optional[int] = None,  # Not supported by OpenAI
        stop_sequences: Optional[List[str]] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate text using OpenAI API."""
        self._ensure_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        api_kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }

        if stop_sequences:
            api_kwargs["stop"] = stop_sequences

        # Make API call with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(**api_kwargs)
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise last_error

    def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI's embedding API."""
        self._ensure_client()

        response = self._client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )

        return response.data[0].embedding

    def get_info(self) -> BackendInfo:
        """Get OpenAI backend info."""
        return BackendInfo(
            name="OpenAIBackend",
            backend_type=BackendType.OPENAI,
            model_name=self.model_name,
            capabilities=self._capabilities,
            max_context_length=128000,  # GPT-4 Turbo
            embedding_dimensions=1536,
            description="OpenAI GPT API backend",
            metadata={
                "api_available": self._openai_available,
                "client_configured": self._client is not None,
            },
        )


# =============================================================================
# BACKEND REGISTRY
# =============================================================================

class BackendRegistry:
    """
    Registry for managing and accessing LLM backends.

    Provides a central location to register, retrieve, and manage
    different backend implementations.

    Usage:
        registry = BackendRegistry()

        # Register backends
        registry.register("mock", MockBackend())
        registry.register("claude", AnthropicBackend())

        # Get backend
        backend = registry.get("mock")

        # Set default
        registry.set_default("claude")
        default = registry.get_default()

        # List available
        print(registry.list_backends())
    """

    def __init__(self):
        self._backends: Dict[str, LLMBackend] = {}
        self._default_name: Optional[str] = None

    def register(
        self,
        name: str,
        backend: LLMBackend,
        set_as_default: bool = False,
    ) -> None:
        """
        Register a backend with a name.

        Args:
            name: Unique name for the backend
            backend: Backend instance
            set_as_default: Whether to set as default backend
        """
        if name in self._backends:
            raise ValueError(f"Backend '{name}' already registered")

        self._backends[name] = backend

        if set_as_default or self._default_name is None:
            self._default_name = name

    def unregister(self, name: str) -> bool:
        """
        Unregister a backend.

        Args:
            name: Name of backend to remove

        Returns:
            True if backend was removed, False if not found
        """
        if name in self._backends:
            del self._backends[name]
            if self._default_name == name:
                self._default_name = next(iter(self._backends), None)
            return True
        return False

    def get(self, name: str) -> LLMBackend:
        """
        Get a backend by name.

        Args:
            name: Name of the backend

        Returns:
            The backend instance

        Raises:
            KeyError: If backend not found
        """
        if name not in self._backends:
            raise KeyError(
                f"Backend '{name}' not found. "
                f"Available: {list(self._backends.keys())}"
            )
        return self._backends[name]

    def get_default(self) -> LLMBackend:
        """
        Get the default backend.

        Returns:
            The default backend instance

        Raises:
            ValueError: If no backends registered
        """
        if self._default_name is None:
            raise ValueError("No backends registered")
        return self._backends[self._default_name]

    def set_default(self, name: str) -> None:
        """
        Set the default backend.

        Args:
            name: Name of backend to set as default

        Raises:
            KeyError: If backend not found
        """
        if name not in self._backends:
            raise KeyError(f"Backend '{name}' not found")
        self._default_name = name

    def list_backends(self) -> List[str]:
        """Get list of registered backend names."""
        return list(self._backends.keys())

    def get_info(self, name: str) -> BackendInfo:
        """Get info about a registered backend."""
        return self.get(name).get_info()

    def get_all_info(self) -> Dict[str, BackendInfo]:
        """Get info about all registered backends."""
        return {name: backend.get_info() for name, backend in self._backends.items()}

    def __contains__(self, name: str) -> bool:
        """Check if a backend is registered."""
        return name in self._backends

    def __len__(self) -> int:
        """Get number of registered backends."""
        return len(self._backends)


# =============================================================================
# GLOBAL REGISTRY AND CONVENIENCE FUNCTIONS
# =============================================================================

# Global registry instance
_global_registry = BackendRegistry()


def register_backend(
    name: str,
    backend: LLMBackend,
    set_as_default: bool = False,
) -> None:
    """Register a backend in the global registry."""
    _global_registry.register(name, backend, set_as_default)


def get_backend(name: Optional[str] = None) -> LLMBackend:
    """Get a backend from the global registry."""
    if name is None:
        return _global_registry.get_default()
    return _global_registry.get(name)


def list_backends() -> List[str]:
    """List all registered backends."""
    return _global_registry.list_backends()


def set_default_backend(name: str) -> None:
    """Set the default backend."""
    _global_registry.set_default(name)


# Auto-register mock backend
register_backend("mock", MockBackend(), set_as_default=True)


# =============================================================================
# EXPORTS
# =============================================================================

__version__ = "0.1.0"

__all__ = [
    # Enums
    "BackendType",
    "ModelCapability",
    # Data classes
    "GenerationConfig",
    "GenerationResult",
    "EmbeddingResult",
    "TokenizationResult",
    "BackendInfo",
    # Base class
    "LLMBackend",
    # Implementations
    "MockBackend",
    "LocalModelBackend",
    "AnthropicBackend",
    "OpenAIBackend",
    # Registry
    "BackendRegistry",
    # Global functions
    "register_backend",
    "get_backend",
    "list_backends",
    "set_default_backend",
]
