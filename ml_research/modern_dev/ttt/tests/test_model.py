"""
TTT Model Tests

Basic tests for Test-Time Training model components.

Run with: pytest tests/test_model.py -v
"""

from __future__ import annotations

import pytest
import torch


class TestTTTConfig:
    """Tests for TTTConfig dataclass."""

    def test_default_config(self):
        """Test default configuration creation."""
        from ..src.model import TTTConfig

        config = TTTConfig()

        assert config.hidden_dim == 768
        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.ttt_type == "linear"
        assert config.ttt_learning_rate == 1.0

    def test_custom_config(self):
        """Test custom configuration creation."""
        from ..src.model import TTTConfig

        config = TTTConfig(
            hidden_dim=512,
            num_layers=6,
            ttt_type="mlp",
            ttt_learning_rate=0.5,
        )

        assert config.hidden_dim == 512
        assert config.num_layers == 6
        assert config.ttt_type == "mlp"
        assert config.ttt_learning_rate == 0.5

    def test_config_to_dict(self):
        """Test configuration serialization."""
        from ..src.model import TTTConfig

        config = TTTConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["hidden_dim"] == 768
        assert "ttt_learning_rate" in config_dict

    def test_config_from_dict(self):
        """Test configuration deserialization."""
        from ..src.model import TTTConfig

        config_dict = {
            "hidden_dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "vocab_size": 50257,
            "max_seq_len": 1024,
            "ttt_type": "mlp",
            "mlp_hidden_dim": 1024,
            "ttt_learning_rate": 0.5,
            "mini_batch_size": 8,
            "use_rope": True,
            "rope_base": 10000,
            "layer_norm_eps": 1e-5,
            "dropout": 0.1,
            "initializer_range": 0.02,
            "tie_word_embeddings": True,
        }

        config = TTTConfig.from_dict(config_dict)

        assert config.hidden_dim == 512
        assert config.ttt_type == "mlp"


class TestRotaryEmbedding:
    """Tests for rotary position embedding."""

    def test_initialization(self):
        """Test rotary embedding initialization."""
        from ..src.layers import RotaryEmbedding

        rope = RotaryEmbedding(dim=64, base=10000)

        assert rope.dim == 64
        assert rope.base == 10000

    def test_forward(self):
        """Test rotary embedding forward pass."""
        from ..src.layers import RotaryEmbedding

        rope = RotaryEmbedding(dim=64)
        x = torch.randn(2, 8, 32, 64)  # (batch, heads, seq, head_dim)

        cos, sin = rope(x, seq_len=32)

        assert cos.shape == (32, 64)
        assert sin.shape == (32, 64)


class TestInnerOptimizer:
    """Tests for inner optimizer."""

    def test_step(self):
        """Test optimizer step."""
        from ..src.layers import InnerOptimizer

        optimizer = InnerOptimizer(learning_rate=0.1)

        weights = torch.randn(10, 10)
        gradient = torch.randn(10, 10)

        new_weights = optimizer.step(weights, gradient)

        # Weights should change
        assert not torch.allclose(weights, new_weights)

        # Change should be proportional to gradient
        expected = weights - 0.1 * gradient
        assert torch.allclose(new_weights, expected)

    def test_momentum(self):
        """Test optimizer with momentum."""
        from ..src.layers import InnerOptimizer

        optimizer = InnerOptimizer(learning_rate=0.1, momentum=0.9)

        weights = torch.randn(10, 10)
        gradient = torch.randn(10, 10)

        # First step
        new_weights = optimizer.step(weights, gradient)

        # Velocity should be initialized
        assert optimizer.velocity is not None


class TestTTTLinear:
    """Tests for TTT-Linear layer."""

    def test_initialization(self):
        """Test TTT-Linear initialization."""
        from ..src.layers import TTTLinear

        layer = TTTLinear(
            hidden_dim=256,
            num_heads=4,
            learning_rate=1.0,
            mini_batch_size=8,
        )

        assert layer.hidden_dim == 256
        assert layer.num_heads == 4
        assert layer.head_dim == 64

    def test_forward(self):
        """Test TTT-Linear forward pass."""
        from ..src.layers import TTTLinear

        layer = TTTLinear(
            hidden_dim=256,
            num_heads=4,
            learning_rate=1.0,
            mini_batch_size=8,
            use_rope=False,  # Disable RoPE for simplicity
        )

        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 256)

        output, hidden = layer(x, return_hidden_state=True)

        assert output.shape == (batch_size, seq_len, 256)
        assert hidden is not None
        assert hidden.shape == (batch_size, 4, 64, 64)  # (batch, heads, d, d)

    def test_stateful_processing(self):
        """Test stateful processing with hidden state."""
        from ..src.layers import TTTLinear

        layer = TTTLinear(
            hidden_dim=256,
            num_heads=4,
            learning_rate=1.0,
            mini_batch_size=8,
            use_rope=False,
        )

        batch_size = 2
        x1 = torch.randn(batch_size, 16, 256)
        x2 = torch.randn(batch_size, 16, 256)

        # Process first chunk
        out1, hidden1 = layer(x1, return_hidden_state=True)

        # Process second chunk with previous hidden state
        out2, hidden2 = layer(x2, hidden_state=hidden1, return_hidden_state=True)

        # Hidden state should have changed
        assert not torch.allclose(hidden1, hidden2)


class TestTTTMLP:
    """Tests for TTT-MLP layer."""

    def test_initialization(self):
        """Test TTT-MLP initialization."""
        from ..src.layers import TTTMLP

        layer = TTTMLP(
            hidden_dim=256,
            mlp_hidden_dim=512,
            num_heads=4,
            learning_rate=1.0,
            mini_batch_size=8,
        )

        assert layer.hidden_dim == 256
        assert layer.mlp_hidden_dim == 512

    def test_forward(self):
        """Test TTT-MLP forward pass."""
        from ..src.layers import TTTMLP

        layer = TTTMLP(
            hidden_dim=256,
            mlp_hidden_dim=512,
            num_heads=4,
            learning_rate=1.0,
            mini_batch_size=8,
            use_rope=False,
        )

        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 256)

        output, hidden = layer(x, return_hidden_state=True)

        assert output.shape == (batch_size, seq_len, 256)
        assert hidden is not None
        assert len(hidden) == 2  # (W1, W2) tuple


class TestTTTBlock:
    """Tests for TTT block."""

    def test_forward(self):
        """Test TTT block forward pass."""
        from ..src.model import TTTBlock, TTTConfig

        config = TTTConfig(
            hidden_dim=256,
            num_heads=4,
            ttt_type="linear",
            mini_batch_size=8,
            use_rope=False,
        )
        block = TTTBlock(config, layer_idx=0)

        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 256)

        output, hidden = block(x, return_hidden_state=True)

        assert output.shape == (batch_size, seq_len, 256)


class TestTTTLanguageModel:
    """Tests for main TTT language model."""

    def test_initialization(self):
        """Test model initialization."""
        from ..src.model import TTTLanguageModel, TTTConfig

        config = TTTConfig(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            vocab_size=1000,
            max_seq_len=128,
            ttt_type="linear",
            use_rope=False,
        )
        model = TTTLanguageModel(config)

        assert model.config == config
        assert len(model.blocks) == 2

    def test_forward(self):
        """Test model forward pass."""
        from ..src.model import TTTLanguageModel, TTTConfig

        config = TTTConfig(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            vocab_size=1000,
            max_seq_len=128,
            mini_batch_size=16,
            ttt_type="linear",
            use_rope=False,
        )
        model = TTTLanguageModel(config)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        outputs = model(input_ids)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 1000)

    def test_forward_with_labels(self):
        """Test model forward pass with labels."""
        from ..src.model import TTTLanguageModel, TTTConfig

        config = TTTConfig(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            vocab_size=1000,
            mini_batch_size=16,
            ttt_type="linear",
            use_rope=False,
        )
        model = TTTLanguageModel(config)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)

        assert "loss" in outputs
        assert outputs["loss"].ndim == 0  # scalar

    def test_generate(self):
        """Test text generation."""
        from ..src.model import TTTLanguageModel, TTTConfig

        config = TTTConfig(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            vocab_size=100,
            max_seq_len=64,
            mini_batch_size=8,
            ttt_type="linear",
            use_rope=False,
        )
        model = TTTLanguageModel(config)
        model.eval()

        input_ids = torch.randint(0, 100, (1, 5))
        generated = model.generate(input_ids, max_new_tokens=10)

        assert generated.shape[1] == 15  # 5 + 10

    def test_parameter_count(self):
        """Test model has trainable parameters."""
        from ..src.model import TTTLanguageModel, TTTConfig

        config = TTTConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            vocab_size=1000,
        )
        model = TTTLanguageModel(config)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params > 0

    def test_gradient_flow(self):
        """Test gradient flow through model."""
        from ..src.model import TTTLanguageModel, TTTConfig

        config = TTTConfig(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            vocab_size=100,
            mini_batch_size=16,
            ttt_type="linear",
            use_rope=False,
        )
        model = TTTLanguageModel(config)

        input_ids = torch.randint(0, 100, (2, 16))
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad


class TestTTTForSequenceClassification:
    """Tests for sequence classification model."""

    def test_forward(self):
        """Test classification forward pass."""
        from ..src.model import TTTForSequenceClassification, TTTConfig

        config = TTTConfig(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            vocab_size=1000,
            mini_batch_size=16,
            ttt_type="linear",
            use_rope=False,
        )
        model = TTTForSequenceClassification(config, num_labels=5)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        outputs = model(input_ids)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, 5)

    def test_forward_with_labels(self):
        """Test classification with labels."""
        from ..src.model import TTTForSequenceClassification, TTTConfig

        config = TTTConfig(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            vocab_size=1000,
            mini_batch_size=16,
            ttt_type="linear",
            use_rope=False,
        )
        model = TTTForSequenceClassification(config, num_labels=5)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 5, (batch_size,))

        outputs = model(input_ids, labels=labels)

        assert "loss" in outputs
        assert outputs["loss"].ndim == 0


class TestTTTEmbedding:
    """Tests for TTT embedding layer."""

    def test_forward(self):
        """Test embedding forward pass."""
        from ..src.model import TTTEmbedding, TTTConfig

        config = TTTConfig(
            hidden_dim=256,
            vocab_size=1000,
            max_seq_len=128,
            use_rope=False,
        )
        embedding = TTTEmbedding(config)

        input_ids = torch.randint(0, 1000, (2, 32))
        output = embedding(input_ids)

        assert output.shape == (2, 32, 256)

    def test_forward_with_rope(self):
        """Test embedding without position embedding (RoPE mode)."""
        from ..src.model import TTTEmbedding, TTTConfig

        config = TTTConfig(
            hidden_dim=256,
            vocab_size=1000,
            max_seq_len=128,
            use_rope=True,
        )
        embedding = TTTEmbedding(config)

        input_ids = torch.randint(0, 1000, (2, 32))
        output = embedding(input_ids)

        assert output.shape == (2, 32, 256)
        assert embedding.position_embedding is None


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
