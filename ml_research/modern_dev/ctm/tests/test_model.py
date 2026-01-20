"""
CTM Model Tests

Basic tests for Continuous Thought Machine model components.

Run with: pytest tests/test_model.py -v
"""

from __future__ import annotations

import pytest
import torch


class TestCTMConfig:
    """Tests for CTMConfig dataclass."""

    def test_default_config(self):
        """Test default configuration creation."""
        from ..src.model import CTMConfig

        config = CTMConfig()

        assert config.hidden_dim == 512
        assert config.num_neurons == 1024
        assert config.history_length == 8
        assert config.max_internal_steps == 32
        assert config.neuron_activation == "gelu"

    def test_custom_config(self):
        """Test custom configuration creation."""
        from ..src.model import CTMConfig

        config = CTMConfig(
            hidden_dim=256,
            num_neurons=512,
            history_length=4,
        )

        assert config.hidden_dim == 256
        assert config.num_neurons == 512
        assert config.history_length == 4

    def test_config_to_dict(self):
        """Test configuration serialization."""
        from ..src.model import CTMConfig

        config = CTMConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["hidden_dim"] == 512
        assert "num_neurons" in config_dict

    def test_config_from_dict(self):
        """Test configuration deserialization."""
        from ..src.model import CTMConfig

        config_dict = {
            "hidden_dim": 256,
            "num_neurons": 512,
            "history_length": 4,
            "max_internal_steps": 16,
            "sync_window": 2,
            "halt_threshold": 0.05,
            "neuron_activation": "relu",
            "input_dim": 512,
            "output_dim": 100,
            "dropout": 0.2,
            "use_adaptive_halt": True,
            "num_sync_heads": 4,
        }

        config = CTMConfig.from_dict(config_dict)

        assert config.hidden_dim == 256
        assert config.num_neurons == 512


class TestTemporalHistory:
    """Tests for TemporalHistory layer."""

    def test_initialization(self):
        """Test temporal history initialization."""
        from ..src.layers import TemporalHistory

        history = TemporalHistory(num_neurons=128, history_length=8)

        assert history.num_neurons == 128
        assert history.history_length == 8

    def test_reset(self):
        """Test history buffer reset."""
        from ..src.layers import TemporalHistory

        history = TemporalHistory(num_neurons=128, history_length=8)
        history.reset(batch_size=4, device=torch.device("cpu"))

        assert history.history.shape == (4, 128, 8)
        assert history.current_idx.item() == 0

    def test_update(self):
        """Test history buffer update."""
        from ..src.layers import TemporalHistory

        history = TemporalHistory(num_neurons=128, history_length=8)
        history.reset(batch_size=4, device=torch.device("cpu"))

        activations = torch.randn(4, 128)
        history.update(activations)

        assert history.current_idx.item() == 1

    def test_get_history(self):
        """Test getting full history tensor."""
        from ..src.layers import TemporalHistory

        history = TemporalHistory(num_neurons=128, history_length=8)
        history.reset(batch_size=4, device=torch.device("cpu"))

        # Add some activations
        for _ in range(4):
            history.update(torch.randn(4, 128))

        hist_tensor = history.get_history()
        assert hist_tensor.shape == (4, 128, 8)

    def test_get_recent_window(self):
        """Test getting recent activation window."""
        from ..src.layers import TemporalHistory

        history = TemporalHistory(num_neurons=128, history_length=8)
        history.reset(batch_size=4, device=torch.device("cpu"))

        # Add some activations
        for _ in range(4):
            history.update(torch.randn(4, 128))

        window = history.get_recent_window(window_size=3)
        assert window.shape == (4, 128, 3)


class TestNeuronLevelModel:
    """Tests for NeuronLevelModel layer."""

    def test_initialization(self):
        """Test NLM initialization."""
        from ..src.layers import NeuronLevelModel

        nlm = NeuronLevelModel(
            num_neurons=128,
            history_length=8,
            activation="gelu",
        )

        assert nlm.num_neurons == 128
        assert nlm.history_length == 8

    def test_forward(self):
        """Test NLM forward pass."""
        from ..src.layers import NeuronLevelModel

        nlm = NeuronLevelModel(
            num_neurons=128,
            history_length=8,
        )

        batch_size = 4
        current = torch.randn(batch_size, 128)
        history = torch.randn(batch_size, 128, 8)

        output = nlm(current, history)

        assert output.shape == (batch_size, 128)

    def test_different_activations(self):
        """Test NLM with different activation functions."""
        from ..src.layers import NeuronLevelModel

        for activation in ["gelu", "relu", "silu", "tanh"]:
            nlm = NeuronLevelModel(
                num_neurons=64,
                history_length=4,
                activation=activation,
            )
            current = torch.randn(2, 64)
            history = torch.randn(2, 64, 4)

            output = nlm(current, history)
            assert output.shape == (2, 64)


class TestSynchronizationLayer:
    """Tests for SynchronizationLayer."""

    def test_initialization(self):
        """Test sync layer initialization."""
        from ..src.layers import SynchronizationLayer

        sync = SynchronizationLayer(
            num_neurons=128,
            num_heads=8,
            window_size=4,
        )

        assert sync.num_neurons == 128
        assert sync.num_heads == 8
        assert sync.window_size == 4

    def test_forward(self):
        """Test sync layer forward pass."""
        from ..src.layers import SynchronizationLayer

        num_neurons = 128
        num_heads = 8
        window_size = 4

        sync = SynchronizationLayer(
            num_neurons=num_neurons,
            num_heads=num_heads,
            window_size=window_size,
        )

        batch_size = 4
        activations = torch.randn(batch_size, num_neurons)
        recent_window = torch.randn(batch_size, num_neurons, window_size)

        sync_matrix, sync_features = sync(activations, recent_window)

        # Check sync matrix shape
        neurons_per_head = num_neurons // num_heads
        assert sync_matrix.shape == (
            batch_size, num_heads, neurons_per_head, neurons_per_head
        )

        # Check sync features shape
        assert sync_features.shape == (batch_size, num_heads * window_size)


class TestCTM:
    """Tests for main CTM model."""

    def test_initialization(self):
        """Test CTM model initialization."""
        from ..src.model import CTM, CTMConfig

        config = CTMConfig(
            hidden_dim=128,
            num_neurons=256,
            input_dim=64,
            output_dim=10,
        )
        model = CTM(config)

        assert model.config == config

    def test_forward(self):
        """Test CTM forward pass."""
        from ..src.model import CTM, CTMConfig

        config = CTMConfig(
            hidden_dim=128,
            num_neurons=256,
            history_length=4,
            max_internal_steps=8,
            sync_window=2,
            num_sync_heads=4,
            input_dim=64,
            output_dim=10,
        )
        model = CTM(config)

        batch_size = 4
        inputs = torch.randn(batch_size, config.input_dim)

        outputs = model(inputs)

        assert "output" in outputs
        assert "activations" in outputs
        assert "sync_matrix" in outputs
        assert "num_steps_used" in outputs

        assert outputs["output"].shape == (batch_size, config.output_dim)
        assert outputs["activations"].shape == (batch_size, config.num_neurons)

    def test_forward_with_intermediates(self):
        """Test CTM forward pass returning intermediates."""
        from ..src.model import CTM, CTMConfig

        config = CTMConfig(
            hidden_dim=128,
            num_neurons=256,
            max_internal_steps=8,
            input_dim=64,
            output_dim=10,
            num_sync_heads=4,
        )
        model = CTM(config)

        inputs = torch.randn(2, config.input_dim)
        outputs = model(inputs, return_intermediates=True)

        assert "intermediates" in outputs
        assert len(outputs["intermediates"]) == config.max_internal_steps

    def test_custom_num_steps(self):
        """Test CTM forward with custom number of steps."""
        from ..src.model import CTM, CTMConfig

        config = CTMConfig(
            hidden_dim=64,
            num_neurons=128,
            max_internal_steps=16,
            input_dim=32,
            output_dim=10,
            num_sync_heads=4,
        )
        model = CTM(config)

        inputs = torch.randn(2, config.input_dim)

        # Test with fewer steps than max
        outputs = model(inputs, num_steps=4, return_intermediates=True)
        assert len(outputs["intermediates"]) == 4

    def test_parameter_count(self):
        """Test that model has trainable parameters."""
        from ..src.model import CTM, CTMConfig

        config = CTMConfig(
            hidden_dim=128,
            num_neurons=256,
            input_dim=64,
            output_dim=10,
        )
        model = CTM(config)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params > 0

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        from ..src.model import CTM, CTMConfig

        config = CTMConfig(
            hidden_dim=64,
            num_neurons=128,
            max_internal_steps=4,
            input_dim=32,
            output_dim=10,
            num_sync_heads=4,
        )
        model = CTM(config)

        inputs = torch.randn(2, config.input_dim, requires_grad=True)
        outputs = model(inputs)

        # Compute loss and backward
        loss = outputs["output"].sum()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestCTMEmbedding:
    """Tests for CTM embedding layer."""

    def test_forward(self):
        """Test embedding layer forward pass."""
        from ..src.model import CTMEmbedding, CTMConfig

        config = CTMConfig(
            num_neurons=256,
            input_dim=64,
        )
        embedding = CTMEmbedding(config)

        inputs = torch.randn(4, config.input_dim)
        outputs = embedding(inputs)

        assert outputs.shape == (4, config.num_neurons)


class TestCTMOutputHead:
    """Tests for CTM output head."""

    def test_forward(self):
        """Test output head forward pass."""
        from ..src.model import CTMOutputHead, CTMConfig

        config = CTMConfig(
            hidden_dim=128,
            num_neurons=256,
            num_sync_heads=8,
            sync_window=4,
            output_dim=10,
        )
        output_head = CTMOutputHead(config)

        batch_size = 4
        activations = torch.randn(batch_size, config.num_neurons)
        sync_features = torch.randn(
            batch_size, config.num_sync_heads * config.sync_window
        )

        output = output_head(activations, sync_features)

        assert output.shape == (batch_size, config.output_dim)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
