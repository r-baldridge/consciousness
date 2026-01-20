"""
Tests for RWKV model components.

Run with: pytest tests/test_model.py -v
"""

from __future__ import annotations

import pytest


class TestRWKVImports:
    """Test that all modules can be imported."""

    def test_import_model(self):
        """Test importing model module."""
        from rwkv.src.model import RWKVConfig, RWKV, RWKVBlock
        assert RWKVConfig is not None
        assert RWKV is not None
        assert RWKVBlock is not None

    def test_import_layers(self):
        """Test importing layers module."""
        from rwkv.src.layers import (
            WKVOperator,
            TimeMixing,
            ChannelMixing,
            TokenShift,
            GroupNorm,
        )
        assert WKVOperator is not None
        assert TimeMixing is not None
        assert ChannelMixing is not None
        assert TokenShift is not None
        assert GroupNorm is not None

    def test_import_src_init(self):
        """Test importing from src __init__."""
        from rwkv.src import (
            RWKVConfig,
            RWKV,
            RWKVBlock,
            WKVOperator,
            TimeMixing,
            ChannelMixing,
        )
        assert RWKVConfig is not None


class TestRWKVConfig:
    """Test RWKVConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from rwkv.src.model import RWKVConfig

        config = RWKVConfig()
        assert config.vocab_size == 50277
        assert config.hidden_dim == 768
        assert config.num_layers == 12
        assert config.head_size == 64
        assert config.context_length == 4096

    def test_num_heads_calculation(self):
        """Test automatic num_heads calculation."""
        from rwkv.src.model import RWKVConfig

        config = RWKVConfig(hidden_dim=768, head_size=64)
        assert config.num_heads == 12

        config = RWKVConfig(hidden_dim=1024, head_size=64)
        assert config.num_heads == 16

    def test_custom_config(self):
        """Test custom configuration."""
        from rwkv.src.model import RWKVConfig

        config = RWKVConfig(
            hidden_dim=1024,
            num_layers=24,
            version=6,
            use_data_dependent_decay=True,
        )
        assert config.hidden_dim == 1024
        assert config.num_layers == 24
        assert config.version == 6
        assert config.use_data_dependent_decay is True


class TestRWKVModel:
    """Test RWKV model instantiation."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        from rwkv.src.model import RWKVConfig
        return RWKVConfig(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=2,
            head_size=16,
            context_length=128,
        )

    def test_model_creation(self, small_config):
        """Test model can be created."""
        from rwkv.src.model import RWKV

        model = RWKV(small_config)
        assert model is not None

    def test_model_parameters(self, small_config):
        """Test model has parameters."""
        from rwkv.src.model import RWKV

        model = RWKV(small_config)
        num_params = model.num_parameters()
        assert num_params > 0

    def test_model_forward(self, small_config):
        """Test forward pass."""
        import torch
        from rwkv.src.model import RWKV

        model = RWKV(small_config)
        model.eval()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)

    def test_model_with_state(self, small_config):
        """Test forward pass with state."""
        import torch
        from rwkv.src.model import RWKV

        model = RWKV(small_config)
        model.eval()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, states = model(input_ids, return_state=True)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert len(states) == small_config.num_layers

    def test_init_state(self, small_config):
        """Test state initialization."""
        import torch
        from rwkv.src.model import RWKV

        model = RWKV(small_config)

        batch_size = 2
        states = model.init_state(batch_size)

        assert len(states) == small_config.num_layers
        for state in states:
            assert "time" in state
            assert "channel" in state


class TestRWKVBlock:
    """Test RWKVBlock component."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        from rwkv.src.model import RWKVConfig
        return RWKVConfig(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=2,
            head_size=16,
        )

    def test_block_creation(self, small_config):
        """Test block can be created."""
        from rwkv.src.model import RWKVBlock

        block = RWKVBlock(small_config, layer_idx=0)
        assert block is not None

    def test_block_forward(self, small_config):
        """Test block forward pass."""
        import torch
        from rwkv.src.model import RWKVBlock

        block = RWKVBlock(small_config, layer_idx=0)
        block.eval()

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, small_config.hidden_dim)

        with torch.no_grad():
            output, state = block(x)

        assert output.shape == x.shape


class TestTimeMixing:
    """Test TimeMixing layer."""

    def test_time_mixing_creation(self):
        """Test time mixing can be created."""
        from rwkv.src.layers import TimeMixing

        layer = TimeMixing(
            hidden_dim=64,
            head_size=16,
            num_heads=4,
            layer_idx=0,
            num_layers=2,
        )
        assert layer is not None

    def test_time_mixing_forward(self):
        """Test time mixing forward pass."""
        import torch
        from rwkv.src.layers import TimeMixing

        layer = TimeMixing(
            hidden_dim=64,
            head_size=16,
            num_heads=4,
            layer_idx=0,
            num_layers=2,
        )
        layer.eval()

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 64)

        with torch.no_grad():
            output, state = layer(x)

        assert output.shape == x.shape


class TestChannelMixing:
    """Test ChannelMixing layer."""

    def test_channel_mixing_creation(self):
        """Test channel mixing can be created."""
        from rwkv.src.layers import ChannelMixing

        layer = ChannelMixing(
            hidden_dim=64,
            layer_idx=0,
            num_layers=2,
        )
        assert layer is not None

    def test_channel_mixing_forward(self):
        """Test channel mixing forward pass."""
        import torch
        from rwkv.src.layers import ChannelMixing

        layer = ChannelMixing(
            hidden_dim=64,
            layer_idx=0,
            num_layers=2,
        )
        layer.eval()

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 64)

        with torch.no_grad():
            output, state = layer(x)

        assert output.shape == x.shape


class TestWKVOperator:
    """Test WKVOperator."""

    def test_wkv_creation(self):
        """Test WKV operator can be created."""
        from rwkv.src.layers import WKVOperator

        wkv = WKVOperator(
            hidden_dim=64,
            head_size=16,
            num_heads=4,
            layer_idx=0,
            num_layers=2,
        )
        assert wkv is not None

    def test_wkv_forward(self):
        """Test WKV forward pass."""
        import torch
        from rwkv.src.layers import WKVOperator

        wkv = WKVOperator(
            hidden_dim=64,
            head_size=16,
            num_heads=4,
            layer_idx=0,
            num_layers=2,
        )

        batch_size = 2
        seq_len = 16
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)

        output, state = wkv(k, v)

        assert output.shape == (batch_size, seq_len, 64)
        assert state is not None


class TestTokenShift:
    """Test TokenShift layer."""

    def test_token_shift_creation(self):
        """Test token shift can be created."""
        from rwkv.src.layers import TokenShift

        shift = TokenShift(hidden_dim=64)
        assert shift is not None

    def test_token_shift_forward(self):
        """Test token shift forward pass."""
        import torch
        from rwkv.src.layers import TokenShift

        shift = TokenShift(hidden_dim=64)

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 64)
        mu = torch.ones(64) * 0.5

        shifted, new_state = shift(x, mu)

        assert shifted.shape == x.shape
        assert new_state.shape == (batch_size, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
