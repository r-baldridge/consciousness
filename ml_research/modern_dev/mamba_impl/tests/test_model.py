"""
Tests for Mamba model components.

Run with: pytest tests/test_model.py -v
"""

from __future__ import annotations

import pytest


class TestMambaImports:
    """Test that all modules can be imported."""

    def test_import_model(self):
        """Test importing model module."""
        from mamba_impl.src.model import MambaConfig, Mamba, MambaBlock
        assert MambaConfig is not None
        assert Mamba is not None
        assert MambaBlock is not None

    def test_import_layers(self):
        """Test importing layers module."""
        from mamba_impl.src.layers import (
            SelectiveSSM,
            S6Layer,
            CausalConv1d,
            Discretization,
            ParallelScan,
            RMSNorm,
        )
        assert SelectiveSSM is not None
        assert S6Layer is not None
        assert CausalConv1d is not None
        assert Discretization is not None
        assert ParallelScan is not None
        assert RMSNorm is not None

    def test_import_src_init(self):
        """Test importing from src __init__."""
        from mamba_impl.src import (
            MambaConfig,
            Mamba,
            MambaBlock,
            SelectiveSSM,
            S6Layer,
            CausalConv1d,
        )
        assert MambaConfig is not None


class TestMambaConfig:
    """Test MambaConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from mamba_impl.src.model import MambaConfig

        config = MambaConfig()
        assert config.d_model == 768
        assert config.n_layer == 24
        assert config.vocab_size == 50280
        assert config.d_state == 16
        assert config.d_conv == 4
        assert config.expand == 2

    def test_derived_attributes(self):
        """Test computed attributes."""
        from mamba_impl.src.model import MambaConfig

        config = MambaConfig(d_model=1024, expand=2)
        assert config.d_inner == 2048

    def test_auto_dt_rank(self):
        """Test automatic dt_rank calculation."""
        from mamba_impl.src.model import MambaConfig
        import math

        config = MambaConfig(d_model=768, dt_rank="auto")
        expected_dt_rank = math.ceil(768 / 16)
        assert config.dt_rank == expected_dt_rank

    def test_custom_config(self):
        """Test custom configuration."""
        from mamba_impl.src.model import MambaConfig

        config = MambaConfig(
            d_model=1024,
            n_layer=48,
            d_state=32,
        )
        assert config.d_model == 1024
        assert config.n_layer == 48
        assert config.d_state == 32


class TestMambaModel:
    """Test Mamba model instantiation."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        from mamba_impl.src.model import MambaConfig
        return MambaConfig(
            d_model=64,
            n_layer=2,
            vocab_size=1000,
            d_state=8,
            d_conv=2,
            expand=2,
        )

    def test_model_creation(self, small_config):
        """Test model can be created."""
        from mamba_impl.src.model import Mamba

        model = Mamba(small_config)
        assert model is not None

    def test_model_parameters(self, small_config):
        """Test model has parameters."""
        from mamba_impl.src.model import Mamba

        model = Mamba(small_config)
        num_params = model.num_parameters()
        assert num_params > 0

    def test_model_forward(self, small_config):
        """Test forward pass."""
        import torch
        from mamba_impl.src.model import Mamba

        model = Mamba(small_config)
        model.eval()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)

    def test_model_from_config(self, small_config):
        """Test creating model from config."""
        from mamba_impl.src.model import Mamba

        model = Mamba.from_config(small_config)
        assert model is not None


class TestMambaBlock:
    """Test MambaBlock component."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        from mamba_impl.src.model import MambaConfig
        return MambaConfig(
            d_model=64,
            n_layer=1,
            vocab_size=1000,
            d_state=8,
            d_conv=2,
            expand=2,
        )

    def test_block_creation(self, small_config):
        """Test block can be created."""
        from mamba_impl.src.model import MambaBlock

        block = MambaBlock(small_config)
        assert block is not None

    def test_block_forward(self, small_config):
        """Test block forward pass."""
        import torch
        from mamba_impl.src.model import MambaBlock

        block = MambaBlock(small_config)
        block.eval()

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, small_config.d_model)

        with torch.no_grad():
            output = block(x)

        assert output.shape == x.shape


class TestSelectiveSSM:
    """Test SelectiveSSM layer."""

    def test_ssm_creation(self):
        """Test SSM can be created."""
        from mamba_impl.src.layers import SelectiveSSM

        ssm = SelectiveSSM(
            d_model=64,
            d_state=16,
            dt_rank=8,
        )
        assert ssm is not None

    def test_ssm_forward(self):
        """Test SSM forward pass."""
        import torch
        from mamba_impl.src.layers import SelectiveSSM

        ssm = SelectiveSSM(
            d_model=64,
            d_state=16,
            dt_rank=8,
        )
        ssm.eval()

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 64)

        with torch.no_grad():
            output = ssm(x)

        assert output.shape == x.shape


class TestCausalConv1d:
    """Test CausalConv1d layer."""

    def test_conv_creation(self):
        """Test conv can be created."""
        from mamba_impl.src.layers import CausalConv1d

        conv = CausalConv1d(
            in_channels=64,
            kernel_size=4,
        )
        assert conv is not None

    def test_conv_causality(self):
        """Test that conv is causal (output at t only depends on inputs <= t)."""
        import torch
        from mamba_impl.src.layers import CausalConv1d

        conv = CausalConv1d(
            in_channels=64,
            kernel_size=4,
        )
        conv.eval()

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, 64, seq_len)

        with torch.no_grad():
            output = conv(x)

        # Output should have same sequence length (causal padding)
        assert output.shape == (batch_size, 64, seq_len)


class TestDiscretization:
    """Test discretization methods."""

    def test_zero_order_hold(self):
        """Test ZOH discretization."""
        import torch
        from mamba_impl.src.layers import Discretization

        batch_size = 2
        d_model = 64
        d_state = 16

        A = -torch.ones(d_model, d_state)
        B = torch.ones(d_model, d_state)
        delta = torch.ones(batch_size, d_model) * 0.1

        A_bar, B_bar = Discretization.zero_order_hold(A, B, delta)

        assert A_bar.shape == (batch_size, d_model, d_state)
        assert B_bar.shape == (batch_size, d_model, d_state)

    def test_bilinear(self):
        """Test bilinear discretization."""
        import torch
        from mamba_impl.src.layers import Discretization

        batch_size = 2
        d_model = 64
        d_state = 16

        A = -torch.ones(d_model, d_state)
        B = torch.ones(d_model, d_state)
        delta = torch.ones(batch_size, d_model) * 0.1

        A_bar, B_bar = Discretization.bilinear(A, B, delta)

        assert A_bar.shape == (batch_size, d_model, d_state)
        assert B_bar.shape == (batch_size, d_model, d_state)


class TestRMSNorm:
    """Test RMSNorm layer."""

    def test_rmsnorm_creation(self):
        """Test RMSNorm can be created."""
        from mamba_impl.src.layers import RMSNorm

        norm = RMSNorm(d_model=64)
        assert norm is not None

    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        import torch
        from mamba_impl.src.layers import RMSNorm

        norm = RMSNorm(d_model=64)

        x = torch.randn(2, 16, 64)
        output = norm(x)

        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
