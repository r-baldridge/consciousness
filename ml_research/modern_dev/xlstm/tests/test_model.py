"""
Tests for xLSTM model components.

Run with: pytest tests/test_model.py -v
"""

from __future__ import annotations

import pytest


class TestxLSTMImports:
    """Test that all modules can be imported."""

    def test_import_model(self):
        """Test importing model module."""
        from xlstm.src.model import xLSTMConfig, xLSTM, xLSTMBlock
        assert xLSTMConfig is not None
        assert xLSTM is not None
        assert xLSTMBlock is not None

    def test_import_layers(self):
        """Test importing layers module."""
        from xlstm.src.layers import (
            sLSTMCell,
            mLSTMCell,
            ExponentialGating,
            MatrixMemory,
            CausalConv1d,
        )
        assert sLSTMCell is not None
        assert mLSTMCell is not None
        assert ExponentialGating is not None
        assert MatrixMemory is not None
        assert CausalConv1d is not None

    def test_import_src_init(self):
        """Test importing from src __init__."""
        from xlstm.src import (
            xLSTMConfig,
            xLSTM,
            xLSTMBlock,
            sLSTMCell,
            mLSTMCell,
        )
        assert xLSTMConfig is not None


class TestxLSTMConfig:
    """Test xLSTMConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from xlstm.src.model import xLSTMConfig

        config = xLSTMConfig()
        assert config.vocab_size == 50257
        assert config.embedding_dim == 768
        assert config.num_layers == 12
        assert config.num_heads == 8
        assert config.head_dim == 64

    def test_auto_layer_types(self):
        """Test automatic layer type generation."""
        from xlstm.src.model import xLSTMConfig

        config = xLSTMConfig(num_layers=12)
        assert len(config.layer_types) == 12
        # Default: sLSTM every 4th layer
        assert config.layer_types[3] == "s"
        assert config.layer_types[7] == "s"
        assert config.layer_types[11] == "s"
        assert config.layer_types[0] == "m"

    def test_custom_layer_types(self):
        """Test custom layer type specification."""
        from xlstm.src.model import xLSTMConfig

        layer_types = ["m", "m", "s", "m"]
        config = xLSTMConfig(num_layers=4, layer_types=layer_types)
        assert config.layer_types == layer_types

    def test_slstm_at_layer_idx(self):
        """Test specifying sLSTM layers by index."""
        from xlstm.src.model import xLSTMConfig

        config = xLSTMConfig(num_layers=8, slstm_at_layer_idx=[2, 5])
        assert config.layer_types[2] == "s"
        assert config.layer_types[5] == "s"
        assert config.layer_types[0] == "m"
        assert config.layer_types[3] == "m"


class TestxLSTMModel:
    """Test xLSTM model instantiation."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        from xlstm.src.model import xLSTMConfig
        return xLSTMConfig(
            vocab_size=1000,
            embedding_dim=64,
            num_layers=4,
            layer_types=["m", "m", "s", "m"],
            num_heads=4,
            head_dim=16,
            up_proj_factor=2.0,
        )

    def test_model_creation(self, small_config):
        """Test model can be created."""
        from xlstm.src.model import xLSTM

        model = xLSTM(small_config)
        assert model is not None

    def test_model_parameters(self, small_config):
        """Test model has parameters."""
        from xlstm.src.model import xLSTM

        model = xLSTM(small_config)
        num_params = model.num_parameters()
        assert num_params > 0

    def test_model_forward(self, small_config):
        """Test forward pass."""
        import torch
        from xlstm.src.model import xLSTM

        model = xLSTM(small_config)
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
        from xlstm.src.model import xLSTM

        model = xLSTM(small_config)
        model.eval()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, states = model(input_ids, return_state=True)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert len(states) == small_config.num_layers

    def test_layer_stats(self, small_config):
        """Test layer statistics."""
        from xlstm.src.model import xLSTM

        model = xLSTM(small_config)
        stats = model.get_layer_stats()

        assert stats["total_layers"] == 4
        assert stats["slstm_layers"] == 1
        assert stats["mlstm_layers"] == 3


class TestxLSTMBlock:
    """Test xLSTMBlock component."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        from xlstm.src.model import xLSTMConfig
        return xLSTMConfig(
            vocab_size=1000,
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
            head_dim=16,
        )

    def test_slstm_block_creation(self, small_config):
        """Test sLSTM block can be created."""
        from xlstm.src.model import xLSTMBlock

        block = xLSTMBlock(small_config, layer_idx=0, block_type="s")
        assert block is not None
        assert block.block_type == "s"

    def test_mlstm_block_creation(self, small_config):
        """Test mLSTM block can be created."""
        from xlstm.src.model import xLSTMBlock

        block = xLSTMBlock(small_config, layer_idx=0, block_type="m")
        assert block is not None
        assert block.block_type == "m"

    def test_block_forward(self, small_config):
        """Test block forward pass."""
        import torch
        from xlstm.src.model import xLSTMBlock

        for block_type in ["s", "m"]:
            block = xLSTMBlock(small_config, layer_idx=0, block_type=block_type)
            block.eval()

            batch_size = 2
            seq_len = 16
            x = torch.randn(batch_size, seq_len, small_config.embedding_dim)

            with torch.no_grad():
                output, state = block(x)

            assert output.shape == x.shape


class TestsLSTMCell:
    """Test sLSTMCell layer."""

    def test_slstm_creation(self):
        """Test sLSTM cell can be created."""
        from xlstm.src.layers import sLSTMCell

        cell = sLSTMCell(
            input_dim=64,
            hidden_dim=64,
            num_heads=4,
        )
        assert cell is not None

    def test_slstm_forward(self):
        """Test sLSTM forward pass."""
        import torch
        from xlstm.src.layers import sLSTMCell

        cell = sLSTMCell(
            input_dim=64,
            hidden_dim=64,
            num_heads=4,
        )

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 64)

        output, state = cell(x)

        assert output.shape == x.shape
        assert len(state) == 3  # h, c, n

    def test_slstm_with_state(self):
        """Test sLSTM with initial state."""
        import torch
        from xlstm.src.layers import sLSTMCell

        cell = sLSTMCell(
            input_dim=64,
            hidden_dim=64,
        )

        batch_size = 2
        state = cell.init_state(batch_size, torch.device("cpu"), torch.float32)
        x = torch.randn(batch_size, 8, 64)

        output, new_state = cell(x, state=state)
        assert output.shape == x.shape


class TestmLSTMCell:
    """Test mLSTMCell layer."""

    def test_mlstm_creation(self):
        """Test mLSTM cell can be created."""
        from xlstm.src.layers import mLSTMCell

        cell = mLSTMCell(
            input_dim=64,
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
        )
        assert cell is not None

    def test_mlstm_forward(self):
        """Test mLSTM forward pass."""
        import torch
        from xlstm.src.layers import mLSTMCell

        cell = mLSTMCell(
            input_dim=64,
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
        )

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 64)

        output, state = cell(x)

        assert output.shape == x.shape
        assert len(state) == 2  # C, n

    def test_mlstm_with_state(self):
        """Test mLSTM with initial state."""
        import torch
        from xlstm.src.layers import mLSTMCell

        cell = mLSTMCell(
            input_dim=64,
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
        )

        batch_size = 2
        state = cell.init_state(batch_size, torch.device("cpu"), torch.float32)
        x = torch.randn(batch_size, 8, 64)

        output, new_state = cell(x, state=state)
        assert output.shape == x.shape


class TestMatrixMemory:
    """Test MatrixMemory layer."""

    def test_matrix_memory_creation(self):
        """Test matrix memory can be created."""
        from xlstm.src.layers import MatrixMemory

        memory = MatrixMemory(
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
        )
        assert memory is not None

    def test_matrix_memory_forward(self):
        """Test matrix memory forward pass."""
        import torch
        from xlstm.src.layers import MatrixMemory

        memory = MatrixMemory(
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
        )

        batch_size = 2
        k = torch.randn(batch_size, 16)
        v = torch.randn(batch_size, 16)
        q = torch.randn(batch_size, 16)
        i_gate = torch.ones(batch_size)
        f_gate = torch.ones(batch_size)

        output, C, n = memory(k, v, q, i_gate, f_gate)

        assert output.shape == (batch_size, 16)
        assert C.shape == (batch_size, 16, 16)
        assert n.shape == (batch_size, 16)


class TestExponentialGating:
    """Test ExponentialGating layer."""

    def test_exp_gating_creation(self):
        """Test exponential gating can be created."""
        from xlstm.src.layers import ExponentialGating

        gating = ExponentialGating(input_dim=64, hidden_dim=64)
        assert gating is not None

    def test_exp_gating_forward(self):
        """Test exponential gating forward pass."""
        import torch
        from xlstm.src.layers import ExponentialGating

        gating = ExponentialGating(input_dim=64, hidden_dim=64)

        batch_size = 2
        x = torch.randn(batch_size, 64)
        h = torch.randn(batch_size, 64)

        i_gate, f_gate = gating(x, h)

        assert i_gate.shape == (batch_size, 64)
        assert f_gate.shape == (batch_size, 64)
        # Exponential gates should be positive
        assert (i_gate > 0).all()
        assert (f_gate > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
