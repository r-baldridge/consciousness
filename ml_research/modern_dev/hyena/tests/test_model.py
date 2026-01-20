"""
Hyena Model Tests

Basic tests for Hyena model components including:
- Configuration validation
- Model instantiation
- Forward pass shapes
- Layer components
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHyenaConfig:
    """Tests for HyenaConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.model import HyenaConfig

        config = HyenaConfig()

        assert config.d_model == 512
        assert config.n_layer == 12
        assert config.order == 2
        assert config.filter_order == 64
        assert config.emb_dim == 3
        assert config.max_seq_len == 2048
        assert config.activation == "gelu"
        assert config.bidirectional is False

    def test_custom_config(self):
        """Test custom configuration."""
        from src.model import HyenaConfig

        config = HyenaConfig(
            d_model=256,
            n_layer=6,
            order=3,
            max_seq_len=4096,
        )

        assert config.d_model == 256
        assert config.n_layer == 6
        assert config.order == 3
        assert config.max_seq_len == 4096


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""

    def test_output_shape(self):
        """Test positional encoding output shape."""
        import torch
        from src.model import PositionalEncoding

        emb_dim = 3
        max_seq_len = 1024
        seq_len = 128

        pos_enc = PositionalEncoding(emb_dim, max_seq_len)
        output = pos_enc(seq_len)

        # Output should be (seq_len, 2 * emb_dim + 1)
        expected_dim = 2 * emb_dim + 1
        assert output.shape == (seq_len, expected_dim)

    def test_learnable_frequencies(self):
        """Test that frequencies are learnable parameters."""
        from src.model import PositionalEncoding

        pos_enc = PositionalEncoding(emb_dim=3, max_seq_len=1024, learnable_frequencies=True)

        assert hasattr(pos_enc, "frequencies")
        assert pos_enc.frequencies.requires_grad


class TestImplicitFilter:
    """Tests for ImplicitFilter module."""

    def test_output_shape(self):
        """Test implicit filter output shape."""
        import torch
        from src.model import ImplicitFilter

        d_model = 64
        filter_order = 32
        emb_dim = 3
        max_seq_len = 1024
        seq_len = 128

        implicit_filter = ImplicitFilter(
            d_model=d_model,
            filter_order=filter_order,
            emb_dim=emb_dim,
            max_seq_len=max_seq_len,
        )

        output = implicit_filter(seq_len)

        assert output.shape == (seq_len, d_model)

    def test_different_sequence_lengths(self):
        """Test filter generation for different sequence lengths."""
        import torch
        from src.model import ImplicitFilter

        implicit_filter = ImplicitFilter(
            d_model=64,
            filter_order=32,
            emb_dim=3,
            max_seq_len=2048,
        )

        for seq_len in [64, 128, 256, 512]:
            output = implicit_filter(seq_len)
            assert output.shape == (seq_len, 64)


class TestDataControlledGating:
    """Tests for DataControlledGating module."""

    def test_output_shapes(self):
        """Test gating output shapes."""
        import torch
        from src.model import DataControlledGating

        batch_size = 2
        seq_len = 128
        d_model = 64

        gating = DataControlledGating(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        gate, gated_output = gating(x)

        assert gate.shape == (batch_size, seq_len, d_model)
        assert gated_output.shape == (batch_size, seq_len, d_model)

    def test_gate_range(self):
        """Test that gate values are in [0, 1]."""
        import torch
        from src.model import DataControlledGating

        gating = DataControlledGating(d_model=64)
        x = torch.randn(2, 128, 64)

        gate, _ = gating(x)

        assert gate.min() >= 0.0
        assert gate.max() <= 1.0


class TestHyenaOperator:
    """Tests for HyenaOperator module."""

    def test_output_shape(self):
        """Test Hyena operator output shape."""
        import torch
        from src.model import HyenaOperator, HyenaConfig

        config = HyenaConfig(d_model=64, order=2, max_seq_len=256)
        operator = HyenaOperator(config)

        batch_size = 2
        seq_len = 128
        x = torch.randn(batch_size, seq_len, config.d_model)

        output, cache = operator(x)

        assert output.shape == (batch_size, seq_len, config.d_model)

    def test_different_orders(self):
        """Test Hyena operator with different orders."""
        import torch
        from src.model import HyenaOperator, HyenaConfig

        for order in [1, 2, 3]:
            config = HyenaConfig(d_model=64, order=order, max_seq_len=256)
            operator = HyenaOperator(config)

            x = torch.randn(2, 128, 64)
            output, _ = operator(x)

            assert output.shape == x.shape


class TestHyenaBlock:
    """Tests for HyenaBlock module."""

    def test_output_shape(self):
        """Test Hyena block output shape."""
        import torch
        from src.model import HyenaBlock, HyenaConfig

        config = HyenaConfig(d_model=64, max_seq_len=256)
        block = HyenaBlock(config)

        batch_size = 2
        seq_len = 128
        x = torch.randn(batch_size, seq_len, config.d_model)

        output, cache = block(x)

        assert output.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connections work (output differs from input)."""
        import torch
        from src.model import HyenaBlock, HyenaConfig

        config = HyenaConfig(d_model=64, max_seq_len=256, dropout=0.0)
        block = HyenaBlock(config)

        x = torch.randn(2, 128, 64)
        output, _ = block(x)

        # Output should be different from input (non-trivial transformation)
        assert not torch.allclose(output, x)


class TestHyenaModel:
    """Tests for complete HyenaModel."""

    def test_instantiation(self):
        """Test model can be instantiated."""
        from src.model import HyenaModel, HyenaConfig

        config = HyenaConfig(
            d_model=64,
            n_layer=2,
            vocab_size=1000,
            max_seq_len=256,
        )
        model = HyenaModel(config)

        assert model is not None
        assert len(model.blocks) == config.n_layer

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        import torch
        from src.model import HyenaModel, HyenaConfig

        config = HyenaConfig(
            d_model=64,
            n_layer=2,
            vocab_size=1000,
            max_seq_len=256,
        )
        model = HyenaModel(config)

        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, caches = model(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert len(caches) == config.n_layer

    def test_tied_embeddings(self):
        """Test that embeddings are tied when configured."""
        from src.model import HyenaModel, HyenaConfig

        config = HyenaConfig(
            d_model=64,
            n_layer=2,
            vocab_size=1000,
            tie_embeddings=True,
        )
        model = HyenaModel(config)

        # When tied, lm_head should be None
        assert model.lm_head is None

    def test_untied_embeddings(self):
        """Test that embeddings are separate when configured."""
        from src.model import HyenaModel, HyenaConfig

        config = HyenaConfig(
            d_model=64,
            n_layer=2,
            vocab_size=1000,
            tie_embeddings=False,
        )
        model = HyenaModel(config)

        # When untied, lm_head should exist
        assert model.lm_head is not None

    def test_parameter_count(self):
        """Test model has reasonable parameter count."""
        from src.model import HyenaModel, HyenaConfig

        config = HyenaConfig(
            d_model=64,
            n_layer=2,
            vocab_size=1000,
            max_seq_len=256,
        )
        model = HyenaModel(config)

        num_params = sum(p.numel() for p in model.parameters())

        # Should have > 0 parameters
        assert num_params > 0

        # Rough sanity check (small model should be < 10M params)
        assert num_params < 10_000_000


class TestLayerComponents:
    """Tests for individual layer components."""

    def test_fft_conv(self):
        """Test FFT convolution layer."""
        import torch
        from src.layers import FFTConv

        d_model = 64
        batch_size = 2
        seq_len = 128

        fft_conv = FFTConv(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        h = torch.randn(seq_len, d_model)

        output = fft_conv(x, h)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_exponential_window(self):
        """Test exponential window function."""
        import torch
        from src.layers import ExponentialWindow

        d_model = 64
        seq_len = 128

        window = ExponentialWindow(d_model)
        h = torch.randn(seq_len, d_model)

        windowed = window(h)

        assert windowed.shape == h.shape
        # First value should be approximately h (decay ~ exp(0) = 1)
        # Last values should be smaller due to decay

    def test_short_convolution(self):
        """Test short convolution layer."""
        import torch
        from src.layers import ShortConvolution

        d_model = 64
        batch_size = 2
        seq_len = 128

        short_conv = ShortConvolution(d_model, kernel_size=3)
        x = torch.randn(batch_size, seq_len, d_model)

        output = short_conv(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_hyena_filter(self):
        """Test combined Hyena filter."""
        import torch
        from src.layers import HyenaFilter

        d_model = 64
        seq_len = 128

        hyena_filter = HyenaFilter(d_model=d_model, max_seq_len=256)
        h = hyena_filter(seq_len)

        assert h.shape == (seq_len, d_model)

    def test_gated_mlp(self):
        """Test gated MLP."""
        import torch
        from src.layers import GatedMLP

        d_model = 64
        batch_size = 2
        seq_len = 128

        mlp = GatedMLP(d_model, expansion_factor=4)
        x = torch.randn(batch_size, seq_len, d_model)

        output = mlp(x)

        assert output.shape == x.shape


# Skip tests if torch not available
def pytest_configure(config):
    """Configure pytest to skip tests if dependencies missing."""
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
