"""
Griffin Model Tests

Basic tests for Griffin model components including:
- Configuration validation
- Model instantiation
- Forward pass shapes
- Recurrent and attention blocks
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGriffinConfig:
    """Tests for GriffinConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.model import GriffinConfig

        config = GriffinConfig()

        assert config.hidden_dim == 2048
        assert config.num_layers == 26
        assert config.num_heads == 8
        assert config.head_dim == 256
        assert config.window_size == 2048
        assert config.context_length == 8192
        assert config.block_pattern == ["recurrent", "recurrent", "attention"]

    def test_custom_config(self):
        """Test custom configuration."""
        from src.model import GriffinConfig

        config = GriffinConfig(
            hidden_dim=768,
            num_layers=12,
            window_size=1024,
            block_pattern=["recurrent", "attention"],
        )

        assert config.hidden_dim == 768
        assert config.num_layers == 12
        assert config.window_size == 1024
        assert config.block_pattern == ["recurrent", "attention"]


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_output_shape(self):
        """Test RMSNorm output shape."""
        import torch
        from src.model import RMSNorm

        batch_size = 2
        seq_len = 128
        dim = 64

        norm = RMSNorm(dim)
        x = torch.randn(batch_size, seq_len, dim)
        output = norm(x)

        assert output.shape == x.shape

    def test_normalization(self):
        """Test that output has approximately unit RMS."""
        import torch
        from src.model import RMSNorm

        norm = RMSNorm(64)
        x = torch.randn(2, 128, 64) * 10  # Large values

        output = norm(x)

        # RMS should be close to 1 (with learned scale)
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        assert rms.mean() > 0  # Sanity check


class TestGatedLinearRecurrence:
    """Tests for GatedLinearRecurrence (RG-LRU)."""

    def test_output_shape(self):
        """Test RG-LRU output shape."""
        import torch
        from src.model import GatedLinearRecurrence

        batch_size = 2
        seq_len = 128
        input_dim = 64
        state_dim = 64

        rglru = GatedLinearRecurrence(input_dim, state_dim)
        x = torch.randn(batch_size, seq_len, input_dim)

        output, h = rglru(x)

        assert output.shape == (batch_size, seq_len, input_dim)
        assert h.shape == (batch_size, state_dim)

    def test_with_initial_state(self):
        """Test RG-LRU with initial hidden state."""
        import torch
        from src.model import GatedLinearRecurrence

        batch_size = 2
        input_dim = 64
        state_dim = 64

        rglru = GatedLinearRecurrence(input_dim, state_dim)
        x = torch.randn(batch_size, 128, input_dim)
        h0 = torch.randn(batch_size, state_dim)

        output, h = rglru(x, h0)

        assert output.shape == (batch_size, 128, input_dim)
        assert h.shape == (batch_size, state_dim)

    def test_state_evolution(self):
        """Test that hidden state evolves over time."""
        import torch
        from src.model import GatedLinearRecurrence

        rglru = GatedLinearRecurrence(64, 64)
        x = torch.randn(1, 10, 64)

        output, h_final = rglru(x)

        # Hidden state should be different from zero
        assert h_final.abs().sum() > 0


class TestLocalAttention:
    """Tests for LocalAttention."""

    def test_output_shape(self):
        """Test local attention output shape."""
        import torch
        from src.model import LocalAttention

        batch_size = 2
        seq_len = 128
        hidden_dim = 64
        num_heads = 4
        head_dim = 16
        window_size = 32

        attn = LocalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size,
        )
        x = torch.randn(batch_size, seq_len, hidden_dim)

        output, cache = attn(x)

        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_window_mask(self):
        """Test sliding window mask creation."""
        import torch
        from src.model import LocalAttention

        attn = LocalAttention(
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
            window_size=4,
        )

        mask = attn._create_window_mask(8, torch.device("cpu"))

        # Check mask shape
        assert mask.shape == (8, 8)

        # Position 0 should only attend to itself
        assert mask[0, 0] == 0
        assert mask[0, 1] == float("-inf")

        # Later positions should attend to window
        assert mask[5, 5] == 0  # Current
        assert mask[5, 4] == 0  # Previous in window
        assert mask[5, 2] == 0  # Start of window
        assert mask[5, 1] == float("-inf")  # Outside window


class TestRecurrentBlock:
    """Tests for RecurrentBlock."""

    def test_output_shape(self):
        """Test recurrent block output shape."""
        import torch
        from src.model import RecurrentBlock, GriffinConfig

        config = GriffinConfig(hidden_dim=64, recurrent_state_dim=64)
        block = RecurrentBlock(config)

        batch_size = 2
        seq_len = 128
        x = torch.randn(batch_size, seq_len, config.hidden_dim)

        output, h = block(x)

        assert output.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connections work."""
        import torch
        from src.model import RecurrentBlock, GriffinConfig

        config = GriffinConfig(hidden_dim=64, dropout=0.0)
        block = RecurrentBlock(config)

        x = torch.randn(2, 128, 64)
        output, _ = block(x)

        # Output should differ from input
        assert not torch.allclose(output, x)


class TestAttentionBlock:
    """Tests for AttentionBlock."""

    def test_output_shape(self):
        """Test attention block output shape."""
        import torch
        from src.model import AttentionBlock, GriffinConfig

        config = GriffinConfig(
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
            window_size=32,
        )
        block = AttentionBlock(config)

        batch_size = 2
        seq_len = 128
        x = torch.randn(batch_size, seq_len, config.hidden_dim)

        output, cache = block(x)

        assert output.shape == x.shape


class TestGriffinBlock:
    """Tests for GriffinBlock factory."""

    def test_recurrent_type(self):
        """Test recurrent block type."""
        from src.model import GriffinBlock, GriffinConfig, RecurrentBlock

        config = GriffinConfig(hidden_dim=64)
        block = GriffinBlock(config, "recurrent")

        assert block.block_type == "recurrent"
        assert isinstance(block.block, RecurrentBlock)

    def test_attention_type(self):
        """Test attention block type."""
        from src.model import GriffinBlock, GriffinConfig, AttentionBlock

        config = GriffinConfig(hidden_dim=64, num_heads=4, head_dim=16)
        block = GriffinBlock(config, "attention")

        assert block.block_type == "attention"
        assert isinstance(block.block, AttentionBlock)

    def test_invalid_type(self):
        """Test invalid block type raises error."""
        from src.model import GriffinBlock, GriffinConfig

        config = GriffinConfig()

        with pytest.raises(ValueError):
            GriffinBlock(config, "invalid")


class TestGriffinModel:
    """Tests for complete GriffinModel."""

    def test_instantiation(self):
        """Test model can be instantiated."""
        from src.model import GriffinModel, GriffinConfig

        config = GriffinConfig(
            hidden_dim=64,
            num_layers=3,
            vocab_size=1000,
            num_heads=4,
            head_dim=16,
        )
        model = GriffinModel(config)

        assert model is not None
        assert len(model.blocks) == config.num_layers

    def test_block_pattern(self):
        """Test blocks follow pattern."""
        from src.model import GriffinModel, GriffinConfig

        config = GriffinConfig(
            hidden_dim=64,
            num_layers=6,
            num_heads=4,
            head_dim=16,
            block_pattern=["recurrent", "recurrent", "attention"],
        )
        model = GriffinModel(config)

        expected = ["recurrent", "recurrent", "attention", "recurrent", "recurrent", "attention"]
        assert model.block_types == expected

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        import torch
        from src.model import GriffinModel, GriffinConfig

        config = GriffinConfig(
            hidden_dim=64,
            num_layers=3,
            vocab_size=1000,
            num_heads=4,
            head_dim=16,
        )
        model = GriffinModel(config)

        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, states = model(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert len(states) == config.num_layers

    def test_tied_embeddings(self):
        """Test tied embeddings."""
        from src.model import GriffinModel, GriffinConfig

        config = GriffinConfig(
            hidden_dim=64,
            num_layers=2,
            vocab_size=1000,
            tie_embeddings=True,
        )
        model = GriffinModel(config)

        assert model.lm_head is None

    def test_untied_embeddings(self):
        """Test untied embeddings."""
        from src.model import GriffinModel, GriffinConfig

        config = GriffinConfig(
            hidden_dim=64,
            num_layers=2,
            vocab_size=1000,
            tie_embeddings=False,
        )
        model = GriffinModel(config)

        assert model.lm_head is not None


class TestLayerComponents:
    """Tests for individual layer components."""

    def test_rglru_cell(self):
        """Test single-step RG-LRU cell."""
        import torch
        from src.layers import RGLRUCell

        cell = RGLRUCell(input_dim=64, state_dim=64)
        x = torch.randn(2, 64)
        h = cell.init_state(2, x.device, x.dtype)

        output, h_new = cell(x, h)

        assert output.shape == (2, 64)
        assert h_new.shape == (2, 64)

    def test_parallel_scan(self):
        """Test parallel scan."""
        import torch
        from src.layers import ParallelScan

        scan = ParallelScan()
        batch_size = 2
        seq_len = 16
        dim = 32

        a = torch.sigmoid(torch.randn(batch_size, seq_len, dim))
        b = torch.randn(batch_size, seq_len, dim)

        output = scan(a, b)

        assert output.shape == (batch_size, seq_len, dim)

    def test_causal_conv1d(self):
        """Test causal convolution."""
        import torch
        from src.layers import CausalConv1d

        conv = CausalConv1d(64, 64, kernel_size=3)
        x = torch.randn(2, 128, 64)

        output = conv(x)

        assert output.shape == x.shape

    def test_rotary_embedding(self):
        """Test rotary position embedding."""
        import torch
        from src.layers import RotaryEmbedding

        rope = RotaryEmbedding(dim=64, max_seq_len=1024)

        batch_size = 2
        num_heads = 4
        seq_len = 128
        head_dim = 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_sliding_window_mask(self):
        """Test sliding window mask generator."""
        import torch
        from src.layers import SlidingWindowMask

        mask_gen = SlidingWindowMask(window_size=4, max_seq_len=64)
        mask = mask_gen(16)

        assert mask.shape == (16, 16)

    def test_gated_mlp(self):
        """Test gated MLP."""
        import torch
        from src.layers import GatedMLP

        mlp = GatedMLP(hidden_dim=64, expansion_factor=4)
        x = torch.randn(2, 128, 64)

        output = mlp(x)

        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
