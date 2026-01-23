"""
Tests for Tiny Recursive Model (TRM)

Run with: pytest tests/test_trm.py -v
"""

import pytest
import torch
import torch.nn as nn

from ..src.model import TRM, TRMConfig
from ..src.layers import (
    TRMBlock,
    DeepRecursion,
    QHead,
    OutputHead,
    GridEmbedding,
    MLPSequence,
    RMSNorm,
    SwiGLU,
    MultiHeadAttention,
    RotaryEmbedding,
)


class TestTRMConfig:
    """Tests for TRMConfig."""

    def test_default_config(self):
        config = TRMConfig()
        assert config.grid_size == 9
        assert config.vocab_size == 10
        assert config.embed_dim == 512
        assert config.n_layers == 2
        assert config.T_cycles == 3
        assert config.n_cycles == 6

    def test_effective_depth(self):
        config = TRMConfig(T_cycles=3, n_cycles=6, n_layers=2)
        # 3 * (6 + 1) * 2 = 42
        assert config.effective_depth == 42

    def test_sudoku_preset(self):
        config = TRMConfig.for_sudoku()
        assert config.grid_size == 9
        assert config.vocab_size == 10
        assert config.use_attention is False

    def test_arc_agi_preset(self):
        config = TRMConfig.for_arc_agi()
        assert config.grid_size == 30
        assert config.vocab_size == 11
        assert config.use_attention is True


class TestLayers:
    """Tests for TRM layer components."""

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def seq_len(self):
        return 81  # 9x9 grid

    @pytest.fixture
    def embed_dim(self):
        return 64  # Smaller for testing

    def test_rms_norm(self, batch_size, seq_len, embed_dim):
        norm = RMSNorm(embed_dim)
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = norm(x)
        assert output.shape == x.shape

    def test_swiglu(self, batch_size, seq_len, embed_dim):
        hidden_dim = embed_dim * 4
        swiglu = SwiGLU(embed_dim, hidden_dim, embed_dim)
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = swiglu(x)
        assert output.shape == x.shape

    def test_rotary_embedding(self, batch_size, seq_len, embed_dim):
        head_dim = embed_dim // 8
        rope = RotaryEmbedding(head_dim)
        x = torch.randn(batch_size, seq_len, embed_dim)
        cos, sin = rope(x, seq_len)
        assert cos.shape == (seq_len, head_dim)
        assert sin.shape == (seq_len, head_dim)

    def test_multi_head_attention(self, batch_size, seq_len, embed_dim):
        attn = MultiHeadAttention(embed_dim, n_heads=8, use_rotary=True)
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = attn(x)
        assert output.shape == x.shape

    def test_trm_block(self, batch_size, seq_len, embed_dim):
        block = TRMBlock(embed_dim, n_heads=8, use_attention=True)
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = block(x)
        assert output.shape == x.shape

    def test_trm_block_mlp_only(self, batch_size, seq_len, embed_dim):
        block = TRMBlock(embed_dim, n_heads=8, use_attention=False)
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = block(x)
        assert output.shape == x.shape

    def test_mlp_sequence(self, batch_size, seq_len, embed_dim):
        mlp_seq = MLPSequence(embed_dim, n_layers=2)
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = mlp_seq(x)
        assert output.shape == x.shape

    def test_grid_embedding(self, batch_size, embed_dim):
        grid_size = 9
        vocab_size = 10
        embedding = GridEmbedding(vocab_size, embed_dim, grid_size)

        # Test 2D input
        x_2d = torch.randint(0, vocab_size, (batch_size, grid_size, grid_size))
        output_2d = embedding(x_2d)
        assert output_2d.shape == (batch_size, grid_size * grid_size, embed_dim)

        # Test 1D input
        x_1d = torch.randint(0, vocab_size, (batch_size, grid_size * grid_size))
        output_1d = embedding(x_1d)
        assert output_1d.shape == (batch_size, grid_size * grid_size, embed_dim)

    def test_output_head(self, batch_size, seq_len, embed_dim):
        vocab_size = 10
        head = OutputHead(embed_dim, vocab_size)
        x = torch.randn(batch_size, seq_len, embed_dim)
        logits = head(x)
        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_q_head(self, batch_size, seq_len, embed_dim):
        head = QHead(embed_dim)
        z = torch.randn(batch_size, seq_len, embed_dim)
        q_hat = head(z)
        assert q_hat.shape == (batch_size,)


class TestDeepRecursion:
    """Tests for DeepRecursion module."""

    @pytest.fixture
    def recursion_module(self):
        return DeepRecursion(
            embed_dim=64,
            n_layers=2,
            n_heads=4,
            T_cycles=2,
            n_cycles=3,
            use_attention=True,
        )

    def test_forward(self, recursion_module):
        batch_size = 2
        seq_len = 81
        embed_dim = 64

        x = torch.randn(batch_size, seq_len, embed_dim)
        y = torch.randn(batch_size, seq_len, embed_dim)
        z = torch.randn(batch_size, seq_len, embed_dim)

        y_out, z_out = recursion_module(x, y, z)

        assert y_out.shape == y.shape
        assert z_out.shape == z.shape

    def test_forward_single_step(self, recursion_module):
        batch_size = 2
        seq_len = 81
        embed_dim = 64

        x = torch.randn(batch_size, seq_len, embed_dim)
        y = torch.randn(batch_size, seq_len, embed_dim)
        z = torch.randn(batch_size, seq_len, embed_dim)

        y_out, z_out = recursion_module.forward_single_step(x, y, z)

        assert y_out.shape == y.shape
        assert z_out.shape == z.shape


class TestTRM:
    """Tests for TRM model."""

    @pytest.fixture
    def small_config(self):
        return TRMConfig(
            grid_size=9,
            vocab_size=10,
            embed_dim=64,
            n_layers=2,
            n_heads=4,
            T_cycles=2,
            n_cycles=2,
            max_supervision_steps=4,
        )

    @pytest.fixture
    def model(self, small_config):
        return TRM(small_config)

    def test_forward(self, model, small_config):
        batch_size = 2
        grid_size = small_config.grid_size

        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, grid_size, grid_size))
        output = model(input_ids)

        assert "logits" in output
        assert "q_hat" in output
        assert output["logits"].shape == (batch_size, grid_size * grid_size, small_config.vocab_size)
        assert output["q_hat"].shape == (batch_size,)

    def test_forward_with_labels(self, model, small_config):
        batch_size = 2
        grid_size = small_config.grid_size

        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, grid_size, grid_size))
        labels = torch.randint(0, small_config.vocab_size, (batch_size, grid_size, grid_size))

        output = model(input_ids, labels=labels)

        assert "loss" in output
        assert "ce_loss" in output
        assert "q_loss" in output
        assert output["loss"].dim() == 0  # Scalar

    def test_train_step(self, model, small_config):
        batch_size = 2
        grid_size = small_config.grid_size

        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, grid_size, grid_size))
        labels = torch.randint(0, small_config.vocab_size, (batch_size, grid_size, grid_size))

        model.train()
        output = model.train_step(input_ids, labels, max_steps=2)

        assert "loss" in output
        assert "steps" in output
        assert "accuracy" in output
        assert 1 <= output["steps"] <= 2

    def test_solve(self, model, small_config):
        batch_size = 2
        grid_size = small_config.grid_size

        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, grid_size, grid_size))

        model.eval()
        result = model.solve(input_ids, max_steps=2)

        assert "solution" in result
        assert "confidence" in result
        assert "steps" in result
        assert result["solution"].shape == (batch_size, grid_size * grid_size)

    def test_solve_with_trajectory(self, model, small_config):
        batch_size = 2
        grid_size = small_config.grid_size

        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, grid_size, grid_size))

        model.eval()
        result = model.solve(input_ids, max_steps=2, return_trajectory=True)

        assert "trajectory" in result
        assert len(result["trajectory"]) == result["steps"]
        for step_info in result["trajectory"]:
            assert "step" in step_info
            assert "prediction" in step_info
            assert "q_hat" in step_info

    def test_num_parameters(self, model):
        params = model.num_parameters()
        assert params > 0
        assert isinstance(params, int)

    def test_save_load(self, model, small_config, tmp_path):
        # Save
        save_path = tmp_path / "model.pt"
        model.save_pretrained(str(save_path))

        # Load
        loaded_model = TRM.from_pretrained(str(save_path))

        # Check config matches
        assert loaded_model.config.grid_size == small_config.grid_size
        assert loaded_model.config.vocab_size == small_config.vocab_size
        assert loaded_model.config.embed_dim == small_config.embed_dim

        # Check outputs match
        input_ids = torch.randint(0, small_config.vocab_size, (1, small_config.grid_size, small_config.grid_size))

        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            out1 = model(input_ids)
            out2 = loaded_model(input_ids)

        assert torch.allclose(out1["logits"], out2["logits"], atol=1e-5)


class TestGradientFlow:
    """Tests for gradient flow through recursion."""

    def test_gradient_flows_through_recursion(self):
        config = TRMConfig(
            grid_size=4,
            vocab_size=5,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            T_cycles=2,
            n_cycles=2,
        )
        model = TRM(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 4, 4))
        labels = torch.randint(0, config.vocab_size, (2, 4, 4))

        model.train()
        output = model(input_ids, labels=labels)
        output["loss"].backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
