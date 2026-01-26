"""
Tests for Code Repair TRM Model (64x48 Grid Architecture)

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import torch.nn as nn

from ..src.model import CodeRepairTRM, CodeRepairConfig, CodeRepairDeepRecursion
from ..src.layers import (
    GridPositionalEncoding,
    GridAttention,
    RecursiveBlock,
    FeedForward,
    IterationController,
    RMSNorm,
)


class TestCodeRepairConfig:
    """Tests for CodeRepairConfig."""

    def test_default_config(self):
        config = CodeRepairConfig()
        assert config.grid_height == 64
        assert config.grid_width == 48
        assert config.vocab_size == 32768
        assert config.embed_dim == 256
        assert config.n_heads == 8
        assert config.max_iterations == 8

    def test_max_seq_len(self):
        config = CodeRepairConfig()
        assert config.max_seq_len == 64 * 48  # 3072

    def test_effective_depth(self):
        config = CodeRepairConfig(max_iterations=8, n_blocks=6)
        # 8 * 6 = 48
        assert config.effective_depth == 48

    def test_small_preset(self):
        config = CodeRepairConfig.for_code_repair_small()
        assert config.embed_dim == 192
        assert config.n_blocks == 4

    def test_base_preset(self):
        config = CodeRepairConfig.for_code_repair_base()
        assert config.embed_dim == 256
        assert config.n_blocks == 6

    def test_large_preset(self):
        config = CodeRepairConfig.for_code_repair_large()
        assert config.embed_dim == 384
        assert config.n_blocks == 8


class TestGridPositionalEncoding:
    """Tests for GridPositionalEncoding."""

    @pytest.fixture
    def pos_encoding(self):
        return GridPositionalEncoding(
            embed_dim=64,
            max_height=64,
            max_width=48,
        )

    def test_forward_3d_input(self, pos_encoding):
        batch_size = 2
        x = torch.randint(0, 100, (batch_size, 64, 48))
        output = pos_encoding(x)
        assert output.shape == (batch_size, 64, 48, 64)

    def test_forward_4d_input(self, pos_encoding):
        batch_size = 2
        x = torch.randn(batch_size, 64, 48, 64)
        output = pos_encoding(x)
        assert output.shape == (batch_size, 64, 48, 64)

    def test_smaller_grid(self, pos_encoding):
        batch_size = 2
        x = torch.randint(0, 100, (batch_size, 32, 24))
        output = pos_encoding(x)
        assert output.shape == (batch_size, 32, 24, 64)


class TestFeedForward:
    """Tests for FeedForward with SwiGLU."""

    def test_swiglu_forward(self):
        ffn = FeedForward(embed_dim=64, hidden_dim=256, use_swiglu=True)
        x = torch.randn(2, 64, 48, 64)
        output = ffn(x)
        assert output.shape == x.shape

    def test_gelu_forward(self):
        ffn = FeedForward(embed_dim=64, hidden_dim=256, use_swiglu=False)
        x = torch.randn(2, 64, 48, 64)
        output = ffn(x)
        assert output.shape == x.shape

    def test_default_hidden_dim(self):
        ffn = FeedForward(embed_dim=64)
        x = torch.randn(2, 64, 48, 64)
        output = ffn(x)
        assert output.shape == x.shape


class TestGridAttention:
    """Tests for GridAttention with 2D relative position encoding."""

    @pytest.fixture
    def attention(self):
        return GridAttention(
            embed_dim=64,
            n_heads=4,
            max_height=64,
            max_width=48,
        )

    def test_forward_shape(self, attention):
        batch_size = 2
        x = torch.randn(batch_size, 64, 48, 64)
        output = attention(x)
        assert output.shape == x.shape

    def test_forward_with_mask(self, attention):
        batch_size = 2
        x = torch.randn(batch_size, 64, 48, 64)
        mask = torch.ones(batch_size, 64, 48)
        mask[:, :10, :] = 0  # Mask first 10 rows
        output = attention(x, mask)
        assert output.shape == x.shape

    def test_causal_attention(self, attention):
        batch_size = 2
        x = torch.randn(batch_size, 64, 48, 64)
        output = attention(x, causal=True)
        assert output.shape == x.shape

    def test_smaller_grid(self, attention):
        batch_size = 2
        x = torch.randn(batch_size, 16, 12, 64)
        output = attention(x)
        assert output.shape == x.shape

    def test_relative_position_bias(self, attention):
        bias = attention._compute_relative_position_bias(8, 8, torch.device('cpu'))
        assert bias.shape == (4, 64, 64)  # (n_heads, seq_len, seq_len)


class TestRecursiveBlock:
    """Tests for RecursiveBlock."""

    @pytest.fixture
    def block(self):
        return RecursiveBlock(
            embed_dim=64,
            n_heads=4,
            ffn_dim=256,
            max_height=64,
            max_width=48,
        )

    def test_forward(self, block):
        batch_size = 2
        x = torch.randn(batch_size, 64, 48, 64)
        output = block(x)
        assert output.shape == x.shape

    def test_forward_with_mask(self, block):
        batch_size = 2
        x = torch.randn(batch_size, 64, 48, 64)
        mask = torch.ones(batch_size, 64, 48)
        output = block(x, mask)
        assert output.shape == x.shape


class TestIterationController:
    """Tests for IterationController."""

    def test_min_iterations(self):
        controller = IterationController(
            q_threshold=0.95,
            min_iterations=2,
            max_iterations=8,
        )
        hidden = torch.randn(2, 64, 48, 64)

        # Should not stop before min iterations
        should_stop, _ = controller.should_stop(hidden, iteration=0)
        assert should_stop is False

    def test_max_iterations(self):
        controller = IterationController(
            q_threshold=0.95,
            min_iterations=2,
            max_iterations=8,
        )
        hidden = torch.randn(2, 64, 48, 64)

        # Should always stop at max iterations
        should_stop, _ = controller.should_stop(hidden, iteration=7)
        assert should_stop is True

    def test_confidence_computation(self):
        controller = IterationController()
        hidden = torch.randn(2, 64, 48, 64)
        confidence = controller._compute_confidence(hidden)

        assert confidence.shape == (2,)
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_early_stopping_with_q_hat(self):
        controller = IterationController(q_threshold=0.5, min_iterations=1)
        hidden = torch.randn(2, 64, 48, 64)

        # High confidence q_hat should trigger stop
        q_hat = torch.tensor([2.0, 2.0])  # sigmoid(2.0) > 0.5
        should_stop, confidence = controller.should_stop(hidden, iteration=2, q_hat=q_hat)
        assert should_stop is True


class TestCodeRepairDeepRecursion:
    """Tests for CodeRepairDeepRecursion."""

    @pytest.fixture
    def small_config(self):
        return CodeRepairConfig(
            grid_height=16,
            grid_width=12,
            embed_dim=64,
            n_heads=4,
            ffn_dim=256,
            n_blocks=2,
            max_iterations=4,
            min_iterations=1,
        )

    @pytest.fixture
    def recursion_module(self, small_config):
        return CodeRepairDeepRecursion(small_config)

    def test_forward_shape(self, recursion_module):
        batch_size = 2
        x = torch.randn(batch_size, 16, 12, 64)
        output, info = recursion_module(x)

        assert output.shape == x.shape
        assert "iterations" in info
        assert "confidence" in info
        assert "q_hat" in info

    def test_iterations_range(self, recursion_module):
        batch_size = 2
        x = torch.randn(batch_size, 16, 12, 64)
        _, info = recursion_module(x)

        assert 1 <= info["iterations"] <= 4

    def test_all_iterations_returned(self, recursion_module):
        batch_size = 2
        x = torch.randn(batch_size, 16, 12, 64)
        _, info = recursion_module(x, return_all_iterations=True)

        assert "all_states" in info
        assert len(info["all_states"]) == info["iterations"]

    def test_training_mode_full_iterations(self, recursion_module):
        recursion_module.train()
        batch_size = 2
        x = torch.randn(batch_size, 16, 12, 64)
        _, info = recursion_module(x)

        # In training mode, should complete all iterations
        assert info["iterations"] == 4


class TestCodeRepairTRM:
    """Tests for CodeRepairTRM model."""

    @pytest.fixture
    def small_config(self):
        return CodeRepairConfig(
            grid_height=16,
            grid_width=12,
            vocab_size=1000,
            embed_dim=64,
            n_heads=4,
            ffn_dim=256,
            n_blocks=2,
            max_iterations=4,
            min_iterations=1,
        )

    @pytest.fixture
    def model(self, small_config):
        return CodeRepairTRM(small_config)

    def test_forward(self, model, small_config):
        batch_size = 2
        x = torch.randint(0, small_config.vocab_size, (batch_size, 16, 12))
        logits, info = model(x)

        assert logits.shape == (batch_size, 16, 12, small_config.vocab_size)
        assert "iterations" in info
        assert "confidence" in info

    def test_forward_with_mask(self, model, small_config):
        batch_size = 2
        x = torch.randint(0, small_config.vocab_size, (batch_size, 16, 12))
        mask = torch.ones(batch_size, 16, 12)
        logits, info = model(x, mask=mask)

        assert logits.shape == (batch_size, 16, 12, small_config.vocab_size)

    def test_forward_with_labels(self, model, small_config):
        batch_size = 2
        x = torch.randint(0, small_config.vocab_size, (batch_size, 16, 12))
        labels = torch.randint(0, small_config.vocab_size, (batch_size, 16, 12))
        logits, info = model(x, labels=labels)

        assert "loss" in info
        assert info["loss"].dim() == 0  # Scalar

    def test_generate(self, model, small_config):
        batch_size = 2
        x = torch.randint(0, small_config.vocab_size, (batch_size, 16, 12))

        model.eval()
        result = model.generate(x)

        assert "output" in result
        assert "logits" in result
        assert "iterations" in result
        assert "confidence" in result
        assert result["output"].shape == (batch_size, 16, 12)

    def test_num_parameters(self, model):
        params = model.num_parameters()
        assert params > 0
        assert isinstance(params, int)

    def test_early_stopping(self, model, small_config):
        batch_size = 2
        x = torch.randint(0, small_config.vocab_size, (batch_size, 16, 12))

        model.eval()
        _, info = model(x)

        # Check iterations is within bounds
        assert 1 <= info["iterations"] <= 4


class TestCodeRepairTRMFullSize:
    """Tests for CodeRepairTRM with full 64x48 grid."""

    @pytest.fixture
    def config(self):
        return CodeRepairConfig(
            grid_height=64,
            grid_width=48,
            vocab_size=32768,
            embed_dim=128,  # Smaller for testing
            n_heads=4,
            ffn_dim=512,
            n_blocks=2,
            max_iterations=2,
        )

    @pytest.fixture
    def model(self, config):
        return CodeRepairTRM(config)

    def test_forward_full_size(self, model, config):
        batch_size = 2
        x = torch.randint(0, config.vocab_size, (batch_size, 64, 48))
        logits, info = model(x)

        assert logits.shape == (batch_size, 64, 48, config.vocab_size)

    def test_acceptance_criteria(self, model, config):
        """Test the acceptance criteria from the task specification."""
        batch_size = 2
        x = torch.randint(0, config.vocab_size, (batch_size, 64, 48))
        mask = torch.ones(batch_size, 64, 48)

        output, info = model(x, mask)

        # Criterion 1: Output shape
        assert output.shape == (batch_size, 64, 48, config.vocab_size)

        # Criterion 2: Early stopping works
        assert 1 <= info["iterations"] <= config.max_iterations


class TestGradientFlow:
    """Tests for gradient flow through the code repair model."""

    def test_gradient_flows_through_recursion(self):
        config = CodeRepairConfig(
            grid_height=8,
            grid_width=8,
            vocab_size=100,
            embed_dim=32,
            n_heads=2,
            ffn_dim=128,
            n_blocks=2,
            max_iterations=2,
        )
        model = CodeRepairTRM(config)

        x = torch.randint(0, config.vocab_size, (2, 8, 8))
        labels = torch.randint(0, config.vocab_size, (2, 8, 8))

        model.train()
        _, info = model(x, labels=labels)
        info["loss"].backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_gradient_checkpointing(self):
        config = CodeRepairConfig(
            grid_height=8,
            grid_width=8,
            vocab_size=100,
            embed_dim=32,
            n_heads=2,
            ffn_dim=128,
            n_blocks=2,
            max_iterations=2,
            use_gradient_checkpointing=True,
        )
        model = CodeRepairTRM(config)

        x = torch.randint(0, config.vocab_size, (2, 8, 8))
        labels = torch.randint(0, config.vocab_size, (2, 8, 8))

        model.train()
        _, info = model(x, labels=labels)
        info["loss"].backward()

        # Check gradients exist with checkpointing
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name} with checkpointing"


class TestSaveLoad:
    """Tests for model saving and loading."""

    def test_save_load(self, tmp_path):
        config = CodeRepairConfig(
            grid_height=8,
            grid_width=8,
            vocab_size=100,
            embed_dim=32,
            n_heads=2,
            ffn_dim=128,
            n_blocks=2,
        )
        model = CodeRepairTRM(config)

        # Save
        save_path = tmp_path / "model.pt"
        model.save_pretrained(str(save_path))

        # Load
        loaded_model = CodeRepairTRM.from_pretrained(str(save_path))

        # Check config matches
        assert loaded_model.config.grid_height == config.grid_height
        assert loaded_model.config.grid_width == config.grid_width
        assert loaded_model.config.vocab_size == config.vocab_size

        # Check outputs match
        x = torch.randint(0, config.vocab_size, (1, 8, 8))

        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            out1, _ = model(x)
            out2, _ = loaded_model(x)

        assert torch.allclose(out1, out2, atol=1e-5)


class TestMemoryUsage:
    """Tests for memory usage estimation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_usage_batch_32(self):
        """Test memory usage for batch_size=32."""
        config = CodeRepairConfig.for_code_repair_base()
        model = CodeRepairTRM(config).cuda()

        # Measure memory
        torch.cuda.reset_peak_memory_stats()

        x = torch.randint(0, config.vocab_size, (32, 64, 48)).cuda()

        with torch.no_grad():
            output, info = model(x)

        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak memory usage (batch_size=32): {peak_memory_mb:.2f} MB")

        # Just verify it runs without OOM
        assert output.shape == (32, 64, 48, config.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
