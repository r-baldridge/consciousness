"""
Comprehensive tests for TRM (Tiny Recursive Model) components.

Tests cover:
- Model architecture and forward pass
- Recursive iteration mechanism
- Early stopping based on confidence
- Gradient flow and training
- Loss functions
- Checkpoint save/load
- Curriculum learning

Run with: pytest tests/test_trm_full.py -v --cov=trm
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Dict, List, Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# TRM Model Architecture Tests
# =============================================================================


class TestTRMModel:
    """Tests for TRM model architecture."""

    def test_forward_shape(self, trm_model, trm_small_config):
        """Test output shapes match expected."""
        batch_size = 2
        x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))

        logits, info = trm_model(x)

        assert logits.shape == (batch_size, 16, 12, trm_small_config.vocab_size)
        assert "iterations" in info
        assert "confidence" in info
        assert "q_hat" in info

    def test_forward_with_different_batch_sizes(self, trm_model, trm_small_config):
        """Test model handles various batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))
            logits, info = trm_model(x)
            assert logits.shape[0] == batch_size

    def test_forward_with_mask(self, trm_model, trm_small_config):
        """Test forward pass with attention mask."""
        batch_size = 2
        x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))
        mask = torch.ones(batch_size, 16, 12)
        mask[:, 8:, :] = 0  # Mask second half

        logits, info = trm_model(x, mask=mask)
        assert logits.shape == (batch_size, 16, 12, trm_small_config.vocab_size)

    def test_forward_with_labels_computes_loss(self, trm_model, trm_small_config):
        """Test loss computation when labels provided."""
        batch_size = 2
        x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))
        labels = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))

        logits, info = trm_model(x, labels=labels)

        assert "loss" in info
        assert info["loss"].dim() == 0  # Scalar
        assert not torch.isnan(info["loss"])
        assert info["loss"] > 0  # Cross-entropy should be positive

    def test_num_parameters(self, trm_model):
        """Test parameter counting."""
        params = trm_model.num_parameters()
        assert params > 0
        assert isinstance(params, int)

        # All params should equal trainable params for this model
        all_params = trm_model.num_parameters(trainable_only=False)
        assert all_params >= params

    def test_model_deterministic(self, trm_model, trm_small_config):
        """Test model produces deterministic outputs."""
        torch.manual_seed(42)
        x = torch.randint(0, trm_small_config.vocab_size, (2, 16, 12))

        logits1, _ = trm_model(x)

        # Reset model to same state
        torch.manual_seed(42)
        logits2, _ = trm_model(x)

        assert torch.allclose(logits1, logits2, atol=1e-6)


class TestTRMRecursiveIterations:
    """Tests for TRM recursive iteration mechanism."""

    def test_recursive_iterations_count(self, trm_model, trm_small_config):
        """Test model runs iterations within expected range."""
        batch_size = 2
        x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))

        _, info = trm_model(x)

        assert info["iterations"] >= 1
        assert info["iterations"] <= trm_small_config.max_iterations

    def test_training_mode_full_iterations(self, trm_model_training, trm_small_config):
        """Test training mode runs all iterations."""
        batch_size = 2
        x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))

        _, info = trm_model_training(x)

        # In training mode, should complete all iterations
        assert info["iterations"] == trm_small_config.max_iterations

    def test_return_all_iterations(self, trm_model, trm_small_config):
        """Test returning intermediate states from all iterations."""
        batch_size = 2
        x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))

        # Access recursion module directly
        emb = trm_model.embedding(x)
        pos = trm_model.pos_encoding(x)
        hidden = emb + pos

        _, info = trm_model.recursion(hidden, return_all_iterations=True)

        assert "all_states" in info
        assert len(info["all_states"]) == info["iterations"]
        for state in info["all_states"]:
            assert state.shape == hidden.shape


class TestTRMEarlyStopping:
    """Tests for TRM early stopping based on confidence."""

    def test_early_stopping_in_eval_mode(self, trm_model, trm_small_config):
        """Test early stopping is active in eval mode."""
        batch_size = 2
        x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))

        trm_model.eval()
        _, info = trm_model(x)

        # Early stopping may or may not trigger depending on input
        # Just verify iterations is within bounds
        assert info["iterations"] >= trm_small_config.min_iterations
        assert info["iterations"] <= trm_small_config.max_iterations

    def test_confidence_in_valid_range(self, trm_model, trm_small_config):
        """Test confidence values are in [0, 1]."""
        batch_size = 2
        x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))

        _, info = trm_model(x)

        assert (info["confidence"] >= 0).all()
        assert (info["confidence"] <= 1).all()

    def test_min_iterations_respected(self, trm_small_config, cpu_device):
        """Test minimum iterations constraint is respected."""
        from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairTRM

        # Create model with high min_iterations
        config = trm_small_config
        config.min_iterations = 3
        model = CodeRepairTRM(config).to(cpu_device)
        model.eval()

        x = torch.randint(0, config.vocab_size, (2, 16, 12))
        _, info = model(x)

        assert info["iterations"] >= 3


class TestTRMGradientFlow:
    """Tests for gradient flow through TRM model."""

    def test_gradient_flows_through_recursion(self, trm_model_training, trm_small_config):
        """Test gradients flow through all parameters."""
        batch_size = 2
        x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))
        labels = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))

        _, info = trm_model_training(x, labels=labels)
        info["loss"].backward()

        # Check gradients exist for all trainable parameters
        for name, param in trm_model_training.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_gradient_checkpointing(self, trm_small_config, cpu_device):
        """Test gradient checkpointing for memory efficiency."""
        from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairTRM

        config = trm_small_config
        config.use_gradient_checkpointing = True
        model = CodeRepairTRM(config).to(cpu_device)
        model.train()

        x = torch.randint(0, config.vocab_size, (2, 16, 12))
        labels = torch.randint(0, config.vocab_size, (2, 16, 12))

        _, info = model(x, labels=labels)
        info["loss"].backward()

        # Check gradients exist with checkpointing
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name} with checkpointing"

    def test_gradients_not_nan_or_inf(self, trm_model_training, trm_small_config):
        """Test gradients are finite."""
        batch_size = 2
        x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))
        labels = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))

        _, info = trm_model_training(x, labels=labels)
        info["loss"].backward()

        for name, param in trm_model_training.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"


class TestTRMTraining:
    """Tests for TRM training components."""

    def test_loss_decreases_over_steps(self, trm_model_training, trm_small_config):
        """Test loss decreases over training steps."""
        batch_size = 4
        x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))
        labels = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))

        optimizer = torch.optim.Adam(trm_model_training.parameters(), lr=1e-3)

        losses = []
        for step in range(10):
            optimizer.zero_grad()
            _, info = trm_model_training(x, labels=labels)
            info["loss"].backward()
            optimizer.step()
            losses.append(info["loss"].item())

        # Loss should generally decrease (allow some variance)
        assert losses[-1] < losses[0], "Loss did not decrease over training"

    def test_optimizer_step_updates_parameters(self, trm_model_training, trm_small_config):
        """Test optimizer step actually updates parameters."""
        # Store initial parameter values
        initial_params = {
            name: param.clone().detach()
            for name, param in trm_model_training.named_parameters()
            if param.requires_grad
        }

        optimizer = torch.optim.Adam(trm_model_training.parameters(), lr=1e-3)

        x = torch.randint(0, trm_small_config.vocab_size, (2, 16, 12))
        labels = torch.randint(0, trm_small_config.vocab_size, (2, 16, 12))

        optimizer.zero_grad()
        _, info = trm_model_training(x, labels=labels)
        info["loss"].backward()
        optimizer.step()

        # Check at least some parameters changed
        params_changed = 0
        for name, param in trm_model_training.named_parameters():
            if name in initial_params:
                if not torch.allclose(param, initial_params[name], atol=1e-7):
                    params_changed += 1

        assert params_changed > 0, "No parameters were updated by optimizer"

    def test_checkpoint_save_load(self, trm_model, trm_small_config):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pt")

            # Save
            trm_model.save_pretrained(save_path)
            assert os.path.exists(save_path)

            # Load
            from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairTRM
            loaded_model = CodeRepairTRM.from_pretrained(save_path)

            # Check config matches
            assert loaded_model.config.grid_height == trm_small_config.grid_height
            assert loaded_model.config.grid_width == trm_small_config.grid_width
            assert loaded_model.config.vocab_size == trm_small_config.vocab_size

            # Check outputs match
            x = torch.randint(0, trm_small_config.vocab_size, (1, 16, 12))

            trm_model.eval()
            loaded_model.eval()

            with torch.no_grad():
                out1, _ = trm_model(x)
                out2, _ = loaded_model(x)

            assert torch.allclose(out1, out2, atol=1e-5)

    def test_learning_rate_affects_training(self, trm_small_config, cpu_device):
        """Test that learning rate affects training speed."""
        from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairTRM

        x = torch.randint(0, trm_small_config.vocab_size, (4, 16, 12))
        labels = torch.randint(0, trm_small_config.vocab_size, (4, 16, 12))

        def train_with_lr(lr: float, steps: int = 5) -> float:
            model = CodeRepairTRM(trm_small_config).to(cpu_device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            initial_loss = None
            final_loss = None

            for step in range(steps):
                optimizer.zero_grad()
                _, info = model(x.clone(), labels=labels.clone())
                loss = info["loss"]
                loss.backward()
                optimizer.step()

                if initial_loss is None:
                    initial_loss = loss.item()
                final_loss = loss.item()

            return initial_loss - final_loss  # Loss decrease

        # Higher LR should decrease loss more initially
        decrease_low_lr = train_with_lr(1e-5)
        decrease_high_lr = train_with_lr(1e-3)

        assert decrease_high_lr > decrease_low_lr


class TestTRMGenerate:
    """Tests for TRM generation/inference."""

    def test_generate_returns_expected_keys(self, trm_model, trm_small_config):
        """Test generate returns all expected output keys."""
        x = torch.randint(0, trm_small_config.vocab_size, (2, 16, 12))

        result = trm_model.generate(x)

        assert "output" in result
        assert "logits" in result
        assert "iterations" in result
        assert "confidence" in result

    def test_generate_output_shape(self, trm_model, trm_small_config):
        """Test generate output has correct shape."""
        batch_size = 2
        x = torch.randint(0, trm_small_config.vocab_size, (batch_size, 16, 12))

        result = trm_model.generate(x)

        assert result["output"].shape == (batch_size, 16, 12)
        assert result["logits"].shape == (batch_size, 16, 12, trm_small_config.vocab_size)

    def test_generate_deterministic_with_same_input(self, trm_model, trm_small_config):
        """Test generate produces same output for same input."""
        torch.manual_seed(42)
        x = torch.randint(0, trm_small_config.vocab_size, (2, 16, 12))

        result1 = trm_model.generate(x)
        result2 = trm_model.generate(x)

        assert torch.equal(result1["output"], result2["output"])

    def test_generate_with_max_iterations_override(self, trm_model, trm_small_config):
        """Test generate respects max_iterations override."""
        x = torch.randint(0, trm_small_config.vocab_size, (2, 16, 12))

        result = trm_model.generate(x, max_iterations=1)

        assert result["iterations"] == 1


class TestTRMLayers:
    """Tests for TRM layer components."""

    def test_grid_positional_encoding_shape(self):
        """Test GridPositionalEncoding output shapes."""
        from consciousness.ml_research.modern_dev.trm.src.layers import GridPositionalEncoding

        pos_enc = GridPositionalEncoding(
            embed_dim=64,
            max_height=16,
            max_width=12,
        )

        # Test with 3D input (token IDs)
        x = torch.randint(0, 100, (2, 16, 12))
        output = pos_enc(x)
        assert output.shape == (2, 16, 12, 64)

        # Test with 4D input (embeddings)
        x = torch.randn(2, 16, 12, 64)
        output = pos_enc(x)
        assert output.shape == (2, 16, 12, 64)

    def test_recursive_block_preserves_shape(self):
        """Test RecursiveBlock preserves input shape."""
        from consciousness.ml_research.modern_dev.trm.src.layers import RecursiveBlock

        block = RecursiveBlock(
            embed_dim=64,
            n_heads=4,
            ffn_dim=256,
            max_height=16,
            max_width=12,
        )

        x = torch.randn(2, 16, 12, 64)
        output = block(x)
        assert output.shape == x.shape

    def test_iteration_controller_min_iterations(self):
        """Test IterationController respects min_iterations."""
        from consciousness.ml_research.modern_dev.trm.src.layers import IterationController

        controller = IterationController(
            q_threshold=0.95,
            min_iterations=3,
            max_iterations=8,
        )

        hidden = torch.randn(2, 16, 12, 64)

        # Should not stop before min iterations
        for i in range(3):
            should_stop, _ = controller.should_stop(hidden, iteration=i)
            assert not should_stop, f"Stopped at iteration {i} before min_iterations"

    def test_iteration_controller_max_iterations(self):
        """Test IterationController stops at max_iterations."""
        from consciousness.ml_research.modern_dev.trm.src.layers import IterationController

        controller = IterationController(
            q_threshold=0.95,
            min_iterations=1,
            max_iterations=8,
        )

        hidden = torch.randn(2, 16, 12, 64)

        # Should always stop at max iterations
        should_stop, _ = controller.should_stop(hidden, iteration=7)
        assert should_stop

    def test_rms_norm_normalizes(self):
        """Test RMSNorm normalizes input."""
        from consciousness.ml_research.modern_dev.trm.src.layers import RMSNorm

        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64) * 10  # Large values

        output = norm(x)

        # Output should have reasonable magnitude
        assert output.abs().mean() < 5


class TestTRMLosses:
    """Tests for TRM loss functions."""

    def test_cross_entropy_loss_positive(self, trm_model_training, trm_small_config):
        """Test cross-entropy loss is positive."""
        x = torch.randint(0, trm_small_config.vocab_size, (2, 16, 12))
        labels = torch.randint(0, trm_small_config.vocab_size, (2, 16, 12))

        _, info = trm_model_training(x, labels=labels)

        assert info["loss"] > 0

    def test_loss_decreases_with_correct_labels(self, trm_model, trm_small_config):
        """Test loss is lower when output matches labels."""
        x = torch.randint(0, trm_small_config.vocab_size, (2, 16, 12))

        # Get model's predictions
        with torch.no_grad():
            logits, _ = trm_model(x)
            predicted = logits.argmax(dim=-1)

        # Loss with random labels
        random_labels = torch.randint(0, trm_small_config.vocab_size, (2, 16, 12))
        _, info_random = trm_model(x, labels=random_labels)

        # Loss with model's own predictions (should be lower)
        _, info_predicted = trm_model(x, labels=predicted)

        assert info_predicted["loss"] < info_random["loss"]

    def test_ignore_index_in_loss(self, trm_model_training, trm_small_config):
        """Test that ignore_index (pad token) is excluded from loss."""
        x = torch.randint(1, trm_small_config.vocab_size, (2, 16, 12))

        # Labels with some padding (0s)
        labels = torch.randint(1, trm_small_config.vocab_size, (2, 16, 12))
        labels[:, 8:, :] = 0  # Set second half to padding

        _, info = trm_model_training(x, labels=labels)

        # Loss should still be valid
        assert not torch.isnan(info["loss"])
        assert not torch.isinf(info["loss"])


class TestTRMConfigurations:
    """Tests for TRM configuration presets."""

    def test_tiny_config_preset(self):
        """Test tiny configuration preset."""
        from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairConfig

        config = CodeRepairConfig.for_code_repair_tiny()

        assert config.vocab_size == 1024
        assert config.embed_dim == 64
        assert config.n_blocks == 2

    def test_small_config_preset(self):
        """Test small configuration preset."""
        from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairConfig

        config = CodeRepairConfig.for_code_repair_small()

        assert config.embed_dim == 128
        assert config.n_blocks == 4

    def test_base_config_preset(self):
        """Test base configuration preset."""
        from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairConfig

        config = CodeRepairConfig.for_code_repair_base()

        assert config.embed_dim == 160
        assert config.n_blocks == 6

    def test_large_config_preset(self):
        """Test large configuration preset."""
        from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairConfig

        config = CodeRepairConfig.for_code_repair_large()

        assert config.embed_dim == 256
        assert config.n_blocks == 6

    def test_max_seq_len_computed(self):
        """Test max_seq_len is computed from grid dimensions."""
        from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairConfig

        config = CodeRepairConfig(grid_height=64, grid_width=48)

        assert config.max_seq_len == 64 * 48

    def test_effective_depth_computed(self):
        """Test effective depth is computed correctly."""
        from consciousness.ml_research.modern_dev.trm.src.model import CodeRepairConfig

        config = CodeRepairConfig(max_iterations=8, n_blocks=6)

        # effective_depth = max_iterations * n_blocks
        assert config.effective_depth == 8 * 6


class TestTRMFullSize:
    """Tests for TRM with full-size 64x48 grid (skip if slow)."""

    @pytest.mark.slow
    def test_full_size_forward(self, cpu_device):
        """Test forward pass with full 64x48 grid."""
        from consciousness.ml_research.modern_dev.trm.src.model import (
            CodeRepairConfig,
            CodeRepairTRM,
        )

        config = CodeRepairConfig(
            grid_height=64,
            grid_width=48,
            vocab_size=1000,  # Smaller vocab for test
            embed_dim=64,
            n_heads=4,
            ffn_dim=256,
            n_blocks=2,
            max_iterations=2,
        )

        model = CodeRepairTRM(config).to(cpu_device)
        model.eval()

        x = torch.randint(0, config.vocab_size, (1, 64, 48))
        logits, info = model(x)

        assert logits.shape == (1, 64, 48, config.vocab_size)

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_full_size_gpu_memory(self):
        """Test GPU memory usage with full-size model."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from consciousness.ml_research.modern_dev.trm.src.model import (
            CodeRepairConfig,
            CodeRepairTRM,
        )

        config = CodeRepairConfig.for_code_repair_base()
        model = CodeRepairTRM(config).cuda()

        torch.cuda.reset_peak_memory_stats()

        x = torch.randint(0, config.vocab_size, (1, 64, 48)).cuda()
        with torch.no_grad():
            logits, info = model(x)

        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak memory: {peak_memory_mb:.2f} MB")

        # Should fit in reasonable GPU memory
        assert peak_memory_mb < 8000  # 8GB limit


# =============================================================================
# Performance Benchmark Tests
# =============================================================================


@pytest.mark.benchmark
class TestTRMPerformance:
    """Performance benchmarks for TRM model."""

    def test_inference_speed(self, trm_model, trm_small_config, benchmark_timer):
        """Benchmark TRM inference speed."""
        x = torch.randint(0, trm_small_config.vocab_size, (1, 16, 12))

        # Warmup
        for _ in range(3):
            trm_model(x)

        # Benchmark
        with benchmark_timer("trm_inference", iterations=10) as timer:
            for _ in range(10):
                with torch.no_grad():
                    trm_model(x)

        print(f"TRM inference: {timer.result.avg_duration_ms:.2f}ms per forward pass")
        assert timer.result.avg_duration_ms < 1000  # 1 second max

    def test_training_step_speed(
        self, trm_model_training, trm_small_config, benchmark_timer
    ):
        """Benchmark TRM training step speed."""
        x = torch.randint(0, trm_small_config.vocab_size, (4, 16, 12))
        labels = torch.randint(0, trm_small_config.vocab_size, (4, 16, 12))
        optimizer = torch.optim.Adam(trm_model_training.parameters())

        # Warmup
        for _ in range(2):
            optimizer.zero_grad()
            _, info = trm_model_training(x, labels=labels)
            info["loss"].backward()
            optimizer.step()

        # Benchmark
        with benchmark_timer("trm_training", iterations=5) as timer:
            for _ in range(5):
                optimizer.zero_grad()
                _, info = trm_model_training(x, labels=labels)
                info["loss"].backward()
                optimizer.step()

        print(f"TRM training step: {timer.result.avg_duration_ms:.2f}ms")
        assert timer.result.avg_duration_ms < 5000  # 5 seconds max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
