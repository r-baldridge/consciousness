"""
Comprehensive tests for Mamba (Selective State Space Model) components.

Tests cover:
- SSM core (S4D kernel)
- Selective mechanism (input-dependent parameters)
- Mamba block architecture
- Parallel scan implementation
- Full Mamba language model
- Generation and caching

Run with: pytest tests/test_mamba_full.py -v --cov=mamba_impl
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SSM Core Tests
# =============================================================================


class TestS4DKernel:
    """Tests for S4D (diagonal state space) kernel."""

    def test_forward_shape(self, cpu_device):
        """Test S4D kernel output shapes."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16).to(cpu_device)

        # Sequence input
        x = torch.randn(2, 32, 64, device=cpu_device)
        y, h = kernel(x)

        assert y.shape == x.shape
        assert h is None  # Conv mode returns no state

    def test_recurrent_mode(self, cpu_device):
        """Test S4D kernel recurrent mode."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16).to(cpu_device)
        kernel.eval()

        # Single step input
        x = torch.randn(2, 64, device=cpu_device)
        y, h = kernel.forward_recurrent(x)

        assert y.shape == (2, 64)
        assert h.shape == (2, 64, 16)

    def test_conv_mode(self, cpu_device):
        """Test S4D kernel convolutional mode."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16).to(cpu_device)

        x = torch.randn(2, 32, 64, device=cpu_device)
        y = kernel.forward_conv(x)

        assert y.shape == x.shape

    def test_recurrent_conv_equivalence(self, cpu_device):
        """Test recurrent and convolutional modes produce similar outputs."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=32, d_state=8).to(cpu_device)
        kernel.eval()

        x = torch.randn(1, 16, 32, device=cpu_device)

        # Convolutional mode
        y_conv = kernel.forward_conv(x)

        # Recurrent mode
        y_rec_list = []
        h = None
        for t in range(x.shape[1]):
            y_t, h = kernel.forward_recurrent(x[:, t], h)
            y_rec_list.append(y_t)
        y_rec = torch.stack(y_rec_list, dim=1)

        # Should be close (not exact due to FFT padding)
        assert torch.allclose(y_conv, y_rec, atol=1e-4)

    def test_hippo_initialization(self, cpu_device):
        """Test HiPPO initialization of A matrix."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16).to(cpu_device)

        # A should be negative (for stability)
        assert (kernel.A < 0).all()

        # A should follow HiPPO pattern: -0.5 * (n + 1) for n in 0..N-1
        expected_A = -0.5 * torch.arange(1, 17, dtype=torch.float32)
        assert torch.allclose(kernel.A, expected_A, atol=1e-5)

    def test_discretization(self, cpu_device):
        """Test discretization produces valid parameters."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16).to(cpu_device)

        A_bar, B_bar = kernel._discretize()

        # A_bar should be in (0, 1) for stability
        assert (A_bar > 0).all()
        assert (A_bar < 1).all()

        # No NaN or Inf
        assert not torch.isnan(A_bar).any()
        assert not torch.isnan(B_bar).any()


class TestS4DLayer:
    """Tests for S4D layer with normalization and residual."""

    def test_forward_preserves_shape(self, cpu_device):
        """Test S4D layer preserves input shape."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.ssm import S4DLayer

        layer = S4DLayer(d_model=64, d_state=16).to(cpu_device)

        x = torch.randn(2, 32, 64, device=cpu_device)
        y = layer(x)

        assert y.shape == x.shape

    def test_bidirectional_mode(self, cpu_device):
        """Test bidirectional S4D layer."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.ssm import S4DLayer

        layer = S4DLayer(d_model=64, d_state=16, bidirectional=True).to(cpu_device)

        x = torch.randn(2, 32, 64, device=cpu_device)
        y = layer(x)

        assert y.shape == x.shape

    def test_residual_connection(self, cpu_device):
        """Test residual connection is applied."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.ssm import S4DLayer

        layer = S4DLayer(d_model=64, d_state=16, dropout=0.0).to(cpu_device)

        x = torch.randn(2, 32, 64, device=cpu_device)
        y = layer(x)

        # Output should be different from input but correlated
        assert not torch.allclose(y, x, atol=1e-3)
        # But not wildly different (residual helps)
        assert (y - x).abs().mean() < x.abs().mean()


# =============================================================================
# Selective Mechanism Tests
# =============================================================================


class TestSelectiveScan:
    """Tests for Mamba selective mechanism."""

    def test_input_dependent_delta(self, cpu_device):
        """Test delta (dt) varies with input."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.selective import SelectiveSSM

        ssm = SelectiveSSM(d_model=64, d_state=16).to(cpu_device)

        x1 = torch.randn(2, 32, 64, device=cpu_device)
        x2 = torch.randn(2, 32, 64, device=cpu_device) * 2  # Different input

        # Get intermediate values
        with torch.no_grad():
            y1, _ = ssm(x1)
            y2, _ = ssm(x2)

        # Outputs should be different (input-dependent)
        assert not torch.allclose(y1, y2, atol=1e-3)

    def test_input_dependent_B_C(self, cpu_device):
        """Test B and C projections vary with input."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.selective import SelectiveSSM

        ssm = SelectiveSSM(d_model=64, d_state=16).to(cpu_device)

        # Two different inputs
        x1 = torch.ones(1, 16, 64, device=cpu_device)
        x2 = torch.ones(1, 16, 64, device=cpu_device) * 2

        y1, _ = ssm(x1)
        y2, _ = ssm(x2)

        # Different inputs should produce different outputs
        assert not torch.allclose(y1, y2, atol=1e-5)

    def test_selective_vs_static_different(self, cpu_device):
        """Test selective SSM differs from static SSM."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.selective import SelectiveSSM
        from consciousness.ml_research.modern_dev.mamba_impl.src.ssm import S4DKernel

        selective = SelectiveSSM(d_model=64, d_state=16).to(cpu_device)
        static = S4DKernel(d_model=64, d_state=16).to(cpu_device)

        x = torch.randn(2, 32, 64, device=cpu_device)

        y_selective, _ = selective(x)
        y_static, _ = static(x)

        # Should produce different results
        assert not torch.allclose(y_selective, y_static, atol=1e-2)

    def test_gradient_flow(self, cpu_device):
        """Test gradients flow through selective mechanism."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.selective import SelectiveSSM

        ssm = SelectiveSSM(d_model=64, d_state=16).to(cpu_device)

        x = torch.randn(2, 32, 64, device=cpu_device, requires_grad=True)
        y, _ = ssm(x)
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        for name, param in ssm.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# =============================================================================
# Mamba Block Tests
# =============================================================================


class TestMambaBlock:
    """Tests for full Mamba block architecture."""

    def test_block_forward_shape(self, cpu_device):
        """Test Mamba block output shape."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.block import MambaBlock

        block = MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2).to(cpu_device)

        x = torch.randn(2, 32, 64, device=cpu_device)
        y = block(x)

        assert y.shape == x.shape

    def test_block_with_different_expand(self, cpu_device):
        """Test Mamba block with different expansion factors."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.block import MambaBlock

        for expand in [1, 2, 4]:
            block = MambaBlock(d_model=64, d_state=16, expand=expand).to(cpu_device)
            x = torch.randn(2, 32, 64, device=cpu_device)
            y = block(x)
            assert y.shape == x.shape

    def test_conv1d_causal(self, cpu_device):
        """Test Conv1D is causal (no future information leakage)."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.block import MambaBlock

        block = MambaBlock(d_model=64, d_state=16, d_conv=4).to(cpu_device)
        block.eval()

        # Create input where second half is zeros
        x = torch.randn(1, 32, 64, device=cpu_device)
        x_masked = x.clone()
        x_masked[:, 16:, :] = 0

        # Output at position t should only depend on x[:t+1]
        y_full = block(x)
        y_masked = block(x_masked)

        # First 16 positions should be identical (causal)
        # Note: Due to conv receptive field, we check a bit before
        assert torch.allclose(y_full[:, :12, :], y_masked[:, :12, :], atol=1e-5)

    def test_gating_mechanism(self, cpu_device):
        """Test gating mechanism modulates output."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.block import MambaBlock

        block = MambaBlock(d_model=64, d_state=16).to(cpu_device)

        x = torch.randn(2, 32, 64, device=cpu_device)
        y = block(x)

        # Output should have reasonable magnitude (gating prevents explosion)
        assert y.abs().mean() < 10 * x.abs().mean()

    def test_inference_caching(self, cpu_device):
        """Test inference with state caching matches full forward."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.block import MambaBlock

        block = MambaBlock(d_model=64, d_state=16, d_conv=4).to(cpu_device)
        block.eval()

        # Full forward
        x = torch.randn(1, 16, 64, device=cpu_device)
        with torch.no_grad():
            y_full = block(x)

        # Step-by-step with caching
        cache = None
        y_steps = []
        with torch.no_grad():
            for t in range(x.shape[1]):
                y_t, cache = block.step(x[:, t:t+1, :], cache)
                y_steps.append(y_t)

        y_cached = torch.cat(y_steps, dim=1)

        # Should match (within numerical precision)
        assert torch.allclose(y_full, y_cached, atol=1e-4)


class TestMambaLayer:
    """Tests for Mamba layer with normalization."""

    def test_layer_forward(self, cpu_device):
        """Test Mamba layer forward pass."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.block import MambaLayer

        layer = MambaLayer(d_model=64).to(cpu_device)

        x = torch.randn(2, 32, 64, device=cpu_device)
        y = layer(x)

        assert y.shape == x.shape

    def test_residual_scaling(self, cpu_device):
        """Test residual scaling is applied."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.block import MambaLayer

        layer = MambaLayer(d_model=64, residual_scale=0.5).to(cpu_device)

        x = torch.randn(2, 32, 64, device=cpu_device)
        y = layer(x)

        # Output should be close to input (residual dominant)
        diff = (y - x).abs().mean()
        assert diff < x.abs().mean()


class TestMambaStack:
    """Tests for stacked Mamba layers."""

    def test_stack_forward(self, cpu_device):
        """Test Mamba stack forward pass."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.block import MambaStack

        stack = MambaStack(d_model=64, n_layers=3).to(cpu_device)

        x = torch.randn(2, 32, 64, device=cpu_device)
        y = stack(x)

        assert y.shape == x.shape

    def test_stack_gradient_flow(self, cpu_device):
        """Test gradients flow through entire stack."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.block import MambaStack

        stack = MambaStack(d_model=64, n_layers=4).to(cpu_device)

        x = torch.randn(2, 32, 64, device=cpu_device, requires_grad=True)
        y = stack(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# =============================================================================
# Parallel Scan Tests
# =============================================================================


class TestParallelScan:
    """Tests for parallel/associative scan implementation."""

    def test_scan_matches_sequential(self, cpu_device):
        """Test parallel scan matches sequential scan."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.scan import (
            parallel_scan,
            sequential_scan,
        )

        batch, length, d_state = 2, 32, 16

        A = torch.rand(batch, length, d_state, device=cpu_device) * 0.9 + 0.05
        B = torch.randn(batch, length, d_state, device=cpu_device)

        y_parallel = parallel_scan(A, B)
        y_sequential = sequential_scan(A, B)

        assert torch.allclose(y_parallel, y_sequential, atol=1e-4)

    def test_scan_gradient_flow(self, cpu_device):
        """Test gradients flow through parallel scan."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.scan import parallel_scan

        batch, length, d_state = 2, 16, 8

        A = torch.rand(batch, length, d_state, device=cpu_device, requires_grad=True) * 0.9
        B = torch.randn(batch, length, d_state, device=cpu_device, requires_grad=True)

        y = parallel_scan(A, B)
        loss = y.sum()
        loss.backward()

        assert A.grad is not None
        assert B.grad is not None
        assert not torch.isnan(A.grad).any()
        assert not torch.isnan(B.grad).any()

    def test_chunked_scan(self, cpu_device):
        """Test chunked scan for long sequences."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.scan import (
            ChunkedScan,
            sequential_scan,
        )

        batch, length, d_state = 2, 128, 8

        A = torch.rand(batch, length, d_state, device=cpu_device) * 0.9
        B = torch.randn(batch, length, d_state, device=cpu_device)

        chunked = ChunkedScan(chunk_size=32)
        y_chunked = chunked(A, B)
        y_sequential = sequential_scan(A, B)

        assert torch.allclose(y_chunked, y_sequential, atol=1e-4)

    def test_scan_edge_cases(self, cpu_device):
        """Test scan handles edge cases."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.scan import parallel_scan

        # Length 1
        A = torch.rand(2, 1, 8, device=cpu_device)
        B = torch.randn(2, 1, 8, device=cpu_device)
        y = parallel_scan(A, B)
        assert y.shape == (2, 1, 8)
        assert torch.allclose(y, B)  # With length 1, output = B

        # Length 2
        A = torch.rand(2, 2, 8, device=cpu_device)
        B = torch.randn(2, 2, 8, device=cpu_device)
        y = parallel_scan(A, B)
        assert y.shape == (2, 2, 8)

    def test_scan_numerical_stability(self, cpu_device):
        """Test scan is numerically stable."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.scan import parallel_scan

        batch, length, d_state = 2, 64, 16

        # A close to 1 (challenging for stability)
        A = torch.ones(batch, length, d_state, device=cpu_device) * 0.99
        B = torch.randn(batch, length, d_state, device=cpu_device)

        y = parallel_scan(A, B)

        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()


# =============================================================================
# Full Mamba Model Tests
# =============================================================================


class TestMambaLM:
    """Tests for full Mamba language model."""

    def test_forward_shape(self, cpu_device):
        """Test Mamba LM output shapes."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            d_state=8,
        )
        model = MambaLM(config).to(cpu_device)

        input_ids = torch.randint(0, 1000, (2, 32), device=cpu_device)
        logits = model(input_ids)

        assert logits.shape == (2, 32, 1000)

    def test_forward_with_labels(self, cpu_device):
        """Test forward pass with labels computes loss."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
        )
        model = MambaLM(config).to(cpu_device)

        input_ids = torch.randint(0, 1000, (2, 32), device=cpu_device)
        labels = torch.randint(0, 1000, (2, 32), device=cpu_device)

        logits, loss = model(input_ids, labels=labels)

        assert logits.shape == (2, 32, 1000)
        assert loss.dim() == 0  # Scalar
        assert loss > 0

    def test_generate_basic(self, cpu_device):
        """Test basic generation."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
        )
        model = MambaLM(config).to(cpu_device)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 8), device=cpu_device)

        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=16)

        assert output.shape[0] == 1
        assert output.shape[1] == 8 + 16  # Original + new tokens

    def test_generate_deterministic(self, cpu_device):
        """Test generation is deterministic with temperature=0."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
        )
        model = MambaLM(config).to(cpu_device)
        model.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, 1000, (1, 8), device=cpu_device)

        with torch.no_grad():
            output1 = model.generate(input_ids, max_new_tokens=8, temperature=0.0)
            output2 = model.generate(input_ids, max_new_tokens=8, temperature=0.0)

        assert torch.equal(output1, output2)

    def test_generate_with_sampling(self, cpu_device):
        """Test generation with sampling produces diverse outputs."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
        )
        model = MambaLM(config).to(cpu_device)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 8), device=cpu_device)

        outputs = []
        with torch.no_grad():
            for _ in range(3):
                output = model.generate(
                    input_ids,
                    max_new_tokens=16,
                    temperature=1.0,
                    do_sample=True,
                )
                outputs.append(output)

        # At least some outputs should be different
        different_count = sum(
            1 for i in range(len(outputs))
            for j in range(i + 1, len(outputs))
            if not torch.equal(outputs[i], outputs[j])
        )
        assert different_count > 0

    def test_tied_embeddings(self, cpu_device):
        """Test embedding weight tying."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            tie_embeddings=True,
        )
        model = MambaLM(config).to(cpu_device)

        # Check weights are tied
        assert model.embedding.weight is model.lm_head.weight

    def test_gradient_flow(self, cpu_device):
        """Test gradients flow through model."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
        )
        model = MambaLM(config).to(cpu_device)

        input_ids = torch.randint(0, 1000, (2, 32), device=cpu_device)
        labels = torch.randint(0, 1000, (2, 32), device=cpu_device)

        _, loss = model(input_ids, labels=labels)
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_loss_decreases(self, cpu_device):
        """Test loss decreases with training."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
        )
        model = MambaLM(config).to(cpu_device)

        input_ids = torch.randint(0, 1000, (4, 32), device=cpu_device)
        labels = torch.randint(0, 1000, (4, 32), device=cpu_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            _, loss = model(input_ids, labels=labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0]


class TestMambaInferenceCache:
    """Tests for Mamba inference caching."""

    def test_cache_creation(self, cpu_device):
        """Test cache is created correctly."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaCache,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            d_state=8,
            d_conv=4,
        )

        cache = MambaCache.empty(config, batch_size=2, device=cpu_device)

        assert len(cache.conv_states) == 2  # n_layers
        assert len(cache.ssm_states) == 2

    def test_cached_generation_matches_full(self, cpu_device):
        """Test cached generation matches non-cached."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
        )
        model = MambaLM(config).to(cpu_device)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 16), device=cpu_device)

        with torch.no_grad():
            # Full forward
            logits_full = model(input_ids)

            # Step-by-step with cache
            cache = None
            logits_list = []
            for t in range(input_ids.shape[1]):
                logits_t, cache = model.forward_with_cache(
                    input_ids[:, t:t+1], cache
                )
                logits_list.append(logits_t)

            logits_cached = torch.cat(logits_list, dim=1)

        assert torch.allclose(logits_full, logits_cached, atol=1e-4)


# =============================================================================
# Performance Benchmarks
# =============================================================================


@pytest.mark.benchmark
class TestMambaPerformance:
    """Performance benchmarks for Mamba model."""

    def test_inference_speed(self, cpu_device, benchmark_timer):
        """Benchmark Mamba inference speed."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
        )
        model = MambaLM(config).to(cpu_device)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 64), device=cpu_device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                model(input_ids)

        # Benchmark
        with benchmark_timer("mamba_inference", iterations=10) as timer:
            for _ in range(10):
                with torch.no_grad():
                    model(input_ids)

        print(f"Mamba inference: {timer.result.avg_duration_ms:.2f}ms")
        assert timer.result.avg_duration_ms < 1000

    def test_generation_speed(self, cpu_device, benchmark_timer):
        """Benchmark Mamba generation speed."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.mamba_model import (
            MambaLM,
            MambaLMConfig,
        )

        config = MambaLMConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
        )
        model = MambaLM(config).to(cpu_device)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 8), device=cpu_device)

        # Warmup
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=16)

        # Benchmark
        with benchmark_timer("mamba_generation", iterations=5) as timer:
            for _ in range(5):
                with torch.no_grad():
                    model.generate(input_ids, max_new_tokens=32)

        print(f"Mamba generation (32 tokens): {timer.result.avg_duration_ms:.2f}ms")

    def test_scan_benchmark(self, cpu_device, benchmark_timer):
        """Benchmark parallel vs sequential scan."""
        from consciousness.ml_research.modern_dev.mamba_impl.src.scan import (
            parallel_scan,
            sequential_scan,
        )

        batch, length, d_state = 4, 256, 16

        A = torch.rand(batch, length, d_state, device=cpu_device) * 0.9
        B = torch.randn(batch, length, d_state, device=cpu_device)

        # Warmup
        parallel_scan(A, B)
        sequential_scan(A, B)

        # Benchmark parallel
        with benchmark_timer("parallel_scan", iterations=10) as t_parallel:
            for _ in range(10):
                parallel_scan(A, B)

        # Benchmark sequential
        with benchmark_timer("sequential_scan", iterations=10) as t_sequential:
            for _ in range(10):
                sequential_scan(A, B)

        print(f"Parallel scan: {t_parallel.result.avg_duration_ms:.2f}ms")
        print(f"Sequential scan: {t_sequential.result.avg_duration_ms:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
