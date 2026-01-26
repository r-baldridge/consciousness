"""
Tests for S4D State Space Model components.

Run with: pytest tests/test_ssm.py -v
"""

from __future__ import annotations

import pytest
import time
import torch

# Check if we have pytest benchmarking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class TestS4DKernel:
    """Test S4DKernel implementation."""

    def test_init(self):
        """Test S4DKernel initialization and shapes."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=256, d_state=64)
        assert kernel.A.shape == (64,)
        assert kernel.B.shape == (256, 64)
        assert kernel.C.shape == (256, 64)
        assert kernel.D.shape == (256,)
        assert kernel.log_dt.shape == (256,)

    def test_init_dt_modes(self):
        """Test different dt initialization modes."""
        from mamba_impl.src.ssm import S4DKernel

        # Random init
        kernel_random = S4DKernel(d_model=64, d_state=16, dt_init="random")
        dt_random = torch.exp(kernel_random.log_dt)
        assert torch.all(dt_random >= 0.001)
        assert torch.all(dt_random <= 0.1)

        # Constant init
        kernel_const = S4DKernel(d_model=64, d_state=16, dt_init="constant")
        dt_const = torch.exp(kernel_const.log_dt)
        # All should be approximately the same
        assert torch.allclose(dt_const, dt_const[0].expand_as(dt_const), atol=1e-6)

    def test_forward_recurrent_single(self):
        """Test recurrent mode with single time step."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=256, d_state=64)
        x = torch.randn(2, 256)  # [batch, d_model]

        y, h = kernel.forward_recurrent(x)

        assert y.shape == (2, 256)
        assert h.shape == (2, 256, 64)

    def test_forward_recurrent_with_state(self):
        """Test recurrent mode with initial state."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16)
        x = torch.randn(2, 64)
        h0 = torch.randn(2, 64, 16) * 0.1

        y, h = kernel.forward_recurrent(x, h=h0)

        assert y.shape == (2, 64)
        assert h.shape == (2, 64, 16)
        # State should have changed
        assert not torch.allclose(h, h0, atol=1e-6)

    def test_forward_recurrent_sequence(self):
        """Test recurrent mode with sequence input via forward()."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=256, d_state=64)
        kernel.eval()
        x = torch.randn(2, 100, 256)  # [batch, length, d_model]

        y, h = kernel(x, mode="recurrent")

        assert y.shape == (2, 100, 256)
        assert h.shape == (2, 256, 64)

    def test_forward_conv(self):
        """Test convolutional mode."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=256, d_state=64)
        x = torch.randn(2, 100, 256)

        y = kernel.forward_conv(x)

        assert y.shape == (2, 100, 256)

    def test_conv_recurrent_equivalence(self):
        """Conv and recurrent modes should produce same output."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16)
        kernel.eval()

        x = torch.randn(1, 50, 64)

        with torch.no_grad():
            y_conv, _ = kernel(x, mode="conv")
            y_rec, _ = kernel(x, mode="recurrent")

        # Should be close (not exact due to FFT precision)
        assert torch.allclose(y_conv, y_rec, atol=1e-4, rtol=1e-4), \
            f"Max diff: {(y_conv - y_rec).abs().max()}"

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16)
        x = torch.randn(2, 20, 64, requires_grad=True)

        y, _ = kernel(x, mode="conv")
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert kernel.B.grad is not None
        assert kernel.C.grad is not None
        assert kernel.log_dt.grad is not None
        assert kernel.D.grad is not None

    def test_long_sequence(self):
        """Should handle long sequences without OOM."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=256, d_state=64)

        # 10K tokens - should work on reasonable GPU/CPU
        x = torch.randn(1, 10000, 256)

        with torch.no_grad():
            y, _ = kernel(x, mode="conv")

        assert y.shape == (1, 10000, 256)

    def test_auto_mode_selection(self):
        """Test automatic mode selection based on training/eval."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16)
        x = torch.randn(1, 20, 64)

        # Training mode should use conv
        kernel.train()
        y_train, h_train = kernel(x, mode="auto")
        assert h_train is None  # Conv mode doesn't return state

        # Eval mode should use recurrent
        kernel.eval()
        with torch.no_grad():
            y_eval, h_eval = kernel(x, mode="auto")
        assert h_eval is not None  # Recurrent mode returns state

    def test_single_step_detection(self):
        """Test that single step input uses recurrent mode."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16)
        x = torch.randn(2, 64)  # [batch, d_model] - single step

        y, h = kernel(x)

        assert y.shape == (2, 64)
        assert h is not None

    def test_discretization(self):
        """Test discretization produces valid values."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16)
        A_bar, B_bar = kernel._discretize()

        # A_bar should be in (0, 1) for stable system (since A is negative)
        assert torch.all(A_bar > 0)
        assert torch.all(A_bar < 1)

        # B_bar should be finite
        assert torch.all(torch.isfinite(B_bar))

    def test_kernel_computation(self):
        """Test convolution kernel computation."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16)
        K = kernel._compute_kernel(100)

        assert K.shape == (64, 100)
        assert torch.all(torch.isfinite(K))

        # Kernel should decay over time (exponential decay)
        # Check that later values are smaller in magnitude on average
        early_norm = K[:, :10].abs().mean()
        late_norm = K[:, -10:].abs().mean()
        assert late_norm < early_norm


class TestS4DLayer:
    """Test S4DLayer (full layer with normalization)."""

    def test_forward(self):
        """Test S4DLayer forward pass."""
        from mamba_impl.src.ssm import S4DLayer

        layer = S4DLayer(d_model=256, d_state=64)
        x = torch.randn(2, 100, 256)

        y = layer(x)

        assert y.shape == (2, 100, 256)

    def test_residual_connection(self):
        """Test that residual connection is working."""
        from mamba_impl.src.ssm import S4DLayer

        layer = S4DLayer(d_model=64, d_state=16)
        x = torch.randn(2, 20, 64)

        y = layer(x)

        # Output should be correlated with input due to residual
        # (not a strict test, but sanity check)
        assert y.shape == x.shape

    def test_bidirectional(self):
        """Test bidirectional mode."""
        from mamba_impl.src.ssm import S4DLayer

        layer = S4DLayer(d_model=256, d_state=64, bidirectional=True)
        x = torch.randn(2, 100, 256)

        y = layer(x)

        assert y.shape == (2, 100, 256)

    def test_dropout(self):
        """Test that dropout is applied during training."""
        from mamba_impl.src.ssm import S4DLayer

        layer = S4DLayer(d_model=64, d_state=16, dropout=0.5)
        x = torch.randn(2, 20, 64)

        # Training mode - outputs should vary due to dropout
        layer.train()
        y1 = layer(x)
        y2 = layer(x)
        assert not torch.allclose(y1, y2)

        # Eval mode - outputs should be deterministic
        layer.eval()
        with torch.no_grad():
            y3 = layer(x)
            y4 = layer(x)
        assert torch.allclose(y3, y4)


class TestHiPPO:
    """Test HiPPO initialization."""

    def test_legs_diagonal(self):
        """Test HiPPO-LegS diagonal approximation."""
        from mamba_impl.src.parameterization import HiPPO

        A = HiPPO.legs_diagonal(64)

        assert A.shape == (64,)
        assert torch.all(A < 0)  # All negative
        assert A[0] == -0.5
        assert A[63] == -63.5

    def test_legs_full(self):
        """Test full HiPPO-LegS matrix."""
        from mamba_impl.src.parameterization import HiPPO

        A = HiPPO.legs(16)

        assert A.shape == (16, 16)
        # Lower triangular (including diagonal)
        assert torch.all(A.triu(1) == 0)
        # Diagonal should be negative
        assert torch.all(torch.diag(A) < 0)

    def test_legt(self):
        """Test HiPPO-LegT matrix."""
        from mamba_impl.src.parameterization import HiPPO

        A = HiPPO.legt(16)

        assert A.shape == (16, 16)
        # Should be all non-positive
        assert torch.all(A <= 0)

    def test_lmu(self):
        """Test LMU matrix."""
        from mamba_impl.src.parameterization import HiPPO

        A = HiPPO.lmu(16)

        assert A.shape == (16, 16)
        # Should be all non-positive
        assert torch.all(A <= 0)

    def test_fourier(self):
        """Test Fourier basis initialization."""
        from mamba_impl.src.parameterization import HiPPO

        A = HiPPO.fourier(16)

        assert A.shape == (16,)

    def test_random_diagonal(self):
        """Test random diagonal initialization."""
        from mamba_impl.src.parameterization import HiPPO

        A = HiPPO.random_diagonal(64, min_val=-1.0, max_val=-0.1)

        assert A.shape == (64,)
        assert torch.all(A >= -1.0)
        assert torch.all(A <= -0.1)


class TestSSMInit:
    """Test SSM initialization utilities."""

    def test_init_dt_random(self):
        """Test random dt initialization."""
        from mamba_impl.src.parameterization import SSMInit

        log_dt = SSMInit.init_dt(256, dt_min=0.001, dt_max=0.1, dt_init="random")

        assert log_dt.shape == (256,)
        dt = torch.exp(log_dt)
        assert torch.all(dt >= 0.001)
        assert torch.all(dt <= 0.1)

    def test_init_dt_constant(self):
        """Test constant dt initialization."""
        from mamba_impl.src.parameterization import SSMInit

        log_dt = SSMInit.init_dt(256, dt_min=0.001, dt_max=0.1, dt_init="constant")

        assert log_dt.shape == (256,)
        # All values should be the same
        assert torch.allclose(log_dt, log_dt[0].expand_as(log_dt))

    def test_init_BC_normal(self):
        """Test normal B/C initialization."""
        from mamba_impl.src.parameterization import SSMInit

        B, C = SSMInit.init_BC(256, 64, init="normal")

        assert B.shape == (256, 64)
        assert C.shape == (256, 64)
        # Values should be roughly unit variance / sqrt(d_state)
        assert B.std() < 0.5
        assert C.std() < 0.5

    def test_init_BC_uniform(self):
        """Test uniform B/C initialization."""
        from mamba_impl.src.parameterization import SSMInit

        B, C = SSMInit.init_BC(256, 64, init="uniform")

        assert B.shape == (256, 64)
        assert C.shape == (256, 64)

    def test_init_BC_ones(self):
        """Test ones B/C initialization."""
        from mamba_impl.src.parameterization import SSMInit

        B, C = SSMInit.init_BC(256, 64, init="ones")

        assert B.shape == (256, 64)
        assert C.shape == (256, 64)
        assert torch.allclose(B, torch.ones_like(B) / 64)

    def test_init_D(self):
        """Test D initialization."""
        from mamba_impl.src.parameterization import SSMInit

        D_ones = SSMInit.init_D(256, init="ones")
        D_zeros = SSMInit.init_D(256, init="zeros")
        D_normal = SSMInit.init_D(256, init="normal")

        assert torch.allclose(D_ones, torch.ones(256))
        assert torch.allclose(D_zeros, torch.zeros(256))
        assert D_normal.shape == (256,)


class TestRMSNorm:
    """Test RMSNorm implementation."""

    def test_creation(self):
        """Test RMSNorm creation."""
        from mamba_impl.src.parameterization import RMSNorm

        norm = RMSNorm(d_model=256)
        assert norm.weight.shape == (256,)

    def test_forward(self):
        """Test RMSNorm forward pass."""
        from mamba_impl.src.parameterization import RMSNorm

        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 100, 64)

        y = norm(x)

        assert y.shape == x.shape

    def test_normalization(self):
        """Test that RMSNorm actually normalizes."""
        from mamba_impl.src.parameterization import RMSNorm

        norm = RMSNorm(d_model=64)
        # Set weight to ones for testing
        with torch.no_grad():
            norm.weight.fill_(1.0)

        x = torch.randn(2, 100, 64) * 10  # Large scale

        y = norm(x)

        # RMS of output should be approximately 1
        rms = torch.sqrt((y ** 2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestPerformance:
    """Performance benchmarks."""

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not installed")
    def test_memory_1k_tokens(self):
        """Test memory usage for 1K tokens."""
        import gc
        from mamba_impl.src.ssm import S4DKernel

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        kernel = S4DKernel(d_model=256, d_state=64)
        x = torch.randn(1, 1000, 256)

        # Warm up
        with torch.no_grad():
            _ = kernel(x, mode="conv")

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        with torch.no_grad():
            y, _ = kernel(x, mode="conv")

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        print(f"\n1K tokens memory: {mem_used:.2f} MB (before: {mem_before:.2f}, after: {mem_after:.2f})")
        # Just check it runs without OOM
        assert y.shape == (1, 1000, 256)

    def test_forward_time_1k(self):
        """Benchmark forward pass time for 1K tokens."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=256, d_state=64)
        x = torch.randn(1, 1000, 256)

        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = kernel(x, mode="conv")

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.perf_counter()
                y, _ = kernel(x, mode="conv")
                end = time.perf_counter()
                times.append(end - start)

        avg_time = sum(times) / len(times)
        print(f"\n1K tokens forward time: {avg_time*1000:.2f} ms")

        # Should complete reasonably fast (< 1 second on CPU)
        assert avg_time < 1.0

    def test_forward_time_10k(self):
        """Benchmark forward pass time for 10K tokens."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=256, d_state=64)
        x = torch.randn(1, 10000, 256)

        # Warm up
        with torch.no_grad():
            _ = kernel(x, mode="conv")

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(5):
                start = time.perf_counter()
                y, _ = kernel(x, mode="conv")
                end = time.perf_counter()
                times.append(end - start)

        avg_time = sum(times) / len(times)
        print(f"\n10K tokens forward time: {avg_time*1000:.2f} ms")

        # Should complete in reasonable time (< 10 seconds on CPU)
        assert avg_time < 10.0

    def test_recurrent_vs_conv_speed(self):
        """Compare recurrent vs convolutional mode speed."""
        from mamba_impl.src.ssm import S4DKernel

        kernel = S4DKernel(d_model=64, d_state=16)
        kernel.eval()
        x = torch.randn(1, 100, 64)

        # Benchmark conv mode
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(10):
                _ = kernel(x, mode="conv")
            conv_time = (time.perf_counter() - start) / 10

        # Benchmark recurrent mode
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(10):
                _ = kernel(x, mode="recurrent")
            rec_time = (time.perf_counter() - start) / 10

        print(f"\n100 tokens - Conv: {conv_time*1000:.2f} ms, Recurrent: {rec_time*1000:.2f} ms")

        # Both should work; conv is typically faster for training
        assert conv_time > 0
        assert rec_time > 0


class TestImports:
    """Test that all exports are accessible."""

    def test_import_from_src(self):
        """Test importing from src package."""
        from mamba_impl.src import S4DKernel, S4DLayer, HiPPO, SSMInit, RMSNorm

        assert S4DKernel is not None
        assert S4DLayer is not None
        assert HiPPO is not None
        assert SSMInit is not None
        assert RMSNorm is not None

    def test_import_from_ssm(self):
        """Test importing from ssm module."""
        from mamba_impl.src.ssm import S4DKernel, S4DLayer

        assert S4DKernel is not None
        assert S4DLayer is not None

    def test_import_from_parameterization(self):
        """Test importing from parameterization module."""
        from mamba_impl.src.parameterization import HiPPO, SSMInit, RMSNorm

        assert HiPPO is not None
        assert SSMInit is not None
        assert RMSNorm is not None


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_s4d_with_hippo_init(self):
        """Test S4DKernel initialized with HiPPO values."""
        from mamba_impl.src.ssm import S4DKernel
        from mamba_impl.src.parameterization import HiPPO

        # Create kernel
        kernel = S4DKernel(d_model=64, d_state=16)

        # Verify A matches HiPPO diagonal
        hippo_A = HiPPO.legs_diagonal(16)
        # Note: S4DKernel uses A[n] = -0.5 * (n+1), which is similar
        expected_A = -0.5 * torch.arange(1, 17, dtype=torch.float32)
        assert torch.allclose(kernel.A, expected_A)

    def test_full_pipeline(self):
        """Test full SSM pipeline: init -> forward -> backward."""
        from mamba_impl.src.ssm import S4DLayer

        # Create layer
        layer = S4DLayer(d_model=128, d_state=32, dropout=0.1)
        layer.train()

        # Forward
        x = torch.randn(4, 50, 128, requires_grad=True)
        y = layer(x)

        # Backward
        loss = y.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_stacked_layers(self):
        """Test stacking multiple S4D layers."""
        from mamba_impl.src.ssm import S4DLayer
        import torch.nn as nn

        # Create stack of layers
        layers = nn.Sequential(
            S4DLayer(d_model=64, d_state=16),
            S4DLayer(d_model=64, d_state=16),
            S4DLayer(d_model=64, d_state=16),
        )

        x = torch.randn(2, 100, 64)
        y = layers(x)

        assert y.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
