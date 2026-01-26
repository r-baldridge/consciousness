#!/usr/bin/env python3
"""
Standalone test runner for S4D SSM tests.

This script runs tests independently of the package hierarchy to avoid
import issues with parent packages.
"""

import sys
import os
import time

# Add the src directory to path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

import torch
import traceback

# Import modules directly
from src.ssm import S4DKernel, S4DLayer
from src.parameterization import HiPPO, SSMInit, RMSNorm


def test_s4d_kernel_init():
    """Test S4DKernel initialization and shapes."""
    print("Testing S4DKernel initialization...")
    kernel = S4DKernel(d_model=256, d_state=64)
    assert kernel.A.shape == (64,), f"A shape mismatch: {kernel.A.shape}"
    assert kernel.B.shape == (256, 64), f"B shape mismatch: {kernel.B.shape}"
    assert kernel.C.shape == (256, 64), f"C shape mismatch: {kernel.C.shape}"
    assert kernel.D.shape == (256,), f"D shape mismatch: {kernel.D.shape}"
    assert kernel.log_dt.shape == (256,), f"log_dt shape mismatch: {kernel.log_dt.shape}"
    print("  PASSED")


def test_forward_recurrent_single():
    """Test recurrent mode with single time step."""
    print("Testing forward_recurrent single step...")
    kernel = S4DKernel(d_model=256, d_state=64)
    x = torch.randn(2, 256)

    y, h = kernel.forward_recurrent(x)

    assert y.shape == (2, 256), f"Output shape mismatch: {y.shape}"
    assert h.shape == (2, 256, 64), f"State shape mismatch: {h.shape}"
    print("  PASSED")


def test_forward_recurrent_sequence():
    """Test recurrent mode with sequence."""
    print("Testing forward_recurrent sequence...")
    kernel = S4DKernel(d_model=256, d_state=64)
    kernel.eval()
    x = torch.randn(2, 100, 256)

    y, h = kernel(x, mode="recurrent")

    assert y.shape == (2, 100, 256), f"Output shape mismatch: {y.shape}"
    assert h.shape == (2, 256, 64), f"State shape mismatch: {h.shape}"
    print("  PASSED")


def test_forward_conv():
    """Test convolutional mode."""
    print("Testing forward_conv...")
    kernel = S4DKernel(d_model=256, d_state=64)
    x = torch.randn(2, 100, 256)

    y = kernel.forward_conv(x)

    assert y.shape == (2, 100, 256), f"Output shape mismatch: {y.shape}"
    print("  PASSED")


def test_conv_recurrent_equivalence():
    """Conv and recurrent modes should produce same output."""
    print("Testing conv-recurrent equivalence...")
    kernel = S4DKernel(d_model=64, d_state=16)
    kernel.eval()

    x = torch.randn(1, 50, 64)

    with torch.no_grad():
        y_conv, _ = kernel(x, mode="conv")
        y_rec, _ = kernel(x, mode="recurrent")

    max_diff = (y_conv - y_rec).abs().max().item()
    assert torch.allclose(y_conv, y_rec, atol=1e-4, rtol=1e-4), \
        f"Max diff: {max_diff}"
    print(f"  PASSED (max diff: {max_diff:.2e})")


def test_gradient_flow():
    """Test that gradients flow correctly."""
    print("Testing gradient flow...")
    kernel = S4DKernel(d_model=64, d_state=16)
    x = torch.randn(2, 20, 64, requires_grad=True)

    y, _ = kernel(x, mode="conv")
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "Input gradient is None"
    assert kernel.B.grad is not None, "B gradient is None"
    assert kernel.C.grad is not None, "C gradient is None"
    assert kernel.log_dt.grad is not None, "log_dt gradient is None"
    assert kernel.D.grad is not None, "D gradient is None"
    print("  PASSED")


def test_long_sequence():
    """Should handle long sequences without OOM."""
    print("Testing 10K token sequence...")
    kernel = S4DKernel(d_model=256, d_state=64)

    x = torch.randn(1, 10000, 256)

    with torch.no_grad():
        y, _ = kernel(x, mode="conv")

    assert y.shape == (1, 10000, 256), f"Output shape mismatch: {y.shape}"
    print("  PASSED")


def test_s4d_layer_forward():
    """Test S4DLayer forward pass."""
    print("Testing S4DLayer forward...")
    layer = S4DLayer(d_model=256, d_state=64)
    x = torch.randn(2, 100, 256)

    y = layer(x)

    assert y.shape == (2, 100, 256), f"Output shape mismatch: {y.shape}"
    print("  PASSED")


def test_s4d_layer_bidirectional():
    """Test bidirectional mode."""
    print("Testing S4DLayer bidirectional...")
    layer = S4DLayer(d_model=256, d_state=64, bidirectional=True)
    x = torch.randn(2, 100, 256)

    y = layer(x)

    assert y.shape == (2, 100, 256), f"Output shape mismatch: {y.shape}"
    print("  PASSED")


def test_hippo_legs_diagonal():
    """Test HiPPO-LegS diagonal."""
    print("Testing HiPPO legs_diagonal...")
    A = HiPPO.legs_diagonal(64)

    assert A.shape == (64,), f"Shape mismatch: {A.shape}"
    assert torch.all(A < 0), "Not all negative"
    assert A[0] == -0.5, f"A[0] = {A[0]}, expected -0.5"
    assert A[63] == -63.5, f"A[63] = {A[63]}, expected -63.5"
    print("  PASSED")


def test_hippo_legs_full():
    """Test full HiPPO-LegS matrix."""
    print("Testing HiPPO legs full...")
    A = HiPPO.legs(16)

    assert A.shape == (16, 16), f"Shape mismatch: {A.shape}"
    assert torch.all(A.triu(1) == 0), "Not lower triangular"
    print("  PASSED")


def test_ssminit_dt():
    """Test dt initialization."""
    print("Testing SSMInit init_dt...")
    log_dt = SSMInit.init_dt(256, dt_min=0.001, dt_max=0.1, dt_init="random")

    assert log_dt.shape == (256,), f"Shape mismatch: {log_dt.shape}"
    dt = torch.exp(log_dt)
    assert torch.all(dt >= 0.001), "dt below min"
    assert torch.all(dt <= 0.1), "dt above max"
    print("  PASSED")


def test_ssminit_BC():
    """Test B/C initialization."""
    print("Testing SSMInit init_BC...")
    B, C = SSMInit.init_BC(256, 64, init="normal")

    assert B.shape == (256, 64), f"B shape mismatch: {B.shape}"
    assert C.shape == (256, 64), f"C shape mismatch: {C.shape}"
    print("  PASSED")


def test_rmsnorm():
    """Test RMSNorm."""
    print("Testing RMSNorm...")
    norm = RMSNorm(d_model=64)
    x = torch.randn(2, 100, 64)

    y = norm(x)

    assert y.shape == x.shape, f"Shape mismatch: {y.shape}"
    print("  PASSED")


def benchmark_forward_1k():
    """Benchmark forward pass for 1K tokens."""
    print("\nBenchmarking 1K tokens...")
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
    print(f"  1K tokens forward time: {avg_time*1000:.2f} ms")
    return avg_time


def benchmark_forward_10k():
    """Benchmark forward pass for 10K tokens."""
    print("\nBenchmarking 10K tokens...")
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
    print(f"  10K tokens forward time: {avg_time*1000:.2f} ms")
    return avg_time


def benchmark_memory():
    """Estimate memory usage."""
    print("\nBenchmarking memory usage...")
    import gc

    gc.collect()

    kernel = S4DKernel(d_model=256, d_state=64)
    x = torch.randn(1, 1000, 256)

    # Parameters memory
    param_bytes = sum(p.numel() * p.element_size() for p in kernel.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in kernel.buffers())
    total_model_mb = (param_bytes + buffer_bytes) / 1024 / 1024

    # Input memory
    input_mb = x.numel() * x.element_size() / 1024 / 1024

    print(f"  Model parameters: {param_bytes / 1024 / 1024:.2f} MB")
    print(f"  Model buffers: {buffer_bytes / 1024 / 1024:.4f} MB")
    print(f"  Input (1K tokens): {input_mb:.2f} MB")

    return total_model_mb, input_mb


def main():
    """Run all tests."""
    print("=" * 60)
    print("S4D SSM Test Suite")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("=" * 60)

    tests = [
        test_s4d_kernel_init,
        test_forward_recurrent_single,
        test_forward_recurrent_sequence,
        test_forward_conv,
        test_conv_recurrent_equivalence,
        test_gradient_flow,
        test_long_sequence,
        test_s4d_layer_forward,
        test_s4d_layer_bidirectional,
        test_hippo_legs_diagonal,
        test_hippo_legs_full,
        test_ssminit_dt,
        test_ssminit_BC,
        test_rmsnorm,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print("BENCHMARKS")
    print("=" * 60)

    try:
        time_1k = benchmark_forward_1k()
        time_10k = benchmark_forward_10k()
        model_mb, input_mb = benchmark_memory()
    except Exception as e:
        print(f"Benchmark failed: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
