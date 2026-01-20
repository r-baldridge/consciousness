"""
Flow Matching Model Tests

Basic tests for flow matching model components including
imports, instantiation, and forward passes.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports:
    """Test that all modules can be imported."""

    def test_import_model(self):
        """Test importing main model module."""
        from src.model import (
            FlowMatchingConfig,
            FlowMatchingModel,
            VectorField,
            ODESolver,
            FlowPath,
            ConditionalFlowMatching,
        )
        assert FlowMatchingConfig is not None
        assert FlowMatchingModel is not None
        assert VectorField is not None
        assert ODESolver is not None
        assert FlowPath is not None
        assert ConditionalFlowMatching is not None

    def test_import_layers(self):
        """Test importing layers module."""
        from src.layers import (
            AdaptiveLayerNorm,
            FourierFeatures,
            ResidualBlock,
            AttentionBlock,
            VelocityUNet,
            OTCoupler,
        )
        assert AdaptiveLayerNorm is not None
        assert FourierFeatures is not None
        assert ResidualBlock is not None
        assert AttentionBlock is not None
        assert VelocityUNet is not None
        assert OTCoupler is not None


class TestConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test creating config with defaults."""
        from src.model import FlowMatchingConfig

        config = FlowMatchingConfig()

        assert config.input_dim == 784
        assert config.hidden_dim == 256
        assert config.num_layers == 4
        assert config.solver == "euler"
        assert config.num_steps == 50

    def test_custom_config(self):
        """Test creating config with custom values."""
        from src.model import FlowMatchingConfig

        config = FlowMatchingConfig(
            input_dim=3072,
            hidden_dim=512,
            num_layers=8,
            solver="heun",
            num_steps=25,
        )

        assert config.input_dim == 3072
        assert config.hidden_dim == 512
        assert config.num_layers == 8
        assert config.solver == "heun"
        assert config.num_steps == 25


class TestVectorField:
    """Test VectorField network."""

    def test_instantiation(self):
        """Test VectorField can be instantiated."""
        from src.model import FlowMatchingConfig, VectorField

        config = FlowMatchingConfig()
        model = VectorField(config)

        assert model is not None
        assert hasattr(model, "forward")

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_forward_pass(self):
        """Test VectorField forward pass."""
        import torch
        from src.model import FlowMatchingConfig, VectorField

        config = FlowMatchingConfig(input_dim=784, hidden_dim=128)
        model = VectorField(config)

        batch_size = 4
        x = torch.randn(batch_size, config.input_dim)
        t = torch.rand(batch_size)

        v = model(x, t)

        assert v.shape == x.shape

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_output_dtype(self):
        """Test VectorField output dtype matches input."""
        import torch
        from src.model import FlowMatchingConfig, VectorField

        config = FlowMatchingConfig()
        model = VectorField(config)

        x = torch.randn(2, config.input_dim)
        t = torch.rand(2)

        v = model(x, t)

        assert v.dtype == x.dtype


class TestODESolver:
    """Test ODE solver."""

    def test_instantiation(self):
        """Test ODESolver can be instantiated."""
        from src.model import ODESolver

        for solver_type in ["euler", "heun", "rk4"]:
            solver = ODESolver(solver_type)
            assert solver.solver_type == solver_type

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_euler_step(self):
        """Test Euler solver step."""
        import torch
        from src.model import ODESolver

        solver = ODESolver("euler")

        # Simple velocity field: v(x, t) = -x
        def velocity_fn(x, t):
            return -x

        x0 = torch.ones(4, 10)
        result = solver.solve(velocity_fn, x0, t_span=(0, 1), num_steps=100)

        # Should decay towards zero
        assert result.abs().mean() < x0.abs().mean()


class TestFlowPath:
    """Test FlowPath interpolation."""

    def test_instantiation(self):
        """Test FlowPath can be instantiated."""
        from src.model import FlowPath

        path = FlowPath()
        assert path is not None

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_interpolation_boundaries(self):
        """Test interpolation at t=0 and t=1."""
        import torch
        from src.model import FlowPath

        path = FlowPath()

        x0 = torch.randn(4, 10)
        x1 = torch.randn(4, 10)

        # At t=0, should be x0
        t0 = torch.zeros(4)
        x_at_0 = path.interpolate(x0, x1, t0)
        assert torch.allclose(x_at_0, x0)

        # At t=1, should be x1
        t1 = torch.ones(4)
        x_at_1 = path.interpolate(x0, x1, t1)
        assert torch.allclose(x_at_1, x1)

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_velocity(self):
        """Test velocity computation."""
        import torch
        from src.model import FlowPath

        path = FlowPath()

        x0 = torch.randn(4, 10)
        x1 = torch.randn(4, 10)

        # Velocity should be x1 - x0 for OT path
        v = path.velocity(x0, x1)
        expected = x1 - x0

        assert torch.allclose(v, expected)


class TestConditionalFlowMatching:
    """Test ConditionalFlowMatching training objective."""

    def test_instantiation(self):
        """Test CFM can be instantiated."""
        from src.model import FlowMatchingConfig, ConditionalFlowMatching

        config = FlowMatchingConfig()
        cfm = ConditionalFlowMatching(config)

        assert cfm is not None
        assert hasattr(cfm, "forward")
        assert hasattr(cfm, "sample")

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_loss_computation(self):
        """Test loss computation."""
        import torch
        from src.model import FlowMatchingConfig, ConditionalFlowMatching

        config = FlowMatchingConfig(input_dim=64, hidden_dim=32, num_layers=2)
        cfm = ConditionalFlowMatching(config)

        x1 = torch.randn(4, config.input_dim)
        loss = cfm(x1)

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Non-negative


class TestFlowMatchingModel:
    """Test main FlowMatchingModel class."""

    def test_instantiation(self):
        """Test FlowMatchingModel can be instantiated."""
        from src.model import FlowMatchingConfig, FlowMatchingModel

        config = FlowMatchingConfig()
        model = FlowMatchingModel(config)

        assert model is not None

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_forward(self):
        """Test forward pass returns loss."""
        import torch
        from src.model import FlowMatchingConfig, FlowMatchingModel

        config = FlowMatchingConfig(input_dim=64, hidden_dim=32, num_layers=2)
        model = FlowMatchingModel(config)

        x = torch.randn(4, config.input_dim)
        loss = model(x)

        assert loss.ndim == 0

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_sample(self):
        """Test sample generation."""
        import torch
        from src.model import FlowMatchingConfig, FlowMatchingModel

        config = FlowMatchingConfig(
            input_dim=64, hidden_dim=32, num_layers=2, num_steps=5
        )
        model = FlowMatchingModel(config)

        samples = model.sample(batch_size=4, device="cpu")

        assert samples.shape == (4, config.input_dim)


class TestLayers:
    """Test custom layers."""

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_adaptive_layer_norm(self):
        """Test AdaptiveLayerNorm."""
        import torch
        from src.layers import AdaptiveLayerNorm

        hidden_dim = 64
        cond_dim = 32
        layer = AdaptiveLayerNorm(hidden_dim, cond_dim)

        x = torch.randn(4, hidden_dim)
        cond = torch.randn(4, cond_dim)

        out = layer(x, cond)

        assert out.shape == x.shape

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_fourier_features(self):
        """Test FourierFeatures."""
        import torch
        from src.layers import FourierFeatures

        dim = 64
        layer = FourierFeatures(dim)

        t = torch.rand(4)

        out = layer(t)

        assert out.shape == (4, dim)


def _torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
