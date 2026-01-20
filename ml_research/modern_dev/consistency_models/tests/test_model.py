"""
Consistency Models Tests

Basic tests for consistency model components including
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
            ConsistencyConfig,
            ConsistencyModel,
            ConsistencyFunction,
            ConsistencyTraining,
            ConsistencyDistillation,
            SkipScaling,
            KarrasSchedule,
        )
        assert ConsistencyConfig is not None
        assert ConsistencyModel is not None
        assert ConsistencyFunction is not None
        assert ConsistencyTraining is not None
        assert ConsistencyDistillation is not None
        assert SkipScaling is not None
        assert KarrasSchedule is not None

    def test_import_layers(self):
        """Test importing layers module."""
        from src.layers import (
            Preconditioner,
            FourierEmbedding,
            AdaGroupNorm,
            ConsistencyResBlock,
            SelfAttention2d,
            ConsistencyUNet,
        )
        assert Preconditioner is not None
        assert FourierEmbedding is not None
        assert AdaGroupNorm is not None
        assert ConsistencyResBlock is not None
        assert SelfAttention2d is not None
        assert ConsistencyUNet is not None


class TestConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test creating config with defaults."""
        from src.model import ConsistencyConfig

        config = ConsistencyConfig()

        assert config.input_dim == 784
        assert config.hidden_dim == 256
        assert config.sigma_min == 0.002
        assert config.sigma_max == 80.0
        assert config.sigma_data == 0.5
        assert config.rho == 7.0
        assert config.num_timesteps == 18

    def test_custom_config(self):
        """Test creating config with custom values."""
        from src.model import ConsistencyConfig

        config = ConsistencyConfig(
            input_dim=3072,
            hidden_dim=512,
            sigma_min=0.001,
            sigma_max=100.0,
            num_timesteps=40,
            distillation=False,
        )

        assert config.input_dim == 3072
        assert config.hidden_dim == 512
        assert config.sigma_min == 0.001
        assert config.sigma_max == 100.0
        assert config.num_timesteps == 40
        assert config.distillation is False


class TestSkipScaling:
    """Test SkipScaling for boundary condition."""

    def test_instantiation(self):
        """Test SkipScaling can be instantiated."""
        from src.model import SkipScaling

        skip = SkipScaling(sigma_data=0.5)
        assert skip is not None

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_boundary_condition(self):
        """Test that f(x, epsilon) = x at boundary."""
        import torch
        from src.model import SkipScaling

        skip = SkipScaling(sigma_data=0.5, sigma_min=0.002)

        # At sigma = sigma_min, c_skip should be ~1 and c_out should be ~0
        sigma = torch.tensor([0.002])
        c_skip = skip.compute_c_skip(sigma)
        c_out = skip.compute_c_out(sigma)

        # c_skip should be close to 1
        assert c_skip.item() > 0.99

        # c_out should be close to 0
        assert abs(c_out.item()) < 0.01


class TestKarrasSchedule:
    """Test Karras noise schedule."""

    def test_instantiation(self):
        """Test KarrasSchedule can be instantiated."""
        from src.model import KarrasSchedule

        schedule = KarrasSchedule()
        assert schedule is not None

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_get_sigmas(self):
        """Test sigma schedule computation."""
        from src.model import KarrasSchedule

        schedule = KarrasSchedule(
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0,
            num_timesteps=18,
        )

        sigmas = schedule.get_sigmas()

        assert sigmas.shape == (18,)
        assert sigmas[0].item() == pytest.approx(80.0, rel=0.01)
        assert sigmas[-1].item() == pytest.approx(0.002, rel=0.01)
        # Sigmas should be decreasing
        assert (sigmas[:-1] > sigmas[1:]).all()

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_adjacent_sigmas(self):
        """Test getting adjacent sigma pairs."""
        import torch
        from src.model import KarrasSchedule

        schedule = KarrasSchedule(num_timesteps=10)

        indices = torch.tensor([0, 3, 5])
        sigma_t, sigma_t_minus_1 = schedule.get_adjacent_sigmas(indices)

        assert sigma_t.shape == (3,)
        assert sigma_t_minus_1.shape == (3,)
        # sigma_t should be greater than sigma_t_minus_1
        assert (sigma_t > sigma_t_minus_1).all()


class TestConsistencyFunction:
    """Test ConsistencyFunction network."""

    def test_instantiation(self):
        """Test ConsistencyFunction can be instantiated."""
        from src.model import ConsistencyConfig, ConsistencyFunction

        config = ConsistencyConfig()
        model = ConsistencyFunction(config)

        assert model is not None
        assert hasattr(model, "forward")

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_forward_pass(self):
        """Test ConsistencyFunction forward pass."""
        import torch
        from src.model import ConsistencyConfig, ConsistencyFunction

        config = ConsistencyConfig(input_dim=64, hidden_dim=32, num_layers=2)
        model = ConsistencyFunction(config)

        batch_size = 4
        x = torch.randn(batch_size, config.input_dim)
        sigma = torch.rand(batch_size) * 80.0 + 0.002  # Random sigmas

        output = model(x, sigma)

        assert output.shape == x.shape


class TestConsistencyTraining:
    """Test ConsistencyTraining objective."""

    def test_instantiation(self):
        """Test ConsistencyTraining can be instantiated."""
        from src.model import ConsistencyConfig, ConsistencyTraining

        config = ConsistencyConfig()
        trainer = ConsistencyTraining(config)

        assert trainer is not None
        assert hasattr(trainer, "model")
        assert hasattr(trainer, "model_ema")

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_loss_computation(self):
        """Test loss computation."""
        import torch
        from src.model import ConsistencyConfig, ConsistencyTraining

        config = ConsistencyConfig(
            input_dim=64, hidden_dim=32, num_layers=2, num_timesteps=5
        )
        trainer = ConsistencyTraining(config)

        x = torch.randn(4, config.input_dim)
        loss = trainer(x)

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Non-negative


class TestConsistencyDistillation:
    """Test ConsistencyDistillation objective."""

    def test_instantiation(self):
        """Test ConsistencyDistillation can be instantiated."""
        from src.model import ConsistencyConfig, ConsistencyDistillation

        config = ConsistencyConfig()
        trainer = ConsistencyDistillation(config, teacher_model=None)

        assert trainer is not None

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_loss_without_teacher(self):
        """Test loss computation without teacher (placeholder behavior)."""
        import torch
        from src.model import ConsistencyConfig, ConsistencyDistillation

        config = ConsistencyConfig(
            input_dim=64, hidden_dim=32, num_layers=2, num_timesteps=5
        )
        trainer = ConsistencyDistillation(config, teacher_model=None)

        x = torch.randn(4, config.input_dim)
        loss = trainer(x)

        assert loss.ndim == 0


class TestConsistencyModel:
    """Test main ConsistencyModel class."""

    def test_instantiation_training(self):
        """Test ConsistencyModel can be instantiated for training."""
        from src.model import ConsistencyConfig, ConsistencyModel

        config = ConsistencyConfig(distillation=False)
        model = ConsistencyModel(config)

        assert model is not None

    def test_instantiation_distillation(self):
        """Test ConsistencyModel can be instantiated for distillation."""
        from src.model import ConsistencyConfig, ConsistencyModel

        config = ConsistencyConfig(distillation=True)
        model = ConsistencyModel(config, teacher_model=None)

        assert model is not None

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_forward(self):
        """Test forward pass returns loss."""
        import torch
        from src.model import ConsistencyConfig, ConsistencyModel

        config = ConsistencyConfig(
            input_dim=64, hidden_dim=32, num_layers=2,
            num_timesteps=5, distillation=False
        )
        model = ConsistencyModel(config)

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
        from src.model import ConsistencyConfig, ConsistencyModel

        config = ConsistencyConfig(
            input_dim=64, hidden_dim=32, num_layers=2,
            num_timesteps=5, distillation=False
        )
        model = ConsistencyModel(config)

        samples = model.sample(batch_size=4, device="cpu", num_steps=1)

        assert samples.shape == (4, config.input_dim)


class TestLayers:
    """Test custom layers."""

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_preconditioner(self):
        """Test Preconditioner."""
        import torch
        from src.layers import Preconditioner

        precond = Preconditioner(sigma_data=0.5)

        x = torch.randn(4, 3, 32, 32)
        F_x = torch.randn(4, 3, 32, 32)
        sigma = torch.rand(4) * 80.0 + 0.002

        out = precond(x, F_x, sigma)

        assert out.shape == x.shape

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_fourier_embedding(self):
        """Test FourierEmbedding."""
        import torch
        from src.layers import FourierEmbedding

        dim = 64
        embed = FourierEmbedding(dim)

        sigma = torch.rand(4) * 80.0

        out = embed(sigma)

        assert out.shape == (4, dim)

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_ada_group_norm(self):
        """Test AdaGroupNorm."""
        import torch
        from src.layers import AdaGroupNorm

        num_channels = 64
        cond_dim = 128
        norm = AdaGroupNorm(num_channels, cond_dim)

        x = torch.randn(4, num_channels, 8, 8)
        cond = torch.randn(4, cond_dim)

        out = norm(x, cond)

        assert out.shape == x.shape


def _torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
