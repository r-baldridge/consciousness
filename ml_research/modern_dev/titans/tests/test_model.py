"""
Titans Model Tests

Basic tests for Titans model components including:
- Configuration validation
- Model instantiation
- Forward pass shapes
- Memory modules and test-time learning
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTitansConfig:
    """Tests for TitansConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.model import TitansConfig

        config = TitansConfig()

        assert config.hidden_dim == 2048
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.memory_dim == 256
        assert config.memory_layers == 2
        assert config.memory_lr == 0.01
        assert config.surprise_threshold == 0.1
        assert config.variant == "MAG"

    def test_custom_config(self):
        """Test custom configuration."""
        from src.model import TitansConfig

        config = TitansConfig(
            hidden_dim=768,
            num_layers=12,
            memory_dim=128,
            variant="MAC",
        )

        assert config.hidden_dim == 768
        assert config.num_layers == 12
        assert config.memory_dim == 128
        assert config.variant == "MAC"


class TestSurpriseMetric:
    """Tests for SurpriseMetric module."""

    def test_output_shape(self):
        """Test surprise metric output shape."""
        import torch
        from src.model import SurpriseMetric

        batch_size = 2
        seq_len = 128
        hidden_dim = 64
        embed_dim = 64

        surprise = SurpriseMetric(hidden_dim, embed_dim)
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        targets = torch.randn(batch_size, seq_len, embed_dim)

        output = surprise(hidden_states, targets)

        assert output.shape == (batch_size, seq_len)

    def test_predictions_without_targets(self):
        """Test predictions returned when no targets."""
        import torch
        from src.model import SurpriseMetric

        surprise = SurpriseMetric(64, 64)
        hidden_states = torch.randn(2, 128, 64)

        output = surprise(hidden_states)

        assert output.shape == (2, 128, 64)

    def test_surprise_is_nonnegative(self):
        """Test that surprise values are non-negative."""
        import torch
        from src.model import SurpriseMetric

        surprise = SurpriseMetric(64, 64)
        hidden_states = torch.randn(2, 128, 64)
        targets = torch.randn(2, 128, 64)

        output = surprise(hidden_states, targets)

        assert (output >= 0).all()


class TestNeuralLongTermMemory:
    """Tests for NeuralLongTermMemory module."""

    def test_output_shape(self):
        """Test memory output shape."""
        import torch
        from src.model import NeuralLongTermMemory

        batch_size = 2
        seq_len = 128
        input_dim = 64
        memory_dim = 32

        memory = NeuralLongTermMemory(input_dim, memory_dim, num_layers=2)
        query = torch.randn(batch_size, seq_len, input_dim)

        output = memory(query)

        assert output.shape == (batch_size, seq_len, input_dim)

    def test_write_loss(self):
        """Test write loss computation."""
        import torch
        from src.model import NeuralLongTermMemory

        memory = NeuralLongTermMemory(64, 32)
        query = torch.randn(2, 128, 64)
        target = torch.randn(2, 128, 64)

        loss = memory.compute_write_loss(query, target)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_test_time_update(self):
        """Test that test-time update modifies parameters."""
        import torch
        from src.model import NeuralLongTermMemory

        memory = NeuralLongTermMemory(64, 32)

        # Save initial parameters
        initial_params = [p.clone() for p in memory.memory_network.parameters()]

        # Perform update
        query = torch.randn(1, 10, 64, requires_grad=True)
        target = torch.randn(1, 10, 64)

        memory.test_time_update(query, target, learning_rate=0.1)

        # Check parameters changed
        changed = False
        for old_p, new_p in zip(initial_params, memory.memory_network.parameters()):
            if not torch.allclose(old_p, new_p):
                changed = True
                break

        assert changed, "Parameters should change after test-time update"


class TestMemoryAsContext:
    """Tests for MemoryAsContext (MAC) module."""

    def test_output_shape(self):
        """Test MAC output shape."""
        import torch
        from src.model import MemoryAsContext

        batch_size = 2
        seq_len = 64
        hidden_dim = 64
        num_heads = 4
        head_dim = 16

        mac = MemoryAsContext(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            memory_dim=32,
        )
        x = torch.randn(batch_size, seq_len, hidden_dim)

        output = mac(x)

        assert output.shape == (batch_size, seq_len, hidden_dim)


class TestMemoryAsGate:
    """Tests for MemoryAsGate (MAG) module."""

    def test_output_shape(self):
        """Test MAG output shape."""
        import torch
        from src.model import MemoryAsGate

        batch_size = 2
        seq_len = 64
        hidden_dim = 64

        mag = MemoryAsGate(
            hidden_dim=hidden_dim,
            num_heads=4,
            head_dim=16,
            memory_dim=32,
        )
        x = torch.randn(batch_size, seq_len, hidden_dim)

        output = mag(x)

        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_gate_initialization(self):
        """Test that gate is initialized to favor attention."""
        from src.model import MemoryAsGate

        mag = MemoryAsGate(
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
            memory_dim=32,
            gate_init_bias=-2.0,
        )

        # Check gate bias is negative (favoring attention)
        gate_bias = mag.gate[-1].bias
        assert (gate_bias < 0).all()


class TestMemoryAsLayer:
    """Tests for MemoryAsLayer (MAL) module."""

    def test_output_shape(self):
        """Test MAL output shape."""
        import torch
        from src.model import MemoryAsLayer

        batch_size = 2
        seq_len = 64
        hidden_dim = 64

        mal = MemoryAsLayer(
            hidden_dim=hidden_dim,
            num_heads=4,
            head_dim=16,
            memory_dim=32,
        )
        x = torch.randn(batch_size, seq_len, hidden_dim)

        output = mal(x)

        assert output.shape == (batch_size, seq_len, hidden_dim)


class TestTitansBlock:
    """Tests for TitansBlock module."""

    def test_output_shape_mag(self):
        """Test Titans block output shape with MAG variant."""
        import torch
        from src.model import TitansBlock, TitansConfig

        config = TitansConfig(
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
            memory_dim=32,
            variant="MAG",
        )
        block = TitansBlock(config)

        batch_size = 2
        seq_len = 64
        x = torch.randn(batch_size, seq_len, config.hidden_dim)

        output, surprise = block(x)

        assert output.shape == x.shape

    def test_output_shape_mac(self):
        """Test Titans block output shape with MAC variant."""
        import torch
        from src.model import TitansBlock, TitansConfig

        config = TitansConfig(
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
            memory_dim=32,
            variant="MAC",
        )
        block = TitansBlock(config)

        x = torch.randn(2, 64, 64)
        output, surprise = block(x)

        assert output.shape == x.shape

    def test_output_shape_mal(self):
        """Test Titans block output shape with MAL variant."""
        import torch
        from src.model import TitansBlock, TitansConfig

        config = TitansConfig(
            hidden_dim=64,
            num_heads=4,
            head_dim=16,
            memory_dim=32,
            variant="MAL",
        )
        block = TitansBlock(config)

        x = torch.randn(2, 64, 64)
        output, surprise = block(x)

        assert output.shape == x.shape

    def test_surprise_computation(self):
        """Test that surprise is computed when targets provided."""
        import torch
        from src.model import TitansBlock, TitansConfig

        config = TitansConfig(hidden_dim=64, num_heads=4, head_dim=16)
        block = TitansBlock(config)

        x = torch.randn(2, 64, 64)
        targets = torch.randn(2, 64, 64)

        output, surprise = block(x, target_embeddings=targets)

        assert surprise is not None
        assert surprise.shape == (2, 64)


class TestTitansModel:
    """Tests for complete TitansModel."""

    def test_instantiation(self):
        """Test model can be instantiated."""
        from src.model import TitansModel, TitansConfig

        config = TitansConfig(
            hidden_dim=64,
            num_layers=2,
            vocab_size=1000,
            num_heads=4,
            head_dim=16,
        )
        model = TitansModel(config)

        assert model is not None
        assert len(model.blocks) == config.num_layers

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        import torch
        from src.model import TitansModel, TitansConfig

        config = TitansConfig(
            hidden_dim=64,
            num_layers=2,
            vocab_size=1000,
            num_heads=4,
            head_dim=16,
        )
        model = TitansModel(config)

        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, surprises = model(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_all_variants(self):
        """Test all integration variants work."""
        import torch
        from src.model import TitansModel, TitansConfig

        for variant in ["MAC", "MAG", "MAL"]:
            config = TitansConfig(
                hidden_dim=64,
                num_layers=2,
                vocab_size=1000,
                num_heads=4,
                head_dim=16,
                variant=variant,
            )
            model = TitansModel(config)

            input_ids = torch.randint(0, 1000, (2, 32))
            logits, _ = model(input_ids)

            assert logits.shape == (2, 32, 1000), f"Failed for variant {variant}"

    def test_memory_reset(self):
        """Test memory reset functionality."""
        import torch
        from src.model import TitansModel, TitansConfig

        config = TitansConfig(
            hidden_dim=64,
            num_layers=2,
            vocab_size=1000,
            num_heads=4,
            head_dim=16,
        )
        model = TitansModel(config)

        # Get initial memory state
        initial_params = []
        for block in model.blocks:
            if hasattr(block.integration, 'memory'):
                for p in block.integration.memory.parameters():
                    initial_params.append(p.clone())

        # Modify memory (simulate usage)
        for p in initial_params[:1]:
            with torch.no_grad():
                p.add_(torch.randn_like(p))

        # Reset memory
        model.reset_memory()

        # Memory parameters should be different after random init
        # (This is a weak test - mainly checking it doesn't crash)


class TestLayerComponents:
    """Tests for individual layer components."""

    def test_test_time_learner(self):
        """Test TestTimeLearner wrapper."""
        import torch
        from src.layers import TestTimeLearner

        # Simple module to wrap
        module = torch.nn.Linear(64, 64)
        learner = TestTimeLearner(module, learning_rate=0.01)

        # Forward pass
        x = torch.randn(2, 64)
        output = learner(x)

        assert output.shape == (2, 64)

    def test_test_time_learner_reset(self):
        """Test TestTimeLearner reset functionality."""
        import torch
        from src.layers import TestTimeLearner

        module = torch.nn.Linear(64, 64)
        learner = TestTimeLearner(module)

        # Modify parameters
        with torch.no_grad():
            module.weight.fill_(0)

        # Reset
        learner.reset()

        # Parameters should be back to initial
        assert not (module.weight == 0).all()

    def test_surprise_gate(self):
        """Test SurpriseGate module."""
        import torch
        from src.layers import SurpriseGate

        gate = SurpriseGate(hidden_dim=64, threshold=0.1)

        surprise = torch.randn(2, 128)
        values = torch.randn(2, 128, 64)

        gated_values, gate_values = gate(surprise, values)

        assert gated_values.shape == values.shape
        assert gate_values.shape == values.shape

    def test_memory_attention(self):
        """Test MemoryAttention module."""
        import torch
        from src.layers import MemoryAttention

        mem_attn = MemoryAttention(
            hidden_dim=64,
            num_heads=4,
            num_memory_slots=32,
        )

        x = torch.randn(2, 128, 64)
        output = mem_attn(x)

        assert output.shape == x.shape

    def test_memory_snapshot(self):
        """Test MemorySnapshot save/load."""
        import torch
        from src.layers import MemorySnapshot

        # Create some modules
        modules = [torch.nn.Linear(64, 64) for _ in range(2)]
        snapshot = MemorySnapshot(modules)

        # Save snapshot
        snapshot.save_snapshot("test")

        # Modify parameters
        with torch.no_grad():
            for m in modules:
                m.weight.fill_(0)

        # Load snapshot
        assert snapshot.load_snapshot("test")

        # Parameters should be restored
        for m in modules:
            assert not (m.weight == 0).all()

    def test_reconstruction_loss(self):
        """Test ReconstructionLoss module."""
        import torch
        from src.layers import ReconstructionLoss

        loss_fn = ReconstructionLoss(hidden_dim=64)

        original = torch.randn(2, 128, 64)
        reconstructed = torch.randn(2, 128, 64)

        total_loss, components = loss_fn(original, reconstructed)

        assert total_loss.ndim == 0  # Scalar
        assert "reconstruction" in components
        assert components["reconstruction"] >= 0

    def test_positional_memory(self):
        """Test PositionalMemory module."""
        import torch
        from src.layers import PositionalMemory

        pos_memory = PositionalMemory(
            hidden_dim=64,
            memory_dim=32,
            max_positions=1024,
        )

        x = torch.randn(2, 128, 64)
        output = pos_memory(x)

        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
