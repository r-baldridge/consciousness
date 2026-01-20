"""
Ring Attention Model Tests

Basic tests for ring attention components including
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
            RingAttentionConfig,
            RingAttentionModel,
            RingAttentionLayer,
            RingCommunication,
            BlockwiseAttention,
            SequenceParallelism,
            OnlineSoftmax,
        )
        assert RingAttentionConfig is not None
        assert RingAttentionModel is not None
        assert RingAttentionLayer is not None
        assert RingCommunication is not None
        assert BlockwiseAttention is not None
        assert SequenceParallelism is not None
        assert OnlineSoftmax is not None

    def test_import_layers(self):
        """Test importing layers module."""
        from src.layers import (
            RotaryEmbedding,
            RMSNorm,
            ParallelLinear,
            GatedMLP,
            AsyncKVBuffer,
            CausalMaskCache,
            MemoryEfficientCrossAttention,
        )
        assert RotaryEmbedding is not None
        assert RMSNorm is not None
        assert ParallelLinear is not None
        assert GatedMLP is not None
        assert AsyncKVBuffer is not None
        assert CausalMaskCache is not None
        assert MemoryEfficientCrossAttention is not None


class TestConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test creating config with defaults."""
        from src.model import RingAttentionConfig

        config = RingAttentionConfig()

        assert config.hidden_dim == 4096
        assert config.num_heads == 32
        assert config.head_dim == 128
        assert config.block_size == 4096
        assert config.num_layers == 32
        assert config.causal is True

    def test_custom_config(self):
        """Test creating config with custom values."""
        from src.model import RingAttentionConfig

        config = RingAttentionConfig(
            hidden_dim=2048,
            num_heads=16,
            head_dim=128,
            block_size=8192,
            num_layers=24,
            causal=False,
        )

        assert config.hidden_dim == 2048
        assert config.num_heads == 16
        assert config.block_size == 8192
        assert config.num_layers == 24
        assert config.causal is False


class TestRingCommunication:
    """Test RingCommunication class."""

    def test_instantiation(self):
        """Test RingCommunication can be instantiated."""
        from src.model import RingCommunication

        comm = RingCommunication(world_size=8, rank=0)

        assert comm.world_size == 8
        assert comm.rank == 0
        assert comm.send_rank == 1
        assert comm.recv_rank == 7

    def test_ring_topology(self):
        """Test ring topology neighbor computation."""
        from src.model import RingCommunication

        # Test various ranks
        for rank in range(8):
            comm = RingCommunication(world_size=8, rank=rank)
            assert comm.send_rank == (rank + 1) % 8
            assert comm.recv_rank == (rank - 1 + 8) % 8

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_send_recv_single_device(self):
        """Test send/recv in single-device mode (copies tensor)."""
        import torch
        from src.model import RingCommunication

        comm = RingCommunication(world_size=1, rank=0)

        tensor = torch.randn(4, 32, 64)
        result = comm.ring_send_recv(tensor)

        assert result.shape == tensor.shape
        assert torch.allclose(result, tensor)


class TestOnlineSoftmax:
    """Test OnlineSoftmax for numerically stable aggregation."""

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_init_state(self):
        """Test initializing online softmax state."""
        import torch
        from src.model import OnlineSoftmax

        output, max_val, sum_exp = OnlineSoftmax.init_state(
            batch_size=4,
            seq_len=128,
            head_dim=64,
            num_heads=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert output.shape == (4, 8, 128, 64)
        assert max_val.shape == (4, 8, 128, 1)
        assert sum_exp.shape == (4, 8, 128, 1)

        # Output should be zeros
        assert (output == 0).all()
        # Max should be -inf
        assert (max_val == float("-inf")).all()

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_finalize(self):
        """Test finalizing online softmax."""
        import torch
        from src.model import OnlineSoftmax

        output = torch.randn(4, 8, 128, 64)
        sum_exp = torch.ones(4, 8, 128, 1) * 10.0

        result = OnlineSoftmax.finalize(output, sum_exp)

        expected = output / 10.0
        assert torch.allclose(result, expected, rtol=1e-5)


class TestBlockwiseAttention:
    """Test BlockwiseAttention class."""

    def test_instantiation(self):
        """Test BlockwiseAttention can be instantiated."""
        from src.model import BlockwiseAttention

        attn = BlockwiseAttention(
            head_dim=64,
            causal=True,
            dropout=0.0,
        )

        assert attn.head_dim == 64
        assert attn.causal is True

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_forward_shapes(self):
        """Test forward pass output shapes."""
        import torch
        from src.model import BlockwiseAttention

        attn = BlockwiseAttention(head_dim=64, causal=True)

        batch_size, num_heads = 4, 8
        q_len, k_len = 128, 128

        query = torch.randn(batch_size, num_heads, q_len, 64)
        key = torch.randn(batch_size, num_heads, k_len, 64)
        value = torch.randn(batch_size, num_heads, k_len, 64)

        scores, vals = attn(query, key, value, 0, 0)

        assert scores.shape == (batch_size, num_heads, q_len, k_len)
        assert vals.shape == value.shape


class TestSequenceParallelism:
    """Test SequenceParallelism class."""

    def test_instantiation(self):
        """Test SequenceParallelism can be instantiated."""
        from src.model import SequenceParallelism

        sp = SequenceParallelism(
            block_size=4096,
            world_size=8,
            rank=3,
        )

        assert sp.block_size == 4096
        assert sp.world_size == 8
        assert sp.rank == 3

    def test_effective_seq_len(self):
        """Test effective sequence length computation."""
        from src.model import SequenceParallelism

        sp = SequenceParallelism(
            block_size=4096,
            world_size=8,
            rank=0,
        )

        assert sp.get_effective_seq_len() == 4096 * 8

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_partition_sequence(self):
        """Test sequence partitioning."""
        import torch
        from src.model import SequenceParallelism

        block_size = 128
        world_size = 4
        seq_len = block_size * world_size

        # Test each rank gets correct partition
        for rank in range(world_size):
            sp = SequenceParallelism(block_size, world_size, rank)

            sequence = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).float()
            sequence = sequence.expand(2, seq_len, 64)  # B=2, L=512, D=64

            block = sp.partition_sequence(sequence)

            assert block.shape == (2, block_size, 64)


class TestRingAttentionLayer:
    """Test RingAttentionLayer class."""

    def test_instantiation(self):
        """Test RingAttentionLayer can be instantiated."""
        from src.model import RingAttentionConfig, RingAttentionLayer

        config = RingAttentionConfig(
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            block_size=128,
        )
        layer = RingAttentionLayer(config)

        assert layer is not None

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_forward(self):
        """Test forward pass."""
        import torch
        from src.model import (
            RingAttentionConfig,
            RingAttentionLayer,
            RingCommunication,
        )

        config = RingAttentionConfig(
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            block_size=128,
        )
        layer = RingAttentionLayer(config)
        comm = RingCommunication(world_size=1, rank=0)

        hidden_states = torch.randn(2, 128, 256)
        output = layer(hidden_states, comm, local_block_idx=0)

        assert output.shape == hidden_states.shape


class TestRingAttentionModel:
    """Test main RingAttentionModel class."""

    def test_instantiation(self):
        """Test RingAttentionModel can be instantiated."""
        from src.model import RingAttentionConfig, RingAttentionModel

        config = RingAttentionConfig(
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            block_size=128,
            num_layers=2,
        )
        model = RingAttentionModel(config, vocab_size=1000)

        assert model is not None

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_forward_with_embeddings(self):
        """Test forward pass with pre-computed embeddings."""
        import torch
        from src.model import RingAttentionConfig, RingAttentionModel

        config = RingAttentionConfig(
            hidden_dim=256,
            num_heads=4,
            head_dim=64,
            block_size=128,
            num_layers=2,
        )
        model = RingAttentionModel(config)

        inputs_embeds = torch.randn(2, 128, 256)
        output = model(inputs_embeds=inputs_embeds)

        assert output.shape == inputs_embeds.shape


class TestLayers:
    """Test custom layers."""

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_rotary_embedding(self):
        """Test RotaryEmbedding."""
        import torch
        from src.layers import RotaryEmbedding

        rope = RotaryEmbedding(dim=64)

        x = torch.randn(2, 8, 128, 64)  # B, H, S, D
        output = rope(x, position_offset=0)

        assert output.shape == x.shape

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_rms_norm(self):
        """Test RMSNorm."""
        import torch
        from src.layers import RMSNorm

        norm = RMSNorm(dim=256)

        x = torch.randn(4, 128, 256)
        output = norm(x)

        assert output.shape == x.shape

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_gated_mlp(self):
        """Test GatedMLP."""
        import torch
        from src.layers import GatedMLP

        mlp = GatedMLP(hidden_dim=256)

        x = torch.randn(4, 128, 256)
        output = mlp(x)

        assert output.shape == x.shape

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_causal_mask_cache(self):
        """Test CausalMaskCache."""
        import torch
        from src.layers import CausalMaskCache

        cache = CausalMaskCache(block_size=128, device=torch.device("cpu"))

        # Future block should be fully masked
        mask = cache.get_mask(query_block_idx=0, kv_block_idx=1)
        assert mask is not None
        assert mask.all()  # All True

        # Past block should have no mask
        mask = cache.get_mask(query_block_idx=1, kv_block_idx=0)
        assert mask is None

        # Same block should have triangular mask
        mask = cache.get_mask(query_block_idx=0, kv_block_idx=0)
        assert mask is not None
        # Upper triangular should be True
        assert not mask[0, 0]  # Diagonal not masked
        assert mask[0, 1]  # Upper triangle masked


def _torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
