"""
Ring Attention Model Implementation

Core model architecture for ring attention enabling near-infinite
context lengths through distributed attention computation.
Implements RingCommunication, BlockwiseAttention, and SequenceParallelism.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RingAttentionConfig:
    """Configuration for Ring Attention.

    Attributes:
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
        block_size: Size of sequence blocks per device
        num_layers: Number of transformer layers
        dropout: Dropout probability
        causal: Whether to use causal masking
        use_flash_attention: Whether to use Flash Attention for local computation
        overlap_communication: Whether to overlap communication with computation
        ring_impl: Ring implementation type ('nccl', 'gloo', 'custom')
    """
    hidden_dim: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    block_size: int = 4096
    num_layers: int = 32
    dropout: float = 0.0
    causal: bool = True
    use_flash_attention: bool = True
    overlap_communication: bool = True
    ring_impl: str = "nccl"


class RingCommunication:
    """Handles ring topology communication for distributed attention.

    Manages the rotation of key-value tensors around a ring of devices,
    enabling each device to eventually attend to all key-value pairs.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        ring_impl: str = "nccl",
    ):
        """Initialize ring communication.

        Args:
            world_size: Total number of devices in the ring
            rank: This device's rank in the ring
            ring_impl: Communication backend ('nccl', 'gloo', 'custom')
        """
        self.world_size = world_size
        self.rank = rank
        self.ring_impl = ring_impl

        # Compute neighbors in ring topology
        self.send_rank = (rank + 1) % world_size
        self.recv_rank = (rank - 1 + world_size) % world_size

    def ring_send_recv(
        self,
        send_tensor: torch.Tensor,
        recv_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Send tensor to next rank and receive from previous rank.

        Args:
            send_tensor: Tensor to send to next rank
            recv_buffer: Optional pre-allocated buffer for receiving

        Returns:
            Received tensor from previous rank
        """
        if recv_buffer is None:
            recv_buffer = torch.empty_like(send_tensor)

        # Placeholder implementation
        # In real implementation with distributed:
        # send_op = dist.isend(send_tensor, self.send_rank)
        # recv_op = dist.irecv(recv_buffer, self.recv_rank)
        # send_op.wait()
        # recv_op.wait()

        # For single-device testing, just copy
        recv_buffer.copy_(send_tensor)

        return recv_buffer

    def ring_send_recv_async(
        self,
        send_tensor: torch.Tensor,
        recv_buffer: torch.Tensor,
    ) -> Tuple[object, object]:
        """Async version for overlapping with computation.

        Args:
            send_tensor: Tensor to send
            recv_buffer: Buffer for receiving

        Returns:
            Tuple of (send_handle, recv_handle) for waiting
        """
        # Placeholder for async operations
        # send_op = dist.isend(send_tensor, self.send_rank)
        # recv_op = dist.irecv(recv_buffer, self.recv_rank)
        # return send_op, recv_op

        recv_buffer.copy_(send_tensor)
        return None, None

    def wait_all(self, handles: List[object]) -> None:
        """Wait for all async operations to complete.

        Args:
            handles: List of operation handles to wait for
        """
        # for handle in handles:
        #     if handle is not None:
        #         handle.wait()
        pass


class OnlineSoftmax:
    """Online softmax computation for numerically stable attention aggregation.

    Computes softmax incrementally across blocks without storing the full
    attention matrix, essential for memory-efficient ring attention.
    """

    @staticmethod
    def init_state(
        batch_size: int,
        seq_len: int,
        head_dim: int,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize online softmax state.

        Args:
            batch_size: Batch size
            seq_len: Query sequence length (local block size)
            head_dim: Dimension per head
            num_heads: Number of attention heads
            device: Device to create tensors on
            dtype: Data type

        Returns:
            Tuple of (output_accumulator, max_so_far, sum_so_far)
        """
        output = torch.zeros(
            batch_size, num_heads, seq_len, head_dim,
            device=device, dtype=dtype
        )
        max_val = torch.full(
            (batch_size, num_heads, seq_len, 1),
            float("-inf"),
            device=device, dtype=dtype
        )
        sum_exp = torch.zeros(
            batch_size, num_heads, seq_len, 1,
            device=device, dtype=dtype
        )
        return output, max_val, sum_exp

    @staticmethod
    def update(
        output: torch.Tensor,
        max_val: torch.Tensor,
        sum_exp: torch.Tensor,
        scores: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update online softmax state with new block.

        Args:
            output: Current output accumulator
            max_val: Current max values
            sum_exp: Current sum of exponentials
            scores: Attention scores for current block (B, H, Q, K)
            values: Values for current block (B, H, K, D)

        Returns:
            Updated (output, max_val, sum_exp)
        """
        # Compute max for this block
        block_max = scores.max(dim=-1, keepdim=True).values

        # Update global max
        new_max = torch.maximum(max_val, block_max)

        # Rescale factors
        old_scale = torch.exp(max_val - new_max)
        new_scale = torch.exp(block_max - new_max)

        # Update sum of exponentials
        block_exp = torch.exp(scores - block_max)
        block_sum = block_exp.sum(dim=-1, keepdim=True)
        sum_exp = sum_exp * old_scale + block_sum * new_scale

        # Compute attention for this block
        attn_weights = block_exp / (block_sum + 1e-8)
        block_output = torch.matmul(attn_weights, values)

        # Update output accumulator
        output = output * old_scale + block_output * new_scale

        return output, new_max, sum_exp

    @staticmethod
    def finalize(
        output: torch.Tensor,
        sum_exp: torch.Tensor,
    ) -> torch.Tensor:
        """Finalize output by normalizing.

        Args:
            output: Accumulated output
            sum_exp: Sum of exponentials

        Returns:
            Normalized attention output
        """
        return output / (sum_exp + 1e-8)


class BlockwiseAttention(nn.Module):
    """Computes attention for a single block pair.

    Handles the local attention computation between a query block
    and a key-value block, optionally using Flash Attention.
    """

    def __init__(
        self,
        head_dim: int,
        causal: bool = True,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
    ):
        """Initialize blockwise attention.

        Args:
            head_dim: Dimension per attention head
            causal: Whether to apply causal masking
            dropout: Attention dropout probability
            use_flash_attention: Whether to use Flash Attention
        """
        super().__init__()
        self.head_dim = head_dim
        self.causal = causal
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        self.scale = head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_block_idx: int,
        kv_block_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention for one block pair.

        Args:
            query: Query tensor (B, H, Q, D)
            key: Key tensor (B, H, K, D)
            value: Value tensor (B, H, K, D)
            query_block_idx: Index of query block (for causal masking)
            kv_block_idx: Index of key-value block (for causal masking)

        Returns:
            Tuple of (attention_output, attention_scores)
            where scores are pre-softmax for online aggregation
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if self.causal:
            scores = self._apply_causal_mask(
                scores, query_block_idx, kv_block_idx
            )

        return scores, value

    def _apply_causal_mask(
        self,
        scores: torch.Tensor,
        query_block_idx: int,
        kv_block_idx: int,
    ) -> torch.Tensor:
        """Apply causal mask to attention scores.

        Args:
            scores: Attention scores (B, H, Q, K)
            query_block_idx: Index of query block
            kv_block_idx: Index of key-value block

        Returns:
            Masked attention scores
        """
        if kv_block_idx > query_block_idx:
            # Future block - fully masked
            return scores.fill_(float("-inf"))
        elif kv_block_idx < query_block_idx:
            # Past block - no masking needed
            return scores
        else:
            # Same block - apply triangular mask
            q_len, k_len = scores.shape[-2], scores.shape[-1]
            mask = torch.triu(
                torch.ones(q_len, k_len, device=scores.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
            return scores


class SequenceParallelism:
    """Manages sequence-level parallelism for ring attention.

    Handles partitioning of sequences across devices and manages
    the assignment of blocks to different ranks.
    """

    def __init__(
        self,
        block_size: int,
        world_size: int,
        rank: int,
    ):
        """Initialize sequence parallelism manager.

        Args:
            block_size: Size of each sequence block
            world_size: Total number of devices
            rank: This device's rank
        """
        self.block_size = block_size
        self.world_size = world_size
        self.rank = rank

    def partition_sequence(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Partition a full sequence and return this rank's block.

        Args:
            sequence: Full sequence tensor (B, L, D)

        Returns:
            This rank's block (B, block_size, D)
        """
        batch_size, seq_len, dim = sequence.shape

        # Calculate start and end indices for this rank
        start_idx = self.rank * self.block_size
        end_idx = start_idx + self.block_size

        # Handle edge cases
        if start_idx >= seq_len:
            # This rank gets padding
            return torch.zeros(
                batch_size, self.block_size, dim,
                device=sequence.device, dtype=sequence.dtype
            )

        end_idx = min(end_idx, seq_len)
        block = sequence[:, start_idx:end_idx, :]

        # Pad if needed
        if block.shape[1] < self.block_size:
            padding = torch.zeros(
                batch_size, self.block_size - block.shape[1], dim,
                device=sequence.device, dtype=sequence.dtype
            )
            block = torch.cat([block, padding], dim=1)

        return block

    def gather_sequence(
        self,
        local_block: torch.Tensor,
    ) -> torch.Tensor:
        """Gather all blocks to reconstruct full sequence.

        Args:
            local_block: This rank's block (B, block_size, D)

        Returns:
            Full sequence (B, L, D) where L = block_size * world_size
        """
        # Placeholder - would use all_gather in distributed setting
        # In real implementation:
        # blocks = [torch.empty_like(local_block) for _ in range(self.world_size)]
        # dist.all_gather(blocks, local_block)
        # return torch.cat(blocks, dim=1)

        # Single-device fallback
        return local_block

    def get_effective_seq_len(self) -> int:
        """Get the effective total sequence length across all devices."""
        return self.block_size * self.world_size


class RingAttentionLayer(nn.Module):
    """Single Ring Attention layer.

    Implements multi-head attention using ring communication pattern
    for sequence parallelism across devices.
    """

    def __init__(self, config: RingAttentionConfig):
        """Initialize ring attention layer.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # Projections
        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_dim)

        # Blockwise attention
        self.blockwise_attn = BlockwiseAttention(
            config.head_dim,
            config.causal,
            config.dropout,
            config.use_flash_attention,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ring_comm: RingCommunication,
        local_block_idx: int,
    ) -> torch.Tensor:
        """Forward pass with ring communication.

        Args:
            hidden_states: Input tensor (B, block_size, hidden_dim)
            ring_comm: Ring communication handler
            local_block_idx: This device's block index

        Returns:
            Output tensor (B, block_size, hidden_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (B, H, S, D)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Initialize online softmax state
        output, max_val, sum_exp = OnlineSoftmax.init_state(
            batch_size, seq_len, self.head_dim, self.num_heads,
            hidden_states.device, hidden_states.dtype
        )

        # Ring attention loop
        current_key = key
        current_value = value

        for step in range(ring_comm.world_size):
            kv_block_idx = (local_block_idx + step) % ring_comm.world_size

            # Compute attention for this block pair
            scores, vals = self.blockwise_attn(
                query, current_key, current_value,
                local_block_idx, kv_block_idx
            )

            # Update online softmax state
            output, max_val, sum_exp = OnlineSoftmax.update(
                output, max_val, sum_exp, scores, vals
            )

            # Ring rotation (except on last step)
            if step < ring_comm.world_size - 1:
                # Pack K and V for communication
                kv_tensor = torch.cat([current_key, current_value], dim=-1)
                kv_recv = ring_comm.ring_send_recv(kv_tensor)

                # Unpack
                current_key, current_value = kv_recv.chunk(2, dim=-1)

        # Finalize attention output
        attn_output = OnlineSoftmax.finalize(output, sum_exp)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return self.dropout(output)


class RingAttentionBlock(nn.Module):
    """Transformer block with ring attention and feed-forward."""

    def __init__(self, config: RingAttentionConfig):
        """Initialize ring attention block.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.attention = RingAttentionLayer(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        ring_comm: RingCommunication,
        local_block_idx: int,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor
            ring_comm: Ring communication handler
            local_block_idx: This device's block index

        Returns:
            Output tensor
        """
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, ring_comm, local_block_idx)
        hidden_states = residual + hidden_states

        # Pre-norm feed-forward
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class RingAttentionModel(nn.Module):
    """Full Ring Attention transformer model.

    Implements a complete transformer with ring attention for
    distributed sequence processing.
    """

    def __init__(
        self,
        config: RingAttentionConfig,
        vocab_size: Optional[int] = None,
    ):
        """Initialize Ring Attention model.

        Args:
            config: Model configuration
            vocab_size: Vocabulary size (None for continuous inputs)
        """
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Embedding (if using vocabulary)
        if vocab_size is not None:
            self.embed = nn.Embedding(vocab_size, config.hidden_dim)
        else:
            self.embed = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            RingAttentionBlock(config)
            for _ in range(config.num_layers)
        ])

        # Output projection
        self.norm_out = nn.LayerNorm(config.hidden_dim)
        if vocab_size is not None:
            self.lm_head = nn.Linear(config.hidden_dim, vocab_size, bias=False)
        else:
            self.lm_head = None

        # Initialize ring communication (placeholder)
        self._ring_comm = None
        self._local_block_idx = 0

    def setup_distributed(
        self,
        world_size: int,
        rank: int,
    ) -> None:
        """Set up distributed training/inference.

        Args:
            world_size: Total number of devices
            rank: This device's rank
        """
        self._ring_comm = RingCommunication(
            world_size, rank, self.config.ring_impl
        )
        self._local_block_idx = rank

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        ring_comm: Optional[RingCommunication] = None,
        local_block_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs (B, seq_len) - requires embed layer
            inputs_embeds: Pre-computed embeddings (B, seq_len, hidden_dim)
            ring_comm: Ring communication handler (uses default if None)
            local_block_idx: Block index (uses default if None)

        Returns:
            Output logits or hidden states
        """
        # Get embeddings
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif input_ids is not None and self.embed is not None:
            hidden_states = self.embed(input_ids)
        else:
            raise ValueError("Must provide input_ids or inputs_embeds")

        # Use defaults if not provided
        if ring_comm is None:
            if self._ring_comm is None:
                # Single-device fallback
                self._ring_comm = RingCommunication(1, 0)
            ring_comm = self._ring_comm

        if local_block_idx is None:
            local_block_idx = self._local_block_idx

        # Apply transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, ring_comm, local_block_idx)

        # Output
        hidden_states = self.norm_out(hidden_states)

        if self.lm_head is not None:
            return self.lm_head(hidden_states)

        return hidden_states

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def estimate_memory_per_device(
        self,
        batch_size: int = 1,
        dtype_bytes: int = 2,
    ) -> float:
        """Estimate memory usage per device in GB.

        Args:
            batch_size: Batch size
            dtype_bytes: Bytes per element (2 for bf16/fp16)

        Returns:
            Estimated memory in GB
        """
        block_size = self.config.block_size
        hidden_dim = self.config.hidden_dim
        num_layers = self.config.num_layers

        # Activations per layer
        activation_mem = batch_size * block_size * hidden_dim * dtype_bytes

        # KV cache per layer
        kv_mem = 2 * batch_size * block_size * hidden_dim * dtype_bytes

        # Total for all layers
        total_bytes = (activation_mem + kv_mem) * num_layers

        # Add model parameters
        param_bytes = self.get_num_params() * dtype_bytes
        total_bytes += param_bytes

        return total_bytes / (1024 ** 3)


# Export all public classes
__all__ = [
    "RingAttentionConfig",
    "RingAttentionModel",
    "RingAttentionLayer",
    "RingAttentionBlock",
    "RingCommunication",
    "BlockwiseAttention",
    "SequenceParallelism",
    "OnlineSoftmax",
]
