"""
Titans Model Implementation

Core model architecture for Titans - Learning to Memorize at Test Time.
Based on "Titans: Learning to Memorize at Test Time" (Google, 2025).

This module contains:
- TitansConfig: Configuration dataclass
- MemoryAsContext (MAC): Memory concatenated with attention KV
- NeuralLongTermMemory: Learnable memory module with test-time updates
- SurpriseMetric: Computes surprise for memory gating
- TitansBlock: Combined attention + memory block
- TitansModel: Complete language model
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TitansConfig:
    """Configuration for Titans model.

    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_dim: Hidden dimension size.
        num_layers: Number of Titans blocks.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        mlp_expansion: MLP expansion factor.
        context_length: Maximum context length.
        memory_dim: Dimension of memory module hidden layers.
        memory_layers: Number of layers in memory MLP.
        memory_lr: Learning rate for test-time memory updates.
        surprise_threshold: Threshold for triggering memory writes.
        variant: Integration variant ("MAC", "MAG", or "MAL").
        dropout: Dropout probability.
        layer_norm_eps: Layer normalization epsilon.
        use_bias: Whether to use bias in linear layers.
        tie_embeddings: Whether to tie input/output embeddings.
    """
    vocab_size: int = 50257
    hidden_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    head_dim: int = 128
    mlp_expansion: int = 4
    context_length: int = 8192
    memory_dim: int = 256
    memory_layers: int = 2
    memory_lr: float = 0.01
    surprise_threshold: float = 0.1
    variant: Literal["MAC", "MAG", "MAL"] = "MAG"
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    use_bias: bool = True
    tie_embeddings: bool = True
    initializer_range: float = 0.02
    memory_loss_weight: float = 0.1
    gate_init_bias: float = -2.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            dim: Dimension to normalize.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SurpriseMetric(nn.Module):
    """Computes surprise signal for memory gating.

    Surprise is computed as the prediction error on the current input:
        s_t = ||f(h_t) - embed(x_{t+1})||^2

    High surprise indicates novel/important information that should be memorized.
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        use_bias: bool = True,
    ):
        """Initialize surprise metric.

        Args:
            hidden_dim: Hidden state dimension.
            embed_dim: Embedding dimension (for prediction target).
            use_bias: Whether to use bias.
        """
        super().__init__()

        # Linear predictor: predicts next token embedding from hidden state
        self.predictor = nn.Linear(hidden_dim, embed_dim, bias=use_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute surprise signal.

        Args:
            hidden_states: Hidden states of shape (batch, seq_len, hidden_dim).
            target_embeddings: Target embeddings (shifted by 1) of shape
                (batch, seq_len, embed_dim). If None, returns predictions.

        Returns:
            Surprise signal of shape (batch, seq_len) or predictions if no target.
        """
        # Predict next token embeddings
        predictions = self.predictor(hidden_states)

        if target_embeddings is None:
            return predictions

        # Compute squared L2 error as surprise
        surprise = torch.sum((predictions - target_embeddings) ** 2, dim=-1)

        return surprise


class NeuralLongTermMemory(nn.Module):
    """Neural Long-Term Memory module.

    A small MLP that learns to store and retrieve information. Unlike
    traditional KV caches, this module can be updated at test time via
    gradient descent to memorize novel information.

    Key operations:
    - Read: Forward pass through the MLP
    - Write: Gradient update based on surprise/reconstruction loss
    """

    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        """Initialize memory module.

        Args:
            input_dim: Input (query) dimension.
            memory_dim: Hidden dimension of memory MLP.
            num_layers: Number of MLP layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim

        # Build memory MLP
        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, memory_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = memory_dim

        # Final layer projects back to input dim
        layers.append(nn.Linear(current_dim, input_dim))

        self.memory_network = nn.Sequential(*layers)

        # Learnable gate for memory contribution
        self.memory_gate = nn.Parameter(torch.zeros(input_dim))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Read from memory.

        Args:
            query: Query tensor of shape (batch, seq_len, input_dim).

        Returns:
            Retrieved memory of shape (batch, seq_len, input_dim).
        """
        return self.memory_network(query)

    def compute_write_loss(
        self,
        query: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for memory write.

        Args:
            query: Query tensor.
            target: Target tensor to memorize.

        Returns:
            Loss for gradient-based memory update.
        """
        retrieved = self.forward(query)
        loss = F.mse_loss(retrieved, target)
        return loss

    @torch.enable_grad()
    def test_time_update(
        self,
        query: torch.Tensor,
        target: torch.Tensor,
        learning_rate: float,
        num_steps: int = 1,
    ) -> None:
        """Update memory at test time via gradient descent.

        This is the key innovation of Titans - the memory module
        learns to memorize during inference.

        Args:
            query: Query tensor for the update.
            target: Target to memorize.
            learning_rate: Learning rate for update.
            num_steps: Number of gradient steps.
        """
        for _ in range(num_steps):
            loss = self.compute_write_loss(query, target)

            # Compute gradients only for memory network parameters
            grads = torch.autograd.grad(
                loss,
                self.memory_network.parameters(),
                retain_graph=False,
                create_graph=False,
            )

            # Apply gradient update
            with torch.no_grad():
                for param, grad in zip(self.memory_network.parameters(), grads):
                    param.sub_(learning_rate * grad)


class MemoryAsContext(nn.Module):
    """Memory as Context (MAC) integration variant.

    Memory output is concatenated with attention key/value pairs,
    providing additional context for the attention mechanism.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        memory_dim: int,
        num_memory_slots: int = 64,
        dropout: float = 0.0,
    ):
        """Initialize MAC module.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            memory_dim: Memory module dimension.
            num_memory_slots: Number of memory key-value slots.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_memory_slots = num_memory_slots

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim)

        # Memory module
        self.memory = NeuralLongTermMemory(
            input_dim=hidden_dim,
            memory_dim=memory_dim,
        )

        # Memory key/value projections
        self.memory_k_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.memory_v_proj = nn.Linear(hidden_dim, num_heads * head_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with memory as context.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Get memory output
        memory_out = self.memory(x)  # (batch, seq_len, hidden_dim)

        # Project memory to K, V
        memory_k = self.memory_k_proj(memory_out).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        memory_v = self.memory_v_proj(memory_out).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        # Concatenate memory with regular K, V
        k = torch.cat([k, memory_k], dim=1)
        v = torch.cat([v, memory_v], dim=1)

        # Transpose for attention
        q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = k.transpose(1, 2)  # (batch, heads, seq + mem_seq, head_dim)
        v = v.transpose(1, 2)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            # Extend mask for memory positions (memory is always attended)
            extended_mask = F.pad(attention_mask, (0, seq_len), value=0)
            attn_weights = attn_weights + extended_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)

        return output


class MemoryAsGate(nn.Module):
    """Memory as Gate (MAG) integration variant.

    Memory modulates the attention output through a learned gate,
    blending short-term attention with long-term memory.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        memory_dim: int,
        dropout: float = 0.0,
        gate_init_bias: float = -2.0,
    ):
        """Initialize MAG module.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            memory_dim: Memory module dimension.
            dropout: Dropout probability.
            gate_init_bias: Initial bias for gate (negative = attention-dominant).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim)

        # Memory module
        self.memory = NeuralLongTermMemory(
            input_dim=hidden_dim,
            memory_dim=memory_dim,
        )

        # Gate network: decides blend between attention and memory
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Initialize gate bias to favor attention initially
        with torch.no_grad():
            self.gate[-1].bias.fill_(gate_init_bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with memory as gate.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Standard attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)

        # Get memory output
        memory_output = self.memory(x)

        # Compute gate
        gate_input = torch.cat([attn_output, memory_output], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))

        # Blend attention and memory
        output = gate * attn_output + (1 - gate) * memory_output

        return output


class MemoryAsLayer(nn.Module):
    """Memory as Layer (MAL) integration variant.

    Memory is applied as a separate processing layer, with its output
    added to the attention output before the MLP.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        memory_dim: int,
        memory_ffn_dim: int = 512,
        dropout: float = 0.0,
    ):
        """Initialize MAL module.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            memory_dim: Memory module dimension.
            memory_ffn_dim: FFN dimension for memory processing.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim)

        # Memory module
        self.memory = NeuralLongTermMemory(
            input_dim=hidden_dim,
            memory_dim=memory_dim,
        )

        # Memory FFN for additional processing
        self.memory_ffn = nn.Sequential(
            nn.Linear(hidden_dim, memory_ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(memory_ffn_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with memory as layer.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Standard attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)

        # Get memory output and process
        memory_output = self.memory(x)
        memory_output = self.memory_ffn(memory_output)

        # Add attention and memory outputs
        output = attn_output + memory_output

        return output


class TitansBlock(nn.Module):
    """Titans block combining attention, memory, and MLP.

    Structure:
        x = x + Integration(RMSNorm(x))  # Attention + Memory
        x = x + MLP(RMSNorm(x))
    """

    def __init__(self, config: TitansConfig):
        """Initialize Titans block.

        Args:
            config: Titans configuration.
        """
        super().__init__()
        self.config = config

        # Normalizations
        self.norm1 = RMSNorm(config.hidden_dim, config.layer_norm_eps)
        self.norm2 = RMSNorm(config.hidden_dim, config.layer_norm_eps)

        # Choose integration variant
        if config.variant == "MAC":
            self.integration = MemoryAsContext(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                memory_dim=config.memory_dim,
                dropout=config.dropout,
            )
        elif config.variant == "MAG":
            self.integration = MemoryAsGate(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                memory_dim=config.memory_dim,
                dropout=config.dropout,
                gate_init_bias=config.gate_init_bias,
            )
        elif config.variant == "MAL":
            self.integration = MemoryAsLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                memory_dim=config.memory_dim,
                dropout=config.dropout,
            )
        else:
            raise ValueError(f"Unknown variant: {config.variant}")

        # Surprise metric for memory gating
        self.surprise = SurpriseMetric(
            hidden_dim=config.hidden_dim,
            embed_dim=config.hidden_dim,
        )

        # MLP
        mlp_dim = config.hidden_dim * config.mlp_expansion
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, mlp_dim, bias=config.use_bias),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_dim, config.hidden_dim, bias=config.use_bias),
            nn.Dropout(config.dropout),
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_embeddings: Optional[torch.Tensor] = None,
        update_memory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Titans block.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            attention_mask: Optional attention mask.
            target_embeddings: Target embeddings for surprise computation.
            update_memory: Whether to update memory at test time.

        Returns:
            Tuple of (output, surprise) where surprise is optional.
        """
        # Attention + Memory with residual
        normed = self.norm1(x)
        integration_out = self.integration(normed, attention_mask)
        x = x + self.dropout(integration_out)

        # Compute surprise if targets provided
        surprise = None
        if target_embeddings is not None:
            surprise = self.surprise(x, target_embeddings)

            # Optionally update memory based on surprise
            if update_memory and hasattr(self.integration, 'memory'):
                # Find positions with high surprise
                high_surprise_mask = surprise > self.config.surprise_threshold
                if high_surprise_mask.any():
                    # Update memory for high-surprise positions
                    self.integration.memory.test_time_update(
                        query=x[high_surprise_mask],
                        target=target_embeddings[high_surprise_mask],
                        learning_rate=self.config.memory_lr,
                    )

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, surprise


class TitansModel(nn.Module):
    """Complete Titans language model with test-time learning memory."""

    def __init__(self, config: TitansConfig):
        """Initialize Titans model.

        Args:
            config: Titans configuration.
        """
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Titans blocks
        self.blocks = nn.ModuleList([
            TitansBlock(config) for _ in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(config.hidden_dim, config.layer_norm_eps)

        # Output projection
        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize module weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)
        float_mask = torch.zeros(seq_len, seq_len, device=device)
        float_mask.masked_fill_(mask, float("-inf"))
        return float_mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        update_memory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass of Titans model.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            attention_mask: Optional attention mask.
            return_logits: Whether to compute and return logits.
            update_memory: Whether to update memory at test time.

        Returns:
            Tuple of (logits or hidden states, list of surprise values).
        """
        batch_size, seq_len = input_ids.shape

        # Token embedding
        x = self.embedding(input_ids)
        x = self.dropout(x)

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_len, x.device)

        # Compute target embeddings (shifted for next-token prediction)
        if update_memory or self.training:
            # Shift input_ids to get targets
            target_ids = F.pad(input_ids[:, 1:], (0, 1), value=0)
            target_embeddings = self.embedding(target_ids)
        else:
            target_embeddings = None

        # Pass through blocks
        surprises = []
        for block in self.blocks:
            x, surprise = block(
                x,
                attention_mask,
                target_embeddings,
                update_memory=update_memory,
            )
            if surprise is not None:
                surprises.append(surprise)

        # Final normalization
        x = self.final_norm(x)

        # Compute logits
        if return_logits:
            if self.lm_head is not None:
                logits = self.lm_head(x)
            else:
                logits = F.linear(x, self.embedding.weight)
            return logits, surprises if surprises else None

        return x, surprises if surprises else None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """Generate tokens with test-time memory updates.

        Args:
            input_ids: Starting token IDs of shape (batch, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            update_memory: Whether to update memory during generation.

        Returns:
            Generated token IDs.
        """
        for _ in range(max_new_tokens):
            # Truncate if too long
            if input_ids.shape[1] > self.config.context_length:
                input_ids = input_ids[:, -self.config.context_length:]

            # Forward pass with optional memory update
            logits, _ = self.forward(
                input_ids,
                update_memory=update_memory,
            )

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def reset_memory(self) -> None:
        """Reset memory modules to initial state.

        Call this before processing a new context to clear
        test-time learned information.
        """
        # Re-initialize memory networks
        for block in self.blocks:
            if hasattr(block.integration, 'memory'):
                for module in block.integration.memory.memory_network.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
