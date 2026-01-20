"""
Test-Time Training (TTT) Model Implementation

This module contains the main TTT model architecture with:
- Configuration dataclass
- Main TTT language model class
- TTT layer implementations (Linear and MLP variants)
- Inner optimizer for test-time learning

Reference: https://arxiv.org/abs/2407.04620
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import TTTLayer, TTTLinear, TTTMLP, InnerOptimizer, TestTimeTrainer


@dataclass
class TTTConfig:
    """Configuration for Test-Time Training model.

    Attributes:
        hidden_dim: Hidden dimension of the model.
        num_layers: Number of TTT layers.
        num_heads: Number of attention heads (for multi-head TTT).
        vocab_size: Vocabulary size for language modeling.
        max_seq_len: Maximum sequence length.
        ttt_type: Type of TTT layer ("linear" or "mlp").
        mlp_hidden_dim: Hidden dimension for TTT-MLP variant.
        ttt_learning_rate: Learning rate for test-time training.
        mini_batch_size: Mini-batch size for batched TTT updates.
        use_rope: Whether to use rotary position embeddings.
        rope_base: Base for rotary embeddings.
        layer_norm_eps: Epsilon for layer normalization.
        dropout: Dropout probability.
        initializer_range: Range for weight initialization.
        tie_word_embeddings: Whether to tie input/output embeddings.
    """
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50257
    max_seq_len: int = 2048
    ttt_type: Literal["linear", "mlp"] = "linear"
    mlp_hidden_dim: int = 2048
    ttt_learning_rate: float = 1.0
    mini_batch_size: int = 16
    use_rope: bool = True
    rope_base: int = 10000
    layer_norm_eps: float = 1e-5
    dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "ttt_type": self.ttt_type,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "ttt_learning_rate": self.ttt_learning_rate,
            "mini_batch_size": self.mini_batch_size,
            "use_rope": self.use_rope,
            "rope_base": self.rope_base,
            "layer_norm_eps": self.layer_norm_eps,
            "dropout": self.dropout,
            "initializer_range": self.initializer_range,
            "tie_word_embeddings": self.tie_word_embeddings,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TTTConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


class TTTEmbedding(nn.Module):
    """Token and position embedding for TTT language model."""

    def __init__(self, config: TTTConfig):
        """Initialize embeddings.

        Args:
            config: TTT configuration.
        """
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Position embeddings (if not using RoPE)
        if not config.use_rope:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        else:
            self.position_embedding = None

        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed tokens.

        Args:
            input_ids: Token IDs (batch_size, seq_len).
            position_ids: Optional position IDs.

        Returns:
            Embedded tokens (batch_size, seq_len, hidden_dim).
        """
        seq_len = input_ids.size(1)

        # Token embedding
        x = self.token_embedding(input_ids)

        # Add position embedding if not using RoPE
        if self.position_embedding is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)
            x = x + self.position_embedding(position_ids)

        x = self.layer_norm(x)
        x = self.dropout(x)

        return x


class TTTBlock(nn.Module):
    """Single TTT block with TTT layer and MLP.

    Replaces transformer attention with TTT layer while keeping
    the feed-forward network structure.
    """

    def __init__(self, config: TTTConfig, layer_idx: int):
        """Initialize TTT block.

        Args:
            config: TTT configuration.
            layer_idx: Index of this layer.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Pre-normalization
        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

        # TTT layer (replaces attention)
        if config.ttt_type == "linear":
            self.ttt = TTTLinear(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                learning_rate=config.ttt_learning_rate,
                mini_batch_size=config.mini_batch_size,
                use_rope=config.use_rope,
                rope_base=config.rope_base,
            )
        else:
            self.ttt = TTTMLP(
                hidden_dim=config.hidden_dim,
                mlp_hidden_dim=config.mlp_hidden_dim,
                num_heads=config.num_heads,
                learning_rate=config.ttt_learning_rate,
                mini_batch_size=config.mini_batch_size,
                use_rope=config.use_rope,
                rope_base=config.rope_base,
            )

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        return_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim).
            hidden_state: Optional previous hidden state (weights).
            return_hidden_state: Whether to return updated hidden state.

        Returns:
            Tuple of:
                - Output tensor (batch_size, seq_len, hidden_dim).
                - Updated hidden state (if requested).
        """
        # TTT layer with residual
        residual = x
        x = self.norm1(x)
        x, new_hidden = self.ttt(x, hidden_state, return_hidden_state=True)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)

        if return_hidden_state:
            return x, new_hidden
        return x, None


class TTTLanguageModel(nn.Module):
    """TTT-based Language Model.

    A language model that replaces attention with learnable hidden states
    updated via gradient descent during inference. Achieves linear
    complexity in sequence length.

    Reference: https://arxiv.org/abs/2407.04620
    """

    def __init__(self, config: TTTConfig):
        """Initialize TTT language model.

        Args:
            config: TTT configuration.
        """
        super().__init__()
        self.config = config

        # Embeddings
        self.embedding = TTTEmbedding(config)

        # TTT blocks
        self.blocks = nn.ModuleList([
            TTTBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Input token IDs (batch_size, seq_len).
            hidden_states: Optional list of hidden states per layer.
            labels: Optional labels for computing loss.
            return_hidden_states: Whether to return hidden states.

        Returns:
            Dictionary containing:
                - logits: Output logits (batch_size, seq_len, vocab_size).
                - loss: Language modeling loss (if labels provided).
                - hidden_states: Updated hidden states (if requested).
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.embedding(input_ids)

        # Process through TTT blocks
        new_hidden_states = []
        for i, block in enumerate(self.blocks):
            layer_hidden = hidden_states[i] if hidden_states else None
            x, new_hidden = block(x, layer_hidden, return_hidden_state=True)
            new_hidden_states.append(new_hidden)

        # Output projection
        x = self.output_norm(x)
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        result = {"logits": logits}
        if loss is not None:
            result["loss"] = loss
        if return_hidden_states:
            result["hidden_states"] = new_hidden_states

        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Initial token IDs (batch_size, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Top-p (nucleus) filtering.

        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens).
        """
        self.eval()

        # Initialize hidden states
        hidden_states = None

        generated = input_ids

        for _ in range(max_new_tokens):
            # Forward pass (use last token for efficiency in stateful model)
            with torch.no_grad():
                outputs = self.forward(
                    generated[:, -1:] if hidden_states else generated,
                    hidden_states=hidden_states,
                    return_hidden_states=True,
                )

            logits = outputs["logits"][:, -1, :]  # Last position
            hidden_states = outputs["hidden_states"]

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def reset_hidden_states(self) -> None:
        """Reset hidden states in all TTT layers.

        Called at the start of processing a new sequence.
        """
        for block in self.blocks:
            if hasattr(block.ttt, "reset_hidden_state"):
                block.ttt.reset_hidden_state()

    @classmethod
    def from_pretrained(cls, path: str) -> "TTTLanguageModel":
        """Load pretrained TTT model.

        Args:
            path: Path to pretrained model checkpoint.

        Returns:
            Loaded TTT model.
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = TTTConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "config": self.config.to_dict(),
            "model_state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)


class TTTForSequenceClassification(nn.Module):
    """TTT model with sequence classification head."""

    def __init__(
        self,
        config: TTTConfig,
        num_labels: int,
        pooling: str = "mean",
    ):
        """Initialize classification model.

        Args:
            config: TTT configuration.
            num_labels: Number of output labels.
            pooling: Pooling type ("mean", "last", "cls").
        """
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.pooling = pooling

        # TTT backbone (without LM head)
        self.embedding = TTTEmbedding(config)
        self.blocks = nn.ModuleList([
            TTTBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        self.output_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Input token IDs (batch_size, seq_len).
            attention_mask: Optional attention mask for padding.
            labels: Optional labels for computing loss.

        Returns:
            Dictionary containing logits and optional loss.
        """
        # Embed and process through blocks
        x = self.embedding(input_ids)

        for block in self.blocks:
            x, _ = block(x)

        x = self.output_norm(x)

        # Pool sequence
        if self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pooling == "last":
            pooled = x[:, -1]
        else:  # cls (first token)
            pooled = x[:, 0]

        # Classify
        logits = self.classifier(pooled)

        result = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss

        return result
