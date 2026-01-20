"""
Transformer (2017)

Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
         Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
Paper: "Attention Is All You Need"
       NeurIPS 2017, arXiv:1706.03762

The Transformer architecture revolutionized sequence modeling by eliminating
recurrence entirely, relying solely on attention mechanisms. This enables
full parallelization and has become the foundation for virtually all modern
large language models.

Key Innovation:
    - No recurrence or convolution - purely attention-based
    - Encoder-decoder architecture with self-attention and cross-attention
    - Position-wise feed-forward networks
    - Residual connections and layer normalization
    - Positional encodings to inject sequence order information

Architecture Overview:
    Encoder (N=6 layers):
        Each layer has:
        1. Multi-Head Self-Attention (all positions attend to all positions)
        2. Position-wise Feed-Forward Network
        Both with residual connections and layer normalization

    Decoder (N=6 layers):
        Each layer has:
        1. Masked Multi-Head Self-Attention (causal masking)
        2. Multi-Head Cross-Attention (attends to encoder output)
        3. Position-wise Feed-Forward Network
        All with residual connections and layer normalization

Mathematical Formulation:
    Layer Normalization:
        LayerNorm(x) = gamma * (x - mu) / sqrt(var + eps) + beta
        Where mu, var are mean and variance over the last dimension

    Position-wise Feed-Forward Network:
        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

        Or with GELU (modern variant):
        FFN(x) = GELU(x W_1 + b_1) W_2 + b_2

        Typical dimensions: d_model=512, d_ff=2048

    Residual Connection:
        output = LayerNorm(x + Sublayer(x))

        Or Post-LN (original):
        output = x + LayerNorm(Sublayer(x))

    Encoder Layer:
        x = LayerNorm(x + MultiHeadAttention(x, x, x))
        x = LayerNorm(x + FFN(x))

    Decoder Layer:
        x = LayerNorm(x + MaskedMultiHeadAttention(x, x, x))
        x = LayerNorm(x + MultiHeadAttention(x, encoder_output, encoder_output))
        x = LayerNorm(x + FFN(x))

Pseudocode:
    ```
    class TransformerEncoder:
        def __init__(self, num_layers, d_model, num_heads, d_ff):
            self.layers = [EncoderLayer(d_model, num_heads, d_ff)
                          for _ in range(num_layers)]
            self.pos_encoding = PositionalEncoding(d_model)

        def forward(self, x, mask=None):
            x = x + self.pos_encoding(x)
            for layer in self.layers:
                x = layer(x, mask)
            return x

    class EncoderLayer:
        def forward(self, x, mask=None):
            # Self-attention with residual
            attn_out, _ = self.self_attention(x, x, x, mask)
            x = self.norm1(x + attn_out)

            # FFN with residual
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

            return x

    class TransformerDecoder:
        def __init__(self, num_layers, d_model, num_heads, d_ff):
            self.layers = [DecoderLayer(d_model, num_heads, d_ff)
                          for _ in range(num_layers)]
            self.pos_encoding = PositionalEncoding(d_model)

        def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
            x = x + self.pos_encoding(x)
            for layer in self.layers:
                x = layer(x, encoder_output, src_mask, tgt_mask)
            return x

    class DecoderLayer:
        def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
            # Masked self-attention
            self_attn_out, _ = self.self_attention(x, x, x, tgt_mask)
            x = self.norm1(x + self_attn_out)

            # Cross-attention to encoder
            cross_attn_out, _ = self.cross_attention(x, encoder_output,
                                                     encoder_output, src_mask)
            x = self.norm2(x + cross_attn_out)

            # FFN
            ffn_out = self.ffn(x)
            x = self.norm3(x + ffn_out)

            return x
    ```

Training Details (Original Paper):
    - Base model: d_model=512, h=8, d_ff=2048, N=6
    - Big model: d_model=1024, h=16, d_ff=4096, N=6
    - Optimizer: Adam with beta1=0.9, beta2=0.98, eps=1e-9
    - Learning rate: warmup + inverse sqrt decay
      lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    - Dropout: 0.1 on attention weights, residuals, embeddings
    - Label smoothing: 0.1

Historical Significance:
    - Eliminated the sequential computation bottleneck of RNNs
    - O(1) path length between any two positions
    - Enabled training on much longer sequences
    - Foundation for BERT, GPT, T5, and all modern LLMs
    - Spawned encoder-only (BERT), decoder-only (GPT), and encoder-decoder variants

Variants:
    - Pre-LN Transformer: LayerNorm before sublayer (more stable training)
    - GPT: Decoder-only with causal masking
    - BERT: Encoder-only with bidirectional attention
    - T5: Encoder-decoder with text-to-text framing
"""

from typing import Tuple, Optional, List
import numpy as np

from consciousness.ml_research.core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# Research index entry
TRANSFORMER = MLMethod(
    method_id="transformer_2017",
    name="Transformer",
    year=2017,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
    authors=[
        "Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit",
        "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"
    ],
    paper_title="Attention Is All You Need",
    paper_url="https://arxiv.org/abs/1706.03762",
    key_innovation=(
        "Eliminated recurrence and convolutions entirely, using only attention "
        "mechanisms. Encoder-decoder architecture with self-attention, cross-attention, "
        "positional encodings, and position-wise FFN. Enables full parallelization."
    ),
    mathematical_formulation="""
    Encoder Layer:
        x = LayerNorm(x + MultiHeadAttention(x, x, x))
        x = LayerNorm(x + FFN(x))

    Decoder Layer:
        x = LayerNorm(x + MaskedMHA(x, x, x))
        x = LayerNorm(x + MHA(x, encoder_out, encoder_out))
        x = LayerNorm(x + FFN(x))

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """,
    predecessors=[
        "bahdanau_attention_2014",
        "self_attention_2017",
        "multi_head_attention_2017",
    ],
    successors=["bert_2018", "gpt_2018", "gpt2_2019", "t5_2019", "gpt3_2020"],
    tags=[
        "transformer",
        "attention",
        "encoder-decoder",
        "parallelizable",
        "neural-machine-translation",
        "foundational",
    ],
    notes=(
        "The paper title 'Attention Is All You Need' became iconic. The architecture "
        "achieved state-of-the-art on WMT 2014 English-German (28.4 BLEU) and "
        "English-French (41.8 BLEU) translation benchmarks."
    ),
)


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Layer normalization.

    LayerNorm(x) = gamma * (x - mu) / sqrt(var + eps) + beta

    Normalizes over the last dimension (feature dimension).

    Args:
        x: Input tensor (..., d_model)
        gamma: Scale parameter (d_model,)
        beta: Shift parameter (d_model,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor (..., d_model)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + eps)
    return gamma * x_norm + beta


def position_wise_ffn(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    activation: str = "relu",
) -> np.ndarray:
    """
    Position-wise feed-forward network.

    FFN(x) = activation(x W_1 + b_1) W_2 + b_2

    Args:
        x: Input tensor (seq_len, d_model)
        W1: First linear layer weights (d_model, d_ff)
        b1: First linear layer bias (d_ff,)
        W2: Second linear layer weights (d_ff, d_model)
        b2: Second linear layer bias (d_model,)
        activation: Activation function ("relu" or "gelu")

    Returns:
        Output tensor (seq_len, d_model)
    """
    # First linear transformation
    hidden = x @ W1 + b1

    # Activation
    if activation == "relu":
        hidden = np.maximum(0, hidden)
    elif activation == "gelu":
        # Approximate GELU
        hidden = 0.5 * hidden * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (hidden + 0.044715 * hidden**3)
        ))
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Second linear transformation
    output = hidden @ W2 + b2

    return output


class TransformerEncoderLayer:
    """
    Single Transformer encoder layer.

    Consists of:
    1. Multi-head self-attention with residual + layer norm
    2. Position-wise FFN with residual + layer norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize encoder layer parameters.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate (not used in numpy implementation)
            seed: Random seed
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        if seed is not None:
            np.random.seed(seed)

        # Initialize weights
        scale_attn = np.sqrt(2.0 / (d_model + d_model))
        scale_ff1 = np.sqrt(2.0 / (d_model + d_ff))
        scale_ff2 = np.sqrt(2.0 / (d_ff + d_model))

        # Multi-head attention weights
        self.W_Q = np.random.randn(d_model, d_model) * scale_attn
        self.W_K = np.random.randn(d_model, d_model) * scale_attn
        self.W_V = np.random.randn(d_model, d_model) * scale_attn
        self.W_O = np.random.randn(d_model, d_model) * scale_attn

        # Layer norm 1 parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)

        # FFN weights
        self.W1 = np.random.randn(d_model, d_ff) * scale_ff1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale_ff2
        self.b2 = np.zeros(d_model)

        # Layer norm 2 parameters
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor (seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        from consciousness.ml_research.attention.self_attention import multi_head_attention

        # Self-attention with residual and layer norm
        attn_out, attn_weights = multi_head_attention(
            x, x, x,
            self.W_Q, self.W_K, self.W_V, self.W_O,
            self.num_heads, mask
        )
        x = layer_norm(x + attn_out, self.gamma1, self.beta1)

        # FFN with residual and layer norm
        ffn_out = position_wise_ffn(x, self.W1, self.b1, self.W2, self.b2)
        x = layer_norm(x + ffn_out, self.gamma2, self.beta2)

        return x, attn_weights


class TransformerDecoderLayer:
    """
    Single Transformer decoder layer.

    Consists of:
    1. Masked multi-head self-attention with residual + layer norm
    2. Multi-head cross-attention with residual + layer norm
    3. Position-wise FFN with residual + layer norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize decoder layer parameters.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate (not used in numpy implementation)
            seed: Random seed
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        if seed is not None:
            np.random.seed(seed)

        # Initialize weights
        scale_attn = np.sqrt(2.0 / (d_model + d_model))
        scale_ff1 = np.sqrt(2.0 / (d_model + d_ff))
        scale_ff2 = np.sqrt(2.0 / (d_ff + d_model))

        # Self-attention weights
        self.W_Q_self = np.random.randn(d_model, d_model) * scale_attn
        self.W_K_self = np.random.randn(d_model, d_model) * scale_attn
        self.W_V_self = np.random.randn(d_model, d_model) * scale_attn
        self.W_O_self = np.random.randn(d_model, d_model) * scale_attn

        # Layer norm 1
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)

        # Cross-attention weights
        self.W_Q_cross = np.random.randn(d_model, d_model) * scale_attn
        self.W_K_cross = np.random.randn(d_model, d_model) * scale_attn
        self.W_V_cross = np.random.randn(d_model, d_model) * scale_attn
        self.W_O_cross = np.random.randn(d_model, d_model) * scale_attn

        # Layer norm 2
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

        # FFN weights
        self.W1 = np.random.randn(d_model, d_ff) * scale_ff1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale_ff2
        self.b2 = np.zeros(d_model)

        # Layer norm 3
        self.gamma3 = np.ones(d_model)
        self.beta3 = np.zeros(d_model)

    def forward(
        self,
        x: np.ndarray,
        encoder_output: np.ndarray,
        src_mask: Optional[np.ndarray] = None,
        tgt_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through decoder layer.

        Args:
            x: Decoder input tensor (seq_len_tgt, d_model)
            encoder_output: Encoder output (seq_len_src, d_model)
            src_mask: Source padding mask
            tgt_mask: Target causal mask

        Returns:
            Tuple of (output, self_attn_weights, cross_attn_weights)
        """
        from consciousness.ml_research.attention.self_attention import multi_head_attention

        # Masked self-attention
        self_attn_out, self_attn_weights = multi_head_attention(
            x, x, x,
            self.W_Q_self, self.W_K_self, self.W_V_self, self.W_O_self,
            self.num_heads, tgt_mask
        )
        x = layer_norm(x + self_attn_out, self.gamma1, self.beta1)

        # Cross-attention to encoder
        cross_attn_out, cross_attn_weights = multi_head_attention(
            x, encoder_output, encoder_output,
            self.W_Q_cross, self.W_K_cross, self.W_V_cross, self.W_O_cross,
            self.num_heads, src_mask
        )
        x = layer_norm(x + cross_attn_out, self.gamma2, self.beta2)

        # FFN
        ffn_out = position_wise_ffn(x, self.W1, self.b1, self.W2, self.b2)
        x = layer_norm(x + ffn_out, self.gamma3, self.beta3)

        return x, self_attn_weights, cross_attn_weights


def create_transformer_lr_schedule(
    d_model: int,
    warmup_steps: int = 4000,
) -> callable:
    """
    Create learning rate schedule from the original Transformer paper.

    lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})

    Args:
        d_model: Model dimension
        warmup_steps: Number of warmup steps

    Returns:
        Function that takes step number and returns learning rate
    """
    def lr_schedule(step: int) -> float:
        step = max(step, 1)  # Avoid division by zero
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    return lr_schedule


def initialize_transformer_params(
    d_model: int,
    vocab_size: int,
    seed: Optional[int] = None,
) -> dict:
    """
    Initialize embedding matrices for Transformer.

    Args:
        d_model: Model dimension
        vocab_size: Vocabulary size
        seed: Random seed

    Returns:
        Dictionary with embedding parameters
    """
    if seed is not None:
        np.random.seed(seed)

    scale = np.sqrt(1.0 / d_model)

    return {
        "token_embedding": np.random.randn(vocab_size, d_model) * scale,
        "output_projection": np.random.randn(d_model, vocab_size) * scale,
    }
