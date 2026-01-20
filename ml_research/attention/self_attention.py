"""
Self-Attention and Multi-Head Attention (2017)

Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
         Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
Paper: "Attention Is All You Need"
       NeurIPS 2017, arXiv:1706.03762

Self-attention allows each position in a sequence to attend to all other positions,
capturing global dependencies without the sequential constraints of RNNs.
Multi-head attention runs multiple attention operations in parallel, allowing the
model to jointly attend to information from different representation subspaces.

Key Innovation:
    - Self-attention: sequence attends to itself (Q, K, V all from same source)
    - Scaled dot-product: efficient attention with scaling factor sqrt(d_k)
    - Multi-head: multiple attention heads capture different relationships
    - Enables full parallelization across sequence positions

Mathematical Formulation:
    Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Where:
        Q = query matrix (seq_len, d_k)
        K = key matrix (seq_len, d_k)
        V = value matrix (seq_len, d_v)
        d_k = dimension of keys (scaling factor)

    For self-attention, Q, K, V are all derived from the same input X:
        Q = X W^Q
        K = X W^K
        V = X W^V

    Multi-Head Attention:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

        Where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    Parameters:
        W_i^Q in R^{d_model x d_k}
        W_i^K in R^{d_model x d_k}
        W_i^V in R^{d_model x d_v}
        W^O in R^{h*d_v x d_model}

    Typically: d_k = d_v = d_model / h

Why Scaling by sqrt(d_k)?
    For large d_k, the dot products can grow large in magnitude, pushing the
    softmax into regions with extremely small gradients. Scaling by sqrt(d_k)
    counteracts this effect, assuming Q and K components have unit variance.

Pseudocode:
    ```
    def scaled_dot_product_attention(Q, K, V, mask=None):
        # Q, K, V: (batch, seq_len, d_k/d_v)
        d_k = Q.shape[-1]

        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / sqrt(d_k)  # (batch, seq_len, seq_len)

        # Apply mask (for decoder self-attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -inf)

        # Softmax over keys
        attention_weights = softmax(scores, dim=-1)

        # Weighted sum of values
        output = attention_weights @ V  # (batch, seq_len, d_v)

        return output, attention_weights

    def multi_head_attention(Q, K, V, W_Q, W_K, W_V, W_O, num_heads, mask=None):
        batch_size, seq_len, d_model = Q.shape
        d_k = d_model // num_heads

        # Linear projections
        Q = Q @ W_Q  # (batch, seq_len, d_model)
        K = K @ W_K
        V = V @ W_V

        # Split into heads: (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

        # Apply attention for each head
        output, weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads: (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        output = output @ W_O

        return output, weights
    ```

Attention Patterns:
    1. Self-Attention (Encoder): Each position attends to all positions
       - Full attention matrix (seq_len x seq_len)

    2. Masked Self-Attention (Decoder): Position i only attends to j <= i
       - Lower triangular attention matrix
       - Prevents "looking ahead" during generation

    3. Cross-Attention (Decoder): Q from decoder, K/V from encoder
       - Attends to encoder outputs based on decoder state

Historical Significance:
    - Eliminated recurrence, enabling parallel training
    - O(1) path length for any two positions (vs O(n) for RNN)
    - Foundation for BERT, GPT, and all modern LLMs
    - Made training on very long sequences practical

Complexity:
    - Time: O(n^2 * d) for sequence length n and dimension d
    - Memory: O(n^2) for attention matrix
    - Motivates efficient attention variants (Linformer, Performer, etc.)
"""

from typing import Tuple, Optional
import numpy as np

from consciousness.ml_research.core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# Research index entries
SELF_ATTENTION = MLMethod(
    method_id="self_attention_2017",
    name="Self-Attention",
    year=2017,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ATTENTION,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
    authors=[
        "Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit",
        "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"
    ],
    paper_title="Attention Is All You Need",
    paper_url="https://arxiv.org/abs/1706.03762",
    key_innovation=(
        "Scaled dot-product self-attention allowing each position to attend to all "
        "others in O(1) sequential operations. Uses Q K^T / sqrt(d_k) scaling to "
        "prevent vanishing gradients in softmax for large dimensions."
    ),
    mathematical_formulation="""
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Self-attention: Q = XW^Q, K = XW^K, V = XW^V
    """,
    predecessors=["bahdanau_attention_2014", "luong_attention_2015"],
    successors=["bert_2018", "gpt_2018", "efficient_attention"],
    tags=[
        "self-attention",
        "scaled-dot-product",
        "transformer",
        "parallelizable",
        "global-context",
    ],
    notes=(
        "The sqrt(d_k) scaling is crucial: for large d_k, dot products grow large, "
        "pushing softmax into saturation regions with tiny gradients."
    ),
)


MULTI_HEAD_ATTENTION = MLMethod(
    method_id="multi_head_attention_2017",
    name="Multi-Head Attention",
    year=2017,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ATTENTION,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
    authors=[
        "Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit",
        "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"
    ],
    paper_title="Attention Is All You Need",
    paper_url="https://arxiv.org/abs/1706.03762",
    key_innovation=(
        "Multiple attention heads operating in parallel, each learning to attend "
        "to different aspects of the input. Heads are projected to lower dimension "
        "(d_k = d_model / h), concatenated, and linearly transformed."
    ),
    mathematical_formulation="""
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

    Where: head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    W_i^Q, W_i^K in R^{d_model x d_k}
    W_i^V in R^{d_model x d_v}
    W^O in R^{h*d_v x d_model}
    """,
    predecessors=["self_attention_2017"],
    successors=["bert_2018", "gpt_2018", "mqa_2019", "gqa_2023"],
    tags=[
        "multi-head",
        "attention",
        "transformer",
        "ensemble",
        "parallel-heads",
    ],
    notes=(
        "Original Transformer uses h=8 heads with d_k=d_v=d_model/h=64. "
        "Different heads learn different attention patterns (syntactic, semantic, etc.)."
    ),
)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Args:
        Q: Query matrix (seq_len_q, d_k) or (batch, seq_len_q, d_k)
        K: Key matrix (seq_len_k, d_k) or (batch, seq_len_k, d_k)
        V: Value matrix (seq_len_k, d_v) or (batch, seq_len_k, d_v)
        mask: Optional mask (seq_len_q, seq_len_k), 0 = masked, 1 = unmasked

    Returns:
        Tuple of (output, attention_weights)
        - output: (seq_len_q, d_v) or (batch, seq_len_q, d_v)
        - attention_weights: (seq_len_q, seq_len_k) or (batch, seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    # Compute attention scores: Q K^T / sqrt(d_k)
    # For 2D: (seq_q, d_k) @ (d_k, seq_k) -> (seq_q, seq_k)
    # For 3D: (batch, seq_q, d_k) @ (batch, d_k, seq_k) -> (batch, seq_q, seq_k)
    if Q.ndim == 2:
        scores = Q @ K.T / np.sqrt(d_k)
    else:
        scores = Q @ np.transpose(K, (0, 2, 1)) / np.sqrt(d_k)

    # Apply mask (set masked positions to -inf before softmax)
    if mask is not None:
        scores = np.where(mask == 1, scores, -1e9)

    # Softmax over keys (last dimension)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    # Weighted sum of values
    output = attention_weights @ V

    return output, attention_weights


def multi_head_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W_O: np.ndarray,
    num_heads: int,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-head attention.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    Where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    Args:
        Q: Query input (seq_len_q, d_model)
        K: Key input (seq_len_k, d_model)
        V: Value input (seq_len_k, d_model)
        W_Q: Query projection weights (d_model, d_model)
        W_K: Key projection weights (d_model, d_model)
        W_V: Value projection weights (d_model, d_model)
        W_O: Output projection weights (d_model, d_model)
        num_heads: Number of attention heads
        mask: Optional attention mask

    Returns:
        Tuple of (output, attention_weights)
        - output: (seq_len_q, d_model)
        - attention_weights: (num_heads, seq_len_q, seq_len_k)
    """
    seq_len_q = Q.shape[0]
    seq_len_k = K.shape[0]
    d_model = Q.shape[1]
    d_k = d_model // num_heads

    # Linear projections
    Q_proj = Q @ W_Q  # (seq_len_q, d_model)
    K_proj = K @ W_K  # (seq_len_k, d_model)
    V_proj = V @ W_V  # (seq_len_k, d_model)

    # Split into heads: reshape (seq_len, d_model) -> (num_heads, seq_len, d_k)
    Q_heads = Q_proj.reshape(seq_len_q, num_heads, d_k).transpose(1, 0, 2)
    K_heads = K_proj.reshape(seq_len_k, num_heads, d_k).transpose(1, 0, 2)
    V_heads = V_proj.reshape(seq_len_k, num_heads, d_k).transpose(1, 0, 2)

    # Apply scaled dot-product attention for each head
    head_outputs = []
    all_weights = []

    for h in range(num_heads):
        head_output, weights = scaled_dot_product_attention(
            Q_heads[h], K_heads[h], V_heads[h], mask
        )
        head_outputs.append(head_output)
        all_weights.append(weights)

    # Concatenate heads: (num_heads, seq_len_q, d_k) -> (seq_len_q, d_model)
    concat_output = np.concatenate(head_outputs, axis=-1)

    # Final linear projection
    output = concat_output @ W_O

    # Stack attention weights: (num_heads, seq_len_q, seq_len_k)
    attention_weights = np.stack(all_weights, axis=0)

    return output, attention_weights


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal (look-ahead) mask for decoder self-attention.

    Prevents position i from attending to positions j > i.

    Args:
        seq_len: Sequence length

    Returns:
        Lower triangular mask (seq_len, seq_len) where 1 = attend, 0 = masked
    """
    return np.tril(np.ones((seq_len, seq_len)))


def create_padding_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    """
    Create padding mask for variable-length sequences.

    Args:
        lengths: Actual lengths of each sequence (batch_size,)
        max_len: Maximum sequence length

    Returns:
        Mask (batch_size, max_len) where 1 = valid, 0 = padding
    """
    batch_size = len(lengths)
    mask = np.zeros((batch_size, max_len))
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask


def initialize_mha_parameters(
    d_model: int,
    num_heads: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize multi-head attention parameters.

    Uses Xavier/Glorot initialization.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        seed: Random seed for reproducibility

    Returns:
        Tuple of (W_Q, W_K, W_V, W_O) weight matrices
    """
    if seed is not None:
        np.random.seed(seed)

    scale = np.sqrt(2.0 / (d_model + d_model))

    W_Q = np.random.randn(d_model, d_model) * scale
    W_K = np.random.randn(d_model, d_model) * scale
    W_V = np.random.randn(d_model, d_model) * scale
    W_O = np.random.randn(d_model, d_model) * scale

    return W_Q, W_K, W_V, W_O
