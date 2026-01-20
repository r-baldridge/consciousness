"""
Positional Encodings

A collection of positional encoding methods that inject sequence order information
into Transformer models, which otherwise have no notion of position.

Methods Included:
    1. Sinusoidal Positional Encoding (2017) - Vaswani et al.
    2. Learned Positional Embeddings (various)
    3. Rotary Position Embedding (RoPE, 2021) - Su et al.
    4. ALiBi (2021) - Press et al.

================================================================================
1. SINUSOIDAL POSITIONAL ENCODING (Original Transformer, 2017)
================================================================================

Authors: Vaswani et al.
Paper: "Attention Is All You Need"

Mathematical Formulation:
    PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    Where:
        pos = position in sequence (0, 1, 2, ...)
        i = dimension index (0, 1, ..., d_model/2 - 1)
        d_model = embedding dimension

Key Properties:
    - Deterministic (no learned parameters)
    - Each position has a unique encoding
    - Can represent relative positions: PE(pos+k) is a linear function of PE(pos)
    - Extrapolates to longer sequences than seen in training

================================================================================
2. LEARNED POSITIONAL EMBEDDINGS
================================================================================

Simply learn a position embedding matrix P in R^{max_len x d_model}.
Add P[pos] to token embedding at position pos.

Pros: Can learn arbitrary position-dependent patterns
Cons: Cannot extrapolate beyond max_len seen in training

================================================================================
3. ROTARY POSITION EMBEDDING (RoPE, 2021)
================================================================================

Authors: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
       arXiv:2104.09864, April 2021

Key Innovation:
    - Encodes position through rotation of query/key vectors
    - Relative position information is automatically captured in attention
    - Natural decay of attention with distance

Mathematical Formulation:
    For position m and dimension pair (2i, 2i+1):

    Rotation matrix R_m:
        R_m = [[cos(m*theta_i), -sin(m*theta_i)],
               [sin(m*theta_i),  cos(m*theta_i)]]

    Where theta_i = 10000^{-2i/d}

    Applied to query q and key k at positions m and n:
        f_q(q, m) = R_m @ q
        f_k(k, n) = R_n @ k

    Inner product captures relative position:
        <f_q(q, m), f_k(k, n)> = <R_m @ q, R_n @ k>
                                = q^T R_{m-n} k

Properties:
    - Relative position dependency through rotation
    - Extrapolates well to longer sequences
    - No additional memory for position embeddings
    - Used in LLaMA, PaLM, and many modern LLMs

================================================================================
4. ALiBi - Attention with Linear Biases (2021)
================================================================================

Authors: Ofir Press, Noah A. Smith, Mike Lewis
Paper: "Train Short, Test Long: Attention with Linear Biases Enables Input
        Length Extrapolation"
       ICLR 2022, arXiv:2108.12409, August 2021

Key Innovation:
    - Add position-dependent linear bias to attention scores
    - No positional encodings added to embeddings
    - Excellent length extrapolation

Mathematical Formulation:
    softmax(q_i K^T / sqrt(d) + m * [0, -1, -2, ..., -(i-1)])

    Where m is a head-specific slope.

    For h heads, slopes are set to:
        m_i = 2^{-8/h} for i = 1, ..., h

    Full attention bias matrix B for position i attending to positions 0..i:
        B[i, j] = -m * (i - j)   for j <= i
        B[i, j] = -inf           for j > i (causal masking)

Properties:
    - Penalizes attention to distant positions linearly
    - Different heads use different slopes (multi-scale attention)
    - Trains on short sequences, extrapolates to much longer
    - Simpler than learned or sinusoidal encodings
    - Used in BLOOM and other models

================================================================================
"""

from typing import Optional, Tuple
import numpy as np

from consciousness.ml_research.core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# Research index entries
SINUSOIDAL_POSITIONAL_ENCODING = MLMethod(
    method_id="sinusoidal_positional_encoding_2017",
    name="Sinusoidal Positional Encoding",
    year=2017,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=[
        "Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit",
        "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"
    ],
    paper_title="Attention Is All You Need",
    paper_url="https://arxiv.org/abs/1706.03762",
    key_innovation=(
        "Fixed sinusoidal functions encode absolute position. Each dimension uses "
        "a different frequency, allowing the model to learn relative positions as "
        "PE(pos+k) is a linear function of PE(pos)."
    ),
    mathematical_formulation="""
    PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    """,
    predecessors=[],
    successors=["learned_positional_embeddings", "rope_2021", "alibi_2021"],
    tags=[
        "positional-encoding",
        "transformer",
        "sinusoidal",
        "fixed-encoding",
    ],
    notes=(
        "The sinusoidal encoding allows extrapolation to longer sequences than "
        "seen during training, unlike learned position embeddings."
    ),
)


ROTARY_POSITION_EMBEDDING = MLMethod(
    method_id="rope_2021",
    name="Rotary Position Embedding (RoPE)",
    year=2021,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=[
        "Jianlin Su", "Yu Lu", "Shengfeng Pan",
        "Ahmed Murtadha", "Bo Wen", "Yunfeng Liu"
    ],
    paper_title="RoFormer: Enhanced Transformer with Rotary Position Embedding",
    paper_url="https://arxiv.org/abs/2104.09864",
    key_innovation=(
        "Encodes position by rotating query and key vectors in 2D subspaces. "
        "The inner product naturally captures relative position m-n through "
        "rotation matrix R_{m-n}. No extra parameters needed."
    ),
    mathematical_formulation="""
    R_m = rotation matrix with angle m*theta_i for each 2D subspace
    f_q(q, m) = R_m @ q
    f_k(k, n) = R_n @ k
    <f_q, f_k> = q^T R_{m-n} k (captures relative position)

    theta_i = 10000^{-2i/d} (different frequency per dimension pair)
    """,
    predecessors=["sinusoidal_positional_encoding_2017"],
    successors=["yarn_2023", "longrope_2024"],
    tags=[
        "positional-encoding",
        "rotary",
        "relative-position",
        "extrapolation",
    ],
    notes=(
        "Used in LLaMA, PaLM, GPT-NeoX, and many modern LLMs. "
        "Enables better length extrapolation than learned embeddings."
    ),
)


ALIBI = MLMethod(
    method_id="alibi_2021",
    name="ALiBi (Attention with Linear Biases)",
    year=2021,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=["Ofir Press", "Noah A. Smith", "Mike Lewis"],
    paper_title="Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation",
    paper_url="https://arxiv.org/abs/2108.12409",
    key_innovation=(
        "Adds linear position-dependent bias to attention scores instead of "
        "positional encodings. Simple penalty -m*(i-j) for attention from "
        "position i to j. Different slopes m for different heads."
    ),
    mathematical_formulation="""
    Attention(i) = softmax(q_i K^T / sqrt(d) + bias_i)
    bias_i[j] = -m * (i - j)  for j <= i
    m_h = 2^{-8/num_heads}  for head h

    Slopes form geometric sequence: 1/2, 1/4, 1/8, ... for 8 heads
    """,
    predecessors=["sinusoidal_positional_encoding_2017"],
    successors=[],
    tags=[
        "positional-encoding",
        "attention-bias",
        "length-extrapolation",
        "simple",
    ],
    notes=(
        "Remarkable extrapolation: trained on 1024 tokens, works on 2048+. "
        "Used in BLOOM. Simpler than RoPE but similar extrapolation ability."
    ),
)


def sinusoidal_encoding(
    seq_len: int,
    d_model: int,
    base: float = 10000.0,
) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.

    PE(pos, 2i) = sin(pos / base^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / base^{2i/d_model})

    Args:
        seq_len: Maximum sequence length
        d_model: Embedding dimension (must be even)
        base: Base for exponential (default 10000)

    Returns:
        Positional encoding matrix (seq_len, d_model)
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")

    # Position indices: (seq_len, 1)
    positions = np.arange(seq_len)[:, np.newaxis]

    # Dimension indices: (1, d_model/2)
    dim_indices = np.arange(0, d_model, 2)[np.newaxis, :]

    # Compute angles: pos / base^{2i/d_model}
    angles = positions / np.power(base, dim_indices / d_model)

    # Interleave sin and cos
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)  # Even indices
    pe[:, 1::2] = np.cos(angles)  # Odd indices

    return pe


def learned_positional_embedding(
    seq_len: int,
    d_model: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Initialize learned positional embedding matrix.

    Args:
        seq_len: Maximum sequence length
        d_model: Embedding dimension
        seed: Random seed

    Returns:
        Position embedding matrix (seq_len, d_model)
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize with small random values
    scale = np.sqrt(1.0 / d_model)
    return np.random.randn(seq_len, d_model) * scale


def rotary_position_embedding(
    x: np.ndarray,
    position: int,
    base: float = 10000.0,
) -> np.ndarray:
    """
    Apply rotary position embedding to a vector.

    Rotates each pair of dimensions by position-dependent angle.

    Args:
        x: Input vector (d_model,) where d_model is even
        position: Position index
        base: Base for frequency computation

    Returns:
        Rotated vector (d_model,)
    """
    d_model = x.shape[0]
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")

    # Compute rotation angles for each dimension pair
    dim_pairs = d_model // 2
    dim_indices = np.arange(dim_pairs)
    thetas = position / np.power(base, 2 * dim_indices / d_model)

    # Apply rotation to each 2D subspace
    x_rotated = np.zeros_like(x)
    for i in range(dim_pairs):
        cos_theta = np.cos(thetas[i])
        sin_theta = np.sin(thetas[i])

        # Rotate pair (x[2i], x[2i+1])
        x_rotated[2*i] = x[2*i] * cos_theta - x[2*i + 1] * sin_theta
        x_rotated[2*i + 1] = x[2*i] * sin_theta + x[2*i + 1] * cos_theta

    return x_rotated


def rope_encode_qk(
    q: np.ndarray,
    k: np.ndarray,
    positions_q: np.ndarray,
    positions_k: np.ndarray,
    base: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply RoPE to query and key matrices.

    Args:
        q: Query matrix (seq_len_q, d_k)
        k: Key matrix (seq_len_k, d_k)
        positions_q: Position indices for queries (seq_len_q,)
        positions_k: Position indices for keys (seq_len_k,)
        base: Base for frequency computation

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    seq_len_q, d_k = q.shape
    seq_len_k = k.shape[0]

    q_rotated = np.zeros_like(q)
    k_rotated = np.zeros_like(k)

    for i in range(seq_len_q):
        q_rotated[i] = rotary_position_embedding(q[i], positions_q[i], base)

    for i in range(seq_len_k):
        k_rotated[i] = rotary_position_embedding(k[i], positions_k[i], base)

    return q_rotated, k_rotated


def alibi_bias(
    seq_len: int,
    num_heads: int,
) -> np.ndarray:
    """
    Compute ALiBi attention bias matrix.

    bias[h, i, j] = -m_h * (i - j)  for j <= i (causal)
                  = -inf            for j > i

    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads

    Returns:
        Bias tensor (num_heads, seq_len, seq_len)
    """
    # Compute slopes: geometric sequence starting from 2^{-8/num_heads}
    # For 8 heads: 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256
    slopes = 2.0 ** (-8.0 / num_heads * np.arange(1, num_heads + 1))

    # Position differences: i - j
    positions = np.arange(seq_len)
    # diff[i, j] = i - j
    diff = positions[:, np.newaxis] - positions[np.newaxis, :]

    # Compute biases for each head
    # bias[h, i, j] = -slopes[h] * max(0, i - j)
    biases = np.zeros((num_heads, seq_len, seq_len))

    for h in range(num_heads):
        biases[h] = -slopes[h] * np.maximum(0, diff)

    # Apply causal mask (set future positions to -inf)
    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    biases = np.where(causal_mask == 1, -1e9, biases)

    return biases


def alibi_slopes(num_heads: int) -> np.ndarray:
    """
    Compute ALiBi slopes for given number of heads.

    m_h = 2^{-8/num_heads * h} for h = 1, ..., num_heads

    Args:
        num_heads: Number of attention heads

    Returns:
        Slopes array (num_heads,)
    """
    return 2.0 ** (-8.0 / num_heads * np.arange(1, num_heads + 1))
