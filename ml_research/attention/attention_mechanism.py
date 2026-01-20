"""
Bahdanau Attention (2014)

Authors: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
Paper: "Neural Machine Translation by Jointly Learning to Align and Translate"
       arXiv:1409.0473, September 2014

The first attention mechanism for neural machine translation, allowing the decoder
to selectively focus on different parts of the input sequence when generating each
output token. Also known as "additive attention" due to its scoring function.

Key Innovation:
    - Broke the information bottleneck of fixed-length context vectors
    - Introduced alignment model that learns soft alignments between source and target
    - Enabled encoder-decoder models to handle long sequences effectively
    - Foundation for all subsequent attention mechanisms

Mathematical Formulation:
    Attention Score (Additive/Bahdanau):
        e_ti = v^T * tanh(W_1 * s_{t-1} + W_2 * h_i)

    Where:
        s_{t-1} = decoder hidden state at previous time step
        h_i = encoder hidden state at position i
        W_1, W_2 = learnable weight matrices
        v = learnable weight vector

    Attention Weights (via softmax):
        alpha_ti = exp(e_ti) / sum_j(exp(e_tj))

    Context Vector:
        c_t = sum_i(alpha_ti * h_i)

    The context vector c_t is then used along with s_{t-1} and y_{t-1}
    to compute the next decoder state s_t and output y_t.

Architecture:
    - Bidirectional RNN encoder produces h_1, ..., h_T
    - Each h_i concatenates forward and backward states: [h_i_fwd; h_i_bwd]
    - Decoder RNN uses attention to select relevant encoder states
    - At each step, decoder "attends" to all encoder states with learned weights

Pseudocode:
    ```
    def bahdanau_attention(decoder_state, encoder_outputs):
        # decoder_state: (batch, hidden_dim)
        # encoder_outputs: (batch, seq_len, hidden_dim)

        # Compute alignment scores
        scores = []
        for i in range(seq_len):
            # Additive attention scoring
            h_i = encoder_outputs[:, i, :]
            score = v @ tanh(W1 @ decoder_state + W2 @ h_i)
            scores.append(score)

        # Normalize with softmax
        attention_weights = softmax(scores)  # (batch, seq_len)

        # Compute context vector as weighted sum
        context = sum(alpha_i * h_i for alpha_i, h_i in zip(attention_weights, encoder_outputs))

        return context, attention_weights

    def decode_step(prev_state, prev_output, encoder_outputs):
        # Compute attention
        context, weights = bahdanau_attention(prev_state, encoder_outputs)

        # Combine context with previous output
        rnn_input = concatenate([prev_output, context])

        # Update decoder state
        new_state = gru_cell(rnn_input, prev_state)

        # Generate output
        output = softmax(W_out @ concatenate([new_state, context]))

        return new_state, output, weights
    ```

Historical Significance:
    - First successful attention mechanism in deep learning
    - Enabled neural machine translation to match/exceed phrase-based SMT
    - Attention weights provide interpretable alignments
    - Directly inspired Transformer's attention (though Transformer uses dot-product)

Comparison with Later Attention:
    - Bahdanau (additive): score = v^T tanh(W_1 s + W_2 h)
    - Luong dot-product: score = s^T h
    - Luong general: score = s^T W h
    - Scaled dot-product: score = (Q K^T) / sqrt(d_k)

Limitations:
    - Still sequential due to RNN backbone
    - Attention computation is O(n) for each decoder step
    - Cannot parallelize across time steps
"""

from typing import List, Tuple, Optional
import numpy as np

from consciousness.ml_research.core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# Research index entry
BAHDANAU_ATTENTION = MLMethod(
    method_id="bahdanau_attention_2014",
    name="Bahdanau Attention",
    year=2014,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ATTENTION,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.RNN_LINE],
    authors=["Dzmitry Bahdanau", "Kyunghyun Cho", "Yoshua Bengio"],
    paper_title="Neural Machine Translation by Jointly Learning to Align and Translate",
    paper_url="https://arxiv.org/abs/1409.0473",
    key_innovation=(
        "First attention mechanism for sequence-to-sequence models. Introduced soft "
        "alignment allowing decoder to focus on relevant encoder states, breaking the "
        "information bottleneck of fixed-length context vectors."
    ),
    mathematical_formulation="""
    Score (additive): e_ti = v^T * tanh(W_1 * s_{t-1} + W_2 * h_i)

    Weights: alpha_ti = softmax(e_ti) = exp(e_ti) / sum_j(exp(e_tj))

    Context: c_t = sum_i(alpha_ti * h_i)
    """,
    predecessors=["seq2seq_2014", "gru_2014"],
    successors=["luong_attention_2015", "self_attention_2017", "transformer_2017"],
    tags=[
        "attention",
        "sequence-to-sequence",
        "neural-machine-translation",
        "alignment",
        "additive-attention",
        "encoder-decoder",
    ],
)


def compute_attention_weights(
    decoder_state: np.ndarray,
    encoder_outputs: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Compute Bahdanau attention weights.

    score_i = v^T * tanh(W1 @ decoder_state + W2 @ encoder_output_i)
    weights = softmax(scores)

    Args:
        decoder_state: Decoder hidden state (hidden_dim,)
        encoder_outputs: All encoder hidden states (seq_len, hidden_dim)
        W1: Weight matrix for decoder state (attention_dim, hidden_dim)
        W2: Weight matrix for encoder outputs (attention_dim, hidden_dim)
        v: Weight vector for final score (attention_dim,)

    Returns:
        Attention weights (seq_len,) summing to 1
    """
    seq_len = encoder_outputs.shape[0]
    scores = np.zeros(seq_len)

    # Project decoder state (same for all positions)
    proj_decoder = W1 @ decoder_state  # (attention_dim,)

    for i in range(seq_len):
        # Project encoder output at position i
        proj_encoder = W2 @ encoder_outputs[i]  # (attention_dim,)

        # Additive attention score
        scores[i] = v @ np.tanh(proj_decoder + proj_encoder)

    # Softmax normalization
    scores_exp = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
    weights = scores_exp / np.sum(scores_exp)

    return weights


def compute_context_vector(
    encoder_outputs: np.ndarray,
    attention_weights: np.ndarray,
) -> np.ndarray:
    """
    Compute context vector as weighted sum of encoder outputs.

    c_t = sum_i(alpha_ti * h_i)

    Args:
        encoder_outputs: All encoder hidden states (seq_len, hidden_dim)
        attention_weights: Attention weights (seq_len,)

    Returns:
        Context vector (hidden_dim,)
    """
    # Weighted sum: (seq_len, hidden_dim).T @ (seq_len,) -> (hidden_dim,)
    context = encoder_outputs.T @ attention_weights
    return context


def bahdanau_attention(
    decoder_state: np.ndarray,
    encoder_outputs: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full Bahdanau attention computation.

    Given decoder state and encoder outputs, compute attention weights
    and context vector.

    Args:
        decoder_state: Decoder hidden state (hidden_dim,)
        encoder_outputs: All encoder hidden states (seq_len, hidden_dim)
        W1: Weight matrix for decoder state (attention_dim, hidden_dim)
        W2: Weight matrix for encoder outputs (attention_dim, hidden_dim)
        v: Weight vector for final score (attention_dim,)

    Returns:
        Tuple of (context_vector, attention_weights)
        - context_vector: (hidden_dim,)
        - attention_weights: (seq_len,)
    """
    weights = compute_attention_weights(decoder_state, encoder_outputs, W1, W2, v)
    context = compute_context_vector(encoder_outputs, weights)
    return context, weights


def initialize_attention_parameters(
    hidden_dim: int,
    attention_dim: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize Bahdanau attention parameters.

    Uses Xavier/Glorot initialization for weight matrices.

    Args:
        hidden_dim: Dimension of encoder/decoder hidden states
        attention_dim: Dimension of attention hidden layer
        seed: Random seed for reproducibility

    Returns:
        Tuple of (W1, W2, v) initialized parameters
    """
    if seed is not None:
        np.random.seed(seed)

    # Xavier initialization
    scale_W = np.sqrt(2.0 / (hidden_dim + attention_dim))
    scale_v = np.sqrt(2.0 / attention_dim)

    W1 = np.random.randn(attention_dim, hidden_dim) * scale_W
    W2 = np.random.randn(attention_dim, hidden_dim) * scale_W
    v = np.random.randn(attention_dim) * scale_v

    return W1, W2, v
