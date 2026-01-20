"""
Simple RNN / Elman Networks (1990)

Jeffrey Elman's foundational work on recurrent neural networks for
discovering temporal structure in sequences.

Paper: "Finding Structure in Time" (1990)
Author: Jeffrey L. Elman
Journal: Cognitive Science

Key Innovation:
    Introduced the concept of hidden state that persists across time steps,
    enabling networks to learn temporal dependencies. The "context units"
    copy the hidden state from the previous time step, allowing the network
    to maintain a form of memory.

Limitations:
    - Vanishing gradient problem: gradients diminish exponentially over time
    - Exploding gradient problem: gradients can grow unboundedly
    - Limited ability to capture long-range dependencies
    - These limitations motivated the development of LSTM and GRU

This is a research index entry, not a runnable implementation.
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# Method Registry Entry
SIMPLE_RNN = MLMethod(
    method_id="simple_rnn_elman_1990",
    name="Simple RNN / Elman Network",
    year=1990,
    era=MethodEra.CLASSICAL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.RNN_LINE],
    authors=["Jeffrey L. Elman"],
    paper_title="Finding Structure in Time",
    paper_url="https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1",
    key_innovation=(
        "Introduced context units that copy hidden state from previous time step, "
        "enabling temporal sequence processing. Demonstrated that RNNs could discover "
        "structure in time without explicit temporal markers."
    ),
    mathematical_formulation="""
    Hidden state update:
        h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)

    Output:
        y_t = W_hy * h_t + b_y

    Where:
        x_t = input at time t
        h_t = hidden state at time t
        y_t = output at time t
        W_hh = hidden-to-hidden weight matrix
        W_xh = input-to-hidden weight matrix
        W_hy = hidden-to-output weight matrix
        b_h, b_y = bias vectors
    """,
    predecessors=["hopfield_1982", "jordan_1986"],
    successors=["lstm_1997", "bptt_1990"],
    tags=["recurrent", "sequence-modeling", "temporal", "context-units", "connectionist"],
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Simple RNN."""
    return SIMPLE_RNN


def pseudocode() -> str:
    """Return pseudocode describing Simple RNN operation."""
    return '''
SIMPLE RNN FORWARD PASS:
========================

function simple_rnn_forward(X, W_xh, W_hh, W_hy, b_h, b_y):
    # Process a sequence through a simple RNN.
    #
    # Args:
    #     X: Input sequence of shape (T, input_dim)
    #     W_xh: Input-to-hidden weights
    #     W_hh: Hidden-to-hidden weights
    #     W_hy: Hidden-to-output weights
    #     b_h: Hidden bias
    #     b_y: Output bias
    #
    # Returns:
    #     Y: Output sequence
    #     H: Hidden state sequence

    T = length(X)
    h_prev = zeros(hidden_dim)  # Initialize hidden state

    H = []  # Store hidden states
    Y = []  # Store outputs

    for t in range(T):
        # Get input at time t
        x_t = X[t]

        # Compute hidden state (context units receive h_{t-1})
        h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b_h)

        # Compute output
        y_t = W_hy @ h_t + b_y

        # Store states
        H.append(h_t)
        Y.append(y_t)

        # Update for next step
        h_prev = h_t

    return Y, H


BACKPROPAGATION THROUGH TIME (BPTT):
====================================

function bptt(X, Y_true, parameters):
    # Compute gradients via backpropagation through time.
    #
    # The vanishing gradient problem:
    # - Gradients flow back through tanh activations
    # - Each step multiplies by W_hh and tanh derivative
    # - For long sequences: gradient ~ (W_hh)^T * (tanh_deriv)^T
    # - If |eigenvalues of W_hh| < 1: gradient vanishes
    # - If |eigenvalues of W_hh| > 1: gradient explodes

    # Forward pass
    Y_pred, H = simple_rnn_forward(X, ...)

    # Backward pass
    dL_dW_hh = 0
    dL_dW_xh = 0
    dL_dW_hy = 0

    dh_next = 0

    for t in reversed(range(T)):
        # Output gradient
        dL_dy = loss_gradient(Y_pred[t], Y_true[t])

        # Hidden state gradient
        dL_dh = W_hy.T @ dL_dy + dh_next

        # Through tanh nonlinearity
        dL_dh_raw = dL_dh * (1 - h[t]^2)  # tanh derivative

        # Accumulate weight gradients
        dL_dW_hh += outer(dL_dh_raw, h[t-1])
        dL_dW_xh += outer(dL_dh_raw, x[t])
        dL_dW_hy += outer(dL_dy, h[t])

        # Gradient to pass to previous timestep
        dh_next = W_hh.T @ dL_dh_raw

    return gradients
'''


def key_equations() -> dict:
    """Return key equations in LaTeX-style notation."""
    return {
        "hidden_state": r"h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)",
        "output": r"y_t = W_{hy} h_t + b_y",
        "loss": r"L = \sum_{t=1}^{T} \ell(y_t, \hat{y}_t)",
        "gradient_flow": r"\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} W_{hh}^T \text{diag}(\tanh'(z_i))",
        "vanishing_condition": r"\text{If } \|W_{hh}\| < 1 \text{ and } \|\tanh'\| \leq 1, \text{ gradients vanish exponentially}",
    }


def get_historical_context() -> str:
    """Return historical context and significance."""
    return """
    Historical Context:

    Jeffrey Elman's 1990 paper "Finding Structure in Time" was groundbreaking in
    demonstrating that neural networks could discover temporal structure in
    sequences without explicit temporal markers.

    Key insights:
    1. Context units (copying hidden state) enable implicit temporal representation
    2. The network learns to predict the next element in a sequence
    3. Internal representations capture abstract grammatical categories

    Elman showed his networks could learn:
    - Grammatical structure from word sequences
    - Number agreement in sentences
    - Long-distance dependencies (within limits)

    The discovery of the vanishing gradient problem (Hochreiter 1991, Bengio 1994)
    in these networks directly motivated the development of LSTM, which solved
    the long-term dependency problem through gating mechanisms.

    Legacy:
    - Foundation for all modern RNN architectures
    - Inspired sequence-to-sequence models
    - Demonstrated connectionist approach to language
    """


def get_limitations() -> list:
    """Return known limitations of Simple RNN."""
    return [
        "Vanishing gradient problem prevents learning long-range dependencies",
        "Exploding gradient problem can cause training instability",
        "Difficulty capturing dependencies beyond ~10-20 timesteps",
        "Sensitive to hyperparameter choices",
        "No mechanism for selective forgetting or gating",
        "Sequential processing limits parallelization",
    ]


def get_applications() -> list:
    """Return historical and modern applications."""
    return [
        "Language modeling (historical)",
        "Part-of-speech tagging (historical)",
        "Simple sequence prediction tasks",
        "Time series with short-term dependencies",
        "Educational demonstrations of RNN concepts",
        "Baseline comparisons for more advanced architectures",
    ]
