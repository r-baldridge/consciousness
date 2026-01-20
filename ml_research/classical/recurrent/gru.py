"""
Gated Recurrent Unit (GRU) Networks (2014)

Cho et al.'s simplified gating mechanism that achieves similar performance
to LSTM with fewer parameters and faster training.

Paper: "Learning Phrase Representations using RNN Encoder-Decoder
        for Statistical Machine Translation" (2014)
Authors: Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre,
         Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
Venue: EMNLP 2014

Key Innovation:
    Simplified gating architecture with only two gates:
    - Reset gate: Controls how much of previous hidden state to forget
    - Update gate: Controls how much of new state vs old state to use

    No separate cell state - the hidden state serves both purposes.
    Fewer parameters than LSTM while achieving comparable performance.

This is a research index entry, not a runnable implementation.
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# Method Registry Entry
GRU = MLMethod(
    method_id="gru_2014",
    name="Gated Recurrent Unit (GRU)",
    year=2014,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.RNN_LINE],
    authors=[
        "Kyunghyun Cho",
        "Bart van Merrienboer",
        "Caglar Gulcehre",
        "Dzmitry Bahdanau",
        "Fethi Bougares",
        "Holger Schwenk",
        "Yoshua Bengio",
    ],
    paper_title="Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
    paper_url="https://arxiv.org/abs/1406.1078",
    key_innovation=(
        "Simplified gating architecture with only reset and update gates, "
        "eliminating the separate cell state of LSTM. Achieves similar "
        "performance with fewer parameters and faster training."
    ),
    mathematical_formulation="""
    GRU Equations:

    Update gate (controls state mixing):
        z_t = sigma(W_z * [h_{t-1}, x_t])

    Reset gate (controls memory access):
        r_t = sigma(W_r * [h_{t-1}, x_t])

    Candidate hidden state:
        h_tilde_t = tanh(W * [r_t * h_{t-1}, x_t])

    Hidden state update (interpolation):
        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t

    Where:
        sigma = sigmoid function
        * = element-wise multiplication
        [a, b] = concatenation

    Note: The update gate z_t controls how much of the new candidate
    state to use. When z_t = 0, the hidden state is just copied
    (like LSTM forget gate = 1). When z_t = 1, completely new state.
    """,
    predecessors=["lstm_1997", "simple_rnn_elman_1990"],
    successors=["minimal_gru", "transformer_2017"],
    tags=["recurrent", "gating", "simplified", "encoder-decoder", "machine-translation"],
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for GRU."""
    return GRU


def pseudocode() -> str:
    """Return pseudocode describing GRU operation."""
    return '''
GRU FORWARD PASS:
=================

function gru_forward(X, W_z, W_r, W, b_z, b_r, b):
    # Process a sequence through a GRU cell.
    #
    # Args:
    #     X: Input sequence of shape (T, input_dim)
    #     W_z: Update gate weights
    #     W_r: Reset gate weights
    #     W: Candidate state weights
    #     b_z, b_r, b: Biases
    #
    # Returns:
    #     H: Hidden state sequence

    T = length(X)
    h_prev = zeros(hidden_dim)  # Initialize hidden state

    H = []  # Store hidden states

    for t in range(T):
        x_t = X[t]

        # Concatenate previous hidden state and current input
        combined = concatenate([h_prev, x_t])

        # === UPDATE GATE ===
        # Controls how much of new state to use
        # z = 0: copy old state, z = 1: use new state entirely
        z_t = sigmoid(W_z @ combined + b_z)

        # === RESET GATE ===
        # Controls how much of previous hidden state to expose
        # r = 0: ignore previous state, r = 1: use all of it
        r_t = sigmoid(W_r @ combined + b_r)

        # === CANDIDATE HIDDEN STATE ===
        # New candidate, with reset gate applied to h_{t-1}
        # Reset gate allows ignoring irrelevant past information
        h_reset = r_t * h_prev
        combined_reset = concatenate([h_reset, x_t])
        h_tilde_t = tanh(W @ combined_reset + b)

        # === HIDDEN STATE UPDATE ===
        # Interpolation between old and new state
        # This is equivalent to LSTM cell state update when:
        #   - z_t = i_t (input gate)
        #   - (1 - z_t) = f_t (forget gate)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde_t

        # Store state
        H.append(h_t)

        # Update for next step
        h_prev = h_t

    return H


GRU VS LSTM COMPARISON:
=======================

LSTM has 4 components:
    - Forget gate: f_t
    - Input gate: i_t
    - Cell state: C_t
    - Output gate: o_t

GRU has 2 components:
    - Reset gate: r_t (similar to forget gate, but applied differently)
    - Update gate: z_t (combines input and forget gate functions)

Key simplifications:
1. No separate cell state (h_t serves both roles)
2. Update gate z_t controls both forgetting and updating
3. Reset gate r_t applied to hidden state before computing candidate
4. No output gate (hidden state is directly the output)

Parameter count comparison (hidden_dim = h, input_dim = x):
    LSTM: 4 * (h * (h + x) + h) = 4h^2 + 4hx + 4h
    GRU:  3 * (h * (h + x) + h) = 3h^2 + 3hx + 3h

    GRU has ~25% fewer parameters!


GRU BACKWARD PASS:
==================

function gru_backward(dH, cache):
    # Backpropagation through GRU.
    #
    # Gradient flow through update gate interpolation:
    #     h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
    #
    #     dh_{t-1} = dh_t * (1 - z_t)  (linear path when z_t small!)
    #
    # Similar to LSTM, when update gate is small (z_t ~ 0),
    # gradients flow through unchanged.

    dh_next = 0

    for t in reversed(range(T)):
        dh = dH[t] + dh_next

        # Gradient of interpolation
        dh_prev_direct = dh * (1 - z[t])
        dh_tilde = dh * z[t]
        dz = dh * (h_tilde[t] - h[t-1])

        # Through candidate computation
        dh_tilde_raw = dh_tilde * tanh_derivative(h_tilde[t])

        # Through reset gate
        dh_reset = W.T @ dh_tilde_raw  # gradient to reset hidden
        dr = dh_reset * h[t-1]
        dh_prev_reset = dh_reset * r[t]

        # Through gates
        dz_raw = dz * sigmoid_derivative(z[t])
        dr_raw = dr * sigmoid_derivative(r[t])

        # Accumulate weight gradients
        # ...

        # Total gradient to previous timestep
        dh_next = dh_prev_direct + dh_prev_reset + gate_gradients

    return gradients
'''


def key_equations() -> dict:
    """
    Return key GRU equations in LaTeX-style notation.

    These are the standard GRU equations as formulated by
    Cho et al. (2014).
    """
    return {
        # Gate equations
        "update_gate": r"z_t = \sigma(W_z \cdot [h_{t-1}, x_t])",
        "reset_gate": r"r_t = \sigma(W_r \cdot [h_{t-1}, x_t])",
        "candidate_hidden": r"\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])",
        "hidden_state": r"h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t",

        # Gradient flow
        "gradient_flow": r"\frac{\partial h_t}{\partial h_{t-1}} = (1 - z_t) + z_t \cdot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}",
        "gradient_preservation": r"\text{When } z_t \approx 0: \frac{\partial h_t}{\partial h_{t-1}} \approx I",
    }


def get_historical_context() -> str:
    """Return historical context and significance."""
    return """
    Historical Context:

    GRU was introduced in 2014 by the Montreal research group led by Yoshua Bengio,
    alongside several other foundational contributions to neural machine translation:

    - Encoder-Decoder architecture (Cho et al., 2014)
    - GRU cell (this paper)
    - Attention mechanism (Bahdanau et al., 2015, same group)

    The motivation was to create a simpler alternative to LSTM that:
    1. Had fewer parameters (easier to train, less overfitting)
    2. Was faster to compute
    3. Maintained similar modeling capacity

    Key empirical findings (Chung et al., 2014):
    - GRU and LSTM perform similarly across most tasks
    - GRU tends to be better on smaller datasets
    - LSTM may have slight edge on tasks requiring precise timing
    - Neither is universally better - task dependent

    The GRU's simplicity made it popular for:
    - Resource-constrained applications
    - Quick prototyping
    - When LSTM is overkill

    However, both GRU and LSTM were largely superseded by Transformers
    after 2017 for most NLP tasks, though they remain useful for:
    - Online/streaming processing
    - Time series with natural temporal ordering
    - Resource-constrained devices
    """


def get_limitations() -> list:
    """Return known limitations of GRU."""
    return [
        "Sequential processing prevents parallelization (like LSTM)",
        "May underperform LSTM on tasks requiring precise timing",
        "Less expressive than LSTM due to fewer parameters",
        "Still has quadratic complexity in hidden dimension",
        "Largely superseded by Transformers for NLP (post-2017)",
        "Reset gate can cause gradient issues if always near 0",
    ]


def get_applications() -> list:
    """Return historical and modern applications."""
    return [
        "Neural machine translation (early systems)",
        "Sequence-to-sequence models",
        "Speech recognition",
        "Language modeling",
        "Time series forecasting",
        "Video understanding",
        "Anomaly detection in sequences",
        "Music generation",
        "Edge devices (fewer parameters than LSTM)",
        "Online/streaming applications",
    ]


def compare_to_lstm() -> dict:
    """Return detailed comparison between GRU and LSTM."""
    return {
        "parameters": {
            "LSTM": "4 * hidden_dim * (hidden_dim + input_dim + 1)",
            "GRU": "3 * hidden_dim * (hidden_dim + input_dim + 1)",
            "ratio": "GRU has ~75% of LSTM parameters",
        },
        "components": {
            "LSTM": ["Forget gate", "Input gate", "Output gate", "Cell state"],
            "GRU": ["Reset gate", "Update gate"],
        },
        "memory": {
            "LSTM": "Separate cell state C_t for long-term memory",
            "GRU": "Single hidden state h_t serves both roles",
        },
        "gating": {
            "LSTM": "Independent forget and input gates",
            "GRU": "Coupled through update gate z_t and (1-z_t)",
        },
        "empirical": {
            "performance": "Generally similar",
            "training_speed": "GRU often faster",
            "best_for_GRU": "Smaller datasets, simpler patterns",
            "best_for_LSTM": "Complex timing, longer sequences",
        },
    }
