"""
Long Short-Term Memory (LSTM) Networks (1997)

Hochreiter & Schmidhuber's breakthrough architecture that solved the
vanishing gradient problem through gating mechanisms and cell state.

Paper: "Long Short-Term Memory" (1997)
Authors: Sepp Hochreiter, Jurgen Schmidhuber
Journal: Neural Computation

Key Innovation:
    Introduced the cell state (memory cell) and three gating mechanisms:
    - Forget gate: Controls what information to discard from cell state
    - Input gate: Controls what new information to store in cell state
    - Output gate: Controls what information to output from cell state

    The constant error carousel (CEC) allows gradients to flow unchanged
    through the cell state, enabling learning of long-range dependencies.

This is a research index entry, not a runnable implementation.
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# Method Registry Entry
LSTM = MLMethod(
    method_id="lstm_1997",
    name="Long Short-Term Memory (LSTM)",
    year=1997,
    era=MethodEra.CLASSICAL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.RNN_LINE],
    authors=["Sepp Hochreiter", "Jurgen Schmidhuber"],
    paper_title="Long Short-Term Memory",
    paper_url="https://www.bioinf.jku.at/publications/older/2604.pdf",
    key_innovation=(
        "Introduced gating mechanisms (forget, input, output gates) and a cell state "
        "with constant error carousel (CEC) to solve the vanishing gradient problem. "
        "Gates learn when to remember, forget, and output information."
    ),
    mathematical_formulation="""
    LSTM Equations:

    Forget gate (what to forget from cell state):
        f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)

    Input gate (what new information to add):
        i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)

    Candidate cell state:
        C_tilde_t = tanh(W_C * [h_{t-1}, x_t] + b_C)

    Cell state update:
        C_t = f_t * C_{t-1} + i_t * C_tilde_t

    Output gate (what to output):
        o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)

    Hidden state (output):
        h_t = o_t * tanh(C_t)

    Where:
        sigma = sigmoid function
        * = element-wise multiplication (Hadamard product)
        [h_{t-1}, x_t] = concatenation of previous hidden state and current input
    """,
    predecessors=["simple_rnn_elman_1990", "jordan_1986"],
    successors=["gru_2014", "bidirectional_lstm", "stacked_lstm"],
    tags=["recurrent", "gating", "memory", "long-term-dependencies", "cell-state", "gradient-flow"],
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for LSTM."""
    return LSTM


def pseudocode() -> str:
    """Return pseudocode describing LSTM operation."""
    return '''
LSTM FORWARD PASS:
==================

function lstm_forward(X, weights, biases):
    # Process a sequence through an LSTM cell.
    #
    # Args:
    #     X: Input sequence of shape (T, input_dim)
    #     weights: Dictionary containing W_f, W_i, W_C, W_o
    #     biases: Dictionary containing b_f, b_i, b_C, b_o
    #
    # Returns:
    #     H: Hidden state sequence
    #     C: Cell state sequence

    T = length(X)
    h_prev = zeros(hidden_dim)  # Previous hidden state
    c_prev = zeros(hidden_dim)  # Previous cell state

    H = []  # Store hidden states
    C = []  # Store cell states

    for t in range(T):
        x_t = X[t]

        # Concatenate previous hidden state and current input
        combined = concatenate([h_prev, x_t])

        # === FORGET GATE ===
        # Decides what information to discard from cell state
        # f_t close to 0 = forget, close to 1 = remember
        f_t = sigmoid(W_f @ combined + b_f)

        # === INPUT GATE ===
        # Decides what new information to store
        i_t = sigmoid(W_i @ combined + b_i)

        # === CANDIDATE CELL STATE ===
        # New candidate values that could be added to cell state
        C_tilde_t = tanh(W_C @ combined + b_C)

        # === CELL STATE UPDATE ===
        # Core of LSTM: linear combination allows gradient flow
        # This is the "Constant Error Carousel" (CEC)
        C_t = f_t * c_prev + i_t * C_tilde_t

        # === OUTPUT GATE ===
        # Decides what parts of cell state to output
        o_t = sigmoid(W_o @ combined + b_o)

        # === HIDDEN STATE ===
        # Filtered cell state becomes the output
        h_t = o_t * tanh(C_t)

        # Store states
        H.append(h_t)
        C.append(C_t)

        # Update for next step
        h_prev = h_t
        c_prev = C_t

    return H, C


LSTM BACKWARD PASS (BPTT):
==========================

function lstm_backward(dH, cache):
    # Backpropagation through LSTM.
    #
    # Key insight: Gradient flows through cell state with only
    # element-wise operations (no matrix multiplication), preventing
    # vanishing gradients.
    #
    # Cell state gradient path:
    #     dC_t/dC_{t-1} = f_t  (just multiplication by forget gate)
    #
    # If forget gate ~ 1, gradients flow unchanged!

    dC_next = 0
    dh_next = 0

    for t in reversed(range(T)):
        # Gradient from output and next timestep
        dh = dH[t] + dh_next

        # Output gate gradient
        do = dh * tanh(C[t])
        do_raw = do * sigmoid_derivative(o[t])

        # Cell state gradient (key: linear path!)
        dC = dh * o[t] * tanh_derivative(C[t]) + dC_next

        # Forget gate gradient
        df = dC * C[t-1]
        df_raw = df * sigmoid_derivative(f[t])

        # Input gate gradient
        di = dC * C_tilde[t]
        di_raw = di * sigmoid_derivative(i[t])

        # Candidate gradient
        dC_tilde = dC * i[t]
        dC_tilde_raw = dC_tilde * tanh_derivative(C_tilde[t])

        # Accumulate weight gradients
        # ... (standard gradient accumulation)

        # Gradients to pass to previous timestep
        dC_next = dC * f[t]  # Linear! No vanishing!
        dh_next = (W_f.T @ df_raw + W_i.T @ di_raw +
                   W_C.T @ dC_tilde_raw + W_o.T @ do_raw)[:hidden_dim]

    return gradients


LSTM WITH PEEPHOLE CONNECTIONS:
===============================

# Peephole connections allow gates to look at cell state directly
# Added by Gers & Schmidhuber (2000)

f_t = sigmoid(W_f @ [h_{t-1}, x_t] + W_cf * C_{t-1} + b_f)  # Note: C_{t-1}
i_t = sigmoid(W_i @ [h_{t-1}, x_t] + W_ci * C_{t-1} + b_i)
C_t = f_t * C_{t-1} + i_t * tanh(W_C @ [h_{t-1}, x_t] + b_C)
o_t = sigmoid(W_o @ [h_{t-1}, x_t] + W_co * C_t + b_o)      # Note: C_t
h_t = o_t * tanh(C_t)
'''


def key_equations() -> dict:
    """
    Return key LSTM equations in LaTeX-style notation.

    These are the standard LSTM equations as formulated by
    Hochreiter & Schmidhuber (1997) with the forget gate
    addition by Gers et al. (2000).
    """
    return {
        # Gate equations
        "forget_gate": r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)",
        "input_gate": r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)",
        "candidate_cell": r"\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)",
        "cell_state": r"C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t",
        "output_gate": r"o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)",
        "hidden_state": r"h_t = o_t \odot \tanh(C_t)",

        # Gradient flow (why LSTM works)
        "cell_gradient": r"\frac{\partial C_t}{\partial C_{t-1}} = f_t",
        "gradient_preservation": r"\frac{\partial C_T}{\partial C_1} = \prod_{t=2}^{T} f_t \approx 1 \text{ if } f_t \approx 1",
    }


def get_historical_context() -> str:
    """Return historical context and significance."""
    return """
    Historical Context:

    LSTM was developed in 1997 by Sepp Hochreiter (in his PhD thesis, supervised
    by Jurgen Schmidhuber) to address the fundamental vanishing gradient problem
    that plagued simple RNNs.

    Key development timeline:
    - 1991: Hochreiter identifies vanishing gradient problem in his diploma thesis
    - 1994: Bengio et al. publish detailed analysis of vanishing gradients
    - 1997: Hochreiter & Schmidhuber publish LSTM paper
    - 2000: Gers et al. add forget gate (crucial improvement!)
    - 2000: Gers & Schmidhuber add peephole connections
    - 2014-2017: LSTM dominates sequence modeling (speech, NLP, translation)
    - 2017: Transformers begin replacing LSTM for many tasks

    The "Constant Error Carousel" (CEC):
    The cell state C_t is updated through element-wise operations only:
        C_t = f_t * C_{t-1} + i_t * C_tilde_t

    This means gradients can flow back through time with only multiplication
    by the forget gate values. If forget gates stay close to 1, gradients
    neither vanish nor explode!

    Impact:
    - Enabled learning dependencies over 1000+ timesteps
    - Foundation for speech recognition breakthroughs (Google, Baidu)
    - Neural machine translation (Google Translate, 2016)
    - Language modeling advances leading to modern NLP
    """


def get_limitations() -> list:
    """Return known limitations of LSTM."""
    return [
        "Sequential processing prevents parallelization",
        "High computational cost (4x parameters of simple RNN)",
        "Saturating gates can still cause gradient issues",
        "Complex architecture with many hyperparameters",
        "Slower to train than feedforward networks",
        "Largely superseded by Transformers for many tasks (post-2017)",
        "Limited context window compared to attention mechanisms",
    ]


def get_applications() -> list:
    """Return historical and modern applications."""
    return [
        "Speech recognition (Google Voice, Apple Siri)",
        "Neural machine translation (Google Translate 2016)",
        "Language modeling",
        "Handwriting recognition and generation",
        "Music generation (Magenta)",
        "Time series forecasting",
        "Video captioning",
        "Sentiment analysis",
        "Named entity recognition",
        "Protein structure prediction (historical)",
    ]


def get_variants() -> dict:
    """Return common LSTM variants and modifications."""
    return {
        "Vanilla LSTM": "Original 1997 formulation without forget gate",
        "LSTM with Forget Gate": "Standard modern LSTM (Gers et al., 2000)",
        "Peephole LSTM": "Gates can observe cell state directly",
        "Coupled Input-Forget Gate": "Use (1-f_t) instead of separate i_t",
        "Bidirectional LSTM": "Process sequence in both directions",
        "Stacked LSTM": "Multiple LSTM layers for hierarchical features",
        "Attention LSTM": "LSTM with attention mechanism (pre-Transformer)",
    }
