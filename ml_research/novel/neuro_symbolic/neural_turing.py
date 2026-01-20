"""
Neural Turing Machine (NTM) - 2014

Graves, Wayne, & Danihelka's memory-augmented neural network architecture.
The NTM extends neural networks with external memory that can be read from
and written to, inspired by the Turing machine's tape. This allows learning
algorithms from examples, such as copying, sorting, and recall.

Paper: "Neural Turing Machines"
arXiv: 1410.5401

Mathematical Formulation:
    Memory: M_t in R^{N x W} (N memory locations, W-dimensional)

    Read operation:
        r_t = sum_i w_t^r(i) * M_t(i)  (weighted read)

    Write operation:
        M_t = M_{t-1} * (1 - w_t^w * e_t) + w_t^w * a_t
        (erase then add)

    Addressing (content + location):
        w_t^c = softmax(beta_t * K(k_t, M_t))  (content addressing)
        w_t = g_t * w_t^c + (1 - g_t) * w_{t-1}  (interpolation)
        w_t = shift(w_t, s_t)  (location shift)
        w_t = sharpen(w_t, gamma_t)  (sharpening)

    Where:
        - k_t: Key vector for content addressing
        - beta_t: Key strength
        - g_t: Interpolation gate
        - s_t: Shift weights
        - gamma_t: Sharpening factor
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

NEURAL_TURING_MACHINE = MLMethod(
    method_id="ntm_2014",
    name="Neural Turing Machine",
    year=2014,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.RNN_LINE, MethodLineage.ATTENTION_LINE],

    authors=["Alex Graves", "Greg Wayne", "Ivo Danihelka"],
    paper_title="Neural Turing Machines",
    paper_url="https://arxiv.org/abs/1410.5401",

    key_innovation=(
        "Extended neural networks with differentiable external memory and "
        "attention-based addressing. This allows networks to learn algorithms "
        "(copying, sorting, recall) by reading from and writing to memory. "
        "Memory access is content-based (by similarity) and location-based "
        "(by shifting)."
    ),

    mathematical_formulation=r"""
Memory Bank:
    M_t in R^{N x W}
    N = number of memory locations
    W = width of each memory location

Controller:
    Neural network (LSTM/feedforward) that produces interface parameters

Read Head:
    Input: Read weights w_t^r in R^N (attention over locations)
    Output: r_t = sum_i w_t^r(i) * M_t(i)    (weighted sum)

Write Head:
    Input: Write weights w_t^w, erase vector e_t, add vector a_t
    Operation:
        M_t(i) = M_{t-1}(i) * [1 - w_t^w(i) * e_t] + w_t^w(i) * a_t
        (First erase old content, then add new content)

Addressing Mechanism (generating w_t):
    1. Content Addressing:
       w_t^c(i) = softmax(beta_t * K(k_t, M_t(i)))
       K(u, v) = u * v / (||u|| * ||v||)    (cosine similarity)

    2. Interpolation (blend content vs previous location):
       w_t^g = g_t * w_t^c + (1 - g_t) * w_{t-1}

    3. Convolutional Shift:
       w_t^s(i) = sum_j w_t^g(j) * s_t(i - j)
       s_t = softmax over shift amounts {-1, 0, +1}

    4. Sharpening:
       w_t(i) = w_t^s(i)^gamma_t / sum_j w_t^s(j)^gamma_t

Controller Interface (per head):
    k_t: Key vector (W dimensions)
    beta_t: Key strength (scalar > 0)
    g_t: Interpolation gate (scalar in [0, 1])
    s_t: Shift weighting (3 values: left, stay, right)
    gamma_t: Sharpening (scalar >= 1)
    e_t: Erase vector (W dimensions, write head only)
    a_t: Add vector (W dimensions, write head only)
""",

    predecessors=["lstm_1997", "attention_mechanism"],
    successors=["dnc_2016", "memory_networks"],

    tags=["memory-augmented", "differentiable-memory", "attention", "algorithmic-tasks"]
)

# Note: NTMs demonstrated that neural networks could learn basic algorithms
# (copy, reverse, sort, associative recall) from input-output examples.
# The hybrid content+location addressing enables both associative lookup
# and sequential access patterns. Succeeded by DNC (Differentiable Neural
# Computer) which added memory linkage and dynamic allocation.


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def memory_addressing() -> Dict[str, str]:
    """
    Memory addressing mechanisms in Neural Turing Machine.

    Returns:
        Dictionary describing addressing components
    """
    return {
        "content_addressing": """
Content-Based Addressing:
    Find memory locations similar to query key k_t

    Similarity: K(k_t, M_t(i)) = cosine_similarity(k_t, M_t(i))
                               = (k_t * M_t(i)) / (||k_t|| * ||M_t(i)||)

    Attention: w_t^c(i) = exp(beta_t * K(k_t, M_t(i))) / sum_j exp(beta_t * K(k_t, M_t(j)))

    beta_t controls sharpness:
        - beta_t -> 0: uniform attention over all locations
        - beta_t -> inf: hard attention on most similar location

    This is soft attention: gradient flows through all locations weighted by similarity.
""",
        "interpolation_gate": """
Interpolation:
    Blend content-based attention with previous location:

    w_t^g = g_t * w_t^c + (1 - g_t) * w_{t-1}

    g_t in [0, 1] controls:
        - g_t = 1: Use only content addressing
        - g_t = 0: Use only previous location (iteration)
        - 0 < g_t < 1: Combine both

    Enables: Look up by content OR continue from previous position
""",
        "convolutional_shift": """
Location-Based Shifting:
    Shift attention to nearby locations:

    w_t^s(i) = sum_{j=0}^{N-1} w_t^g(j) * s_t(i - j mod N)

    s_t is a shift distribution over {-1, 0, +1}:
        - s_t(-1) > 0: Shift left (decrease index)
        - s_t(0) > 0: Stay at current position
        - s_t(+1) > 0: Shift right (increase index)

    Enables: Sequential traversal, iteration over memory
""",
        "sharpening": """
Sharpening:
    Focus attention after shifting:

    w_t(i) = w_t^s(i)^gamma_t / sum_j w_t^s(j)^gamma_t

    gamma_t >= 1 controls:
        - gamma_t = 1: No change
        - gamma_t > 1: Sharpen (make peaked distribution more peaked)

    Prevents: Attention blurring after repeated shifts
"""
    }


def read_write_operations() -> Dict[str, str]:
    """
    Read and write operations in NTM.

    Returns:
        Dictionary describing read/write mechanisms
    """
    return {
        "read_operation": """
Read Operation:
    r_t = sum_{i=0}^{N-1} w_t^r(i) * M_t(i)

    - w_t^r: Read attention weights (computed via addressing)
    - M_t(i): i-th row of memory matrix
    - r_t: Weighted sum of memory contents

    Differentiable: Gradient flows through attention weights
    Output dimension: W (same as memory width)

    Multiple read heads possible: each produces independent r_t
""",
        "write_operation": """
Write Operation (Erase + Add):
    M_t(i) = M_{t-1}(i) * [1 - w_t^w(i) * e_t] + w_t^w(i) * a_t

    Step 1 - Erase:
        tilde{M}_t(i) = M_{t-1}(i) * [1 - w_t^w(i) * e_t]

        - e_t in [0, 1]^W: Erase vector
        - Where w_t^w(i) * e_t approaches 1, memory is erased

    Step 2 - Add:
        M_t(i) = tilde{M}_t(i) + w_t^w(i) * a_t

        - a_t in R^W: Add vector
        - New information written where attention is high

    Combined: Content can be selectively erased and replaced
""",
        "controller_outputs": """
Controller Outputs (per head):
    Read head:
        - k_t in R^W: Key for content addressing
        - beta_t > 0: Key strength
        - g_t in [0, 1]: Interpolation gate
        - s_t: Shift distribution
        - gamma_t >= 1: Sharpening factor

    Write head (additional):
        - e_t in [0, 1]^W: Erase vector
        - a_t in R^W: Add vector

    Production:
        - Use linear layer + appropriate activation
        - beta_t = softplus(unbounded)
        - g_t = sigmoid(unbounded)
        - s_t = softmax(unbounded)
        - gamma_t = 1 + softplus(unbounded)
        - e_t = sigmoid(unbounded)
"""
    }


def ntm_training() -> Dict[str, str]:
    """
    Training procedure for NTM.

    Returns:
        Dictionary describing training details
    """
    return {
        "end_to_end": """
End-to-End Training:
    NTM is fully differentiable:
        - Soft attention allows gradient flow through memory access
        - All addressing operations are differentiable
        - Controller is differentiable (LSTM/MLP)

    Loss: Standard cross-entropy or MSE depending on task
    Optimizer: RMSprop (used in original paper)

    Key: All operations (read, write, addressing) are "soft"
         so gradients flow from output through memory to inputs
""",
        "initialization": """
Memory Initialization:
    M_0 = small random values (e.g., N(0, 0.01))

    Read/Write weights initialization:
        w_0 = softmax of small random values
        (approximately uniform attention)

    Controller: Standard neural network initialization
""",
        "tasks": """
Benchmark Tasks (from paper):
    1. Copy: Output sequence identical to input
    2. Repeat Copy: Output input sequence k times
    3. Associative Recall: Output value paired with query key
    4. Dynamic N-Grams: Predict next symbol from context
    5. Priority Sort: Sort input by associated priorities

    All tasks test:
        - Memory read/write capabilities
        - Attention over memory
        - Learning algorithmic structure
"""
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class NTMArchitecture:
    """Reference architecture for Neural Turing Machine."""

    memory_size: int = 128  # N: Number of memory locations
    memory_width: int = 20  # W: Size of each location
    controller_size: int = 100  # Hidden size of controller
    num_read_heads: int = 1
    num_write_heads: int = 1

    @staticmethod
    def full_architecture() -> str:
        """Complete NTM architecture."""
        return """
NTM Architecture:

Input x_t -> Controller -> (Read from M_t) -> Output y_t
                |                   ^
                v                   |
            Write to M_t      Read vector r_t

Components:
    1. Controller: LSTM or feedforward network
       Input: [x_t, r_{t-1}] (input + previous read vectors)
       Output: Interface parameters + output

    2. Memory: M_t in R^{N x W}
       Persistent across time steps

    3. Read Heads (one or more):
       Generate: k_t, beta_t, g_t, s_t, gamma_t
       Compute: w_t^r via addressing mechanism
       Output: r_t = M_t^T * w_t^r

    4. Write Heads (one or more):
       Generate: k_t, beta_t, g_t, s_t, gamma_t, e_t, a_t
       Compute: w_t^w via addressing mechanism
       Update: M_t using erase/add operations

Forward Pass (single step):
    1. Concatenate input with previous read vectors: [x_t; r_{t-1}]
    2. Controller forward: h_t = Controller([x_t; r_{t-1}])
    3. Generate head parameters from h_t
    4. For each write head: update memory M_t
    5. For each read head: read r_t from M_t
    6. Generate output: y_t = f(h_t, r_t)
"""

    @staticmethod
    def comparison_with_lstm() -> str:
        """Compare NTM with LSTM."""
        return """
NTM vs LSTM:

LSTM:
    - Memory: Hidden state h_t (fixed size, typically 256-1024)
    - Access: Implicit through gated updates
    - Capacity: Limited by hidden size
    - Reading: All of state used at once

NTM:
    - Memory: External matrix M_t (e.g., 128 x 20 = 2560 scalars)
    - Access: Explicit via attention-based addressing
    - Capacity: Can be arbitrarily large
    - Reading: Selective via attention weights

Key Differences:
    1. Scalability: NTM memory can scale independently of computation
    2. Precision: NTM can address specific locations precisely
    3. Iteration: NTM can implement loops via location-based shifting
    4. Persistence: NTM memory explicitly persists across time

NTM advantages for algorithmic tasks:
    - Can learn to copy sequences of arbitrary length
    - Can implement sequential access patterns
    - Can store and retrieve by content
"""


# =============================================================================
# Successors and Related Work
# =============================================================================

RELATED_WORK = {
    "dnc_2016": {
        "name": "Differentiable Neural Computer",
        "authors": ["Graves et al."],
        "paper": "Hybrid computing using a neural network with dynamic external memory",
        "innovations": [
            "Temporal memory linkage (track write order)",
            "Dynamic memory allocation",
            "Multiple read/write heads with independent addressing"
        ],
        "improvements": "More sophisticated memory management"
    },
    "memory_networks": {
        "name": "Memory Networks (2015)",
        "authors": ["Weston et al."],
        "focus": "Question answering with memory",
        "difference": "Simpler addressing (no location-based), more focused on NLP"
    },
    "stack_augmented": {
        "name": "Stack-Augmented RNN",
        "structure": "Neural network with differentiable stack",
        "advantage": "Well-suited for context-free languages",
        "disadvantage": "Less general than arbitrary memory"
    },
    "sparse_access_memory": {
        "name": "Sparse Access Memory (2016)",
        "innovation": "Only access k << N memory locations",
        "benefit": "Scales to very large memories",
        "trade_off": "Cannot implement exact algorithms"
    }
}
