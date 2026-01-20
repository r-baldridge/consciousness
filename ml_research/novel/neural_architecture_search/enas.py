"""
Efficient Neural Architecture Search via Parameter Sharing (ENAS) - 2018

Pham et al.'s breakthrough in NAS efficiency. Achieved 1000x speedup over
original NAS by having child models share parameters in a directed acyclic
graph (DAG) supernetwork. A controller RNN learns to generate architectures
by sampling subgraphs from this DAG.

Paper: "Efficient Neural Architecture Search via Parameter Sharing"
arXiv: 1802.03268

Mathematical Formulation:
    ENAS trains two sets of parameters:
        - omega: Shared parameters of the supernet
        - theta: Controller RNN parameters

    Training alternates between:
        1. Train omega to minimize E_{m~pi_theta}[L(m; omega)]
        2. Train theta to maximize E_{m~pi_theta}[R(m)]

    Where:
        - m: Architecture (subgraph sampled from DAG)
        - L: Cross-entropy loss on training data
        - R: Reward (validation accuracy)
        - pi_theta: Controller policy parameterized by theta
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

ENAS = MLMethod(
    method_id="enas_2018",
    name="Efficient Neural Architecture Search (ENAS)",
    year=2018,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE, MethodLineage.RL_LINE],

    authors=["Hieu Pham", "Melody Guan", "Barret Zoph", "Quoc V. Le", "Jeff Dean"],
    paper_title="Efficient Neural Architecture Search via Parameter Sharing",
    paper_url="https://arxiv.org/abs/1802.03268",

    key_innovation=(
        "Introduced parameter sharing across candidate architectures in NAS. "
        "Child models share weights by being subgraphs of a single large DAG, "
        "reducing search time from 450 GPU-days to 0.45 GPU-days (1000x speedup) "
        "while maintaining competitive accuracy."
    ),

    mathematical_formulation=r"""
ENAS Training Objective:
    Two sets of parameters to optimize:
        - omega: Shared parameters of the DAG (supernet)
        - theta: Controller LSTM parameters

Alternating Training:
    Step 1: Fix theta, train omega for one epoch
        omega* = argmin_omega E_{m ~ pi_theta(m)}[L(m; omega)]

        In practice: Sample m from controller, update omega via SGD on L(m; omega)

    Step 2: Fix omega, train theta via REINFORCE
        theta <- theta + eta * grad_theta E_{m ~ pi_theta}[R(m)]

        Gradient estimation (REINFORCE with baseline):
            grad_theta J(theta) = E_m[(R(m) - b) * grad_theta log pi_theta(m)]

        Where b is moving average baseline to reduce variance

Controller LSTM:
    Samples architecture as sequence of decisions:
    For cell-based search (each node i):
        - Sample activation function for node i
        - For each previous node j < i:
            - Sample whether to connect j -> i
            - If connected, sample operation type

    Architecture m = (connections, operations) ~ pi_theta(m)

Reward Signal:
    R(m) = Accuracy(m; omega, D_val)

    Child model m is evaluated on validation set using shared weights omega

DAG Representation:
    Supernet is a DAG where:
        - Each node is a local computation
        - Each edge has multiple operation choices
        - Child architecture = subgraph with one operation per edge
""",

    predecessors=["nas_2017", "nasnet_2018"],
    successors=["darts_2018", "proxylessnas"],

    tags=["architecture-search", "weight-sharing", "reinforcement-learning", "automl"]
)

# Note: ENAS was the first major efficiency breakthrough in NAS. The key insight
# is that training thousands of architectures from scratch is wasteful -
# instead, share parameters so each new architecture starts from a trained
# state. This approach influenced all subsequent weight-sharing NAS methods.


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def controller_rnn_structure() -> Dict[str, str]:
    """
    ENAS Controller LSTM structure.

    The controller is an LSTM that generates architectures as sequences
    of decisions. For each node in the cell, it decides which previous
    nodes to connect to and which operations to use.

    Returns:
        Dictionary describing controller structure
    """
    return {
        "architecture": """
Controller LSTM:
    hidden_size = 100
    num_layers = 1

    For each node i in {1, 2, ..., N}:
        1. Sample activation function:
           p_act = softmax(W_act * h)
           activation_i ~ Categorical(p_act)

        2. For each previous node j in {0, 1, ..., i-1}:
           Input: Embedding of node j's activation
           p_conn = sigmoid(W_conn * h)
           connect_j_i ~ Bernoulli(p_conn)

           If connect_j_i = 1:
               p_op = softmax(W_op * h)
               operation_j_i ~ Categorical(p_op)
""",
        "sampling": """
Sampling Process (Cell-based):
    For each node i = 1 to N:
        # Decide which previous nodes to connect
        anchors = []
        for j in range(i):
            h_j = embed(node_j)
            h_i = LSTM(h_{t-1}, h_j)
            p_select = sigmoid(v^T * tanh(W * [h_j; h_i]))
            if sample(p_select) == 1:
                anchors.append(j)

        # If no anchors selected, use previous node
        if len(anchors) == 0:
            anchors = [i-1]

        # For each anchor, select operation
        for j in anchors:
            h_op = LSTM(h_{t-1}, embed(anchor_j))
            p_op = softmax(W_op * h_op)
            op_j_i = sample(p_op)
""",
        "operations": [
            "identity",
            "3x3 separable conv",
            "5x5 separable conv",
            "3x3 average pooling",
            "3x3 max pooling"
        ]
    }


def enas_training_loop() -> Dict[str, str]:
    """
    ENAS training procedure pseudocode.

    Returns:
        Dictionary with training algorithm description
    """
    return {
        "algorithm": """
ENAS Training Algorithm:

Initialize:
    omega: Shared parameters of DAG (random)
    theta: Controller LSTM parameters (random)

For epoch = 1 to num_epochs:
    # Phase 1: Train shared parameters
    For batch in training_data:
        m = sample_architecture(controller)  # Sample subgraph
        loss = compute_loss(m, omega, batch)  # Forward pass through subgraph
        omega = omega - lr_omega * grad(loss, omega)  # Update shared params

    # Phase 2: Train controller
    For step = 1 to controller_steps:
        m = sample_architecture(controller)
        R = evaluate_accuracy(m, omega, validation_data)  # Reward

        # REINFORCE gradient
        log_prob = compute_log_prob(m, theta)
        baseline = moving_average(R)
        advantage = R - baseline

        grad_theta = advantage * grad(log_prob, theta)
        theta = theta + lr_theta * grad_theta

        # Update baseline
        baseline = 0.99 * baseline + 0.01 * R
""",
        "hyperparameters": {
            "controller_lstm_size": 100,
            "controller_learning_rate": 0.00035,
            "controller_entropy_weight": 0.0001,
            "child_learning_rate": 0.05,
            "child_grad_clip": 5.0,
            "num_epochs": 150,
            "controller_train_steps": 2000
        }
    }


def reinforce_gradient() -> Dict[str, str]:
    """
    REINFORCE gradient estimation for ENAS controller.

    Returns:
        Dictionary with gradient formulation
    """
    return {
        "formula": """
Policy Gradient (REINFORCE):
    J(theta) = E_{m ~ pi_theta}[R(m)]

    grad_theta J(theta) = E_m[R(m) * grad_theta log pi_theta(m)]

With baseline (variance reduction):
    grad_theta J(theta) = E_m[(R(m) - b) * grad_theta log pi_theta(m)]

    Where b = exponential moving average of R(m)
""",
        "log_probability": """
Log Probability of Architecture:
    log pi_theta(m) = sum_{decision d in m} log p_theta(d | previous decisions)

    For cell-based ENAS:
        log pi(m) = sum_i [log p(activation_i) + sum_{j<i} log p(connect_j_i) * connect_j_i
                         + sum_{j in anchors_i} log p(operation_j_i)]
""",
        "entropy_regularization": """
Entropy Bonus:
    J'(theta) = J(theta) + beta * H(pi_theta)

    H(pi_theta) = -E[log pi_theta(m)]

    Encourages exploration by rewarding high-entropy policies
"""
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class ENASArchitecture:
    """Reference architecture for ENAS."""

    num_cells: int = 6  # Number of cells in final network
    num_nodes: int = 4  # Intermediate nodes per cell
    num_channels: int = 36  # Initial number of channels

    @staticmethod
    def dag_structure() -> str:
        """DAG supernet structure in ENAS."""
        return """
ENAS DAG Supernet:

    Each node in DAG can compute:
        h_i = f_i(sum_{j in inputs_i} g_{j,i}(h_j))

    Where:
        - f_i: Activation function (tanh, ReLU, sigmoid, identity)
        - g_{j,i}: Operation from j to i (conv, pool, identity)

    Cell DAG Structure:
        Nodes: {h_{-1}, h_0, h_1, h_2, ..., h_N}
        - h_{-1}: Output of cell k-2
        - h_0: Output of cell k-1
        - h_1 to h_N: Intermediate nodes

        Output: h_out = average(h_i for i not used as input to other nodes)

    Example 4-node cell:
        h_{-1} ----[ops]----> h_1 ----[ops]----> h_3
                              \\                 ^
                               \\----[ops]----> h_4
        h_0   ----[ops]----> h_2 ----[ops]-----/

        Output = average(loose ends)
"""

    @staticmethod
    def macro_search() -> str:
        """ENAS macro architecture search."""
        return """
ENAS Macro Search (for entire network, not cells):

    Controller generates entire architecture layer by layer:

    For layer l = 1 to L:
        1. Select layer type:
           - Convolution (3x3, 5x5)
           - Separable convolution (3x3, 5x5)
           - Max pooling (3x3)
           - Average pooling (3x3)

        2. Select skip connections from previous layers:
           For each previous layer j < l:
               connect_j_l ~ Bernoulli(p_j_l)

        3. Combine inputs:
           If multiple inputs: h_l = concat([inputs]) then 1x1 conv
           If single input: h_l = input

    Reduction layers placed at layer L/3 and 2L/3
"""

    @staticmethod
    def micro_search() -> str:
        """ENAS micro/cell-based search."""
        return """
ENAS Micro Search (cell-based):

    Search for two types of cells:
        1. Normal Cell: Maintains spatial dimensions
        2. Reduction Cell: Reduces dimensions by stride 2

    Cell Discovery:
        For each intermediate node i in {1, ..., B}:
            1. Select two input nodes from {0, ..., i-1}
               (node 0 = h_{k-2}, node 1 = h_{k-1})

            2. Select operation for each selected input:
               ops = {identity, 3x3 sep conv, 5x5 sep conv,
                      3x3 avg pool, 3x3 max pool}

            3. Node output: h_i = op_1(h_a) + op_2(h_b)

        Cell output: concat all unused intermediate nodes

    Network from Cells:
        Stack N copies of (Normal, Normal, Reduction)
        Total cells = 3N for CIFAR-10 experiments
"""


# =============================================================================
# Results and Comparison
# =============================================================================

ENAS_RESULTS = {
    "cifar10_macro": {
        "test_error": "4.23%",
        "search_cost": "0.32 GPU-days",
        "params": "21.3M"
    },
    "cifar10_micro": {
        "test_error": "2.89%",
        "search_cost": "0.45 GPU-days",
        "params": "4.6M",
        "note": "Cell discovered by ENAS, then stacked deeper with more channels"
    },
    "comparison_with_nas": {
        "nas_2017": {
            "test_error": "3.65%",
            "search_cost": "22400 GPU-hours"
        },
        "enas_2018": {
            "test_error": "2.89%",
            "search_cost": "10 GPU-hours"
        },
        "speedup": "~1000x"
    },
    "penn_treebank": {
        "task": "Language modeling",
        "test_perplexity": "55.8",
        "search_cost": "10 GPU-hours"
    }
}
