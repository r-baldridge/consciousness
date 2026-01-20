"""
HyperNetworks - Ha et al. (2016)

Networks that generate weights for other networks, enabling
dynamic weight generation and efficient weight sharing.

Paper: "HyperNetworks"
arXiv: 1609.09106

Mathematical Formulation:
    Given a hypernetwork H and target network T:
    W_T = H(z)

    Where:
        z: Input embedding or conditioning information
        H: Hypernetwork function
        W_T: Generated weights for target network T

Key Innovation:
    Instead of learning fixed weights, learn a function that
    generates weights. This enables weight sharing, adaptation,
    and meta-learning capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

HYPERNETWORK = MLMethod(
    method_id="hypernetwork_2016",
    name="HyperNetworks",
    year=2016,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.PERCEPTRON_LINE],

    authors=["David Ha", "Andrew Dai", "Quoc V. Le"],
    paper_title="HyperNetworks",
    paper_url="https://arxiv.org/abs/1609.09106",

    key_innovation=(
        "Introduced the concept of using one network (the hypernetwork) to generate "
        "weights for another network (the target network). This enables: 1) Parameter "
        "efficiency through weight sharing, 2) Dynamic weight adaptation based on input, "
        "3) Connection to meta-learning and model generation. Demonstrated on CNNs, "
        "RNNs, and achieved state-of-the-art compression rates."
    ),

    mathematical_formulation=r"""
Basic HyperNetwork:
    Given target network layer with weight matrix W in R^{d_out x d_in}

    Hypernetwork generates W:
        W = H(z)

    Where z in R^{d_z} is an embedding vector

Static HyperNetwork (weight sharing):
    z_i: Learnable embedding for layer i
    W_i = H(z_i)

    Parameters: |z_1| + ... + |z_L| + |H|
    vs Original: |W_1| + ... + |W_L|

    Compression if: sum|z_i| + |H| << sum|W_i|

Dynamic HyperNetwork (input-conditioned):
    z = Encoder(input)
    W = H(z)
    output = TargetNetwork(input; W)

    Weights change based on input!

Relaxed Weight Generation:
    Generate weight matrix row-by-row:
    W[j, :] = H(z, j)  # j-th row

    Or column-by-column, or block-by-block

HyperLSTM:
    For LSTM with gates (i, f, o, g):

    Standard: [i; f; o; g] = W_h h_{t-1} + W_x x_t + b

    HyperLSTM:
        # Hypernetwork LSTM processes input
        h_hyper_t = LSTM_hyper(x_t, h_hyper_{t-1})

        # Generate weights for main LSTM
        W_h = H_1(z_h)
        W_x = H_2(z_x)
        b = H_3(z_b)

        # Or scaling approach:
        W_h = W_h_base * d_h  # Element-wise scaling
        W_x = W_x_base * d_x
        b = b_base * d_b

        Where d_h, d_x, d_b = HyperNet(h_hyper_t)
""",

    predecessors=["mlp", "weight_tying"],
    successors=["hypernet_meta_learning", "neural_architecture_search"],

    tags=[
        "hypernetwork", "weight-generation", "meta-learning",
        "model-compression", "dynamic-weights", "weight-sharing"
    ],
    notes=(
        "HyperNetworks connect to many important concepts: meta-learning (learning "
        "to learn), neural architecture search (generating architectures), model "
        "compression (weight sharing), and fast adaptation (input-conditioned weights). "
        "The paper showed 100x compression on neural machine translation with <1% loss."
    )
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for HyperNetworks."""
    return HYPERNETWORK


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class HyperNetworkArchitecture:
    """Reference architecture for HyperNetworks."""

    # Hypernetwork parameters
    embedding_dim: int = 64
    hyper_hidden_dim: int = 256

    # Target network parameters
    target_layers: int = 3
    target_hidden_dim: int = 512

    @staticmethod
    def basic_hypernetwork() -> str:
        """Basic hypernetwork structure."""
        return """
Basic HyperNetwork:

Components:
    1. Embedding layer: z_i for each target layer i
    2. Hypernetwork H: maps z_i to weight matrix

Target Layer i (to be generated):
    W_i in R^{d_out x d_in}  (e.g., 512 x 512)

Embedding:
    z_i in R^{d_z}  (e.g., 64)

Hypernetwork (MLP):
    H: R^{d_z} -> R^{d_out x d_in}

    h = ReLU(Linear(d_z -> hidden))
    W_i = Reshape(Linear(hidden -> d_out * d_in))

Parameter Comparison:
    Direct: d_out * d_in = 512 * 512 = 262,144 per layer
    HyperNet: d_z + |H| = 64 + (64*256 + 256*262144) ≈ very large!

Problem: Full weight generation often INCREASES parameters!

Solution: Generate low-rank or structured weights
"""

    @staticmethod
    def factorized_hypernetwork() -> str:
        """Factorized hypernetwork for efficiency."""
        return """
Factorized HyperNetwork (practical):

Instead of generating full W in R^{d_out x d_in}:

Option 1: Low-rank factorization
    W = U V^T
    U in R^{d_out x r}, V in R^{d_in x r}

    Generate U and V separately:
    U = H_U(z)  [output: d_out * r]
    V = H_V(z)  [output: d_in * r]

    Total: 2 * r * max(d_out, d_in) << d_out * d_in

Option 2: Kernel generation (for CNNs)
    Kernel K in R^{out_ch x in_ch x k x k}

    Generate per-kernel:
    K[i, j] = H(z_layer, i, j)  # k x k kernel

    Or generate kernel basis:
    K[i, j] = sum_b alpha_{i,j,b} * K_basis[b]
    Generate only alpha coefficients

Option 3: Scaling approach (HyperLSTM style)
    W = W_base * diag(d)

    Where:
    W_base: Fixed base weights
    d = H(z): Scaling vector

    Very efficient: only generate d_out + d_in values
"""

    @staticmethod
    def hyperlstm_structure() -> str:
        """HyperLSTM structure."""
        return """
HyperLSTM:

Two-level architecture:
    1. Hypernetwork LSTM (small)
    2. Main LSTM (large, with generated weights)

Hypernetwork LSTM:
    Input: x_t
    Hidden: h_hyper_t in R^{d_hyper}  (e.g., 128)

    h_hyper_t, c_hyper_t = LSTM_hyper(x_t, h_hyper_{t-1}, c_hyper_{t-1})

Weight Generation (scaling approach):
    For main LSTM gates [i; f; o; g] = W_h h + W_x x + b

    # Generate scaling vectors from hypernetwork
    d_h = Linear(h_hyper_t)  # R^{4*d_main}
    d_x = Linear(h_hyper_t)  # R^{4*d_main}
    d_b = Linear(h_hyper_t)  # R^{4*d_main}

    # Apply scaling to base weights
    W_h_scaled = W_h_base * expand(d_h)
    W_x_scaled = W_x_base * expand(d_x)
    b_scaled = b_base * d_b

Main LSTM (with generated weights):
    gates = W_h_scaled @ h_main_{t-1} + W_x_scaled @ x_t + b_scaled
    i, f, o, g = split(gates, 4)
    c_main_t = sigmoid(f) * c_main_{t-1} + sigmoid(i) * tanh(g)
    h_main_t = sigmoid(o) * tanh(c_main_t)

Output: h_main_t (from main LSTM)

Key Insight:
    Hypernetwork modulates main LSTM's dynamics based on input
    Different inputs -> different effective weights
    Enables input-dependent temporal processing
"""

    @staticmethod
    def dynamic_hypernetwork() -> str:
        """Dynamic (input-conditioned) hypernetwork."""
        return """
Dynamic HyperNetwork:

The weights of the target network depend on the input itself.

Architecture:
    Input: x

    # Encode input to conditioning vector
    z = Encoder(x)

    # Generate weights
    W_1, ..., W_L = H(z)

    # Apply target network with generated weights
    y = TargetNetwork(x; W_1, ..., W_L)

Example: Image classification with input-dependent convolutions

    z = GlobalAvgPool(ResNet_small(image))
    conv_weights = H(z)
    features = Conv(image, conv_weights)
    logits = Classifier(features)

Applications:
    1. Few-shot learning: z encodes task, H generates task-specific weights
    2. Personalization: z encodes user, H generates user-specific model
    3. Domain adaptation: z encodes domain, H generates domain-adapted weights
    4. Conditional generation: z encodes condition, H generates conditional weights

Key Difference from Standard Nets:
    Standard: Same weights for all inputs
    Dynamic HyperNet: Different effective weights per input
"""


# =============================================================================
# Mathematical Functions (Reference)
# =============================================================================

def hypernetwork_forward(z, target_architecture, hypernetwork):
    """
    Generate target network weights and perform forward pass.

    Args:
        z: Conditioning embedding
        target_architecture: Structure of target network
        hypernetwork: Weight generation network

    Returns:
        Generated weights for target network
    """
    return {
        "algorithm": """
            weights = {}
            for layer_name, layer_spec in target_architecture:
                # Generate weights for this layer
                w_flat = hypernetwork(z, layer_spec)
                weights[layer_name] = reshape(w_flat, layer_spec.shape)
            return weights
        """,
        "complexity": "O(|H| + sum_layers |W_layer|)",
        "tradeoff": """
            Compression achieved when:
            |z| * num_layers + |H| < sum |W_layer|
        """
    }


def weight_scaling_approach(base_weights, scaling_network, context):
    """
    Efficient weight generation via scaling.

    Instead of generating full weight matrices,
    generate scaling factors for base weights.

    Args:
        base_weights: Fixed base weight matrix
        scaling_network: Network that outputs scaling factors
        context: Conditioning information

    Returns:
        Scaled weights
    """
    return {
        "formula": """
            d = scaling_network(context)  # [d_out + d_in] or [d_out]

            # Row scaling
            W_scaled = diag(d) @ W_base

            # Or column scaling
            W_scaled = W_base @ diag(d)

            # Or both (rank-1 modulation)
            W_scaled = diag(d_out) @ W_base @ diag(d_in)
        """,
        "efficiency": """
            Full generation: O(d_out * d_in) outputs
            Scaling: O(d_out + d_in) outputs

            For 512x512: 262,144 vs 1,024 (256x reduction!)
        """,
        "expressiveness": """
            Row scaling: Change output neuron importance
            Column scaling: Change input feature importance
            Both: Rank-1 modulation, limited but efficient
        """
    }


def chunked_weight_generation():
    """
    Generate weights in chunks for large layers.

    For very large weight matrices, generating all at once
    may exceed memory. Generate in chunks instead.
    """
    return {
        "method_1_row_chunks": """
            # Generate weight matrix row by row
            W = []
            for i in range(d_out):
                row_i = H(z, positional_encoding(i))
                W.append(row_i)
            W = stack(W)
        """,
        "method_2_block_chunks": """
            # Generate weight matrix block by block
            W = zeros(d_out, d_in)
            for i in range(0, d_out, block_size):
                for j in range(0, d_in, block_size):
                    block = H(z, positional_encoding(i, j))
                    W[i:i+block_size, j:j+block_size] = block
        """,
        "method_3_basis_combination": """
            # Generate as linear combination of basis matrices
            alphas = H(z)  # [num_basis]
            W = sum(alpha_k * W_basis[k] for k in range(num_basis))

            Only need to generate num_basis coefficients!
        """
    }


# =============================================================================
# Key Insights and Applications
# =============================================================================

HYPERNETWORK_INSIGHTS = {
    "weight_as_function": """
        Traditional view: Weights are parameters to optimize
        HyperNet view: Weights are outputs of a function

        This shift enables:
        - Sharing: Same H, different z -> different but related W
        - Adaptation: Change z -> adapt W without retraining
        - Generation: Learn a distribution over weights
    """,

    "compression_via_sharing": """
        Weight Sharing Compression:
        - Multiple layers share single hypernetwork H
        - Each layer has unique embedding z_i
        - Compression ratio: (L * |z| + |H|) / (L * |W|)

        Works best when layers should be similar.
    """,

    "connection_to_meta_learning": """
        Meta-Learning Connection:
        - Task embedding z encodes task identity
        - H generates task-specific weights
        - Learning H = learning to learn

        MAML vs HyperNets:
        - MAML: Initialize + gradient steps
        - HyperNet: Direct weight generation (faster)
    """,

    "soft_weight_sharing": """
        Unlike hard weight tying (W_1 = W_2):
        - Soft sharing: W_1 = H(z_1), W_2 = H(z_2)
        - Related but not identical
        - z_1 close to z_2 -> W_1 close to W_2
        - Enables smooth interpolation between configurations
    """
}


HYPERNETWORK_APPLICATIONS = {
    "model_compression": {
        "approach": "Share weights via hypernetwork",
        "result": "100x compression on NMT with <1% quality loss",
        "key": "Layers that should behave similarly share embeddings"
    },

    "meta_learning": {
        "approach": "Generate task-specific weights from task embedding",
        "examples": "Few-shot classification, fast adaptation",
        "advantage": "No gradient steps at test time (faster than MAML)"
    },

    "neural_architecture_search": {
        "approach": "HyperNet generates weights for candidate architectures",
        "benefit": "Don't need to train each architecture from scratch",
        "method": "Architecture encoding -> weights"
    },

    "continual_learning": {
        "approach": "Task-conditioned weight generation",
        "benefit": "Avoid catastrophic forgetting by task-specific weights",
        "method": "Task ID -> task-specific weights"
    },

    "personalization": {
        "approach": "User embedding -> user-specific model",
        "examples": "Personalized recommendations, federated learning",
        "advantage": "Single model serves many users with adaptation"
    },

    "conditional_generation": {
        "approach": "Condition -> generator weights",
        "examples": "Class-conditional GANs, style transfer",
        "method": "Style/class embedding generates decoder weights"
    }
}


HYPERNETWORK_VARIANTS = {
    "hyperlstm_2016": {
        "description": "HyperLSTM with scaling approach",
        "efficiency": "Small hypernetwork modulates large LSTM"
    },
    "hypernetwork_nas_2018": {
        "description": "HyperNets for neural architecture search",
        "benefit": "Amortize training across architectures"
    },
    "hypernetwork_meta_2020": {
        "description": "HyperNets for meta-learning",
        "methods": "LEO, MetaNets, etc."
    },
    "hypernetwork_nerf_2021": {
        "description": "HyperNeRF for dynamic scenes",
        "method": "Time embedding -> NeRF weights"
    }
}
