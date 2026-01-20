"""
Graph Neural Networks - Message Passing Neural Networks

Research index for graph neural network architectures that learn
representations on graph-structured data through message passing.

Methods included:
- GNN Basics: Message passing framework
- GCN (2016): Graph Convolutional Networks - Kipf & Welling
- GAT (2017): Graph Attention Networks - Velickovic et al.
- GraphSAGE (2017): Inductive node embeddings - Hamilton et al.

Core Formulation (Message Passing):
    h_v^{(k+1)} = UPDATE(h_v^{(k)}, AGGREGATE({h_u^{(k)} : u in N(v)}))

    Where:
        h_v: Node v's hidden representation
        N(v): Neighbors of node v
        k: Layer index
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entries
# =============================================================================

GNN_BASIC = MLMethod(
    method_id="gnn_message_passing",
    name="Message Passing Neural Networks",
    year=2017,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.PERCEPTRON_LINE],

    authors=["Justin Gilmer", "Samuel S. Schoenholz", "Patrick F. Riley",
             "Oriol Vinyals", "George E. Dahl"],
    paper_title="Neural Message Passing for Quantum Chemistry",
    paper_url="https://arxiv.org/abs/1704.01212",

    key_innovation=(
        "Unified framework for graph neural networks as message passing: nodes "
        "iteratively update their representations by aggregating messages from "
        "neighbors. This abstraction encompasses GCN, GAT, GraphSAGE, and many "
        "others as special cases with different message and aggregation functions."
    ),

    mathematical_formulation=r"""
Message Passing Framework:

Message function:
    m_v^{(k+1)} = AGGREGATE({MESSAGE(h_v^{(k)}, h_u^{(k)}, e_{vu}) : u in N(v)})

Update function:
    h_v^{(k+1)} = UPDATE(h_v^{(k)}, m_v^{(k+1)})

Common choices:
    AGGREGATE: sum, mean, max, attention-weighted sum
    MESSAGE: Linear(concat(h_v, h_u)), Linear(h_u), etc.
    UPDATE: GRU(h_v, m_v), MLP(concat(h_v, m_v)), etc.

Readout (for graph-level tasks):
    h_G = READOUT({h_v^{(K)} : v in V})

    READOUT: sum, mean, attention pooling, set2set

General form:
    h_v^{(k+1)} = sigma(W^{(k)} * AGG({h_u^{(k)} : u in N(v) union {v}}))
""",

    predecessors=["spectral_graph_theory", "recurrent_gnn_2009"],
    successors=["gcn_2016", "gat_2017", "graphsage_2017"],

    tags=["graph", "message-passing", "aggregation", "node-embedding"],
    notes=(
        "Message passing provides a unified view of graph learning. The choice of "
        "message and aggregation functions determines the model's expressivity. "
        "Most GNN variants can be expressed as message passing with specific choices."
    )
)


GCN = MLMethod(
    method_id="gcn_2016",
    name="Graph Convolutional Networks",
    year=2016,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],

    authors=["Thomas N. Kipf", "Max Welling"],
    paper_title="Semi-Supervised Classification with Graph Convolutional Networks",
    paper_url="https://arxiv.org/abs/1609.02907",

    key_innovation=(
        "Simplified spectral graph convolutions to a first-order approximation that "
        "is efficient and effective. The layer-wise propagation rule with symmetric "
        "normalization became the foundation for modern graph neural networks. "
        "Demonstrated strong semi-supervised learning with very few labels."
    ),

    mathematical_formulation=r"""
Spectral Graph Convolution (background):
    g_theta * x = U g_theta(Lambda) U^T x

    Where:
        L = U Lambda U^T is eigendecomposition of graph Laplacian
        Expensive: O(n^3) for eigendecomposition

Chebyshev Approximation:
    g_theta * x ≈ sum_{k=0}^{K} theta_k T_k(L_tilde) x

    Where T_k are Chebyshev polynomials
    L_tilde = 2L/lambda_max - I (scaled Laplacian)

First-Order Approximation (GCN):
    With K=1, lambda_max=2:
    g_theta * x ≈ theta_0 x + theta_1 (L - I) x
                = theta_0 x - theta_1 D^{-1/2} A D^{-1/2} x

    Further simplify with theta = theta_0 = -theta_1:
    g_theta * x = theta (I + D^{-1/2} A D^{-1/2}) x

Renormalization Trick:
    I + D^{-1/2} A D^{-1/2} -> D_tilde^{-1/2} A_tilde D_tilde^{-1/2}

    Where:
        A_tilde = A + I (add self-loops)
        D_tilde = D + I (adjusted degree matrix)

GCN Layer:
    H^{(l+1)} = sigma(D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H^{(l)} W^{(l)})

    Or per-node:
    h_v^{(l+1)} = sigma(W^{(l)} * mean({h_u^{(l)} : u in N(v) union {v}}))
""",

    predecessors=["spectral_gnn", "gnn_message_passing"],
    successors=["gat_2017", "graphsage_2017", "gin_2018"],

    tags=["graph", "convolution", "spectral", "semi-supervised", "normalization"],
    notes=(
        "GCN uses symmetric normalization which averages neighbor features weighted "
        "by degree. This prevents scale issues but can cause over-smoothing with "
        "many layers. The simplicity and effectiveness made GCN extremely popular."
    )
)


GAT = MLMethod(
    method_id="gat_2017",
    name="Graph Attention Networks",
    year=2017,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ATTENTION,
    lineages=[MethodLineage.ATTENTION_LINE],

    authors=["Petar Velickovic", "Guillem Cucurull", "Arantxa Casanova",
             "Adriana Romero", "Pietro Lio", "Yoshua Bengio"],
    paper_title="Graph Attention Networks",
    paper_url="https://arxiv.org/abs/1710.10903",

    key_innovation=(
        "Applied attention mechanism to graphs, learning to weight neighbor "
        "contributions dynamically. Unlike GCN's fixed normalization, GAT learns "
        "which neighbors are most relevant for each node. Multi-head attention "
        "stabilizes training and captures different relationship types."
    ),

    mathematical_formulation=r"""
Attention Coefficients:
    e_{vu} = LeakyReLU(a^T [W h_v || W h_u])

    Where:
        W in R^{F' x F}: Shared linear transformation
        a in R^{2F'}: Attention weight vector
        ||: Concatenation

Normalized Attention (softmax over neighbors):
    alpha_{vu} = softmax_u(e_{vu}) = exp(e_{vu}) / sum_{k in N(v)} exp(e_{vk})

GAT Layer (single head):
    h_v' = sigma(sum_{u in N(v)} alpha_{vu} W h_u)

Multi-Head Attention:
    h_v' = ||_{k=1}^{K} sigma(sum_{u in N(v)} alpha_{vu}^k W^k h_u)

    Where || is concatenation of K heads

    Output dimension: K * F' (or averaged for final layer)

Final Layer (averaging):
    h_v' = sigma(1/K sum_{k=1}^{K} sum_{u in N(v)} alpha_{vu}^k W^k h_u)

Self-Attention (include self-loops):
    N(v) includes v itself
    Node attends to its own features
""",

    predecessors=["gcn_2016", "attention_bahdanau_2014"],
    successors=["gatv2_2021", "graphtransformer_2020"],

    tags=["graph", "attention", "multi-head", "dynamic-weighting"],
    notes=(
        "GAT's attention mechanism allows learning edge importance without explicit "
        "edge features. However, the original GAT has a 'static attention' issue where "
        "attention depends only on node features, not their interaction. GATv2 fixes this."
    )
)


GRAPHSAGE = MLMethod(
    method_id="graphsage_2017",
    name="GraphSAGE",
    year=2017,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.PERCEPTRON_LINE],

    authors=["William L. Hamilton", "Rex Ying", "Jure Leskovec"],
    paper_title="Inductive Representation Learning on Large Graphs",
    paper_url="https://arxiv.org/abs/1706.02216",

    key_innovation=(
        "Introduced inductive graph learning through sampling and aggregation. "
        "Unlike transductive methods (GCN) that need all nodes during training, "
        "GraphSAGE can generalize to unseen nodes by learning aggregator functions. "
        "Neighbor sampling enables scaling to large graphs."
    ),

    mathematical_formulation=r"""
GraphSAGE Layer:
    # Aggregate neighbor features
    h_{N(v)}^{(k)} = AGGREGATE_k({h_u^{(k-1)} : u in N(v)})

    # Combine with self
    h_v^{(k)} = sigma(W^{(k)} * CONCAT(h_v^{(k-1)}, h_{N(v)}^{(k)}))

    # Normalize
    h_v^{(k)} = h_v^{(k)} / ||h_v^{(k)}||_2

Aggregator Variants:
    Mean Aggregator:
        h_{N(v)} = mean({h_u : u in N(v)})

    LSTM Aggregator:
        h_{N(v)} = LSTM(shuffle({h_u : u in N(v)}))

    Pooling Aggregator:
        h_{N(v)} = max({sigma(W_pool h_u + b) : u in N(v)})

Sampling Strategy:
    For each layer k:
        Sample S_k neighbors uniformly at random

    Neighborhood expansion:
        Layer 1: Sample S_1 neighbors of v
        Layer 2: Sample S_2 neighbors of each sampled node
        ...

    Receptive field: |S_1| * |S_2| * ... * |S_K| nodes

Unsupervised Loss:
    J_G(z_u) = -log(sigma(z_u^T z_v)) - Q * E_{v_n ~ P_n(v)} log(sigma(-z_u^T z_{v_n}))

    Where:
        v is a neighbor of u (positive sample)
        v_n is a random node (negative sample)
        Q is number of negative samples
""",

    predecessors=["gcn_2016", "deepwalk_2014"],
    successors=["pinsage_2018", "cluster_gcn_2019"],

    tags=["graph", "inductive", "sampling", "aggregation", "scalable"],
    notes=(
        "GraphSAGE's key contribution is inductive learning: train on one set of "
        "nodes, apply to different/new nodes. This is crucial for dynamic graphs "
        "where new nodes appear (e.g., new users in social networks)."
    )
)


def get_gnn_info() -> MLMethod:
    """Return the MLMethod entry for Message Passing GNNs."""
    return GNN_BASIC


def get_gcn_info() -> MLMethod:
    """Return the MLMethod entry for GCN."""
    return GCN


def get_gat_info() -> MLMethod:
    """Return the MLMethod entry for GAT."""
    return GAT


def get_graphsage_info() -> MLMethod:
    """Return the MLMethod entry for GraphSAGE."""
    return GRAPHSAGE


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class GNNArchitecture:
    """Reference architecture for Graph Neural Networks."""

    # Common parameters
    num_layers: int = 2
    hidden_dim: int = 64
    num_heads: int = 8  # For GAT

    # GraphSAGE sampling
    sample_sizes: List[int] = field(default_factory=lambda: [25, 10])

    @staticmethod
    def message_passing_framework() -> str:
        """Message passing framework structure."""
        return """
Message Passing Framework:

For K layers of message passing:

def forward(x, edge_index):
    h = x  # Initial node features

    for k in range(K):
        # Message phase
        messages = []
        for v in nodes:
            m_v = aggregate([message(h[u], h[v], e_uv) for u in neighbors(v)])
            messages.append(m_v)

        # Update phase
        h_new = []
        for v in nodes:
            h_v_new = update(h[v], messages[v])
            h_new.append(h_v_new)

        h = stack(h_new)

    return h

Efficient Implementation (sparse):
    # Using adjacency matrix
    H_new = sigma(A @ H @ W + bias)

    # Or edge_index format
    messages = H[edge_index[0]]  # Source node features
    aggregated = scatter_add(messages, edge_index[1])  # Sum by target
    H_new = sigma(W @ concat(H, aggregated))
"""

    @staticmethod
    def gcn_layer() -> str:
        """GCN layer implementation."""
        return """
GCN Layer:

Input:
    H in R^{N x F}: Node feature matrix
    A in R^{N x N}: Adjacency matrix

Preprocessing (once):
    A_tilde = A + I  # Add self-loops
    D_tilde = diag(sum(A_tilde, axis=1))  # Degree matrix
    A_norm = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}  # Symmetric normalization

Layer computation:
    H' = sigma(A_norm H W)

    Where W in R^{F x F'} is learnable

Per-node view:
    h_v' = sigma(W^T * (1/sqrt(d_v)) * sum_{u in N(v)} (1/sqrt(d_u)) * h_u)

         ≈ sigma(W^T * mean({h_u : u in N(v) union {v}}))

Intuition:
    Each node's new representation is a normalized average of
    its neighbors' (and its own) features, transformed by W.
"""

    @staticmethod
    def gat_layer() -> str:
        """GAT layer implementation."""
        return """
GAT Layer:

Input:
    H in R^{N x F}: Node feature matrix
    edge_index: Edge connectivity

For each head k:
    # Linear transformation
    H_k = H W_k  # [N, F']

    # Compute attention scores for all edges
    src_features = H_k[edge_index[0]]  # [E, F']
    dst_features = H_k[edge_index[1]]  # [E, F']
    edge_features = concat(src_features, dst_features)  # [E, 2F']

    e = LeakyReLU(edge_features @ a_k)  # [E]

    # Normalize per node (softmax over incoming edges)
    alpha = softmax_by_target(e, edge_index[1])  # [E]

    # Weighted aggregation
    messages = alpha.unsqueeze(-1) * src_features  # [E, F']
    H_k_new = scatter_add(messages, edge_index[1])  # [N, F']

Combine heads:
    # Intermediate layers: concatenate
    H' = concat([H_1_new, ..., H_K_new])  # [N, K*F']

    # Output layer: average
    H' = mean([H_1_new, ..., H_K_new])  # [N, F']

    H' = sigma(H')  # Activation
"""

    @staticmethod
    def graphsage_layer() -> str:
        """GraphSAGE layer implementation."""
        return """
GraphSAGE Layer:

Input:
    H in R^{N x F}: Node feature matrix
    edge_index: Edge connectivity

Sampling (for large graphs):
    For each node v:
        Sample S neighbors uniformly at random
        N_sample(v) = random_sample(N(v), S)

Aggregation (mean aggregator):
    for v in nodes:
        h_N_v = mean([H[u] for u in N_sample(v)])
        H_new[v] = sigma(W @ concat(H[v], h_N_v))
        H_new[v] = H_new[v] / norm(H_new[v])  # L2 normalize

Aggregation (pooling aggregator):
    for v in nodes:
        neighbor_transformed = [sigma(W_pool @ H[u]) for u in N_sample(v)]
        h_N_v = element_wise_max(neighbor_transformed)
        H_new[v] = sigma(W @ concat(H[v], h_N_v))
        H_new[v] = H_new[v] / norm(H_new[v])

Mini-batch Training:
    1. Sample batch of target nodes
    2. Expand to multi-hop neighborhood (sample S per layer)
    3. Forward pass on subgraph
    4. Compute loss only on target nodes
"""


# =============================================================================
# Mathematical Functions (Reference)
# =============================================================================

def spectral_graph_convolution():
    """
    Background on spectral graph convolution.
    """
    return {
        "graph_laplacian": """
            L = D - A  (unnormalized)
            L_sym = I - D^{-1/2} A D^{-1/2}  (symmetric normalized)
            L_rw = I - D^{-1} A  (random walk normalized)
        """,
        "spectral_filtering": """
            Signal on graph: x in R^N

            Fourier transform: x_hat = U^T x
            Where L = U Lambda U^T (eigendecomposition)

            Filtering: y = U g_theta(Lambda) U^T x
            g_theta(Lambda) = diag(g_theta(lambda_1), ..., g_theta(lambda_N))
        """,
        "gcn_approximation": """
            Full spectral: O(N^2) or O(N^3) for eigendecomposition
            Chebyshev: O(K|E|) using K-hop localized filter
            GCN: O(|E|) using 1-hop filter
        """
    }


def over_smoothing_problem():
    """
    The over-smoothing problem in deep GNNs.
    """
    return {
        "description": """
            As GNN depth increases, node representations converge to
            similar values, losing discriminative power.
        """,
        "cause": """
            Repeated averaging operations:
            h_v^{(K)} approx mean(h_u : u in K-hop neighborhood of v)

            If graph is connected, K large -> all nodes see same neighbors
        """,
        "solutions": {
            "residual_connections": "h' = h + GNNLayer(h)",
            "jumping_knowledge": "h_final = concat(h^{(0)}, h^{(1)}, ..., h^{(K)})",
            "dropout": "DropEdge: randomly drop edges during training",
            "normalization": "PairNorm, NodeNorm to preserve feature variance"
        }
    }


def gnn_expressivity():
    """
    Expressivity of GNNs and connection to WL test.
    """
    return {
        "weisfeiler_lehman_test": """
            WL test: Graph isomorphism test based on iterative color refinement

            1. Initialize: c_v^{(0)} = label(v)
            2. Iterate: c_v^{(k+1)} = hash(c_v^{(k)}, multiset({c_u^{(k)} : u in N(v)}))
            3. Compare: Graphs isomorphic if color histograms match
        """,
        "gnn_wl_connection": """
            Standard message-passing GNNs are at most as powerful as 1-WL test.

            Proof sketch:
            - GNN aggregation similar to WL color update
            - GNN can't distinguish graphs that 1-WL can't distinguish
        """,
        "more_powerful_gnns": {
            "GIN": "Graph Isomorphism Network, as powerful as 1-WL",
            "k-WL": "Higher-order WL tests, higher-order GNNs",
            "subgraph_GNN": "Include subgraph patterns for more power"
        }
    }


# =============================================================================
# Key Insights and Comparisons
# =============================================================================

GNN_COMPARISON = {
    "normalization": {
        "GCN": "Symmetric (degree-based): 1/sqrt(d_v * d_u)",
        "GAT": "Learned (attention): alpha_vu",
        "GraphSAGE": "Mean or max pooling"
    },
    "scalability": {
        "GCN": "Full-batch, all nodes needed",
        "GAT": "Full-batch or mini-batch with sampling",
        "GraphSAGE": "Mini-batch with sampling, highly scalable"
    },
    "inductive": {
        "GCN": "Transductive (needs all nodes at training)",
        "GAT": "Can be inductive",
        "GraphSAGE": "Designed for inductive learning"
    },
    "attention": {
        "GCN": "Fixed (degree-based)",
        "GAT": "Learned (but static in original)",
        "GraphSAGE": "None (mean/max aggregation)"
    }
}


GNN_APPLICATIONS = {
    "node_classification": {
        "task": "Predict label for each node",
        "examples": "Paper topic, user interest, protein function",
        "loss": "Cross-entropy on labeled nodes"
    },
    "link_prediction": {
        "task": "Predict if edge exists between nodes",
        "examples": "Friend recommendation, drug-target interaction",
        "method": "Score = dot(h_u, h_v) or MLP(concat(h_u, h_v))"
    },
    "graph_classification": {
        "task": "Predict label for entire graph",
        "examples": "Molecule property, program bug detection",
        "method": "Readout: pool node embeddings -> graph embedding"
    },
    "graph_generation": {
        "task": "Generate new graphs",
        "examples": "Drug design, circuit design",
        "methods": "Autoregressive, VAE-based, diffusion"
    }
}


GNN_INSIGHTS = {
    "locality": """
        GNNs respect graph structure through local message passing.
        K layers = K-hop neighborhood aggregation.
        This is both a strength (locality) and limitation (limited receptive field).
    """,

    "permutation_equivariance": """
        GNNs are permutation equivariant:
        If we reorder nodes, outputs are reordered the same way.
        This is essential for graphs (no canonical node ordering).
    """,

    "heterophily_challenge": """
        Most GNNs assume homophily (similar nodes connect).
        For heterophilic graphs (dissimilar nodes connect),
        standard GNNs struggle. Solutions: signed messages, separate
        ego and neighbor representations.
    """,

    "edge_features": """
        Basic GNNs don't use edge features.
        Extensions: Edge-conditioned convolution, message functions
        that depend on edge attributes.
    """
}
