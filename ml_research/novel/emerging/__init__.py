"""
Emerging Methods Research Index

This module contains research index entries for novel and emerging
neural network architectures that represent new directions in deep learning.

Methods included:
- KAN: Kolmogorov-Arnold Networks (2024)
- Liquid Neural Networks (2021)
- HyperNetworks (2016)
- Graph Neural Networks (GCN, GAT, GraphSAGE)
"""

from .kolmogorov_arnold import (
    get_method_info as get_kan_info,
    KAN_NETWORK,
    KANArchitecture,
)
from .liquid_networks import (
    get_method_info as get_liquid_info,
    LIQUID_NETWORK,
    LTCArchitecture,
)
from .hypernetworks import (
    get_method_info as get_hypernetwork_info,
    HYPERNETWORK,
    HyperNetworkArchitecture,
)
from .graph_neural_networks import (
    get_gnn_info,
    get_gcn_info,
    get_gat_info,
    get_graphsage_info,
    GNN_BASIC,
    GCN,
    GAT,
    GRAPHSAGE,
    GNNArchitecture,
)

__all__ = [
    # KAN
    "get_kan_info",
    "KAN_NETWORK",
    "KANArchitecture",
    # Liquid Networks
    "get_liquid_info",
    "LIQUID_NETWORK",
    "LTCArchitecture",
    # HyperNetworks
    "get_hypernetwork_info",
    "HYPERNETWORK",
    "HyperNetworkArchitecture",
    # Graph Neural Networks
    "get_gnn_info",
    "get_gcn_info",
    "get_gat_info",
    "get_graphsage_info",
    "GNN_BASIC",
    "GCN",
    "GAT",
    "GRAPHSAGE",
    "GNNArchitecture",
]
