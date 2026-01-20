"""
Novel ML Methods Module

This module contains research indices for novel/emerging ML architectures
that represent significant departures from traditional approaches.

Submodules:
    - state_space: State space sequence models (S4, Mamba, Jamba)
    - mixture_of_experts: Conditional computation via expert routing
    - world_models: Latent dynamics models for imagination and planning
    - emerging: Novel architectures (KAN, Liquid Networks, HyperNets, GNNs)

State Space Models:
    A paradigm for sequence modeling based on continuous-time dynamical systems.
    Key advantage: O(L) complexity vs O(L^2) for attention.

    x'(t) = Ax(t) + Bu(t)    (state dynamics)
    y(t)  = Cx(t) + Du(t)    (observation)

    Evolution:
        S4 (2021) -> Mamba (2023) -> Jamba (2024)

Mixture of Experts:
    Conditional computation that scales parameters without scaling compute.
    Key idea: route inputs to specialized expert networks.

    y = sum_i g(x)_i * Expert_i(x)

    Evolution:
        MoE (1991) -> Switch Transformer (2021) -> Mixtral (2023)

World Models:
    Learn latent dynamics for model-based RL and imagination.
    Key idea: learn world model, then plan/learn in latent space.

    z_t = Encode(o_t)
    z_{t+1} = Dynamics(z_t, a_t)

    Evolution:
        World Models (2018) -> Dreamer (2019-2023) -> Genie (2024)

Emerging Methods:
    Novel architectures exploring new paradigms:
    - KAN (2024): Learnable activation functions on edges
    - Liquid Networks (2021): Continuous-time variable dynamics
    - HyperNetworks (2016): Networks generating weights for other networks
    - Graph Neural Networks: Message passing on graph-structured data

These represent the frontier of efficient large-scale sequence modeling,
combining insights from multiple paradigms in hybrid architectures.
"""

# State Space Models
from .state_space import (
    # S4
    STRUCTURED_STATE_SPACE,
    state_space_continuous,
    discretize_bilinear,
    compute_ssm_kernel,
    hippo_matrix,
    dplr_representation,
    S4Architecture,
    S4_VARIANTS,
    # Mamba
    MAMBA,
    selective_scan,
    selection_mechanism,
    parallel_associative_scan,
    mamba_block,
    MambaArchitecture,
    MAMBA_VARIANTS,
    MAMBA_INSIGHTS,
    # Jamba
    JAMBA,
    jamba_layer_pattern,
    hybrid_kv_cache,
    jamba_moe_configuration,
    jamba_attention_layer,
    JambaArchitecture,
    JAMBA_COMPARISONS,
    JAMBA_DESIGN_RATIONALE,
)

# Mixture of Experts
from .mixture_of_experts import (
    # MoE Basics
    MIXTURE_OF_EXPERTS,
    dense_moe_forward,
    sparse_moe_forward,
    gating_function,
    load_balancing_loss,
    expert_capacity,
    MoEArchitecture,
    MOE_CONCEPTS,
    MOE_HISTORY,
    # Switch Transformer
    SWITCH_TRANSFORMER,
    switch_routing,
    expert_capacity_computation,
    load_balancing_loss_switch,
    selective_precision,
    switch_layer_forward,
    SwitchTransformerArchitecture,
    SWITCH_SCALING,
    SWITCH_IMPLEMENTATION,
    # Mixtral
    MIXTRAL,
    mixtral_routing,
    mixtral_expert_ffn,
    sliding_window_attention,
    mixtral_moe_layer,
    MixtralArchitecture,
    MIXTRAL_BENCHMARKS,
    MIXTRAL_VARIANTS,
    MIXTRAL_DEPLOYMENT,
)

# World Models
from .world_models import (
    # World Model Basics
    get_world_model_info,
    WORLD_MODEL,
    VMCArchitecture,
    # Dreamer
    get_dreamer_v1_info,
    get_dreamer_v2_info,
    get_dreamer_v3_info,
    DREAMER_V1,
    DREAMER_V2,
    DREAMER_V3,
    RSSMArchitecture,
    # Genie
    get_genie_info,
    GENIE,
    GenieArchitecture,
)

# Emerging Methods
from .emerging import (
    # KAN
    get_kan_info,
    KAN_NETWORK,
    KANArchitecture,
    # Liquid Networks
    get_liquid_info,
    LIQUID_NETWORK,
    LTCArchitecture,
    # HyperNetworks
    get_hypernetwork_info,
    HYPERNETWORK,
    HyperNetworkArchitecture,
    # Graph Neural Networks
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
    # =========================================================================
    # State Space Models
    # =========================================================================
    # S4: Structured State Spaces
    "STRUCTURED_STATE_SPACE",
    "state_space_continuous",
    "discretize_bilinear",
    "compute_ssm_kernel",
    "hippo_matrix",
    "dplr_representation",
    "S4Architecture",
    "S4_VARIANTS",
    # Mamba: Selective State Spaces
    "MAMBA",
    "selective_scan",
    "selection_mechanism",
    "parallel_associative_scan",
    "mamba_block",
    "MambaArchitecture",
    "MAMBA_VARIANTS",
    "MAMBA_INSIGHTS",
    # Jamba: Hybrid Architecture
    "JAMBA",
    "jamba_layer_pattern",
    "hybrid_kv_cache",
    "jamba_moe_configuration",
    "jamba_attention_layer",
    "JambaArchitecture",
    "JAMBA_COMPARISONS",
    "JAMBA_DESIGN_RATIONALE",
    # =========================================================================
    # Mixture of Experts
    # =========================================================================
    # MoE Basics
    "MIXTURE_OF_EXPERTS",
    "dense_moe_forward",
    "sparse_moe_forward",
    "gating_function",
    "load_balancing_loss",
    "expert_capacity",
    "MoEArchitecture",
    "MOE_CONCEPTS",
    "MOE_HISTORY",
    # Switch Transformer
    "SWITCH_TRANSFORMER",
    "switch_routing",
    "expert_capacity_computation",
    "load_balancing_loss_switch",
    "selective_precision",
    "switch_layer_forward",
    "SwitchTransformerArchitecture",
    "SWITCH_SCALING",
    "SWITCH_IMPLEMENTATION",
    # Mixtral
    "MIXTRAL",
    "mixtral_routing",
    "mixtral_expert_ffn",
    "sliding_window_attention",
    "mixtral_moe_layer",
    "MixtralArchitecture",
    "MIXTRAL_BENCHMARKS",
    "MIXTRAL_VARIANTS",
    "MIXTRAL_DEPLOYMENT",
    # =========================================================================
    # World Models
    # =========================================================================
    # World Model Basics
    "get_world_model_info",
    "WORLD_MODEL",
    "VMCArchitecture",
    # Dreamer
    "get_dreamer_v1_info",
    "get_dreamer_v2_info",
    "get_dreamer_v3_info",
    "DREAMER_V1",
    "DREAMER_V2",
    "DREAMER_V3",
    "RSSMArchitecture",
    # Genie
    "get_genie_info",
    "GENIE",
    "GenieArchitecture",
    # =========================================================================
    # Emerging Methods
    # =========================================================================
    # KAN: Kolmogorov-Arnold Networks
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
