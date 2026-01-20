"""
Neural Architecture Search (NAS) Fundamentals

Overview of NAS components: search space definition, search strategies,
and performance estimation methods. NAS automates the process of designing
neural network architectures, replacing human expertise with automated discovery.

Key Papers:
- "Neural Architecture Search with Reinforcement Learning" (Zoph & Le, 2017)
- "Learning Transferable Architectures for Scalable Image Recognition" (Zoph et al., 2018)

Mathematical Formulation:
    NAS as optimization:
        alpha* = argmax_alpha E_{D_val}[Accuracy(w*(alpha), alpha)]
        s.t. w*(alpha) = argmin_w E_{D_train}[Loss(w, alpha)]

    Where:
        - alpha: Architecture parameters (topology, operations)
        - w: Network weights
        - D_train, D_val: Training and validation datasets

    The search consists of three components:
        1. Search Space A: Set of possible architectures
        2. Search Strategy S: Method to explore A
        3. Performance Estimation P: Evaluate architectures efficiently
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

NAS_BASICS = MLMethod(
    method_id="nas_fundamentals",
    name="Neural Architecture Search Fundamentals",
    year=2017,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],

    authors=["Barret Zoph", "Quoc V. Le"],
    paper_title="Neural Architecture Search with Reinforcement Learning",
    paper_url="https://arxiv.org/abs/1611.01578",

    key_innovation=(
        "Introduced the paradigm of using machine learning to automate neural "
        "architecture design. Demonstrated that RL-based controllers can discover "
        "architectures that match or exceed human-designed networks."
    ),

    mathematical_formulation=r"""
NAS Optimization Problem:
    alpha* = argmax_{alpha in A} Accuracy(N(alpha, w*))
    s.t. w* = argmin_w Loss(N(alpha, w))

Search Space Definition:
    A = {alpha | alpha = (nodes, edges, operations)}
    - Nodes: Computation units (layers)
    - Edges: Connections between nodes
    - Operations: Transformations (conv, pool, identity, etc.)

Cell-Based Search Space:
    Architecture = Stack(Normal_Cell, Reduction_Cell)
    Cell = DAG(nodes, operations)

    Normal Cell: Maintains spatial dimensions
    Reduction Cell: Reduces spatial dimensions (stride=2)

Search Strategies:
    1. Evolution: alpha_{t+1} = Mutate(Select(Population_t))
    2. RL: alpha ~ pi_theta(a|s), theta <- theta + grad_theta J(theta)
    3. Gradient: alpha = alpha - lr * grad_alpha L_val(w*, alpha)
""",

    predecessors=["hyperparameter_optimization", "automl"],
    successors=["enas_2018", "darts_2018", "proxylessnas"],

    tags=["architecture-search", "automl", "meta-learning", "optimization"]
)

# Note: Original NAS required 800+ GPU days for CIFAR-10. Subsequent work focused
# on efficiency: ENAS achieved 1000x speedup via weight sharing, DARTS made
# search differentiable. NAS has discovered cells used in EfficientNet and
# other state-of-the-art architectures.


# =============================================================================
# Search Space Types
# =============================================================================

def search_space_types() -> Dict[str, Dict]:
    """
    Types of search spaces used in NAS.

    Returns:
        Dictionary describing different search space formulations
    """
    return {
        "macro_search_space": {
            "description": "Search entire network topology",
            "parameters": ["layer_types", "connections", "hyperparameters"],
            "complexity": "O(L^N) where L=operations, N=layers",
            "examples": ["Original NAS (2017)"],
            "pros": "Maximum flexibility",
            "cons": "Computationally expensive, hard to transfer"
        },
        "cell_based_search_space": {
            "description": "Search for cells, stack them to form network",
            "components": {
                "normal_cell": "Maintains feature map dimensions",
                "reduction_cell": "Reduces spatial dimensions by factor of 2"
            },
            "formula": "Network = Initial_Conv + Stack(N_cells) + Classifier",
            "complexity": "O(B * O^E) where B=blocks, O=ops, E=edges",
            "examples": ["NASNet", "ENAS", "DARTS"],
            "pros": "Transferable cells, reduced search space",
            "cons": "May miss novel macro architectures"
        },
        "hierarchical_search_space": {
            "description": "Multi-level search space with motifs",
            "levels": ["primitives", "motifs", "cells", "network"],
            "examples": ["Hierarchical NAS"],
            "pros": "Captures patterns at multiple scales"
        },
        "constrained_search_space": {
            "description": "Search space with hardware constraints",
            "constraints": ["latency", "FLOPs", "memory", "energy"],
            "examples": ["MnasNet", "FBNet"],
            "formula": "max Accuracy s.t. Latency < T"
        }
    }


def search_strategy_types() -> Dict[str, Dict]:
    """
    Types of search strategies used in NAS.

    Returns:
        Dictionary describing different search strategies
    """
    return {
        "random_search": {
            "description": "Randomly sample architectures from search space",
            "algorithm": "alpha ~ Uniform(A)",
            "pros": "Simple baseline, surprisingly effective",
            "cons": "Inefficient for large search spaces",
            "note": "Random search often competitive with sophisticated methods"
        },
        "reinforcement_learning": {
            "description": "Train controller to generate architectures",
            "algorithm": """
                Controller RNN generates sequence of actions (architecture)
                alpha = Controller(theta)
                R = Accuracy(alpha)
                theta <- theta + alpha_lr * grad_theta E[R]
            """,
            "methods": ["REINFORCE", "PPO"],
            "examples": ["Original NAS", "NASNet", "ENAS"],
            "pros": "Can optimize non-differentiable objectives",
            "cons": "High variance, sample inefficient"
        },
        "evolutionary_algorithms": {
            "description": "Evolve population of architectures",
            "algorithm": """
                Initialize population P_0
                For t = 1 to T:
                    Select parents from P_{t-1}
                    Offspring = Mutate(Crossover(Parents))
                    P_t = Select(P_{t-1} + Offspring)
            """,
            "operators": {
                "mutation": "Change operation, add/remove edge",
                "crossover": "Combine parts from two architectures"
            },
            "examples": ["AmoebaNet", "Regularized Evolution"],
            "pros": "Simple, parallelizable, robust",
            "cons": "May be sample inefficient"
        },
        "gradient_based": {
            "description": "Make architecture search differentiable",
            "algorithm": """
                Relax discrete choices to continuous:
                o(x) = sum_i alpha_i * o_i(x), alpha = softmax(beta)

                Bi-level optimization:
                    Inner: w* = argmin_w L_train(w, alpha)
                    Outer: alpha* = argmin_alpha L_val(w*, alpha)
            """,
            "examples": ["DARTS", "ProxylessNAS", "SNAS"],
            "pros": "Fast search (single GPU days)",
            "cons": "Memory intensive, may collapse to suboptimal"
        },
        "one_shot": {
            "description": "Train supernet containing all architectures",
            "algorithm": """
                Train supernet with all possible paths
                Evaluate subnetworks by inheriting weights
            """,
            "examples": ["SMASH", "Understanding NAS"],
            "pros": "Decouples training and search",
            "cons": "Weight sharing may affect ranking fidelity"
        }
    }


def performance_estimation_types() -> Dict[str, Dict]:
    """
    Types of performance estimation strategies in NAS.

    Returns:
        Dictionary describing different performance estimation methods
    """
    return {
        "full_training": {
            "description": "Train each architecture to convergence",
            "cost": "Extremely high (days per architecture)",
            "accuracy": "Gold standard",
            "examples": ["Original NAS"]
        },
        "early_stopping": {
            "description": "Train for fewer epochs, extrapolate final performance",
            "techniques": ["Learning curve prediction", "Successive halving"],
            "cost": "Reduced by factor of epochs",
            "examples": ["Hyperband"]
        },
        "weight_sharing": {
            "description": "Share weights across architectures in supernet",
            "algorithm": """
                Train supernet containing all architectures
                For evaluation: sample path, use inherited weights
            """,
            "speedup": "1000x+ over full training",
            "concern": "Ranking correlation with true performance",
            "examples": ["ENAS", "DARTS", "One-shot NAS"]
        },
        "proxy_tasks": {
            "description": "Evaluate on simpler related task",
            "approaches": [
                "Smaller dataset (subset of CIFAR)",
                "Smaller network (fewer cells/filters)",
                "Lower resolution images"
            ],
            "examples": ["ProxylessNAS", "FBNet"]
        },
        "performance_prediction": {
            "description": "Train predictor to estimate architecture performance",
            "model_types": ["MLP", "GNN", "Bayesian optimization"],
            "features": ["Architecture encoding", "Network statistics"],
            "examples": ["NAO", "BANANAS"]
        },
        "zero_cost_proxies": {
            "description": "Evaluate at initialization without training",
            "metrics": [
                "Gradient norm",
                "Jacobian covariance",
                "Number of linear regions",
                "Fisher information"
            ],
            "cost": "Seconds per architecture",
            "examples": ["NASWOT", "SynFlow"]
        }
    }


# =============================================================================
# Architecture Representations
# =============================================================================

@dataclass
class CellStructure:
    """Reference structure for cell-based NAS."""

    num_nodes: int = 4  # Intermediate nodes in cell
    num_input_nodes: int = 2  # Previous cell outputs
    num_ops: int = 7  # Number of operation choices

    @staticmethod
    def operations() -> List[str]:
        """Standard operations in cell-based search."""
        return [
            "identity",           # Skip connection
            "3x3_conv",           # Standard convolution
            "5x5_conv",           # Larger receptive field
            "3x3_separable_conv", # Depthwise separable
            "5x5_separable_conv", # Depthwise separable
            "3x3_max_pool",       # Max pooling
            "3x3_avg_pool",       # Average pooling
            "zero",               # No connection (DARTS)
        ]

    @staticmethod
    def cell_representation() -> str:
        """Cell as directed acyclic graph."""
        return """
Cell as DAG:
    Input nodes: h_{k-1}, h_{k-2} (previous two cell outputs)

    Intermediate nodes: For each node j in {2, 3, ..., N+1}:
        h_j = sum_{i < j} o_{i,j}(h_i)

        where o_{i,j} is operation on edge (i, j)

    Output: Concatenate all intermediate node outputs
        h_k = concat(h_2, h_3, ..., h_{N+1})

Example Cell (N=4):
    h_0 ----[op01]----> h_2
    h_1 ----[op12]----> h_2
    h_0 ----[op02]----> h_3
    h_2 ----[op23]----> h_3
    ...
    h_k = concat(h_2, h_3, h_4, h_5)
"""


# =============================================================================
# NAS Benchmark Datasets
# =============================================================================

NAS_BENCHMARKS = {
    "nasbench_101": {
        "description": "First tabular NAS benchmark",
        "search_space": "423k unique architectures",
        "dataset": "CIFAR-10",
        "metrics": "Accuracy at 4, 12, 36, 108 epochs",
        "paper": "NAS-Bench-101: Towards Reproducible Neural Architecture Search",
        "year": 2019
    },
    "nasbench_201": {
        "description": "Smaller search space, multiple datasets",
        "search_space": "15,625 architectures",
        "datasets": ["CIFAR-10", "CIFAR-100", "ImageNet16-120"],
        "paper": "NAS-Bench-201: Extending the Scope of Reproducible NAS Research",
        "year": 2020
    },
    "nasbench_301": {
        "description": "Surrogate benchmark for DARTS space",
        "search_space": "10^18 architectures (DARTS space)",
        "approach": "Ensemble of surrogate models",
        "paper": "NAS-Bench-301: Benchmarking NAS on DARTS Search Space",
        "year": 2020
    },
    "hw_nasbench": {
        "description": "Hardware-aware NAS benchmark",
        "metrics": ["Accuracy", "Latency", "Energy"],
        "devices": ["GPU", "TPU", "Mobile", "FPGA"],
        "paper": "HW-NAS-Bench: Hardware-Aware NAS Benchmark",
        "year": 2021
    }
}
