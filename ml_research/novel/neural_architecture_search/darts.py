"""
DARTS: Differentiable Architecture Search - 2018

Liu et al.'s gradient-based approach to NAS. Key insight: relax the discrete
architecture search space to be continuous, allowing gradient descent on
architecture parameters. Uses bi-level optimization to jointly learn
architecture and network weights.

Paper: "DARTS: Differentiable Architecture Search"
arXiv: 1806.09055

Mathematical Formulation:
    Bi-level optimization:
        min_alpha L_val(w*(alpha), alpha)
        s.t. w*(alpha) = argmin_w L_train(w, alpha)

    Continuous relaxation:
        Instead of selecting one operation, compute weighted sum:
        o_bar(x) = sum_i (exp(alpha_i) / sum_j exp(alpha_j)) * o_i(x)
                 = sum_i softmax(alpha)_i * o_i(x)

    Where:
        - alpha: Architecture parameters (mixing weights)
        - w: Network weights
        - o_i: Candidate operations (conv, pool, etc.)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

DARTS = MLMethod(
    method_id="darts_2018",
    name="DARTS: Differentiable Architecture Search",
    year=2018,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],

    authors=["Hanxiao Liu", "Karen Simonyan", "Yiming Yang"],
    paper_title="DARTS: Differentiable Architecture Search",
    paper_url="https://arxiv.org/abs/1806.09055",

    key_innovation=(
        "Made NAS fully differentiable by relaxing discrete operation choices "
        "to continuous mixing weights. This allows using gradient descent on "
        "architecture parameters, reducing search cost to single GPU-days. "
        "Introduced bi-level optimization for joint architecture-weight learning."
    ),

    mathematical_formulation=r"""
Continuous Relaxation:
    Instead of selecting single operation, mix all operations:

    o_bar^{(i,j)}(x) = sum_{o in O} alpha_o^{(i,j)} * o(x)

    Where alpha_o^{(i,j)} = exp(beta_o^{(i,j)}) / sum_{o'} exp(beta_{o'}^{(i,j)})

    beta: Learnable architecture parameters (before softmax)

Cell as DAG:
    Each cell is a DAG with N nodes:

    x^{(j)} = sum_{i<j} o_bar^{(i,j)}(x^{(i)})

    Each edge (i,j) has mixed operation o_bar^{(i,j)}

Bi-level Optimization:
    Outer (architecture):
        min_alpha L_val(w*(alpha), alpha)

    Inner (weights):
        w*(alpha) = argmin_w L_train(w, alpha)

    Approximate solution via alternating gradient descent

Gradient Approximation:
    grad_alpha L_val(w*(alpha), alpha)

    First-order approximation:
        approx grad_alpha L_val(w - xi * grad_w L_train(w, alpha), alpha)

    Second-order approximation (finite difference):
        grad_alpha L_val(w*, alpha) - xi * grad^2_{alpha,w} L_train(w, alpha) * grad_w L_val(w*, alpha)

        approx [grad_alpha L_val(w^+, alpha) - grad_alpha L_val(w^-, alpha)] / (2 * epsilon)

        where w^+/- = w +/- epsilon * grad_w L_val(w, alpha)

Discretization (after search):
    For each edge (i,j): keep only strongest operation
    o^{(i,j)} = argmax_{o != zero} alpha_o^{(i,j)}

    For each node j: keep top-k input edges by alpha strength
""",

    predecessors=["enas_2018", "nas_2017"],
    successors=["pcdarts", "gdas", "darts_plus"],

    tags=["architecture-search", "differentiable", "gradient-based", "bi-level-optimization"]
)

# Note: DARTS popularized gradient-based NAS and inspired many follow-up works.
# However, it suffers from issues like collapse to skip connections and
# instability. Variants like PC-DARTS, DARTS+, and Fair DARTS address
# these limitations. Search cost: ~1 GPU-day on CIFAR-10.


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def bilevel_optimization() -> Dict[str, str]:
    """
    Bi-level optimization in DARTS.

    Returns:
        Dictionary describing the bi-level optimization procedure
    """
    return {
        "formulation": """
Bi-level Optimization Problem:
    Upper level (architecture): min_alpha L_val(w*(alpha), alpha)
    Lower level (weights):      w*(alpha) = argmin_w L_train(w, alpha)

The optimal weights w* depend on architecture alpha.
We need gradient of validation loss w.r.t. alpha, accounting for this dependence.
""",
        "first_order_approximation": """
First-Order Approximation:
    Assume w*(alpha) approx w - xi * grad_w L_train(w, alpha)

    Then:
        grad_alpha L_val(w*(alpha), alpha)
        approx grad_alpha L_val(w - xi * grad_w L_train(w, alpha), alpha)

    Setting xi = 0 gives:
        grad_alpha L_val(w, alpha)

    This ignores the dependency of w* on alpha but is fast and often works well.
""",
        "second_order_approximation": """
Second-Order Approximation:
    Using chain rule:
        grad_alpha L_val(w*, alpha) = grad_alpha L_val(w*, alpha)|_w_fixed
                                    - xi * grad_alpha (grad_w L_train * grad_w L_val)

    The second term accounts for how alpha affects w*.

    Finite difference approximation:
        Let epsilon be small
        w^+ = w + epsilon * grad_w L_val(w, alpha)
        w^- = w - epsilon * grad_w L_val(w, alpha)

        grad_alpha L_val(w*, alpha) approx
            [grad_alpha L_train(w^+, alpha) - grad_alpha L_train(w^-, alpha)] / (2 * epsilon)

    This requires 2 extra forward passes but is more accurate.
""",
        "alternating_optimization": """
DARTS Training Algorithm:
    Initialize: alpha (architecture), w (weights)

    while not converged:
        # Step 1: Update weights w (inner optimization)
        w = w - lr_w * grad_w L_train(w, alpha)

        # Step 2: Update architecture alpha (outer optimization)
        # Using first-order approximation:
        alpha = alpha - lr_alpha * grad_alpha L_val(w, alpha)

        # Or using second-order approximation:
        w_plus = w + epsilon * grad_w L_val
        w_minus = w - epsilon * grad_w L_val
        alpha_grad = (grad_alpha L_train(w_plus) - grad_alpha L_train(w_minus)) / (2*epsilon)
        alpha = alpha - lr_alpha * alpha_grad
"""
    }


def continuous_relaxation() -> Dict[str, str]:
    """
    Continuous relaxation of discrete architecture choices.

    Returns:
        Dictionary describing the relaxation
    """
    return {
        "discrete_choice": """
Discrete Architecture (standard):
    On each edge (i,j), select ONE operation from candidate set O:
    o^{(i,j)}(x) = o_k(x) for some k in {1, ..., |O|}

    Search space is discrete and non-differentiable.
""",
        "continuous_relaxation": """
Continuous Relaxation (DARTS):
    On each edge (i,j), compute weighted sum of ALL operations:

    o_bar^{(i,j)}(x) = sum_{k=1}^{|O|} alpha_k^{(i,j)} * o_k(x)

    where alpha_k^{(i,j)} = softmax(beta^{(i,j)})_k
                          = exp(beta_k^{(i,j)}) / sum_l exp(beta_l^{(i,j)})

    beta^{(i,j)} are learnable parameters (logits before softmax)
""",
        "gradient_flow": """
Gradients Flow Through Mixing Weights:
    d(L)/d(beta_k^{(i,j)}) = sum_x d(L)/d(o_bar(x)) * d(o_bar(x))/d(alpha_k) * d(alpha_k)/d(beta_k)

    d(alpha_k)/d(beta_k) = alpha_k * (1 - alpha_k)  (softmax derivative)
    d(o_bar)/d(alpha_k) = o_k(x)

    This allows gradient descent on architecture parameters beta.
""",
        "operations": """
Candidate Operations O (DARTS CIFAR-10):
    1. 3x3 separable convolution
    2. 5x5 separable convolution
    3. 3x3 dilated separable convolution
    4. 5x5 dilated separable convolution
    5. 3x3 max pooling
    6. 3x3 average pooling
    7. skip connection (identity)
    8. zero (no connection)

    The 'zero' operation allows edges to be removed.
"""
    }


def darts_discretization() -> Dict[str, str]:
    """
    Discretization procedure after DARTS search.

    Returns:
        Dictionary describing discretization
    """
    return {
        "procedure": """
After Search Discretization:

1. For each edge (i,j):
   - Select operation with largest alpha (excluding 'zero'):
     o^{(i,j)} = argmax_{o != zero} alpha_o^{(i,j)}

2. For each intermediate node j:
   - Keep only top-2 incoming edges by alpha strength
   - Remove other edges

3. Cell structure:
   - Output = concat(all intermediate nodes not used as inputs)

Example:
    After search, edge (1,3) has: alpha = [0.8_conv3, 0.1_conv5, 0.05_pool, 0.05_zero]
    Selected operation: 3x3 conv (highest non-zero alpha)
""",
        "final_architecture": """
Building Final Architecture:
    1. Take discretized cells (normal and reduction)
    2. Stack cells to form full network:

       Input -> Stem (3x3 conv)
             -> Normal Cell x N
             -> Reduction Cell (stride 2)
             -> Normal Cell x N
             -> Reduction Cell (stride 2)
             -> Normal Cell x N
             -> Global Average Pool
             -> Classifier

    3. Train from scratch with:
       - More channels
       - Longer training
       - Additional regularization (cutout, drop-path)
"""
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class DARTSArchitecture:
    """Reference architecture for DARTS."""

    num_cells: int = 20  # Cells in final network
    init_channels: int = 36  # Initial channels
    num_nodes: int = 4  # Intermediate nodes per cell

    @staticmethod
    def search_space() -> str:
        """DARTS search space definition."""
        return """
DARTS Search Space:

Cell Structure:
    - 2 input nodes: outputs of previous two cells
    - 4 intermediate nodes: computed from previous nodes
    - 1 output node: concatenation of all intermediate nodes

    Each edge (i,j) has 8 candidate operations:
        O = {3x3_sep_conv, 5x5_sep_conv, 3x3_dil_conv, 5x5_dil_conv,
             3x3_max_pool, 3x3_avg_pool, skip_connect, zero}

Architecture Parameters:
    For edge (i,j): beta^{(i,j)} in R^8 (one weight per operation)
    Total parameters:
        - Normal cell: 14 edges x 8 ops = 112 parameters
        - Reduction cell: 14 edges x 8 ops = 112 parameters
        - Total: 224 architecture parameters

Search Cost:
    - 1.5 GPU-days on CIFAR-10
    - Uses proxy: 8 cells, 16 initial channels
"""

    @staticmethod
    def cell_structure() -> str:
        """Cell computation in DARTS."""
        return """
DARTS Cell Computation:

Input: h_{k-2}, h_{k-1} (outputs of previous two cells)

For normal cell: both inputs same resolution
For reduction cell: h_{k-2} has 2x spatial dim, apply stride-2 conv

Intermediate Nodes:
    x^{(0)} = h_{k-2}  (after potential reduction)
    x^{(1)} = h_{k-1}

    For j = 2, 3, 4, 5:
        x^{(j)} = sum_{i<j} o_bar^{(i,j)}(x^{(i)})

        where o_bar^{(i,j)}(x) = sum_{o in O} softmax(beta_o^{(i,j)}) * o(x)

Output:
    h_k = concat(x^{(2)}, x^{(3)}, x^{(4)}, x^{(5)})

    Then 1x1 conv to adjust channels: h_k = conv1x1(h_k)

Memory Footprint:
    During search, all operations computed for each edge (high memory)
    Strategies to reduce: partial channel connection (PC-DARTS)
"""

    @staticmethod
    def training_configuration() -> Dict[str, any]:
        """DARTS training hyperparameters."""
        return {
            "search_phase": {
                "epochs": 50,
                "batch_size": 64,
                "weight_lr": 0.025,
                "weight_momentum": 0.9,
                "weight_decay": 3e-4,
                "arch_lr": 3e-4,
                "arch_weight_decay": 1e-3,
                "cells": 8,
                "init_channels": 16
            },
            "evaluation_phase": {
                "epochs": 600,
                "batch_size": 96,
                "learning_rate": 0.025,
                "momentum": 0.9,
                "weight_decay": 3e-4,
                "cells": 20,
                "init_channels": 36,
                "auxiliary_weight": 0.4,
                "drop_path_prob": 0.2,
                "cutout_length": 16
            }
        }


# =============================================================================
# Known Issues and Variants
# =============================================================================

DARTS_ISSUES = {
    "skip_connection_collapse": {
        "description": "DARTS tends to select too many skip connections",
        "reason": "Skip connections have no learnable parameters, easier to optimize",
        "symptom": "Searched architecture has most edges as skip connections",
        "solutions": ["Early stopping", "Auxiliary skip penalty (Fair DARTS)", "Progressive DARTS"]
    },
    "performance_collapse": {
        "description": "Search performance doesn't correlate with final performance",
        "reason": "Bi-level approximation errors, depth gap between search and eval",
        "solutions": ["DARTS+", "P-DARTS", "Search on larger proxy"]
    },
    "high_memory": {
        "description": "Computing all operations per edge is memory intensive",
        "reason": "O(|O| * edges * batch_size * channels * H * W)",
        "solutions": ["PC-DARTS (partial channels)", "ProxylessNAS (binary gates)"]
    },
    "instability": {
        "description": "Search results vary significantly across runs",
        "solutions": ["DARTS- (early stopping)", "Perturbation-based regularization"]
    }
}

DARTS_VARIANTS = {
    "pcdarts": {
        "full_name": "Partially-Connected DARTS",
        "innovation": "Compute mixed operations on partial channels only",
        "benefit": "Reduced memory, faster search, more stable"
    },
    "pdarts": {
        "full_name": "Progressive DARTS",
        "innovation": "Gradually increase depth during search",
        "benefit": "Addresses depth gap between search and evaluation"
    },
    "fair_darts": {
        "innovation": "Exclusive competition between skip and other ops",
        "benefit": "Prevents skip connection collapse"
    },
    "sdarts": {
        "full_name": "Stabilizing DARTS",
        "innovation": "Regularization via perturbation of architecture weights",
        "benefit": "More stable, better generalization"
    },
    "gdas": {
        "full_name": "Searching for A Robust Neural Architecture in Four GPU Hours",
        "innovation": "Gumbel-softmax for differentiable discrete sampling",
        "benefit": "Single path during search (memory efficient)"
    }
}


# =============================================================================
# Results
# =============================================================================

DARTS_RESULTS = {
    "cifar10": {
        "test_error": "2.76% +/- 0.09%",
        "params": "3.3M",
        "search_cost": "1.5 GPU-days",
        "note": "With cutout regularization"
    },
    "cifar10_second_order": {
        "test_error": "2.83% +/- 0.06%",
        "note": "Using second-order approximation"
    },
    "imagenet": {
        "top1_error": "26.7%",
        "top5_error": "8.7%",
        "params": "4.7M",
        "flops": "574M",
        "note": "Mobile setting, transferred from CIFAR-10"
    },
    "comparison": {
        "enas": {"error": "2.89%", "search_cost": "0.45 GPU-days"},
        "nasnet": {"error": "2.65%", "search_cost": "1800 GPU-days"},
        "amoebanet": {"error": "2.55%", "search_cost": "3150 GPU-days"},
        "darts": {"error": "2.76%", "search_cost": "1.5 GPU-days"}
    }
}
