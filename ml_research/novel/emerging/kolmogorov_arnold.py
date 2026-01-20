"""
KAN: Kolmogorov-Arnold Networks - Liu et al. (2024)

A novel neural network architecture where learnable activation functions
are placed on edges (weights) rather than nodes, based on the
Kolmogorov-Arnold representation theorem.

Paper: "KAN: Kolmogorov-Arnold Networks"
arXiv: 2404.19756

Kolmogorov-Arnold Representation Theorem:
    Any continuous multivariate function f: [0,1]^n -> R can be written as:

    f(x_1, ..., x_n) = sum_{q=0}^{2n} Phi_q( sum_{p=1}^{n} phi_{q,p}(x_p) )

    Where phi_{q,p}: [0,1] -> R and Phi_q: R -> R are continuous univariate functions.

Key Innovation:
    Instead of fixed activation functions on nodes (MLPs),
    KANs learn activation functions on edges using B-splines.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

KAN_NETWORK = MLMethod(
    method_id="kan_2024",
    name="Kolmogorov-Arnold Networks",
    year=2024,

    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.PERCEPTRON_LINE],

    authors=[
        "Ziming Liu", "Yixuan Wang", "Sachin Vaidya", "Fabian Ruehle",
        "James Halverson", "Marin Soljacic", "Thomas Y. Hou", "Max Tegmark"
    ],
    paper_title="KAN: Kolmogorov-Arnold Networks",
    paper_url="https://arxiv.org/abs/2404.19756",

    key_innovation=(
        "Replaces fixed activation functions on nodes (as in MLPs) with learnable "
        "activation functions on edges, parameterized as B-splines. Based on the "
        "Kolmogorov-Arnold representation theorem. KANs can achieve comparable or "
        "better accuracy than MLPs with fewer parameters, and the learned functions "
        "are often interpretable, revealing symbolic formulas."
    ),

    mathematical_formulation=r"""
Kolmogorov-Arnold Representation Theorem:
    f(x_1, ..., x_n) = sum_{q=0}^{2n} Phi_q( sum_{p=1}^{n} phi_{q,p}(x_p) )

    This states ANY continuous function can be represented by
    compositions and sums of univariate functions.

KAN Layer:
    Input: x in R^{n_in}
    Output: y in R^{n_out}

    y_j = sum_{i=1}^{n_in} phi_{i,j}(x_i)

    Where phi_{i,j}: R -> R is a learnable univariate function

    Compare to MLP layer:
        y_j = sigma( sum_i w_{i,j} * x_i + b_j )
        Fixed activation sigma, learned weights w

Spline Parameterization:
    phi(x) = w_b * b(x) + w_s * spline(x)

    Where:
        b(x) = silu(x) = x / (1 + exp(-x))  # Base function
        spline(x) = sum_i c_i * B_i(x)      # B-spline
        B_i(x): B-spline basis functions
        c_i: Learnable coefficients

Full KAN:
    x^{(0)} = input
    x^{(l+1)}_j = sum_i phi^{(l)}_{i,j}(x^{(l)}_i)

    KAN(x) = (Phi_{L-1} o ... o Phi_1 o Phi_0)(x)
""",

    predecessors=["mlp", "spline_networks"],
    successors=[],

    tags=[
        "kolmogorov-arnold", "splines", "interpretable", "activation-functions",
        "symbolic-regression", "function-approximation"
    ],
    notes=(
        "KANs offer a fundamentally different approach to function approximation. "
        "Key advantages: 1) Better scaling (accuracy improves faster with parameters), "
        "2) Interpretability (learned splines reveal functional form), "
        "3) Symbolic regression (can extract equations). "
        "Limitations: slower training than MLPs, more hyperparameters."
    )
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for KAN."""
    return KAN_NETWORK


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class KANArchitecture:
    """Reference architecture for Kolmogorov-Arnold Networks."""

    # Layer structure
    layer_widths: List[int] = field(default_factory=lambda: [2, 5, 1])

    # Spline parameters
    spline_order: int = 3  # Cubic B-splines
    num_grid_points: int = 5  # Number of grid intervals
    grid_range: tuple = (-1, 1)

    @staticmethod
    def kan_layer_structure() -> str:
        """KAN layer structure."""
        return """
KAN Layer:
    Input: x in R^{n_in}
    Output: y in R^{n_out}

    For each output j in [1, n_out]:
        y_j = sum_{i=1}^{n_in} phi_{i,j}(x_i)

    Total learnable functions: n_in * n_out
    Each phi_{i,j} parameterized by ~(num_grid_points + spline_order) coefficients

MLP Layer (for comparison):
    y_j = sigma( sum_i w_{i,j} * x_i + b_j )

    Total parameters: n_in * n_out + n_out
    Fixed activation sigma

Key Difference:
    MLP: Linear combination -> Fixed nonlinearity
    KAN: Sum of learned nonlinearities -> Identity

    Both are universal approximators, but KANs learn the
    nonlinearities that best fit the data.
"""

    @staticmethod
    def spline_parameterization() -> str:
        """B-spline parameterization for phi functions."""
        return """
B-Spline Parameterization:

For each edge function phi_{i,j}:

    phi(x) = w_b * b(x) + w_s * spline(x)

Base function (residual connection):
    b(x) = silu(x) = x * sigmoid(x)

B-spline component:
    spline(x) = sum_{k=0}^{G+K-1} c_k * B_{k,K}(x)

    Where:
        G = number of grid intervals
        K = spline order (typically 3 for cubic)
        B_{k,K}(x) = B-spline basis function
        c_k = learnable coefficients

B-spline basis (Cox-de Boor recursion):
    B_{k,0}(x) = 1 if t_k <= x < t_{k+1} else 0

    B_{k,n}(x) = (x - t_k)/(t_{k+n} - t_k) * B_{k,n-1}(x)
               + (t_{k+n+1} - x)/(t_{k+n+1} - t_{k+1}) * B_{k+1,n-1}(x)

    Where t_k are knot positions

Grid Extension (for extrapolation):
    When x falls outside training range, grid is extended
    to maintain smooth extrapolation behavior.
"""

    @staticmethod
    def training_procedure() -> str:
        """KAN training procedure."""
        return """
KAN Training Procedure:

1. Initialization:
    - Initialize grid points uniformly in [grid_min, grid_max]
    - Initialize spline coefficients near identity function
    - Set w_b = 0, w_s = 1 initially

2. Forward Pass:
    for layer in KAN.layers:
        x_next = zeros(layer.n_out)
        for j in range(layer.n_out):
            for i in range(layer.n_in):
                x_next[j] += layer.phi[i,j](x[i])
        x = x_next

3. Backward Pass:
    Standard backpropagation through spline functions
    d_loss/d_c_k = d_loss/d_phi * d_phi/d_spline * d_spline/d_c_k

4. Grid Update (periodically):
    Adapt grid points based on data distribution
    More grid points where function varies quickly
    Fewer where function is smooth

5. Regularization:
    L1 on spline coefficients for smoothness
    L1 on w_s for sparsity (prune unnecessary edges)

6. Pruning and Simplification:
    Remove edges with small w_s
    Symbolically fit simple functions to learned splines
    Replace splines with symbolic forms when possible
"""

    @staticmethod
    def symbolic_regression() -> str:
        """Symbolic regression from KAN."""
        return """
Symbolic Regression (Interpretability):

After training, extract symbolic formulas:

1. For each edge phi_{i,j}:
    - Visualize the learned spline
    - Fit candidate symbolic functions:
      * Linear: a*x + b
      * Polynomial: sum_k a_k * x^k
      * Trigonometric: a*sin(b*x + c)
      * Exponential: a*exp(b*x)
      * Logarithmic: a*log(x + b)
    - Select best fit (lowest error + simplicity)

2. Compose layer-by-layer:
    - Substitute symbolic forms
    - Simplify algebraically
    - Result: closed-form expression

Example:
    Original: f(x, y) = exp(sin(pi*x) + y^2)

    KAN learns:
        phi_1(x) ≈ sin(pi*x)
        phi_2(y) ≈ y^2
        Phi(z) ≈ exp(z)

    Composed: f ≈ exp(sin(pi*x) + y^2) [Exact recovery!]
"""


# =============================================================================
# Mathematical Functions (Reference)
# =============================================================================

def kan_forward(x, layers):
    """
    Forward pass through KAN.

    KAN(x) = (Phi_{L-1} o ... o Phi_1 o Phi_0)(x)

    Where each Phi_l is a KAN layer with learned edge functions.

    Args:
        x: Input vector [batch, n_0]
        layers: List of KAN layers

    Returns:
        Output vector [batch, n_L]
    """
    return {
        "algorithm": """
            for l in range(L):
                x_new = zeros(batch, n_{l+1})
                for j in range(n_{l+1}):
                    for i in range(n_l):
                        x_new[:, j] += phi_{l,i,j}(x[:, i])
                x = x_new
            return x
        """,
        "complexity": "O(sum_l n_l * n_{l+1} * spline_eval)",
        "parallelizable": "Spline evaluations independent per edge"
    }


def b_spline_basis(x, knots, degree):
    """
    Compute B-spline basis functions.

    Uses Cox-de Boor recursion formula.

    Args:
        x: Input values
        knots: Knot vector
        degree: Spline degree (order - 1)

    Returns:
        Basis function values at x
    """
    return {
        "recursion": """
            B_{i,0}(x) = 1 if t_i <= x < t_{i+1} else 0

            B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x)
                       + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
        """,
        "properties": """
            - Local support: B_{i,k}(x) = 0 outside [t_i, t_{i+k+1}]
            - Partition of unity: sum_i B_{i,k}(x) = 1
            - Non-negative: B_{i,k}(x) >= 0
            - Continuous: C^{k-1} continuous for degree k
        """,
        "efficiency": "Use matrix form for batch evaluation"
    }


def grid_update(phi_function, data_x, num_grid_points):
    """
    Update grid points based on data distribution.

    Place more grid points where the function varies quickly.

    Args:
        phi_function: Current learned function
        data_x: Training data distribution
        num_grid_points: Number of grid intervals

    Returns:
        New grid point positions
    """
    return {
        "algorithm": """
            # Compute second derivative (curvature)
            curvature = |d^2 phi / dx^2|

            # Weight by data density
            weighted_curvature = curvature * data_density(x)

            # Place grid points to equalize integral
            # between consecutive points
            new_grid = quantiles of weighted_curvature CDF
        """,
        "intuition": """
            More grid points where:
            1. Function changes rapidly (high curvature)
            2. Data is dense (more samples to fit)

            Fewer grid points where:
            1. Function is linear or constant
            2. Little data present
        """,
        "frequency": "Update every few hundred epochs"
    }


# =============================================================================
# Key Insights and Comparisons
# =============================================================================

KAN_VS_MLP = {
    "architecture": {
        "MLP": "Linear weights on edges, fixed activations on nodes",
        "KAN": "Learnable activations on edges, sum on nodes"
    },
    "parameters": {
        "MLP": "O(n_l * n_{l+1}) per layer",
        "KAN": "O(n_l * n_{l+1} * G) per layer (G = grid points)"
    },
    "expressiveness": {
        "MLP": "Universal approximator (width or depth)",
        "KAN": "Universal approximator (from KA theorem)"
    },
    "scaling": {
        "MLP": "Accuracy scales as O(N^{-4/d}) for smooth functions",
        "KAN": "Accuracy scales as O(N^{-4}) regardless of input dimension"
    },
    "interpretability": {
        "MLP": "Black box, features emerge in hidden layers",
        "KAN": "Each edge function visualizable, symbolic extraction possible"
    },
    "training_speed": {
        "MLP": "Highly optimized (matmul)",
        "KAN": "Slower (spline evaluation, less optimized)"
    }
}


KAN_INSIGHTS = {
    "curse_of_dimensionality": """
        MLPs suffer from curse of dimensionality for smooth functions.
        KANs factor high-dimensional functions into 1D functions,
        potentially breaking the curse for certain function classes.
    """,

    "interpretability": """
        KAN edge functions can often be identified as known functions:
        - Linear/polynomial -> Algebraic relationships
        - Trigonometric -> Periodic patterns
        - Exponential -> Growth/decay

        This enables scientific discovery from data.
    """,

    "sparsity_and_pruning": """
        Edges with near-zero w_s can be pruned.
        Remaining edges reveal important input-output relationships.
        Sparse KANs are more interpretable and faster.
    """,

    "continual_learning": """
        Grid refinement enables local function updates.
        New data can refine specific regions without catastrophic forgetting.
        More natural for online/continual learning.
    """
}


KAN_APPLICATIONS = {
    "scientific_discovery": {
        "description": "Discover equations from data",
        "example": "Recover Planck's law from black body radiation data",
        "method": "Train KAN, extract symbolic form, verify"
    },
    "pde_solving": {
        "description": "Solve PDEs with physics-informed KANs",
        "advantage": "Better accuracy per parameter than PINNs",
        "method": "KAN as solution ansatz, physics loss"
    },
    "function_approximation": {
        "description": "Approximate complex functions efficiently",
        "advantage": "Fewer parameters for same accuracy",
        "caveat": "Slower training, so useful when inference matters"
    }
}
