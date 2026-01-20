"""
XGBoost (eXtreme Gradient Boosting) - 2016

Authors: Tianqi Chen and Carlos Guestrin

Paper: "XGBoost: A Scalable Tree Boosting System"
Conference: KDD 2016
URL: https://arxiv.org/abs/1603.02754

Key Innovations:
================
1. Regularized objective function preventing overfitting
2. Sparsity-aware algorithm for handling missing values
3. Weighted quantile sketch for approximate tree learning
4. Cache-aware access patterns for efficient computation
5. Blocks for out-of-core computation

Mathematical Formulation:
=========================

Regularized Objective:
----------------------
L(phi) = sum_{i=1}^{n} l(y_i, y_hat_i) + sum_{k=1}^{K} Omega(f_k)

Where:
  - l(y_i, y_hat_i) is the training loss (e.g., squared error, logistic)
  - y_hat_i = sum_{k=1}^{K} f_k(x_i) is the prediction (sum of K trees)
  - Omega(f_k) is the regularization term for tree f_k

Regularization Term:
-------------------
Omega(f) = gamma * T + (1/2) * lambda * ||w||^2

Where:
  - T = number of leaves in the tree
  - w = vector of leaf scores
  - gamma = complexity penalty on number of leaves (L0)
  - lambda = L2 regularization on leaf weights

Full Objective:
  L = sum_{i=1}^{n} l(y_i, y_hat_i) + sum_{k=1}^{K} [gamma * T_k + (1/2) * lambda * ||w_k||^2]

Additive Training (Taylor Expansion):
-------------------------------------
At iteration t, we want to find f_t that minimizes:

L^(t) = sum_{i=1}^{n} l(y_i, y_hat_i^(t-1) + f_t(x_i)) + Omega(f_t)

Second-order Taylor expansion around y_hat^(t-1):

L^(t) approx sum_{i=1}^{n} [l(y_i, y_hat_i^(t-1)) + g_i * f_t(x_i) + (1/2) * h_i * f_t(x_i)^2] + Omega(f_t)

Where:
  - g_i = d l(y_i, y_hat_i^(t-1)) / d y_hat_i^(t-1)  (first derivative)
  - h_i = d^2 l(y_i, y_hat_i^(t-1)) / d (y_hat_i^(t-1))^2  (second derivative)

Tree Structure Optimization:
----------------------------
For a tree structure q that maps x to leaf indices:

  f_t(x) = w_{q(x)}

Define for each leaf j:
  - I_j = {i : q(x_i) = j}  (instances in leaf j)
  - G_j = sum_{i in I_j} g_i  (sum of gradients)
  - H_j = sum_{i in I_j} h_i  (sum of hessians)

Rewriting the objective for a fixed structure q:

  L^(t)(q) = sum_{j=1}^{T} [G_j * w_j + (1/2) * (H_j + lambda) * w_j^2] + gamma * T

Optimal Leaf Weight:
  w_j* = -G_j / (H_j + lambda)

Optimal Objective Value:
  L^(t)*(q) = -(1/2) * sum_{j=1}^{T} G_j^2 / (H_j + lambda) + gamma * T

Split Finding:
--------------
Gain from splitting node into left (L) and right (R):

  Gain = (1/2) * [G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda) - (G_L + G_R)^2 / (H_L + H_R + lambda)] - gamma

Where gamma is the minimum gain required (acts as pre-pruning)

Sparsity-Aware Split Finding:
-----------------------------
Default direction for missing values:
  - Try both left and right
  - Choose direction that maximizes gain
  - Learned during training, used during prediction

Weighted Quantile Sketch:
-------------------------
For approximate split finding, XGBoost uses quantile sketches with weights h_i.

The rank function is defined as:
  r_k(z) = (1 / sum_i h_i) * sum_{(x,h): x<z} h

Find candidate split points {s_k1, s_k2, ...} such that:
  |r_k(s_{k,j}) - r_k(s_{k,j+1})| < epsilon

This ensures each bucket has approximately equal sum of hessians.

Pseudocode:
===========

def XGBoostTrain(D, K, params):
    '''
    D: training data {(x_i, y_i)}
    K: number of boosting rounds
    params: {eta, gamma, lambda, max_depth, ...}
    '''
    y_hat = initial_prediction(D)
    trees = []

    for k in range(K):
        # Compute gradients and hessians
        g = compute_gradient(y, y_hat)
        h = compute_hessian(y, y_hat)

        # Build tree using gradient statistics
        tree = build_tree(D, g, h, params)

        # Update predictions with learning rate
        y_hat = y_hat + params.eta * tree.predict(X)
        trees.append(tree)

    return trees

def build_tree(D, g, h, params, depth=0):
    '''Build a single tree using gradient and hessian information'''

    if depth >= params.max_depth:
        return create_leaf(g, h, params.lambda_)

    if sum(h) < params.min_child_weight:
        return create_leaf(g, h, params.lambda_)

    best_gain = 0
    best_split = None

    for feature in features:
        # Get split candidates (exact or approximate)
        candidates = get_split_candidates(D, feature, h, params.sketch_eps)

        for threshold in candidates:
            # Split data
            left_mask = D[feature] <= threshold
            right_mask = ~left_mask

            # Handle missing values - try both directions
            missing_mask = is_missing(D[feature])

            # Try missing -> left
            gain_left = compute_gain(
                g[left_mask | missing_mask], h[left_mask | missing_mask],
                g[right_mask], h[right_mask],
                params.gamma, params.lambda_
            )

            # Try missing -> right
            gain_right = compute_gain(
                g[left_mask], h[left_mask],
                g[right_mask | missing_mask], h[right_mask | missing_mask],
                params.gamma, params.lambda_
            )

            gain, default_left = max((gain_left, True), (gain_right, False))

            if gain > best_gain:
                best_gain = gain
                best_split = (feature, threshold, default_left)

    if best_gain <= 0:  # gamma already subtracted in gain calculation
        return create_leaf(g, h, params.lambda_)

    # Create split node
    node = SplitNode(*best_split)

    # Recurse
    left_indices, right_indices = split_with_default(D, best_split)
    node.left = build_tree(D[left_indices], g[left_indices], h[left_indices], params, depth+1)
    node.right = build_tree(D[right_indices], g[right_indices], h[right_indices], params, depth+1)

    return node

def create_leaf(g, h, lambda_):
    '''Create leaf with optimal weight'''
    G = sum(g)
    H = sum(h)
    weight = -G / (H + lambda_)
    return LeafNode(weight)

def compute_gain(g_L, h_L, g_R, h_R, gamma, lambda_):
    '''Compute gain from a split'''
    G_L, H_L = sum(g_L), sum(h_L)
    G_R, H_R = sum(g_R), sum(h_R)

    gain = 0.5 * (
        G_L**2 / (H_L + lambda_) +
        G_R**2 / (H_R + lambda_) -
        (G_L + G_R)**2 / (H_L + H_R + lambda_)
    ) - gamma

    return gain

System Optimizations:
=====================

1. Column Block Structure:
   - Data stored in column-major CSC format
   - Pre-sorted indices for each column
   - Enables parallel split finding

2. Cache-Aware Access:
   - Allocate internal buffer for gradient statistics
   - Choose appropriate block size for cache lines
   - Mini-batch style accumulation

3. Out-of-Core Computation:
   - Block compression (LZ4/ZLIB)
   - Block sharding across multiple disks
   - Pre-fetching with separate threads

4. Parallel and Distributed:
   - Parallel split finding across features
   - Row-parallel gradient computation
   - Column-parallel split candidate generation

Complexity Analysis:
--------------------
Training: O(K * d * n * log(n)) where K=trees, d=depth, n=samples
  - With approximate: O(K * d * n * (1/eps))
Prediction: O(K * d)
Space: O(n * features) for data blocks
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


@dataclass
class XGBoostMethod(MLMethod):
    """
    XGBoost method entry.

    Extends MLMethod with XGBoost-specific attributes including
    regularization parameters and system optimizations.
    """
    # Core Parameters
    n_estimators: int = 100
    learning_rate: float = 0.3  # eta
    max_depth: int = 6

    # Regularization Parameters
    gamma: float = 0.0          # min_split_loss
    lambda_: float = 1.0        # L2 regularization (reg_lambda)
    alpha: float = 0.0          # L1 regularization (reg_alpha)

    # Tree Construction
    min_child_weight: float = 1.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0

    # System Parameters
    tree_method: str = "auto"   # 'exact', 'approx', 'hist', 'gpu_hist'

    # Key innovations
    innovations: List[str] = field(default_factory=lambda: [
        "Regularized objective with L1/L2 on leaf weights",
        "Second-order Taylor expansion for optimization",
        "Sparsity-aware split finding for missing values",
        "Weighted quantile sketch for approximate learning",
        "Cache-aware block structure for efficiency",
        "Out-of-core computation for large datasets",
    ])

    typical_use_cases: List[str] = field(default_factory=lambda: [
        "Kaggle competitions (historically dominant)",
        "Tabular data classification/regression",
        "Ranking (learning to rank)",
        "Click-through rate prediction",
        "Fraud detection",
        "High-dimensional sparse data",
    ])


def create_xgboost_entry() -> XGBoostMethod:
    """Create the canonical XGBoost method entry."""
    return XGBoostMethod(
        method_id="xgboost_2016",
        name="XGBoost (eXtreme Gradient Boosting)",
        year=2016,

        era=MethodEra.DEEP_LEARNING,
        category=MethodCategory.ARCHITECTURE,
        lineages=[MethodLineage.PERCEPTRON_LINE, MethodLineage.PERCEPTRON_LINE],

        authors=["Tianqi Chen", "Carlos Guestrin"],
        paper_title="XGBoost: A Scalable Tree Boosting System",
        paper_url="https://arxiv.org/abs/1603.02754",

        key_innovation=(
            "Combines regularized boosting objective with system optimizations. "
            "The regularized objective adds complexity penalty gamma*T and L2 "
            "penalty lambda*||w||^2 on leaf weights. Uses second-order Taylor "
            "expansion for efficient optimization. Introduces sparsity-aware "
            "algorithm for missing values and weighted quantile sketch for "
            "approximate split finding on large datasets."
        ),

        mathematical_formulation="""
Regularized Objective:
  L = sum_i l(y_i, y_hat_i) + sum_k Omega(f_k)

  Where Omega(f) = gamma * T + (1/2) * lambda * ||w||^2

  - T: number of leaves
  - w: leaf weight vector
  - gamma: complexity penalty (L0 on leaves)
  - lambda: L2 regularization on weights

Taylor Expansion (2nd order):
  L^(t) approx sum_i [g_i * f_t(x_i) + (1/2) * h_i * f_t^2(x_i)] + Omega(f_t)

  Where:
    g_i = d l / d y_hat  (gradient)
    h_i = d^2 l / d y_hat^2  (hessian)

Optimal Leaf Weight:
  w_j* = -G_j / (H_j + lambda)

  Where G_j = sum_{i in I_j} g_i, H_j = sum_{i in I_j} h_i

Split Gain:
  Gain = (1/2)[G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)] - gamma
""",

        predecessors=["gradient_boosting_2001", "random_forest_2001"],
        successors=["lightgbm_2017", "catboost_2018", "ngboost_2019"],

        tags=[
            "boosting",
            "gradient-boosting",
            "regularization",
            "scalable",
            "ensemble",
            "tree-based",
            "sparsity-aware",
            "second-order",
        ],

        notes=(
            "XGBoost dominated machine learning competitions from 2015-2019, "
            "winning numerous Kaggle competitions. Key advantages: "
            "(1) built-in regularization prevents overfitting, "
            "(2) handles missing values natively, "
            "(3) highly optimized C++ implementation, "
            "(4) supports parallel and distributed training. "
            "Later succeeded by LightGBM (histogram-based, faster) and "
            "CatBoost (better categorical handling). The paper has over "
            "30,000 citations and introduced tree boosting to production ML."
        ),
    )


# Additional utility functions for working with XGBoost formulations

def compute_xgboost_gain(
    G_L: float, H_L: float,
    G_R: float, H_R: float,
    gamma: float = 0.0,
    lambda_: float = 1.0
) -> float:
    """
    Compute the gain from a split in XGBoost.

    Parameters:
    -----------
    G_L, H_L : float
        Sum of gradients and hessians for left child
    G_R, H_R : float
        Sum of gradients and hessians for right child
    gamma : float
        Minimum loss reduction required (complexity penalty)
    lambda_ : float
        L2 regularization term

    Returns:
    --------
    float : The gain from the split (negative if split not beneficial)

    Formula:
        Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
                      - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma
    """
    G = G_L + G_R
    H = H_L + H_R

    score_L = G_L ** 2 / (H_L + lambda_) if H_L + lambda_ > 0 else 0
    score_R = G_R ** 2 / (H_R + lambda_) if H_R + lambda_ > 0 else 0
    score_parent = G ** 2 / (H + lambda_) if H + lambda_ > 0 else 0

    gain = 0.5 * (score_L + score_R - score_parent) - gamma
    return gain


def compute_optimal_weight(G: float, H: float, lambda_: float = 1.0) -> float:
    """
    Compute the optimal leaf weight in XGBoost.

    Parameters:
    -----------
    G : float
        Sum of gradients for instances in the leaf
    H : float
        Sum of hessians for instances in the leaf
    lambda_ : float
        L2 regularization term

    Returns:
    --------
    float : The optimal leaf weight

    Formula:
        w* = -G / (H + lambda)
    """
    return -G / (H + lambda_) if H + lambda_ > 0 else 0.0


def compute_objective_value(
    G_leaves: List[float],
    H_leaves: List[float],
    gamma: float = 0.0,
    lambda_: float = 1.0
) -> float:
    """
    Compute the objective value for a tree structure.

    Parameters:
    -----------
    G_leaves : List[float]
        Sum of gradients for each leaf
    H_leaves : List[float]
        Sum of hessians for each leaf
    gamma : float
        Complexity penalty on number of leaves
    lambda_ : float
        L2 regularization term

    Returns:
    --------
    float : The objective value (lower is better)

    Formula:
        L* = -0.5 * sum_j [G_j^2 / (H_j + lambda)] + gamma * T
    """
    T = len(G_leaves)

    score = sum(
        G ** 2 / (H + lambda_) if H + lambda_ > 0 else 0
        for G, H in zip(G_leaves, H_leaves)
    )

    return -0.5 * score + gamma * T
