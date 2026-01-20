"""
Boosting Methods

This module covers the foundational boosting algorithms:
1. AdaBoost (1997) - Freund & Schapire
2. Gradient Boosting (2001) - Friedman

================================================================================
ADABOOST (Adaptive Boosting)
================================================================================

Authors: Yoav Freund and Robert Schapire (1997)

Paper: "A Decision-Theoretic Generalization of On-Line Learning and an
        Application to Boosting"
Journal: Journal of Computer and System Sciences, 55(1), 119-139

Key Innovation:
- Sequentially trains weak learners on weighted versions of the data
- Misclassified examples get higher weights for the next learner
- Final prediction is a weighted majority vote
- Provably reduces training error exponentially with number of rounds

Mathematical Formulation:
=========================

Given: Training set {(x_1, y_1), ..., (x_n, y_n)}, y_i in {-1, +1}
Initialize: Sample weights w_i^(1) = 1/n for i = 1, ..., n

For t = 1 to T (number of boosting rounds):

    1. Train weak learner h_t using distribution w^(t)

    2. Compute weighted error:
       epsilon_t = sum_{i: h_t(x_i) != y_i} w_i^(t)

    3. Compute learner weight:
       alpha_t = (1/2) * ln((1 - epsilon_t) / epsilon_t)

    4. Update sample weights:
       w_i^(t+1) = w_i^(t) * exp(-alpha_t * y_i * h_t(x_i))

    5. Normalize weights:
       w_i^(t+1) = w_i^(t+1) / sum_j w_j^(t+1)

Final Classifier:
    H(x) = sign(sum_{t=1}^{T} alpha_t * h_t(x))

Training Error Bound:
    Training_error(H) <= exp(-2 * sum_{t=1}^{T} gamma_t^2)

    Where gamma_t = (1/2) - epsilon_t is the "edge" of weak learner t

================================================================================
GRADIENT BOOSTING
================================================================================

Author: Jerome Friedman (2001)

Paper: "Greedy Function Approximation: A Gradient Boosting Machine"
Journal: Annals of Statistics, 29(5), 1189-1232

Key Innovation:
- Views boosting as gradient descent in function space
- Can optimize any differentiable loss function
- Generalizes AdaBoost to regression and other tasks
- Foundation for XGBoost, LightGBM, CatBoost

Mathematical Formulation:
=========================

Objective:
    Minimize: L(F) = sum_{i=1}^{n} l(y_i, F(x_i))

Where:
    - l(y, F(x)) is a differentiable loss function
    - F(x) = sum_{m=0}^{M} f_m(x) is an additive model

Algorithm (Gradient Descent in Function Space):
-----------------------------------------------

1. Initialize:
   F_0(x) = argmin_gamma sum_{i=1}^{n} l(y_i, gamma)

2. For m = 1 to M (boosting iterations):

   a) Compute pseudo-residuals (negative gradient):
      r_im = -[partial l(y_i, F(x_i)) / partial F(x_i)]_{F=F_{m-1}}

   b) Fit a base learner h_m(x) to pseudo-residuals {(x_i, r_im)}

   c) Compute optimal step size:
      gamma_m = argmin_gamma sum_{i=1}^{n} l(y_i, F_{m-1}(x_i) + gamma * h_m(x_i))

   d) Update model:
      F_m(x) = F_{m-1}(x) + nu * gamma_m * h_m(x)

      Where nu in (0, 1] is the learning rate (shrinkage)

3. Final model:
   F_M(x) = F_0(x) + nu * sum_{m=1}^{M} gamma_m * h_m(x)

Common Loss Functions:
----------------------

Squared Error (Regression):
    l(y, F) = (1/2)(y - F)^2
    r_i = y_i - F(x_i)  (residuals)

Absolute Error (Regression):
    l(y, F) = |y - F|
    r_i = sign(y_i - F(x_i))

Logistic Loss (Classification):
    l(y, F) = log(1 + exp(-2yF))  for y in {-1, +1}
    r_i = 2y_i / (1 + exp(2y_i * F(x_i)))

Pseudocode:
===========

def GradientBoosting(X, y, M, nu, loss):
    '''
    X, y: training data
    M: number of boosting rounds
    nu: learning rate (shrinkage)
    loss: loss function with gradient method
    '''
    # Initialize with constant prediction
    F = [initial_prediction(y, loss)]

    for m in range(M):
        # Compute pseudo-residuals (negative gradient)
        residuals = -loss.gradient(y, F[-1](X))

        # Fit base learner to residuals
        h_m = fit_tree(X, residuals)

        # Line search for optimal step
        gamma_m = line_search(y, F[-1], h_m, loss)

        # Update model
        F_m = lambda x: F[-1](x) + nu * gamma_m * h_m(x)
        F.append(F_m)

    return F[-1]

def AdaBoost(X, y, T):
    '''
    X, y: training data (y in {-1, +1})
    T: number of boosting rounds
    '''
    n = len(y)
    weights = np.ones(n) / n

    alphas = []
    weak_learners = []

    for t in range(T):
        # Train weak learner with current weights
        h_t = train_weak_learner(X, y, weights)

        # Compute weighted error
        predictions = h_t.predict(X)
        errors = (predictions != y).astype(float)
        epsilon_t = np.dot(weights, errors)

        # Compute learner weight
        alpha_t = 0.5 * np.log((1 - epsilon_t) / (epsilon_t + 1e-10))

        # Update sample weights
        weights = weights * np.exp(-alpha_t * y * predictions)
        weights = weights / weights.sum()

        alphas.append(alpha_t)
        weak_learners.append(h_t)

    return alphas, weak_learners

Complexity Analysis:
--------------------
AdaBoost Training: O(T * cost_of_weak_learner)
Gradient Boosting Training: O(M * n * log(n) * d) for tree-based learners
Prediction: O(T) or O(M * depth) depending on weak learner
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


@dataclass
class AdaBoostMethod(MLMethod):
    """
    AdaBoost (Adaptive Boosting) method entry.

    Extends MLMethod with AdaBoost-specific attributes.
    """
    n_estimators: int = 50
    base_estimator: str = "decision_stump"
    algorithm: str = "SAMME.R"  # 'SAMME' or 'SAMME.R'

    typical_use_cases: List[str] = field(default_factory=lambda: [
        "Binary classification",
        "Face detection (Viola-Jones)",
        "Feature selection through attention weighting",
        "Combining weak learners",
    ])


@dataclass
class GradientBoostingMethod(MLMethod):
    """
    Gradient Boosting method entry.

    Extends MLMethod with Gradient Boosting-specific attributes.
    """
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    loss: str = "squared_error"  # or 'deviance' for classification

    typical_use_cases: List[str] = field(default_factory=lambda: [
        "Regression with complex relationships",
        "Classification with tabular data",
        "Ranking problems",
        "Click-through rate prediction",
    ])


def create_adaboost_entry() -> AdaBoostMethod:
    """Create the canonical AdaBoost method entry."""
    return AdaBoostMethod(
        method_id="adaboost_1997",
        name="AdaBoost (Adaptive Boosting)",
        year=1997,

        era=MethodEra.CLASSICAL,
        category=MethodCategory.ARCHITECTURE,
        lineages=[MethodLineage.PERCEPTRON_LINE],

        authors=["Yoav Freund", "Robert Schapire"],
        paper_title=(
            "A Decision-Theoretic Generalization of On-Line Learning "
            "and an Application to Boosting"
        ),
        paper_url="https://www.sciencedirect.com/science/article/pii/S002200009791504X",

        key_innovation=(
            "Sequential training of weak learners where each learner focuses on "
            "examples misclassified by previous learners. Sample weights are "
            "adaptively adjusted, and learner weights are computed based on accuracy. "
            "Provides theoretical guarantees on training error reduction."
        ),

        mathematical_formulation="""
Weighted Error:
  epsilon_t = sum_{i: h_t(x_i) != y_i} w_i^(t)

Learner Weight:
  alpha_t = (1/2) * ln((1 - epsilon_t) / epsilon_t)

Weight Update:
  w_i^(t+1) = w_i^(t) * exp(-alpha_t * y_i * h_t(x_i)) / Z_t

Final Classifier:
  H(x) = sign(sum_{t=1}^{T} alpha_t * h_t(x))

Training Error Bound:
  error(H) <= exp(-2 * sum_{t=1}^{T} gamma_t^2)
  where gamma_t = 1/2 - epsilon_t
""",

        predecessors=["boosting_by_majority_1995"],
        successors=["gradient_boosting_2001", "logitboost_2000"],

        tags=[
            "boosting",
            "ensemble",
            "adaptive-weighting",
            "weak-learner",
            "exponential-loss",
        ],

        notes=(
            "AdaBoost revolutionized machine learning by showing that many "
            "weak learners can be combined to create a strong learner. "
            "The algorithm is sensitive to noisy data and outliers since "
            "these receive increasingly high weights. Later shown to minimize "
            "exponential loss, connecting it to logistic regression."
        ),
    )


def create_gradient_boosting_entry() -> GradientBoostingMethod:
    """Create the canonical Gradient Boosting method entry."""
    return GradientBoostingMethod(
        method_id="gradient_boosting_2001",
        name="Gradient Boosting Machine (GBM)",
        year=2001,

        era=MethodEra.CLASSICAL,
        category=MethodCategory.ARCHITECTURE,
        lineages=[MethodLineage.PERCEPTRON_LINE, MethodLineage.PERCEPTRON_LINE],

        authors=["Jerome H. Friedman"],
        paper_title="Greedy Function Approximation: A Gradient Boosting Machine",
        paper_url="https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full",

        key_innovation=(
            "Reformulates boosting as gradient descent in function space, allowing "
            "optimization of arbitrary differentiable loss functions. Each new "
            "learner fits the negative gradient (pseudo-residuals) of the loss. "
            "Introduces shrinkage (learning rate) for regularization."
        ),

        mathematical_formulation="""
Objective:
  min_F sum_{i=1}^{n} l(y_i, F(x_i))

Pseudo-residuals (negative gradient):
  r_im = -[d l(y_i, F(x_i)) / d F(x_i)]_{F=F_{m-1}}

Model Update:
  F_m(x) = F_{m-1}(x) + nu * gamma_m * h_m(x)

Optimal Step Size:
  gamma_m = argmin_gamma sum_i l(y_i, F_{m-1}(x_i) + gamma * h_m(x_i))

Common Losses:
  Squared Error: l = (1/2)(y - F)^2, r = y - F
  Logistic: l = log(1 + exp(-yF)), r = y / (1 + exp(yF))
""",

        predecessors=["adaboost_1997", "decision_tree_cart_1984"],
        successors=["xgboost_2016", "lightgbm_2017", "catboost_2018"],

        tags=[
            "boosting",
            "gradient-descent",
            "function-approximation",
            "ensemble",
            "regression-tree",
        ],

        notes=(
            "Gradient Boosting unifies and generalizes boosting methods through "
            "the lens of functional gradient descent. Key practical considerations: "
            "(1) shallow trees (depth 3-6) work best, (2) small learning rate with "
            "many trees reduces overfitting, (3) subsampling (stochastic GB) adds "
            "regularization. Foundation for modern implementations like XGBoost."
        ),
    )
