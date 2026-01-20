"""
Random Forest (2001)
Author: Leo Breiman

Paper: "Random Forests"
Journal: Machine Learning, 45(1), 5-32

Key Innovation:
- Combines bagging (bootstrap aggregating) with random feature selection
- Each tree is trained on a bootstrap sample of the data
- At each split, only a random subset of features is considered
- Reduces variance through averaging while maintaining low bias

Mathematical Formulation:
========================

Bootstrap Aggregating (Bagging):
--------------------------------
Given training set D = {(x_1, y_1), ..., (x_n, y_n)}

For b = 1 to B (number of trees):
    1. Draw bootstrap sample D_b of size n from D (with replacement)
    2. Train decision tree T_b on D_b

Final prediction (regression):
    f(x) = (1/B) * sum_{b=1}^{B} T_b(x)

Final prediction (classification):
    f(x) = argmax_c sum_{b=1}^{B} I(T_b(x) = c)

Random Feature Selection:
-------------------------
At each node split:
    - Select m features randomly from p total features
    - m = sqrt(p) for classification (typical)
    - m = p/3 for regression (typical)
    - Find best split among these m features

Out-of-Bag (OOB) Error:
-----------------------
For each observation (x_i, y_i):
    - Compute predictions only from trees where x_i was NOT in bootstrap sample
    - OOB error approximates leave-one-out cross-validation error

OOB_error = (1/n) * sum_{i=1}^{n} L(y_i, f_OOB(x_i))

Where f_OOB(x_i) uses only trees trained without x_i

Variable Importance:
--------------------
Mean Decrease Impurity (MDI):
    Imp(x_j) = sum over all trees sum over all nodes splitting on x_j
               of (p(t) * delta_i(s_t, t))

Where:
    - p(t) = proportion of samples reaching node t
    - delta_i = decrease in impurity from split s_t

Permutation Importance:
    Imp(x_j) = (1/B) * sum_{b=1}^{B} [OOB_error(T_b, x_j permuted) - OOB_error(T_b)]

Pseudocode:
===========

def RandomForest(D, B, m):
    '''
    D: training data {(x_1, y_1), ..., (x_n, y_n)}
    B: number of trees
    m: number of features to consider at each split
    '''
    forest = []

    for b in range(B):
        # Bootstrap sampling
        D_b = bootstrap_sample(D)

        # Build tree with random feature selection
        T_b = build_tree(D_b, m)
        forest.append(T_b)

    return forest

def build_tree(D, m):
    '''Build decision tree with random feature selection'''
    if stopping_criteria_met(D):
        return create_leaf(D)

    # Randomly select m features
    features = random_subset(all_features, m)

    # Find best split among selected features
    best_feature, best_threshold = find_best_split(D, features)

    # Split data and recurse
    D_left, D_right = split_data(D, best_feature, best_threshold)

    node = TreeNode(best_feature, best_threshold)
    node.left = build_tree(D_left, m)
    node.right = build_tree(D_right, m)

    return node

def predict(forest, x):
    '''Aggregate predictions from all trees'''
    predictions = [tree.predict(x) for tree in forest]

    if classification:
        return mode(predictions)  # Majority vote
    else:
        return mean(predictions)  # Average

Complexity Analysis:
--------------------
Training: O(B * n * log(n) * m) where B=trees, n=samples, m=features per split
Prediction: O(B * log(n))
Space: O(B * n * log(n)) for storing trees
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
class RandomForestMethod(MLMethod):
    """
    Random Forest ensemble method entry.

    Extends MLMethod with Random Forest-specific attributes.
    """
    n_estimators: int = 100
    max_features: str = "sqrt"  # 'sqrt', 'log2', or int
    bootstrap: bool = True
    oob_score: bool = True

    # Additional metadata
    typical_use_cases: List[str] = field(default_factory=lambda: [
        "Classification with high-dimensional data",
        "Regression with complex nonlinear relationships",
        "Feature importance ranking",
        "Handling missing values (through surrogate splits)",
        "Robust to outliers",
    ])


def create_random_forest_entry() -> RandomForestMethod:
    """Create the canonical Random Forest method entry."""
    return RandomForestMethod(
        method_id="random_forest_2001",
        name="Random Forest",
        year=2001,

        era=MethodEra.CLASSICAL,
        category=MethodCategory.ARCHITECTURE,
        lineages=[MethodLineage.PERCEPTRON_LINE, MethodLineage.PERCEPTRON_LINE],

        authors=["Leo Breiman"],
        paper_title="Random Forests",
        paper_url="https://link.springer.com/article/10.1023/A:1010933404324",

        key_innovation=(
            "Combines bagging with random feature selection at each split, "
            "creating decorrelated trees that reduce variance when averaged. "
            "Introduces out-of-bag error estimation for model validation."
        ),

        mathematical_formulation="""
Ensemble Prediction:
  Classification: f(x) = argmax_c sum_{b=1}^{B} I(T_b(x) = c)
  Regression:     f(x) = (1/B) * sum_{b=1}^{B} T_b(x)

Bootstrap Sampling:
  Draw n samples with replacement from training set

Random Feature Selection:
  At each split, consider m random features
  Typical m: sqrt(p) for classification, p/3 for regression

OOB Error:
  OOB_error = (1/n) * sum L(y_i, f_OOB(x_i))
  Where f_OOB uses only trees not trained on x_i
""",

        predecessors=["bagging_1996", "decision_tree_cart_1984"],
        successors=["extremely_randomized_trees_2006", "xgboost_2016"],

        tags=[
            "ensemble",
            "bagging",
            "decision-tree",
            "feature-importance",
            "out-of-bag",
            "variance-reduction",
        ],

        notes=(
            "Random Forests remain one of the most successful and widely-used "
            "ensemble methods. Key strengths include: (1) minimal hyperparameter "
            "tuning required, (2) robust to overfitting with more trees, "
            "(3) natural handling of mixed feature types, (4) built-in feature "
            "importance measures, and (5) efficient parallelization."
        ),
    )
