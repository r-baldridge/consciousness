"""
L1/L2 Weight Regularization

Classic regularization techniques that add penalty terms to the loss
function based on the magnitude of model weights.

Key Methods:
    - L1 (Lasso): Encourages sparsity
    - L2 (Ridge): Encourages small weights
    - Elastic Net: Combination of L1 and L2
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# L1 Regularization (Lasso)
# =============================================================================

L1_REGULARIZATION = MLMethod(
    method_id="l1_regularization",
    name="L1 Regularization (Lasso)",
    year=1996,
    era=MethodEra.CLASSICAL,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Robert Tibshirani"],
    paper_title="Regression Shrinkage and Selection via the Lasso",
    paper_url="https://www.jstor.org/stable/2346178",
    key_innovation="""
    Adds the L1 norm (sum of absolute values) of weights as a penalty
    to the loss function. The key property is that L1 regularization
    encourages SPARSE solutions - many weights become exactly zero.
    """,
    mathematical_formulation="""
    L1 Regularization (Lasso):
    --------------------------

    Regularized Loss:
        L_total = L(theta) + lambda * ||theta||_1
                = L(theta) + lambda * sum_i |theta_i|

    Where:
        L(theta)        = original loss function
        lambda          = regularization strength (hyperparameter)
        ||theta||_1     = L1 norm of parameters
        theta_i         = individual parameters

    Gradient of L1 penalty:
        d/d(theta_i) [lambda * |theta_i|] = lambda * sign(theta_i)

        Where sign(x) = +1 if x > 0, -1 if x < 0, undefined at x = 0

    Update rule (subgradient method):
        theta_i = theta_i - eta * (dL/d(theta_i) + lambda * sign(theta_i))

    Proximal Operator (more stable):
        theta_i = soft_threshold(theta_i - eta * dL/d(theta_i), eta * lambda)

        soft_threshold(x, tau) = sign(x) * max(|x| - tau, 0)

    Why L1 Produces Sparsity:
        Consider the 1D case: minimize |theta| subject to constraint
        - L1 ball: {theta : |theta| <= c} is a diamond shape
        - Optimal solution often lies at vertex (theta = 0)
        - This is why L1 "encourages" exact zeros

    Comparison with L2:
        L1: ||theta||_1 = sum |theta_i|       -> sparse solutions
        L2: ||theta||_2^2 = sum theta_i^2     -> small but non-zero
    """,
    predecessors=[],
    successors=["elastic_net"],
    tags=["regularization", "sparsity", "feature-selection", "lasso"],
    notes="""
    Properties:
    - Produces sparse models (feature selection)
    - Induces exact zeros in weights
    - Not differentiable at zero (requires subgradient)
    - Can select at most min(n, p) features in linear regression

    When to Use:
    - Feature selection is desired
    - Interpretable models needed
    - High-dimensional data with few relevant features
    - Embedded in networks for structured sparsity

    Practical Considerations:
    - Non-smooth: requires proximal methods or subgradient
    - Can be unstable with correlated features
    - Often combined with L2 (Elastic Net)

    In Deep Learning:
    - Less common than L2 due to non-smoothness
    - Used for structured sparsity (group lasso)
    - Weight pruning often more practical
    """
)


def l1_penalty(weights: "ndarray", lambda_: float) -> float:
    """
    Compute L1 regularization penalty.

    Pseudocode:
        penalty = lambda * sum(|w| for w in weights)
    """
    # return lambda_ * np.sum(np.abs(weights))
    pass


def l1_gradient(weights: "ndarray", lambda_: float) -> "ndarray":
    """
    Compute gradient of L1 penalty (subgradient at 0).

    Pseudocode:
        gradient = lambda * sign(weights)
    """
    # return lambda_ * np.sign(weights)
    pass


def soft_threshold(x: "ndarray", tau: float) -> "ndarray":
    """
    Proximal operator for L1 regularization.

    Pseudocode:
        result = sign(x) * max(|x| - tau, 0)
    """
    # return np.sign(x) * np.maximum(np.abs(x) - tau, 0)
    pass


# =============================================================================
# L2 Regularization (Ridge / Weight Decay)
# =============================================================================

L2_REGULARIZATION = MLMethod(
    method_id="l2_regularization",
    name="L2 Regularization (Ridge / Weight Decay)",
    year=1970,
    era=MethodEra.FOUNDATIONAL,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Arthur Hoerl", "Robert Kennard"],
    paper_title="Ridge Regression: Biased Estimation for Nonorthogonal Problems",
    paper_url="https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634",
    key_innovation="""
    Adds the squared L2 norm of weights as a penalty, encouraging small
    weights without forcing them to zero. Equivalent to assuming a
    Gaussian prior on weights in Bayesian interpretation.
    """,
    mathematical_formulation="""
    L2 Regularization (Ridge):
    --------------------------

    Regularized Loss:
        L_total = L(theta) + (lambda/2) * ||theta||_2^2
                = L(theta) + (lambda/2) * sum_i theta_i^2

    Note: The factor of 1/2 is conventional (simplifies gradient)

    Where:
        L(theta)        = original loss function
        lambda          = regularization strength
        ||theta||_2^2   = squared L2 norm of parameters

    Gradient of L2 penalty:
        d/d(theta_i) [(lambda/2) * theta_i^2] = lambda * theta_i

    Update rule:
        theta_i = theta_i - eta * (dL/d(theta_i) + lambda * theta_i)

    Weight Decay Form (equivalent for SGD):
        theta_i = (1 - eta * lambda) * theta_i - eta * dL/d(theta_i)

        The term (1 - eta * lambda) "decays" the weight toward zero.

    Bayesian Interpretation:
        L2 regularization is equivalent to MAP estimation with Gaussian prior:
        p(theta) = N(0, 1/lambda * I)

        Prior probability: exp(-lambda/2 * ||theta||^2)

    Closed-Form Solution (linear regression):
        theta* = (X^T X + lambda I)^{-1} X^T y

        The lambda I term stabilizes the inversion (ridge = adds to diagonal)
    """,
    predecessors=[],
    successors=["elastic_net", "adamw"],
    tags=["regularization", "weight-decay", "ridge", "gaussian-prior"],
    notes="""
    Properties:
    - Encourages small weights, not exact zeros
    - Smooth and differentiable everywhere
    - Equivalent to weight decay in SGD (but NOT in Adam!)
    - Shrinks weights toward zero proportionally

    Weight Decay vs L2:
    - For SGD: equivalent
    - For Adam: NOT equivalent (AdamW fixes this)
    - Weight decay: theta = (1-lr*wd)*theta - lr*grad
    - L2 regularization: theta = theta - lr*(grad + wd*theta)

    When to Use:
    - Default regularization for neural networks
    - When all features might be relevant
    - Stability in ill-conditioned problems
    - Preventing large weight magnitudes

    Typical Values:
    - lambda (or weight_decay): 1e-4 to 1e-2
    - Larger = more regularization = smaller weights
    - Too large = underfitting

    In Deep Learning:
    - Most common form of explicit regularization
    - Applied to weight matrices, usually not biases
    - Often called "weight decay" in practice
    """
)


def l2_penalty(weights: "ndarray", lambda_: float) -> float:
    """
    Compute L2 regularization penalty.

    Pseudocode:
        penalty = (lambda / 2) * sum(w^2 for w in weights)
    """
    # return (lambda_ / 2) * np.sum(weights ** 2)
    pass


def l2_gradient(weights: "ndarray", lambda_: float) -> "ndarray":
    """
    Compute gradient of L2 penalty.

    Pseudocode:
        gradient = lambda * weights
    """
    # return lambda_ * weights
    pass


# =============================================================================
# Elastic Net
# =============================================================================

ELASTIC_NET = MLMethod(
    method_id="elastic_net",
    name="Elastic Net",
    year=2005,
    era=MethodEra.CLASSICAL,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Hui Zou", "Trevor Hastie"],
    paper_title="Regularization and Variable Selection via the Elastic Net",
    paper_url="https://www.jstor.org/stable/3647580",
    key_innovation="""
    Combines L1 and L2 regularization to get benefits of both: sparsity
    from L1 and stability from L2. Particularly useful when features
    are correlated, where pure L1 is unstable.
    """,
    mathematical_formulation="""
    Elastic Net:
    ------------

    Regularized Loss:
        L_total = L(theta) + lambda_1 * ||theta||_1 + lambda_2 * ||theta||_2^2

    Or equivalently (common formulation):
        L_total = L(theta) + lambda * (alpha * ||theta||_1 + (1-alpha)/2 * ||theta||_2^2)

    Where:
        lambda      = overall regularization strength
        alpha       = mixing parameter in [0, 1]
        alpha = 1   -> pure L1 (Lasso)
        alpha = 0   -> pure L2 (Ridge)

    Gradient:
        d/d(theta) = dL/d(theta) + lambda * (alpha * sign(theta) + (1-alpha) * theta)

    L1 Ratio (alpha):
        - alpha = 1.0: Pure L1 (Lasso) - maximum sparsity
        - alpha = 0.5: Equal mix
        - alpha = 0.0: Pure L2 (Ridge) - no sparsity

    Advantages over Pure L1:
        - Can select more than min(n, p) features
        - Handles correlated features better (grouping effect)
        - More stable optimization

    Advantages over Pure L2:
        - Produces sparse solutions
        - Feature selection capability
        - More interpretable models
    """,
    predecessors=["l1_regularization", "l2_regularization"],
    successors=[],
    tags=["regularization", "sparsity", "feature-selection", "elastic-net"],
    notes="""
    When to Use:
    - Features are correlated
    - Want sparsity but L1 is unstable
    - Number of features > number of samples
    - Group selection of correlated features

    Practical Guidelines:
    - Start with alpha = 0.5
    - If need more sparsity: increase alpha
    - If unstable or too sparse: decrease alpha
    - Use cross-validation to tune both lambda and alpha

    In Deep Learning:
    - Less common than pure L2
    - Useful for structured sparsity
    - Can be combined with other regularization
    """
)


def elastic_net_penalty(
    weights: "ndarray",
    lambda_: float,
    alpha: float = 0.5
) -> float:
    """
    Compute Elastic Net penalty.

    Pseudocode:
        l1_term = sum(|w| for w in weights)
        l2_term = sum(w^2 for w in weights)
        penalty = lambda * (alpha * l1_term + (1-alpha)/2 * l2_term)
    """
    # l1 = np.sum(np.abs(weights))
    # l2 = np.sum(weights ** 2)
    # return lambda_ * (alpha * l1 + (1 - alpha) / 2 * l2)
    pass


def elastic_net_gradient(
    weights: "ndarray",
    lambda_: float,
    alpha: float = 0.5
) -> "ndarray":
    """
    Compute gradient of Elastic Net penalty.

    Pseudocode:
        gradient = lambda * (alpha * sign(weights) + (1-alpha) * weights)
    """
    # return lambda_ * (alpha * np.sign(weights) + (1 - alpha) * weights)
    pass


# =============================================================================
# Comparison
# =============================================================================

REGULARIZATION_COMPARISON = """
Weight Regularization Comparison:
=================================

| Method      | Formula              | Sparsity | Stability | Use Case            |
|-------------|---------------------|----------|-----------|---------------------|
| L1 (Lasso)  | lambda * sum|w|     | Yes      | Lower     | Feature selection   |
| L2 (Ridge)  | lambda/2 * sum(w^2) | No       | High      | Default, stability  |
| Elastic Net | L1 + L2 combination | Partial  | Medium    | Correlated features |

Geometric Interpretation:
- L1: Diamond-shaped constraint (corners at axes)
- L2: Circular constraint (smooth)
- Elastic Net: Between diamond and circle

Bayesian Interpretation:
- L1: Laplace prior on weights
- L2: Gaussian prior on weights
- Elastic Net: Combination prior
"""


def get_all_regularization_methods():
    """Return all weight regularization MLMethod entries."""
    return [L1_REGULARIZATION, L2_REGULARIZATION, ELASTIC_NET]
