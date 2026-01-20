"""
Gradient Descent Optimization Methods

This module contains research indices for gradient descent variants,
the foundational optimization algorithms for training neural networks.

Key Methods:
    - SGD (Stochastic Gradient Descent): Basic gradient-based optimization
    - Momentum: Accelerated SGD with velocity accumulation
    - Nesterov Accelerated Gradient: "Look-ahead" momentum variant

Mathematical Foundation:
    All gradient descent methods minimize a loss function L(theta) by
    iteratively updating parameters in the direction of steepest descent.
"""

from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# Stochastic Gradient Descent (SGD)
# =============================================================================

SGD = MLMethod(
    method_id="sgd",
    name="Stochastic Gradient Descent",
    year=1951,
    era=MethodEra.FOUNDATIONAL,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Herbert Robbins", "Sutton Monro"],
    paper_title="A Stochastic Approximation Method",
    paper_url="https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full",
    key_innovation="""
    Uses stochastic approximation to estimate gradients from mini-batches
    rather than the full dataset, enabling efficient optimization on large
    datasets. The stochastic noise provides implicit regularization.
    """,
    mathematical_formulation="""
    SGD Update Rule:
    ----------------
    theta_{t+1} = theta_t - eta * nabla L(theta_t; x_i, y_i)

    Where:
        theta_t     = parameters at time t
        eta         = learning rate (step size)
        nabla L     = gradient of loss function
        (x_i, y_i)  = mini-batch sample(s)

    Full Batch Gradient Descent:
        theta_{t+1} = theta_t - eta * (1/N) * sum_{i=1}^{N} nabla L(theta_t; x_i, y_i)

    Mini-Batch SGD (typical):
        theta_{t+1} = theta_t - eta * (1/B) * sum_{i=1}^{B} nabla L(theta_t; x_i, y_i)
        where B = batch size

    Convergence (convex case):
        E[L(theta_T)] - L(theta*) <= O(1/sqrt(T))
    """,
    predecessors=[],
    successors=["momentum", "nesterov_accelerated_gradient", "adagrad"],
    tags=["optimization", "gradient-descent", "foundational", "stochastic"],
    notes="""
    - Simple and widely used baseline optimizer
    - Learning rate selection is critical
    - Can oscillate in ravine-like loss surfaces
    - Stochastic noise helps escape shallow local minima
    - Often requires learning rate decay for convergence
    """
)


def sgd_update(theta: "ndarray", gradient: "ndarray", lr: float) -> "ndarray":
    """
    Pseudocode for SGD parameter update.

    Args:
        theta: Current parameters
        gradient: Gradient of loss w.r.t. parameters
        lr: Learning rate

    Returns:
        Updated parameters

    Pseudocode:
        theta = theta - lr * gradient
    """
    # theta_new = theta - lr * gradient
    pass


# =============================================================================
# Momentum
# =============================================================================

MOMENTUM = MLMethod(
    method_id="momentum",
    name="Momentum",
    year=1964,
    era=MethodEra.FOUNDATIONAL,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Boris Polyak"],
    paper_title="Some methods of speeding up the convergence of iteration methods",
    paper_url="https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=zvmmf&paperid=7713&option_lang=eng",
    key_innovation="""
    Introduces a velocity term that accumulates gradients over time,
    accelerating convergence in consistent gradient directions while
    dampening oscillations in directions with high curvature.
    """,
    mathematical_formulation="""
    Momentum Update Rule:
    --------------------
    v_t = gamma * v_{t-1} + eta * nabla L(theta_t)
    theta_{t+1} = theta_t - v_t

    Alternative formulation:
    v_t = gamma * v_{t-1} + nabla L(theta_t)
    theta_{t+1} = theta_t - eta * v_t

    Where:
        v_t         = velocity at time t
        gamma       = momentum coefficient (typically 0.9)
        eta         = learning rate
        nabla L     = gradient of loss function

    Physical Interpretation:
        - theta represents position
        - v represents velocity
        - gamma represents friction/damping
        - nabla L represents force (negative gradient)

    Effective Learning Rate:
        With momentum, the effective step size in consistent
        gradient direction approaches: eta / (1 - gamma)
        For gamma = 0.9, this is 10x the base learning rate
    """,
    predecessors=["sgd"],
    successors=["nesterov_accelerated_gradient", "adam"],
    tags=["optimization", "gradient-descent", "momentum", "acceleration"],
    notes="""
    - Accelerates convergence in low-curvature directions
    - Dampens oscillations in high-curvature directions
    - Common momentum values: 0.9, 0.95, 0.99
    - Higher momentum = faster acceleration but potential overshoot
    - Works well with learning rate warmup
    """
)


def momentum_update(
    theta: "ndarray",
    gradient: "ndarray",
    velocity: "ndarray",
    lr: float,
    gamma: float = 0.9
) -> tuple:
    """
    Pseudocode for Momentum parameter update.

    Args:
        theta: Current parameters
        gradient: Gradient of loss w.r.t. parameters
        velocity: Accumulated velocity from previous steps
        lr: Learning rate
        gamma: Momentum coefficient

    Returns:
        Tuple of (updated_parameters, updated_velocity)

    Pseudocode:
        v = gamma * v + lr * gradient
        theta = theta - v
    """
    # v_new = gamma * velocity + lr * gradient
    # theta_new = theta - v_new
    # return theta_new, v_new
    pass


# =============================================================================
# Nesterov Accelerated Gradient (NAG)
# =============================================================================

NESTEROV_ACCELERATED_GRADIENT = MLMethod(
    method_id="nesterov_accelerated_gradient",
    name="Nesterov Accelerated Gradient",
    year=1983,
    era=MethodEra.FOUNDATIONAL,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Yurii Nesterov"],
    paper_title="A method for solving the convex programming problem with convergence rate O(1/k^2)",
    paper_url=None,
    key_innovation="""
    Computes the gradient at a "look-ahead" position where the momentum
    would take us, providing a more accurate gradient estimate. This
    anticipatory correction allows for more responsive updates and
    faster convergence, achieving optimal O(1/k^2) rate for convex functions.
    """,
    mathematical_formulation="""
    Nesterov Accelerated Gradient:
    ------------------------------

    Original formulation (look-ahead):
    theta_lookahead = theta_t - gamma * v_{t-1}
    v_t = gamma * v_{t-1} + eta * nabla L(theta_lookahead)
    theta_{t+1} = theta_t - v_t

    Equivalent reformulation (for implementation):
    v_t = gamma * v_{t-1} + eta * nabla L(theta_t - gamma * v_{t-1})
    theta_{t+1} = theta_t - v_t

    Common implementation (using auxiliary variable):
    Let theta_aux = theta + gamma * v (accumulated position)

    v_t = gamma * v_{t-1} + eta * nabla L(theta_aux_{t-1})
    theta_aux_t = theta_aux_{t-1} - eta * nabla L(theta_aux_{t-1})
                  + gamma * (v_t - v_{t-1})

    Where:
        v_t             = velocity at time t
        gamma           = momentum coefficient (typically 0.9)
        eta             = learning rate
        nabla L         = gradient of loss function
        theta_lookahead = anticipated position

    Convergence Rate:
        Convex: O(1/T^2) vs O(1/T) for standard gradient descent
        This is provably optimal for first-order methods on convex functions
    """,
    predecessors=["momentum", "sgd"],
    successors=["adam", "nadam"],
    tags=["optimization", "gradient-descent", "momentum", "nesterov", "accelerated"],
    notes="""
    - Also known as NAG or Nesterov Momentum
    - Look-ahead gradient provides better correction
    - Optimal convergence rate for convex optimization
    - More responsive to loss surface changes than standard momentum
    - Sutskever et al. (2013) popularized NAG for deep learning
    - Forms the basis for NAdam (Nesterov + Adam)
    """
)


def nesterov_update(
    theta: "ndarray",
    gradient_fn: callable,
    velocity: "ndarray",
    lr: float,
    gamma: float = 0.9
) -> tuple:
    """
    Pseudocode for Nesterov Accelerated Gradient update.

    Args:
        theta: Current parameters
        gradient_fn: Function to compute gradient at a given point
        velocity: Accumulated velocity from previous steps
        lr: Learning rate
        gamma: Momentum coefficient

    Returns:
        Tuple of (updated_parameters, updated_velocity)

    Pseudocode:
        # Look ahead
        theta_lookahead = theta - gamma * velocity
        gradient = gradient_fn(theta_lookahead)

        # Update velocity and parameters
        v = gamma * velocity + lr * gradient
        theta = theta - v
    """
    # theta_lookahead = theta - gamma * velocity
    # gradient = gradient_fn(theta_lookahead)
    # v_new = gamma * velocity + lr * gradient
    # theta_new = theta - v_new
    # return theta_new, v_new
    pass


# =============================================================================
# Comparison Summary
# =============================================================================

GRADIENT_DESCENT_COMPARISON = """
Gradient Descent Method Comparison:
===================================

| Method    | Update Rule                    | Convergence | Best For              |
|-----------|--------------------------------|-------------|------------------------|
| SGD       | theta -= lr * grad             | O(1/sqrt(T))| Simple problems        |
| Momentum  | v = gamma*v + lr*grad          | O(1/T)      | Smooth landscapes      |
|           | theta -= v                     |             |                        |
| NAG       | grad at (theta - gamma*v)     | O(1/T^2)    | Convex optimization    |
|           | theta -= gamma*v + lr*grad     |             |                        |

Key Insights:
- SGD: Noisy but good generalization
- Momentum: Accelerates through consistent gradients
- NAG: Look-ahead provides better correction

Typical Hyperparameters:
- Learning rate (eta): 0.01 - 0.1 (task-dependent)
- Momentum (gamma): 0.9 (standard), 0.99 (high momentum)
"""


# =============================================================================
# Utility Functions
# =============================================================================

def get_all_gradient_descent_methods():
    """Return all gradient descent MLMethod entries."""
    return [SGD, MOMENTUM, NESTEROV_ACCELERATED_GRADIENT]


def get_gradient_descent_evolution():
    """Return the evolution chain of gradient descent methods."""
    return {
        "sgd": {
            "name": "Stochastic Gradient Descent",
            "year": 1951,
            "innovation": "Stochastic gradient estimation",
            "led_to": ["momentum", "adagrad"]
        },
        "momentum": {
            "name": "Momentum",
            "year": 1964,
            "innovation": "Velocity accumulation",
            "led_to": ["nesterov", "adam"]
        },
        "nesterov": {
            "name": "Nesterov Accelerated Gradient",
            "year": 1983,
            "innovation": "Look-ahead gradient",
            "led_to": ["nadam"]
        }
    }
