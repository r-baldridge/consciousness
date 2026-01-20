"""
RMSProp - Root Mean Square Propagation

Hinton (2012) - Unpublished, presented in Coursera Neural Networks course

RMSProp addresses AdaGrad's diminishing learning rate problem by using
an exponential moving average of squared gradients instead of accumulating them.

Key Innovation:
    Uses a decaying average of squared gradients, preventing the learning
    rate from becoming infinitesimally small while maintaining per-parameter
    adaptation.
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


RMSPROP = MLMethod(
    method_id="rmsprop",
    name="RMSProp",
    year=2012,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Geoffrey Hinton"],
    paper_title="Neural Networks for Machine Learning - Lecture 6e",
    paper_url="https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf",
    key_innovation="""
    Replaces AdaGrad's sum of squared gradients with an exponential moving
    average (EMA). This allows the learning rate to recover if gradients
    become small, solving AdaGrad's "learning rate dying" problem. The
    name comes from dividing by the root mean square of gradients.
    """,
    mathematical_formulation="""
    RMSProp Update Rule:
    --------------------

    Compute exponential moving average of squared gradients:
        E[g^2]_t = gamma * E[g^2]_{t-1} + (1 - gamma) * g_t^2

    Update parameters:
        theta_{t+1} = theta_t - eta / sqrt(E[g^2]_t + epsilon) * g_t

    Where:
        E[g^2]_t    = exponential moving average of squared gradients
        gamma       = decay rate (typically 0.9 or 0.99)
        g_t         = gradient at time t
        eta         = learning rate (typically 0.001)
        epsilon     = small constant for numerical stability (typically 1e-8)

    Interpretation:
        The denominator sqrt(E[g^2]_t) is approximately the RMS (root mean square)
        of recent gradients, hence the name "RMSProp".

    Effective Learning Rate:
        eta_eff(t) = eta / RMS(g)

        Unlike AdaGrad, this can increase if recent gradients are small,
        allowing continued learning.

    RMSProp with Momentum (variant):
        v_t = gamma_m * v_{t-1} + eta / sqrt(E[g^2]_t + epsilon) * g_t
        theta_{t+1} = theta_t - v_t

        Where gamma_m is the momentum coefficient.

    Centered RMSProp (variant):
        E[g]_t = gamma * E[g]_{t-1} + (1 - gamma) * g_t
        E[g^2]_t = gamma * E[g^2]_{t-1} + (1 - gamma) * g_t^2
        Var[g]_t = E[g^2]_t - E[g]_t^2
        theta_{t+1} = theta_t - eta / sqrt(Var[g]_t + epsilon) * g_t

        Uses variance instead of second moment for normalization.
    """,
    predecessors=["adagrad"],
    successors=["adam", "adadelta"],
    tags=["optimization", "adaptive-learning-rate", "deep-learning", "unpublished"],
    notes="""
    History:
    - Presented in Geoffrey Hinton's Coursera course (2012)
    - Never formally published, but widely adopted
    - Independently developed alongside Adadelta (Zeiler, 2012)

    Advantages:
    - Learning rate can recover (unlike AdaGrad)
    - Works well for non-stationary objectives
    - Effective for RNNs and training deep networks
    - Simple modification from AdaGrad

    Typical Hyperparameters:
    - eta (learning rate): 0.001
    - gamma (decay rate): 0.9 or 0.99
    - epsilon: 1e-8

    Relationship to Adam:
    - Adam = RMSProp + Momentum + Bias Correction
    - RMSProp can be seen as the adaptive learning rate component of Adam
    """
)


def rmsprop_update(
    theta: "ndarray",
    gradient: "ndarray",
    E_g2: "ndarray",
    lr: float = 0.001,
    gamma: float = 0.9,
    epsilon: float = 1e-8
) -> tuple:
    """
    Pseudocode for RMSProp parameter update.

    Args:
        theta: Current parameters
        gradient: Gradient of loss w.r.t. parameters
        E_g2: Exponential moving average of squared gradients
        lr: Learning rate
        gamma: Decay rate for moving average
        epsilon: Small constant for numerical stability

    Returns:
        Tuple of (updated_parameters, updated_E_g2)

    Pseudocode:
        # Update moving average of squared gradients
        E_g2 = gamma * E_g2 + (1 - gamma) * gradient^2

        # Update parameters
        theta = theta - lr / sqrt(E_g2 + epsilon) * gradient
    """
    # E_g2_new = gamma * E_g2 + (1 - gamma) * gradient ** 2
    # theta_new = theta - lr / np.sqrt(E_g2_new + epsilon) * gradient
    # return theta_new, E_g2_new
    pass


def rmsprop_with_momentum(
    theta: "ndarray",
    gradient: "ndarray",
    E_g2: "ndarray",
    velocity: "ndarray",
    lr: float = 0.001,
    gamma: float = 0.9,
    momentum: float = 0.9,
    epsilon: float = 1e-8
) -> tuple:
    """
    RMSProp with momentum for improved convergence.

    Pseudocode:
        # Update moving average of squared gradients
        E_g2 = gamma * E_g2 + (1 - gamma) * gradient^2

        # Compute adaptive gradient
        adaptive_grad = lr / sqrt(E_g2 + epsilon) * gradient

        # Apply momentum
        velocity = momentum * velocity + adaptive_grad

        # Update parameters
        theta = theta - velocity
    """
    pass


def centered_rmsprop_update(
    theta: "ndarray",
    gradient: "ndarray",
    E_g: "ndarray",
    E_g2: "ndarray",
    lr: float = 0.001,
    gamma: float = 0.9,
    epsilon: float = 1e-8
) -> tuple:
    """
    Centered RMSProp using variance instead of second moment.

    Pseudocode:
        # Update moving averages
        E_g = gamma * E_g + (1 - gamma) * gradient
        E_g2 = gamma * E_g2 + (1 - gamma) * gradient^2

        # Compute variance
        var_g = E_g2 - E_g^2

        # Update parameters
        theta = theta - lr / sqrt(var_g + epsilon) * gradient
    """
    pass


# =============================================================================
# RMSProp Properties and Comparisons
# =============================================================================

RMSPROP_VS_ADAGRAD = """
RMSProp vs AdaGrad Comparison:
==============================

AdaGrad:
    G_t = G_{t-1} + g_t^2                    (accumulating sum)
    theta = theta - eta / sqrt(G_t + eps) * g

RMSProp:
    E[g^2]_t = gamma * E[g^2]_{t-1} + (1-gamma) * g_t^2    (EMA)
    theta = theta - eta / sqrt(E[g^2]_t + eps) * g

Key Differences:
    - AdaGrad: G only increases -> learning rate only decreases
    - RMSProp: E[g^2] can decrease -> learning rate can recover

When to Use:
    - AdaGrad: Sparse data, convex optimization
    - RMSProp: Deep learning, non-stationary objectives
"""

RMSPROP_INTUITION = """
RMSProp Intuition:
==================

The key insight is dividing by the RMS of recent gradients:

1. If recent gradients are large:
   - RMS is large
   - Effective learning rate is small
   - Prevents overshooting

2. If recent gradients are small:
   - RMS is small
   - Effective learning rate is large
   - Maintains learning progress

3. The "memory" controlled by gamma:
   - gamma = 0.9: remembers ~10 steps
   - gamma = 0.99: remembers ~100 steps
   - gamma = 0.999: remembers ~1000 steps

   Effective window = 1 / (1 - gamma)
"""


def get_rmsprop_info():
    """Return the RMSProp MLMethod entry."""
    return RMSPROP
