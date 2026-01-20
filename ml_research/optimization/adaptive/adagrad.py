"""
AdaGrad - Adaptive Gradient Algorithm

Duchi et al. (2011) - "Adaptive Subgradient Methods for Online Learning
and Stochastic Optimization"

AdaGrad adapts the learning rate for each parameter based on the historical
sum of squared gradients, giving larger updates to infrequent features.

Key Innovation:
    Per-parameter learning rates that decrease based on gradient history,
    enabling automatic learning rate adaptation for sparse features.
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


ADAGRAD = MLMethod(
    method_id="adagrad",
    name="AdaGrad",
    year=2011,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["John Duchi", "Elad Hazan", "Yoram Singer"],
    paper_title="Adaptive Subgradient Methods for Online Learning and Stochastic Optimization",
    paper_url="https://jmlr.org/papers/v12/duchi11a.html",
    key_innovation="""
    Introduces per-parameter adaptive learning rates based on the historical
    sum of squared gradients. Parameters with large accumulated gradients
    receive smaller updates, while parameters with small gradients receive
    larger updates. This is particularly effective for sparse data and
    natural language processing tasks.
    """,
    mathematical_formulation="""
    AdaGrad Update Rule:
    --------------------

    For each parameter theta_i:

    Accumulate squared gradients:
        G_{t,i} = G_{t-1,i} + g_{t,i}^2

    Or in matrix form:
        G_t = G_{t-1} + g_t odot g_t    (element-wise square)

    Update parameters:
        theta_{t+1,i} = theta_{t,i} - eta / sqrt(G_{t,i} + epsilon) * g_{t,i}

    Or in vector form:
        theta_{t+1} = theta_t - eta / sqrt(G_t + epsilon) odot g_t

    Where:
        G_t         = accumulated sum of squared gradients (diagonal matrix)
        g_t         = gradient at time t
        eta         = global learning rate (typically 0.01)
        epsilon     = small constant for numerical stability (typically 1e-8)
        odot        = element-wise multiplication

    Effective Learning Rate:
        eta_eff(t, i) = eta / sqrt(G_{t,i} + epsilon)

        This decreases monotonically over training, as G_t only accumulates.

    Per-Parameter Adaptation:
        - Frequent parameters (large G_i): smaller effective learning rate
        - Infrequent parameters (small G_i): larger effective learning rate

    Regret Bound (online convex optimization):
        R_T <= O(sqrt(T))

        For sparse gradients with d non-zero components on average:
        R_T <= O(sqrt(dT))   (much better when d << D, total dimensions)
    """,
    predecessors=["sgd"],
    successors=["rmsprop", "adadelta", "adam"],
    tags=["optimization", "adaptive-learning-rate", "sparse-data", "NLP"],
    notes="""
    Advantages:
    - No manual learning rate tuning (in theory)
    - Excellent for sparse features (NLP, recommendation systems)
    - Good for embeddings with varying frequency
    - Provable regret bounds for online learning

    Limitations:
    - Learning rate only decreases, never increases
    - Can stop learning too early in training (accumulated G becomes large)
    - Not well-suited for non-convex deep learning
    - Memory: stores G (same size as parameters)

    This limitation led to RMSProp, which uses exponential moving average
    instead of sum, preventing the learning rate from vanishing.

    Typical Hyperparameters:
    - eta (initial learning rate): 0.01
    - epsilon: 1e-8 to 1e-6
    """
)


def adagrad_update(
    theta: "ndarray",
    gradient: "ndarray",
    G: "ndarray",
    lr: float = 0.01,
    epsilon: float = 1e-8
) -> tuple:
    """
    Pseudocode for AdaGrad parameter update.

    Args:
        theta: Current parameters
        gradient: Gradient of loss w.r.t. parameters
        G: Accumulated sum of squared gradients
        lr: Global learning rate
        epsilon: Small constant for numerical stability

    Returns:
        Tuple of (updated_parameters, updated_G)

    Pseudocode:
        # Accumulate squared gradient
        G = G + gradient^2

        # Compute adaptive update
        theta = theta - lr / sqrt(G + epsilon) * gradient
    """
    # G_new = G + gradient ** 2
    # theta_new = theta - lr / np.sqrt(G_new + epsilon) * gradient
    # return theta_new, G_new
    pass


def adagrad_update_with_clipping(
    theta: "ndarray",
    gradient: "ndarray",
    G: "ndarray",
    lr: float = 0.01,
    epsilon: float = 1e-8,
    grad_clip: float = 1.0
) -> tuple:
    """
    AdaGrad with gradient clipping for stability.

    Pseudocode:
        # Clip gradient norm
        grad_norm = ||gradient||_2
        if grad_norm > grad_clip:
            gradient = gradient * (grad_clip / grad_norm)

        # Standard AdaGrad update
        G = G + gradient^2
        theta = theta - lr / sqrt(G + epsilon) * gradient
    """
    pass


# =============================================================================
# AdaGrad Variants
# =============================================================================

ADAGRAD_DIAGONAL = """
AdaGrad (Diagonal Form - Standard):
-----------------------------------
Uses diagonal approximation of the full matrix.

Full matrix form (theoretical):
    G_t = G_{t-1} + g_t @ g_t.T    (outer product)
    theta_{t+1} = theta_t - eta * G_t^{-1/2} @ g_t

Diagonal approximation (practical):
    G_t = G_{t-1} + g_t^2          (element-wise)
    theta_{t+1} = theta_t - eta / sqrt(G_t + eps) * g_t

The diagonal form is used in practice for computational efficiency.
"""

ADAGRAD_PROPERTIES = """
AdaGrad Key Properties:
=======================

1. Monotonic Learning Rate Decay:
   - eta_eff(t) = eta / sqrt(sum_s g_s^2) always decreases
   - Can lead to premature convergence

2. Sparse Feature Handling:
   - Infrequent features get larger updates
   - Ideal for word embeddings, user/item factors

3. Scale Invariance:
   - Robust to gradient magnitude variations
   - Works well without extensive hyperparameter tuning

4. Memory Cost:
   - O(|theta|) additional memory for G
   - Same as parameter count

5. Convergence:
   - Convex: O(1/sqrt(T)) regret bound
   - Non-convex: May stop learning due to accumulated G
"""


def get_adagrad_info():
    """Return the AdaGrad MLMethod entry."""
    return ADAGRAD
