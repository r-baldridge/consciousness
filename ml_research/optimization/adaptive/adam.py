"""
Adam - Adaptive Moment Estimation

Kingma & Ba (2014) - "Adam: A Method for Stochastic Optimization"

Adam combines the benefits of AdaGrad (adaptive learning rates) and
RMSProp (exponential moving average) with momentum, plus bias correction
for the initial steps.

Key Innovation:
    Maintains exponential moving averages of both first moment (mean) and
    second moment (uncentered variance) of gradients, with bias correction
    to account for initialization at zero.
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


ADAM = MLMethod(
    method_id="adam",
    name="Adam",
    year=2014,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Diederik P. Kingma", "Jimmy Lei Ba"],
    paper_title="Adam: A Method for Stochastic Optimization",
    paper_url="https://arxiv.org/abs/1412.6980",
    key_innovation="""
    Combines exponential moving averages of both the first moment (momentum)
    and second moment (adaptive learning rate) of gradients. Crucially includes
    bias correction to compensate for the zero-initialization of moment
    estimates, which is essential in early training steps.
    """,
    mathematical_formulation="""
    Adam Update Rule:
    -----------------

    Initialize:
        m_0 = 0      (first moment estimate)
        v_0 = 0      (second moment estimate)
        t = 0        (timestep)

    At each timestep t:

    1. Compute gradient:
        g_t = nabla L(theta_{t-1})

    2. Update biased first moment estimate (momentum):
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t

    3. Update biased second moment estimate (adaptive LR):
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2

    4. Compute bias-corrected first moment:
        m_hat_t = m_t / (1 - beta_1^t)

    5. Compute bias-corrected second moment:
        v_hat_t = v_t / (1 - beta_2^t)

    6. Update parameters:
        theta_t = theta_{t-1} - eta * m_hat_t / (sqrt(v_hat_t) + epsilon)

    Where:
        m_t         = biased first moment estimate
        v_t         = biased second moment estimate
        m_hat_t     = bias-corrected first moment
        v_hat_t     = bias-corrected second moment
        beta_1      = exponential decay rate for first moment (default 0.9)
        beta_2      = exponential decay rate for second moment (default 0.999)
        eta         = learning rate (default 0.001)
        epsilon     = small constant for stability (default 1e-8)

    Bias Correction Intuition:
        - At t=1 with beta_1=0.9: m_1 = 0.1*g_1 (underestimated by 10x)
        - Correction: m_hat_1 = m_1/(1-0.9^1) = m_1/0.1 = g_1
        - As t -> infinity: correction approaches 1

    Relationship to Other Methods:
        - Adam = RMSProp + Momentum + Bias Correction
        - Without v_t: becomes SGD with momentum
        - Without m_t: becomes RMSProp

    Effective Step Size Bounds:
        |Delta theta_t| <= eta * sqrt(1 - beta_2^t) / (1 - beta_1^t)
        This bound approaches eta as t -> infinity
    """,
    predecessors=["rmsprop", "momentum", "adagrad"],
    successors=["adamw", "nadam", "lamb", "radam"],
    tags=["optimization", "adaptive-learning-rate", "deep-learning", "default-optimizer"],
    notes="""
    Why Adam is Popular:
    - Works well out of the box with default hyperparameters
    - Robust to hyperparameter choices
    - Efficient computation and memory
    - Works well for large datasets and parameters

    Default Hyperparameters (from paper):
    - eta (learning rate): 0.001
    - beta_1: 0.9
    - beta_2: 0.999
    - epsilon: 1e-8

    Known Issues:
    - May not converge in some cases (fixed by AMSGrad)
    - Weight decay is not properly regularized (fixed by AdamW)
    - Can generalize worse than SGD on some tasks
    - Bias correction adds computation

    Variants:
    - AdamW: Decoupled weight decay
    - NAdam: Nesterov momentum variant
    - RAdam: Rectified Adam (adaptive learning rate warmup)
    - AMSGrad: Uses max of past squared gradients

    Citation Count (2024): >180,000 - most cited deep learning paper
    """
)


def adam_update(
    theta: "ndarray",
    gradient: "ndarray",
    m: "ndarray",
    v: "ndarray",
    t: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8
) -> tuple:
    """
    Pseudocode for Adam parameter update.

    Args:
        theta: Current parameters
        gradient: Gradient of loss w.r.t. parameters
        m: First moment estimate (momentum)
        v: Second moment estimate (adaptive LR)
        t: Current timestep (starts at 1)
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        epsilon: Small constant for numerical stability

    Returns:
        Tuple of (updated_theta, updated_m, updated_v)

    Pseudocode:
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * gradient

        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * gradient^2

        # Bias correction
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)

        # Update parameters
        theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
    """
    # m_new = beta1 * m + (1 - beta1) * gradient
    # v_new = beta2 * v + (1 - beta2) * gradient ** 2
    # m_hat = m_new / (1 - beta1 ** t)
    # v_hat = v_new / (1 - beta2 ** t)
    # theta_new = theta - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    # return theta_new, m_new, v_new
    pass


def adam_update_fused(
    theta: "ndarray",
    gradient: "ndarray",
    m: "ndarray",
    v: "ndarray",
    t: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8
) -> tuple:
    """
    Fused Adam update with incorporated bias correction.

    More efficient formulation that avoids computing m_hat and v_hat explicitly.

    Pseudocode:
        # Update moments
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient^2

        # Compute step size with incorporated bias correction
        step_size = lr * sqrt(1 - beta2^t) / (1 - beta1^t)

        # Update parameters
        theta = theta - step_size * m / (sqrt(v) + epsilon)
    """
    pass


# =============================================================================
# Adam Variants
# =============================================================================

ADAM_VARIANTS = """
Adam Variants:
==============

1. AMSGrad (Reddi et al., 2018):
   - Uses maximum of past squared gradients
   - v_hat_t = max(v_hat_{t-1}, v_t / (1 - beta_2^t))
   - Guarantees convergence in convex settings
   - Paper: "On the Convergence of Adam and Beyond"

2. NAdam (Dozat, 2016):
   - Incorporates Nesterov momentum
   - m_hat_t = beta_1 * m_hat_t + (1 - beta_1) * g_t / (1 - beta_1^t)
   - Better on some tasks

3. RAdam (Liu et al., 2019):
   - Rectified Adam with adaptive learning rate warmup
   - Addresses variance of adaptive learning rate
   - No need for manual warmup
   - Paper: "On the Variance of the Adaptive Learning Rate and Beyond"

4. AdaMax (Kingma & Ba, 2014):
   - Uses L-infinity norm instead of L2
   - u_t = max(beta_2 * u_{t-1}, |g_t|)
   - theta = theta - eta * m_hat_t / u_t
   - No need for bias correction on v_t

5. Yogi (Zaheer et al., 2018):
   - Adaptive effective learning rate control
   - v_t = v_{t-1} + (1-beta_2) * sign(g_t^2 - v_{t-1}) * g_t^2
"""

ADAM_VS_SGD_DEBATE = """
Adam vs SGD Debate:
===================

Adam Advantages:
- Faster convergence in early training
- Less sensitive to learning rate
- Works well out of the box
- Good for large-scale training

SGD with Momentum Advantages:
- Often better generalization
- Preferred for computer vision
- State-of-the-art results on ImageNet
- Simpler, fewer hyperparameters

Research Findings:
- Wilson et al. (2017): SGD generalizes better
- Choi et al. (2019): With proper tuning, both similar
- Zhang et al. (2019): Adam faster but SGD better asymptotically

Current Best Practice:
- Vision: SGD with momentum + cosine annealing
- NLP/Transformers: AdamW with warmup
- Reinformcement Learning: Adam
- General: Start with AdamW, tune if needed
"""


def get_adam_info():
    """Return the Adam MLMethod entry."""
    return ADAM
