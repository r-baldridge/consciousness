"""
AdamW - Adam with Decoupled Weight Decay

Loshchilov & Hutter (2017) - "Decoupled Weight Decay Regularization"

AdamW fixes Adam's improper handling of L2 regularization by decoupling
weight decay from the gradient-based update, leading to better generalization.

Key Innovation:
    Separates weight decay from the adaptive gradient computation,
    applying it directly to weights rather than through gradients.
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


ADAMW = MLMethod(
    method_id="adamw",
    name="AdamW",
    year=2017,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Ilya Loshchilov", "Frank Hutter"],
    paper_title="Decoupled Weight Decay Regularization",
    paper_url="https://arxiv.org/abs/1711.05101",
    key_innovation="""
    Reveals that L2 regularization and weight decay are NOT equivalent in
    adaptive optimizers like Adam. Standard Adam with L2 regularization
    scales the regularization by the adaptive learning rate, weakening its
    effect on parameters with large gradients. AdamW applies weight decay
    directly, independent of the gradient adaptation.
    """,
    mathematical_formulation="""
    The Problem with Adam + L2 Regularization:
    ------------------------------------------

    Standard Adam with L2 regularization:
        L_reg = L + (lambda/2) * ||theta||^2
        g_t = nabla L + lambda * theta        (gradient includes L2 term)
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
        theta = theta - lr * m_hat_t / sqrt(v_hat_t)

    Problem: The weight decay term (lambda * theta) is scaled by 1/sqrt(v_t),
    so parameters with large historical gradients get LESS regularization!

    AdamW Solution - Decoupled Weight Decay:
    ----------------------------------------

    1. Compute gradient (without L2 term):
        g_t = nabla L(theta_{t-1})

    2. Update biased first moment estimate:
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t

    3. Update biased second moment estimate:
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2

    4. Compute bias-corrected estimates:
        m_hat_t = m_t / (1 - beta_1^t)
        v_hat_t = v_t / (1 - beta_2^t)

    5. Update parameters with DECOUPLED weight decay:
        theta_t = theta_{t-1} - eta * (m_hat_t / (sqrt(v_hat_t) + epsilon) + lambda * theta_{t-1})

    Or equivalently (more common implementation):
        theta_t = (1 - eta * lambda) * theta_{t-1} - eta * m_hat_t / (sqrt(v_hat_t) + epsilon)

    Where:
        lambda = weight decay coefficient (NOT L2 regularization)
        The weight decay is applied at full strength, independent of v_t

    Comparison:
        Adam + L2:  effective_decay = lambda / sqrt(v_t)    (variable per parameter)
        AdamW:      effective_decay = lambda                (constant)

    Key Insight:
        L2 regularization != weight decay for adaptive optimizers
        They are only equivalent for vanilla SGD
    """,
    predecessors=["adam"],
    successors=["lamb", "sophia"],
    tags=["optimization", "adaptive-learning-rate", "weight-decay", "regularization"],
    notes="""
    Why AdamW Matters:
    - Standard optimizer for transformers and large language models
    - Used in BERT, GPT, and most modern NLP models
    - Better generalization than Adam with L2

    Empirical Results:
    - Significantly better than Adam on ImageNet
    - Better generalization across various tasks
    - Particularly important for large models

    Relationship to L2 Regularization:
    - For SGD: L2 regularization = weight decay
    - For Adam: L2 regularization != weight decay
    - AdamW uses true weight decay

    Typical Hyperparameters:
    - lr: 1e-4 to 1e-3 (often with warmup)
    - weight_decay: 0.01 to 0.1
    - beta1: 0.9
    - beta2: 0.999 (or 0.98 for transformers)
    - epsilon: 1e-8

    Usage:
    - Default for transformers (BERT, GPT, etc.)
    - PyTorch: torch.optim.AdamW
    - Usually combined with learning rate warmup

    Impact:
    - Published at ICLR 2019
    - Widely adopted in NLP and vision
    - Clarified misconception about regularization in Adam
    """
)


def adamw_update(
    theta: "ndarray",
    gradient: "ndarray",
    m: "ndarray",
    v: "ndarray",
    t: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.01,
    epsilon: float = 1e-8
) -> tuple:
    """
    Pseudocode for AdamW parameter update.

    Args:
        theta: Current parameters
        gradient: Gradient of loss w.r.t. parameters (WITHOUT L2 term)
        m: First moment estimate (momentum)
        v: Second moment estimate (adaptive LR)
        t: Current timestep (starts at 1)
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        weight_decay: Decoupled weight decay coefficient
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

        # Update with DECOUPLED weight decay
        theta = theta - lr * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * theta)

        # Or equivalently:
        # theta = (1 - lr * weight_decay) * theta - lr * m_hat / (sqrt(v_hat) + epsilon)
    """
    # m_new = beta1 * m + (1 - beta1) * gradient
    # v_new = beta2 * v + (1 - beta2) * gradient ** 2
    # m_hat = m_new / (1 - beta1 ** t)
    # v_hat = v_new / (1 - beta2 ** t)
    # theta_new = theta - lr * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * theta)
    # return theta_new, m_new, v_new
    pass


def adamw_update_fused(
    theta: "ndarray",
    gradient: "ndarray",
    m: "ndarray",
    v: "ndarray",
    t: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.01,
    epsilon: float = 1e-8
) -> tuple:
    """
    Fused AdamW update (common implementation).

    Pseudocode:
        # Apply weight decay first
        theta = theta * (1 - lr * weight_decay)

        # Then apply Adam update
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)

    Note: Weight decay applied first, then gradient update.
    """
    pass


# =============================================================================
# Comparison: Adam vs AdamW
# =============================================================================

ADAM_VS_ADAMW = """
Adam vs AdamW Comparison:
=========================

Adam with L2 Regularization:
----------------------------
Loss function: L_total = L + (lambda/2) * ||theta||^2

Gradient: g = nabla L + lambda * theta

Update:
    m = beta1 * m + (1-beta1) * (nabla L + lambda * theta)
    v = beta2 * v + (1-beta2) * (nabla L + lambda * theta)^2
    theta = theta - lr * m_hat / sqrt(v_hat)

Problem: The regularization term is scaled by 1/sqrt(v_hat)
- Parameters with large gradients get LESS regularization
- This is not the intended effect of regularization

AdamW (Decoupled Weight Decay):
-------------------------------
Gradient: g = nabla L  (no regularization term)

Update:
    m = beta1 * m + (1-beta1) * nabla L
    v = beta2 * v + (1-beta2) * (nabla L)^2
    theta = theta - lr * m_hat / sqrt(v_hat) - lr * lambda * theta

The regularization is constant across all parameters.

Mathematical Equivalence (SGD only):
------------------------------------
For SGD: L2 regularization == weight decay
    nabla L_reg = nabla L + lambda * theta
    theta = theta - lr * (nabla L + lambda * theta)
          = theta - lr * nabla L - lr * lambda * theta
          = (1 - lr * lambda) * theta - lr * nabla L  <<<< Weight decay form

For Adam: L2 regularization != weight decay
    The adaptive scaling breaks the equivalence.
"""

ADAMW_USAGE_GUIDE = """
AdamW Usage Guide:
==================

Recommended for:
- Transformers (BERT, GPT, T5, etc.)
- Vision Transformers (ViT)
- Large language models
- Fine-tuning pretrained models

Typical Settings for Transformers:
    lr: 1e-4 to 5e-4
    weight_decay: 0.01 to 0.1
    beta1: 0.9
    beta2: 0.98 or 0.999
    epsilon: 1e-8
    warmup_steps: 1000-10000

Example Configuration (BERT fine-tuning):
    optimizer = AdamW(
        lr=2e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )

    # With warmup
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
"""


def get_adamw_info():
    """Return the AdamW MLMethod entry."""
    return ADAMW
