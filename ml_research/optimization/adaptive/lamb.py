"""
LAMB - Layer-wise Adaptive Moments for Batch training

You et al. (2019) - "Large Batch Optimization for Deep Learning:
Training BERT in 76 Minutes"

LAMB enables training with very large batch sizes by applying layer-wise
learning rate adaptation on top of Adam, achieving near-linear scaling.

Key Innovation:
    Layer-wise trust ratio that scales updates based on the ratio of
    parameter norm to update norm, enabling stable large-batch training.
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


LAMB = MLMethod(
    method_id="lamb",
    name="LAMB",
    year=2019,
    era=MethodEra.ATTENTION,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Yang You", "Jing Li", "Sashank Reddi", "Jonathan Hseu",
             "Sanjiv Kumar", "Diederik Kingma", "Yonghui Wu", "Bryan Barkley"],
    paper_title="Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes",
    paper_url="https://arxiv.org/abs/1904.00962",
    key_innovation="""
    Introduces layer-wise adaptive learning rates based on the trust ratio:
    the ratio of parameter norm to update norm. This prevents any single
    layer from taking disproportionately large steps, enabling stable
    training with batch sizes up to 64K while maintaining model quality.
    """,
    mathematical_formulation="""
    LAMB Update Rule:
    -----------------

    LAMB builds on Adam with layer-wise learning rate adaptation.

    For each layer l:

    1. Compute Adam update direction:
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
        m_hat_t = m_t / (1 - beta_1^t)
        v_hat_t = v_t / (1 - beta_2^t)

    2. Compute update with weight decay:
        r_t = m_hat_t / (sqrt(v_hat_t) + epsilon) + lambda * theta_{t-1}

    3. Compute layer-wise trust ratio:
        phi(theta) = ||theta||_2                (parameter norm)
        phi(r) = ||r_t||_2                      (update norm)

        trust_ratio = phi(theta) / phi(r)       if phi(theta) > 0 and phi(r) > 0
                    = 1                          otherwise

    4. Update parameters:
        theta_t = theta_{t-1} - eta * trust_ratio * r_t

    Where:
        m_t, v_t    = Adam's first and second moment estimates
        lambda      = weight decay coefficient
        r_t         = Adam update direction (with weight decay)
        trust_ratio = layer-wise scaling factor (LARS-style)
        eta         = global learning rate

    Relationship to LARS:
        LARS (Layer-wise Adaptive Rate Scaling) for SGD:
            trust_ratio = ||theta|| / ||grad||

        LAMB extends this concept to Adam:
            trust_ratio = ||theta|| / ||adam_update||

    Key Properties:
        - Trust ratio ~1 when norms are balanced
        - Small trust ratio prevents large updates
        - Large trust ratio boosts small updates
        - Layer-wise application handles varying scales
    """,
    predecessors=["adamw", "lars"],
    successors=["sophia"],
    tags=["optimization", "large-batch", "transformer", "distributed-training"],
    notes="""
    Motivation:
    - Standard Adam fails with very large batch sizes
    - Linear scaling rule fails for adaptive optimizers
    - Need layer-wise adaptation for stability

    Key Results:
    - Trained BERT in 76 minutes (64K batch size)
    - Near-linear scaling up to 32K batch size
    - Maintains model quality with large batches

    Comparison:
    - LAMB with batch 32K matches Adam with batch 512
    - No warmup heuristics needed
    - Works across various architectures

    Typical Hyperparameters:
    - eta (base lr): scales with sqrt(batch_size)
    - weight_decay: 0.01
    - beta1: 0.9
    - beta2: 0.999
    - epsilon: 1e-6
    - clip_norm: Optional gradient clipping

    Practical Notes:
    - Clip trust_ratio to [0, 10] for stability
    - Exclude bias and LayerNorm from weight decay
    - Works well with linear warmup
    - Memory efficient (same as AdamW)

    LAMB vs LARS:
    - LARS: Layer-wise SGD with momentum
    - LAMB: Layer-wise Adam
    - LAMB better for attention-based models
    """
)


def lamb_update(
    theta: "ndarray",
    gradient: "ndarray",
    m: "ndarray",
    v: "ndarray",
    t: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.01,
    epsilon: float = 1e-6,
    trust_ratio_clip: float = 10.0
) -> tuple:
    """
    Pseudocode for LAMB parameter update (single layer).

    Args:
        theta: Current parameters for this layer
        gradient: Gradient of loss w.r.t. parameters
        m: First moment estimate (momentum)
        v: Second moment estimate (adaptive LR)
        t: Current timestep (starts at 1)
        lr: Base learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        weight_decay: Weight decay coefficient
        epsilon: Small constant for numerical stability
        trust_ratio_clip: Maximum value for trust ratio

    Returns:
        Tuple of (updated_theta, updated_m, updated_v)

    Pseudocode:
        # Standard Adam moment updates
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient^2

        # Bias correction
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)

        # Compute update direction with weight decay
        update = m_hat / (sqrt(v_hat) + epsilon) + weight_decay * theta

        # Compute trust ratio
        param_norm = ||theta||_2
        update_norm = ||update||_2

        if param_norm > 0 and update_norm > 0:
            trust_ratio = param_norm / update_norm
            trust_ratio = min(trust_ratio, trust_ratio_clip)  # clip for stability
        else:
            trust_ratio = 1.0

        # Apply layer-wise scaled update
        theta = theta - lr * trust_ratio * update
    """
    # m_new = beta1 * m + (1 - beta1) * gradient
    # v_new = beta2 * v + (1 - beta2) * gradient ** 2
    # m_hat = m_new / (1 - beta1 ** t)
    # v_hat = v_new / (1 - beta2 ** t)
    #
    # update = m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * theta
    #
    # param_norm = np.linalg.norm(theta)
    # update_norm = np.linalg.norm(update)
    #
    # if param_norm > 0 and update_norm > 0:
    #     trust_ratio = min(param_norm / update_norm, trust_ratio_clip)
    # else:
    #     trust_ratio = 1.0
    #
    # theta_new = theta - lr * trust_ratio * update
    # return theta_new, m_new, v_new
    pass


# =============================================================================
# LAMB Theory and Comparisons
# =============================================================================

LAMB_THEORY = """
LAMB Theoretical Foundation:
============================

Trust Ratio Intuition:
----------------------
The trust ratio addresses a fundamental problem in large-batch training:
different layers can have vastly different gradient scales.

Without layer-wise adaptation:
    - Some layers may take steps that are too large (diverge)
    - Some layers may take steps that are too small (slow learning)

Trust ratio ensures:
    - ||actual_step|| ~ ||theta|| (proportional to parameter scale)
    - Larger parameters can take larger steps
    - Prevents any layer from dominating

Mathematical Analysis:
----------------------
For a layer with parameters theta and update r:

    actual_step = lr * trust_ratio * r
                = lr * (||theta|| / ||r||) * r

    ||actual_step|| = lr * ||theta||  (independent of r magnitude!)

This normalizes the update magnitude to be proportional to parameter norm.

Convergence:
-----------
Under standard assumptions (Lipschitz continuous gradients, bounded variance):
- LAMB converges at rate O(1/sqrt(T)) for convex objectives
- Empirically matches small-batch performance with large batches
"""

LAMB_VS_LARS = """
LAMB vs LARS Comparison:
========================

LARS (Layer-wise Adaptive Rate Scaling) - You et al., 2017:
----------------------------------------------------------
For SGD with momentum:
    v = momentum * v + grad + weight_decay * theta
    trust_ratio = ||theta|| / ||v||
    theta = theta - lr * trust_ratio * v

Key: Applies to SGD update

LAMB (Layer-wise Adaptive Moments):
----------------------------------
For Adam:
    adam_update = m_hat / sqrt(v_hat) + weight_decay * theta
    trust_ratio = ||theta|| / ||adam_update||
    theta = theta - lr * trust_ratio * adam_update

Key: Applies to Adam update

When to Use:
- LARS: CNNs, ResNet (SGD typically better baseline)
- LAMB: Transformers, BERT, GPT (Adam typically better baseline)

Both enable:
- Near-linear scaling to 32K+ batch size
- Faster wall-clock training with more GPUs
- Maintained model quality
"""

LAMB_PRACTICAL_GUIDE = """
LAMB Practical Guide:
=====================

Learning Rate Scaling:
---------------------
Unlike SGD where lr scales linearly with batch size,
LAMB uses square root scaling:

    lr = base_lr * sqrt(batch_size / base_batch_size)

Example:
    base_lr = 1e-3 at batch_size = 256
    For batch_size = 32768:
    lr = 1e-3 * sqrt(32768/256) = 1e-3 * sqrt(128) ~ 1.13e-2

Warmup:
------
Still beneficial but less critical than standard Adam:
- Linear warmup over 1-5% of training
- Helps with initial instability

Parameter Groups:
----------------
Exclude from weight decay:
- Bias terms
- LayerNorm parameters
- Embedding parameters (often)

Typical Configuration (BERT pre-training):
-----------------------------------------
optimizer = LAMB(
    params,
    lr=1e-2,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-6
)

# Learning rate warmup
warmup_proportion = 0.01
scheduler = warmup_linear(warmup_proportion)

# Gradient clipping (optional but recommended)
max_grad_norm = 1.0
"""


def get_lamb_info():
    """Return the LAMB MLMethod entry."""
    return LAMB
