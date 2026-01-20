"""
Lion - Evolved Sign Momentum

Chen et al. (2023) - "Symbolic Discovery of Optimization Algorithms"

Lion is an optimizer discovered through program search, using only the
sign of momentum for updates. Surprisingly simple yet effective.

Key Innovation:
    Uses sign(momentum) instead of magnitude, reducing memory and
    improving generalization. Discovered via symbolic search over
    possible optimizer update rules.
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


LION = MLMethod(
    method_id="lion",
    name="Lion",
    year=2023,
    era=MethodEra.NOVEL,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Xiangning Chen", "Chen Liang", "Da Huang", "Esteban Real",
             "Kaiyuan Wang", "Yao Liu", "Hieu Pham", "Xuanyi Dong",
             "Thang Luong", "Cho-Jui Hsieh", "Yifeng Lu", "Quoc V. Le"],
    paper_title="Symbolic Discovery of Optimization Algorithms",
    paper_url="https://arxiv.org/abs/2302.06675",
    key_innovation="""
    Discovered through automated program search (symbolic discovery),
    Lion uses only the SIGN of the momentum, not its magnitude. This
    makes updates uniform across parameters (all have magnitude eta),
    reduces memory by not storing second moments, and empirically
    achieves better generalization on vision and language tasks.
    """,
    mathematical_formulation="""
    Lion Update Rule:
    -----------------

    Initialize:
        m_0 = 0      (momentum buffer)

    At each timestep t:

    1. Compute gradient:
        g_t = nabla L(theta_{t-1})

    2. Compute update direction using INTERPOLATED momentum:
        c_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t

    3. Update parameters using SIGN of interpolation:
        theta_t = theta_{t-1} - eta * sign(c_t) - eta * lambda * theta_{t-1}

        Or with decoupled weight decay:
        theta_t = (1 - eta * lambda) * theta_{t-1} - eta * sign(c_t)

    4. Update momentum (using DIFFERENT interpolation):
        m_t = beta_2 * m_{t-1} + (1 - beta_2) * g_t

    Where:
        m_t         = momentum buffer
        c_t         = interpolated momentum (for direction)
        beta_1      = interpolation for update (default 0.9)
        beta_2      = interpolation for momentum update (default 0.99)
        eta         = learning rate (typically 3x smaller than Adam)
        lambda      = weight decay (typically 3x larger than AdamW)
        sign(x)     = element-wise sign function (+1, 0, or -1)

    Key Observations:
        - Update magnitude is ALWAYS eta (no adaptive scaling)
        - sign() provides direction only
        - Two different beta values discovered by search
        - No second moment (v) needed -> memory savings

    Comparison to SignSGD:
        SignSGD:  theta = theta - eta * sign(g_t)
        Lion:     theta = theta - eta * sign(beta_1 * m + (1-beta_1) * g)

        Lion's momentum smoothing is crucial for stability.

    Memory Comparison:
        Adam/AdamW: O(2N) for m and v
        Lion:       O(N)  for m only
        ~50% memory savings for optimizer states
    """,
    predecessors=["adamw", "signsgd"],
    successors=[],
    tags=["optimization", "discovered", "sign-based", "memory-efficient", "automl"],
    notes="""
    Discovery Process:
    - Program search over space of possible optimizers
    - Symbolic representation of update rules
    - Evolutionary algorithm to find best programs
    - Regularization favored simpler solutions

    Key Findings:
    - Sign-based update works surprisingly well
    - Two different betas (0.9 and 0.99) are optimal
    - Decoupled weight decay is important
    - Works better with smaller lr, larger weight decay

    Empirical Results:
    - Better than AdamW on ImageNet (ViT, various sizes)
    - Better on language modeling (T5, GPT-2)
    - Better compute-accuracy tradeoff
    - 50% memory savings on optimizer states

    Hyperparameter Conversion from AdamW:
    - lr_lion = lr_adamw / 3 to / 10
    - weight_decay_lion = weight_decay_adamw * 3 to * 10
    - Example: AdamW(lr=1e-4, wd=0.01) -> Lion(lr=3e-5, wd=0.03)

    Typical Hyperparameters:
    - lr: 1e-5 to 1e-4 (smaller than Adam)
    - weight_decay: 0.1 to 1.0 (larger than AdamW)
    - beta1: 0.9
    - beta2: 0.99

    Caveats:
    - Requires careful lr/wd tuning when switching from Adam
    - May be sensitive to batch size
    - Sign function not differentiable (but not needed for optimization)
    - Best practices still emerging

    Open Questions:
    - Why does sign work so well?
    - Optimal beta1, beta2 for different architectures?
    - Theoretical convergence guarantees?
    """
)


def lion_update(
    theta: "ndarray",
    gradient: "ndarray",
    m: "ndarray",
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.1
) -> tuple:
    """
    Pseudocode for Lion parameter update.

    Args:
        theta: Current parameters
        gradient: Gradient of loss w.r.t. parameters
        m: Momentum buffer
        lr: Learning rate (use ~3-10x smaller than Adam)
        beta1: Interpolation coefficient for update direction
        beta2: Interpolation coefficient for momentum update
        weight_decay: Decoupled weight decay (use ~3-10x larger than AdamW)

    Returns:
        Tuple of (updated_theta, updated_m)

    Pseudocode:
        # Compute update direction (interpolate momentum and gradient)
        update_direction = beta1 * m + (1 - beta1) * gradient

        # Update parameters using SIGN
        theta = theta - lr * sign(update_direction)

        # Apply decoupled weight decay
        theta = theta - lr * weight_decay * theta
        # Or equivalently: theta = (1 - lr * weight_decay) * theta - lr * sign(...)

        # Update momentum (with different interpolation)
        m = beta2 * m + (1 - beta2) * gradient
    """
    # # Compute update direction
    # update_direction = beta1 * m + (1 - beta1) * gradient
    #
    # # Update with sign and weight decay
    # theta_new = theta - lr * np.sign(update_direction) - lr * weight_decay * theta
    #
    # # Update momentum
    # m_new = beta2 * m + (1 - beta2) * gradient
    #
    # return theta_new, m_new
    pass


def lion_update_fused(
    theta: "ndarray",
    gradient: "ndarray",
    m: "ndarray",
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.1
) -> tuple:
    """
    Fused Lion update (common implementation order).

    Pseudocode:
        # Apply weight decay first
        theta = theta * (1 - lr * weight_decay)

        # Compute and apply sign update
        update_direction = beta1 * m + (1 - beta1) * gradient
        theta = theta - lr * sign(update_direction)

        # Update momentum
        m = beta2 * m + (1 - beta2) * gradient
    """
    pass


# =============================================================================
# Lion Theory and Analysis
# =============================================================================

LION_DISCOVERY = """
Lion Discovery via Symbolic Search:
===================================

Search Space:
-------------
The authors defined a program search space of possible optimizer updates:

1. Basic operations: +, -, *, /, exp, log, sqrt, sign, abs
2. State variables: theta, gradient, momentum, second_moment
3. Hyperparameters: learning_rate, beta1, beta2, epsilon, weight_decay

Search Method:
--------------
1. Start with population of random programs
2. Evaluate each on proxy tasks (small models, short training)
3. Apply evolutionary operations (mutation, crossover)
4. Select best performers
5. Add regularization to favor simpler programs
6. Transfer best to larger-scale evaluation

Key Discoveries:
----------------
1. Sign-based update emerged naturally
2. Two different beta values (not typical)
3. Weight decay should be decoupled
4. Second moment (v in Adam) not needed

The simplicity of Lion was NOT designed - it emerged from search.
"""

LION_VS_ADAM = """
Lion vs AdamW Comparison:
=========================

Update Rules:
-------------
AdamW:
    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g^2
    theta = theta - lr * m/(sqrt(v)+eps) - lr*wd*theta

Lion:
    c = b1*m + (1-b1)*g
    theta = theta - lr * sign(c) - lr*wd*theta
    m = b2*m + (1-b2)*g

Key Differences:
1. Lion uses sign(c) instead of m/sqrt(v)
   - Uniform step size across all parameters
   - No second moment needed

2. Lion uses different betas for update vs momentum
   - b1=0.9 for direction computation
   - b2=0.99 for momentum update

3. Memory: Lion saves ~50% optimizer state memory

Hyperparameter Mapping:
----------------------
AdamW                    Lion
lr = 1e-4       ->      lr = 1e-5 to 3e-5
wd = 0.01       ->      wd = 0.03 to 0.1
beta1 = 0.9     ->      beta1 = 0.9
beta2 = 0.999   ->      beta2 = 0.99

Performance:
-----------
- Lion generally better on vision (ViT, ConvNets)
- Lion generally better on language (LLMs)
- Adam/AdamW may still win on some tasks
- Lion more memory efficient
"""

LION_INTUITION = """
Why Does Lion Work?
===================

Hypotheses:

1. Regularization Effect:
   - sign() prevents large updates
   - All parameters updated by exactly +/- lr
   - May act as implicit gradient clipping

2. Direction > Magnitude:
   - The direction of update matters more than scale
   - Adaptive scaling in Adam may not always help
   - Consistent step size may improve exploration

3. Noise Injection:
   - sign() introduces quantization noise
   - May help escape local minima
   - Similar to adding noise to gradients

4. Scale Invariance:
   - Updates independent of gradient magnitude
   - More robust to loss scaling issues
   - Handles heterogeneous parameter scales

5. Simplicity:
   - Fewer moving parts = fewer ways to fail
   - Evolutionary pressure for simplicity found optimal simple solution

Open Questions:
- Theoretical convergence analysis
- When does Adam beat Lion?
- Optimal hyperparameters for different scales
"""


def get_lion_info():
    """Return the Lion MLMethod entry."""
    return LION
