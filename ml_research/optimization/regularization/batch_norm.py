"""
Normalization Methods

Normalization techniques that stabilize and accelerate training by
normalizing activations across different dimensions.

Key Methods:
    - BatchNorm: Normalize across batch dimension
    - LayerNorm: Normalize across feature dimension
    - GroupNorm: Normalize across groups of channels
    - RMSNorm: Simplified normalization without centering
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# Batch Normalization
# =============================================================================

BATCH_NORMALIZATION = MLMethod(
    method_id="batch_normalization",
    name="Batch Normalization",
    year=2015,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Sergey Ioffe", "Christian Szegedy"],
    paper_title="Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
    paper_url="https://arxiv.org/abs/1502.03167",
    key_innovation="""
    Normalizes activations across the batch dimension at each layer,
    reducing internal covariate shift. Enables higher learning rates,
    faster convergence, and acts as a regularizer.
    """,
    mathematical_formulation="""
    Batch Normalization:
    --------------------

    For a mini-batch B = {x_1, ..., x_m}:

    1. Compute batch mean:
        mu_B = (1/m) * sum_{i=1}^{m} x_i

    2. Compute batch variance:
        sigma_B^2 = (1/m) * sum_{i=1}^{m} (x_i - mu_B)^2

    3. Normalize:
        x_hat_i = (x_i - mu_B) / sqrt(sigma_B^2 + epsilon)

    4. Scale and shift (learnable parameters):
        y_i = gamma * x_hat_i + beta

    Where:
        mu_B        = batch mean
        sigma_B^2   = batch variance
        epsilon     = small constant (1e-5) for numerical stability
        gamma       = learnable scale parameter (initialized to 1)
        beta        = learnable shift parameter (initialized to 0)

    Training vs Inference:
        Training: Use batch statistics (mu_B, sigma_B^2)

        Inference: Use running statistics (exponential moving average)
            mu_running = momentum * mu_running + (1 - momentum) * mu_B
            var_running = momentum * var_running + (1 - momentum) * sigma_B^2

    For Convolutional Layers:
        Input shape: (N, C, H, W)
        Normalize over: (N, H, W) for each channel C
        Parameters: gamma, beta of shape (C,)

    Gradient Flow:
        dL/dx = gamma / sqrt(sigma_B^2 + eps) * (dL/dy - mean(dL/dy)
                - x_hat * mean(dL/dy * x_hat))

        The normalization stabilizes gradients across layers.
    """,
    predecessors=[],
    successors=["layer_normalization", "group_normalization", "instance_normalization"],
    tags=["normalization", "regularization", "covariate-shift", "batch"],
    notes="""
    Benefits:
    - Higher learning rates (10x or more)
    - Faster convergence
    - Reduces sensitivity to initialization
    - Acts as regularizer (can reduce dropout need)
    - Enables training very deep networks

    Limitations:
    - Depends on batch size (small batch = noisy statistics)
    - Different behavior train vs test
    - Not ideal for RNNs (varying sequence lengths)
    - Batch dependency breaks for some architectures

    Placement:
    - Original: Before activation (conv -> BN -> ReLU)
    - Alternative: After activation (conv -> ReLU -> BN)
    - Both work, slight performance differences

    Typical Momentum:
    - momentum = 0.1 (PyTorch default)
    - momentum = 0.99 (TensorFlow/Keras default)
    - Note: conventions differ between frameworks!

    Batch Size Sensitivity:
    - Works well: batch >= 32
    - Degraded: batch < 16
    - Use GroupNorm or LayerNorm for small batches
    """
)


def batch_norm_forward(
    x: "ndarray",
    gamma: "ndarray",
    beta: "ndarray",
    running_mean: "ndarray" = None,
    running_var: "ndarray" = None,
    training: bool = True,
    momentum: float = 0.1,
    epsilon: float = 1e-5
) -> tuple:
    """
    Pseudocode for batch normalization forward pass.

    Args:
        x: Input of shape (N, C, ...) where N is batch size
        gamma: Scale parameter of shape (C,)
        beta: Shift parameter of shape (C,)
        running_mean: Running mean for inference
        running_var: Running variance for inference
        training: Whether in training mode
        momentum: Momentum for running statistics
        epsilon: Numerical stability constant

    Pseudocode:
        if training:
            # Compute batch statistics
            mean = x.mean(axis=0)        # across batch
            var = x.var(axis=0)          # across batch

            # Update running statistics
            running_mean = momentum * mean + (1 - momentum) * running_mean
            running_var = momentum * var + (1 - momentum) * running_var

            # Normalize using batch statistics
            x_hat = (x - mean) / sqrt(var + epsilon)
        else:
            # Normalize using running statistics
            x_hat = (x - running_mean) / sqrt(running_var + epsilon)

        # Scale and shift
        output = gamma * x_hat + beta
    """
    pass


# =============================================================================
# Layer Normalization
# =============================================================================

LAYER_NORMALIZATION = MLMethod(
    method_id="layer_normalization",
    name="Layer Normalization",
    year=2016,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Jimmy Lei Ba", "Jamie Ryan Kiros", "Geoffrey E. Hinton"],
    paper_title="Layer Normalization",
    paper_url="https://arxiv.org/abs/1607.06450",
    key_innovation="""
    Normalizes across the feature dimension instead of batch dimension,
    making it independent of batch size. Essential for transformers and
    recurrent networks where batch normalization is problematic.
    """,
    mathematical_formulation="""
    Layer Normalization:
    --------------------

    For each sample independently:

    1. Compute mean across features:
        mu = (1/H) * sum_{i=1}^{H} x_i

    2. Compute variance across features:
        sigma^2 = (1/H) * sum_{i=1}^{H} (x_i - mu)^2

    3. Normalize:
        x_hat = (x - mu) / sqrt(sigma^2 + epsilon)

    4. Scale and shift:
        y = gamma * x_hat + beta

    Where:
        H           = number of features (hidden size)
        mu          = mean over features (computed per sample)
        sigma^2     = variance over features (computed per sample)
        gamma       = learnable scale (shape: H)
        beta        = learnable shift (shape: H)

    For Transformers (typical):
        Input shape: (batch, seq_len, hidden_dim)
        Normalize over: hidden_dim (last axis)
        Each token normalized independently

    Comparison to BatchNorm:
        BatchNorm: normalize over (N,) for each feature
        LayerNorm: normalize over (H,) for each sample

        BatchNorm needs batch statistics; LayerNorm doesn't.

    Pre-Norm vs Post-Norm (in Transformers):
        Post-Norm (original): x + LayerNorm(Sublayer(x))
        Pre-Norm:  x + Sublayer(LayerNorm(x))

        Pre-Norm often more stable for very deep models.
    """,
    predecessors=["batch_normalization"],
    successors=["rms_normalization"],
    tags=["normalization", "transformer", "RNN", "batch-independent"],
    notes="""
    Key Advantages:
    - No batch dependency (works with batch size 1)
    - Same computation train and test
    - Essential for transformers
    - Works well for RNNs

    Used In:
    - Transformers (BERT, GPT, etc.)
    - RNNs/LSTMs
    - Batch size 1 scenarios
    - Online learning

    Placement in Transformers:
    - Post-LayerNorm: After residual (original Transformer)
    - Pre-LayerNorm: Before attention/FFN (GPT-2, more stable)

    Comparison:
    - BatchNorm: Better for CNNs with large batches
    - LayerNorm: Better for transformers, RNNs, small batches
    """
)


def layer_norm_forward(
    x: "ndarray",
    gamma: "ndarray",
    beta: "ndarray",
    epsilon: float = 1e-5
) -> "ndarray":
    """
    Pseudocode for layer normalization forward pass.

    Args:
        x: Input of shape (..., H) where H is normalized dimension
        gamma: Scale parameter of shape (H,)
        beta: Shift parameter of shape (H,)
        epsilon: Numerical stability constant

    Pseudocode:
        # Compute statistics over last axis (features)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        # Normalize
        x_hat = (x - mean) / sqrt(var + epsilon)

        # Scale and shift
        output = gamma * x_hat + beta
    """
    pass


# =============================================================================
# Group Normalization
# =============================================================================

GROUP_NORMALIZATION = MLMethod(
    method_id="group_normalization",
    name="Group Normalization",
    year=2018,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Yuxin Wu", "Kaiming He"],
    paper_title="Group Normalization",
    paper_url="https://arxiv.org/abs/1803.08494",
    key_innovation="""
    Divides channels into groups and normalizes within each group,
    providing a middle ground between BatchNorm (batch dependency)
    and LayerNorm (normalizes all channels together). Works well
    with small batch sizes.
    """,
    mathematical_formulation="""
    Group Normalization:
    --------------------

    Divide C channels into G groups, each with C/G channels.

    For each sample and each group:

    1. Compute group mean:
        mu_g = (1/(C/G * H * W)) * sum_{c in group g} sum_{h,w} x_{c,h,w}

    2. Compute group variance:
        sigma_g^2 = (1/(C/G * H * W)) * sum_{c in group g} sum_{h,w} (x_{c,h,w} - mu_g)^2

    3. Normalize:
        x_hat = (x - mu_g) / sqrt(sigma_g^2 + epsilon)

    4. Scale and shift (per channel):
        y = gamma * x_hat + beta

    Where:
        G           = number of groups
        C           = number of channels
        C/G         = channels per group
        gamma, beta = learnable, shape (C,)

    Special Cases:
        G = C:        Instance Normalization (each channel is a group)
        G = 1:        Layer Normalization (all channels in one group)
        G = C/k:      Group Normalization with k channels per group

    Input shape: (N, C, H, W)
    Reshape to:  (N, G, C/G, H, W)
    Normalize over: (C/G, H, W) for each (N, G)

    Typical Groups:
        G = 32 is a common default
        G must divide C evenly
    """,
    predecessors=["batch_normalization", "layer_normalization"],
    successors=[],
    tags=["normalization", "groups", "batch-independent", "CNN"],
    notes="""
    Key Advantages:
    - Independent of batch size
    - Works well with small batches
    - Better than BatchNorm for detection/segmentation

    Comparison:
    - BatchNorm: Best for large batch classification
    - GroupNorm: Best for small batch or detection
    - LayerNorm: Best for transformers

    Typical Settings:
    - G = 32 (common default)
    - G should divide number of channels evenly

    Use Cases:
    - Object detection (small batch due to large images)
    - Segmentation
    - Video processing
    - Any small batch scenario

    Performance:
    - Matches BatchNorm with large batches
    - Significantly better than BatchNorm with small batches
    - Slight overhead from grouping
    """
)


def group_norm_forward(
    x: "ndarray",
    gamma: "ndarray",
    beta: "ndarray",
    num_groups: int = 32,
    epsilon: float = 1e-5
) -> "ndarray":
    """
    Pseudocode for group normalization forward pass.

    Args:
        x: Input of shape (N, C, H, W)
        gamma: Scale parameter of shape (C,)
        beta: Shift parameter of shape (C,)
        num_groups: Number of groups G
        epsilon: Numerical stability constant

    Pseudocode:
        N, C, H, W = x.shape
        G = num_groups

        # Reshape to separate groups
        x = x.reshape(N, G, C // G, H, W)

        # Compute statistics over (C//G, H, W) for each (N, G)
        mean = x.mean(axis=(2, 3, 4), keepdims=True)
        var = x.var(axis=(2, 3, 4), keepdims=True)

        # Normalize
        x_hat = (x - mean) / sqrt(var + epsilon)

        # Reshape back
        x_hat = x_hat.reshape(N, C, H, W)

        # Scale and shift
        output = gamma * x_hat + beta
    """
    pass


# =============================================================================
# RMS Normalization
# =============================================================================

RMS_NORMALIZATION = MLMethod(
    method_id="rms_normalization",
    name="RMSNorm",
    year=2019,
    era=MethodEra.ATTENTION,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Biao Zhang", "Rico Sennrich"],
    paper_title="Root Mean Square Layer Normalization",
    paper_url="https://arxiv.org/abs/1910.07467",
    key_innovation="""
    Simplifies Layer Normalization by removing mean centering, using
    only the root mean square for normalization. Reduces computation
    while maintaining or improving performance, adopted in many modern
    large language models (LLaMA, Mistral, etc.).
    """,
    mathematical_formulation="""
    RMS Normalization:
    ------------------

    For input x of dimension H:

    1. Compute RMS (root mean square):
        RMS(x) = sqrt((1/H) * sum_{i=1}^{H} x_i^2)

    2. Normalize:
        x_hat = x / RMS(x)
        # or with epsilon: x_hat = x / sqrt((1/H) * sum x_i^2 + epsilon)

    3. Scale (no shift by default):
        y = gamma * x_hat

    Where:
        H           = number of features
        gamma       = learnable scale parameter (shape: H)

    Comparison to LayerNorm:
        LayerNorm:  (x - mean(x)) / std(x) * gamma + beta
        RMSNorm:    x / RMS(x) * gamma

        RMSNorm removes:
        - Mean centering (no subtraction of mean)
        - Shift parameter (no beta)

    Why It Works:
        - Re-centering may not be necessary
        - RMS provides sufficient normalization
        - Simpler = potentially more stable

    Computational Savings:
        LayerNorm: needs mean, variance, 2 parameters
        RMSNorm: needs only RMS, 1 parameter
        ~10-15% faster in practice
    """,
    predecessors=["layer_normalization"],
    successors=[],
    tags=["normalization", "transformer", "efficient", "LLM"],
    notes="""
    Adoption:
    - LLaMA (Meta)
    - Mistral
    - Many recent large language models
    - Becoming standard for LLMs

    Why Popular for LLMs:
    - Faster computation
    - Good performance
    - Simpler gradient flow
    - Memory efficient (one fewer parameter)

    Implementation:
        rms = sqrt(mean(x^2) + epsilon)
        output = x / rms * gamma

    Typical epsilon:
    - 1e-5 or 1e-6

    Trade-offs:
    + Faster than LayerNorm
    + Simpler
    + Works well for transformers
    - Less studied than LayerNorm
    - May not work for all architectures
    """
)


def rms_norm_forward(
    x: "ndarray",
    gamma: "ndarray",
    epsilon: float = 1e-5
) -> "ndarray":
    """
    Pseudocode for RMS normalization forward pass.

    Args:
        x: Input of shape (..., H)
        gamma: Scale parameter of shape (H,)
        epsilon: Numerical stability constant

    Pseudocode:
        # Compute RMS
        rms = sqrt(mean(x^2, axis=-1, keepdims=True) + epsilon)

        # Normalize
        x_hat = x / rms

        # Scale (no shift)
        output = gamma * x_hat
    """
    pass


# =============================================================================
# Comparison
# =============================================================================

NORMALIZATION_COMPARISON = """
Normalization Methods Comparison:
=================================

| Method    | Normalizes Over    | Batch Dep | Best For           |
|-----------|-------------------|-----------|---------------------|
| BatchNorm | (N, H, W) per C   | Yes       | CNNs, large batch   |
| LayerNorm | (C, H, W) per N   | No        | Transformers, RNNs  |
| GroupNorm | (C/G, H, W) per G | No        | Small batch CNNs    |
| RMSNorm   | (H,) per sample   | No        | LLMs, efficient     |

For 2D input (N, C, H, W):
    BatchNorm: mean/var over (N, H, W), stats shape (C,)
    GroupNorm: mean/var over (C/G, H, W), G groups
    InstanceNorm: mean/var over (H, W), per channel per sample
    LayerNorm: mean/var over (C, H, W), per sample

Formulas:
    BatchNorm:  (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
    LayerNorm:  (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
    RMSNorm:    x / RMS(x) * gamma

When to Use:
    - CNNs with batch >= 32: BatchNorm
    - CNNs with small batch: GroupNorm
    - Transformers: LayerNorm or RMSNorm
    - LLMs: RMSNorm (efficient)
    - RNNs: LayerNorm
"""


def get_all_normalization_methods():
    """Return all normalization MLMethod entries."""
    return [BATCH_NORMALIZATION, LAYER_NORMALIZATION,
            GROUP_NORMALIZATION, RMS_NORMALIZATION]
