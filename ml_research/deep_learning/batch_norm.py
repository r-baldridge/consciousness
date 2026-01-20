"""
Batch Normalization (2015)

Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift
Authors: Sergey Ioffe, Christian Szegedy

Batch Normalization (BatchNorm) normalizes layer inputs to have zero mean
and unit variance within each mini-batch, then applies a learnable scale
and shift. This seemingly simple technique had a profound impact on deep
learning, enabling much faster training and higher learning rates.

Key Insight - Internal Covariate Shift:
    During training, the distribution of each layer's inputs changes as the
    parameters of the preceding layers change. This "internal covariate shift"
    makes training difficult. BatchNorm addresses this by normalizing inputs.

Key Benefits:
    - Enables much higher learning rates (10-100x in some cases)
    - Reduces sensitivity to weight initialization
    - Acts as a regularizer (reduces need for dropout)
    - Smooths the loss landscape
    - Enables training of very deep networks

The Technique:
    1. Compute mean and variance over mini-batch
    2. Normalize: (x - mean) / sqrt(var + epsilon)
    3. Scale and shift: gamma * x_norm + beta (learnable)
"""


from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from typing import Dict, List


BATCH_NORMALIZATION = MLMethod(
    method_id="batch_normalization_2015",
    name="Batch Normalization",
    year=2015,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Sergey Ioffe", "Christian Szegedy"],
    paper_title="Batch Normalization: Accelerating Deep Network Training by "
                "Reducing Internal Covariate Shift",
    paper_url="https://arxiv.org/abs/1502.03167",
    key_innovation="Normalizing layer inputs within each mini-batch followed by "
                   "learnable scale and shift parameters, dramatically accelerating "
                   "training and enabling higher learning rates",
    mathematical_formulation="""
    Given a mini-batch B = {x_1, ..., x_m}:

    1. Mini-batch mean:
        mu_B = (1/m) * sum_{i=1}^{m} x_i

    2. Mini-batch variance:
        sigma_B^2 = (1/m) * sum_{i=1}^{m} (x_i - mu_B)^2

    3. Normalize:
        x_hat_i = (x_i - mu_B) / sqrt(sigma_B^2 + epsilon)

    4. Scale and shift (learnable parameters gamma, beta):
        y_i = gamma * x_hat_i + beta = BN_{gamma,beta}(x_i)

    At inference time, use running estimates:
        mu_running = momentum * mu_running + (1 - momentum) * mu_B
        sigma_running^2 = momentum * sigma_running^2 + (1 - momentum) * sigma_B^2
        y = gamma * ((x - mu_running) / sqrt(sigma_running^2 + epsilon)) + beta
    """,
    predecessors=["deep_networks", "normalization_methods"],
    successors=["layer_normalization_2016", "group_normalization_2018", "instance_normalization_2016"],
    tags=["normalization", "regularization", "training_acceleration", "optimization"],
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Batch Normalization."""
    return BATCH_NORMALIZATION


def pseudocode() -> str:
    """Return pseudocode describing Batch Normalization."""
    return """
    BATCH NORMALIZATION (Training):

    function BatchNorm_train(x, gamma, beta, epsilon=1e-5):
        # x: input tensor of shape (batch_size, ..., channels)
        # gamma, beta: learnable parameters (per channel)
        # For 2D inputs: (N, C) - normalize over N
        # For 4D inputs: (N, C, H, W) - normalize over N, H, W

        # Step 1: Compute mini-batch statistics
        mu = mean(x, axis=batch_axes)       # Shape: (C,)
        var = variance(x, axis=batch_axes)  # Shape: (C,)

        # Step 2: Normalize
        x_norm = (x - mu) / sqrt(var + epsilon)

        # Step 3: Scale and shift
        y = gamma * x_norm + beta

        # Step 4: Update running statistics (for inference)
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var

        return y


    BATCH NORMALIZATION (Inference):

    function BatchNorm_inference(x, gamma, beta, running_mean, running_var, epsilon=1e-5):
        # Use running statistics instead of batch statistics
        x_norm = (x - running_mean) / sqrt(running_var + epsilon)
        y = gamma * x_norm + beta
        return y


    BATCH NORMALIZATION PLACEMENT:

    # Original paper: BN after linear, before activation
    def conv_bn_relu(x):
        x = Conv2D(x)
        x = BatchNorm(x)
        x = ReLU(x)
        return x

    # Alternative (sometimes used): BN after activation
    def conv_relu_bn(x):
        x = Conv2D(x)
        x = ReLU(x)
        x = BatchNorm(x)
        return x


    GRADIENT COMPUTATION:

    function BatchNorm_backward(dy, x, gamma, mu, var, epsilon):
        # Compute gradients for backpropagation
        N = batch_size

        # Gradient w.r.t. gamma and beta
        x_norm = (x - mu) / sqrt(var + epsilon)
        dgamma = sum(dy * x_norm, axis=batch_axes)
        dbeta = sum(dy, axis=batch_axes)

        # Gradient w.r.t. input x
        dx_norm = dy * gamma
        dvar = sum(dx_norm * (x - mu) * (-0.5) * (var + epsilon)^(-1.5), axis=batch)
        dmu = sum(dx_norm * (-1/sqrt(var + epsilon)), axis=batch) + dvar * (-2/N) * sum(x - mu)
        dx = dx_norm / sqrt(var + epsilon) + dvar * (2/N) * (x - mu) + dmu / N

        return dx, dgamma, dbeta


    TRAINING WITH BATCH NORMALIZATION:

    function train_with_batchnorm(model, dataset):
        # Benefits of BatchNorm:

        # 1. Higher learning rate
        optimizer = SGD(lr=0.1)  # vs 0.01 without BatchNorm

        # 2. Less careful initialization
        init = random_normal(std=0.01)  # Works fine with BatchNorm

        # 3. Can remove dropout (optional)
        # BatchNorm provides regularization effect

        # 4. Faster convergence
        for batch in dataset:
            # Forward pass (BatchNorm computes batch statistics)
            predictions = model(batch.images, training=True)
            loss = compute_loss(predictions, batch.labels)

            # Backward pass
            gradients = backprop(loss)
            optimizer.step(gradients)

        return model
    """


def key_equations() -> Dict[str, str]:
    """Return key equations for Batch Normalization in LaTeX-style notation."""
    return {
        "batch_mean":
            "\\mu_B = \\frac{1}{m} \\sum_{i=1}^{m} x_i",

        "batch_variance":
            "\\sigma_B^2 = \\frac{1}{m} \\sum_{i=1}^{m} (x_i - \\mu_B)^2",

        "normalize":
            "\\hat{x}_i = \\frac{x_i - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}}",

        "scale_shift":
            "y_i = \\gamma \\hat{x}_i + \\beta \\equiv \\text{BN}_{\\gamma,\\beta}(x_i)",

        "running_mean_update":
            "\\mu_{running} \\leftarrow \\alpha \\mu_{running} + (1-\\alpha) \\mu_B",

        "running_var_update":
            "\\sigma^2_{running} \\leftarrow \\alpha \\sigma^2_{running} + (1-\\alpha) \\sigma_B^2",

        "inference_normalize":
            "y = \\gamma \\frac{x - \\mu_{running}}{\\sqrt{\\sigma^2_{running} + \\epsilon}} + \\beta",

        "gradient_wrt_gamma":
            "\\frac{\\partial L}{\\partial \\gamma} = \\sum_{i=1}^{m} \\frac{\\partial L}{\\partial y_i} \\cdot \\hat{x}_i",

        "gradient_wrt_beta":
            "\\frac{\\partial L}{\\partial \\beta} = \\sum_{i=1}^{m} \\frac{\\partial L}{\\partial y_i}",
    }


def normalization_variants() -> Dict[str, Dict]:
    """Return descriptions of normalization variants."""
    return {
        "batch_norm": {
            "normalize_over": "Batch dimension (and spatial for CNNs)",
            "stats_shape": "(C,) for features, one per channel",
            "use_case": "Standard for CNNs with large batches",
            "limitation": "Poor with small batches, not for RNNs",
        },
        "layer_norm": {
            "normalize_over": "Feature dimension (all features per sample)",
            "stats_shape": "Per sample, across all features",
            "use_case": "RNNs, Transformers, small batches",
            "limitation": "May not work well for CNNs",
        },
        "instance_norm": {
            "normalize_over": "Spatial dimensions only (per channel, per sample)",
            "stats_shape": "(N, C) for CNN feature maps",
            "use_case": "Style transfer, image generation",
            "limitation": "Loses batch statistics, not for classification",
        },
        "group_norm": {
            "normalize_over": "Groups of channels (within each sample)",
            "stats_shape": "(N, G) where G is number of groups",
            "use_case": "Small batches, object detection",
            "limitation": "Requires tuning number of groups",
        },
    }


def get_historical_context() -> str:
    """Return historical context and significance of Batch Normalization."""
    return """
    Batch Normalization (2015) transformed deep learning practice, enabling
    training of networks that were previously intractable.

    The Problem:
    - Deep networks were hard to train (vanishing/exploding gradients)
    - Required careful initialization (e.g., Xavier, He initialization)
    - Learning rates had to be small
    - Training was slow (many epochs needed)

    The Original Explanation (Internal Covariate Shift):
    - Layer inputs change distribution during training
    - Layers must constantly adapt to new distributions
    - This slows learning and requires low learning rates
    - BatchNorm fixes distributions, accelerating training

    Modern Understanding:
    - The "internal covariate shift" explanation is now disputed
    - BatchNorm's success is more likely due to:
        * Smoothing the loss landscape (making it more convex)
        * Enabling larger learning rates
        * Providing regularization (similar to dropout)
        * Improving gradient flow
    - Paper "How Does Batch Normalization Help Optimization?" (NeurIPS 2018)
      showed the smoothing effect empirically

    Impact:
    - Standard component in nearly all modern architectures
    - Enabled training of ResNets and other very deep networks
    - Spawned many variants (LayerNorm for Transformers, GroupNorm, etc.)
    - One of the most cited deep learning papers
    """


def get_limitations() -> List[str]:
    """Return known limitations of Batch Normalization."""
    return [
        "Poor performance with small batch sizes (statistics become noisy)",
        "Different behavior at train vs inference time (train/test discrepancy)",
        "Not suitable for RNNs (sequence length varies)",
        "Memory overhead for storing running statistics",
        "Batch statistics create dependence between samples",
        "Distributed training requires synchronized batch statistics",
    ]


def get_applications() -> List[str]:
    """Return applications of Batch Normalization."""
    return [
        "Convolutional neural networks (standard practice)",
        "Image classification",
        "Object detection",
        "Semantic segmentation",
        "Generative adversarial networks",
        "Any feedforward network with large batch sizes",
    ]
