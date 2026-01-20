"""
Dropout Regularization

Dropout and its variants are regularization techniques that randomly
zero out activations during training, preventing co-adaptation and
improving generalization.

Key Methods:
    - Standard Dropout: Random neuron deactivation
    - Spatial Dropout: Drop entire feature maps (for CNNs)
    - DropConnect: Drop connections (weights) instead of activations
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# Standard Dropout
# =============================================================================

DROPOUT = MLMethod(
    method_id="dropout",
    name="Dropout",
    year=2012,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Geoffrey E. Hinton", "Nitish Srivastava", "Alex Krizhevsky",
             "Ilya Sutskever", "Ruslan Salakhutdinov"],
    paper_title="Improving neural networks by preventing co-adaptation of feature detectors",
    paper_url="https://arxiv.org/abs/1207.0580",
    key_innovation="""
    Randomly sets a fraction of neuron activations to zero during training,
    preventing neurons from co-adapting and forcing the network to learn
    more robust features. Equivalent to training an ensemble of sub-networks.
    """,
    mathematical_formulation="""
    Dropout Forward Pass:
    ---------------------

    Training:
        r_i ~ Bernoulli(p)                    # mask: 1 with probability p
        y_i = r_i * x_i                       # apply mask
        # or with inverted dropout:
        y_i = (r_i * x_i) / p                 # scale by 1/p during training

    Inference (no inverted dropout):
        y_i = p * x_i                         # scale by keep probability

    Inference (with inverted dropout):
        y_i = x_i                             # no scaling needed

    Where:
        x_i         = input activation
        r_i         = binary mask (0 or 1)
        p           = keep probability (e.g., 0.5)
        1 - p       = dropout rate

    Inverted Dropout (standard practice):
        Instead of scaling at test time, scale by 1/p during training.
        This keeps the expected value of outputs the same:
        E[y] = E[r * x / p] = p * x / p = x

    Layer Application:
        h = dropout(activation(W @ x + b))

        Typically applied after activation, before next layer.

    Gradient During Training:
        dL/dx_i = dL/dy_i * r_i / p   (inverted)
        dL/dx_i = dL/dy_i * r_i       (standard)

        Gradient is zero for dropped units.

    Ensemble Interpretation:
        - Each training step uses a different "thinned" network
        - With n units, there are 2^n possible sub-networks
        - Training samples all sub-networks
        - Test time approximates geometric mean of all sub-networks
    """,
    predecessors=[],
    successors=["spatial_dropout", "dropconnect", "dropblock"],
    tags=["regularization", "dropout", "ensemble", "deep-learning"],
    notes="""
    Key Properties:
    - Reduces overfitting significantly
    - Approximately trains exponential ensemble
    - Forces redundant representations
    - Acts as noise injection regularizer

    Typical Values:
    - p = 0.5: Standard for fully connected layers
    - p = 0.8-0.9: Common for input layer
    - p = 0.7-0.9: Common for convolutional layers

    Best Practices:
    - Use after activation functions
    - Don't use on batch norm layers (redundant)
    - Lower dropout for larger models
    - Increase epochs when using dropout

    When NOT to Use:
    - Very small datasets (need all signal)
    - Already regularized heavily (redundant)
    - Batch normalization often sufficient

    Theoretical Connections:
    - Similar to L2 regularization in linear models
    - Equivalent to noise injection in linear case
    - Related to Bayesian neural networks
    """
)


def dropout_forward(
    x: "ndarray",
    p: float = 0.5,
    training: bool = True,
    inverted: bool = True
) -> tuple:
    """
    Pseudocode for dropout forward pass.

    Args:
        x: Input activations
        p: Keep probability (NOT drop rate)
        training: Whether in training mode
        inverted: Whether to use inverted dropout

    Returns:
        Tuple of (output, mask)

    Pseudocode:
        if training:
            mask = random_binary(shape=x.shape, p=p)
            if inverted:
                output = x * mask / p
            else:
                output = x * mask
        else:
            if inverted:
                output = x
            else:
                output = x * p
            mask = None
    """
    # if training:
    #     mask = np.random.binomial(1, p, x.shape)
    #     if inverted:
    #         return x * mask / p, mask
    #     else:
    #         return x * mask, mask
    # else:
    #     if inverted:
    #         return x, None
    #     else:
    #         return x * p, None
    pass


def dropout_backward(
    grad_output: "ndarray",
    mask: "ndarray",
    p: float = 0.5,
    inverted: bool = True
) -> "ndarray":
    """
    Pseudocode for dropout backward pass.

    Pseudocode:
        if inverted:
            grad_input = grad_output * mask / p
        else:
            grad_input = grad_output * mask
    """
    pass


# =============================================================================
# Spatial Dropout
# =============================================================================

SPATIAL_DROPOUT = MLMethod(
    method_id="spatial_dropout",
    name="Spatial Dropout",
    year=2015,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Jonathan Tompson", "Ross Goroshin", "Arjun Jain",
             "Yann LeCun", "Christoph Bregler"],
    paper_title="Efficient Object Localization Using Convolutional Networks",
    paper_url="https://arxiv.org/abs/1411.4280",
    key_innovation="""
    Drops entire feature maps instead of individual activations. This is
    more appropriate for convolutional layers where adjacent pixels are
    highly correlated, preventing the network from working around dropout
    by simply copying information to nearby pixels.
    """,
    mathematical_formulation="""
    Spatial Dropout (2D):
    ---------------------

    Standard Dropout on CNN feature maps:
        Drop individual pixels: mask shape = (batch, channels, height, width)
        Problem: Adjacent pixels correlated, info can "leak" around dropped pixels

    Spatial Dropout:
        Drop entire channels: mask shape = (batch, channels, 1, 1)
        Mask is broadcast across spatial dimensions

    For input x of shape (N, C, H, W):
        r ~ Bernoulli(p) of shape (N, C, 1, 1)
        y = x * r / p                          (inverted dropout)

    Pseudocode:
        # Generate per-channel mask
        mask = random_binary(shape=(batch_size, num_channels, 1, 1), p=p)
        # Apply to all spatial positions
        output = input * mask / p

    1D Variant (for sequences):
        For input x of shape (N, C, L):
        r ~ Bernoulli(p) of shape (N, C, 1)
        y = x * r / p

    3D Variant (for video/volumetric):
        For input x of shape (N, C, D, H, W):
        r ~ Bernoulli(p) of shape (N, C, 1, 1, 1)
    """,
    predecessors=["dropout"],
    successors=["dropblock"],
    tags=["regularization", "dropout", "convolutional", "spatial"],
    notes="""
    When to Use:
    - Convolutional neural networks
    - Spatially correlated inputs
    - When standard dropout ineffective for CNNs

    Typical Values:
    - p = 0.7-0.9 (higher than standard dropout)
    - More aggressive than standard dropout

    Comparison to Standard Dropout:
    - Standard: drops ~50% of pixels
    - Spatial: drops ~50% of channels (more aggressive)

    Implementation in Frameworks:
    - PyTorch: nn.Dropout2d, nn.Dropout3d
    - TensorFlow: tf.keras.layers.SpatialDropout2D
    """
)


def spatial_dropout_forward(
    x: "ndarray",
    p: float = 0.5,
    training: bool = True
) -> tuple:
    """
    Pseudocode for spatial dropout forward pass.

    Args:
        x: Input of shape (N, C, H, W) for 2D
        p: Keep probability
        training: Whether in training mode

    Pseudocode:
        if training:
            # Mask shape: (N, C, 1, 1) - same mask for all spatial positions
            mask = random_binary(shape=(N, C, 1, 1), p=p)
            output = x * mask / p
        else:
            output = x
    """
    pass


# =============================================================================
# DropConnect
# =============================================================================

DROPCONNECT = MLMethod(
    method_id="dropconnect",
    name="DropConnect",
    year=2013,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Li Wan", "Matthew Zeiler", "Sixin Zhang",
             "Yann LeCun", "Rob Fergus"],
    paper_title="Regularization of Neural Networks using DropConnect",
    paper_url="https://proceedings.mlr.press/v28/wan13.html",
    key_innovation="""
    Drops connections (weights) instead of activations. This generalizes
    dropout - when a weight is dropped, only that specific input-output
    pathway is disabled, not the entire neuron. Provides finer-grained
    regularization.
    """,
    mathematical_formulation="""
    DropConnect:
    ------------

    Standard fully connected layer:
        y = W @ x + b

    With DropConnect:
        M ~ Bernoulli(p)                  # mask same shape as W
        y = (M odot W) @ x + b            # element-wise mask on weights

    Comparison to Dropout:
        Dropout:      y = W @ (m odot x) + b     (mask on activations)
        DropConnect:  y = (M odot W) @ x + b     (mask on weights)

    Ensemble Size:
        Dropout:      2^n sub-networks (n = neurons)
        DropConnect:  2^m sub-networks (m = weights >> n)

    Inference:
        The expected output is:
        E[y] = (p * W) @ x + b

        But variance is higher than dropout, so sampling is recommended:
        - Sample k random masks
        - Compute mean prediction

    Theoretical Analysis:
        - More sub-networks than dropout
        - Finer-grained regularization
        - Higher inference cost (if sampling)

    Pseudocode:
        Training:
            mask = random_binary(shape=W.shape, p=p)
            W_masked = mask * W
            output = W_masked @ input + bias

        Inference (approximation):
            output = (p * W) @ input + bias

        Inference (sampling):
            outputs = []
            for k times:
                mask = random_binary(shape=W.shape, p=p)
                outputs.append((mask * W) @ input + bias)
            output = mean(outputs)
    """,
    predecessors=["dropout"],
    successors=[],
    tags=["regularization", "dropout", "connections", "weights"],
    notes="""
    Comparison to Dropout:
    - DropConnect: mask weights (more granular)
    - Dropout: mask activations (coarser)
    - DropConnect has more sub-networks

    Trade-offs:
    + More expressive regularization
    + Larger implicit ensemble
    - Higher computational cost
    - More complex inference

    When to Use:
    - When dropout is insufficient
    - Fully connected layers (expensive for conv)
    - When can afford inference sampling

    Practical Notes:
    - Less commonly used than standard dropout
    - Dropout usually sufficient in practice
    - Inference sampling increases cost
    - Implementation more complex
    """
)


def dropconnect_forward(
    x: "ndarray",
    W: "ndarray",
    b: "ndarray",
    p: float = 0.5,
    training: bool = True
) -> tuple:
    """
    Pseudocode for DropConnect forward pass.

    Pseudocode:
        if training:
            mask = random_binary(shape=W.shape, p=p)
            output = (mask * W / p) @ x + b
        else:
            output = W @ x + b
    """
    pass


# =============================================================================
# Comparison
# =============================================================================

DROPOUT_VARIANTS_COMPARISON = """
Dropout Variants Comparison:
============================

| Method          | What's Dropped   | Use Case             | Ensemble Size |
|-----------------|------------------|----------------------|---------------|
| Dropout         | Activations      | Fully connected      | 2^neurons     |
| Spatial Dropout | Feature maps     | Convolutional        | 2^channels    |
| DropConnect     | Weights          | FC (more granular)   | 2^weights     |
| DropBlock       | Contiguous regions| CNN (structured)    | varies        |

Common Keep Probabilities:
- Dropout (FC): 0.5
- Dropout (Conv): 0.8-0.9
- Spatial Dropout: 0.7-0.9
- DropConnect: 0.5

Modern Practice:
- Standard dropout in FC layers
- Less dropout with batch norm
- Spatial dropout for CNNs
- Transformers: dropout 0.1 typical
"""


def get_all_dropout_methods():
    """Return all dropout MLMethod entries."""
    return [DROPOUT, SPATIAL_DROPOUT, DROPCONNECT]
