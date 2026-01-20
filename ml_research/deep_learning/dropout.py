"""
Dropout (2014)

Dropout: A Simple Way to Prevent Neural Networks from Overfitting
Authors: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky,
         Ilya Sutskever, Ruslan Salakhutdinov

Dropout is a regularization technique that randomly "drops out" (sets to zero)
a fraction of neurons during training. Each neuron is kept with probability p
(typically 0.5). This prevents complex co-adaptations between neurons and
forces the network to learn more robust features.

Key Insight:
    Training with dropout is approximately equivalent to averaging an
    exponentially large ensemble of neural networks that share weights.
    Each training example uses a different "thinned" network.

Key Benefits:
    - Prevents overfitting (implicit ensemble regularization)
    - Reduces neuron co-adaptation
    - Provides approximate Bayesian inference
    - Simple to implement

Implementation:
    - Training: Randomly mask neurons with probability (1-p)
    - Inference: Use all neurons, scale by p (or scale at training time)
"""


from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from typing import Dict, List


DROPOUT = MLMethod(
    method_id="dropout_2014",
    name="Dropout",
    year=2014,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Nitish Srivastava", "Geoffrey Hinton", "Alex Krizhevsky",
             "Ilya Sutskever", "Ruslan Salakhutdinov"],
    paper_title="Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
    paper_url="https://jmlr.org/papers/v15/srivastava14a.html",
    key_innovation="Randomly setting a fraction of neurons to zero during training, "
                   "creating an implicit ensemble of exponentially many networks that "
                   "share weights, effectively regularizing the model",
    mathematical_formulation="""
    Standard Dropout (Training):
        r ~ Bernoulli(p)           # Mask: r_i = 1 with probability p
        y = r * x                   # Element-wise multiplication
        Forward pass uses y instead of x

    Standard Dropout (Inference):
        y = p * x                   # Scale activations by keep probability

    Inverted Dropout (Training) - More Common:
        r ~ Bernoulli(p)
        y = (r * x) / p            # Scale during training
        No scaling needed at inference

    Ensemble Interpretation:
        Each training step samples one of 2^n possible networks
        (where n is number of dropout-able units)
        At test time, the full network approximates the ensemble average

    Approximate Weight Scaling:
        E[y] = p * x               # Expected value with dropout
        Full network outputs ≈ ensemble average when scaled by p
    """,
    predecessors=["neural_networks", "ensemble_methods"],
    successors=["dropconnect_2013", "spatial_dropout_2015", "variational_dropout_2016"],
    tags=["regularization", "ensemble", "overfitting", "co_adaptation"],
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Dropout."""
    return DROPOUT


def pseudocode() -> str:
    """Return pseudocode describing Dropout."""
    return """
    DROPOUT (Standard Version):

    function dropout_train(x, p=0.5):
        # x: input activations (any shape)
        # p: keep probability (typically 0.5)

        # Generate random mask
        mask = random_uniform(shape=x.shape) < p  # Boolean mask
        mask = mask.astype(float)  # Convert to 0.0/1.0

        # Apply mask (drop neurons)
        y = x * mask

        # Store mask for backward pass
        cache = mask

        return y, cache


    function dropout_inference(x, p=0.5):
        # At inference, scale by keep probability
        y = x * p
        return y


    function dropout_backward(dy, cache, p):
        # Gradient only flows through kept neurons
        mask = cache
        dx = dy * mask
        return dx


    INVERTED DROPOUT (More Common):

    function inverted_dropout_train(x, p=0.5):
        # Scale during training so no scaling needed at test time

        # Generate random mask
        mask = (random_uniform(shape=x.shape) < p).astype(float)

        # Apply mask and scale
        y = (x * mask) / p  # Scale by 1/p during training

        return y, mask


    function inverted_dropout_inference(x):
        # No scaling needed!
        return x


    DROPOUT IN NEURAL NETWORK:

    function forward_with_dropout(x, weights, training=True):
        # Layer 1
        z1 = matmul(x, W1) + b1
        a1 = relu(z1)
        if training:
            a1, mask1 = inverted_dropout_train(a1, p=0.5)

        # Layer 2
        z2 = matmul(a1, W2) + b2
        a2 = relu(z2)
        if training:
            a2, mask2 = inverted_dropout_train(a2, p=0.5)

        # Output layer (no dropout on output)
        z3 = matmul(a2, W3) + b3
        output = softmax(z3)

        return output


    DROPOUT VARIANTS:

    function spatial_dropout_2d(x, p=0.5):
        # For CNNs: drop entire feature maps (channels)
        # x shape: (N, C, H, W)
        mask = (random_uniform(shape=(N, C, 1, 1)) < p).astype(float)
        y = (x * mask) / p
        return y


    function dropconnect(x, W, p=0.5):
        # Drop weights instead of activations
        mask = (random_uniform(shape=W.shape) < p).astype(float)
        W_masked = (W * mask) / p
        y = matmul(x, W_masked)
        return y


    function dropout_scheduled(x, p_initial, p_final, current_epoch, total_epochs):
        # Gradually increase dropout rate during training
        p = p_initial + (p_final - p_initial) * (current_epoch / total_epochs)
        return inverted_dropout_train(x, p)
    """


def key_equations() -> Dict[str, str]:
    """Return key equations for Dropout in LaTeX-style notation."""
    return {
        "bernoulli_mask":
            "r_j \\sim \\text{Bernoulli}(p)",

        "training_forward":
            "\\tilde{y} = r \\odot y, \\quad \\text{where } y = f(Wx + b)",

        "inference_forward":
            "y_{test} = p \\cdot f(Wx + b)",

        "inverted_training":
            "\\tilde{y} = \\frac{r \\odot y}{p}",

        "expected_value":
            "\\mathbb{E}[\\tilde{y}] = p \\cdot y",

        "ensemble_size":
            "|\\text{ensemble}| = 2^n \\text{ (for } n \\text{ droppable units)}",

        "geometric_mean_approx":
            "\\hat{y}_{ensemble} \\approx \\left(\\prod_{m} y_m^{1/M}\\right) \\approx p \\cdot y",

        "gradient_masking":
            "\\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial \\tilde{y}} \\odot r",
    }


def dropout_guidelines() -> Dict[str, str]:
    """Return guidelines for using dropout effectively."""
    return {
        "typical_rates": {
            "input_layer": "p = 0.8 (keep 80%)",
            "hidden_layers": "p = 0.5 (keep 50%)",
            "output_layer": "No dropout",
            "convolutional_layers": "p = 0.8 or spatial dropout",
        },
        "when_to_use": [
            "Large networks prone to overfitting",
            "Limited training data",
            "Fully connected layers (especially)",
            "When ensemble effect is desired",
        ],
        "when_not_to_use": [
            "Very small networks (already limited capacity)",
            "With batch normalization (may conflict)",
            "In attention mechanisms (usually)",
            "When training data is abundant",
        ],
        "practical_tips": [
            "Use inverted dropout (scale during training)",
            "Higher dropout may require more training epochs",
            "Can combine with L2 regularization",
            "Increase network size when using dropout",
        ],
    }


def get_historical_context() -> str:
    """Return historical context and significance of Dropout."""
    return """
    Dropout (2014) revolutionized how we regularize neural networks,
    providing a simple yet powerful technique for preventing overfitting.

    Origins:
    - Geoffrey Hinton conceived dropout while thinking about how sexual
      reproduction evolved to prevent co-adaptation of genes
    - If genes that work well together always appear together, complex
      co-adaptations form that break when genes are recombined
    - Similarly, neurons that only work together can't generalize well

    Key Insights:
    1. Ensemble Interpretation: Training with dropout implicitly trains
       an exponential number of weight-sharing networks
    2. Co-adaptation Prevention: Neurons can't rely on specific other
       neurons being present, so must learn robust features
    3. Feature Redundancy: Forces networks to learn redundant representations
    4. Bayesian Connection: Provides approximate Bayesian inference

    Impact:
    - First major neural network regularization technique
    - Enabled training of much larger networks without overfitting
    - Used in AlexNet, VGG, and many subsequent architectures
    - Led to many variants: DropConnect, SpatialDropout, DropBlock
    - Partially superseded by Batch Normalization in some contexts

    Relationship to Other Techniques:
    - Batch Normalization provides regularization and may reduce need for dropout
    - Modern architectures (ResNets, Transformers) often use dropout minimally
    - Still widely used in recurrent networks and attention mechanisms
    """


def get_limitations() -> List[str]:
    """Return known limitations of Dropout."""
    return [
        "Increases training time (more epochs needed for convergence)",
        "May conflict with Batch Normalization",
        "Not well-suited for convolutional layers (use SpatialDropout)",
        "Fixed dropout rate may not be optimal for all layers",
        "Monte Carlo dropout for uncertainty requires multiple forward passes",
        "Standard dropout doesn't provide meaningful uncertainty estimates",
    ]


def get_applications() -> List[str]:
    """Return applications of Dropout."""
    return [
        "Regularizing fully connected layers",
        "Preventing overfitting in any neural network",
        "Monte Carlo dropout for uncertainty estimation",
        "Training large language models (moderate dropout)",
        "Computer vision (with spatial dropout variant)",
        "Recurrent neural networks (variational dropout)",
        "Ensemble-like predictions from single network",
    ]


def dropout_variants() -> Dict[str, str]:
    """Return descriptions of dropout variants."""
    return {
        "DropConnect (2013)": "Drops weights instead of activations",
        "Spatial Dropout (2015)": "Drops entire feature maps in CNNs",
        "Variational Dropout (2016)": "Learned, data-dependent dropout rates",
        "DropBlock (2018)": "Drops contiguous regions of feature maps",
        "Concrete Dropout (2017)": "Dropout rate learned via continuous relaxation",
        "Targeted Dropout (2019)": "Dropout targeting specific weight magnitudes",
        "R-Drop (2021)": "Regularize dropout by minimizing KL divergence between sub-models",
    }
