"""
Multi-Layer Perceptron (MLP) - 1986

Research index entry for the multi-layer perceptron architecture,
which demonstrated that neural networks with hidden layers could
learn arbitrary mappings through the universal approximation theorem.

Key developments:
- Multiple hidden layers with nonlinear activation functions
- Universal approximation capability (Cybenko 1989, Hornik 1991)
- Foundation for deep learning architectures
"""

from typing import Dict, List

from ..core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Multi-Layer Perceptron."""
    return MLMethod(
        method_id="mlp_1986",
        name="Multi-Layer Perceptron",
        year=1986,
        era=MethodEra.CLASSICAL,
        category=MethodCategory.ARCHITECTURE,
        lineages=[MethodLineage.PERCEPTRON_LINE, MethodLineage.PERCEPTRON],
        authors=[
            "David E. Rumelhart",
            "Geoffrey E. Hinton",
            "Ronald J. Williams",
        ],
        paper_title="Learning Internal Representations by Error Propagation",
        paper_url="https://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapter8.pdf",
        key_innovation="""
        The MLP architecture introduced hidden layers between input and output,
        allowing networks to learn internal representations. Combined with
        backpropagation, this enabled learning of complex nonlinear mappings
        that single-layer perceptrons could not represent (solving XOR problem).

        The universal approximation theorem (proven later) showed that MLPs
        with a single hidden layer and sufficient neurons can approximate
        any continuous function to arbitrary precision.
        """,
        mathematical_formulation="""
        For an L-layer MLP:

        Forward pass (layer l):
            z^(l) = W^(l) * a^(l-1) + b^(l)
            a^(l) = f(z^(l))

        Where:
            a^(0) = x (input)
            a^(L) = y_hat (output prediction)
            f = activation function (sigmoid, tanh, ReLU)
            W^(l) = weight matrix for layer l
            b^(l) = bias vector for layer l

        Common activation functions:
            Sigmoid: f(z) = 1 / (1 + exp(-z))
            Tanh: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
            ReLU: f(z) = max(0, z)
        """,
        predecessors=["perceptron_1958", "adaline_1960"],
        successors=["deep_networks", "residual_networks"],
        tags=[
            "feedforward",
            "hidden_layers",
            "universal_approximation",
            "activation_functions",
            "nonlinear",
        ],
        notes="""
        The MLP architecture became the foundation for most neural network
        models. The choice of activation function, number of layers, and
        neurons per layer became key hyperparameters. Training challenges
        (vanishing gradients, local minima) led to later innovations like
        ReLU activation and batch normalization.
        """,
    )


def pseudocode() -> str:
    """Return pseudocode for MLP forward and backward pass."""
    return """
    MULTI-LAYER PERCEPTRON ALGORITHM
    ================================

    ARCHITECTURE DEFINITION:
        layers = [input_dim, hidden1_dim, ..., hiddenN_dim, output_dim]
        Initialize weights W[l] ~ N(0, sqrt(2/fan_in))  # Xavier/He init
        Initialize biases b[l] = 0

    FORWARD PASS:
        a[0] = x  # input

        for l in 1 to L:
            z[l] = W[l] @ a[l-1] + b[l]  # linear transformation
            a[l] = activation(z[l])       # nonlinear activation

        return a[L]  # output prediction

    BACKWARD PASS (with backpropagation):
        # Output layer error
        delta[L] = (a[L] - y) * activation_derivative(z[L])

        # Hidden layer errors (backpropagate)
        for l in L-1 down to 1:
            delta[l] = (W[l+1].T @ delta[l+1]) * activation_derivative(z[l])

        # Compute gradients
        for l in 1 to L:
            dW[l] = delta[l] @ a[l-1].T
            db[l] = delta[l]

    WEIGHT UPDATE (gradient descent):
        for l in 1 to L:
            W[l] = W[l] - learning_rate * dW[l]
            b[l] = b[l] - learning_rate * db[l]

    TRAINING LOOP:
        for epoch in 1 to max_epochs:
            for batch in training_data:
                predictions = FORWARD_PASS(batch.x)
                loss = compute_loss(predictions, batch.y)
                BACKWARD_PASS(loss)
                WEIGHT_UPDATE()

            if validation_loss not improving:
                early_stop()
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for MLP."""
    return {
        "forward_linear": "z^(l) = W^(l) a^(l-1) + b^(l)",
        "forward_activation": "a^(l) = f(z^(l))",
        "sigmoid": "sigma(z) = 1 / (1 + e^(-z))",
        "tanh": "tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))",
        "relu": "ReLU(z) = max(0, z)",
        "sigmoid_derivative": "sigma'(z) = sigma(z)(1 - sigma(z))",
        "tanh_derivative": "tanh'(z) = 1 - tanh^2(z)",
        "relu_derivative": "ReLU'(z) = 1 if z > 0 else 0",
        "mse_loss": "L = (1/n) sum((y_hat - y)^2)",
        "cross_entropy": "L = -sum(y * log(y_hat))",
        "universal_approximation": "For continuous f on compact K, exists MLP g: ||f - g||_inf < epsilon",
        "xavier_init": "W ~ N(0, sqrt(2 / (fan_in + fan_out)))",
        "he_init": "W ~ N(0, sqrt(2 / fan_in))",
    }


# Convenience functions for method exploration
def get_activation_functions() -> List[Dict[str, str]]:
    """Return list of common activation functions with properties."""
    return [
        {
            "name": "Sigmoid",
            "formula": "1 / (1 + exp(-z))",
            "range": "(0, 1)",
            "pros": "Smooth, bounded, probabilistic interpretation",
            "cons": "Vanishing gradients, not zero-centered",
        },
        {
            "name": "Tanh",
            "formula": "(exp(z) - exp(-z)) / (exp(z) + exp(-z))",
            "range": "(-1, 1)",
            "pros": "Zero-centered, stronger gradients than sigmoid",
            "cons": "Still suffers from vanishing gradients",
        },
        {
            "name": "ReLU",
            "formula": "max(0, z)",
            "range": "[0, inf)",
            "pros": "No vanishing gradient (positive), computationally efficient",
            "cons": "Dead neurons, not zero-centered",
        },
        {
            "name": "Leaky ReLU",
            "formula": "max(alpha*z, z) where alpha ~ 0.01",
            "range": "(-inf, inf)",
            "pros": "No dead neurons",
            "cons": "Inconsistent predictions across runs",
        },
        {
            "name": "ELU",
            "formula": "z if z > 0 else alpha*(exp(z) - 1)",
            "range": "(-alpha, inf)",
            "pros": "Zero-centered outputs, no dead neurons",
            "cons": "Computationally more expensive",
        },
        {
            "name": "GELU",
            "formula": "z * Phi(z) where Phi is standard normal CDF",
            "range": "(-inf, inf)",
            "pros": "Smooth, used in transformers",
            "cons": "Computationally expensive",
        },
    ]


def get_universal_approximation_details() -> Dict[str, str]:
    """Return details about the universal approximation theorem."""
    return {
        "theorem_statement": """
            Let f be any continuous function on a compact subset K of R^n.
            For any epsilon > 0, there exists a feedforward network g with
            a single hidden layer and finite number of neurons such that
            |f(x) - g(x)| < epsilon for all x in K.
        """,
        "cybenko_1989": """
            Cybenko (1989) proved UAT for sigmoid activation functions,
            showing sigmoidal networks are dense in C(I^n).
        """,
        "hornik_1991": """
            Hornik (1991) extended to broader class of activation functions,
            showing result holds for any non-polynomial activation.
        """,
        "practical_implications": """
            - Single hidden layer is theoretically sufficient
            - But may require exponentially many neurons
            - Deep networks can be exponentially more efficient
            - Depth vs width tradeoff is problem-dependent
        """,
        "limitations": """
            - Existence theorem, not construction
            - Does not guarantee learnability
            - Does not specify required network size
            - Does not address optimization difficulty
        """,
    }
