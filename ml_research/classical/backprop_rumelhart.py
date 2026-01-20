"""
Modern Backpropagation - Rumelhart, Hinton, Williams (1986)

Research index entry for the modern formulation of backpropagation,
which enabled training of multi-layer neural networks and sparked
the connectionist revival of the 1980s.

Paper: "Learning representations by back-propagating errors"
Nature, 323(6088), 533-536

Key contributions:
- Clear presentation of chain rule for multi-layer networks
- Demonstration of learning internal representations
- Foundation for all gradient-based deep learning
"""

from typing import Dict, List

from ..core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Backpropagation (Rumelhart et al.)."""
    return MLMethod(
        method_id="backprop_rumelhart_1986",
        name="Backpropagation (Rumelhart, Hinton, Williams)",
        year=1986,
        era=MethodEra.CLASSICAL,
        category=MethodCategory.LEARNING_RULE,
        lineages=[MethodLineage.PERCEPTRON_LINE, MethodLineage.PERCEPTRON_LINE],
        authors=[
            "David E. Rumelhart",
            "Geoffrey E. Hinton",
            "Ronald J. Williams",
        ],
        paper_title="Learning representations by back-propagating errors",
        paper_url="https://www.nature.com/articles/323533a0",
        key_innovation="""
        While the chain rule and error backpropagation were known earlier
        (Werbos 1974, Linnainmaa 1970), Rumelhart, Hinton, and Williams
        provided the clearest and most influential presentation, demonstrating
        that multi-layer networks could learn useful internal representations.

        Key insights:
        1. Error signals can propagate backward through layers via chain rule
        2. Hidden units learn to encode meaningful features automatically
        3. The "credit assignment problem" is solved by gradient flow

        This paper ignited the connectionist movement and enabled the
        training of networks with multiple hidden layers.
        """,
        mathematical_formulation="""
        For network with layers l = 1, ..., L:

        Forward pass:
            a^(0) = x (input)
            z^(l) = W^(l) a^(l-1) + b^(l)
            a^(l) = f(z^(l))

        Error at output layer:
            delta^(L) = nabla_a L * f'(z^(L))

        Backpropagation (chain rule):
            delta^(l) = ((W^(l+1))^T delta^(l+1)) * f'(z^(l))

        Gradients:
            dL/dW^(l) = delta^(l) (a^(l-1))^T
            dL/db^(l) = delta^(l)

        Weight update:
            W^(l) := W^(l) - eta * dL/dW^(l)
            b^(l) := b^(l) - eta * dL/db^(l)
        """,
        predecessors=[
            "perceptron_1958",
            "adaline_1960",
            "werbos_backprop_1974",
        ],
        successors=[
            "momentum_sgd",
            "adam_optimizer",
            "batch_normalization",
            "residual_connections",
        ],
        tags=[
            "gradient_descent",
            "chain_rule",
            "credit_assignment",
            "learning_rule",
            "PDP",
            "connectionism",
        ],
        notes="""
        Historical context: Backpropagation was independently discovered
        multiple times (Bryson & Ho 1969, Werbos 1974, Rumelhart et al. 1986).
        The 1986 Nature paper had the most impact due to its clarity and
        demonstration of learning internal representations.

        The paper was part of the influential PDP (Parallel Distributed
        Processing) research group that helped establish connectionism
        as a computational paradigm for cognitive science.
        """,
    )


def pseudocode() -> str:
    """Return pseudocode for the backpropagation algorithm."""
    return """
    BACKPROPAGATION ALGORITHM
    =========================

    INPUT:
        - Training data: {(x_1, y_1), ..., (x_n, y_n)}
        - Network architecture: layers, activation functions
        - Learning rate: eta
        - Number of epochs: T

    INITIALIZATION:
        Initialize weights W[l] randomly (small values)
        Initialize biases b[l] = 0

    MAIN LOOP:
        for epoch in 1 to T:
            for each (x, y) in training_data:

                # ========== FORWARD PASS ==========
                a[0] = x
                for l in 1 to L:
                    z[l] = W[l] @ a[l-1] + b[l]
                    a[l] = activation(z[l])

                # ========== COMPUTE LOSS ==========
                loss = loss_function(a[L], y)

                # ========== BACKWARD PASS ==========
                # Output layer gradient
                delta[L] = loss_gradient(a[L], y) * activation_derivative(z[L])

                # Hidden layer gradients (backpropagate)
                for l in L-1 down to 1:
                    delta[l] = (W[l+1].T @ delta[l+1]) * activation_derivative(z[l])

                # ========== WEIGHT UPDATES ==========
                for l in 1 to L:
                    W[l] = W[l] - eta * (delta[l] @ a[l-1].T)
                    b[l] = b[l] - eta * delta[l]

            # Optional: compute and report epoch loss
            epoch_loss = mean([loss(forward(x), y) for (x, y) in training_data])

    RETURN trained weights W, b

    CHAIN RULE INTUITION:
    ---------------------
    dL/dW[l] = dL/da[L] * da[L]/dz[L] * dz[L]/da[L-1] * ... * dz[l]/dW[l]

    Each delta[l] accumulates the product of partial derivatives
    from the loss back to layer l, enabling efficient computation
    via dynamic programming (memoization of intermediate gradients).
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for backpropagation."""
    return {
        # Forward pass
        "linear_transform": "z^(l) = W^(l) a^(l-1) + b^(l)",
        "activation": "a^(l) = f(z^(l))",
        # Loss functions
        "mse_loss": "L = (1/2n) sum_i ||y_i - a^(L)_i||^2",
        "cross_entropy": "L = -sum_i [y_i log(a^(L)_i) + (1-y_i) log(1-a^(L)_i)]",
        # Output layer delta
        "output_delta_mse": "delta^(L) = (a^(L) - y) * f'(z^(L))",
        "output_delta_softmax_ce": "delta^(L) = a^(L) - y  (simplified when using softmax + cross-entropy)",
        # Backpropagation (chain rule)
        "hidden_delta": "delta^(l) = ((W^(l+1))^T delta^(l+1)) * f'(z^(l))",
        # Gradient computation
        "weight_gradient": "dL/dW^(l) = delta^(l) (a^(l-1))^T",
        "bias_gradient": "dL/db^(l) = delta^(l)",
        # Weight update
        "weight_update": "W^(l) := W^(l) - eta * dL/dW^(l)",
        "bias_update": "b^(l) := b^(l) - eta * dL/db^(l)",
        # Chain rule
        "chain_rule": "dL/dW^(l) = dL/da^(L) * da^(L)/da^(L-1) * ... * da^(l)/dW^(l)",
        # Computational complexity
        "forward_complexity": "O(sum_l n_l * n_(l-1))  [same as matrix multiplications]",
        "backward_complexity": "O(sum_l n_l * n_(l-1))  [symmetric to forward]",
    }


def get_historical_context() -> Dict[str, str]:
    """Return historical context of backpropagation development."""
    return {
        "1960_kelley": """
            Kelley (1960) - Control theory application of chain rule
            for optimal control, precursor to automatic differentiation.
        """,
        "1969_bryson_ho": """
            Bryson & Ho (1969) - Applied optimal control using dynamic
            programming and adjoint methods, mathematically equivalent
            to backpropagation.
        """,
        "1970_linnainmaa": """
            Linnainmaa (1970) - Master's thesis describing automatic
            differentiation via the chain rule.
        """,
        "1974_werbos": """
            Werbos (1974) - PhD thesis explicitly describing backpropagation
            for neural networks, but paper was not widely circulated.
        """,
        "1986_rumelhart": """
            Rumelhart, Hinton, Williams (1986) - Nature paper that brought
            backpropagation to mainstream attention with clear exposition
            and compelling demonstrations of learned representations.
        """,
        "1986_pdp": """
            PDP Research Group (1986) - "Parallel Distributed Processing"
            volumes provided comprehensive treatment of connectionist
            models and learning algorithms.
        """,
    }


def get_gradient_flow_challenges() -> List[Dict[str, str]]:
    """Return common challenges with gradient flow in backpropagation."""
    return [
        {
            "problem": "Vanishing Gradients",
            "description": """
                Gradients shrink exponentially as they backpropagate through
                many layers, making it difficult to train deep networks.
            """,
            "cause": """
                Chain rule multiplication: if |f'(z)| < 1 and |W| < 1,
                gradients decay as O(c^L) where c < 1 and L is depth.
            """,
            "solutions": [
                "ReLU activation (gradient = 1 for positive inputs)",
                "Residual connections (skip connections)",
                "LSTM/GRU gating mechanisms",
                "Careful weight initialization",
                "Batch/layer normalization",
            ],
        },
        {
            "problem": "Exploding Gradients",
            "description": """
                Gradients grow exponentially, causing numerical instability
                and wild weight updates.
            """,
            "cause": """
                If |f'(z)| > 1 or spectral norm of W > 1, gradients
                can grow exponentially with depth.
            """,
            "solutions": [
                "Gradient clipping",
                "Weight regularization",
                "Careful initialization (Xavier, He)",
                "Batch normalization",
            ],
        },
        {
            "problem": "Saddle Points and Local Minima",
            "description": """
                Non-convex loss landscape has many critical points where
                gradients vanish but are not global minima.
            """,
            "cause": "High-dimensional non-convex optimization landscape.",
            "solutions": [
                "Momentum-based optimizers",
                "Adaptive learning rates (Adam, RMSprop)",
                "Learning rate schedules",
                "Multiple random restarts",
            ],
        },
        {
            "problem": "Credit Assignment",
            "description": """
                Determining which weights are responsible for errors is
                challenging in deep networks.
            """,
            "cause": "Error signal must traverse many layers.",
            "solutions": [
                "Backpropagation (the solution!)",
                "Attention mechanisms for explicit credit",
                "Skip connections for direct paths",
            ],
        },
    ]
