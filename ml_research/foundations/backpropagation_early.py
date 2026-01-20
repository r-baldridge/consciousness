"""
Early Backpropagation - 1970s

The foundational work on automatic differentiation and gradient computation
through multilayer networks. Key contributors include Paul Werbos (1974),
Seppo Linnainmaa (1970), and others who developed the mathematical
framework that would later revolutionize deep learning.

Papers:
    - Linnainmaa (1970): "The representation of the cumulative rounding error"
    - Werbos (1974): "Beyond Regression: New Tools for Prediction and Analysis"
    - Rumelhart, Hinton, Williams (1986): Popularized for neural networks

Authors: Seppo Linnainmaa, Paul Werbos, David Rumelhart, Geoffrey Hinton, Ronald Williams
Key Innovation: Efficient gradient computation through computational graphs
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from .base_neuron import BaseNeuron


FORMULATION = r"""
Backpropagation (Chain Rule for Gradient Computation):

For a network with layers l = 1, ..., L:

Forward Pass:
    z^{(l)} = W^{(l)} * a^{(l-1)} + b^{(l)}    (pre-activation)
    a^{(l)} = f(z^{(l)})                         (activation)

Loss Function:
    L = L(a^{(L)}, y)                            (e.g., MSE, cross-entropy)

Backward Pass (chain rule):
    delta^{(L)} = dL/da^{(L)} * f'(z^{(L)})      (output layer error)

    FOR l = L-1 down to 1:
        delta^{(l)} = (W^{(l+1)})^T * delta^{(l+1)} * f'(z^{(l)})

Gradients:
    dL/dW^{(l)} = delta^{(l)} * (a^{(l-1)})^T
    dL/db^{(l)} = delta^{(l)}

Key Insight (Werbos):
    Gradient computation cost is O(W), same as forward pass,
    NOT O(W^2) as naive differentiation would require.

Automatic Differentiation:
    Forward mode:  d/dx [f(g(x))] computed left-to-right
    Reverse mode:  d/dx [f(g(x))] computed right-to-left (backprop)

    Reverse mode is efficient when outputs << inputs.
"""


@dataclass
class BackpropagationConfig:
    """Configuration for backpropagation."""
    learning_rate: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    gradient_clip: Optional[float] = None
    batch_size: int = 32


class EarlyBackpropagation(BaseNeuron):
    """
    Research index entry for Early Backpropagation (1970-1986).

    Backpropagation is the algorithm that made training deep neural networks
    practical. It efficiently computes gradients of a loss function with
    respect to all parameters using the chain rule.

    Historical development:
    - 1960s: Control theory applications (Bryson, Dreyfus)
    - 1970: Linnainmaa - automatic differentiation in computers
    - 1974: Werbos - PhD thesis applying to neural networks
    - 1986: Rumelhart, Hinton, Williams - popularization

    This enables gradient descent optimization in multilayer networks.
    """

    METHOD_ID = "backpropagation_1974"
    NAME = "Backpropagation"
    YEAR = 1974
    ERA = MethodEra.FOUNDATIONAL
    CATEGORY = MethodCategory.LEARNING_RULE

    @classmethod
    def get_method_info(cls) -> MLMethod:
        return MLMethod(
            method_id=cls.METHOD_ID,
            name=cls.NAME,
            year=cls.YEAR,
            era=cls.ERA,
            category=cls.CATEGORY,
            lineages=[MethodLineage.PERCEPTRON_LINE],
            authors=[
                "Seppo Linnainmaa",
                "Paul Werbos",
                "David Rumelhart",
                "Geoffrey Hinton",
                "Ronald Williams",
            ],
            paper_title="Beyond Regression: New Tools for Prediction and Analysis "
                       "in the Behavioral Sciences",
            paper_url="https://www.nature.com/articles/323533a0",  # 1986 Nature paper
            key_innovation="Efficient gradient computation through multilayer networks "
                          "via the chain rule (reverse-mode automatic differentiation)",
            mathematical_formulation=FORMULATION,
            predecessors=[
                "adaline_1960",
                "perceptron_1958",
            ],
            successors=[
                "multilayer_perceptron",
                "convolutional_neural_network",
                "recurrent_neural_network",
                "transformer",
            ],
            tags=[
                "gradient_descent",
                "chain_rule",
                "automatic_differentiation",
                "deep_learning",
                "foundational",
                "learning_rule",
            ]
        )

    @staticmethod
    def pseudocode() -> str:
        return """
        Backpropagation Algorithm:
        --------------------------
        INPUT: Network with L layers, weights W, biases b
        INPUT: Training sample (x, y)
        INPUT: Learning rate eta

        # Forward Pass
        a[0] = x
        FOR l = 1 to L:
            z[l] = W[l] @ a[l-1] + b[l]
            a[l] = activation(z[l])

        # Compute Loss
        loss = loss_function(a[L], y)

        # Backward Pass
        delta[L] = d_loss(a[L], y) * d_activation(z[L])

        FOR l = L-1 down to 1:
            delta[l] = (W[l+1].T @ delta[l+1]) * d_activation(z[l])

        # Compute Gradients
        FOR l = 1 to L:
            dW[l] = delta[l] @ a[l-1].T
            db[l] = delta[l]

        # Update Parameters
        FOR l = 1 to L:
            W[l] = W[l] - eta * dW[l]
            b[l] = b[l] - eta * db[l]

        RETURN loss
        """

    @staticmethod
    def key_equations() -> Dict[str, str]:
        return {
            "forward_preactivation": r"z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}",
            "forward_activation": r"a^{(l)} = f(z^{(l)})",
            "output_error": r"\delta^{(L)} = \frac{\partial L}{\partial a^{(L)}} \odot f'(z^{(L)})",
            "hidden_error": r"\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot f'(z^{(l)})",
            "weight_gradient": r"\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T",
            "bias_gradient": r"\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}",
            "chain_rule": r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}",
        }

    @classmethod
    def get_lineage(cls) -> List[MethodLineage]:
        return [MethodLineage.PERCEPTRON_LINE]

    @classmethod
    def get_historical_context(cls) -> str:
        return """
        The history of backpropagation spans multiple decades and disciplines:

        1960s - Control Theory Roots:
        - Optimal control and adjoint methods (Pontryagin, Bryson)
        - Dynamic programming (Bellman)

        1970 - Linnainmaa's Thesis:
        - First published description of reverse-mode automatic differentiation
        - "Cumulative rounding error of algorithms as Taylor expansion"

        1974 - Werbos PhD Thesis:
        - Applied backprop to neural networks
        - "Beyond Regression" - Harvard PhD thesis
        - Not widely known at the time

        1982 - Werbos Publication:
        - Applied to training neural networks
        - Still not widely noticed

        1985-1986 - Rediscovery and Popularization:
        - Parker (1985) independently derived backprop
        - LeCun (1985) derived for neural networks
        - Rumelhart, Hinton, Williams (1986) - Nature paper
        - "Learning representations by back-propagating errors"
        - Finally achieved widespread recognition

        Modern Impact:
        - Foundation of all deep learning
        - Automatic differentiation in PyTorch, TensorFlow
        - Enables training of billion-parameter models
        """

    @classmethod
    def get_limitations(cls) -> List[str]:
        return [
            "Vanishing gradients in deep networks (pre-ReLU, pre-ResNet)",
            "Exploding gradients without proper initialization",
            "Local minima (though less problematic than once thought)",
            "Not biologically plausible (weight transport problem)",
            "Sequential computation (limits parallelization)",
            "Memory scales with network depth",
        ]

    @classmethod
    def get_applications(cls) -> List[str]:
        return [
            "Training all feedforward neural networks",
            "Training recurrent neural networks (BPTT)",
            "Training convolutional neural networks",
            "Training transformers",
            "Automatic differentiation in scientific computing",
            "Gradient-based optimization in general",
        ]

    @staticmethod
    def historical_timeline() -> List[Dict]:
        """Return timeline of backpropagation development."""
        return [
            {
                "year": 1960,
                "author": "Kelley",
                "contribution": "Gradient methods for optimal control",
            },
            {
                "year": 1961,
                "author": "Bryson",
                "contribution": "Adjoint method for optimal control",
            },
            {
                "year": 1970,
                "author": "Linnainmaa",
                "contribution": "Reverse-mode automatic differentiation",
            },
            {
                "year": 1974,
                "author": "Werbos",
                "contribution": "PhD thesis applying backprop to neural nets",
            },
            {
                "year": 1982,
                "author": "Werbos",
                "contribution": "Published applications to neural networks",
            },
            {
                "year": 1985,
                "author": "Parker",
                "contribution": "Independent rediscovery",
            },
            {
                "year": 1985,
                "author": "LeCun",
                "contribution": "Procedure for learning in neural nets",
            },
            {
                "year": 1986,
                "author": "Rumelhart, Hinton, Williams",
                "contribution": "Nature paper - widespread popularization",
            },
        ]

    @staticmethod
    def variants() -> Dict[str, Dict]:
        """Return backpropagation variants and extensions."""
        return {
            "standard_bp": {
                "name": "Standard Backpropagation",
                "description": "Vanilla gradient descent with backprop",
            },
            "momentum": {
                "name": "Backprop with Momentum",
                "description": "Accumulates gradient history for faster convergence",
                "equation": "v = mu*v - eta*gradient; w = w + v",
            },
            "bptt": {
                "name": "Backpropagation Through Time",
                "description": "Backprop unrolled through time steps for RNNs",
            },
            "truncated_bptt": {
                "name": "Truncated BPTT",
                "description": "Limits backprop horizon for computational efficiency",
            },
            "real_time_recurrent": {
                "name": "Real-Time Recurrent Learning",
                "description": "Forward-mode gradient computation for RNNs",
            },
        }


class AutomaticDifferentiation:
    """
    Research index entry for Automatic Differentiation concepts.

    Automatic differentiation is the computational technique underlying
    backpropagation. It computes exact derivatives of functions specified
    by computer programs.
    """

    METHOD_ID = "automatic_differentiation_1970"
    NAME = "Automatic Differentiation"
    YEAR = 1970

    @staticmethod
    def modes() -> Dict[str, Dict]:
        """Compare forward and reverse mode AD."""
        return {
            "forward_mode": {
                "name": "Forward Mode AD (Tangent Mode)",
                "description": "Propagates derivatives forward through computation",
                "complexity": "O(n) per input, where n is #operations",
                "best_for": "Few inputs, many outputs",
                "implementation": "Dual numbers, operator overloading",
            },
            "reverse_mode": {
                "name": "Reverse Mode AD (Adjoint Mode)",
                "description": "Propagates derivatives backward through computation",
                "complexity": "O(n) total for all inputs",
                "best_for": "Many inputs, few outputs (neural networks!)",
                "implementation": "Tape-based, computation graphs",
            },
        }

    @staticmethod
    def comparison_with_alternatives() -> Dict[str, Dict]:
        """Compare AD with other differentiation methods."""
        return {
            "symbolic_diff": {
                "name": "Symbolic Differentiation",
                "pros": "Exact, interpretable formulas",
                "cons": "Expression swell, cannot handle conditionals",
            },
            "numerical_diff": {
                "name": "Numerical Differentiation",
                "pros": "Simple, works for any function",
                "cons": "Approximation errors, O(n) cost per input",
            },
            "automatic_diff": {
                "name": "Automatic Differentiation",
                "pros": "Exact, efficient, handles control flow",
                "cons": "Memory for computation graph (reverse mode)",
            },
        }
