"""
ADALINE and MADALINE - 1960

Adaptive Linear Neuron (ADALINE) and its multilayer extension (MADALINE).
Introduced the Least Mean Squares (LMS) algorithm, also known as the
Widrow-Hoff rule or delta rule, which uses gradient descent for learning.

Paper: "Adaptive Switching Circuits" (1960)
Authors: Bernard Widrow, Marcian Hoff
Key Innovation: Continuous error signal and gradient descent optimization
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from .base_neuron import BaseNeuron, BaseNeuronConfig


FORMULATION = r"""
ADALINE (Adaptive Linear Neuron):

Architecture:
    z = sum_{i=1}^{n} w_i * x_i + b    (linear combination)
    y_hat = sign(z)                     (output via threshold)

LMS Learning Rule (Widrow-Hoff / Delta Rule):

    Error computed on linear output (before threshold):
    e = y - z     (NOT y - y_hat)

    Weight update (gradient descent on MSE):
    w = w + eta * e * x
    b = b + eta * e

    This is equivalent to:
    w = w - eta * d/dw[(y - z)^2 / 2]
      = w + eta * (y - z) * x

Cost Function:
    J(w) = (1/2) * E[(y - w^T x)^2]

    The LMS rule performs stochastic gradient descent on J(w).

MADALINE (Multiple ADALINE):

    Multiple ADALINE units combined:
    - First multilayer adaptive network
    - Uses majority vote or other combination rules
    - Training is more complex (credit assignment)

MADALINE Rule II (1988):
    For hidden units, update based on:
    - Which hidden unit is closest to its threshold
    - Trial weight changes to improve output
"""


@dataclass
class ADALINEConfig(BaseNeuronConfig):
    """Configuration for ADALINE."""
    n_inputs: int = 2
    learning_rate: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 1e-6
    use_bias: bool = True


@dataclass
class MADALINEConfig:
    """Configuration for MADALINE."""
    n_inputs: int = 2
    n_hidden: int = 4
    learning_rate: float = 0.01
    max_iterations: int = 1000
    combination_rule: str = "majority"  # "majority", "and", "or"


class ADALINE(BaseNeuron):
    """
    Research index entry for ADALINE (1960).

    ADALINE (Adaptive Linear Neuron) introduced gradient-based learning
    to neural networks. Unlike the perceptron which updates only on
    misclassifications, ADALINE minimizes mean squared error using
    the LMS (Least Mean Squares) algorithm.

    Key differences from perceptron:
    - Error computed on linear output (continuous)
    - Gradient descent optimization
    - Smoother learning trajectory
    - Converges to optimal linear solution (not just separating hyperplane)
    """

    METHOD_ID = "adaline_1960"
    NAME = "ADALINE"
    YEAR = 1960
    ERA = MethodEra.FOUNDATIONAL
    CATEGORY = MethodCategory.NEURON_MODEL

    @classmethod
    def get_method_info(cls) -> MLMethod:
        return MLMethod(
            method_id=cls.METHOD_ID,
            name=cls.NAME,
            year=cls.YEAR,
            era=cls.ERA,
            category=cls.CATEGORY,
            lineages=[MethodLineage.PERCEPTRON_LINE],
            authors=["Bernard Widrow", "Marcian Hoff"],
            paper_title="Adaptive Switching Circuits",
            paper_url="https://doi.org/10.1109/IRE-WESCON.1960.5",
            key_innovation="LMS algorithm - gradient descent on mean squared error "
                          "with continuous error signal",
            mathematical_formulation=FORMULATION,
            predecessors=[
                "perceptron_1958",
            ],
            successors=[
                "multilayer_perceptron",
                "backpropagation",
                "adaptive_filters",
            ],
            tags=[
                "gradient_descent",
                "lms_algorithm",
                "delta_rule",
                "adaptive_filter",
                "linear_regression",
                "foundational",
            ]
        )

    @staticmethod
    def pseudocode() -> str:
        return """
        ADALINE with LMS Learning:
        --------------------------
        INPUT: Training data D = {(x_1, y_1), ..., (x_N, y_N)}
        INPUT: Learning rate eta (typically 0.01)
        INPUT: Convergence tolerance epsilon

        1. Initialize:
           w = small random values
           b = 0

        2. REPEAT:
               total_error = 0
               FOR each (x, y) in D:
                   # Compute linear output (before threshold)
                   z = dot(w, x) + b

                   # Compute error
                   e = y - z

                   # LMS update (gradient descent)
                   w = w + eta * e * x
                   b = b + eta * e

                   total_error += e^2

               mse = total_error / N

           UNTIL mse < epsilon or max_iterations

        RETURN (w, b)

        Classification: y_hat = sign(w^T x + b)
        """

    @staticmethod
    def key_equations() -> Dict[str, str]:
        return {
            "forward": r"z = \sum_{i=1}^{n} w_i x_i + b",
            "output": r"\hat{y} = \text{sign}(z)",
            "error": r"e = y - z",
            "lms_update": r"w \leftarrow w + \eta \cdot e \cdot x",
            "cost_function": r"J(w) = \frac{1}{2} E[(y - w^T x)^2]",
            "gradient": r"\nabla_w J = -E[(y - w^T x) \cdot x]",
        }

    @classmethod
    def get_lineage(cls) -> List[MethodLineage]:
        return [MethodLineage.PERCEPTRON_LINE]

    @classmethod
    def get_historical_context(cls) -> str:
        return """
        Bernard Widrow and his graduate student Ted Hoff developed ADALINE
        at Stanford in 1960. The name originally stood for "ADAptive LInear
        NEuron" but was later changed to "ADAptive LINear Element."

        Key contributions:
        - First use of gradient descent in neural networks
        - LMS algorithm is still widely used in adaptive filtering
        - Ted Hoff later invented the microprocessor at Intel

        The LMS algorithm became foundational in:
        - Adaptive signal processing
        - Echo cancellation (telephones)
        - Noise cancellation
        - Equalization in communications
        - Modern deep learning (SGD is a descendant)

        MADALINE (1962) extended ADALINE to multiple layers, predating
        backpropagation by over two decades. However, training was difficult
        due to the credit assignment problem.
        """

    @classmethod
    def get_limitations(cls) -> List[str]:
        return [
            "Still limited to linearly separable problems",
            "Learning rate must be carefully tuned",
            "Sensitive to input scaling",
            "MADALINE training is heuristic (no clean gradient)",
            "Convergence can be slow for large learning rates",
        ]

    @classmethod
    def get_applications(cls) -> List[str]:
        return [
            "Adaptive filtering",
            "Echo cancellation",
            "Noise reduction",
            "Channel equalization",
            "System identification",
            "Linear regression",
            "Pattern classification",
        ]


class MADALINE(BaseNeuron):
    """
    Research index entry for MADALINE (1962).

    MADALINE (Multiple ADALINE) was the first multilayer neural network,
    combining multiple ADALINE units. It predated backpropagation and
    used heuristic training methods.
    """

    METHOD_ID = "madaline_1962"
    NAME = "MADALINE"
    YEAR = 1962
    ERA = MethodEra.FOUNDATIONAL
    CATEGORY = MethodCategory.NEURON_MODEL

    @classmethod
    def get_method_info(cls) -> MLMethod:
        return MLMethod(
            method_id=cls.METHOD_ID,
            name=cls.NAME,
            year=cls.YEAR,
            era=cls.ERA,
            category=cls.CATEGORY,
            lineages=[MethodLineage.PERCEPTRON_LINE],
            authors=["Bernard Widrow", "Marcian Hoff"],
            paper_title="Adaptive Switching Circuits",
            paper_url="https://doi.org/10.1109/IRE-WESCON.1960.5",
            key_innovation="First multilayer adaptive network architecture",
            mathematical_formulation="""
MADALINE Architecture:

    Input layer -> Hidden ADALINE units -> Output combination

    Hidden layer: h_j = sign(sum_i w_ji * x_i + b_j)

    Output: Majority vote, AND, or OR of hidden outputs

MADALINE Rule II (1988):
    1. Find hidden unit closest to threshold (most uncertain)
    2. Tentatively flip its output
    3. If overall error decreases, update that unit's weights
    4. Otherwise, try next closest unit
""",
            predecessors=["adaline_1960"],
            successors=["backpropagation"],
            tags=[
                "multilayer",
                "adaptive_network",
                "heuristic_training",
                "foundational",
            ]
        )

    @staticmethod
    def pseudocode() -> str:
        return """
        MADALINE Rule II Training:
        --------------------------
        INPUT: Training data, MADALINE network with hidden ADALINE units

        FOR each training sample (x, y):
            1. Forward pass through all units
            2. IF output is incorrect:
                a. Find hidden unit closest to threshold
                   (smallest |z_j| where z is pre-threshold output)
                b. Tentatively flip this unit's output
                c. IF new output matches y:
                       Update that unit using LMS toward flipped state
                   ELSE:
                       Try next closest unit to threshold
                d. Repeat until output correct or all units tried
        """

    @staticmethod
    def key_equations() -> Dict[str, str]:
        return {
            "hidden_activation": r"h_j = \text{sign}\left(\sum_i w_{ji} x_i + b_j\right)",
            "majority_vote": r"y = \text{sign}\left(\sum_j h_j\right)",
            "uncertainty": r"\text{uncertainty}_j = |z_j| = \left|\sum_i w_{ji} x_i + b_j\right|",
        }

    @classmethod
    def get_lineage(cls) -> List[MethodLineage]:
        return [MethodLineage.PERCEPTRON_LINE]


def compare_perceptron_adaline() -> Dict[str, Dict]:
    """Compare Perceptron and ADALINE learning rules."""
    return {
        "perceptron": {
            "error_signal": "Binary (correct/incorrect)",
            "update_trigger": "Only on misclassification",
            "cost_function": "Number of misclassifications",
            "convergence": "Finds any separating hyperplane",
            "learning_rate": "Typically 1.0",
        },
        "adaline": {
            "error_signal": "Continuous (y - z)",
            "update_trigger": "Every sample",
            "cost_function": "Mean squared error",
            "convergence": "Finds minimum MSE solution",
            "learning_rate": "Typically small (0.01)",
        },
    }
