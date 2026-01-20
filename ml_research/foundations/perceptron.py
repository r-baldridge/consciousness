"""
Perceptron - 1958

The first machine learning algorithm that could learn from data.
A linear classifier with a supervised learning rule that adjusts
weights based on classification errors.

Paper: "The Perceptron: A Probabilistic Model for Information Storage
        and Organization in the Brain"
Authors: Frank Rosenblatt
Key Innovation: Supervised learning rule with convergence guarantee
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from .base_neuron import BaseNeuron, BaseNeuronConfig


FORMULATION = r"""
Rosenblatt Perceptron:

Forward Pass:
    z = sum_{i=1}^{n} w_i * x_i + b    (weighted sum with bias)
    y_hat = sign(z)                     (threshold at 0)

Perceptron Learning Rule (error-correcting):

    IF y_hat != y (misclassification):
        w = w + eta * y * x
        b = b + eta * y
    ELSE:
        no update

where:
    - x in R^n     : Input vector
    - w in R^n     : Weight vector
    - b in R       : Bias term
    - y in {-1, +1}: True label
    - y_hat        : Predicted label
    - eta          : Learning rate

Perceptron Convergence Theorem:
    If the training data is linearly separable, the perceptron
    learning algorithm will converge in a finite number of steps.

    Maximum iterations <= (R / gamma)^2

    where:
    - R = max||x_i|| (maximum norm of inputs)
    - gamma = margin (minimum distance to decision boundary)
"""


@dataclass
class PerceptronConfig(BaseNeuronConfig):
    """Configuration for Perceptron."""
    n_inputs: int = 2
    learning_rate: float = 1.0
    max_iterations: int = 1000
    use_bias: bool = True
    weights_init: str = "zeros"


class Perceptron(BaseNeuron):
    """
    Research index entry for Rosenblatt Perceptron (1958).

    The perceptron is a single-layer linear classifier that learns
    from labeled examples using an error-correcting rule. It was the
    first algorithm to demonstrate machine learning from data.

    Key properties:
    - Guaranteed convergence for linearly separable data
    - Simple, online learning algorithm
    - Foundation for neural network learning

    The 1969 book "Perceptrons" by Minsky & Papert proved limitations
    (cannot learn XOR), which temporarily halted neural network research.
    """

    METHOD_ID = "perceptron_1958"
    NAME = "Rosenblatt Perceptron"
    YEAR = 1958
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
            authors=["Frank Rosenblatt"],
            paper_title="The Perceptron: A Probabilistic Model for Information "
                       "Storage and Organization in the Brain",
            paper_url="https://doi.org/10.1037/h0042519",
            key_innovation="First supervised learning algorithm with proven "
                          "convergence for linearly separable data",
            mathematical_formulation=FORMULATION,
            predecessors=[
                "mcculloch_pitts_1943",
                "hebbian_learning_1949",
            ],
            successors=[
                "adaline_1960",
                "multilayer_perceptron",
                "backpropagation",
            ],
            tags=[
                "linear_classifier",
                "supervised_learning",
                "convergence_theorem",
                "error_correction",
                "online_learning",
                "foundational",
            ]
        )

    @staticmethod
    def pseudocode() -> str:
        return """
        Perceptron Learning Algorithm:
        ------------------------------
        INPUT: Training data D = {(x_1, y_1), ..., (x_N, y_N)}
               where y_i in {-1, +1}
        INPUT: Learning rate eta (typically 1.0)
        INPUT: Maximum iterations T

        1. Initialize:
           w = [0, 0, ..., 0]  # weight vector
           b = 0               # bias

        2. FOR t = 1 to T:
               converged = True
               FOR each (x, y) in D:
                   # Compute prediction
                   z = dot(w, x) + b
                   y_hat = sign(z)  # +1 if z >= 0, else -1

                   # Update on misclassification
                   IF y_hat != y:
                       w = w + eta * y * x
                       b = b + eta * y
                       converged = False

               IF converged:
                   BREAK

        RETURN (w, b)

        Decision boundary: w^T * x + b = 0
        """

    @staticmethod
    def key_equations() -> Dict[str, str]:
        return {
            "forward": r"\hat{y} = \text{sign}\left(\sum_{i=1}^{n} w_i x_i + b\right)",
            "update_weights": r"w \leftarrow w + \eta \cdot y \cdot x \quad \text{if } \hat{y} \neq y",
            "update_bias": r"b \leftarrow b + \eta \cdot y \quad \text{if } \hat{y} \neq y",
            "decision_boundary": r"w^T x + b = 0",
            "convergence_bound": r"t \leq \left(\frac{R}{\gamma}\right)^2",
        }

    @classmethod
    def get_lineage(cls) -> List[MethodLineage]:
        return [MethodLineage.PERCEPTRON_LINE]

    @classmethod
    def get_historical_context(cls) -> str:
        return """
        Frank Rosenblatt's perceptron (1958) was a landmark achievement in
        AI and machine learning. Implemented on the Mark I Perceptron machine
        at Cornell, it could learn to classify simple visual patterns.

        The New York Times reported it as the "embryo of an electronic computer
        that [the Navy] expects will be able to walk, talk, see, write,
        reproduce itself and be conscious of its existence."

        This hype led to significant funding but also to the "AI Winter" when
        Minsky and Papert's 1969 book "Perceptrons" proved that single-layer
        perceptrons cannot learn functions like XOR. They suggested (incorrectly)
        that multilayer networks would be similarly limited.

        Key moments:
        - 1958: Rosenblatt publishes perceptron
        - 1960: Widrow develops ADALINE (similar but uses gradient descent)
        - 1969: Minsky & Papert publish limitations
        - 1986: Backpropagation revives multilayer networks

        The perceptron convergence theorem remains fundamental to learning theory.
        """

    @classmethod
    def get_limitations(cls) -> List[str]:
        return [
            "Cannot learn non-linearly separable functions (e.g., XOR)",
            "Single layer - no hidden representations",
            "Decision boundary must pass through origin (without bias)",
            "No probabilistic output (hard classification only)",
            "Sensitive to feature scaling",
            "Multiple solutions possible (depends on presentation order)",
        ]

    @classmethod
    def get_applications(cls) -> List[str]:
        return [
            "Binary classification (linearly separable data)",
            "Online learning scenarios",
            "Document classification",
            "Spam filtering",
            "Foundation for kernel perceptron",
            "Teaching linear classifiers",
        ]

    @staticmethod
    def variants() -> Dict[str, Dict]:
        """Return descriptions of perceptron variants."""
        return {
            "basic": {
                "name": "Basic Perceptron",
                "year": 1958,
                "author": "Rosenblatt",
                "description": "Original error-correcting rule",
            },
            "voted": {
                "name": "Voted Perceptron",
                "year": 1999,
                "author": "Freund & Schapire",
                "description": "Ensemble of perceptrons from training",
            },
            "averaged": {
                "name": "Averaged Perceptron",
                "year": 1999,
                "author": "Freund & Schapire",
                "description": "Average weights over all iterations",
            },
            "kernel": {
                "name": "Kernel Perceptron",
                "year": 1964,
                "author": "Aizerman et al.",
                "description": "Perceptron in kernel-induced feature space",
            },
            "multiclass": {
                "name": "Multiclass Perceptron",
                "year": 1958,
                "author": "Rosenblatt",
                "description": "One-vs-all or multiclass extension",
            },
        }

    @staticmethod
    def convergence_analysis() -> Dict[str, str]:
        """Return convergence analysis details."""
        return {
            "theorem": "Perceptron Convergence Theorem",
            "statement": "If training data is linearly separable with margin gamma, "
                        "the perceptron converges in at most (R/gamma)^2 iterations",
            "R_definition": "R = max_i ||x_i|| (maximum input norm)",
            "gamma_definition": "gamma = min_i |w* . x_i| / ||w*|| (geometric margin)",
            "note": "No convergence guarantee for non-separable data",
        }
