"""
McCulloch-Pitts Neuron - 1943

The first mathematical model of an artificial neuron, establishing
the foundation for neural network research. A binary threshold unit
that demonstrated how networks of simple logical units could perform
computation.

Paper: "A Logical Calculus of the Ideas Immanent in Nervous Activity"
Authors: Warren McCulloch, Walter Pitts
Key Innovation: Showed neural activity could be modeled with propositional logic
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from .base_neuron import BaseNeuron, BaseNeuronConfig


FORMULATION = r"""
McCulloch-Pitts Neuron:

The neuron computes a binary output based on weighted inputs:

    y(t+1) = H( sum_{i=1}^{n} w_i * x_i(t) - theta )

where:
    - x_i in {0, 1}  : Binary inputs
    - w_i in {-1, +1}: Excitatory (+1) or inhibitory (-1) weights
    - theta          : Threshold value
    - H(z)           : Heaviside step function
                       H(z) = 1 if z >= 0, else 0

Special property: A single inhibitory input can veto the neuron output.

Logical operations:
    - AND gate:  theta = n (requires all inputs)
    - OR gate:   theta = 1 (requires any input)
    - NOT gate:  Single inhibitory input with theta = 0
"""


@dataclass
class McCullochPittsConfig(BaseNeuronConfig):
    """Configuration for McCulloch-Pitts neuron."""
    n_inputs: int = 2
    threshold: float = 1.0
    use_inhibitory: bool = False
    inhibitory_indices: List[int] = None

    def __post_init__(self):
        if self.inhibitory_indices is None:
            self.inhibitory_indices = []


class McCullochPittsNeuron(BaseNeuron):
    """
    Research index entry for McCulloch-Pitts Neuron (1943).

    The M-P neuron is a binary threshold device that:
    - Takes binary inputs {0, 1}
    - Uses fixed weights (excitatory +1, inhibitory -1)
    - Outputs 1 if weighted sum >= threshold, else 0
    - Has a special inhibitory veto capability

    This model proved that any logical function could be computed
    by an appropriately designed network of such neurons.
    """

    METHOD_ID = "mcculloch_pitts_1943"
    NAME = "McCulloch-Pitts Neuron"
    YEAR = 1943
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
            authors=["Warren McCulloch", "Walter Pitts"],
            paper_title="A Logical Calculus of the Ideas Immanent in Nervous Activity",
            paper_url="https://doi.org/10.1007/BF02478259",
            key_innovation="First mathematical model showing neural computation "
                          "equivalent to propositional logic",
            mathematical_formulation=FORMULATION,
            predecessors=[],
            successors=[
                "hebbian_learning_1949",
                "perceptron_1958",
            ],
            tags=[
                "binary_neuron",
                "threshold_logic",
                "computational_neuroscience",
                "logic_gates",
                "foundational",
            ]
        )

    @staticmethod
    def pseudocode() -> str:
        return """
        McCulloch-Pitts Neuron:
        -----------------------
        INPUT: Binary vector x = [x_1, ..., x_n]
        INPUT: Weight vector w = [w_1, ..., w_n] where w_i in {-1, +1}
        INPUT: Threshold theta

        1. Check inhibitory veto:
           IF any inhibitory input x_i = 1 where w_i = -1:
               RETURN 0  # Veto activated

        2. Compute weighted sum:
           z = sum(w_i * x_i for all excitatory inputs)

        3. Apply threshold:
           IF z >= theta:
               RETURN 1
           ELSE:
               RETURN 0
        """

    @staticmethod
    def key_equations() -> Dict[str, str]:
        return {
            "weighted_sum": r"z = \sum_{i=1}^{n} w_i \cdot x_i",
            "threshold": r"y = H(z - \theta) = \begin{cases} 1 & \text{if } z \geq \theta \\ 0 & \text{otherwise} \end{cases}",
            "AND_gate": r"y = H(x_1 + x_2 - 2)",
            "OR_gate": r"y = H(x_1 + x_2 - 1)",
            "NOT_gate": r"y = H(-x_1)",
        }

    @classmethod
    def get_lineage(cls) -> List[MethodLineage]:
        return [MethodLineage.PERCEPTRON_LINE]

    @classmethod
    def get_historical_context(cls) -> str:
        return """
        The McCulloch-Pitts neuron (1943) marks the birth of computational
        neuroscience and artificial neural networks. Warren McCulloch, a
        neurophysiologist, and Walter Pitts, a mathematician, showed that
        networks of simple binary threshold units could compute any logical
        function, establishing a theoretical foundation for understanding
        the brain as an information processing system.

        This work directly inspired:
        - Alan Turing's work on neural networks (1948)
        - Hebb's learning rule (1949)
        - Rosenblatt's perceptron (1958)
        - Modern connectionism and deep learning

        The paper is considered one of the most influential works in both
        neuroscience and computer science.
        """

    @classmethod
    def get_limitations(cls) -> List[str]:
        return [
            "Binary inputs only (no continuous values)",
            "Fixed weights (no learning algorithm)",
            "Cannot learn from examples",
            "Synchronous operation assumed",
            "No temporal dynamics beyond discrete time steps",
            "Cannot represent graded responses",
        ]

    @classmethod
    def get_applications(cls) -> List[str]:
        return [
            "Digital logic circuit design",
            "Theoretical neuroscience models",
            "Boolean function computation",
            "Historical foundation for neural networks",
            "Teaching tool for understanding neural computation",
        ]

    @staticmethod
    def logical_gates() -> Dict[str, Dict]:
        """Return configurations for basic logical gates."""
        return {
            "AND": {
                "weights": [1, 1],
                "threshold": 2,
                "description": "Outputs 1 only if all inputs are 1",
            },
            "OR": {
                "weights": [1, 1],
                "threshold": 1,
                "description": "Outputs 1 if any input is 1",
            },
            "NOT": {
                "weights": [-1],
                "threshold": 0,
                "description": "Inverts the input",
            },
            "NAND": {
                "weights": [-1, -1],
                "threshold": -1,
                "description": "NOT AND - universal gate",
            },
            "NOR": {
                "weights": [-1, -1],
                "threshold": 0,
                "description": "NOT OR",
            },
        }
