"""
Hebbian Learning - 1949

The first biologically-inspired learning rule, formalizing the principle
that "neurons that fire together, wire together." Provides a mechanism
for synaptic plasticity based on correlated activity.

Paper: "The Organization of Behavior"
Authors: Donald O. Hebb
Key Innovation: Activity-dependent synaptic modification rule
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from .base_neuron import BaseNeuron, BaseNeuronConfig


FORMULATION = r"""
Hebbian Learning Rule:

Basic Hebb Rule (pure correlation):

    Delta w_{ij} = eta * x_i * y_j

where:
    - w_{ij}    : Weight from neuron i to neuron j
    - x_i       : Pre-synaptic activity (input)
    - y_j       : Post-synaptic activity (output)
    - eta       : Learning rate

Variants:

1. Oja's Rule (normalized Hebbian):
    Delta w_{ij} = eta * y_j * (x_i - y_j * w_{ij})

2. Covariance Rule:
    Delta w_{ij} = eta * (x_i - <x>) * (y_j - <y>)

3. BCM Rule (Bienenstock-Cooper-Munro):
    Delta w_{ij} = eta * x_i * y_j * (y_j - theta_M)
    where theta_M is a sliding threshold

Properties:
- Unsupervised learning (no external teacher)
- Local computation (only needs local information)
- Can lead to unbounded weight growth (basic rule)
"""


@dataclass
class HebbianLearningConfig(BaseNeuronConfig):
    """Configuration for Hebbian learning."""
    learning_rate: float = 0.01
    normalize_weights: bool = False
    use_oja_rule: bool = False
    use_covariance: bool = False
    decay_rate: float = 0.0  # Weight decay for stability


class HebbianLearning(BaseNeuron):
    """
    Research index entry for Hebbian Learning (1949).

    The Hebbian learning rule captures the principle that synaptic
    connections strengthen when pre- and post-synaptic neurons are
    active simultaneously. This biologically plausible rule forms
    the basis for unsupervised learning and has deep connections to:
    - Principal Component Analysis
    - Self-organizing maps
    - Associative memory
    - Synaptic plasticity in neuroscience
    """

    METHOD_ID = "hebbian_learning_1949"
    NAME = "Hebbian Learning"
    YEAR = 1949
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
            authors=["Donald O. Hebb"],
            paper_title="The Organization of Behavior: A Neuropsychological Theory",
            paper_url="https://doi.org/10.4324/9781410612403",
            key_innovation="First biologically plausible learning rule based on "
                          "correlated neural activity",
            mathematical_formulation=FORMULATION,
            predecessors=["mcculloch_pitts_1943"],
            successors=[
                "perceptron_1958",
                "hopfield_network_1982",
                "oja_rule_1982",
                "bcm_rule_1982",
            ],
            tags=[
                "learning_rule",
                "unsupervised",
                "biological_plausibility",
                "synaptic_plasticity",
                "correlation_learning",
                "foundational",
            ]
        )

    @staticmethod
    def pseudocode() -> str:
        return """
        Hebbian Learning:
        -----------------
        INPUT: Training data X = [x_1, ..., x_N]
        INPUT: Initial weights W
        INPUT: Learning rate eta

        FOR each epoch:
            FOR each training sample x:
                1. Compute output:
                   y = W * x  (or apply activation)

                2. Update weights (basic Hebb):
                   FOR each weight w_ij:
                       w_ij = w_ij + eta * x_i * y_j

                3. Optional: Normalize weights
                   W = W / ||W||

        RETURN learned weights W

        Oja's Rule (prevents unbounded growth):
        ---------------------------------------
        w_ij = w_ij + eta * y_j * (x_i - y_j * w_ij)
        """

    @staticmethod
    def key_equations() -> Dict[str, str]:
        return {
            "basic_hebb": r"\Delta w_{ij} = \eta \cdot x_i \cdot y_j",
            "oja_rule": r"\Delta w_{ij} = \eta \cdot y_j \cdot (x_i - y_j \cdot w_{ij})",
            "covariance": r"\Delta w_{ij} = \eta \cdot (x_i - \bar{x}) \cdot (y_j - \bar{y})",
            "bcm": r"\Delta w_{ij} = \eta \cdot x_i \cdot y_j \cdot (y_j - \theta_M)",
            "weight_decay": r"\Delta w_{ij} = \eta \cdot x_i \cdot y_j - \lambda \cdot w_{ij}",
        }

    @classmethod
    def get_lineage(cls) -> List[MethodLineage]:
        return [MethodLineage.PERCEPTRON_LINE]

    @classmethod
    def get_historical_context(cls) -> str:
        return """
        Donald Hebb's 1949 book "The Organization of Behavior" proposed
        a neurophysiological theory of learning and memory. His famous
        postulate states:

        "When an axon of cell A is near enough to excite a cell B and
        repeatedly or persistently takes part in firing it, some growth
        process or metabolic change takes place in one or both cells such
        that A's efficiency, as one of the cells firing B, is increased."

        This was later simplified to "neurons that fire together, wire together."

        The rule was remarkably prescient:
        - Long-term potentiation (LTP) discovered in 1973 confirmed
          Hebbian-like mechanisms in the hippocampus
        - NMDA receptors act as Hebbian coincidence detectors
        - Spike-timing dependent plasticity (STDP) extends Hebb's ideas

        In machine learning, Hebbian learning connects to:
        - Principal Component Analysis (Oja's rule converges to PCA)
        - Self-organizing maps
        - Hopfield networks and associative memory
        """

    @classmethod
    def get_limitations(cls) -> List[str]:
        return [
            "Basic rule leads to unbounded weight growth",
            "Cannot learn XOR or other non-linearly separable functions",
            "Unsupervised only - no error correction",
            "Weights can only strengthen (no LTD in basic form)",
            "Sensitive to input correlations",
            "May not converge to useful representations",
        ]

    @classmethod
    def get_applications(cls) -> List[str]:
        return [
            "Unsupervised feature extraction",
            "Principal component analysis (via Oja's rule)",
            "Associative memory networks",
            "Self-organizing maps",
            "Computational neuroscience models",
            "Biological synapse modeling",
        ]

    @staticmethod
    def variants() -> Dict[str, Dict]:
        """Return descriptions of Hebbian learning variants."""
        return {
            "basic_hebb": {
                "name": "Basic Hebbian Rule",
                "year": 1949,
                "author": "Donald Hebb",
                "equation": "dw = eta * x * y",
                "properties": ["Unstable", "Unbounded growth"],
            },
            "oja": {
                "name": "Oja's Rule",
                "year": 1982,
                "author": "Erkki Oja",
                "equation": "dw = eta * y * (x - y * w)",
                "properties": ["Normalized", "Converges to first PC"],
            },
            "sanger": {
                "name": "Sanger's Rule (GHA)",
                "year": 1989,
                "author": "Terence Sanger",
                "equation": "dw_ij = eta * y_j * (x_i - sum_{k<=j} y_k * w_ik)",
                "properties": ["Extracts multiple PCs", "Orthogonal components"],
            },
            "bcm": {
                "name": "BCM Rule",
                "year": 1982,
                "author": "Bienenstock, Cooper, Munro",
                "equation": "dw = eta * x * y * (y - theta_M)",
                "properties": ["Sliding threshold", "Bidirectional plasticity"],
            },
        }
