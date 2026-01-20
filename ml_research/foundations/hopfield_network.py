"""
Hopfield Network - 1982

A recurrent neural network serving as content-addressable (associative)
memory with binary threshold units. Energy-based model that guaranteed
convergence to stable states, providing a theoretical bridge between
physics and neural computation.

Paper: "Neural networks and physical systems with emergent collective
        computational abilities"
Authors: John Hopfield
Key Innovation: Energy function analysis showing convergence to attractors
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from .base_neuron import BaseNeuron


FORMULATION = r"""
Hopfield Network:

Network Configuration:
    - N binary neurons: s_i in {-1, +1} (or {0, 1})
    - Symmetric weights: W_{ij} = W_{ji}
    - No self-connections: W_{ii} = 0

Energy Function:
    E = -1/2 * sum_{i,j} W_{ij} * s_i * s_j - sum_i theta_i * s_i

    Or in matrix form:
    E = -1/2 * s^T W s - theta^T s

Update Rule (asynchronous):
    s_i <- sign(sum_j W_{ij} * s_j - theta_i)

    This is equivalent to:
    s_i <- sign(h_i)  where h_i = sum_j W_{ij} s_j

Convergence Theorem:
    With asynchronous updates and symmetric weights (W = W^T),
    the energy function decreases (or stays same) at each update:

    Delta E <= 0

    Therefore, the network converges to a local minimum of E.

Storage Rule (Hebbian):
    For P patterns {xi^(1), ..., xi^(P)}:

    W_{ij} = (1/N) * sum_{mu=1}^{P} xi_i^{(mu)} * xi_j^{(mu)}

    Or in matrix form:
    W = (1/N) * sum_mu xi^{(mu)} (xi^{(mu)})^T

Storage Capacity:
    Maximum number of patterns that can be reliably stored:
    P_max ≈ 0.138 * N (Amit, Gutfreund, Sompolinsky, 1985)
"""


@dataclass
class HopfieldConfig:
    """Configuration for Hopfield Network."""
    n_neurons: int = 100
    threshold: float = 0.0
    update_mode: str = "asynchronous"  # "asynchronous" or "synchronous"
    max_iterations: int = 1000
    energy_tolerance: float = 1e-10


class HopfieldNetwork(BaseNeuron):
    """
    Research index entry for Hopfield Network (1982).

    The Hopfield network is a fully-connected recurrent neural network
    that functions as an associative (content-addressable) memory.
    It stores patterns as attractors of a dynamical system.

    Key contributions:
    - Introduced energy-based analysis to neural networks
    - Provided mathematical proof of convergence
    - Connected neural computation to statistical physics
    - Inspired Boltzmann machines and modern energy-based models

    Patterns are stored by Hebbian learning and retrieved by
    letting the network settle to a nearby attractor.
    """

    METHOD_ID = "hopfield_network_1982"
    NAME = "Hopfield Network"
    YEAR = 1982
    ERA = MethodEra.FOUNDATIONAL
    CATEGORY = MethodCategory.GENERATIVE

    @classmethod
    def get_method_info(cls) -> MLMethod:
        return MLMethod(
            method_id=cls.METHOD_ID,
            name=cls.NAME,
            year=cls.YEAR,
            era=cls.ERA,
            category=cls.CATEGORY,
            lineages=[MethodLineage.GENERATIVE_LINE],
            authors=["John Hopfield"],
            paper_title="Neural networks and physical systems with emergent "
                       "collective computational abilities",
            paper_url="https://doi.org/10.1073/pnas.79.8.2554",
            key_innovation="Energy function guaranteeing convergence to stable states, "
                          "connecting neural networks to statistical physics",
            mathematical_formulation=FORMULATION,
            predecessors=[
                "mcculloch_pitts_1943",
                "hebbian_learning_1949",
            ],
            successors=[
                "boltzmann_machine_1985",
                "restricted_boltzmann_machine",
                "modern_hopfield_2016",
                "energy_based_models",
            ],
            tags=[
                "associative_memory",
                "content_addressable",
                "energy_based",
                "attractor_network",
                "statistical_physics",
                "recurrent",
                "foundational",
            ]
        )

    @staticmethod
    def pseudocode() -> str:
        return """
        Hopfield Network:
        -----------------

        STORAGE PHASE (Learning):
        -------------------------
        INPUT: Patterns {p_1, ..., p_P} where p_mu in {-1, +1}^N

        Initialize W = zeros(N, N)

        FOR each pattern p_mu:
            W = W + outer_product(p_mu, p_mu)

        W = W / N
        Set diagonal to zero: W[i,i] = 0 for all i

        RETURN W


        RETRIEVAL PHASE (Recall):
        -------------------------
        INPUT: Probe pattern x (possibly corrupted)
        INPUT: Weight matrix W

        s = x  # Initialize state

        REPEAT until convergence:
            # Asynchronous update (random order)
            FOR i in random_permutation(1..N):
                h_i = sum_j(W[i,j] * s[j])
                s[i] = sign(h_i)  # +1 if h_i >= 0, else -1

            # Check energy
            E = -0.5 * s^T @ W @ s

            IF energy unchanged:
                BREAK

        RETURN s  # Retrieved pattern (attractor)


        ENERGY COMPUTATION:
        -------------------
        E(s) = -0.5 * sum_{i,j} W[i,j] * s[i] * s[j]
             = -0.5 * s^T @ W @ s
        """

    @staticmethod
    def key_equations() -> Dict[str, str]:
        return {
            "energy": r"E = -\frac{1}{2} \sum_{i,j} W_{ij} s_i s_j - \sum_i \theta_i s_i",
            "energy_matrix": r"E = -\frac{1}{2} s^T W s - \theta^T s",
            "update_rule": r"s_i \leftarrow \text{sign}\left(\sum_j W_{ij} s_j\right)",
            "hebbian_storage": r"W_{ij} = \frac{1}{N} \sum_{\mu=1}^{P} \xi_i^{(\mu)} \xi_j^{(\mu)}",
            "capacity": r"P_{\max} \approx 0.138 N",
            "energy_decrease": r"\Delta E \leq 0 \text{ (asynchronous updates)}",
        }

    @classmethod
    def get_lineage(cls) -> List[MethodLineage]:
        return [MethodLineage.GENERATIVE_LINE]

    @classmethod
    def get_historical_context(cls) -> str:
        return """
        John Hopfield's 1982 paper revitalized neural network research by
        providing a rigorous mathematical framework connecting neural computation
        to statistical physics. This was a pivotal moment between the
        "AI Winter" following Minsky & Papert (1969) and the connectionist
        revival of the mid-1980s.

        Key insights:
        - Neural networks can be analyzed as dynamical systems
        - Energy functions provide convergence guarantees
        - Memories are attractors in state space
        - Connection to spin glasses in physics (Ising model)

        Impact on the field:
        - Legitimized neural network research in physics community
        - Directly inspired Boltzmann machines (Hinton & Sejnowski, 1985)
        - Led to mean-field theory analysis of neural networks
        - Foundation for energy-based models in modern deep learning
        - 2024 Nobel Prize in Physics to Hopfield (with Hinton)

        The Hopfield network showed that computation could emerge from
        the collective behavior of simple interconnected units, providing
        a new paradigm for understanding both brains and machines.
        """

    @classmethod
    def get_limitations(cls) -> List[str]:
        return [
            "Limited storage capacity (~0.14N patterns)",
            "Spurious attractors (mixture states)",
            "Correlated patterns reduce capacity",
            "No temporal sequences (static attractors)",
            "Binary states only (discrete)",
            "Complete connectivity doesn't scale",
            "Slow convergence for large networks",
        ]

    @classmethod
    def get_applications(cls) -> List[str]:
        return [
            "Associative memory and pattern completion",
            "Content-addressable memory systems",
            "Optimization problems (TSP via continuous Hopfield)",
            "Error correction in communications",
            "Image restoration and denoising",
            "Constraint satisfaction problems",
            "Theoretical neuroscience models",
        ]

    @staticmethod
    def variants() -> Dict[str, Dict]:
        """Return Hopfield network variants."""
        return {
            "discrete": {
                "name": "Discrete Hopfield Network",
                "year": 1982,
                "description": "Original binary network",
                "states": "{-1, +1} or {0, 1}",
            },
            "continuous": {
                "name": "Continuous Hopfield Network",
                "year": 1984,
                "author": "Hopfield",
                "description": "Graded response units with sigmoid",
                "application": "Optimization (e.g., TSP)",
            },
            "dense_associative": {
                "name": "Dense Associative Memory",
                "year": 2016,
                "author": "Krotov & Hopfield",
                "description": "Higher-order interactions, exponential capacity",
            },
            "modern_hopfield": {
                "name": "Modern Hopfield Networks",
                "year": 2020,
                "author": "Ramsauer et al.",
                "description": "Connection to attention mechanisms",
                "capacity": "Exponential in dimension",
            },
        }

    @staticmethod
    def physics_connections() -> Dict[str, str]:
        """Return connections to statistical physics."""
        return {
            "ising_model": "Hopfield network is equivalent to Ising spin glass",
            "energy": "Network state evolves to minimize energy (like physical systems)",
            "temperature": "Can add noise (T > 0) for probabilistic dynamics",
            "phase_transitions": "Sharp transitions in retrieval quality vs. load",
            "replica_method": "Storage capacity analyzed using replica theory",
            "mean_field": "Mean-field approximation for large N analysis",
        }

    @staticmethod
    def storage_capacity_analysis() -> Dict[str, str]:
        """Return storage capacity analysis details."""
        return {
            "palm_bound": "P < N / (4 * log(N)) for perfect retrieval",
            "ags_result": "P_max ≈ 0.138 * N (Amit, Gutfreund, Sompolinsky)",
            "error_probability": "As P/N increases, retrieval errors grow",
            "critical_load": "alpha_c ≈ 0.138 where alpha = P/N",
            "spurious_states": "Number grows with P, causing retrieval errors",
        }
