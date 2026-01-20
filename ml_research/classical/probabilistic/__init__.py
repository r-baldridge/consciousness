"""
Classical Probabilistic Methods Module

This module contains research indices for classical probabilistic ML methods,
particularly energy-based models from the connectionist era.

Key Methods:
    - Boltzmann Machine (1985): Stochastic recurrent neural network
    - Restricted Boltzmann Machine (RBM): Bipartite energy-based model
    - Deep Belief Networks (2006): Stacked RBMs with greedy pretraining
"""

from .boltzmann_machine import (
    BOLTZMANN_MACHINE,
    boltzmann_energy,
    gibbs_sampling_step,
)
from .rbm import (
    RESTRICTED_BOLTZMANN_MACHINE,
    rbm_energy,
    contrastive_divergence_update,
)
from .dbn import (
    DEEP_BELIEF_NETWORK,
    greedy_layerwise_pretrain,
)

__all__ = [
    # Boltzmann Machine
    "BOLTZMANN_MACHINE",
    "boltzmann_energy",
    "gibbs_sampling_step",
    # Restricted Boltzmann Machine
    "RESTRICTED_BOLTZMANN_MACHINE",
    "rbm_energy",
    "contrastive_divergence_update",
    # Deep Belief Networks
    "DEEP_BELIEF_NETWORK",
    "greedy_layerwise_pretrain",
]
