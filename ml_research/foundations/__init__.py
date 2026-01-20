"""
Foundations Era Module (1943-1980)

This module contains research index entries for foundational methods
in neural network and machine learning history. These methods laid
the groundwork for modern deep learning.

Timeline:
    1943 - McCulloch-Pitts: First artificial neuron model
    1949 - Hebbian Learning: "Neurons that fire together wire together"
    1958 - Perceptron: First supervised learning algorithm
    1960 - ADALINE: Gradient descent with LMS algorithm
    1970s - Backpropagation: Efficient gradient computation
    1982 - Hopfield Network: Energy-based associative memory

These are research index entries (documentation/reference), not
runnable implementations. They capture:
    - Historical context and significance
    - Mathematical formulations
    - Key innovations and limitations
    - Connections to subsequent methods
"""

from .base_neuron import BaseNeuron, BaseNeuronConfig
from .mcculloch_pitts import McCullochPittsNeuron, McCullochPittsConfig
from .hebbian_learning import HebbianLearning, HebbianLearningConfig
from .perceptron import Perceptron, PerceptronConfig
from .adaline_madaline import (
    ADALINE,
    MADALINE,
    ADALINEConfig,
    MADALINEConfig,
    compare_perceptron_adaline,
)
from .backpropagation_early import (
    EarlyBackpropagation,
    BackpropagationConfig,
    AutomaticDifferentiation,
)
from .hopfield_network import HopfieldNetwork, HopfieldConfig

__all__ = [
    # Base classes
    "BaseNeuron",
    "BaseNeuronConfig",
    # 1943 - McCulloch-Pitts
    "McCullochPittsNeuron",
    "McCullochPittsConfig",
    # 1949 - Hebbian Learning
    "HebbianLearning",
    "HebbianLearningConfig",
    # 1958 - Perceptron
    "Perceptron",
    "PerceptronConfig",
    # 1960 - ADALINE/MADALINE
    "ADALINE",
    "MADALINE",
    "ADALINEConfig",
    "MADALINEConfig",
    "compare_perceptron_adaline",
    # 1970s - Backpropagation
    "EarlyBackpropagation",
    "BackpropagationConfig",
    "AutomaticDifferentiation",
    # 1982 - Hopfield Network
    "HopfieldNetwork",
    "HopfieldConfig",
    # Registry and utilities
    "FOUNDATIONAL_METHODS",
    "get_all_methods",
    "get_method_by_id",
    "get_timeline",
    "print_era_summary",
]

# Method registry for programmatic access
FOUNDATIONAL_METHODS = {
    "mcculloch_pitts_1943": McCullochPittsNeuron,
    "hebbian_learning_1949": HebbianLearning,
    "perceptron_1958": Perceptron,
    "adaline_1960": ADALINE,
    "madaline_1962": MADALINE,
    "backpropagation_1974": EarlyBackpropagation,
    "hopfield_network_1982": HopfieldNetwork,
}


def get_all_methods():
    """Return all foundational method info objects."""
    return [cls.get_method_info() for cls in FOUNDATIONAL_METHODS.values()]


def get_method_by_id(method_id: str):
    """Get a specific method by its ID."""
    if method_id in FOUNDATIONAL_METHODS:
        return FOUNDATIONAL_METHODS[method_id].get_method_info()
    raise KeyError(f"Unknown method: {method_id}")


def get_timeline():
    """Return methods in chronological order."""
    methods = get_all_methods()
    return sorted(methods, key=lambda m: m.year)


def print_era_summary():
    """Print a summary of the foundational era."""
    print("=" * 60)
    print("FOUNDATIONAL ERA (1943-1982)")
    print("=" * 60)
    print()
    for method in get_timeline():
        print(f"{method.year} - {method.name}")
        print(f"       Authors: {', '.join(method.authors[:2])}")
        print(f"       Key: {method.key_innovation[:60]}...")
        print()
