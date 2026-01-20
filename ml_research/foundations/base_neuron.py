"""
Base Neuron - Abstract Base Class

Abstract base class defining the interface for all neuron models
in the foundational era. Provides common structure for McCulloch-Pitts,
Perceptron, ADALINE, and other early neuron models.

This is a research index template, not a runnable implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


@dataclass
class BaseNeuronConfig:
    """Base configuration for neuron models."""
    n_inputs: int = 2
    threshold: float = 0.5
    learning_rate: float = 0.1
    weights_init: str = "zeros"  # "zeros", "random", "ones"


class BaseNeuron(ABC):
    """
    Abstract base class for foundational neuron models.

    All neuron models in the foundations era share common characteristics:
    - Input aggregation (weighted sum or similar)
    - Activation/threshold function
    - Optional learning rule

    Subclasses implement specific behaviors for each historical model.
    """

    # Class attributes to be overridden
    METHOD_ID: str = "base_neuron"
    NAME: str = "Base Neuron"
    YEAR: int = 0
    ERA: MethodEra = MethodEra.FOUNDATIONAL
    CATEGORY: MethodCategory = MethodCategory.NEURON_MODEL

    @classmethod
    @abstractmethod
    def get_method_info(cls) -> MLMethod:
        """Return the MLMethod entry for this neuron model."""
        pass

    @staticmethod
    @abstractmethod
    def pseudocode() -> str:
        """Return pseudocode describing the model's operation."""
        pass

    @staticmethod
    @abstractmethod
    def key_equations() -> Dict[str, str]:
        """Return key equations in LaTeX-style notation."""
        pass

    @staticmethod
    def activation_functions() -> Dict[str, str]:
        """Return descriptions of applicable activation functions."""
        return {
            "step": "f(x) = 1 if x >= theta else 0",
            "sign": "f(x) = +1 if x >= 0 else -1",
            "linear": "f(x) = x",
            "sigmoid": "f(x) = 1 / (1 + exp(-x))",
        }

    @classmethod
    def get_lineage(cls) -> List["MethodLineage"]:
        """Return the lineages this model belongs to."""
        return []

    @classmethod
    def get_historical_context(cls) -> str:
        """Return historical context and significance."""
        return ""

    @classmethod
    def get_limitations(cls) -> List[str]:
        """Return known limitations of the model."""
        return []

    @classmethod
    def get_applications(cls) -> List[str]:
        """Return historical and modern applications."""
        return []
