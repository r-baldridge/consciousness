"""
Neuro-Symbolic Methods

This module contains research index entries for neuro-symbolic AI methods
that combine neural network learning with symbolic reasoning capabilities.

Methods included:
- Neural Turing Machine (2014): Memory-augmented neural networks
- Differentiable Programming: Making discrete operations differentiable
- Neurosymbolic AI: Hybrid neural + symbolic reasoning systems

Neuro-symbolic approaches bridge the gap between:
- Neural networks: Learning from data, pattern recognition, generalization
- Symbolic AI: Logic, reasoning, interpretability, compositionality
"""

from .neural_turing import (
    NEURAL_TURING_MACHINE,
    NTMArchitecture,
    memory_addressing,
    read_write_operations,
)
from .differentiable_programming import (
    DIFFERENTIABLE_PROGRAMMING,
    soft_attention_mechanisms,
    gumbel_softmax,
    neural_module_networks,
)
from .neurosymbolic_ai import (
    NEUROSYMBOLIC_AI,
    neural_theorem_prover,
    knowledge_graph_reasoning,
    concept_learning,
)

__all__ = [
    # Neural Turing Machine
    "NEURAL_TURING_MACHINE",
    "NTMArchitecture",
    "memory_addressing",
    "read_write_operations",
    # Differentiable Programming
    "DIFFERENTIABLE_PROGRAMMING",
    "soft_attention_mechanisms",
    "gumbel_softmax",
    "neural_module_networks",
    # Neurosymbolic AI
    "NEUROSYMBOLIC_AI",
    "neural_theorem_prover",
    "knowledge_graph_reasoning",
    "concept_learning",
]
