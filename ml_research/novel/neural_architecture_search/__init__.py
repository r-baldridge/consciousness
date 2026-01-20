"""
Neural Architecture Search (NAS) Methods

This module contains research index entries for Neural Architecture Search methods
that automate the design of neural network architectures.

Methods included:
- NAS Basics: Search space, search strategy, performance estimation
- ENAS (2018): Efficient NAS via parameter sharing
- DARTS (2018): Differentiable Architecture Search

NAS represents a paradigm shift from hand-designed to automatically discovered
neural architectures, achieving or surpassing human-designed networks on
benchmarks like CIFAR-10 and ImageNet.
"""

from .nas_basics import (
    NAS_BASICS,
    search_space_types,
    search_strategy_types,
    performance_estimation_types,
)
from .enas import (
    ENAS,
    ENASArchitecture,
    controller_rnn_structure,
)
from .darts import (
    DARTS,
    DARTSArchitecture,
    bilevel_optimization,
)

__all__ = [
    # NAS Basics
    "NAS_BASICS",
    "search_space_types",
    "search_strategy_types",
    "performance_estimation_types",
    # ENAS
    "ENAS",
    "ENASArchitecture",
    "controller_rnn_structure",
    # DARTS
    "DARTS",
    "DARTSArchitecture",
    "bilevel_optimization",
]
