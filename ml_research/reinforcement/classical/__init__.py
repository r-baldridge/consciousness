"""
Classical Reinforcement Learning Methods

Foundational algorithms that established the theoretical basis for
reinforcement learning. These methods work with tabular representations
and form the basis for modern deep RL approaches.

Key Concepts:
    - Value Functions: V(s), Q(s,a)
    - Temporal Difference Learning
    - On-policy vs Off-policy
    - Model-free vs Model-based
"""

from .q_learning import Q_LEARNING
from .sarsa import SARSA
from .td_learning import TD_LEARNING
from .policy_gradient import POLICY_GRADIENT

__all__ = [
    "Q_LEARNING",
    "SARSA",
    "TD_LEARNING",
    "POLICY_GRADIENT",
]
