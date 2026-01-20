"""
Reinforcement Learning Methods - Research Index

A comprehensive collection of reinforcement learning algorithms from classical
methods (Q-Learning, SARSA, TD Learning) to modern deep RL (DQN, PPO, SAC)
and alignment techniques (RLHF).

Timeline:
    1988: Temporal Difference Learning (Sutton)
    1989: Q-Learning (Watkins)
    1992: Policy Gradient / REINFORCE (Williams)
    1994: SARSA (Rummery & Niranjan)
    2013: DQN (Mnih et al.)
    2015: DDPG (Lillicrap et al.)
    2016: A3C (Mnih et al.)
    2017: PPO (Schulman et al.)
    2017: Rainbow (Hessel et al.)
    2017: RLHF (Christiano et al.)
    2018: SAC (Haarnoja et al.)
    2022: InstructGPT RLHF (Ouyang et al.)
"""

# Classical RL methods
from .classical import (
    Q_LEARNING,
    SARSA,
    TD_LEARNING,
    POLICY_GRADIENT,
)

# Deep RL methods
from .deep_rl import (
    DQN,
    DDPG,
    A3C,
    PPO,
    SAC,
    RAINBOW,
)

# RLHF for LLM alignment
from .rlhf import RLHF

__all__ = [
    # Classical
    "Q_LEARNING",
    "SARSA",
    "TD_LEARNING",
    "POLICY_GRADIENT",
    # Deep RL
    "DQN",
    "DDPG",
    "A3C",
    "PPO",
    "SAC",
    "RAINBOW",
    # Alignment
    "RLHF",
]
