"""
Deep Reinforcement Learning Methods

Modern RL algorithms that combine deep neural networks with reinforcement
learning. These methods can learn directly from high-dimensional inputs
(images, continuous states) and handle complex control tasks.

Timeline:
    2013: DQN - Deep Q-Networks (Mnih et al.)
    2015: DDPG - Deep Deterministic Policy Gradient
    2016: A3C - Asynchronous Advantage Actor-Critic
    2017: PPO - Proximal Policy Optimization
    2017: Rainbow - Combining DQN improvements
    2018: SAC - Soft Actor-Critic

Key Innovations:
    - Experience replay for sample efficiency
    - Target networks for stability
    - Actor-critic architectures
    - Maximum entropy frameworks
"""

from .dqn import DQN
from .ddpg import DDPG
from .a3c import A3C
from .ppo import PPO
from .sac import SAC
from .rainbow import RAINBOW

__all__ = [
    "DQN",
    "DDPG",
    "A3C",
    "PPO",
    "SAC",
    "RAINBOW",
]
