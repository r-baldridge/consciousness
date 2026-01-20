"""
Rainbow DQN - 2017

Hessel et al.'s comprehensive study combining six improvements to DQN.
Demonstrated that these enhancements are largely complementary and
together achieve state-of-the-art performance on Atari.

Paper: "Rainbow: Combining Improvements in Deep Reinforcement Learning" (AAAI 2018)
arXiv: 1710.02298

Components Combined:
    1. Double DQN - Reduce overestimation
    2. Prioritized Experience Replay - Focus on important transitions
    3. Dueling Networks - Separate value and advantage
    4. Multi-step Learning - n-step returns
    5. Distributional RL - Learn return distribution (C51)
    6. Noisy Networks - Parameter-space exploration

Key Finding:
    Each component contributes; combining all achieves best results.
    Prioritized replay and multi-step were most important individually.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

RAINBOW = MLMethod(
    method_id="rainbow_2017",
    name="Rainbow DQN",
    year=2017,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RL_LINE],

    authors=["Matteo Hessel", "Joseph Modayil", "Hado van Hasselt",
             "Tom Schaul", "Georg Ostrovski", "Will Dabney",
             "Dan Horgan", "Bilal Piot", "Mohammad Azar", "David Silver"],
    paper_title="Rainbow: Combining Improvements in Deep Reinforcement Learning",
    paper_url="https://arxiv.org/abs/1710.02298",

    key_innovation=(
        "Systematic integration of six orthogonal improvements to DQN, demonstrating "
        "that they are largely complementary. Achieved state-of-the-art on Atari with "
        "significantly improved sample efficiency and final performance."
    ),

    mathematical_formulation=r"""
Rainbow combines six improvements:

1. Double DQN (reduce overestimation):
    y = r + gamma * Q_target(s', argmax_a Q(s', a))
    (Use online network to select action, target to evaluate)

2. Prioritized Experience Replay:
    P(i) = p_i^alpha / sum_j p_j^alpha
    p_i = |delta_i| + epsilon  (priority based on TD error)
    w_i = (N * P(i))^{-beta}  (importance sampling weights)

3. Dueling Networks:
    Q(s,a) = V(s) + A(s,a) - mean_a' A(s,a')
    (Separate streams for value and advantage)

4. Multi-step Learning:
    G_t^{(n)} = sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n Q(s_{t+n}, a*)
    (Use n-step returns, typically n=3)

5. Distributional RL (C51):
    Model distribution of returns: Z(s,a) = d (not just E[Z])
    Support: 51 atoms from V_min to V_max
    Output: p_i(s,a) = probability of return in bin i
    Project Bellman update onto fixed support

6. Noisy Networks:
    y = Wx + b  becomes  y = (mu_w + sigma_w * epsilon_w)x + (mu_b + sigma_b * epsilon_b)
    (Learnable noise for exploration, no epsilon-greedy needed)

Combined Loss (Distributional + Prioritized):
    L = sum_i w_i * D_KL(m || p(s,a))
    m: Projected target distribution
    w_i: Importance sampling weight
""",

    predecessors=["dqn_2013", "double_dqn", "dueling_dqn", "prioritized_replay", "c51", "noisy_nets"],
    successors=["iqn", "r2d2", "agent57"],

    tags=["deep-rl", "dqn", "distributional", "atari", "sample-efficient", "comprehensive"]
)

# Historical Note:
# Rainbow showed that many DQN improvements are orthogonal and can be combined.
# The ablation study found prioritized replay and multi-step learning contributed
# most individually. Rainbow became a strong baseline for Atari and influenced
# subsequent work like IQN, R2D2, and Agent57.


# =============================================================================
# Component Details
# =============================================================================

RAINBOW_COMPONENTS = {
    "double_dqn": {
        "paper": "Deep Reinforcement Learning with Double Q-learning",
        "authors": ["van Hasselt", "Guez", "Silver"],
        "year": 2015,
        "problem": "Q-learning overestimates values due to max operator",
        "solution": "Decouple action selection from evaluation",
        "formula": "y = r + gamma * Q_target(s', argmax_a Q_online(s',a))",
        "intuition": "Online network picks action, target evaluates it"
    },

    "prioritized_replay": {
        "paper": "Prioritized Experience Replay",
        "authors": ["Schaul", "Quan", "Antonoglou", "Silver"],
        "year": 2015,
        "problem": "Uniform sampling wastes capacity on easy transitions",
        "solution": "Sample proportional to TD error magnitude",
        "formula": {
            "priority": "p_i = |delta_i|^alpha + epsilon",
            "probability": "P(i) = p_i / sum_j p_j",
            "importance": "w_i = (1/N * 1/P(i))^beta"
        },
        "params": {
            "alpha": "0.6 (prioritization exponent)",
            "beta": "0.4 -> 1.0 (importance sampling annealing)"
        }
    },

    "dueling_networks": {
        "paper": "Dueling Network Architectures for Deep RL",
        "authors": ["Wang", "Schaul", "Hessel", "van Hasselt", "Lanctot", "de Freitas"],
        "year": 2016,
        "problem": "Q values don't generalize well across actions",
        "solution": "Separate value and advantage streams",
        "formula": "Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))",
        "benefit": "Learn state value even when actions don't matter"
    },

    "multi_step": {
        "concept": "Use n-step returns instead of single-step",
        "formula": "G_t^(n) = sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n V(s_{t+n})",
        "n": "3 (in Rainbow)",
        "benefit": "Faster reward propagation",
        "tradeoff": "More bias from bootstrap, less variance"
    },

    "distributional_rl": {
        "paper": "A Distributional Perspective on Reinforcement Learning",
        "authors": ["Bellemare", "Dabney", "Munos"],
        "year": 2017,
        "problem": "Expected value loses distributional information",
        "solution": "Learn full return distribution",
        "method": "C51 - 51 atoms over fixed support",
        "formula": {
            "support": "z_i linearly spaced in [V_min, V_max]",
            "output": "p_i(s,a) = softmax(network_output)",
            "q_value": "Q(s,a) = sum_i z_i * p_i(s,a)",
            "loss": "KL divergence to projected Bellman target"
        },
        "params": {
            "num_atoms": 51,
            "v_min": -10,
            "v_max": 10
        }
    },

    "noisy_nets": {
        "paper": "Noisy Networks for Exploration",
        "authors": ["Fortunato", "Azar", "Piot", "et al."],
        "year": 2017,
        "problem": "Epsilon-greedy is state-independent",
        "solution": "Add learnable noise to network weights",
        "formula": "y = (mu_w + sigma_w * eps_w)x + (mu_b + sigma_b * eps_b)",
        "benefit": "State-dependent, self-annealing exploration",
        "note": "No epsilon schedule needed"
    }
}


# =============================================================================
# Algorithm Reference
# =============================================================================

@dataclass
class RainbowAlgorithm:
    """Reference implementation structure for Rainbow."""

    # Hyperparameters
    gamma: float = 0.99
    n_step: int = 3
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    target_update_freq: int = 32000
    batch_size: int = 32
    lr: float = 6.25e-5
    adam_eps: float = 1.5e-4

    @staticmethod
    def algorithm_pseudocode() -> str:
        """Complete Rainbow algorithm."""
        return """
Rainbow Algorithm:
    Initialize:
        - Noisy dueling distributional network Q(s,a; theta)
        - Target network Q(s,a; theta^-)
        - Prioritized replay buffer D

    for frame = 1 to max_frames:
        # Select action (noisy net provides exploration)
        a = argmax_a E[Z(s,a)]  # No epsilon-greedy needed

        # Execute and store with max priority
        s', r, done = env.step(a)
        D.store((s, a, r, s', done), priority=max_priority)

        # Sample prioritized batch
        batch, indices, weights = D.sample(batch_size)

        # Compute n-step distributional Bellman target
        for each transition (s, a, R^(n), s_n, done):
            if done:
                target_dist = delta(R^(n))  # Point mass at reward
            else:
                # Double DQN action selection
                a* = argmax_a E[Z(s_n, a; theta)]

                # Get target distribution
                target_dist = project_distribution(
                    R^(n) + gamma^n * Z(s_n, a*; theta^-)
                )

        # Compute distributional loss with importance weights
        loss = sum_i w_i * KL(target_dist_i || Z(s_i, a_i; theta))

        # Update priorities
        for i, delta in enumerate(td_errors):
            D.update_priority(indices[i], |delta| + eps)

        # Gradient descent
        theta <- theta - lr * nabla loss

        # Update target network periodically
        if frame % target_update_freq == 0:
            theta^- <- theta
"""

    @staticmethod
    def distributional_projection() -> str:
        """Project distribution onto support."""
        return """
Distributional Projection (Categorical DQN / C51):

Support: z = [z_0, ..., z_{N-1}] linearly spaced in [V_min, V_max]
Delta_z = (V_max - V_min) / (N - 1)

For target distribution after Bellman backup:
    1. Compute shifted support: z' = r + gamma * z
    2. Clip to [V_min, V_max]
    3. Project onto original support

Projection:
    For each atom z'_j:
        b_j = (z'_j - V_min) / Delta_z
        l = floor(b_j), u = ceil(b_j)

        # Distribute probability to neighbors
        m_l += p_j * (u - b_j)
        m_u += p_j * (b_j - l)

Result: m is the projected distribution on original support
"""


# =============================================================================
# Ablation Study Results
# =============================================================================

RAINBOW_ABLATIONS = {
    "full_rainbow": {
        "median_human_normalized": "223%",
        "note": "All components combined"
    },
    "no_double": {
        "drop": "Moderate",
        "note": "Overestimation hurts performance"
    },
    "no_priority": {
        "drop": "Large",
        "note": "One of most important components"
    },
    "no_dueling": {
        "drop": "Moderate",
        "note": "Helps generalization"
    },
    "no_multistep": {
        "drop": "Large",
        "note": "Crucial for sample efficiency"
    },
    "no_distributional": {
        "drop": "Moderate to Large",
        "note": "Especially important for some games"
    },
    "no_noisy": {
        "drop": "Small to Moderate",
        "note": "Can use epsilon-greedy instead"
    },
    "ranking": [
        "Multi-step (most important)",
        "Prioritized replay",
        "Distributional",
        "Double",
        "Dueling",
        "Noisy (least important but still helps)"
    ]
}


# =============================================================================
# Successors
# =============================================================================

RAINBOW_SUCCESSORS = {
    "iqn": {
        "year": 2018,
        "name": "Implicit Quantile Networks",
        "improvement": "Learn quantile function instead of fixed atoms"
    },
    "r2d2": {
        "year": 2019,
        "name": "Recurrent Replay Distributed DQN",
        "innovation": "Add LSTM for partial observability + distributed"
    },
    "agent57": {
        "year": 2020,
        "name": "Agent57",
        "achievement": "Superhuman on all 57 Atari games",
        "techniques": "Population-based training, adaptive exploration"
    },
    "muesli": {
        "year": 2021,
        "description": "Combines model-based planning with Rainbow"
    }
}
