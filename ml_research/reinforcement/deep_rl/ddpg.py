"""
Deep Deterministic Policy Gradient (DDPG) - 2015

Lillicrap et al.'s actor-critic algorithm for continuous control.
Combines DQN's stability techniques (replay, target networks) with
deterministic policy gradients for continuous action spaces.

Paper: "Continuous Control with Deep Reinforcement Learning" (2015)
arXiv: 1509.02971

Mathematical Formulation:
    Deterministic Policy Gradient (Silver et al., 2014):
        nabla_theta J = E[nabla_a Q(s,a)|_{a=mu(s)} * nabla_theta mu(s)]

    DDPG uses:
        - Actor mu(s; theta): Deterministic policy (outputs action directly)
        - Critic Q(s,a; phi): Action-value function

Key Properties:
    - Off-policy (uses replay buffer)
    - Continuous action spaces
    - Deterministic policy + noise for exploration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

DDPG = MLMethod(
    method_id="ddpg_2015",
    name="Deep Deterministic Policy Gradient",
    year=2015,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RL_LINE],

    authors=["Timothy P. Lillicrap", "Jonathan J. Hunt", "Alexander Pritzel",
             "Nicolas Heess", "Tom Erez", "Yuval Tassa", "David Silver", "Daan Wierstra"],
    paper_title="Continuous Control with Deep Reinforcement Learning",
    paper_url="https://arxiv.org/abs/1509.02971",

    key_innovation=(
        "Extended DQN to continuous action spaces using deterministic policy gradients. "
        "Combines actor-critic architecture with DQN's replay buffer and target networks. "
        "Enabled deep RL for robotics and continuous control tasks."
    ),

    mathematical_formulation=r"""
Architecture:
    Actor mu(s; theta): State -> Action (deterministic)
    Critic Q(s, a; phi): (State, Action) -> Value

Critic Update (like DQN):
    y = r + gamma * Q(s', mu(s'; theta^-); phi^-)
    L(phi) = E[(y - Q(s, a; phi))^2]

Actor Update (Deterministic Policy Gradient):
    nabla_theta J = E[nabla_a Q(s, a; phi)|_{a=mu(s)} * nabla_theta mu(s; theta)]

    Interpretation: Move policy in direction that increases Q

Target Networks (Soft Update):
    theta^- <- tau * theta + (1 - tau) * theta^-
    phi^- <- tau * phi + (1 - tau) * phi^-
    tau << 1 (typically 0.001)

Exploration (Ornstein-Uhlenbeck Noise):
    a_t = mu(s_t; theta) + N_t
    N_t = theta_ou * (mu_ou - N_{t-1}) + sigma_ou * W_t
    (Temporally correlated noise for physical systems)

Alternative: Gaussian Noise
    a_t = mu(s_t; theta) + epsilon, epsilon ~ N(0, sigma^2)
""",

    predecessors=["dqn_2013", "deterministic_policy_gradient_2014"],
    successors=["td3", "sac_2018"],

    tags=["deep-rl", "actor-critic", "continuous-control", "off-policy", "deterministic-policy"]
)

# Historical Note:
# DDPG was the first deep RL algorithm to work well on continuous control tasks
# like MuJoCo. However, it can be brittle - hyperparameter sensitive and can
# fail catastrophically. This led to improvements like TD3 and SAC.


# =============================================================================
# Mathematical Functions (Reference Implementation)
# =============================================================================

def critic_loss(q_values, rewards, next_q_values, dones,
                gamma: float = 0.99) -> Dict:
    """
    Compute DDPG critic loss.

    L = E[(y - Q(s, a))^2]
    y = r + gamma * Q_target(s', mu_target(s'))

    Args:
        q_values: Q(s, a) from current critic
        rewards: Rewards received
        next_q_values: Q(s', mu(s')) from TARGET critic and actor
        dones: Terminal flags
        gamma: Discount factor

    Returns:
        Critic MSE loss
    """
    return {
        "formula": "L_critic = mean((y - Q(s, a))^2)",
        "td_target": "y = r + gamma * Q_target(s', mu_target(s')) * (1 - done)",
        "note": "Next action from target actor, Q from target critic"
    }


def actor_loss(q_values) -> Dict:
    """
    Compute DDPG actor loss (negative of Q for gradient ascent).

    L = -E[Q(s, mu(s))]

    We want to maximize Q, so minimize -Q.

    Args:
        q_values: Q(s, mu(s)) from critic

    Returns:
        Actor loss (negative mean Q)
    """
    return {
        "formula": "L_actor = -mean(Q(s, mu(s)))",
        "gradient": "nabla_theta mu * nabla_a Q",
        "interpretation": "Move policy toward higher Q actions"
    }


def ornstein_uhlenbeck_noise(previous_noise: float, mu: float = 0.0,
                             theta: float = 0.15, sigma: float = 0.2) -> float:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    dN = theta * (mu - N) * dt + sigma * dW

    Properties:
        - Mean-reverting
        - Temporally correlated (smooth)
        - Good for physical systems with momentum

    Args:
        previous_noise: N_{t-1}
        mu: Mean to revert to (usually 0)
        theta: Rate of mean reversion
        sigma: Volatility

    Returns:
        New noise value
    """
    return {
        "formula": "N_t = N_{t-1} + theta * (mu - N_{t-1}) * dt + sigma * sqrt(dt) * randn()",
        "properties": "Mean-reverting, temporally correlated",
        "use_case": "Smooth exploration for continuous control"
    }


# =============================================================================
# Algorithm Reference
# =============================================================================

@dataclass
class DDPGAlgorithm:
    """Reference implementation structure for DDPG."""

    gamma: float = 0.99
    tau: float = 0.001           # Soft update rate
    actor_lr: float = 0.0001
    critic_lr: float = 0.001
    batch_size: int = 64
    buffer_size: int = 1000000

    @staticmethod
    def algorithm_pseudocode() -> str:
        """Complete DDPG algorithm."""
        return """
DDPG Algorithm:
    Initialize actor mu(s; theta) and critic Q(s, a; phi)
    Initialize target networks: theta^- <- theta, phi^- <- phi
    Initialize replay buffer D
    Initialize OU noise process N

    for episode = 1 to M:
        Initialize state s_1
        Reset OU noise: N_0 = 0

        for t = 1 to T:
            # Select action with exploration noise
            a_t = mu(s_t; theta) + N_t
            a_t = clip(a_t, action_low, action_high)

            # Execute action
            s_{t+1}, r_t, done = env.step(a_t)

            # Store transition
            D.store((s_t, a_t, r_t, s_{t+1}, done))

            # Sample mini-batch
            batch = D.sample(batch_size)

            # Update critic
            for (s, a, r, s', done) in batch:
                y = r + gamma * Q(s', mu(s'; theta^-); phi^-) * (1 - done)
            L_critic = mean((y - Q(s, a; phi))^2)
            phi <- phi - lr_critic * nabla_phi L_critic

            # Update actor
            L_actor = -mean(Q(s, mu(s; theta); phi))
            theta <- theta - lr_actor * nabla_theta L_actor

            # Soft update target networks
            theta^- <- tau * theta + (1 - tau) * theta^-
            phi^- <- tau * phi + (1 - tau) * phi^-
"""

    @staticmethod
    def network_architecture() -> str:
        """Typical DDPG network architecture."""
        return """
Actor Network:
    Input: State s [state_dim]
    FC1: 400 units, ReLU
    FC2: 300 units, ReLU
    Output: tanh(Linear(300, action_dim)) * action_scale

    Note: tanh bounds output to [-1, 1], then scale to action range

Critic Network:
    Input: State s [state_dim]
    State path: FC1(state_dim, 400), ReLU
    Action inserted: concat(FC1_output, action)
    FC2: 300 units on concatenated input, ReLU
    Output: Linear(300, 1)  # Single Q-value

    Note: Action inserted after first layer (empirically better)
"""


# =============================================================================
# DDPG Variants and Improvements
# =============================================================================

DDPG_VARIANTS = {
    "td3": {
        "year": 2018,
        "name": "Twin Delayed DDPG",
        "authors": ["Fujimoto", "van Hoof", "Meger"],
        "improvements": [
            "Twin critics: Take min of two Q-networks",
            "Delayed policy updates: Update actor less frequently",
            "Target policy smoothing: Add noise to target action"
        ],
        "paper": "Addressing Function Approximation Error in Actor-Critic Methods"
    },
    "d4pg": {
        "year": 2018,
        "name": "Distributed Distributional DDPG",
        "authors": ["Barth-Maron", "et al."],
        "improvements": [
            "Distributional critic (returns full distribution)",
            "N-step returns",
            "Prioritized experience replay",
            "Distributed training"
        ]
    },
    "cem_ddpg": {
        "description": "Combine with Cross-Entropy Method",
        "idea": "Use CEM population for exploration"
    }
}


# =============================================================================
# Comparison with Other Methods
# =============================================================================

CONTINUOUS_CONTROL_COMPARISON = {
    "ddpg": {
        "policy": "Deterministic",
        "exploration": "Additive noise (OU or Gaussian)",
        "issues": "Brittle, overestimates Q"
    },
    "td3": {
        "policy": "Deterministic",
        "exploration": "Gaussian noise + target smoothing",
        "improvement": "Addresses Q overestimation"
    },
    "sac": {
        "policy": "Stochastic (Gaussian)",
        "exploration": "Entropy bonus (intrinsic)",
        "benefit": "More robust, principled exploration"
    },
    "ppo_continuous": {
        "policy": "Stochastic (Gaussian)",
        "type": "On-policy",
        "note": "Less sample efficient but more stable"
    }
}
