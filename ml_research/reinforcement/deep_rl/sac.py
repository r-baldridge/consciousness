"""
Soft Actor-Critic (SAC) - 2018

Haarnoja et al.'s maximum entropy RL algorithm. SAC augments the
standard RL objective with an entropy bonus, encouraging exploration
and robustness while remaining sample-efficient through off-policy learning.

Paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
        with a Stochastic Actor" (ICML 2018)
arXiv: 1801.01290

Mathematical Formulation:
    Maximum Entropy Objective:
        J(pi) = sum_t E[r(s_t, a_t) + alpha * H(pi(.|s_t))]

    Where:
        H(pi) = -E[log pi(a|s)] (policy entropy)
        alpha: Temperature parameter (how much to value entropy)

Key Properties:
    - Off-policy (uses replay buffer)
    - Stochastic policy (Gaussian for continuous actions)
    - Automatic temperature tuning
    - State of the art sample efficiency for continuous control
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

SAC = MLMethod(
    method_id="sac_2018",
    name="Soft Actor-Critic",
    year=2018,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RL_LINE],

    authors=["Tuomas Haarnoja", "Aurick Zhou", "Pieter Abbeel", "Sergey Levine"],
    paper_title="Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor",
    paper_url="https://arxiv.org/abs/1801.01290",

    key_innovation=(
        "Maximum entropy framework that adds entropy bonus to reward, encouraging "
        "exploration through policy stochasticity. Combined with off-policy learning "
        "and automatic temperature tuning, achieves state-of-the-art sample efficiency "
        "and robustness on continuous control benchmarks."
    ),

    mathematical_formulation=r"""
Maximum Entropy RL Objective:
    J(pi) = sum_{t=0}^T E_{pi}[r(s_t, a_t) + alpha * H(pi(.|s_t))]
          = sum_t E[r + alpha * (-log pi(a|s))]

Soft Value Functions:
    V(s) = E_{a~pi}[Q(s,a) - alpha * log pi(a|s)]
    Q(s,a) = E[r + gamma * V(s')]
           = E[r + gamma * (Q(s',a') - alpha * log pi(a'|s'))]

Soft Bellman Equation:
    Q(s,a) = r + gamma * E_{s'}[V(s')]
           = r + gamma * E_{s',a'}[Q(s',a') - alpha * log pi(a'|s')]

Critic Loss (Two Q-networks):
    y = r + gamma * (min(Q_1(s',a'), Q_2(s',a')) - alpha * log pi(a'|s'))
    L_Q = E[(Q_i(s,a) - y)^2]  for i = 1, 2

Actor Loss:
    L_pi = E_{s,a~pi}[alpha * log pi(a|s) - Q(s,a)]
         = E_s[D_KL(pi(.|s) || exp(Q(s,.))/Z)]

    Note: Minimize KL from policy to softmax of Q

Automatic Temperature (alpha) Adjustment:
    Target entropy: H_target = -dim(A)  (negative action dimension)
    L_alpha = E[-alpha * (log pi(a|s) + H_target)]

    Intuition: Increase alpha if entropy too low, decrease if too high

Reparameterization (for continuous actions):
    a = tanh(mu(s) + sigma(s) * epsilon), epsilon ~ N(0, I)
    log pi(a|s) = log N(u; mu, sigma) - sum log(1 - tanh^2(u))
""",

    predecessors=["ddpg_2015", "soft_q_learning", "maximum_entropy_rl"],
    successors=["sac_discrete", "drq", "curl"],

    tags=["deep-rl", "actor-critic", "maximum-entropy", "off-policy", "continuous-control", "sample-efficient"]
)

# Historical Note:
# SAC is notable for being both sample-efficient (off-policy) and robust
# (entropy regularization). The entropy bonus naturally encourages exploration
# and makes the policy more robust to perturbations. Automatic temperature tuning
# (SAC v2) removed the need to tune alpha manually.


# =============================================================================
# Mathematical Functions (Reference Implementation)
# =============================================================================

def soft_q_target(rewards, next_q1, next_q2, next_log_probs,
                  dones, gamma: float = 0.99, alpha: float = 0.2) -> Dict:
    """
    Compute soft Q-learning target.

    y = r + gamma * (min(Q_1(s',a'), Q_2(s',a')) - alpha * log pi(a'|s'))

    Args:
        rewards: Immediate rewards
        next_q1: Q_1(s', a') where a' ~ pi(.|s')
        next_q2: Q_2(s', a')
        next_log_probs: log pi(a'|s')
        dones: Terminal flags
        gamma: Discount factor
        alpha: Temperature parameter

    Returns:
        Soft Bellman targets
    """
    return {
        "formula": "y = r + gamma * (min(Q_1', Q_2') - alpha * log_pi') * (1 - done)",
        "min_trick": "Use minimum of two Q-networks to reduce overestimation",
        "entropy_penalty": "-alpha * log_pi' adds entropy bonus to target"
    }


def actor_loss_sac(log_probs, q_values, alpha: float = 0.2) -> Dict:
    """
    Compute SAC actor loss.

    L_pi = E[alpha * log pi(a|s) - Q(s,a)]

    Minimize this = maximize Q while keeping entropy high.

    Args:
        log_probs: log pi(a|s) for sampled actions
        q_values: min(Q_1(s,a), Q_2(s,a))
        alpha: Temperature

    Returns:
        Actor loss
    """
    return {
        "formula": "L = mean(alpha * log_pi - Q(s,a))",
        "interpretation": "Maximize Q while maximizing entropy",
        "reparameterization": "Sample a = tanh(mu + sigma * eps), backprop through"
    }


def temperature_loss(log_probs, target_entropy: float, alpha) -> Dict:
    """
    Compute loss for automatic temperature adjustment.

    L_alpha = E[-alpha * (log pi(a|s) + H_target)]

    Args:
        log_probs: log pi(a|s)
        target_entropy: Target entropy (typically -dim(action_space))
        alpha: Current temperature

    Returns:
        Temperature loss
    """
    return {
        "formula": "L = mean(-alpha * (log_pi + H_target))",
        "target_entropy": "-dim(A) for continuous, -0.98 * log(|A|) for discrete",
        "gradient": "If entropy < target: increase alpha; else decrease",
        "log_alpha": "Often optimize log(alpha) for numerical stability"
    }


def squashed_gaussian_log_prob(mean, std, action) -> Dict:
    """
    Compute log probability for squashed (tanh) Gaussian.

    Since a = tanh(u) where u ~ N(mu, sigma^2):
        log pi(a|s) = log N(u; mu, sigma) - sum log(1 - tanh^2(u))

    The second term is the Jacobian correction for the tanh transform.

    Args:
        mean: Policy mean mu(s)
        std: Policy standard deviation sigma(s)
        action: Squashed action a = tanh(u)

    Returns:
        Log probability
    """
    return {
        "unsquashed": "u = arctanh(a) = 0.5 * log((1+a)/(1-a))",
        "gaussian_log_prob": "log N(u; mu, sigma) = -0.5 * ((u - mu)/sigma)^2 - log(sigma) - 0.5*log(2*pi)",
        "jacobian_correction": "-sum log(1 - a^2 + eps)",
        "total": "gaussian_log_prob + jacobian_correction"
    }


# =============================================================================
# Algorithm Reference
# =============================================================================

@dataclass
class SACAlgorithm:
    """Reference implementation structure for SAC."""

    gamma: float = 0.99
    tau: float = 0.005           # Soft update rate
    alpha: float = 0.2           # Initial temperature (if not auto-tuned)
    auto_alpha: bool = True      # Automatic temperature
    target_entropy: float = None # Auto-set to -dim(action)
    lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000

    @staticmethod
    def algorithm_pseudocode() -> str:
        """Complete SAC algorithm."""
        return """
SAC Algorithm:
    Initialize:
        - Policy network pi(a|s; theta)
        - Two Q-networks Q_1(s,a; phi_1), Q_2(s,a; phi_2)
        - Target Q-networks: phi_1' = phi_1, phi_2' = phi_2
        - Temperature alpha (and log_alpha if auto-tuning)
        - Replay buffer D

    for each iteration:
        # Collect experience
        for each environment step:
            a ~ pi(.|s; theta)
            s', r, done = env.step(a)
            D.store((s, a, r, s', done))
            s = s'

        # Update for each gradient step:
        for each gradient step:
            # Sample batch
            batch = D.sample(batch_size)

            # Compute target Q
            a' ~ pi(.|s'; theta)
            log_pi' = log pi(a'|s')
            y = r + gamma * (1 - done) * (min(Q_1'(s',a'), Q_2'(s',a')) - alpha * log_pi')

            # Update critics
            L_Q1 = mean((Q_1(s,a) - y)^2)
            L_Q2 = mean((Q_2(s,a) - y)^2)
            phi_1 <- phi_1 - lr * nabla L_Q1
            phi_2 <- phi_2 - lr * nabla L_Q2

            # Update actor
            a_new ~ pi(.|s; theta)  # Reparameterized
            log_pi = log pi(a_new|s)
            L_pi = mean(alpha * log_pi - min(Q_1(s,a_new), Q_2(s,a_new)))
            theta <- theta - lr * nabla L_pi

            # Update temperature (if auto-tuning)
            L_alpha = mean(-alpha * (log_pi + target_entropy))
            log_alpha <- log_alpha - lr * nabla L_alpha

            # Soft update targets
            phi_1' <- tau * phi_1 + (1-tau) * phi_1'
            phi_2' <- tau * phi_2 + (1-tau) * phi_2'
"""

    @staticmethod
    def network_architecture() -> str:
        """Typical SAC network architecture."""
        return """
Actor (Policy) Network:
    Input: State s [state_dim]
    FC1: 256 units, ReLU
    FC2: 256 units, ReLU
    Mean: Linear(256, action_dim)
    Log_std: Linear(256, action_dim)

    Output: mu, log_sigma
    Action: a = tanh(mu + exp(log_sigma) * epsilon)

    Note: Log_std typically clamped to [-20, 2]

Critic (Q) Networks (two separate):
    Input: State s [state_dim], Action a [action_dim]
    Concat: [s, a]
    FC1: 256 units, ReLU
    FC2: 256 units, ReLU
    Output: Linear(256, 1)  # Single Q-value
"""


# =============================================================================
# SAC Variants and Extensions
# =============================================================================

SAC_VARIANTS = {
    "sac_v1": {
        "year": 2018,
        "paper": "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL",
        "alpha": "Fixed or manually tuned"
    },
    "sac_v2": {
        "year": 2018,
        "paper": "Soft Actor-Critic Algorithms and Applications",
        "innovation": "Automatic temperature adjustment",
        "benefit": "One less hyperparameter to tune"
    },
    "sac_discrete": {
        "description": "SAC for discrete action spaces",
        "policy": "Categorical instead of Gaussian",
        "entropy": "Sum over discrete actions"
    },
    "rad_sac": {
        "year": 2020,
        "name": "Reinforcement Learning with Augmented Data",
        "innovation": "Data augmentation for vision-based RL"
    },
    "drq": {
        "year": 2020,
        "name": "Data-Regularized Q (DrQ)",
        "innovation": "Image augmentation in Q-learning"
    },
    "curl": {
        "year": 2020,
        "name": "Contrastive Unsupervised Representations for RL",
        "innovation": "Contrastive learning for representation"
    }
}


# =============================================================================
# Maximum Entropy Framework
# =============================================================================

MAXIMUM_ENTROPY_RL = {
    "motivation": [
        "Exploration: Stochastic policy explores naturally",
        "Robustness: Policy captures multiple solutions",
        "Compositionality: Can combine learned skills",
        "Optimization: Smoother objective landscape"
    ],
    "connection_to_vi": {
        "description": "Soft Q-learning is variational inference",
        "policy": "Variational distribution q(a|s)",
        "target": "Optimal policy exp(Q(s,a))/Z (energy-based)",
        "kl_divergence": "Actor loss minimizes KL(pi || exp(Q)/Z)"
    },
    "temperature_interpretation": {
        "high_alpha": "More exploration, more entropic policy",
        "low_alpha": "More exploitation, deterministic-like",
        "alpha_to_zero": "Recovers standard RL objective"
    }
}
