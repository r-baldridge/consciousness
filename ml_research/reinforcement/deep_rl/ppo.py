"""
Proximal Policy Optimization (PPO) - 2017

Schulman et al.'s stable and efficient policy gradient algorithm.
PPO constrains policy updates to prevent catastrophically large changes,
achieving strong performance with simple implementation.

Paper: "Proximal Policy Optimization Algorithms" (2017)
arXiv: 1707.06347

Mathematical Formulation:
    Clipped Surrogate Objective:
        L^CLIP(theta) = E[min(r_t(theta) * A_t, clip(r_t(theta), 1-eps, 1+eps) * A_t)]

    Where:
        r_t(theta) = pi(a|s; theta) / pi(a|s; theta_old)  (probability ratio)
        A_t: Advantage estimate (typically GAE)
        eps: Clipping parameter (typically 0.1 or 0.2)

Key Properties:
    - On-policy (but can reuse data for multiple epochs)
    - Simple to implement
    - Robust to hyperparameters
    - State of the art in many domains
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

PPO = MLMethod(
    method_id="ppo_2017",
    name="Proximal Policy Optimization",
    year=2017,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RL_LINE],

    authors=["John Schulman", "Filip Wolski", "Prafulla Dhariwal",
             "Alec Radford", "Oleg Klimov"],
    paper_title="Proximal Policy Optimization Algorithms",
    paper_url="https://arxiv.org/abs/1707.06347",

    key_innovation=(
        "Simplified surrogate objective that constrains policy updates through "
        "clipping, achieving the stability benefits of TRPO without its complexity. "
        "The clipped objective prevents the policy from changing too much in a "
        "single update, enabling multiple epochs of minibatch updates."
    ),

    mathematical_formulation=r"""
Probability Ratio:
    r_t(theta) = pi(a_t | s_t; theta) / pi(a_t | s_t; theta_old)

    r > 1: New policy more likely to take action
    r < 1: New policy less likely to take action

Surrogate Objective (without clipping):
    L^CPI(theta) = E[r_t(theta) * A_t]

    Problem: Can make very large updates when r >> 1 or r << 1

Clipped Surrogate Objective:
    L^CLIP(theta) = E[min(
        r_t(theta) * A_t,
        clip(r_t(theta), 1 - eps, 1 + eps) * A_t
    )]

    Effect of clipping:
        - When A > 0: Clip at 1 + eps (don't increase prob too much)
        - When A < 0: Clip at 1 - eps (don't decrease prob too much)

Value Function Loss:
    L^VF(phi) = (V(s; phi) - V_target)^2

    Value clipping (optional):
        V_clipped = V_old + clip(V - V_old, -eps, eps)
        L^VF = max((V - V_target)^2, (V_clipped - V_target)^2)

Entropy Bonus:
    S(pi) = -sum_a pi(a|s) log pi(a|s)

Total Objective:
    L = L^CLIP - c_1 * L^VF + c_2 * S(pi)

    Typical values: c_1 = 0.5, c_2 = 0.01, eps = 0.2

Generalized Advantage Estimation (GAE):
    A_t^GAE(gamma, lambda) = sum_{l=0}^inf (gamma * lambda)^l * delta_{t+l}
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
""",

    predecessors=["a3c_2016", "trpo"],
    successors=["rlhf_2017", "ppo_fine_tuning"],

    tags=["deep-rl", "policy-gradient", "actor-critic", "on-policy", "stable", "practical"]
)

# Historical Note:
# PPO has become the default choice for many RL applications due to its
# simplicity, stability, and strong performance. It's used in OpenAI Five
# (Dota 2), RLHF for language models (ChatGPT), and robotics. The clipped
# objective is remarkably simple yet effective at preventing destructive
# large policy updates.


# =============================================================================
# Mathematical Functions (Reference Implementation)
# =============================================================================

def compute_probability_ratio(new_log_probs: List[float],
                             old_log_probs: List[float]) -> List[float]:
    """
    Compute probability ratios for PPO.

    r_t = pi_new(a|s) / pi_old(a|s) = exp(log_pi_new - log_pi_old)

    Args:
        new_log_probs: log pi(a|s) under new policy
        old_log_probs: log pi(a|s) under old policy

    Returns:
        Probability ratios
    """
    return {
        "formula": "r = exp(log_pi_new - log_pi_old)",
        "interpretation": {
            "r > 1": "New policy more likely to take this action",
            "r < 1": "New policy less likely to take this action",
            "r = 1": "No change in action probability"
        }
    }


def ppo_clip_objective(ratios: List[float], advantages: List[float],
                       clip_eps: float = 0.2) -> Dict:
    """
    Compute PPO clipped surrogate objective.

    L^CLIP = E[min(r * A, clip(r, 1-eps, 1+eps) * A)]

    Args:
        ratios: Probability ratios r_t
        advantages: Advantage estimates A_t
        clip_eps: Clipping parameter epsilon

    Returns:
        Clipped objective value
    """
    return {
        "formula": "L = mean(min(r * A, clip(r, 1-eps, 1+eps) * A))",
        "unclipped": "r * A",
        "clipped": "clip(r, 1-eps, 1+eps) * A",
        "min_effect": "Takes pessimistic bound - conservative update",
        "cases": {
            "A > 0, r > 1+eps": "Clipped - don't increase prob too much",
            "A > 0, r < 1-eps": "Unclipped - encourage prob increase",
            "A < 0, r < 1-eps": "Clipped - don't decrease prob too much",
            "A < 0, r > 1+eps": "Unclipped - encourage prob decrease"
        }
    }


def ppo_value_loss(values: List[float], returns: List[float],
                   old_values: Optional[List[float]] = None,
                   clip_eps: float = 0.2,
                   use_clipping: bool = True) -> Dict:
    """
    Compute PPO value function loss with optional clipping.

    L^VF = max((V - R)^2, (V_clipped - R)^2)
    V_clipped = V_old + clip(V - V_old, -eps, eps)

    Args:
        values: Current value predictions
        returns: Target returns
        old_values: Previous value predictions
        clip_eps: Clipping parameter
        use_clipping: Whether to use value clipping

    Returns:
        Value loss
    """
    return {
        "basic": "L = (V - R)^2",
        "clipped": {
            "v_clipped": "V_old + clip(V - V_old, -eps, eps)",
            "loss": "max((V - R)^2, (V_clipped - R)^2)"
        },
        "note": "Value clipping is debated - may not always help"
    }


# =============================================================================
# Algorithm Reference
# =============================================================================

@dataclass
class PPOAlgorithm:
    """Reference implementation structure for PPO."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    num_epochs: int = 10
    num_minibatches: int = 32
    lr: float = 3e-4

    @staticmethod
    def algorithm_pseudocode() -> str:
        """Complete PPO algorithm."""
        return """
PPO Algorithm:
    Initialize policy pi(a|s; theta) and value function V(s; phi)

    for iteration = 1, 2, ...:
        # Collect trajectories
        for actor = 1, ..., N:
            Run policy pi_old for T timesteps
            Collect {s_t, a_t, r_t, s_{t+1}, log pi_old(a_t|s_t)}

        # Compute advantages using GAE
        for t = T-1, T-2, ..., 0:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t = delta_t + (gamma * lambda) * A_{t+1}

        # Normalize advantages
        A = (A - mean(A)) / (std(A) + eps)

        # Compute returns
        R_t = A_t + V(s_t)

        # Optimize for K epochs
        for epoch = 1, ..., K:
            # Shuffle and create minibatches
            for minibatch in minibatches:
                # Compute current log probs and values
                log_pi = log pi(a|s; theta)
                V = V(s; phi)

                # Probability ratio
                ratio = exp(log_pi - log_pi_old)

                # Clipped surrogate objective
                L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

                # Value loss
                L_value = (V - R)^2

                # Entropy bonus
                S = entropy(pi)

                # Total loss
                L = -L_clip + c_1 * L_value - c_2 * S

                # Update
                theta, phi <- optimize(L)
"""

    @staticmethod
    def hyperparameters() -> str:
        """Recommended PPO hyperparameters."""
        return """
Common PPO Hyperparameters:

General:
    gamma: 0.99              # Discount factor
    gae_lambda: 0.95         # GAE lambda
    clip_eps: 0.2            # Clipping parameter

Loss coefficients:
    value_coef: 0.5          # Value loss weight
    entropy_coef: 0.01       # Entropy bonus weight

Training:
    num_epochs: 10           # Epochs per iteration
    num_minibatches: 32      # Minibatches per epoch
    batch_size: 2048         # Total samples per iteration
    learning_rate: 3e-4      # Adam learning rate
    max_grad_norm: 0.5       # Gradient clipping

Notes:
    - These are starting points; tune for your domain
    - Atari typically uses fewer epochs (3-4)
    - Continuous control may need larger batches
    - Decrease learning rate over training
"""


# =============================================================================
# PPO Variants and Applications
# =============================================================================

PPO_VARIANTS = {
    "ppo_penalty": {
        "description": "Original PPO variant with KL penalty",
        "objective": "L = E[r*A] - beta * KL(pi_old || pi)",
        "adaptive_beta": "Increase beta if KL too large, decrease if too small",
        "note": "PPO-Clip is more commonly used"
    },
    "ppo_rollback": {
        "description": "Roll back update if KL too large",
        "method": "Check KL after update, revert if > threshold"
    },
    "recurrent_ppo": {
        "description": "PPO with LSTM/GRU for partial observability",
        "key": "Must pass hidden states through rollout"
    },
    "ppo_multi_agent": {
        "applications": ["OpenAI Five (Dota 2)", "Hide and Seek"],
        "technique": "Self-play with population training"
    }
}


PPO_APPLICATIONS = {
    "games": [
        "OpenAI Five (Dota 2)",
        "Capture the Flag",
        "Hide and Seek"
    ],
    "robotics": [
        "Dexterous manipulation",
        "Locomotion control",
        "Sim-to-real transfer"
    ],
    "llm_alignment": [
        "RLHF for instruction following",
        "ChatGPT/InstructGPT training",
        "Constitutional AI"
    ],
    "why_popular": [
        "Simple implementation",
        "Robust to hyperparameters",
        "Scales well",
        "Works across diverse domains"
    ]
}
