"""
Asynchronous Advantage Actor-Critic (A3C) - 2016

Mnih et al.'s parallel RL algorithm that eliminated the need for
experience replay by running multiple agents asynchronously.
Introduced advantage estimation for variance reduction.

Paper: "Asynchronous Methods for Deep Reinforcement Learning" (ICML 2016)
arXiv: 1602.01783

Mathematical Formulation:
    Advantage Function:
        A(s,a) = Q(s,a) - V(s)
        Estimated: A_t = sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n V(s_{t+n}) - V(s_t)

    Policy Gradient with Advantage:
        nabla_theta J = E[nabla_theta log pi(a|s; theta) * A(s,a)]

    Value Loss:
        L_V = (V(s) - G_t)^2

Key Innovations:
    1. Asynchronous parallel actors (no replay buffer needed)
    2. Advantage estimation reduces variance
    3. Shared global parameters with local gradients
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

A3C = MLMethod(
    method_id="a3c_2016",
    name="Asynchronous Advantage Actor-Critic",
    year=2016,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RL_LINE],

    authors=["Volodymyr Mnih", "Adria Puigdomenech Badia", "Mehdi Mirza",
             "Alex Graves", "Timothy Lillicrap", "Tim Harley",
             "David Silver", "Koray Kavukcuoglu"],
    paper_title="Asynchronous Methods for Deep Reinforcement Learning",
    paper_url="https://arxiv.org/abs/1602.01783",

    key_innovation=(
        "Multiple parallel actors running asynchronously provide decorrelated "
        "training data, eliminating need for experience replay. This enables "
        "on-policy learning and allows training on CPUs effectively. Combined "
        "with n-step returns and advantage estimation for efficient learning."
    ),

    mathematical_formulation=r"""
Actor-Critic Architecture:
    Policy: pi(a|s; theta) - Probability of action given state
    Value: V(s; phi) - Estimated value of state
    Often shared network with two heads

N-Step Return:
    G_t^(n) = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1} + gamma^n*V(s_{t+n})

Advantage Estimation:
    A_t = G_t^(n) - V(s_t)
        = sum_{k=0}^{n-1} gamma^k*r_{t+k} + gamma^n*V(s_{t+n}) - V(s_t)

Policy Gradient (Actor Loss):
    L_pi = -log pi(a_t|s_t; theta) * A_t

Value Loss (Critic Loss):
    L_V = (G_t^(n) - V(s_t; phi))^2

Entropy Bonus (for exploration):
    H(pi) = -sum_a pi(a|s) * log pi(a|s)
    L_entropy = -beta * H(pi)

Total Loss:
    L = L_pi + c_v * L_V - beta * H(pi)

    c_v: Value loss coefficient (typically 0.5)
    beta: Entropy coefficient (typically 0.01)

Asynchronous Update:
    Each worker:
        1. Copy global params to local
        2. Run n steps in environment
        3. Compute gradients locally
        4. Apply gradients to global params asynchronously
""",

    predecessors=["policy_gradient_1992", "actor_critic", "dqn_2013"],
    successors=["a2c", "ppo_2017", "impala"],

    tags=["deep-rl", "actor-critic", "asynchronous", "parallel", "on-policy", "advantage"]
)

# Historical Note:
# A3C showed that parallel execution could replace experience replay for
# stabilization. The synchronous variant A2C (Advantage Actor-Critic) is
# often preferred in practice - same algorithm but with synchronized updates.
# The architecture (shared network, entropy bonus) influenced PPO and others.


# =============================================================================
# Mathematical Functions (Reference Implementation)
# =============================================================================

def compute_n_step_returns(rewards: List[float], values: List[float],
                           next_value: float, dones: List[bool],
                           gamma: float = 0.99) -> List[float]:
    """
    Compute n-step returns for a trajectory segment.

    G_t = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1} + gamma^n*V(s_{t+n})

    Args:
        rewards: List of rewards [r_t, ..., r_{t+n-1}]
        values: Value estimates [V(s_t), ..., V(s_{t+n-1})]
        next_value: V(s_{t+n}) - bootstrap value
        dones: Terminal flags
        gamma: Discount factor

    Returns:
        List of n-step returns
    """
    return {
        "formula": "G_t = sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * V(s_{t+n})",
        "terminal_handling": "If done, don't bootstrap (G_t = rewards only)",
        "backward_computation": "Compute returns from end of trajectory backward"
    }


def compute_advantages(returns: List[float], values: List[float]) -> List[float]:
    """
    Compute advantage estimates.

    A_t = G_t - V(s_t)

    Args:
        returns: n-step returns
        values: Value estimates

    Returns:
        Advantages
    """
    return {
        "formula": "A_t = G_t - V(s_t)",
        "interpretation": "How much better was action vs. average",
        "normalization": "Often normalize: (A - mean(A)) / (std(A) + eps)"
    }


def actor_critic_loss(log_probs: List[float], advantages: List[float],
                      returns: List[float], values: List[float],
                      entropy: float, value_coef: float = 0.5,
                      entropy_coef: float = 0.01) -> Dict:
    """
    Compute combined actor-critic loss.

    L = L_policy + c_v * L_value - beta * H(pi)

    Args:
        log_probs: log pi(a|s) for actions taken
        advantages: Advantage estimates
        returns: Target returns for value function
        values: Value predictions
        entropy: Policy entropy
        value_coef: Weight for value loss
        entropy_coef: Weight for entropy bonus

    Returns:
        Combined loss and components
    """
    return {
        "policy_loss": "-mean(log_prob * advantage)",
        "value_loss": "mean((return - value)^2)",
        "entropy_bonus": "-entropy_coef * entropy",
        "total": "policy_loss + value_coef * value_loss + entropy_bonus"
    }


# =============================================================================
# Algorithm Reference
# =============================================================================

@dataclass
class A3CAlgorithm:
    """Reference implementation structure for A3C."""

    gamma: float = 0.99
    n_steps: int = 5            # Steps before update
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 40   # Gradient clipping
    num_workers: int = 16

    @staticmethod
    def algorithm_pseudocode() -> str:
        """Complete A3C algorithm."""
        return """
A3C Algorithm:

Global:
    Initialize shared parameters theta (policy) and phi (value)
    Initialize thread step counter T = 0

Each Worker Thread:
    Initialize thread step counter t = 1
    Initialize local parameters theta' = theta, phi' = phi

    repeat:
        # Synchronize with global
        theta' <- theta
        phi' <- phi

        t_start = t
        s = current_state

        # Collect n-step experience
        states, actions, rewards = [], [], []
        while not terminal and t - t_start < n_steps:
            a ~ pi(a|s; theta')
            s', r, done = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = s'
            t += 1
            T += 1

        # Compute n-step returns
        if terminal:
            R = 0
        else:
            R = V(s; phi')  # Bootstrap

        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        # Compute advantages
        advantages = [G - V(s; phi') for G, s in zip(returns, states)]

        # Compute gradients
        d_theta = nabla_theta sum(-log pi(a|s; theta') * A - beta * H(pi))
        d_phi = nabla_phi sum((G - V(s; phi'))^2)

        # Apply gradients to global parameters asynchronously
        theta <- theta + d_theta
        phi <- phi + d_phi

    until T > T_max
"""

    @staticmethod
    def a2c_variant() -> str:
        """A2C: Synchronous version of A3C."""
        return """
A2C (Advantage Actor-Critic) - Synchronous A3C:
    - Same algorithm but all workers synchronized
    - Wait for all workers to finish n steps
    - Average gradients across workers
    - Single update to shared parameters

    Benefits:
        - More efficient GPU utilization
        - Easier to implement and debug
        - Often matches or exceeds A3C performance

    Modern implementations typically use A2C over A3C.
"""


# =============================================================================
# Generalized Advantage Estimation (GAE)
# =============================================================================

def generalized_advantage_estimation(rewards: List[float], values: List[float],
                                     next_value: float, dones: List[bool],
                                     gamma: float = 0.99,
                                     gae_lambda: float = 0.95) -> Dict:
    """
    Generalized Advantage Estimation (Schulman et al., 2015).

    A^GAE = sum_{l=0}^inf (gamma * lambda)^l * delta_{t+l}

    Where: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    Args:
        rewards: Rewards from trajectory
        values: Value estimates
        next_value: V(s_{t+n}) bootstrap
        dones: Terminal flags
        gamma: Discount factor
        gae_lambda: GAE lambda (0=TD, 1=MC)

    Returns:
        GAE advantages
    """
    return {
        "td_residual": "delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)",
        "gae_formula": "A_t^GAE = sum_l (gamma * lambda)^l * delta_{t+l}",
        "lambda_0": "A_t = delta_t = TD(0) advantage",
        "lambda_1": "A_t = sum gamma^l * delta_{t+l} = MC advantage",
        "typical_lambda": "0.95 - balances bias/variance"
    }


# =============================================================================
# Variants and Related Methods
# =============================================================================

A3C_VARIANTS = {
    "a2c": {
        "description": "Synchronous version of A3C",
        "difference": "All workers sync before update",
        "benefit": "Better GPU utilization, easier to implement"
    },
    "impala": {
        "year": 2018,
        "name": "Importance Weighted Actor-Learner Architecture",
        "innovation": "V-trace for off-policy correction",
        "scale": "Distributed training at massive scale"
    },
    "acktr": {
        "year": 2017,
        "name": "Actor-Critic using Kronecker-Factored Trust Region",
        "innovation": "Second-order optimization via K-FAC",
        "benefit": "More sample efficient than A2C"
    },
    "ga3c": {
        "description": "GPU-accelerated A3C",
        "innovation": "Batched inference on GPU"
    }
}
