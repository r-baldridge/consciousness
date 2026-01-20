"""
Policy Gradient / REINFORCE - 1992

Williams' REINFORCE algorithm: the foundational policy gradient method.
Unlike value-based methods, policy gradient directly optimizes the policy
by following the gradient of expected return.

Paper: "Simple Statistical Gradient-Following Algorithms for Connectionist
        Reinforcement Learning" (Machine Learning, 1992)

Mathematical Formulation:
    Policy Gradient Theorem:
        nabla_theta J(theta) = E_pi [nabla_theta log pi(a|s; theta) * Q^pi(s,a)]

    REINFORCE Update:
        theta <- theta + alpha * nabla_theta log pi(a|s; theta) * G_t

    Where:
        - pi(a|s; theta): Policy parameterized by theta
        - G_t: Return from time t (Monte Carlo estimate of Q)
        - log pi: Log probability of action (the "score function")

Key Properties:
    - Directly optimizes the policy
    - Naturally handles continuous action spaces
    - High variance (addressed by baselines)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

POLICY_GRADIENT = MLMethod(
    method_id="policy_gradient_1992",
    name="Policy Gradient / REINFORCE",
    year=1992,

    era=MethodEra.CLASSICAL,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RL_LINE],

    authors=["Ronald J. Williams"],
    paper_title="Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning",
    paper_url="https://link.springer.com/article/10.1007/BF00992696",

    key_innovation=(
        "Introduced the REINFORCE algorithm using the likelihood ratio trick "
        "(score function estimator) to compute gradients of expected reward "
        "without differentiating through the environment dynamics."
    ),

    mathematical_formulation=r"""
Policy Gradient Theorem:
    J(theta) = E_pi[sum_t gamma^t * r_t]  # Objective: expected return

    nabla_theta J(theta) = E_pi[nabla_theta log pi(a|s; theta) * Q^pi(s,a)]

    Derivation (Likelihood Ratio / Score Function):
        nabla_theta J = nabla_theta sum_a pi(a|s) Q(s,a)
                      = sum_a Q(s,a) nabla_theta pi(a|s)
                      = sum_a Q(s,a) pi(a|s) nabla_theta log pi(a|s)
                      = E_pi[Q(s,a) nabla_theta log pi(a|s)]

REINFORCE Update (Monte Carlo Policy Gradient):
    theta <- theta + alpha * nabla_theta log pi(a_t|s_t) * G_t

    Where G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k} is the return from time t

Baseline Subtraction (Variance Reduction):
    theta <- theta + alpha * nabla_theta log pi(a_t|s_t) * (G_t - b(s_t))

    Common baselines:
        - b(s) = V(s): Value function (gives advantage A(s,a) = Q(s,a) - V(s))
        - b = mean(G): Average return (simple but effective)

    Key: Baseline doesn't bias gradient, only reduces variance
    Proof: E[nabla log pi * b] = b * sum_a nabla pi(a|s) = b * nabla 1 = 0

Softmax Policy (for discrete actions):
    pi(a|s; theta) = exp(h(s,a;theta)) / sum_a' exp(h(s,a';theta))
    nabla_theta log pi(a|s) = nabla_theta h(s,a) - E_pi[nabla_theta h(s,a')]

Gaussian Policy (for continuous actions):
    pi(a|s; theta) = N(a; mu(s;theta), sigma^2)
    nabla_theta log pi = (a - mu(s)) / sigma^2 * nabla_theta mu(s)
""",

    predecessors=["td_learning_1988", "backprop"],
    successors=["actor_critic", "a3c_2016", "ppo_2017", "trpo"],

    tags=["reinforcement-learning", "policy-gradient", "model-free", "continuous-actions", "monte-carlo"]
)

# Historical Note:
# REINFORCE is elegant but has high variance because it uses Monte Carlo
# returns. This led to actor-critic methods which use learned value functions
# to reduce variance. Despite limitations, policy gradients are the foundation
# of modern deep RL (A3C, PPO, SAC) and RLHF for language models.


# =============================================================================
# Mathematical Functions (Reference Implementation)
# =============================================================================

def reinforce_loss(log_probs: List[float], returns: List[float]) -> Dict:
    """
    Compute REINFORCE loss for a trajectory.

    L = -sum_t log pi(a_t|s_t) * G_t

    Negative because we minimize loss = maximize expected return.

    Args:
        log_probs: Log probabilities of actions taken
        returns: Returns (rewards-to-go) from each timestep

    Returns:
        Loss value and gradient info
    """
    return {
        "formula": "L = -sum_t log pi(a_t|s_t) * G_t",
        "gradient": "nabla L = -sum_t nabla log pi(a_t|s_t) * G_t",
        "interpretation": "Increase prob of actions with positive return",
        "monte_carlo": "Uses full episode returns (high variance)"
    }


def reinforce_with_baseline(log_probs: List[float], returns: List[float],
                            baseline_values: List[float]) -> Dict:
    """
    REINFORCE with baseline for variance reduction.

    L = -sum_t log pi(a_t|s_t) * (G_t - b(s_t))

    The baseline b(s) doesn't change the expected gradient but
    reduces variance significantly.

    Args:
        log_probs: Log probabilities of actions
        returns: Returns from each timestep
        baseline_values: Baseline values b(s) for each state

    Returns:
        Loss with advantage weighting
    """
    return {
        "formula": "L = -sum_t log pi(a_t|s_t) * (G_t - b(s_t))",
        "advantage": "A_t = G_t - b(s_t)",
        "common_baseline": "b(s) = V(s) (learned value function)",
        "unbiased": "E[nabla log pi * b] = 0 (baseline doesn't add bias)"
    }


def compute_policy_gradient(trajectories: List, gamma: float = 0.99) -> Dict:
    """
    Compute policy gradient estimate from sampled trajectories.

    nabla J ~ (1/N) sum_i sum_t nabla log pi(a_t|s_t) * (G_t - b)

    Args:
        trajectories: List of (states, actions, rewards) tuples
        gamma: Discount factor

    Returns:
        Estimated policy gradient
    """
    return {
        "formula": "nabla J = (1/N) sum_i sum_t nabla log pi * (G_t - b)",
        "returns": "G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}",
        "variance_reduction": [
            "Baseline subtraction",
            "Reward normalization",
            "Temporal structure (causality)"
        ]
    }


def gaussian_policy_gradient(action: float, mean: float, std: float) -> Dict:
    """
    Gradient of log probability for Gaussian policy.

    For pi(a|s) = N(a; mu(s), sigma^2):
        nabla_mu log pi = (a - mu) / sigma^2
        nabla_sigma log pi = ((a - mu)^2 - sigma^2) / sigma^3

    Args:
        action: Action taken
        mean: Policy mean mu(s)
        std: Policy standard deviation sigma

    Returns:
        Gradients w.r.t. mean and std
    """
    return {
        "log_prob": "log pi(a|s) = -0.5 * ((a - mu) / sigma)^2 - log(sigma) - 0.5*log(2*pi)",
        "grad_mu": "d/d_mu log pi = (a - mu) / sigma^2",
        "grad_sigma": "d/d_sigma log pi = ((a - mu)^2 - sigma^2) / sigma^3",
        "reparameterization": "a = mu + sigma * epsilon, epsilon ~ N(0,1)"
    }


# =============================================================================
# Algorithm Reference
# =============================================================================

@dataclass
class REINFORCEAlgorithm:
    """Reference implementation structure for REINFORCE."""

    alpha: float = 0.001      # Learning rate
    gamma: float = 0.99       # Discount factor
    use_baseline: bool = True

    @staticmethod
    def algorithm_pseudocode() -> str:
        """Complete REINFORCE algorithm."""
        return """
REINFORCE Algorithm:
    Initialize policy parameters theta randomly
    Initialize baseline b(s) (optional, e.g., value network)

    for each episode:
        # Generate episode
        tau = []
        s = env.reset()
        while not done:
            a ~ pi(a|s; theta)
            s', r, done = env.step(a)
            tau.append((s, a, r))
            s = s'

        # Compute returns
        G = 0
        returns = []
        for (s, a, r) in reversed(tau):
            G = r + gamma * G
            returns.insert(0, G)

        # Update policy
        for t, (s, a, r) in enumerate(tau):
            advantage = returns[t] - b(s)  # or just returns[t] if no baseline
            loss = -log pi(a|s; theta) * advantage
            theta <- theta - alpha * nabla loss

        # Update baseline (if using learned baseline)
        if use_baseline:
            baseline_loss = sum((returns[t] - b(s_t))^2)
            update b to minimize baseline_loss
"""

    @staticmethod
    def variance_reduction_techniques() -> str:
        """Techniques to reduce variance in policy gradients."""
        return """
Variance Reduction Techniques:

1. Baseline Subtraction:
   - Use b(s) to reduce variance without adding bias
   - Common: b(s) = V(s) (value function baseline)
   - Gives advantage: A(s,a) = Q(s,a) - V(s)

2. Reward Normalization:
   - Normalize returns: (G - mean(G)) / std(G)
   - Helps with varying reward scales

3. Temporal Structure (Causality):
   - Don't use future actions to credit past rewards
   - nabla log pi(a_t) should only be weighted by G_t, not G_{<t}

4. Generalized Advantage Estimation (GAE):
   - A_t^GAE = sum_{l=0}^inf (gamma * lambda)^l * delta_{t+l}
   - Interpolates between TD and MC advantage estimates

5. Multiple Trajectories:
   - Average gradients over batch of trajectories
   - Reduces variance by factor of sqrt(N)

6. Control Variates:
   - More sophisticated than simple baseline
   - Correlate with gradient to reduce variance
"""


# =============================================================================
# Policy Gradient Family
# =============================================================================

POLICY_GRADIENT_METHODS = {
    "reinforce": {
        "year": 1992,
        "type": "Monte Carlo Policy Gradient",
        "variance": "High",
        "sample_efficiency": "Low"
    },
    "actor_critic": {
        "year": 1999,
        "description": "Use learned value function as baseline/critic",
        "variance": "Lower (bootstrapping)",
        "bias": "Some (from value function)"
    },
    "a2c": {
        "year": 2016,
        "description": "Advantage Actor-Critic (synchronous)",
        "advantage": "A(s,a) = r + gamma * V(s') - V(s)"
    },
    "a3c": {
        "year": 2016,
        "description": "Asynchronous Advantage Actor-Critic",
        "innovation": "Parallel actors for stability"
    },
    "ppo": {
        "year": 2017,
        "description": "Proximal Policy Optimization",
        "innovation": "Clipped surrogate objective for stability"
    },
    "trpo": {
        "year": 2015,
        "description": "Trust Region Policy Optimization",
        "innovation": "Constrained optimization with KL divergence"
    }
}
