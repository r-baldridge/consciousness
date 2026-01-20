"""
Temporal Difference Learning - 1988

Sutton's foundational work combining ideas from Monte Carlo methods and
dynamic programming. TD learning bootstraps: it updates estimates based
on other estimates, without waiting for final outcomes.

Paper: "Learning to Predict by the Methods of Temporal Differences" (1988)

Mathematical Formulation:
    TD(0) Update:
        V(s) <- V(s) + alpha * [r + gamma * V(s') - V(s)]

    TD(lambda) with Eligibility Traces:
        V(s) <- V(s) + alpha * delta * e(s)
        Where:
            delta = r + gamma * V(s') - V(s)  (TD error)
            e(s) <- gamma * lambda * e(s) + 1  (eligibility trace)

Key Insight:
    TD methods learn from incomplete episodes by bootstrapping -
    using current estimates to update other estimates.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

TD_LEARNING = MLMethod(
    method_id="td_learning_1988",
    name="Temporal Difference Learning",
    year=1988,

    era=MethodEra.FOUNDATIONAL,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RL_LINE],

    authors=["Richard S. Sutton"],
    paper_title="Learning to Predict by the Methods of Temporal Differences",
    paper_url="https://link.springer.com/article/10.1007/BF00115009",

    key_innovation=(
        "Introduced bootstrapping in reinforcement learning - learning predictions "
        "from predictions without waiting for final outcomes. Combined the sampling "
        "of Monte Carlo with the bootstrapping of dynamic programming."
    ),

    mathematical_formulation=r"""
TD(0) - One-Step TD:
    V(s) <- V(s) + alpha * [r + gamma * V(s') - V(s)]

    TD Target: r + gamma * V(s')
    TD Error: delta = r + gamma * V(s') - V(s)

    Key: Updates V(s) immediately using estimate V(s'), not waiting for episode end

TD(lambda) - Eligibility Traces:
    Forward View (theoretical):
        G_t^(n) = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1} + gamma^n*V(s_{t+n})
        G_t^lambda = (1-lambda) * sum_{n=1}^inf lambda^{n-1} * G_t^(n)

    Backward View (computational):
        delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        e_t(s) = gamma*lambda*e_{t-1}(s) + I(s_t = s)  (accumulating traces)
        V(s) <- V(s) + alpha * delta_t * e_t(s)

    Special Cases:
        lambda = 0: TD(0), one-step bootstrapping
        lambda = 1: Monte Carlo (equivalent to full episode returns)

Unified View (lambda interpolates between TD(0) and MC):
    TD(0): Low variance, some bias (bootstrapping)
    MC:    High variance, zero bias (no bootstrapping)
    TD(lambda): Interpolates between these extremes
""",

    predecessors=["dynamic_programming", "monte_carlo_methods"],
    successors=["q_learning_1989", "sarsa_1994", "td_gammon"],

    tags=["reinforcement-learning", "temporal-difference", "prediction", "bootstrapping", "eligibility-traces"]
)

# Historical Note:
# TD learning is perhaps Sutton's most important contribution to RL.
# The key insight is that we can learn from incomplete sequences by
# bootstrapping - using our current estimates as targets. TD methods
# are the foundation for nearly all modern RL algorithms.


# =============================================================================
# Mathematical Functions (Reference Implementation)
# =============================================================================

def td_zero_update(V: Dict, state, reward, next_state,
                   alpha: float = 0.1, gamma: float = 0.99) -> Dict:
    """
    TD(0) value function update.

    V(s) <- V(s) + alpha * [r + gamma * V(s') - V(s)]

    Args:
        V: Value function as dict {state: value}
        state: Current state
        reward: Reward received
        next_state: Next state
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        Updated V dictionary
    """
    return {
        "formula": "V(s) <- V(s) + alpha * [r + gamma * V(s') - V(s)]",
        "td_target": "r + gamma * V(s')",
        "td_error": "delta = td_target - V(s)",
        "bootstrapping": "Uses estimate V(s') rather than true return"
    }


def td_lambda_update(V: Dict, eligibility: Dict, state, reward, next_state,
                     alpha: float = 0.1, gamma: float = 0.99,
                     lambd: float = 0.9) -> Dict:
    """
    TD(lambda) update with eligibility traces.

    For all states s:
        delta = r + gamma * V(s') - V(s)
        e(s) = gamma * lambda * e(s) + I(s == current_state)
        V(s) <- V(s) + alpha * delta * e(s)

    Args:
        V: Value function
        eligibility: Eligibility traces for all states
        state: Current state
        reward: Reward received
        next_state: Next state
        alpha: Learning rate
        gamma: Discount factor
        lambd: Trace decay parameter (0 = TD(0), 1 = MC)

    Returns:
        Updated V and eligibility dictionaries
    """
    return {
        "td_error": "delta = r + gamma * V(s') - V(s)",
        "trace_update": "e(s) = gamma * lambda * e(s) + I(current_state)",
        "value_update": "V(s) <- V(s) + alpha * delta * e(s) for all s",
        "lambda_0": "Only current state updated (TD(0))",
        "lambda_1": "All visited states updated equally (like MC)"
    }


def compute_n_step_return(rewards: List[float], next_value: float,
                          gamma: float = 0.99, n: int = 1) -> float:
    """
    Compute n-step return (used in n-step TD).

    G_t^(n) = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1} + gamma^n*V(s_{t+n})

    Args:
        rewards: List of n rewards [r_t, r_{t+1}, ..., r_{t+n-1}]
        next_value: V(s_{t+n})
        gamma: Discount factor
        n: Number of steps

    Returns:
        n-step return value
    """
    return {
        "formula": "G^(n) = sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * V(s_{t+n})",
        "n_1": "G^(1) = r + gamma * V(s') (TD(0))",
        "n_inf": "G^(inf) = r + gamma*r' + gamma^2*r'' + ... (MC)",
        "bias_variance": "Larger n: less bias, more variance"
    }


# =============================================================================
# Algorithm Reference
# =============================================================================

@dataclass
class TDLearningAlgorithm:
    """Reference implementation structure for TD Learning."""

    alpha: float = 0.1
    gamma: float = 0.99
    lambd: float = 0.0  # TD(0) by default

    @staticmethod
    def td_zero_pseudocode() -> str:
        """TD(0) prediction algorithm."""
        return """
TD(0) Prediction (for estimating V^pi):
    Initialize V(s) arbitrarily

    for each episode:
        Initialize state s

        while s is not terminal:
            a = policy(s)           # Action from policy being evaluated
            s', r = env.step(a)     # Take action, observe next state and reward

            # TD(0) update
            V(s) <- V(s) + alpha * [r + gamma * V(s') - V(s)]

            s <- s'
"""

    @staticmethod
    def td_lambda_pseudocode() -> str:
        """TD(lambda) with eligibility traces."""
        return """
TD(lambda) Prediction (backward view):
    Initialize V(s) arbitrarily

    for each episode:
        Initialize eligibility traces e(s) = 0 for all s
        Initialize state s

        while s is not terminal:
            a = policy(s)
            s', r = env.step(a)

            # Compute TD error
            delta = r + gamma * V(s') - V(s)

            # Update eligibility of current state
            e(s) <- e(s) + 1  # Accumulating traces

            # Update all state values
            for all states s:
                V(s) <- V(s) + alpha * delta * e(s)
                e(s) <- gamma * lambda * e(s)  # Decay traces

            s <- s'

    Trace Variants:
        Accumulating: e(s) <- e(s) + 1
        Replacing:    e(s) <- 1
        Dutch:        e(s) <- gamma * lambda * e(s) + (1 - alpha * gamma * lambda * e(s))
"""


# =============================================================================
# TD Methods Comparison
# =============================================================================

TD_METHODS_COMPARISON = {
    "monte_carlo": {
        "target": "Actual return G_t = sum_{k=0}^T gamma^k * r_{t+k}",
        "variance": "High (depends on full trajectory)",
        "bias": "Zero (uses true returns)",
        "requires": "Complete episodes",
        "updates": "End of episode"
    },
    "td_zero": {
        "target": "r + gamma * V(s')",
        "variance": "Low (one-step)",
        "bias": "Some (bootstrapping from estimate)",
        "requires": "Single transition",
        "updates": "Every step (online)"
    },
    "td_lambda": {
        "target": "Weighted average of n-step returns",
        "variance": "Medium (tunable)",
        "bias": "Medium (tunable)",
        "requires": "Eligibility traces",
        "updates": "Every step with credit assignment"
    },
    "n_step_td": {
        "target": "n-step return G^(n)",
        "variance": "Between TD(0) and MC",
        "bias": "Between TD(0) and MC",
        "requires": "n transitions stored",
        "updates": "Every step with n-step delay"
    }
}


# =============================================================================
# Historical Significance
# =============================================================================

TD_HISTORICAL = {
    "td_gammon": {
        "year": 1992,
        "author": "Gerald Tesauro",
        "description": "TD-Gammon: TD learning for backgammon",
        "achievement": "World-class play through self-play",
        "significance": "First major success of RL in complex games"
    },
    "dyna": {
        "year": 1991,
        "author": "Rich Sutton",
        "description": "Integrated planning and learning",
        "idea": "Learn model, then use TD with simulated experience"
    },
    "eligibility_traces": {
        "origin": "Klopf (1972), formalized by Sutton (1988)",
        "interpretation": "Credit assignment mechanism",
        "neuroscience": "Related to synaptic eligibility in neurons"
    }
}
