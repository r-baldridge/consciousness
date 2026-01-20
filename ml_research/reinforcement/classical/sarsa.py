"""
SARSA - 1994

State-Action-Reward-State-Action: On-policy temporal difference control.
Unlike Q-Learning, SARSA learns the value of the policy being followed,
not the optimal policy.

Paper: "On-Line Q-Learning Using Connectionist Systems" (Rummery & Niranjan, 1994)
Name coined by: Sutton (1996)

Mathematical Formulation:
    SARSA Update Rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]

    Where:
        - Q(s,a): Action-value function for current policy
        - s, a: Current state and action
        - r: Reward received
        - s', a': Next state and ACTUAL next action (chosen by policy)
        - alpha: Learning rate
        - gamma: Discount factor

Key Difference from Q-Learning:
    Q-Learning: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    SARSA:      Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]

    Q-Learning uses max (off-policy), SARSA uses actual action a' (on-policy)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

SARSA = MLMethod(
    method_id="sarsa_1994",
    name="SARSA",
    year=1994,

    era=MethodEra.CLASSICAL,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RL_LINE],

    authors=["Gavin A. Rummery", "Mahesan Niranjan"],
    paper_title="On-Line Q-Learning Using Connectionist Systems",
    paper_url="https://www.cs.cmu.edu/afs/cs/academic/class/15782-f06/slides/rummeryThesis.pdf",

    key_innovation=(
        "On-policy temporal difference control that learns the value of the policy "
        "being followed rather than the optimal policy. The name SARSA reflects the "
        "quintuple (S, A, R, S', A') used in each update."
    ),

    mathematical_formulation=r"""
SARSA Update Rule:
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]

    TD Target: r + gamma * Q(s',a')
    TD Error: delta = r + gamma * Q(s',a') - Q(s,a)

    Note: a' is the ACTUAL action selected by the policy (not max)

On-Policy Property:
    - Evaluates and improves the policy being followed
    - Q converges to Q^pi for the behavior policy pi
    - If pi is epsilon-greedy w.r.t. Q, converges to Q* as epsilon -> 0

Comparison with Q-Learning:
    Q-Learning (off-policy): delta = r + gamma * max_a' Q(s',a') - Q(s,a)
    SARSA (on-policy):       delta = r + gamma * Q(s',a') - Q(s,a)

Expected SARSA:
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * E_pi[Q(s',a')] - Q(s,a)]
    Where: E_pi[Q(s',a')] = sum_a' pi(a'|s') * Q(s',a')
""",

    predecessors=["td_learning_1988", "q_learning_1989"],
    successors=["expected_sarsa", "sarsa_lambda"],

    tags=["reinforcement-learning", "on-policy", "model-free", "temporal-difference", "tabular"]
)

# Historical Note:
# SARSA is generally safer than Q-Learning in practice because it accounts for
# exploration in its value estimates. In the cliff-walking problem, SARSA learns
# a safer path while Q-Learning learns the optimal but riskier edge path.
# The name was coined by Rich Sutton, standing for State-Action-Reward-State-Action.


# =============================================================================
# Mathematical Functions (Reference Implementation)
# =============================================================================

def sarsa_update(Q: Dict, state, action, reward, next_state, next_action,
                 alpha: float = 0.1, gamma: float = 0.99) -> Dict:
    """
    Single SARSA update step.

    Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]

    Args:
        Q: Action-value function
        state: Current state s
        action: Current action a
        reward: Reward r
        next_state: Next state s'
        next_action: Next action a' (already chosen by policy)
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        Updated Q dictionary
    """
    return {
        "formula": "Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]",
        "td_target": "r + gamma * Q(s',a')",
        "key_difference": "Uses actual next action a', not max over actions",
        "on_policy": "Learns value of policy being followed"
    }


def expected_sarsa_update(Q: Dict, state, action, reward, next_state,
                          policy, actions: List,
                          alpha: float = 0.1, gamma: float = 0.99) -> Dict:
    """
    Expected SARSA update - uses expectation over next actions.

    Q(s,a) <- Q(s,a) + alpha * [r + gamma * sum_a' pi(a'|s') Q(s',a') - Q(s,a)]

    This reduces variance compared to SARSA while maintaining on-policy nature.

    Args:
        Q: Action-value function
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Next state
        policy: Policy function pi(a|s) returning probabilities
        actions: List of possible actions
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        Updated Q dictionary
    """
    return {
        "formula": "Q(s,a) <- Q(s,a) + alpha * [r + gamma * E[Q(s',a')] - Q(s,a)]",
        "expectation": "E[Q(s',a')] = sum_a' pi(a'|s') * Q(s',a')",
        "benefit": "Lower variance than SARSA",
        "special_case": "When pi is greedy, Expected SARSA = Q-Learning"
    }


# =============================================================================
# Algorithm Reference
# =============================================================================

@dataclass
class SARSAAlgorithm:
    """Reference implementation structure for SARSA."""

    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1

    @staticmethod
    def algorithm_pseudocode() -> str:
        """Complete SARSA algorithm."""
        return """
SARSA Algorithm:
    Initialize Q(s,a) arbitrarily

    for each episode:
        Initialize state s
        Choose action a using policy derived from Q (e.g., epsilon-greedy)

        while s is not terminal:
            Take action a, observe r, s'
            Choose a' from s' using policy derived from Q  # KEY: choose BEFORE update

            # SARSA update
            Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]

            s <- s'
            a <- a'
"""

    @staticmethod
    def sarsa_lambda_pseudocode() -> str:
        """SARSA with eligibility traces."""
        return """
SARSA(lambda) Algorithm:
    Initialize Q(s,a) arbitrarily

    for each episode:
        Initialize eligibility traces e(s,a) = 0 for all s,a
        Initialize s, choose a from policy

        while s is not terminal:
            Take action a, observe r, s'
            Choose a' from s' using policy

            # Compute TD error
            delta = r + gamma * Q(s',a') - Q(s,a)

            # Update eligibility (accumulating traces)
            e(s,a) <- e(s,a) + 1

            # Update all Q values
            for all (s,a):
                Q(s,a) <- Q(s,a) + alpha * delta * e(s,a)
                e(s,a) <- gamma * lambda * e(s,a)

            s <- s', a <- a'

    Note: lambda = 0 gives standard SARSA
          lambda = 1 gives Monte Carlo-like behavior
"""


# =============================================================================
# Comparison: SARSA vs Q-Learning
# =============================================================================

SARSA_VS_QLEARNING = {
    "sarsa": {
        "type": "On-policy",
        "update": "Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]",
        "learns": "Value of policy being followed",
        "behavior": "Conservative in risky environments",
        "example": "Cliff walking: learns safe path away from cliff"
    },
    "q_learning": {
        "type": "Off-policy",
        "update": "Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]",
        "learns": "Optimal value function Q*",
        "behavior": "Can be riskier during learning",
        "example": "Cliff walking: learns optimal path along cliff edge"
    },
    "key_insight": (
        "Q-Learning is optimistic about future actions (assumes optimal), "
        "while SARSA is realistic about what the current policy will actually do. "
        "This makes SARSA safer when exploration is dangerous."
    )
}


# =============================================================================
# Variants
# =============================================================================

SARSA_VARIANTS = {
    "expected_sarsa": {
        "year": 2009,
        "authors": ["van Seijen", "van Hasselt", "Wiering", "Whiteson"],
        "update": "Uses E[Q(s',a')] instead of Q(s',a')",
        "benefit": "Lower variance, interpolates between SARSA and Q-Learning"
    },
    "sarsa_lambda": {
        "description": "SARSA with eligibility traces",
        "parameter": "lambda in [0,1] controls trace decay",
        "lambda_0": "Standard SARSA (one-step)",
        "lambda_1": "Monte Carlo-like (full episode)"
    },
    "true_online_sarsa": {
        "year": 2014,
        "authors": ["van Seijen", "Sutton"],
        "description": "Exact online equivalence to forward view",
        "benefit": "Better theoretical properties with function approximation"
    }
}
