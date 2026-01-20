"""
Q-Learning - 1989

Watkins' off-policy temporal difference control algorithm. Learns the
optimal action-value function Q*(s,a) directly, independent of the
policy being followed.

Paper: "Learning from Delayed Rewards" (PhD Thesis, 1989)
       "Q-Learning" (Machine Learning, 1992)

Mathematical Formulation:
    Q-Learning Update Rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    Where:
        - Q(s,a): Action-value function
        - alpha: Learning rate
        - gamma: Discount factor
        - r: Immediate reward
        - s': Next state
        - max_a' Q(s',a'): Maximum Q-value over all actions in next state

    Bellman Optimality Equation:
        Q*(s,a) = E[r + gamma * max_a' Q*(s',a') | s, a]

Key Properties:
    - Off-policy: Learns Q* while following any policy with exploration
    - Model-free: No need for environment model
    - Tabular: Original form uses table lookup
    - Guaranteed convergence to Q* under certain conditions
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

Q_LEARNING = MLMethod(
    method_id="q_learning_1989",
    name="Q-Learning",
    year=1989,

    era=MethodEra.CLASSICAL,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RL_LINE],

    authors=["Christopher J.C.H. Watkins"],
    paper_title="Learning from Delayed Rewards",
    paper_url="http://www.cs.rhul.ac.uk/~chrisw/thesis.html",

    key_innovation=(
        "First off-policy temporal difference control algorithm that directly "
        "learns the optimal action-value function Q* without requiring a model "
        "of the environment. Proved convergence to optimal policy."
    ),

    mathematical_formulation=r"""
Q-Learning Update Rule:
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    TD Target: r + gamma * max_a' Q(s',a')
    TD Error: delta = r + gamma * max_a' Q(s',a') - Q(s,a)

Bellman Optimality Equation (Fixed Point):
    Q*(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * max_a' Q*(s',a')]

Optimal Policy Extraction:
    pi*(s) = argmax_a Q*(s,a)

Exploration Strategy (epsilon-greedy):
    a = { argmax_a Q(s,a)  with probability 1 - epsilon
        { random action    with probability epsilon

Convergence Conditions (Watkins & Dayan, 1992):
    1. All state-action pairs visited infinitely often
    2. sum_t alpha_t = infinity
    3. sum_t alpha_t^2 < infinity
    4. Bounded rewards
""",

    predecessors=["td_learning_1988"],
    successors=["dqn_2013", "double_q_learning", "sarsa"],

    tags=["reinforcement-learning", "off-policy", "model-free", "temporal-difference", "tabular"]
)

# Historical Note:
# Q-Learning is perhaps the most important algorithm in RL history.
# Its off-policy nature allows separation of exploration and exploitation.
# The key insight is using max over next-state actions regardless of actual
# action taken, enabling learning optimal policy while following exploratory behavior.


# =============================================================================
# Mathematical Functions (Reference Implementation)
# =============================================================================

def q_learning_update(Q: Dict, state, action, reward, next_state,
                      alpha: float = 0.1, gamma: float = 0.99,
                      actions: List = None) -> Dict:
    """
    Single Q-Learning update step.

    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    Args:
        Q: Action-value function as dict {(state, action): value}
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Resulting state
        alpha: Learning rate
        gamma: Discount factor
        actions: List of possible actions

    Returns:
        Updated Q dictionary
    """
    return {
        "formula": "Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]",
        "td_target": "r + gamma * max_a' Q(s',a')",
        "td_error": "delta = td_target - Q(s,a)",
        "key_property": "Off-policy: uses max over actions regardless of policy"
    }


def epsilon_greedy_policy(Q: Dict, state, epsilon: float, actions: List):
    """
    Epsilon-greedy action selection for exploration.

    pi(a|s) = { 1 - epsilon + epsilon/|A|  if a = argmax Q(s,a)
              { epsilon / |A|               otherwise

    Args:
        Q: Action-value function
        state: Current state
        epsilon: Exploration probability
        actions: Available actions

    Returns:
        Selected action
    """
    return {
        "formula": "With prob epsilon: random, else argmax_a Q(s,a)",
        "exploration": "Ensures all actions tried infinitely often",
        "decay": "Often epsilon decreases over time: epsilon_t = max(epsilon_min, epsilon_0 * decay^t)"
    }


def compute_td_error(Q: Dict, state, action, reward, next_state,
                     gamma: float, actions: List) -> float:
    """
    Compute the temporal difference error.

    delta = r + gamma * max_a' Q(s',a') - Q(s,a)

    The TD error measures surprise: how much better (or worse) was the
    outcome compared to our current estimate?

    Args:
        Q: Action-value function
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Resulting state
        gamma: Discount factor
        actions: Available actions

    Returns:
        TD error value
    """
    return {
        "formula": "delta = r + gamma * max_a' Q(s',a') - Q(s,a)",
        "interpretation": "Prediction error: actual vs expected",
        "positive_delta": "Outcome better than expected",
        "negative_delta": "Outcome worse than expected"
    }


# =============================================================================
# Algorithm Reference
# =============================================================================

@dataclass
class QLearningAlgorithm:
    """Reference implementation structure for Q-Learning."""

    alpha: float = 0.1          # Learning rate
    gamma: float = 0.99         # Discount factor
    epsilon: float = 0.1        # Exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01

    @staticmethod
    def algorithm_pseudocode() -> str:
        """Complete Q-Learning algorithm."""
        return """
Q-Learning Algorithm:
    Initialize Q(s,a) arbitrarily (often to zeros)

    for each episode:
        Initialize state s

        while s is not terminal:
            # Choose action using epsilon-greedy
            if random() < epsilon:
                a = random action
            else:
                a = argmax_a Q(s,a)

            # Take action, observe result
            s', r = environment.step(a)

            # Q-Learning update (key step)
            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

            s <- s'

        # Optionally decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
"""

    @staticmethod
    def convergence_theorem() -> str:
        """Q-Learning convergence guarantee."""
        return """
Convergence Theorem (Watkins & Dayan, 1992):
    Q-Learning converges to Q* with probability 1 if:

    1. State-action pairs represented discretely
    2. Q(s,a) stored in lookup table
    3. Each (s,a) pair visited infinitely often
    4. Learning rates satisfy:
        - sum_t alpha_t(s,a) = infinity (for all s,a)
        - sum_t alpha_t(s,a)^2 < infinity
    5. Rewards bounded: |r| <= R_max
    6. Discount factor 0 <= gamma < 1

    Note: With function approximation (neural nets), convergence
    is NOT guaranteed - this led to the innovations in DQN.
"""


# =============================================================================
# Variants and Extensions
# =============================================================================

Q_LEARNING_VARIANTS = {
    "double_q_learning": {
        "year": 2010,
        "authors": ["Hado van Hasselt"],
        "description": "Uses two Q-functions to reduce overestimation bias",
        "update": "Q_A(s,a) <- Q_A(s,a) + alpha * [r + gamma * Q_B(s', argmax_a' Q_A(s',a')) - Q_A(s,a)]",
        "benefit": "More stable learning, reduced maximization bias"
    },
    "delayed_q_learning": {
        "year": 2006,
        "description": "PAC-MDP variant with exploration bounds",
        "property": "Provably efficient exploration"
    },
    "speedy_q_learning": {
        "year": 2011,
        "description": "Faster convergence via stochastic gradient descent view",
        "benefit": "Better sample complexity"
    }
}
