"""
Deep Q-Network (DQN) - 2013

Authors: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves,
         Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller (DeepMind)

Paper: "Playing Atari with Deep Reinforcement Learning" (NIPS Workshop, 2013)
       "Human-level control through deep reinforcement learning" (Nature, 2015)

DQN was the breakthrough that demonstrated deep neural networks could learn
directly from high-dimensional sensory inputs (raw pixels) to achieve human-level
performance on complex control tasks. This marked the beginning of the deep
reinforcement learning era.

Key Innovations:
    1. Experience Replay: Store transitions in buffer, sample randomly for training
       - Breaks correlations in sequential data
       - More efficient use of experience (each sample used multiple times)
       - Stabilizes training dynamics

    2. Target Network: Separate network for computing Q-targets
       - Updated periodically (every C steps) by copying Q-network weights
       - Prevents oscillations from bootstrapping moving target
       - Dramatically improves stability

    3. Frame Stacking: Stack last 4 frames as input
       - Provides velocity/motion information
       - Handles partial observability

    4. Preprocessing: Convert to grayscale, resize, clip rewards
       - Reduces input dimensionality
       - Normalizes reward signal across games

Mathematical Formulation:
    Loss Function (Bellman Error):
        L(theta) = E[(r + gamma * max_a' Q(s', a'; theta^-) - Q(s, a; theta))^2]

    Where:
        - theta: Q-network parameters
        - theta^-: Target network parameters (frozen copy)
        - r: Reward
        - gamma: Discount factor
        - s, a, s': Current state, action, next state
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# Mathematical Formulation
# =============================================================================

FORMULATION = r"""
DQN Loss Function (Huber Loss variant):
    L(theta) = E_{(s,a,r,s') ~ D} [L_delta(y - Q(s, a; theta))]

    Where:
        y = r + gamma * max_a' Q(s', a'; theta^-)  (target)
        L_delta(x) = { 0.5 * x^2           if |x| <= delta
                    { delta * |x| - 0.5 * delta^2  otherwise

Gradient:
    nabla_theta L = E[(y - Q(s,a;theta)) * nabla_theta Q(s,a;theta)]

Target Network Update (every C steps):
    theta^- <- theta

Experience Replay Sampling:
    (s, a, r, s', done) ~ Uniform(D)
    Where D is replay buffer of capacity N

Epsilon-Greedy Exploration:
    a = { argmax_a Q(s, a; theta)  with probability 1 - epsilon
        { random action            with probability epsilon

    epsilon_t = max(epsilon_end, epsilon_start - t * (epsilon_start - epsilon_end) / T)

Q-Network Architecture (Atari):
    Input: 84x84x4 (4 stacked grayscale frames)
    Conv1: 32 filters, 8x8, stride 4, ReLU
    Conv2: 64 filters, 4x4, stride 2, ReLU
    Conv3: 64 filters, 3x3, stride 1, ReLU
    FC1: 512 units, ReLU
    Output: |A| units (one per action)

Frame Preprocessing:
    1. Convert RGB to grayscale
    2. Resize to 84x84
    3. Stack last 4 frames
    4. Normalize pixel values to [0, 1]
"""


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DQNConfig:
    """Configuration for DQN algorithm."""

    # Network
    input_shape: tuple = (84, 84, 4)
    hidden_dim: int = 512

    # Learning
    learning_rate: float = 0.00025
    gamma: float = 0.99
    batch_size: int = 32

    # Replay buffer
    buffer_size: int = 1_000_000
    min_buffer_size: int = 50_000  # Start training after this many steps

    # Target network
    target_update_freq: int = 10_000  # Steps between target network updates

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 1_000_000

    # Training
    train_freq: int = 4  # Train every N environment steps
    max_grad_norm: float = 10.0

    # Huber loss
    huber_delta: float = 1.0

    # Frame processing
    frame_skip: int = 4
    num_stacked_frames: int = 4


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

DQN = MLMethod(
    method_id="dqn_2013",
    name="Deep Q-Network (DQN)",
    year=2013,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RL_LINE],

    authors=[
        "Volodymyr Mnih", "Koray Kavukcuoglu", "David Silver", "Alex Graves",
        "Ioannis Antonoglou", "Daan Wierstra", "Martin Riedmiller"
    ],
    paper_title="Playing Atari with Deep Reinforcement Learning",
    paper_url="https://arxiv.org/abs/1312.5602",

    key_innovation=(
        "Combined deep neural networks with Q-learning using experience replay "
        "and target networks to stabilize training. First algorithm to learn "
        "control policies directly from high-dimensional sensory input, achieving "
        "human-level performance on Atari games."
    ),

    mathematical_formulation=r"""
DQN Loss:
    L(theta) = E[(r + gamma * max_a' Q(s', a'; theta^-) - Q(s, a; theta))^2]

Key Components:
    1. Experience Replay: (s, a, r, s') ~ Uniform(D)
    2. Target Network: theta^- updated every C steps
    3. Epsilon-greedy: epsilon decays from 1.0 to 0.1

Architecture (Atari):
    Conv(8x8, 32, stride=4) -> Conv(4x4, 64, stride=2) -> Conv(3x3, 64, stride=1)
    -> FC(512) -> FC(|A|)
""",

    predecessors=["q_learning_1989", "td_learning_1988"],
    successors=["double_dqn_2015", "dueling_dqn_2015", "prioritized_replay_2015", "rainbow_2017"],

    tags=[
        "deep-rl", "value-based", "off-policy", "experience-replay",
        "target-network", "atari", "game-playing", "breakthrough"
    ]
)

# Historical Note:
# DQN marked the beginning of the deep RL revolution. The Nature 2015 paper
# demonstrated human-level or superhuman performance on 29 of 49 Atari games
# using the same architecture and hyperparameters. This work sparked intense
# research interest in deep RL and led to AlphaGo and subsequent breakthroughs.


# =============================================================================
# Main Class
# =============================================================================

class DeepQNetwork:
    """
    Deep Q-Network (DQN) research reference.

    This class provides documentation and pseudocode for the DQN algorithm,
    not a runnable implementation.
    """

    METHOD_ID = "dqn_2013"
    NAME = "Deep Q-Network (DQN)"
    YEAR = 2013
    ERA = MethodEra.DEEP_LEARNING
    CATEGORY = MethodCategory.REINFORCEMENT

    def __init__(self, config: Optional[DQNConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or DQNConfig()

    @staticmethod
    def pseudocode() -> str:
        """Return algorithm pseudocode."""
        return """
DQN Algorithm:

    Initialize replay buffer D with capacity N
    Initialize Q-network with random weights theta
    Initialize target network with weights theta^- = theta

    for episode = 1 to M:
        Initialize state s_1 (stack of 4 preprocessed frames)

        for t = 1 to T:
            # Epsilon-greedy action selection
            With probability epsilon:
                a_t = random action
            Otherwise:
                a_t = argmax_a Q(s_t, a; theta)

            # Execute action and observe
            Execute a_t, observe reward r_t and next frame x_{t+1}
            Preprocess and stack frames to get s_{t+1}

            # Store transition
            Store (s_t, a_t, r_t, s_{t+1}, done) in D

            # Sample and train (every train_freq steps)
            if t % train_freq == 0 and |D| > min_buffer_size:
                Sample minibatch of (s, a, r, s', done) from D

                # Compute targets
                y = r + gamma * max_a' Q(s', a'; theta^-) * (1 - done)

                # Gradient descent on (y - Q(s, a; theta))^2
                Perform gradient descent step on Huber loss

            # Update target network (every target_update_freq steps)
            if t % target_update_freq == 0:
                theta^- <- theta

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon_start - t * decay_rate)

        # Optionally: evaluate and log performance
"""

    @staticmethod
    def key_equations() -> Dict[str, str]:
        """Return key equations in LaTeX-style notation."""
        return {
            "loss_function":
                r"L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} "
                r"\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]",

            "target":
                r"y = r + \gamma \max_{a'} Q(s', a'; \theta^-)",

            "epsilon_decay":
                r"\epsilon_t = \max(\epsilon_{end}, \epsilon_{start} - t \cdot \frac{\epsilon_{start} - \epsilon_{end}}{T})",

            "target_update":
                r"\theta^- \leftarrow \theta \text{ every } C \text{ steps}",

            "huber_loss":
                r"L_\delta(x) = \begin{cases} \frac{1}{2}x^2 & |x| \leq \delta \\ "
                r"\delta|x| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}",
        }

    @staticmethod
    def get_architecture() -> Dict:
        """Return the DQN network architecture for Atari."""
        return {
            "input": "84x84x4 (4 stacked grayscale frames)",
            "layers": [
                {"type": "Conv2D", "filters": 32, "kernel": "8x8", "stride": 4, "activation": "ReLU"},
                {"type": "Conv2D", "filters": 64, "kernel": "4x4", "stride": 2, "activation": "ReLU"},
                {"type": "Conv2D", "filters": 64, "kernel": "3x3", "stride": 1, "activation": "ReLU"},
                {"type": "Flatten"},
                {"type": "Dense", "units": 512, "activation": "ReLU"},
                {"type": "Dense", "units": "|A|", "activation": "Linear"},
            ],
            "output": "Q-values for each action",
            "total_params": "~1.7M for 18 actions",
        }


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for DQN."""
    return DQN


def pseudocode() -> str:
    """Return algorithm pseudocode."""
    return DeepQNetwork.pseudocode()


def key_equations() -> Dict[str, str]:
    """Return key equations."""
    return DeepQNetwork.key_equations()


# =============================================================================
# Variants and Extensions
# =============================================================================

DQN_VARIANTS = {
    "double_dqn": {
        "year": 2015,
        "authors": ["Hado van Hasselt", "Arthur Guez", "David Silver"],
        "paper": "Deep Reinforcement Learning with Double Q-learning",
        "description": "Decouples action selection from evaluation to reduce overestimation",
        "target": "y = r + gamma * Q(s', argmax_a' Q(s', a'; theta); theta^-)",
    },
    "dueling_dqn": {
        "year": 2015,
        "authors": ["Ziyu Wang", "Tom Schaul", "Matteo Hessel", "et al."],
        "paper": "Dueling Network Architectures for Deep Reinforcement Learning",
        "description": "Separates value and advantage streams",
        "formula": "Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')",
    },
    "prioritized_replay": {
        "year": 2015,
        "authors": ["Tom Schaul", "John Quan", "Ioannis Antonoglou", "David Silver"],
        "paper": "Prioritized Experience Replay",
        "description": "Sample transitions based on TD error magnitude",
        "priority": "p_i = |delta_i| + epsilon",
    },
    "noisy_nets": {
        "year": 2017,
        "authors": ["Meire Fortunato", "et al."],
        "paper": "Noisy Networks for Exploration",
        "description": "Parametric noise for exploration instead of epsilon-greedy",
    },
}


def get_historical_context() -> str:
    """Return historical context and significance of DQN."""
    return """
    DQN (2013/2015) marked a watershed moment in AI research, demonstrating that
    deep neural networks could be combined with reinforcement learning to learn
    complex behaviors directly from raw sensory input.

    Before DQN:
        - Q-learning with function approximation was known to be unstable
        - Convergence guarantees only existed for tabular methods
        - RL on high-dimensional inputs required hand-crafted features
        - Playing Atari from pixels seemed impossibly hard

    Key Insights:
        1. Experience replay breaks correlations in sequential data
        2. Target networks provide stable bootstrapping targets
        3. Deep CNNs can extract relevant features from pixels
        4. Same architecture works across many different games

    Impact:
        - Sparked the deep RL revolution
        - Led directly to AlphaGo (2016) and AlphaZero (2017)
        - Inspired policy gradient methods (A3C, PPO)
        - Foundation for robotic control research
        - Nature cover article brought mainstream attention

    The same network architecture and hyperparameters achieved human-level
    performance on 29 of 49 tested Atari games, demonstrating remarkable
    generalization of the approach.
    """
