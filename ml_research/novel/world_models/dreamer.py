"""
Dreamer (2019-2023) - Hafner et al., DeepMind

The Dreamer family of world model algorithms that learn behaviors
through imagination using the Recurrent State Space Model (RSSM).

Papers:
- DreamerV1 (2019): "Dream to Control: Learning Behaviors by Latent Imagination"
- DreamerV2 (2020): "Mastering Atari with Discrete World Models"
- DreamerV3 (2023): "Mastering Diverse Domains through World Models"

Mathematical Core (RSSM):
    Deterministic path: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    Stochastic state:   z_t ~ p(z_t | h_t) [prior]
                        z_t ~ q(z_t | h_t, o_t) [posterior]

    Full state: s_t = (h_t, z_t)

Key Innovation:
    Actor-critic learning entirely in imagination using learned
    value estimates, enabling sample-efficient model-based RL.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entries
# =============================================================================

DREAMER_V1 = MLMethod(
    method_id="dreamer_v1_2019",
    name="DreamerV1",
    year=2019,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RNN_LINE, MethodLineage.GENERATIVE_LINE],

    authors=["Danijar Hafner", "Timothy Lillicrap", "Ian Fischer",
             "Ruben Villegas", "David Ha", "Honglak Lee", "James Davidson"],
    paper_title="Dream to Control: Learning Behaviors by Latent Imagination",
    paper_url="https://arxiv.org/abs/1912.01603",

    key_innovation=(
        "Introduced actor-critic learning entirely in imagination using the RSSM "
        "(Recurrent State Space Model). Instead of CMA-ES used in World Models, "
        "Dreamer learns both value function and policy via backpropagation through "
        "imagined trajectories, enabling efficient gradient-based policy optimization."
    ),

    mathematical_formulation=r"""
RSSM (Recurrent State Space Model):
    Deterministic state: h_t = f_theta(h_{t-1}, z_{t-1}, a_{t-1})
    Prior:               z_t ~ p_theta(z_t | h_t)
    Posterior:           z_t ~ q_phi(z_t | h_t, o_t)

    Full model state: s_t = (h_t, z_t)

Observation Model:
    o_t ~ p_theta(o_t | h_t, z_t)

Reward Model:
    r_t ~ p_theta(r_t | h_t, z_t)

World Model Loss:
    L_model = E[ -ln p(o_t|s_t) - ln p(r_t|s_t) + KL(q(z_t|h_t,o_t) || p(z_t|h_t)) ]

Actor-Critic in Imagination:
    # Imagine trajectories using prior (no observations needed)
    s_tau, a_tau ~ p_theta, pi_psi  for tau = t, t+1, ..., t+H

    # Value estimation with lambda-returns
    V_lambda(s_tau) = (1-lambda) sum_{n=1}^{H-1} lambda^{n-1} V_n(s_tau) + lambda^{H-1} V_H(s_tau)
    V_n(s_tau) = E[sum_{k=tau}^{tau+n-1} gamma^{k-tau} r_k + gamma^n v_xi(s_{tau+n})]

    # Actor loss: maximize imagined returns
    L_actor = -E[sum_{tau=t}^{t+H} V_lambda(s_tau)]

    # Critic loss: predict lambda-returns
    L_critic = E[sum_{tau=t}^{t+H} (v_xi(s_tau) - V_lambda(s_tau))^2]
""",

    predecessors=["world_models_2018", "planet_2018"],
    successors=["dreamer_v2_2020", "dreamer_v3_2023"],

    tags=[
        "world-model", "model-based-rl", "imagination", "actor-critic",
        "rssm", "latent-dynamics", "lambda-returns"
    ],
    notes=(
        "DreamerV1 uses continuous latent states z_t. Evaluated on DeepMind Control Suite, "
        "achieving 20x better sample efficiency than model-free methods. The RSSM enables "
        "accurate long-horizon imagination by combining deterministic and stochastic components."
    )
)


DREAMER_V2 = MLMethod(
    method_id="dreamer_v2_2020",
    name="DreamerV2",
    year=2020,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RNN_LINE, MethodLineage.GENERATIVE_LINE],

    authors=["Danijar Hafner", "Timothy Lillicrap", "Mohammad Norouzi", "Jimmy Ba"],
    paper_title="Mastering Atari with Discrete World Models",
    paper_url="https://arxiv.org/abs/2010.02193",

    key_innovation=(
        "Replaced continuous latent states with discrete representations using "
        "categorical distributions. This enables more expressive world models that "
        "can capture multimodal state distributions and diverse game dynamics. "
        "First world model approach to achieve human-level performance on Atari."
    ),

    mathematical_formulation=r"""
Discrete RSSM:
    Deterministic: h_t = f_theta(h_{t-1}, z_{t-1}, a_{t-1})
    Prior:         z_t ~ Categorical(p_theta(z_t | h_t))
    Posterior:     z_t ~ Categorical(q_phi(z_t | h_t, o_t))

    Discrete latent: z_t in {0,1}^{K x L}
    K = number of categorical variables (e.g., 32)
    L = number of classes per variable (e.g., 32)
    Total discrete states: L^K (e.g., 32^32)

Straight-Through Gradients:
    Forward: z = one_hot(argmax(logits))
    Backward: dL/d_logits = dL/d_z (straight-through estimator)

KL Balancing:
    L_KL = alpha * KL(sg(posterior) || prior) + (1-alpha) * KL(posterior || sg(prior))

    Where sg = stop_gradient
    alpha = 0.8 (prioritize training posterior to match prior)

Actor Loss with Entropy:
    L_actor = -E[sum_tau (V_lambda(s_tau) - eta * H(pi(a|s_tau)))]

    Entropy bonus encourages exploration
""",

    predecessors=["dreamer_v1_2019", "vq_vae_2017"],
    successors=["dreamer_v3_2023"],

    tags=[
        "world-model", "model-based-rl", "discrete-latents", "atari",
        "categorical", "straight-through", "kl-balancing"
    ],
    notes=(
        "DreamerV2 achieved human-level performance on all 55 Atari games using only "
        "200M environment frames. The discrete latent space with 32 categorical variables "
        "of 32 classes provides a 32^32 state space - far more expressive than continuous."
    )
)


DREAMER_V3 = MLMethod(
    method_id="dreamer_v3_2023",
    name="DreamerV3",
    year=2023,

    era=MethodEra.NOVEL,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RNN_LINE, MethodLineage.ATTENTION_LINE],

    authors=["Danijar Hafner", "Jurgis Pasukonis", "Jimmy Ba", "Timothy Lillicrap"],
    paper_title="Mastering Diverse Domains through World Models",
    paper_url="https://arxiv.org/abs/2301.04104",

    key_innovation=(
        "A single fixed set of hyperparameters works across radically different domains "
        "(Atari, continuous control, Minecraft, DMLab). Key innovations: symlog predictions "
        "for scale-invariance, percentile-based return normalization, and free bits for "
        "KL regularization. First algorithm to collect diamonds in Minecraft from scratch."
    ),

    mathematical_formulation=r"""
Symlog Encoding (Scale Invariance):
    symlog(x) = sign(x) * ln(|x| + 1)
    symexp(x) = sign(x) * (exp(|x|) - 1)

    Decoder predicts: symlog(target)
    Loss: ||symlog(target) - prediction||^2
    Decoding: output = symexp(prediction)

Percentile Return Normalization:
    R_norm = (R - Percentile_5(R)) / (Percentile_95(R) - Percentile_5(R))

    Robust to reward scale across domains

Free Bits for KL:
    L_KL = max(free_bits, KL(posterior || prior))

    Prevents posterior collapse while allowing some free nats

Two-Hot Encoding for Regression:
    Discretize continuous targets into B bins
    Predict distribution over bins using two adjacent bins
    Enables distributional value learning

Actor Objective:
    L_actor = -E[sum_tau R_norm(s_tau)] - eta * H(pi)

    With return normalization and entropy regularization

Full Model Architecture:
    Encoder: CNN -> MLP -> posterior logits
    RSSM: GRU(h, z, a) -> MLP -> prior logits
    Decoder: MLP -> CNN (reconstruct o)
    Reward: MLP(h, z) -> two-hot distribution
    Continue: MLP(h, z) -> Bernoulli (episode continuation)
    Actor: MLP(h, z) -> action distribution
    Critic: MLP(h, z) -> two-hot value distribution
""",

    predecessors=["dreamer_v2_2020"],
    successors=[],

    tags=[
        "world-model", "model-based-rl", "symlog", "minecraft",
        "domain-agnostic", "two-hot", "free-bits", "foundation-rl"
    ],
    notes=(
        "DreamerV3 is a major step toward general-purpose world models. The same "
        "hyperparameters work for BSuite, Atari, DMC, Crafter, and Minecraft without "
        "tuning. In Minecraft, it learns from scratch to collect diamonds - requiring "
        "~30 minute episodes with sparse rewards and complex subgoals."
    )
)


def get_dreamer_v1_info() -> MLMethod:
    """Return the MLMethod entry for DreamerV1."""
    return DREAMER_V1


def get_dreamer_v2_info() -> MLMethod:
    """Return the MLMethod entry for DreamerV2."""
    return DREAMER_V2


def get_dreamer_v3_info() -> MLMethod:
    """Return the MLMethod entry for DreamerV3."""
    return DREAMER_V3


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class RSSMArchitecture:
    """Reference architecture for Recurrent State Space Model."""

    # State dimensions
    deterministic_dim: int = 4096  # GRU hidden state (DreamerV3)
    stochastic_dim: int = 32      # Number of categorical variables
    classes_per_var: int = 32     # Classes per categorical variable

    # Network dimensions
    hidden_dim: int = 1024
    cnn_depth: int = 96

    @staticmethod
    def rssm_structure() -> str:
        """RSSM core structure."""
        return """
RSSM (Recurrent State Space Model):

Components:
    h_t: Deterministic state (GRU hidden, captures history)
    z_t: Stochastic state (captures uncertainty)
    s_t = (h_t, z_t): Full model state

Update Equations:
    1. Deterministic transition (GRU):
       h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])

    2. Prior (imagination, no observation):
       z_t^prior ~ Categorical(MLP(h_t))
       logits_prior = MLP(h_t)  # [batch, K, L]

    3. Posterior (learning, with observation):
       e_t = Encoder(o_t)
       z_t^post ~ Categorical(MLP([h_t, e_t]))
       logits_post = MLP([h_t, e_t])  # [batch, K, L]

For DreamerV2/V3 (discrete):
    z_t is a [K, L] binary tensor (K one-hot vectors of dim L)
    Flattened: z_t in R^{K*L} (e.g., 32*32 = 1024)
    Sampling: straight-through Gumbel-softmax

For DreamerV1 (continuous):
    z_t ~ N(mu(h_t), sigma(h_t))
    z_t in R^{stochastic_dim}
"""

    @staticmethod
    def imagination_rollout() -> str:
        """Imagination rollout procedure."""
        return """
Imagination Rollout (Training Actor-Critic):

Input: Initial state s_0 = (h_0, z_0) from real experience
Horizon: H steps into the future

for tau in range(H):
    # Use prior (no observations in imagination)
    z_tau ~ p(z | h_tau)  # Sample from prior

    # Predict quantities
    r_tau = reward_model(h_tau, z_tau)
    c_tau = continue_model(h_tau, z_tau)  # Episode continuation prob

    # Select action using current policy
    a_tau ~ pi(a | h_tau, z_tau)

    # Transition to next state
    h_{tau+1} = GRU(h_tau, [z_tau, a_tau])

# Compute lambda-returns for value learning
V_lambda[tau] = r_tau + gamma * c_tau * ((1-lambda) * v(s_{tau+1}) + lambda * V_lambda[tau+1])

# Actor loss: maximize imagined returns
L_actor = -mean(V_lambda)

# Critic loss: predict returns
L_critic = mean((v(s_tau) - sg(V_lambda[tau]))^2)
"""

    @staticmethod
    def world_model_training() -> str:
        """World model training procedure."""
        return """
World Model Training:

Data: Buffer of (o_t, a_t, r_t, done_t) from environment

for batch in buffer.sample_sequences(batch_size, seq_len):
    # Initialize hidden state
    h = zeros(batch_size, deterministic_dim)

    # Process sequence
    losses = []
    for t in range(seq_len):
        # Encode observation
        e_t = encoder(o_t)

        # Compute posterior and prior
        z_post = sample(posterior(h, e_t))
        z_prior_logits = prior(h)

        # Predictions
        o_pred = decoder(h, z_post)
        r_pred = reward_model(h, z_post)
        c_pred = continue_model(h, z_post)

        # Losses
        L_recon = -log_prob(o_t | o_pred)
        L_reward = -log_prob(r_t | r_pred)
        L_continue = -log_prob(not done_t | c_pred)
        L_kl = KL(posterior || prior)

        losses.append(L_recon + L_reward + L_continue + beta * L_kl)

        # Transition
        h = GRU(h, [z_post, a_t])

    total_loss = mean(losses)
    total_loss.backward()
    optimizer.step()
"""


# =============================================================================
# Mathematical Functions (Reference)
# =============================================================================

def symlog(x):
    """
    Symlog encoding for scale-invariant predictions (DreamerV3).

    symlog(x) = sign(x) * ln(|x| + 1)

    Properties:
        - Compresses large magnitudes
        - Preserves sign
        - symlog(0) = 0
        - Approximately linear near 0
        - Bounded gradient: |d symlog/dx| <= 1

    Args:
        x: Input value or tensor

    Returns:
        Symlog-encoded value
    """
    return {
        "formula": "symlog(x) = sign(x) * ln(|x| + 1)",
        "inverse": "symexp(x) = sign(x) * (exp(|x|) - 1)",
        "gradient": "d/dx symlog(x) = 1 / (|x| + 1)",
        "use_case": "Predicting rewards/values with varying scales"
    }


def two_hot_encode(x, bins, low, high):
    """
    Two-hot encoding for continuous regression (DreamerV3).

    Distributes probability mass across two adjacent bins
    based on where the value falls.

    Args:
        x: Continuous value to encode
        bins: Number of discrete bins
        low: Minimum value
        high: Maximum value

    Returns:
        Distribution over bins
    """
    return {
        "formula": """
            bin_size = (high - low) / (bins - 1)
            index = (x - low) / bin_size
            lower_idx = floor(index)
            upper_idx = ceil(index)
            upper_weight = index - lower_idx
            lower_weight = 1 - upper_weight
            two_hot[lower_idx] = lower_weight
            two_hot[upper_idx] = upper_weight
        """,
        "decoding": "x = sum(bins * softmax(logits) * bin_values)",
        "advantage": "Enables distributional regression with multimodal support"
    }


def kl_balancing(posterior_logits, prior_logits, alpha=0.8):
    """
    KL balancing for training posterior and prior (DreamerV2/V3).

    Balances two objectives:
    1. Train posterior to match prior (regularization)
    2. Train prior to match posterior (prediction)

    Args:
        posterior_logits: Logits from q(z|h,o)
        prior_logits: Logits from p(z|h)
        alpha: Balance coefficient (0.8 = prioritize posterior regularization)

    Returns:
        Balanced KL loss
    """
    return {
        "formula": """
            L_kl = alpha * KL(sg(posterior) || prior)
                 + (1 - alpha) * KL(posterior || sg(prior))
        """,
        "alpha_0.8": "Prioritizes training posterior to be close to prior",
        "intuition": """
            - First term: trains prior to predict posterior
            - Second term: regularizes posterior toward prior
            - sg() = stop_gradient prevents gradients to one term
        """,
        "free_bits": "L_kl = max(free_nats, L_kl) prevents posterior collapse"
    }


# =============================================================================
# Dreamer Evolution and Insights
# =============================================================================

DREAMER_EVOLUTION = {
    "v1_to_v2": {
        "latent_space": "Continuous Gaussian -> Discrete categorical",
        "reason": "Discrete enables multimodal states, better for diverse games",
        "kl_handling": "Standard KL -> KL balancing with stop-gradient",
        "domains": "DMC continuous control -> Also Atari discrete control"
    },

    "v2_to_v3": {
        "normalization": "None -> Symlog predictions + percentile returns",
        "reason": "Handle vastly different reward/observation scales",
        "regression": "MSE -> Two-hot distributional",
        "reason2": "Better represent uncertainty in value predictions",
        "hyperparameters": "Domain-specific -> Single fixed hyperparameters",
        "domains": "Atari/DMC -> Also Minecraft, DMLab, BSuite, Crafter"
    }
}


DREAMER_INSIGHTS = {
    "rssm_design": """
        The RSSM combines deterministic and stochastic components:
        - Deterministic h: Captures history, enables precise prediction
        - Stochastic z: Captures uncertainty, enables exploration

        Key insight: Using ONLY deterministic would be overconfident,
        using ONLY stochastic would lose temporal coherence.
    """,

    "imagination_horizon": """
        Typical imagination horizon: 15 steps
        - Too short: Can't see long-term consequences
        - Too long: Compounding errors degrade predictions
        - Lambda-returns blend short and long-term estimates
    """,

    "world_model_as_simulator": """
        The world model serves as a learned simulator:
        - Train on real experience (few samples)
        - Generate unlimited synthetic experience
        - Policy learns from synthetic data
        - Much more sample-efficient than model-free
    """,

    "discrete_vs_continuous": """
        Discrete latents (V2/V3) vs Continuous (V1):
        - Discrete: Natural for games with distinct states
        - Discrete: Can represent multimodal distributions
        - Continuous: Smoother interpolation
        - Continuous: Natural for continuous control

        V2/V3 show discrete works well even for continuous domains.
    """
}
