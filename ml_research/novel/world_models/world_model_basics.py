"""
World Models - Ha & Schmidhuber (2018)

Foundational work on learning world models in latent space for
model-based reinforcement learning. The architecture separates
visual perception, memory/prediction, and control.

Paper: "World Models"
arXiv: 1803.10122
Interactive: https://worldmodels.github.io/

Mathematical Formulation:
    Vision (V) - VAE:
        z_t = V(o_t)  # Compress observation to latent
        z ~ q(z|o) = N(mu(o), sigma^2(o))

    Memory (M) - MDN-RNN:
        h_t = M(h_{t-1}, z_{t-1}, a_{t-1})  # Update hidden state
        P(z_t|h_t) = MDN(h_t)  # Predict next latent as mixture of Gaussians

    Controller (C):
        a_t = C(z_t, h_t)  # Simple linear controller
        a_t = W_c [z_t, h_t] + b_c

Key Innovation:
    Train controller entirely in "dream" (latent space) using
    the learned world model for imagination-based planning.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

WORLD_MODEL = MLMethod(
    method_id="world_models_2018",
    name="World Models",
    year=2018,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RNN_LINE, MethodLineage.GENERATIVE_LINE],  # VAE + RNN

    authors=["David Ha", "Jurgen Schmidhuber"],
    paper_title="World Models",
    paper_url="https://arxiv.org/abs/1803.10122",

    key_innovation=(
        "Introduced a modular architecture (Vision-Memory-Controller) for learning "
        "world models in latent space. Demonstrated that agents can learn behaviors "
        "entirely inside a 'dream' - training controllers using imagined rollouts "
        "from a learned dynamics model without interacting with the real environment."
    ),

    mathematical_formulation=r"""
Vision Model V (VAE):
    Encoder: z ~ q_phi(z|o) = N(mu_phi(o), sigma_phi^2(o))
    Decoder: o ~ p_theta(o|z)
    Loss: L_VAE = E[log p(o|z)] - KL(q(z|o) || p(z))

Memory Model M (MDN-RNN):
    Hidden state update:
        h_t = tanh(W_h h_{t-1} + W_z z_{t-1} + W_a a_{t-1} + b)

    Mixture Density Network output:
        P(z_t | h_t) = sum_i pi_i N(z_t; mu_i(h_t), sigma_i(h_t))

    Where: pi_i, mu_i, sigma_i are outputs of linear layers from h_t

Controller C:
    a_t = W_c [z_t; h_t] + b_c

    Or with tanh: a_t = tanh(W_c [z_t; h_t] + b_c)

Full Forward Pass:
    1. z_t = V_encode(o_t)
    2. h_t = M(h_{t-1}, z_{t-1}, a_{t-1})
    3. a_t = C(z_t, h_t)
    4. Execute a_t in environment
""",

    predecessors=["vae_2013", "lstm_1997", "mdn_1994"],
    successors=["dreamer_v1_2019", "planet_2018", "muzero_2019"],

    tags=[
        "world-model", "model-based-rl", "imagination", "latent-space",
        "vae", "mdn-rnn", "controller", "dream-learning"
    ],
    notes=(
        "Key insight: the world model can be trained unsupervised on random "
        "exploration data, then the controller trained entirely in imagination. "
        "Demonstrated on CarRacing-v0 and VizDoom. The MDN-RNN learns to predict "
        "stochastic futures, enabling robust policy learning."
    )
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for World Models."""
    return WORLD_MODEL


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class VMCArchitecture:
    """Reference architecture for World Models (Vision-Memory-Controller)."""

    # VAE parameters
    observation_shape: tuple = (64, 64, 3)
    latent_dim: int = 32

    # MDN-RNN parameters
    hidden_dim: int = 256
    num_mixtures: int = 5
    action_dim: int = 3

    # Controller parameters
    controller_hidden: int = 0  # Linear controller (no hidden layer)

    @staticmethod
    def vision_model_structure() -> str:
        """VAE encoder-decoder structure."""
        return """
Vision Model V (Convolutional VAE):

Encoder:
    Input: o_t [batch, 64, 64, 3]
    Conv2d(3->32, 4x4, stride=2)  -> [batch, 32, 32, 32]
    Conv2d(32->64, 4x4, stride=2) -> [batch, 16, 16, 64]
    Conv2d(64->128, 4x4, stride=2) -> [batch, 8, 8, 128]
    Conv2d(128->256, 4x4, stride=2) -> [batch, 4, 4, 256]
    Flatten -> [batch, 4096]
    Linear(4096 -> 2*latent_dim) -> mu, log_var

    z = mu + sigma * epsilon  (reparameterization)

Decoder:
    Linear(latent_dim -> 4096)
    Reshape -> [batch, 4, 4, 256]
    ConvTranspose2d(256->128, 4x4, stride=2)
    ConvTranspose2d(128->64, 4x4, stride=2)
    ConvTranspose2d(64->32, 4x4, stride=2)
    ConvTranspose2d(32->3, 4x4, stride=2)
    Sigmoid -> [batch, 64, 64, 3]
"""

    @staticmethod
    def memory_model_structure() -> str:
        """MDN-RNN structure."""
        return """
Memory Model M (MDN-RNN):

LSTM Core:
    Input: [z_{t-1}, a_{t-1}] concatenated
    x_t = [z_{t-1}; a_{t-1}]  [batch, latent_dim + action_dim]

    # LSTM update equations
    f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)  # Forget gate
    i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)  # Input gate
    o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)  # Output gate
    c_t = f_t * c_{t-1} + i_t * tanh(W_c x_t + U_c h_{t-1} + b_c)
    h_t = o_t * tanh(c_t)

MDN Output Layer:
    For K mixtures and D latent dimensions:

    Linear(hidden_dim -> K)          -> pi (mixture weights)
    Linear(hidden_dim -> K * D)      -> mu (means)
    Linear(hidden_dim -> K * D)      -> log_sigma (log std devs)

    pi = softmax(pi_logits)
    sigma = exp(log_sigma)

    P(z_t | h_t) = sum_{k=1}^K pi_k * N(z_t; mu_k, diag(sigma_k^2))
"""

    @staticmethod
    def controller_structure() -> str:
        """Controller structure."""
        return """
Controller C:

Simple Linear Controller:
    Input: [z_t, h_t] concatenated
    x = [z_t; h_t]  [batch, latent_dim + hidden_dim]

    a_t = W_c x + b_c  [batch, action_dim]
    a_t = tanh(a_t)  # Bound actions to [-1, 1]

    Total parameters: (latent_dim + hidden_dim) * action_dim + action_dim
    For defaults: (32 + 256) * 3 + 3 = 867 parameters

    Note: The controller is intentionally simple because the
    heavy lifting is done by V and M. This shows that a good
    world model enables simple controllers to succeed.
"""

    @staticmethod
    def training_procedure() -> str:
        """Training procedure for World Models."""
        return """
Training Procedure:

Phase 1: Collect Random Rollouts
    for episode in range(num_episodes):
        while not done:
            a = random_action()
            o, r, done = env.step(a)
            buffer.store(o, a, r)

Phase 2: Train Vision Model V
    for batch in shuffle(buffer.observations):
        z, mu, log_var = V.encode(batch)
        o_recon = V.decode(z)
        loss = reconstruction_loss(o_recon, batch) + KL_loss(mu, log_var)
        loss.backward()
        optimizer_V.step()

Phase 3: Train Memory Model M
    # First encode all observations
    z_data = [V.encode(o) for o in buffer.observations]

    for (z_seq, a_seq) in sequences(z_data, buffer.actions):
        h = M.initial_hidden()
        loss = 0
        for t in range(seq_len):
            h = M.forward(h, z_seq[t], a_seq[t])
            pi, mu, sigma = M.mdn_output(h)
            loss += -log_prob_mdn(z_seq[t+1], pi, mu, sigma)
        loss.backward()
        optimizer_M.step()

Phase 4: Train Controller C (in dream)
    # Using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    population_size = 64
    weights = flatten(C.parameters())

    for generation in range(num_generations):
        # Sample weight perturbations
        candidates = [weights + sigma * N(0, I) for _ in range(population_size)]

        # Evaluate in dream
        rewards = []
        for candidate in candidates:
            C.load_weights(unflatten(candidate))
            dream_reward = rollout_in_dream(V, M, C)
            rewards.append(dream_reward)

        # CMA-ES update
        weights = cma_es_update(candidates, rewards)

    Note: Controller training uses NO environment interaction!
"""


# =============================================================================
# Mathematical Functions (Reference)
# =============================================================================

def mdn_loss(z_target, pi, mu, sigma):
    """
    Compute MDN negative log-likelihood loss.

    P(z | pi, mu, sigma) = sum_k pi_k * N(z; mu_k, sigma_k^2)

    Loss = -log P(z_target | pi, mu, sigma)

    Args:
        z_target: Target latent vector [batch, latent_dim]
        pi: Mixture weights [batch, num_mixtures]
        mu: Mixture means [batch, num_mixtures, latent_dim]
        sigma: Mixture std devs [batch, num_mixtures, latent_dim]

    Returns:
        Negative log-likelihood loss
    """
    return {
        "formula": "L = -log(sum_k pi_k * N(z; mu_k, sigma_k))",
        "gaussian": "N(z; mu, sigma) = (1/sqrt(2*pi*sigma^2)) * exp(-(z-mu)^2 / (2*sigma^2))",
        "log_sum_exp": "Use log-sum-exp trick for numerical stability",
        "note": "Sum over mixture components, product over latent dimensions"
    }


def temperature_sampling(pi, mu, sigma, temperature=1.0):
    """
    Sample from MDN with temperature control.

    Higher temperature -> more stochastic/diverse samples
    Lower temperature -> more deterministic (mode-seeking)

    At temperature=0: return mean of highest-weight mixture

    Args:
        pi: Mixture weights [batch, num_mixtures]
        mu: Mixture means [batch, num_mixtures, latent_dim]
        sigma: Mixture std devs [batch, num_mixtures, latent_dim]
        temperature: Sampling temperature (default 1.0)

    Returns:
        Sampled latent vector
    """
    return {
        "formula": "sigma_adjusted = sigma * sqrt(temperature)",
        "mixture_selection": "k ~ Categorical(pi) or pi_adjusted = softmax(log(pi) / temperature)",
        "sampling": "z = mu_k + sigma_k * sqrt(temperature) * epsilon, epsilon ~ N(0, I)",
        "deterministic": "If temperature=0: z = mu[argmax(pi)]"
    }


# =============================================================================
# Key Insights and Variants
# =============================================================================

WORLD_MODEL_INSIGHTS = {
    "latent_space_learning": """
        Learning in latent space is more efficient than pixel space:
        - Dimensionality reduction: 64x64x3 = 12288 -> 32 latent dims
        - Captures relevant structure, discards noise
        - Enables fast imagination rollouts
    """,

    "dream_training": """
        Controller training in dreams has advantages:
        - No environment interaction needed during policy learning
        - Can simulate many more episodes than real-world allows
        - Learned model captures environment stochasticity
        - Risk-free exploration of dangerous states
    """,

    "mdn_for_stochasticity": """
        MDN (Mixture Density Network) captures multimodal futures:
        - Environment transitions can be stochastic
        - Multiple valid next states from same (s, a)
        - Single Gaussian would average modes (blurry)
        - Mixture models distinct possibilities
    """,

    "separation_of_concerns": """
        V-M-C architecture cleanly separates:
        - V: Perception (what do I see?)
        - M: Prediction (what will happen?)
        - C: Decision (what should I do?)
        This modularity enables independent training and analysis.
    """
}


WORLD_MODEL_VARIANTS = {
    "planet_2018": {
        "description": "PlaNet: Learning Latent Dynamics for Planning",
        "authors": "Hafner et al.",
        "innovation": "Model-predictive control with learned latent dynamics",
        "difference": "Uses cross-entropy method for planning instead of learned controller"
    },
    "dreamer_2019": {
        "description": "Dream to Control: Learning Behaviors by Latent Imagination",
        "authors": "Hafner et al.",
        "innovation": "Actor-critic learning entirely in imagination",
        "difference": "Replaces CMA-ES with learned value function and policy"
    },
    "simple_2019": {
        "description": "Simulated Policy Learning",
        "innovation": "Improved training stability and performance",
        "difference": "Better VAE training, temperature annealing"
    }
}
