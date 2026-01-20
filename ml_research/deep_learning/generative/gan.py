"""
Generative Adversarial Network (GAN) - 2014

Goodfellow et al.'s groundbreaking framework for training generative models
through adversarial training. Two networks (generator and discriminator)
compete in a minimax game.

Paper: "Generative Adversarial Nets" (NIPS 2014)
arXiv: 1406.2661

Mathematical Formulation:
    Minimax Objective:
        min_G max_D V(D, G) = E_x~p_data[log D(x)] + E_z~p_z[log(1 - D(G(z)))]

    Where:
        - G: Generator network (maps noise z to data space)
        - D: Discriminator network (classifies real vs fake)
        - p_data: Real data distribution
        - p_z: Prior noise distribution (typically N(0, I) or Uniform)

    At Nash equilibrium:
        - D(x) = 0.5 for all x
        - p_g = p_data (generator matches data distribution)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

GENERATIVE_ADVERSARIAL_NETWORK = MLMethod(
    method_id="gan_2014",
    name="Generative Adversarial Network",
    year=2014,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.GENERATIVE,
    lineages=[MethodLineage.GENERATIVE_LINE],

    authors=["Ian J. Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza",
             "Bing Xu", "David Warde-Farley", "Sherjil Ozair",
             "Aaron Courville", "Yoshua Bengio"],
    paper_title="Generative Adversarial Nets",
    paper_url="https://arxiv.org/abs/1406.2661",

    key_innovation=(
        "Introduced adversarial training framework where a generator and discriminator "
        "compete in a two-player minimax game. Avoids explicit density estimation and "
        "Markov chains, enabling generation of sharp, realistic samples."
    ),

    mathematical_formulation=r"""
Minimax Game Objective:
    min_G max_D V(D, G) = E_x~p_data[log D(x)] + E_z~p_z[log(1 - D(G(z)))]

Discriminator Objective (maximize):
    L_D = E_x~p_data[log D(x)] + E_z~p_z[log(1 - D(G(z)))]
    = (1/m) * sum_i [log D(x_i) + log(1 - D(G(z_i)))]

Generator Objective (minimize):
    L_G = E_z~p_z[log(1 - D(G(z)))]

    In practice, maximize log D(G(z)) instead (non-saturating loss):
    L_G = -E_z~p_z[log D(G(z))]

Optimal Discriminator (for fixed G):
    D*_G(x) = p_data(x) / (p_data(x) + p_g(x))

Global Optimum:
    p_g = p_data
    D*(x) = 1/2 for all x
    V(G*, D*) = -log(4)
""",

    predecessors=["boltzmann_machine_1985", "autoencoder"],
    successors=["dcgan_2015", "wgan_2017", "stylegan_2018"],

    tags=["generative", "adversarial", "minimax", "game-theory", "deep-learning"]
)


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def discriminator_loss(real_output, fake_output):
    """
    Compute discriminator loss for GAN training.

    L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
        = BCE(D(x), 1) + BCE(D(G(z)), 0)

    The discriminator tries to:
    - Output 1 (real) for real data x
    - Output 0 (fake) for generated data G(z)

    Args:
        real_output: D(x) - discriminator output for real data
        fake_output: D(G(z)) - discriminator output for fake data

    Returns:
        Total discriminator loss

    Note:
        Minimize this loss = Maximize log D(x) + log(1 - D(G(z)))
    """
    return {
        "formula": "L_D = -[log(D(x)) + log(1 - D(G(z)))]",
        "real_loss": "BCE(D(x), ones) = -log(D(x))",
        "fake_loss": "BCE(D(G(z)), zeros) = -log(1 - D(G(z)))",
        "total": "real_loss + fake_loss"
    }


def generator_loss(fake_output, loss_type="non_saturating"):
    """
    Compute generator loss for GAN training.

    Original (saturating) loss:
        L_G = E[log(1 - D(G(z)))]
        Problem: Gradient vanishes when D(G(z)) -> 0

    Non-saturating loss (recommended):
        L_G = -E[log D(G(z))]
        Provides stronger gradients early in training

    Args:
        fake_output: D(G(z)) - discriminator output for generated data
        loss_type: "saturating" or "non_saturating"

    Returns:
        Generator loss

    Note:
        Generator wants D(G(z)) -> 1 (fool discriminator)
    """
    return {
        "saturating": "L_G = log(1 - D(G(z)))",
        "non_saturating": "L_G = -log(D(G(z)))",
        "heuristic": "Use non-saturating loss in practice for better gradients",
        "gradient_analysis": {
            "saturating_early": "When D(G(z)) ≈ 0: gradient ≈ 0 (vanishes)",
            "non_saturating_early": "When D(G(z)) ≈ 0: gradient ≈ -1/epsilon (strong)"
        }
    }


def minimax_objective():
    """
    The complete minimax objective for GAN training.

    V(D, G) = E_x~p_data[log D(x)] + E_z~p_z[log(1 - D(G(z)))]

    Training dynamics:
    1. Sample minibatch of m noise samples {z_1, ..., z_m} from p_z
    2. Sample minibatch of m real examples {x_1, ..., x_m} from p_data
    3. Update discriminator by ascending gradient:
       nabla_theta_d (1/m) sum_i [log D(x_i) + log(1 - D(G(z_i)))]
    4. Update generator by descending gradient:
       nabla_theta_g (1/m) sum_i log(1 - D(G(z_i)))

    Returns:
        Description of minimax objective
    """
    return {
        "objective": "min_G max_D V(D,G)",
        "value_function": "V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]",
        "interpretation": {
            "discriminator": "Maximize probability of correct classification",
            "generator": "Minimize probability of discriminator being correct"
        },
        "theoretical_optimum": {
            "condition": "p_g = p_data",
            "discriminator_output": "D*(x) = 0.5 for all x",
            "value": "V* = -log(4)"
        }
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class GANArchitecture:
    """Reference architecture for vanilla GAN."""

    noise_dim: int = 100
    image_dim: int = 784  # e.g., 28x28 flattened
    hidden_dim: int = 256

    @staticmethod
    def generator_structure() -> str:
        """Generator network structure."""
        return """
Generator G(z):
    Input: z [batch_size, noise_dim] ~ N(0, I) or Uniform[-1, 1]

    h = LeakyReLU(Linear(noise_dim -> hidden_dim))
    h = LeakyReLU(Linear(hidden_dim -> hidden_dim))
    h = LeakyReLU(Linear(hidden_dim -> hidden_dim))

    output = Tanh(Linear(hidden_dim -> image_dim))
    # or Sigmoid for [0, 1] output

    Output: fake_image [batch_size, image_dim]
"""

    @staticmethod
    def discriminator_structure() -> str:
        """Discriminator network structure."""
        return """
Discriminator D(x):
    Input: x [batch_size, image_dim]

    h = LeakyReLU(Linear(image_dim -> hidden_dim))
    h = Dropout(LeakyReLU(Linear(hidden_dim -> hidden_dim)))
    h = Dropout(LeakyReLU(Linear(hidden_dim -> hidden_dim)))

    output = Sigmoid(Linear(hidden_dim -> 1))

    Output: probability [batch_size, 1] in [0, 1]
"""

    @staticmethod
    def training_procedure() -> str:
        """Training procedure pseudocode."""
        return """
GAN Training Loop:
    for iteration in range(num_iterations):
        # ===== Train Discriminator =====
        for k in range(d_steps):  # Often k=1
            # Sample real data
            x_real = sample_data(batch_size)

            # Sample noise and generate fake data
            z = sample_noise(batch_size, noise_dim)
            x_fake = G(z).detach()  # Detach to not update G

            # Discriminator predictions
            d_real = D(x_real)
            d_fake = D(x_fake)

            # Discriminator loss
            d_loss = -mean(log(d_real) + log(1 - d_fake))

            # Update discriminator
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # ===== Train Generator =====
        # Sample noise
        z = sample_noise(batch_size, noise_dim)

        # Generate fake data
        x_fake = G(z)

        # Discriminator prediction on fake
        d_fake = D(x_fake)

        # Generator loss (non-saturating)
        g_loss = -mean(log(d_fake))

        # Update generator
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
"""


# =============================================================================
# Training Challenges Reference
# =============================================================================

GAN_CHALLENGES = {
    "mode_collapse": {
        "description": "Generator produces limited variety of samples",
        "cause": "G finds few modes that fool D and exploits them",
        "solutions": ["Minibatch discrimination", "Unrolled GANs", "Feature matching"]
    },
    "training_instability": {
        "description": "Loss oscillates, training diverges",
        "cause": "Minimax optimization is inherently unstable",
        "solutions": ["Two-timescale update rule", "Spectral normalization", "Gradient penalty"]
    },
    "vanishing_gradients": {
        "description": "Generator receives no useful gradient signal",
        "cause": "Discriminator becomes too strong",
        "solutions": ["Non-saturating loss", "Wasserstein loss", "Label smoothing"]
    },
    "evaluation_difficulty": {
        "description": "No single metric captures sample quality and diversity",
        "metrics": ["Inception Score (IS)", "Frechet Inception Distance (FID)",
                   "Precision/Recall", "Kernel Inception Distance (KID)"]
    }
}


# =============================================================================
# Theoretical Analysis
# =============================================================================

THEORETICAL_RESULTS = {
    "optimal_discriminator": {
        "theorem": "For fixed G, the optimal discriminator is D*_G(x) = p_data(x) / (p_data(x) + p_g(x))",
        "proof_sketch": "Take derivative of V(D,G) w.r.t. D(x), set to zero"
    },
    "global_optimum": {
        "theorem": "The global minimum of C(G) = max_D V(D,G) is achieved iff p_g = p_data",
        "value": "C(G*) = -log(4)",
        "divergence": "C(G) = -log(4) + 2 * JSD(p_data || p_g)",
        "note": "JSD = Jensen-Shannon Divergence"
    },
    "convergence": {
        "theorem": "If G and D have enough capacity and D is optimized to convergence at each step, p_g converges to p_data",
        "caveat": "In practice, G and D are updated simultaneously, so theoretical guarantees don't directly apply"
    }
}
