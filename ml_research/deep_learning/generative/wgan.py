"""
Wasserstein GAN (WGAN) - 2017

Arjovsky, Chintala, and Bottou's theoretical and practical contribution
to GAN training stability using the Earth Mover (Wasserstein-1) distance
instead of Jensen-Shannon divergence.

Paper: "Wasserstein GAN" (ICML 2017)
arXiv: 1701.07875

Follow-up Paper: "Improved Training of Wasserstein GANs" (NIPS 2017)
arXiv: 1704.00028 (Gradient Penalty variant - WGAN-GP)

Mathematical Formulation:
    Wasserstein Distance (Earth Mover Distance):
        W(p_data, p_g) = inf_{gamma in Pi(p_data, p_g)} E_(x,y)~gamma[||x - y||]

    Kantorovich-Rubinstein Duality:
        W(p_data, p_g) = sup_{||f||_L <= 1} E_x~p_data[f(x)] - E_x~p_g[f(x)]

    WGAN Objective:
        L = E_x~p_data[D(x)] - E_z~p_z[D(G(z))]

    Critic (not discriminator) must be 1-Lipschitz:
        Original: Weight clipping to [-c, c]
        Improved: Gradient penalty (WGAN-GP)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

WASSERSTEIN_GAN = MLMethod(
    method_id="wgan_2017",
    name="Wasserstein GAN",
    year=2017,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.GENERATIVE,
    lineages=[MethodLineage.GENERATIVE_LINE],

    authors=["Martin Arjovsky", "Soumith Chintala", "Leon Bottou"],
    paper_title="Wasserstein GAN",
    paper_url="https://arxiv.org/abs/1701.07875",

    key_innovation=(
        "Replaced JS divergence with Wasserstein (Earth Mover) distance, providing "
        "meaningful loss that correlates with sample quality. Eliminates mode collapse "
        "and provides stable training gradients even when distributions don't overlap. "
        "Critic (not discriminator) trained to approximate Wasserstein distance."
    ),

    mathematical_formulation=r"""
Wasserstein-1 Distance (Earth Mover Distance):
    W_1(P_r, P_g) = inf_{gamma in Pi(P_r, P_g)} E_(x,y)~gamma[||x - y||]

    Interpretation: Minimum cost to transform P_r into P_g, where cost is distance moved.

Kantorovich-Rubinstein Duality:
    W_1(P_r, P_g) = sup_{||f||_L <= 1} E_x~P_r[f(x)] - E_x~P_g[f(x)]

    Where ||f||_L <= 1 means f is 1-Lipschitz: |f(x) - f(y)| <= |x - y|

WGAN Objective:
    Critic (D, no longer discriminator):
        L_critic = E_x~p_data[D(x)] - E_z~p_z[D(G(z))]
        Maximize this to estimate W_1 distance

    Generator:
        L_generator = -E_z~p_z[D(G(z))]
        Minimize this to reduce W_1 distance

Lipschitz Constraint Enforcement:

    1. Weight Clipping (Original WGAN):
        After each update: w = clip(w, -c, c), typically c = 0.01
        Problem: Limits capacity, can lead to vanishing/exploding gradients

    2. Gradient Penalty (WGAN-GP, recommended):
        L = L_critic + lambda * E_x_hat[(||nabla_x_hat D(x_hat)||_2 - 1)^2]

        Where x_hat = epsilon * x_real + (1 - epsilon) * x_fake
        and epsilon ~ Uniform[0, 1]

        lambda = 10 (typically)

Why Wasserstein Distance Works Better:

    1. JS Divergence Problem:
        When P_r and P_g have disjoint supports: JS(P_r, P_g) = log(2) (constant!)
        Gradient = 0, no learning signal

    2. Wasserstein Advantage:
        W_1 provides meaningful distance even for disjoint distributions
        W_1 = 0 iff P_r = P_g
        Continuous in distribution parameters (weak topology)
""",

    predecessors=["gan_2014", "dcgan_2015"],
    successors=["wgan_gp_2017", "stylegan_2018", "progressive_gan"],

    tags=["generative", "wasserstein", "optimal-transport", "adversarial", "deep-learning"]
)


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def wasserstein_distance():
    """
    Wasserstein-1 (Earth Mover) Distance definition and properties.

    W_1(P_r, P_g) = inf_{gamma} E_(x,y)~gamma[||x - y||]

    Intuition:
        - Think of P_r as pile of dirt, P_g as hole to fill
        - W_1 = minimum cost to move dirt to fill hole
        - Cost = amount of dirt * distance moved

    Properties:
        1. W_1 >= 0
        2. W_1 = 0 iff P_r = P_g
        3. Symmetric: W_1(P, Q) = W_1(Q, P)
        4. Triangle inequality: W_1(P, R) <= W_1(P, Q) + W_1(Q, R)
        5. Metrizes weak convergence (continuous in distribution parameters)

    Returns:
        Description of Wasserstein distance
    """
    return {
        "primal_form": "W_1(P_r, P_g) = inf_{gamma in Pi(P_r, P_g)} E_(x,y)~gamma[||x - y||]",
        "dual_form": "W_1(P_r, P_g) = sup_{||f||_L <= 1} E_x~P_r[f(x)] - E_x~P_g[f(x)]",
        "computation": "Dual form used in practice; critic approximates f",
        "comparison_to_js": {
            "js_problem": "JS divergence constant when supports don't overlap",
            "w1_advantage": "W_1 provides continuous, meaningful gradient everywhere"
        }
    }


def critic_loss(real_output, fake_output):
    """
    Compute critic loss for WGAN training.

    L_critic = -E[D(x_real)] + E[D(G(z))]
             = -mean(D(x_real)) + mean(D(x_fake))

    Note: This is negated because we MAXIMIZE critic output for real
    and MINIMIZE for fake, but optimizers minimize, so we negate.

    The critic output is unbounded (no sigmoid) and represents
    the Wasserstein distance estimate.

    Args:
        real_output: D(x_real) - critic output for real data (no sigmoid)
        fake_output: D(G(z)) - critic output for generated data

    Returns:
        Critic loss (to be minimized, which maximizes Wasserstein estimate)
    """
    return {
        "formula": "L_critic = -mean(D(x_real)) + mean(D(x_fake))",
        "equivalent": "L_critic = mean(D(x_fake)) - mean(D(x_real))",
        "goal": "Minimize L_critic = Maximize [E[D(real)] - E[D(fake)]]",
        "output_range": "Unbounded (no sigmoid); represents W_1 estimate"
    }


def gradient_penalty(critic, real_samples, fake_samples, lambda_gp=10):
    """
    Compute gradient penalty for WGAN-GP.

    GP = lambda * E_x_hat[(||grad_x_hat D(x_hat)||_2 - 1)^2]

    Where x_hat = epsilon * x_real + (1 - epsilon) * x_fake
    samples points along straight lines between real and fake distributions.

    This enforces the 1-Lipschitz constraint softly by penalizing
    gradients that deviate from norm 1.

    Args:
        critic: The critic network
        real_samples: Batch of real data
        fake_samples: Batch of generated data
        lambda_gp: Gradient penalty coefficient (typically 10)

    Returns:
        Gradient penalty term

    Why Along Interpolations?
        - Theory requires Lipschitz everywhere
        - In practice, only need Lipschitz where critic is evaluated
        - Interpolations sample the region between p_data and p_g
    """
    return {
        "formula": "GP = lambda * E_x_hat[(||grad D(x_hat)||_2 - 1)^2]",
        "interpolation": "x_hat = epsilon * x_real + (1 - epsilon) * x_fake",
        "epsilon": "epsilon ~ Uniform[0, 1], sampled per example",
        "gradient_computation": "grad = autograd.grad(D(x_hat), x_hat, create_graph=True)",
        "norm": "||grad||_2 computed per example, then averaged",
        "lambda_gp": "10 (from WGAN-GP paper)",
        "total_critic_loss": "L = -E[D(real)] + E[D(fake)] + lambda * GP"
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class WGANArchitecture:
    """Reference architecture for WGAN/WGAN-GP."""

    image_size: int = 64
    num_channels: int = 3
    latent_dim: int = 100
    n_critic: int = 5  # Critic updates per generator update

    @staticmethod
    def critic_structure() -> str:
        """WGAN Critic (not discriminator) architecture."""
        return """
WGAN Critic (similar to DCGAN discriminator, key differences noted):
    Input: x [batch_size, nc, 64, 64]

    # Note: No Sigmoid at output!
    # Note: For WGAN-GP, no BatchNorm (use LayerNorm or no norm instead)

    Conv2d(nc, ndf, 4, 2, 1)               # [ndf, 32, 32]
    LeakyReLU(0.2)

    Conv2d(ndf, ndf*2, 4, 2, 1)            # [ndf*2, 16, 16]
    LayerNorm or InstanceNorm (not BatchNorm for WGAN-GP)
    LeakyReLU(0.2)

    Conv2d(ndf*2, ndf*4, 4, 2, 1)          # [ndf*4, 8, 8]
    LayerNorm or InstanceNorm
    LeakyReLU(0.2)

    Conv2d(ndf*4, ndf*8, 4, 2, 1)          # [ndf*8, 4, 4]
    LayerNorm or InstanceNorm
    LeakyReLU(0.2)

    Conv2d(ndf*8, 1, 4, 1, 0)              # [1, 1, 1]
    # NO Sigmoid! Output is unbounded "score"

    Output: critic_score [batch_size, 1, 1, 1] (unbounded)

Key Differences from DCGAN Discriminator:
    1. No Sigmoid activation (output is unbounded)
    2. For WGAN-GP: No BatchNorm (conflicts with gradient penalty)
    3. Called "critic" not "discriminator" (different interpretation)
"""

    @staticmethod
    def generator_structure() -> str:
        """Generator architecture (same as DCGAN)."""
        return """
WGAN Generator (same as DCGAN):
    Input: z [batch_size, 100, 1, 1]

    ConvTranspose2d(100, ngf*8, 4, 1, 0)
    BatchNorm2d(ngf*8)
    ReLU

    ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1)
    BatchNorm2d(ngf*4)
    ReLU

    ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1)
    BatchNorm2d(ngf*2)
    ReLU

    ConvTranspose2d(ngf*2, ngf, 4, 2, 1)
    BatchNorm2d(ngf)
    ReLU

    ConvTranspose2d(ngf, nc, 4, 2, 1)
    Tanh

    Output: image [batch_size, 3, 64, 64]
"""

    @staticmethod
    def training_procedure_weight_clip() -> str:
        """Training with weight clipping (original WGAN)."""
        return """
WGAN Training (Weight Clipping):
    n_critic = 5  # Train critic more often than generator
    clip_value = 0.01

    for iteration in range(num_iterations):
        # ===== Train Critic =====
        for _ in range(n_critic):
            x_real = sample_data(batch_size)
            z = sample_noise(batch_size, latent_dim)
            x_fake = G(z).detach()

            # Critic loss (maximize E[D(real)] - E[D(fake)])
            critic_loss = -mean(D(x_real)) + mean(D(x_fake))

            d_optimizer.zero_grad()
            critic_loss.backward()
            d_optimizer.step()

            # Weight clipping
            for p in D.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # ===== Train Generator =====
        z = sample_noise(batch_size, latent_dim)
        x_fake = G(z)

        # Generator loss (maximize E[D(fake)])
        generator_loss = -mean(D(x_fake))

        g_optimizer.zero_grad()
        generator_loss.backward()
        g_optimizer.step()

    # Use RMSprop optimizer (not Adam) with lr = 0.00005
"""

    @staticmethod
    def training_procedure_gp() -> str:
        """Training with gradient penalty (WGAN-GP)."""
        return """
WGAN-GP Training (Gradient Penalty):
    n_critic = 5
    lambda_gp = 10

    for iteration in range(num_iterations):
        # ===== Train Critic =====
        for _ in range(n_critic):
            x_real = sample_data(batch_size)
            z = sample_noise(batch_size, latent_dim)
            x_fake = G(z).detach()

            # Critic outputs
            d_real = D(x_real)
            d_fake = D(x_fake)

            # Gradient penalty
            epsilon = rand(batch_size, 1, 1, 1)
            x_hat = epsilon * x_real + (1 - epsilon) * x_fake
            x_hat.requires_grad = True
            d_hat = D(x_hat)
            gradients = autograd.grad(d_hat, x_hat, grad_outputs=ones_like(d_hat),
                                      create_graph=True, retain_graph=True)[0]
            gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
            gradient_penalty = lambda_gp * mean((gradient_norm - 1) ** 2)

            # Total critic loss
            critic_loss = -mean(d_real) + mean(d_fake) + gradient_penalty

            d_optimizer.zero_grad()
            critic_loss.backward()
            d_optimizer.step()

        # ===== Train Generator =====
        z = sample_noise(batch_size, latent_dim)
        x_fake = G(z)
        generator_loss = -mean(D(x_fake))

        g_optimizer.zero_grad()
        generator_loss.backward()
        g_optimizer.step()

    # Can use Adam optimizer with WGAN-GP (lr=0.0001, beta1=0.0, beta2=0.9)
"""


# =============================================================================
# Theoretical Comparison
# =============================================================================

DISTANCE_COMPARISON = {
    "kl_divergence": {
        "formula": "KL(P||Q) = E_P[log(P/Q)]",
        "problem": "Undefined when P(x) > 0 and Q(x) = 0; asymmetric",
        "gradient": "Can explode or vanish"
    },
    "js_divergence": {
        "formula": "JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q)",
        "problem": "Saturates to log(2) when supports don't overlap",
        "gradient": "Zero when distributions don't overlap"
    },
    "total_variation": {
        "formula": "TV(P,Q) = 0.5 * integral |P(x) - Q(x)| dx",
        "problem": "Also saturates when supports don't overlap",
        "gradient": "Discontinuous"
    },
    "wasserstein": {
        "formula": "W_1(P,Q) = inf_gamma E_gamma[||x-y||]",
        "advantage": "Continuous everywhere; provides meaningful gradients",
        "gradient": "Always informative, correlates with sample quality"
    }
}


# =============================================================================
# Practical Considerations
# =============================================================================

WGAN_TIPS = {
    "weight_clipping_issues": {
        "capacity_underuse": "Clipping limits critic capacity",
        "gradient_problems": "Can lead to vanishing or exploding gradients",
        "recommendation": "Use gradient penalty (WGAN-GP) instead"
    },
    "wgan_gp_tips": {
        "no_batch_norm": "BatchNorm causes issues with gradient penalty; use LayerNorm/InstanceNorm",
        "two_sided_penalty": "Penalize ||grad|| != 1, not just ||grad|| > 1",
        "optimizer": "Adam works with GP (unlike original WGAN which needed RMSprop)"
    },
    "loss_interpretation": {
        "critic_loss": "Negative of Wasserstein estimate (lower = better)",
        "generator_loss": "Wasserstein estimate (lower = better)",
        "correlation": "Both losses now correlate with sample quality!"
    }
}
