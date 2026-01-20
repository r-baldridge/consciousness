"""
Variational Autoencoder (VAE) - 2013

Kingma & Welling's foundational work on variational inference for
deep generative models. Introduced the reparameterization trick enabling
efficient gradient-based optimization of latent variable models.

Paper: "Auto-Encoding Variational Bayes" (ICLR 2014)
arXiv: 1312.6114

Mathematical Formulation:
    ELBO (Evidence Lower Bound):
        L(theta, phi; x) = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

    Where:
        - q(z|x) = N(mu(x), sigma^2(x)) is the encoder (recognition model)
        - p(x|z) is the decoder (generative model)
        - p(z) = N(0, I) is the prior over latent space

    Reparameterization Trick:
        z = mu + sigma * epsilon, where epsilon ~ N(0, I)

    KL Divergence (Gaussian):
        KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

VARIATIONAL_AUTOENCODER = MLMethod(
    method_id="vae_2013",
    name="Variational Autoencoder",
    year=2013,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.GENERATIVE,
    lineages=[MethodLineage.GENERATIVE_LINE],

    authors=["Diederik P. Kingma", "Max Welling"],
    paper_title="Auto-Encoding Variational Bayes",
    paper_url="https://arxiv.org/abs/1312.6114",

    key_innovation=(
        "Introduced the reparameterization trick enabling backpropagation through "
        "stochastic sampling, making variational inference scalable to deep neural "
        "networks. Combined variational Bayesian inference with autoencoder architecture."
    ),

    mathematical_formulation=r"""
ELBO (Evidence Lower Bound):
    L(theta, phi; x) = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

    Reconstruction term: E_q(z|x)[log p(x|z)]
    Regularization term: -KL(q(z|x) || p(z))

Encoder (Recognition Model):
    q_phi(z|x) = N(z; mu(x), sigma^2(x))
    mu, log_sigma = Encoder_phi(x)

Decoder (Generative Model):
    p_theta(x|z) = N(x; mu_out(z), sigma^2_out) or Bernoulli(x; p(z))

Reparameterization Trick:
    z = mu + sigma * epsilon, epsilon ~ N(0, I)
    Enables: dL/d_phi via chain rule through deterministic transformation

KL Divergence (Gaussian prior):
    KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
""",

    predecessors=["boltzmann_machine_1985", "autoencoder"],
    successors=["beta_vae", "vq_vae", "cvae"],

    tags=["generative", "variational-inference", "latent-variable", "autoencoder", "deep-learning"]
)


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def elbo_loss(x, reconstructed_x, mu, log_var, reconstruction_loss_fn="mse"):
    """
    Compute the ELBO loss for VAE training.

    L = E[log p(x|z)] - KL(q(z|x) || p(z))
      = -reconstruction_loss - kl_divergence

    Args:
        x: Original input data
        reconstructed_x: Decoder output
        mu: Mean of latent distribution q(z|x)
        log_var: Log variance of latent distribution
        reconstruction_loss_fn: "mse" or "bce"

    Returns:
        Dictionary with total loss, reconstruction loss, and KL divergence

    Mathematical Form:
        ELBO = sum_i [ -||x_i - x_hat_i||^2 ] - 0.5 * sum_j [ 1 + log(sigma_j^2) - mu_j^2 - sigma_j^2 ]
    """
    # Reference implementation structure
    return {
        "formula": "L = reconstruction_loss + beta * kl_divergence",
        "reconstruction": "E_q[log p(x|z)] approximated by -||x - decoder(z)||^2",
        "kl_divergence": "-0.5 * sum(1 + log_var - mu^2 - exp(log_var))",
        "note": "Minimize negative ELBO = maximize ELBO"
    }


def reparameterization_trick(mu, log_var):
    """
    Sample z from q(z|x) using the reparameterization trick.

    z = mu + sigma * epsilon, where epsilon ~ N(0, I)

    This allows gradients to flow through the sampling operation
    since epsilon is sampled independently of the parameters.

    Args:
        mu: Mean of the latent distribution [batch_size, latent_dim]
        log_var: Log variance of latent distribution [batch_size, latent_dim]

    Returns:
        z: Sampled latent vector [batch_size, latent_dim]

    Key Insight:
        Instead of sampling z ~ N(mu, sigma^2), we sample epsilon ~ N(0, I)
        and compute z = mu + sigma * epsilon. This makes the sampling
        deterministic given epsilon, enabling backpropagation through mu and sigma.
    """
    return {
        "formula": "z = mu + exp(0.5 * log_var) * epsilon",
        "epsilon": "epsilon ~ N(0, I)",
        "gradient_flow": "d_loss/d_mu and d_loss/d_log_var computable via chain rule"
    }


def kl_divergence_gaussian(mu, log_var):
    """
    Compute KL divergence between N(mu, sigma^2) and N(0, I).

    KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Derivation:
        KL = integral q(z) log(q(z)/p(z)) dz
           = integral N(mu, sigma^2) log(N(mu, sigma^2) / N(0, 1)) dz
           = 0.5 * (tr(sigma^2) + mu^T mu - k - log det(sigma^2))
           = -0.5 * sum_j (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)

    Args:
        mu: Mean of approximate posterior [batch_size, latent_dim]
        log_var: Log variance of approximate posterior [batch_size, latent_dim]

    Returns:
        KL divergence value (scalar, summed over latent dimensions)
    """
    return {
        "formula": "KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))",
        "per_dimension": "KL_j = 0.5 * (sigma_j^2 + mu_j^2 - 1 - log(sigma_j^2))",
        "closed_form": "Analytical solution exists for Gaussian-to-Gaussian KL"
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class VAEArchitecture:
    """Reference architecture for Variational Autoencoder."""

    input_dim: int = 784  # e.g., 28x28 MNIST
    latent_dim: int = 20
    hidden_dims: List[int] = field(default_factory=lambda: [400, 200])

    @staticmethod
    def encoder_structure() -> str:
        """Encoder network structure."""
        return """
Encoder q_phi(z|x):
    Input: x [batch_size, input_dim]

    h = ReLU(Linear(input_dim -> hidden_dim))
    h = ReLU(Linear(hidden_dim -> hidden_dim))

    mu = Linear(hidden_dim -> latent_dim)      # Mean
    log_var = Linear(hidden_dim -> latent_dim) # Log variance

    Output: mu, log_var [batch_size, latent_dim] each
"""

    @staticmethod
    def decoder_structure() -> str:
        """Decoder network structure."""
        return """
Decoder p_theta(x|z):
    Input: z [batch_size, latent_dim]

    h = ReLU(Linear(latent_dim -> hidden_dim))
    h = ReLU(Linear(hidden_dim -> hidden_dim))

    reconstruction = Sigmoid(Linear(hidden_dim -> input_dim))

    Output: reconstruction [batch_size, input_dim]
"""

    @staticmethod
    def training_procedure() -> str:
        """Training procedure pseudocode."""
        return """
VAE Training Loop:
    for batch in data_loader:
        # Encode
        mu, log_var = encoder(batch)

        # Reparameterize
        epsilon = sample_normal(0, 1)
        z = mu + exp(0.5 * log_var) * epsilon

        # Decode
        reconstruction = decoder(z)

        # Compute loss
        recon_loss = reconstruction_loss(batch, reconstruction)
        kl_loss = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        total_loss = recon_loss + kl_loss

        # Backpropagate
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
"""


# =============================================================================
# Variants Reference
# =============================================================================

VAE_VARIANTS = {
    "beta_vae": {
        "description": "VAE with weighted KL term for disentanglement",
        "loss": "L = reconstruction_loss + beta * kl_divergence",
        "paper": "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
    },
    "cvae": {
        "description": "Conditional VAE with label conditioning",
        "encoder": "q(z|x, y)",
        "decoder": "p(x|z, y)",
        "paper": "Learning Structured Output Representation using Deep Conditional Generative Models"
    },
    "vq_vae": {
        "description": "VAE with discrete latent codes via vector quantization",
        "latent": "Discrete codebook instead of continuous Gaussian",
        "paper": "Neural Discrete Representation Learning"
    },
    "hierarchical_vae": {
        "description": "Multiple levels of latent variables",
        "structure": "z_L -> z_{L-1} -> ... -> z_1 -> x",
        "paper": "Ladder Variational Autoencoders"
    }
}
