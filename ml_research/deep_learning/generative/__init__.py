"""
Deep Learning Generative Methods Module

This module contains research indices for generative deep learning models
from the deep learning era (2013-present).

Key Methods:
    - Variational Autoencoder (VAE, 2013): Latent variable model with ELBO
    - Generative Adversarial Network (GAN, 2014): Adversarial training framework
    - Deep Convolutional GAN (DCGAN, 2015): CNN-based GAN architecture
    - Wasserstein GAN (WGAN, 2017): Earth mover distance objective
    - StyleGAN (2018-2021): Style-based generator architecture
"""

from .vae import (
    VARIATIONAL_AUTOENCODER,
    elbo_loss,
    reparameterization_trick,
    kl_divergence_gaussian,
)
from .gan import (
    GENERATIVE_ADVERSARIAL_NETWORK,
    discriminator_loss,
    generator_loss,
    minimax_objective,
)
from .dcgan import (
    DEEP_CONVOLUTIONAL_GAN,
    dcgan_generator_block,
    dcgan_discriminator_block,
)
from .wgan import (
    WASSERSTEIN_GAN,
    wasserstein_distance,
    critic_loss,
    gradient_penalty,
)
from .stylegan import (
    STYLEGAN,
    adaptive_instance_norm,
    mapping_network,
    style_mixing,
)

__all__ = [
    # Variational Autoencoder
    "VARIATIONAL_AUTOENCODER",
    "elbo_loss",
    "reparameterization_trick",
    "kl_divergence_gaussian",
    # Generative Adversarial Network
    "GENERATIVE_ADVERSARIAL_NETWORK",
    "discriminator_loss",
    "generator_loss",
    "minimax_objective",
    # Deep Convolutional GAN
    "DEEP_CONVOLUTIONAL_GAN",
    "dcgan_generator_block",
    "dcgan_discriminator_block",
    # Wasserstein GAN
    "WASSERSTEIN_GAN",
    "wasserstein_distance",
    "critic_loss",
    "gradient_penalty",
    # StyleGAN
    "STYLEGAN",
    "adaptive_instance_norm",
    "mapping_network",
    "style_mixing",
]
