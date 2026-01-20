"""
Deep Convolutional GAN (DCGAN) - 2015

Radford, Metz, and Chintala's influential work establishing architectural
guidelines for stable GAN training with convolutional networks. Became
the de facto standard for image generation with GANs.

Paper: "Unsupervised Representation Learning with Deep Convolutional
        Generative Adversarial Networks" (ICLR 2016)
arXiv: 1511.06434

Key Architectural Guidelines:
    1. Replace pooling with strided convolutions (discriminator) and
       fractionally-strided convolutions (generator)
    2. Use batch normalization in both G and D
    3. Remove fully connected hidden layers for deeper architectures
    4. Use ReLU in generator (except output: Tanh)
    5. Use LeakyReLU in discriminator

Mathematical Formulation:
    Same minimax objective as vanilla GAN:
    min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]

    Innovation is architectural, not in the loss function.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

DEEP_CONVOLUTIONAL_GAN = MLMethod(
    method_id="dcgan_2015",
    name="Deep Convolutional GAN",
    year=2015,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.GENERATIVE,
    lineages=[MethodLineage.GENERATIVE_LINE, MethodLineage.CNN_LINE],

    authors=["Alec Radford", "Luke Metz", "Soumith Chintala"],
    paper_title="Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks",
    paper_url="https://arxiv.org/abs/1511.06434",

    key_innovation=(
        "Established architectural guidelines for stable training of convolutional GANs. "
        "Key insights: strided convolutions instead of pooling, batch normalization, "
        "removal of fully connected layers, and specific activation functions. "
        "Also demonstrated that learned features transfer to other tasks."
    ),

    mathematical_formulation=r"""
Loss Function (same as vanilla GAN):
    min_G max_D V(D, G) = E_x~p_data[log D(x)] + E_z~p_z[log(1 - D(G(z)))]

Architectural Guidelines:

Generator:
    z [batch, 100] -> reshape -> [batch, 100, 1, 1]

    ConvTranspose2d: [100, 1, 1] -> [ngf*8, 4, 4]    # Project and reshape
    BatchNorm2d + ReLU

    ConvTranspose2d: [ngf*8, 4, 4] -> [ngf*4, 8, 8]
    BatchNorm2d + ReLU

    ConvTranspose2d: [ngf*4, 8, 8] -> [ngf*2, 16, 16]
    BatchNorm2d + ReLU

    ConvTranspose2d: [ngf*2, 16, 16] -> [ngf, 32, 32]
    BatchNorm2d + ReLU

    ConvTranspose2d: [ngf, 32, 32] -> [nc, 64, 64]
    Tanh

Discriminator:
    x [batch, nc, 64, 64]

    Conv2d: [nc, 64, 64] -> [ndf, 32, 32]
    LeakyReLU(0.2)

    Conv2d: [ndf, 32, 32] -> [ndf*2, 16, 16]
    BatchNorm2d + LeakyReLU(0.2)

    Conv2d: [ndf*2, 16, 16] -> [ndf*4, 8, 8]
    BatchNorm2d + LeakyReLU(0.2)

    Conv2d: [ndf*4, 8, 8] -> [ndf*8, 4, 4]
    BatchNorm2d + LeakyReLU(0.2)

    Conv2d: [ndf*8, 4, 4] -> [1, 1, 1]
    Sigmoid

Where:
    nc = number of channels (3 for RGB)
    ngf = generator feature maps (typically 64)
    ndf = discriminator feature maps (typically 64)
""",

    predecessors=["gan_2014"],
    successors=["wgan_2017", "progressive_gan", "stylegan_2018"],

    tags=["generative", "convolutional", "adversarial", "architecture", "deep-learning"]
)


# =============================================================================
# Architectural Components (Pseudocode/Reference)
# =============================================================================

def dcgan_generator_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, activation="relu"):
    """
    Standard DCGAN generator block: ConvTranspose -> BatchNorm -> ReLU

    Fractionally-strided convolution (transposed convolution) for upsampling.

    Spatial relationship:
        H_out = (H_in - 1) * stride - 2 * padding + kernel_size
        For stride=2, kernel=4, padding=1: H_out = 2 * H_in (doubles spatial size)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Typically 4
        stride: Typically 2 for 2x upsampling
        padding: Typically 1
        batch_norm: Whether to apply batch normalization
        activation: "relu" for all layers except final (use "tanh")

    Returns:
        Description of generator block
    """
    return {
        "structure": [
            "ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)",
            "BatchNorm2d(out_channels)" if batch_norm else "None",
            "ReLU(inplace=True)" if activation == "relu" else "Tanh()"
        ],
        "output_size": "(H_in - 1) * stride - 2 * padding + kernel_size",
        "note": "Use bias=False when using BatchNorm (BN has its own bias)"
    }


def dcgan_discriminator_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, first_layer=False):
    """
    Standard DCGAN discriminator block: Conv -> BatchNorm -> LeakyReLU

    Strided convolution for downsampling (replaces pooling).

    Spatial relationship:
        H_out = (H_in + 2 * padding - kernel_size) / stride + 1
        For stride=2, kernel=4, padding=1: H_out = H_in / 2 (halves spatial size)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Typically 4
        stride: Typically 2 for 2x downsampling
        padding: Typically 1
        batch_norm: Whether to apply batch normalization
        first_layer: If True, skip BatchNorm (first layer shouldn't have BN)

    Returns:
        Description of discriminator block
    """
    return {
        "structure": [
            "Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)",
            "BatchNorm2d(out_channels)" if batch_norm and not first_layer else "None",
            "LeakyReLU(0.2, inplace=True)"
        ],
        "output_size": "(H_in + 2 * padding - kernel_size) / stride + 1",
        "note": "No BatchNorm in first layer; LeakyReLU slope of 0.2 is standard"
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class DCGANArchitecture:
    """Reference architecture for DCGAN (64x64 images)."""

    image_size: int = 64
    num_channels: int = 3  # RGB
    latent_dim: int = 100
    ngf: int = 64  # Generator feature maps
    ndf: int = 64  # Discriminator feature maps

    @staticmethod
    def generator_structure() -> str:
        """DCGAN Generator architecture."""
        return """
DCGAN Generator (64x64 output):
    Input: z [batch_size, 100, 1, 1]

    # Project and reshape
    ConvTranspose2d(100, ngf*8, 4, 1, 0)  # [ngf*8, 4, 4]
    BatchNorm2d(ngf*8)
    ReLU

    # Upsample blocks
    ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1)  # [ngf*4, 8, 8]
    BatchNorm2d(ngf*4)
    ReLU

    ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1)  # [ngf*2, 16, 16]
    BatchNorm2d(ngf*2)
    ReLU

    ConvTranspose2d(ngf*2, ngf, 4, 2, 1)    # [ngf, 32, 32]
    BatchNorm2d(ngf)
    ReLU

    # Final layer (no BatchNorm, Tanh activation)
    ConvTranspose2d(ngf, nc, 4, 2, 1)       # [nc, 64, 64]
    Tanh

    Output: image [batch_size, 3, 64, 64] in [-1, 1]
"""

    @staticmethod
    def discriminator_structure() -> str:
        """DCGAN Discriminator architecture."""
        return """
DCGAN Discriminator (64x64 input):
    Input: x [batch_size, nc, 64, 64]

    # First layer (no BatchNorm)
    Conv2d(nc, ndf, 4, 2, 1)               # [ndf, 32, 32]
    LeakyReLU(0.2)

    # Downsample blocks
    Conv2d(ndf, ndf*2, 4, 2, 1)            # [ndf*2, 16, 16]
    BatchNorm2d(ndf*2)
    LeakyReLU(0.2)

    Conv2d(ndf*2, ndf*4, 4, 2, 1)          # [ndf*4, 8, 8]
    BatchNorm2d(ndf*4)
    LeakyReLU(0.2)

    Conv2d(ndf*4, ndf*8, 4, 2, 1)          # [ndf*8, 4, 4]
    BatchNorm2d(ndf*8)
    LeakyReLU(0.2)

    # Final layer
    Conv2d(ndf*8, 1, 4, 1, 0)              # [1, 1, 1]
    Sigmoid

    Output: probability [batch_size, 1, 1, 1]
"""

    @staticmethod
    def weight_initialization() -> str:
        """DCGAN weight initialization."""
        return """
Weight Initialization:
    All weights initialized from N(0, 0.02)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
"""


# =============================================================================
# Architectural Guidelines Summary
# =============================================================================

DCGAN_GUIDELINES = {
    "replace_pooling": {
        "description": "Use strided convolutions instead of pooling",
        "generator": "Fractionally-strided convolutions (ConvTranspose2d) for upsampling",
        "discriminator": "Strided convolutions (Conv2d with stride > 1) for downsampling",
        "rationale": "Allows network to learn its own spatial downsampling/upsampling"
    },
    "batch_normalization": {
        "description": "Use BatchNorm in both G and D",
        "exceptions": [
            "No BatchNorm in generator output layer",
            "No BatchNorm in discriminator input layer"
        ],
        "rationale": "Stabilizes training by normalizing inputs to each layer"
    },
    "remove_fc_layers": {
        "description": "Remove fully connected hidden layers",
        "generator": "Global average pooling not used; direct conv to spatial",
        "discriminator": "Final conv reduces to 1x1 spatial",
        "rationale": "Fully convolutional design improves stability"
    },
    "activations": {
        "generator": {
            "hidden": "ReLU",
            "output": "Tanh (output in [-1, 1])"
        },
        "discriminator": {
            "all_layers": "LeakyReLU with slope 0.2"
        },
        "rationale": "Bounded Tanh prevents saturation; LeakyReLU allows gradient flow for negative values"
    },
    "optimizer": {
        "algorithm": "Adam",
        "learning_rate": 0.0002,
        "beta1": 0.5,  # Lower than default 0.9
        "beta2": 0.999,
        "rationale": "Lower beta1 found to stabilize training"
    }
}


# =============================================================================
# Latent Space Properties
# =============================================================================

LATENT_SPACE_PROPERTIES = {
    "vector_arithmetic": {
        "description": "Semantic operations in latent space",
        "example": "vector('man with glasses') - vector('man') + vector('woman') = vector('woman with glasses')",
        "method": "Average z vectors from multiple examples, then compute arithmetic",
        "significance": "Demonstrates disentangled representations emerge from unsupervised training"
    },
    "interpolation": {
        "description": "Smooth transitions between generated images",
        "method": "Linear interpolation: z = alpha * z1 + (1 - alpha) * z2",
        "observation": "Intermediate points produce coherent images, not noise",
        "significance": "Indicates generator learned continuous data manifold"
    },
    "feature_visualization": {
        "description": "Individual units correspond to semantic features",
        "observation": "Certain feature maps activate for specific objects (windows, beds, etc.)",
        "method": "Identify units via guided backpropagation or activation maximization"
    }
}
