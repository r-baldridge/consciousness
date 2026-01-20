"""
StyleGAN - 2018-2021

Karras et al. (NVIDIA) style-based generator architecture that revolutionized
high-resolution image synthesis. Introduced mapping network, adaptive instance
normalization (AdaIN), and style mixing for unprecedented control over generation.

Papers:
    - StyleGAN (2018): "A Style-Based Generator Architecture for GANs"
      arXiv: 1812.04948
    - StyleGAN2 (2019): "Analyzing and Improving the Image Quality of StyleGAN"
      arXiv: 1912.04958
    - StyleGAN3 (2021): "Alias-Free Generative Adversarial Networks"
      arXiv: 2106.12423

Mathematical Formulation:
    Mapping Network:
        w = f(z), where f: Z -> W is 8-layer MLP
        z ~ N(0, I), w in intermediate latent space W

    Adaptive Instance Normalization (AdaIN):
        AdaIN(x, y) = y_s * (x - mu(x)) / sigma(x) + y_b
        Where y_s, y_b are style parameters derived from w

    Style Injection:
        A(w) = [y_s, y_b] learned affine transformation
        Applied at each resolution level
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

STYLEGAN = MLMethod(
    method_id="stylegan_2018",
    name="StyleGAN",
    year=2018,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.GENERATIVE,
    lineages=[MethodLineage.GENERATIVE_LINE, MethodLineage.CNN_LINE],

    authors=["Tero Karras", "Samuli Laine", "Timo Aila"],
    paper_title="A Style-Based Generator Architecture for Generative Adversarial Networks",
    paper_url="https://arxiv.org/abs/1812.04948",

    key_innovation=(
        "Introduced style-based generator with mapping network transforming latent z "
        "to intermediate space w, enabling scale-specific style control via AdaIN. "
        "Demonstrated disentangled latent space with style mixing and unprecedented "
        "image quality at 1024x1024 resolution. Progressive training inherited from "
        "ProGAN for stable high-resolution generation."
    ),

    mathematical_formulation=r"""
StyleGAN Architecture Overview:

1. Mapping Network f: Z -> W
    z ~ N(0, I)^512
    w = MLP_8layers(z)     # 8 fully-connected layers with LeakyReLU
    w in R^512

    Purpose: Transform entangled z to disentangled w

2. Synthesis Network g: W -> Image
    Starting from learned constant input (4x4)

    For each layer:
        - Upsample (bilinear or learned)
        - Conv2d
        - Add noise (B * noise, learned per-feature scaling B)
        - AdaIN(features, style_from_w)

3. Adaptive Instance Normalization (AdaIN):
    style = A(w)  # Affine transformation: [y_s, y_b] = Linear(w)

    AdaIN(x, w) = y_s * ((x - mu(x)) / sigma(x)) + y_b

    Where:
        mu(x), sigma(x) = mean, std computed per feature map per sample
        y_s = scale (multiplicative style)
        y_b = bias (additive style)

4. Noise Injection:
    For each layer: output = conv_output + B * noise
    Where:
        B is learned per-feature scalar
        noise ~ N(0, 1) is spatial (H x W)

    Purpose: Stochastic variation (hair details, freckles, background)

5. Style Mixing (Regularization):
    During training, with probability p:
        Use w1 for layers [0, crossover_point)
        Use w2 for layers [crossover_point, end]

    Encourages each layer to be independent, improves disentanglement

StyleGAN2 Improvements:
    - Weight demodulation instead of AdaIN (removes blob artifacts)
    - Path length regularization
    - No progressive growing (direct training at full resolution)
    - Improved architecture without normalization artifacts

StyleGAN3 Improvements:
    - Alias-free operations (texture sticking problem)
    - Rotation/translation equivariant generator
    - Continuous signal interpretation
""",

    predecessors=["dcgan_2015", "wgan_2017", "progressive_gan_2017"],
    successors=["stylegan2_2019", "stylegan3_2021", "stylegan_xl"],

    tags=["generative", "style-based", "high-resolution", "face-synthesis", "deep-learning"]
)


# =============================================================================
# Mathematical Functions (Pseudocode/Reference)
# =============================================================================

def adaptive_instance_norm(content, style_mean, style_std):
    """
    Adaptive Instance Normalization (AdaIN).

    AdaIN(x, y) = y_s * ((x - mu(x)) / sigma(x)) + y_b

    Combines content from x with style from y by:
    1. Normalizing x to zero mean, unit variance (per instance, per channel)
    2. Scaling by style std (y_s) and shifting by style mean (y_b)

    Originally from style transfer, adapted for conditional generation in StyleGAN.

    Args:
        content: Feature maps [batch, channels, height, width]
        style_mean: Style bias (y_b) [batch, channels]
        style_std: Style scale (y_s) [batch, channels]

    Returns:
        Styled features [batch, channels, height, width]

    Note in StyleGAN:
        style = A(w) where A is learned affine transformation
        [style_std, style_mean] = Linear(w)
    """
    return {
        "formula": "AdaIN(x, y) = y_s * ((x - mu(x)) / sigma(x)) + y_b",
        "normalization": "Instance norm: mean/std over H, W only (not batch)",
        "style_source": "Derived from w via learned affine transformation A(w)",
        "intuition": "Separates content (normalized activations) from style (statistics)"
    }


def mapping_network(z, num_layers=8, latent_dim=512):
    """
    Mapping network: Z -> W

    8-layer MLP that transforms input latent z to intermediate latent w.

    Purpose:
        - z space is entangled (features may be correlated)
        - w space is more disentangled (features more independent)
        - Enables meaningful interpolation and manipulation

    Architecture:
        z [batch, 512]
        -> FC(512, 512), LeakyReLU(0.2)
        -> FC(512, 512), LeakyReLU(0.2)
        ... (8 layers total)
        -> w [batch, 512]

    Args:
        z: Random latent vector [batch, latent_dim]
        num_layers: Number of FC layers (default 8)
        latent_dim: Dimensionality (default 512)

    Returns:
        w: Intermediate latent vector [batch, latent_dim]

    Properties of W space:
        - More linear interpolation
        - Better disentanglement
        - Easier to find meaningful directions (age, pose, etc.)
    """
    return {
        "architecture": "8-layer MLP with LeakyReLU(0.2)",
        "input": "z ~ N(0, I) in R^512",
        "output": "w in R^512",
        "normalization": "Pixel norm on z, then mapping",
        "purpose": "Disentangle latent factors; z is entangled, w is not"
    }


def style_mixing(w1, w2, crossover_point):
    """
    Style mixing regularization.

    During training, with some probability:
        - Sample two latent codes z1, z2
        - Generate w1 = f(z1), w2 = f(z2)
        - Use w1 for early layers (coarse styles)
        - Use w2 for later layers (fine styles)

    This regularization:
        - Prevents adjacent styles from becoming correlated
        - Encourages each layer to specialize
        - Improves overall disentanglement

    Layer-to-style mapping (approximate, for 1024x1024):
        - Coarse (4-8px): Pose, face shape, eyeglasses
        - Middle (16-32px): Facial features, hairstyle
        - Fine (64-1024px): Colors, micro-features

    Args:
        w1: First intermediate latent [batch, latent_dim]
        w2: Second intermediate latent [batch, latent_dim]
        crossover_point: Layer at which to switch from w1 to w2

    Returns:
        Description of style mixing procedure
    """
    return {
        "procedure": [
            "1. Sample z1, z2 from N(0, I)",
            "2. Compute w1 = mapping(z1), w2 = mapping(z2)",
            "3. Randomly select crossover_point",
            "4. Use w1 for layers < crossover_point",
            "5. Use w2 for layers >= crossover_point"
        ],
        "mixing_probability": "0.9 during training (StyleGAN1)",
        "effect": "Forces layers to be style-independent",
        "at_inference": "Can mix styles from different images for creative control"
    }


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class StyleGANArchitecture:
    """Reference architecture for StyleGAN (1024x1024 face generation)."""

    latent_dim: int = 512
    mapping_layers: int = 8
    num_channels: int = 3
    resolution: int = 1024

    @staticmethod
    def synthesis_network_structure() -> str:
        """StyleGAN Synthesis Network."""
        return """
StyleGAN Synthesis Network:
    Input: Learned constant [1, 512, 4, 4] (no z/w directly!)

    # Each resolution block:
    Resolution 4x4:
        Constant input [512, 4, 4]
        + Noise
        AdaIN(features, A(w))  # Style from w
        Conv 3x3
        + Noise
        AdaIN(features, A(w))

    Resolution 8x8:
        Upsample (bilinear)
        Conv 3x3
        + Noise
        AdaIN(features, A(w))
        Conv 3x3
        + Noise
        AdaIN(features, A(w))

    ... continue doubling resolution ...

    Resolution 1024x1024:
        Upsample
        Conv 3x3, + Noise, AdaIN
        Conv 3x3, + Noise, AdaIN
        ToRGB (1x1 conv)

    # Progressive skip connections (ProGAN inheritance):
    Output = sum of ToRGB outputs, weighted during training

    Total: 18 style injection points for 1024x1024
           (2 per resolution: 4, 8, 16, 32, 64, 128, 256, 512, 1024)
"""

    @staticmethod
    def noise_injection() -> str:
        """Noise injection mechanism."""
        return """
Noise Injection:
    For each layer after convolution:

    noise = randn(batch, 1, H, W)     # Spatial noise, single channel
    B = learned_parameter(channels)   # Per-feature scaling

    output = conv_output + B * noise  # Broadcast across channels

    Purpose:
        - Adds stochastic variation to generation
        - Controls fine details: hair strands, pores, background texture
        - Style controls "what" (structure), noise controls "where" (variation)

    Training: Same noise across batch
    Inference: Can randomize for variation, or fix for consistency
"""

    @staticmethod
    def stylegan2_improvements() -> str:
        """Key improvements in StyleGAN2."""
        return """
StyleGAN2 Key Changes:

1. Weight Demodulation (replaces AdaIN):
    Problem: AdaIN causes "blob" artifacts (droplet-like patterns)
    Solution: Demodulate convolution weights instead of normalizing activations

    Modulation: w'_ijk = s_i * w_ijk           # Scale weights by style
    Demodulation: w''_ijk = w'_ijk / sqrt(sum_ik w'_ijk^2 + epsilon)

    Effect: Equivalent to InstanceNorm but operates on weights, no artifacts

2. Path Length Regularization:
    Encourage J^T_w J_w to be close to identity
    Preserves distances: small change in w -> proportional change in image

    L_pl = E_w,y [ (||J_w^T y||_2 - a)^2 ]
    Where J_w is Jacobian of g(w), y is random image

3. No Progressive Growing:
    Train directly at full resolution with skip connections
    Simpler, no transition artifacts

4. Lazy Regularization:
    Apply regularization every 16 steps (not every step)
    Faster training with negligible quality loss
"""

    @staticmethod
    def stylegan3_improvements() -> str:
        """Key improvements in StyleGAN3."""
        return """
StyleGAN3 Key Changes (Alias-Free):

Problem: "Texture sticking" - fine details attached to pixel coordinates,
         not to underlying surfaces. Visible when interpolating latent codes.

Root Cause: Aliasing from non-ideal operations (upsampling, nonlinearities)

1. Continuous Signal Interpretation:
    Treat features as continuous signals, not discrete grids
    Apply ideal resampling (sinc filters)

2. Rotation/Translation Equivariance:
    Generator is equivariant to geometric transformations
    Moving in latent space = geometric transformation in image

3. Alias-Free Operations:
    - Lowpass filter before downsampling
    - Lowpass filter after nonlinearities
    - Careful handling of boundary conditions

4. Fourier Features:
    Replace learned constant with Fourier features
    Enables continuous positional encoding

Result: Smooth latent interpolation without texture sticking
"""


# =============================================================================
# Latent Space Structure
# =============================================================================

LATENT_SPACES = {
    "Z_space": {
        "description": "Input latent space",
        "distribution": "z ~ N(0, I) in R^512",
        "properties": "Entangled; spherically symmetric",
        "usage": "Initial sampling point"
    },
    "W_space": {
        "description": "Intermediate latent space (after mapping network)",
        "distribution": "Learned; no longer Gaussian",
        "properties": "Disentangled; linear interpolation works better",
        "usage": "One w vector for all layers (single style code)"
    },
    "W_plus_space": {
        "description": "Extended W space",
        "structure": "Separate w for each layer: [w_1, w_2, ..., w_18]",
        "properties": "Most expressive; used for GAN inversion",
        "usage": "Image editing, reconstruction from real images"
    },
    "S_space": {
        "description": "Style space (after affine transform A)",
        "structure": "Per-layer style parameters [s_1, s_2, ..., s_18]",
        "properties": "Most disentangled; channel-aligned directions",
        "usage": "Fine-grained attribute manipulation"
    }
}


# =============================================================================
# Style-Attribute Mapping
# =============================================================================

STYLE_LAYERS = {
    "coarse_styles": {
        "layers": "4x4 to 8x8 (first 4 style vectors)",
        "controls": ["Pose", "Face shape", "Eyeglasses presence", "General structure"],
        "example": "Changing these changes overall face orientation"
    },
    "middle_styles": {
        "layers": "16x16 to 32x32 (middle 4 style vectors)",
        "controls": ["Facial features", "Hairstyle", "Eyes open/closed"],
        "example": "Changing these modifies facial expressions"
    },
    "fine_styles": {
        "layers": "64x64 to 1024x1024 (last ~10 style vectors)",
        "controls": ["Color scheme", "Skin texture", "Hair color", "Background"],
        "example": "Changing these adjusts colors without structure changes"
    }
}


# =============================================================================
# Applications
# =============================================================================

STYLEGAN_APPLICATIONS = {
    "face_generation": {
        "description": "Photorealistic face synthesis",
        "dataset": "FFHQ (Flickr-Faces-HQ) 70K images",
        "quality": "1024x1024, often indistinguishable from real photos"
    },
    "gan_inversion": {
        "description": "Project real images into latent space",
        "methods": ["Optimization-based", "Encoder-based", "Hybrid"],
        "applications": "Image editing, style transfer, age progression"
    },
    "attribute_editing": {
        "description": "Modify specific attributes in generated/inverted images",
        "method": "Find directions in W/S space corresponding to attributes",
        "examples": ["InterfaceGAN", "GANSpace", "StyleFlow", "StyleCLIP"]
    },
    "domain_adaptation": {
        "description": "Adapt pretrained StyleGAN to new domains",
        "methods": ["Fine-tuning", "Few-shot adaptation", "Transfer learning"],
        "examples": "Toonify, artistic styles, different face types"
    },
    "video_generation": {
        "description": "Generate temporally coherent video sequences",
        "approach": "Interpolate in latent space; StyleGAN3 better for this"
    }
}
