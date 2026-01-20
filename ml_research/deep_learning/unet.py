"""
U-Net (2015)

U-Net: Convolutional Networks for Biomedical Image Segmentation
Authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

U-Net introduced a symmetric encoder-decoder architecture with skip connections
for precise segmentation of biomedical images. The architecture resembles
a "U" shape: contracting path (encoder) on the left, expanding path (decoder)
on the right, with skip connections between corresponding levels.

Key Innovations:
    - Encoder-decoder with symmetric expanding path
    - Skip connections: concatenate encoder features with decoder features
    - Data augmentation for small datasets (elastic deformations)
    - Overlap-tile strategy for seamless segmentation of large images
    - Works with very few training images

Design Philosophy:
    - Contracting path captures context (what)
    - Expanding path enables precise localization (where)
    - Skip connections combine low-level details with high-level semantics
    - Originally designed for cell segmentation in microscopy images
"""


from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from typing import Dict, List


UNET = MLMethod(
    method_id="unet_2015",
    name="U-Net",
    year=2015,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Olaf Ronneberger", "Philipp Fischer", "Thomas Brox"],
    paper_title="U-Net: Convolutional Networks for Biomedical Image Segmentation",
    paper_url="https://arxiv.org/abs/1505.04597",
    key_innovation="Symmetric encoder-decoder architecture with skip connections that "
                   "concatenate feature maps from encoder to decoder, enabling precise "
                   "localization while using context from the contracting path",
    mathematical_formulation="""
    Encoder (Contracting Path):
        For each level l:
            h_l = MaxPool(Conv(Conv(h_{l-1})))
            Spatial dimensions halved, channels doubled

    Decoder (Expanding Path):
        For each level l:
            up = UpConv(h_l)                    # Upsample
            concat = Concat(up, encoder_l)      # Skip connection
            h_{l-1} = Conv(Conv(concat))         # Process

    Skip Connections:
        decoder_input_l = Concat(upsampled_{l+1}, encoder_output_l)
        Preserves high-resolution features from encoder

    Output:
        For C classes: output = Conv1x1(final_decoder_output)
        Each pixel classified into one of C classes

    Loss (weighted cross-entropy + optional):
        L = sum_{x} w(x) * log(p(class(x) | x))
        where w(x) emphasizes cell borders
    """,
    predecessors=["fcn_2014", "alexnet_2012"],
    successors=["unet++_2018", "attention_unet_2018", "unet_3d_2016"],
    tags=["segmentation", "biomedical", "encoder_decoder", "skip_connections", "small_data"],
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for U-Net."""
    return UNET


def pseudocode() -> str:
    """Return pseudocode describing U-Net architecture."""
    return """
    U-NET ARCHITECTURE:

    function UNet(input_image, num_classes):
        # Input: H x W x C (e.g., 572 x 572 x 1 for original paper)

        # ============= ENCODER (Contracting Path) =============

        # Level 1: 572 -> 568 -> 284
        e1 = Conv2D(input_image, filters=64, kernel=3x3)  # -> 570
        e1 = ReLU(e1)
        e1 = Conv2D(e1, filters=64, kernel=3x3)  # -> 568
        e1 = ReLU(e1)
        p1 = MaxPool(e1, size=2x2, stride=2)  # -> 284

        # Level 2: 284 -> 280 -> 140
        e2 = Conv2D(p1, filters=128, kernel=3x3)  # -> 282
        e2 = ReLU(e2)
        e2 = Conv2D(e2, filters=128, kernel=3x3)  # -> 280
        e2 = ReLU(e2)
        p2 = MaxPool(e2, size=2x2, stride=2)  # -> 140

        # Level 3: 140 -> 136 -> 68
        e3 = Conv2D(p2, filters=256, kernel=3x3)  # -> 138
        e3 = ReLU(e3)
        e3 = Conv2D(e3, filters=256, kernel=3x3)  # -> 136
        e3 = ReLU(e3)
        p3 = MaxPool(e3, size=2x2, stride=2)  # -> 68

        # Level 4: 68 -> 64 -> 32
        e4 = Conv2D(p3, filters=512, kernel=3x3)  # -> 66
        e4 = ReLU(e4)
        e4 = Conv2D(e4, filters=512, kernel=3x3)  # -> 64
        e4 = ReLU(e4)
        p4 = MaxPool(e4, size=2x2, stride=2)  # -> 32

        # ============= BOTTLENECK =============

        # Level 5 (bottom): 32 -> 28
        b = Conv2D(p4, filters=1024, kernel=3x3)  # -> 30
        b = ReLU(b)
        b = Conv2D(b, filters=1024, kernel=3x3)  # -> 28
        b = ReLU(b)

        # ============= DECODER (Expanding Path) =============

        # Level 4: 28 -> 56, concat with e4 (cropped to 56)
        d4 = UpConv(b, filters=512, kernel=2x2, stride=2)  # -> 56
        e4_crop = CenterCrop(e4, size=56)  # Crop e4 to match d4
        d4 = Concat([d4, e4_crop])  # -> 56x56x1024
        d4 = Conv2D(d4, filters=512, kernel=3x3)  # -> 54
        d4 = ReLU(d4)
        d4 = Conv2D(d4, filters=512, kernel=3x3)  # -> 52
        d4 = ReLU(d4)

        # Level 3: 52 -> 104, concat with e3 (cropped to 104)
        d3 = UpConv(d4, filters=256, kernel=2x2, stride=2)  # -> 104
        e3_crop = CenterCrop(e3, size=104)
        d3 = Concat([d3, e3_crop])  # -> 104x104x512
        d3 = Conv2D(d3, filters=256, kernel=3x3)
        d3 = ReLU(d3)
        d3 = Conv2D(d3, filters=256, kernel=3x3)
        d3 = ReLU(d3)

        # Level 2: -> 200, concat with e2 (cropped to 200)
        d2 = UpConv(d3, filters=128, kernel=2x2, stride=2)
        e2_crop = CenterCrop(e2, size=d2.shape)
        d2 = Concat([d2, e2_crop])
        d2 = Conv2D(d2, filters=128, kernel=3x3)
        d2 = ReLU(d2)
        d2 = Conv2D(d2, filters=128, kernel=3x3)
        d2 = ReLU(d2)

        # Level 1: -> 392, concat with e1 (cropped to 392)
        d1 = UpConv(d2, filters=64, kernel=2x2, stride=2)
        e1_crop = CenterCrop(e1, size=d1.shape)
        d1 = Concat([d1, e1_crop])
        d1 = Conv2D(d1, filters=64, kernel=3x3)
        d1 = ReLU(d1)
        d1 = Conv2D(d1, filters=64, kernel=3x3)
        d1 = ReLU(d1)

        # ============= OUTPUT =============

        output = Conv2D(d1, filters=num_classes, kernel=1x1)  # -> 388x388x num_classes
        # For binary: sigmoid activation
        # For multi-class: softmax activation

        return output


    MODERN U-NET (with padding='same'):

    # Using padding='same' preserves dimensions:
    # Input: 256 x 256 x C
    # Encoder outputs: 256, 128, 64, 32, 16
    # Decoder outputs: 32, 64, 128, 256
    # Output: 256 x 256 x num_classes (same as input!)
    # No cropping needed when using same padding


    DATA AUGMENTATION (critical for biomedical):

    function augment_biomedical(image, mask):
        # Elastic deformations (most important)
        image, mask = elastic_transform(image, mask, alpha=2000, sigma=40)

        # Standard augmentations
        image, mask = random_rotation(image, mask, angle_range=[-180, 180])
        image, mask = random_flip(image, mask)
        image, mask = random_shift(image, mask)

        # Intensity augmentations (image only)
        image = random_brightness(image)
        image = random_contrast(image)
        image = add_gaussian_noise(image)

        return image, mask
    """


def key_equations() -> Dict[str, str]:
    """Return key equations for U-Net in LaTeX-style notation."""
    return {
        "encoder_block":
            "h_l = \\text{MaxPool}(\\text{ReLU}(\\text{Conv}(\\text{ReLU}(\\text{Conv}(h_{l-1})))))",

        "decoder_block":
            "h_l = \\text{Conv}(\\text{Conv}(\\text{Concat}(\\text{UpConv}(h_{l+1}), e_l)))",

        "skip_connection":
            "\\text{decoder\\_input}_l = [\\text{UpConv}(h_{l+1}); \\text{Crop}(e_l)]",

        "upconv":
            "\\text{UpConv}(x) = \\text{ConvTranspose2D}(x, \\text{stride}=2)",

        "weighted_cross_entropy":
            "L = \\sum_{x \\in \\Omega} w(x) \\log(p_{l(x)}(x))",

        "border_weight_map":
            "w(x) = w_c(x) + w_0 \\exp\\left(-\\frac{(d_1(x) + d_2(x))^2}{2\\sigma^2}\\right)",

        "dice_loss":
            "L_{Dice} = 1 - \\frac{2 \\sum_i p_i g_i + \\epsilon}{\\sum_i p_i + \\sum_i g_i + \\epsilon}",

        "output_size_original":
            "H_{out} = H_{in} - 184 \\quad (\\text{original U-Net without padding})",
    }


def architecture_details() -> Dict:
    """Return U-Net architecture specifications."""
    return {
        "original_unet": {
            "input_size": "572 x 572 x 1",
            "output_size": "388 x 388 x 2",
            "encoder_channels": [64, 128, 256, 512, 1024],
            "decoder_channels": [512, 256, 128, 64],
            "convolution": "3x3, no padding (valid)",
            "upsampling": "2x2 transposed convolution",
            "skip_connections": "Crop + Concatenate",
            "params": "~31M",
        },
        "modern_unet": {
            "input_size": "any H x W x C (typically 256 or 512)",
            "output_size": "same as input",
            "convolution": "3x3, padding='same'",
            "normalization": "BatchNorm or GroupNorm",
            "dropout": "Optional, typically 0.5",
            "attention": "Optional attention gates",
        },
        "training": {
            "optimizer": "SGD with momentum (0.99 in original)",
            "loss": "Weighted cross-entropy or Dice loss",
            "augmentation": "Elastic deformations, rotations, flips",
            "batch_size": "1 (due to large image size)",
        }
    }


def get_historical_context() -> str:
    """Return historical context and significance of U-Net."""
    return """
    U-Net (2015) was designed specifically for biomedical image segmentation
    where annotated training data is extremely limited.

    Problem Context:
    - Biomedical images require pixel-precise segmentation
    - Annotation is expensive (requires domain experts)
    - Typical datasets have only 10-50 annotated images
    - Cell boundaries must be accurately delineated

    Key Innovations:
    1. Symmetric encoder-decoder: Unlike FCN which had asymmetric paths
    2. Skip connections via concatenation: Preserves fine spatial information
    3. Extensive augmentation: Elastic deformations mimic biological variability
    4. Overlap-tile strategy: Allows segmentation of arbitrarily large images
    5. Weighted loss: Emphasizes cell borders and rare classes

    Competition Results:
    - Won ISBI 2015 cell tracking challenge
    - Achieved IoU of 92% with only ~30 training images
    - Significantly outperformed second place

    Legacy:
    - U-Net became THE architecture for medical image segmentation
    - Inspired countless variants: U-Net++, Attention U-Net, V-Net (3D), etc.
    - Extended to 3D, multi-scale, recurrent versions
    - Still competitive in 2020s with modern modifications
    - Over 50,000 citations - one of the most cited DL papers
    """


def get_limitations() -> List[str]:
    """Return known limitations of U-Net."""
    return [
        "Original version loses border pixels (572 -> 388)",
        "Skip connections are simple concatenation (no attention)",
        "No multi-scale context aggregation",
        "Memory intensive for large/3D images",
        "Sensitive to class imbalance without proper weighting",
        "Fixed receptive field size",
    ]


def get_applications() -> List[str]:
    """Return applications of U-Net."""
    return [
        "Cell segmentation (original application)",
        "Medical image segmentation (CT, MRI, X-ray)",
        "Pathology slide analysis",
        "Satellite image segmentation",
        "Autonomous driving (semantic segmentation)",
        "Industrial defect detection",
        "General image-to-image translation",
        "Video segmentation",
    ]


def unet_variants() -> Dict[str, str]:
    """Return descriptions of U-Net variants."""
    return {
        "V-Net (2016)": "3D U-Net for volumetric segmentation, uses residual connections",
        "U-Net++ (2018)": "Nested skip connections with dense connections between levels",
        "Attention U-Net (2018)": "Attention gates to focus on relevant features",
        "ResU-Net": "Residual connections within encoder/decoder blocks",
        "R2U-Net": "Recurrent residual U-Net with recurrent conv blocks",
        "U-Net 3+ (2020)": "Full-scale skip connections across all levels",
        "TransUNet (2021)": "Combines U-Net with Vision Transformer encoder",
        "nnU-Net (2021)": "Self-configuring U-Net with automatic hyperparameter selection",
    }
