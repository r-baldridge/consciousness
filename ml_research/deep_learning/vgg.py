"""
VGG Networks (2014)

Very Deep Convolutional Networks for Large-Scale Image Recognition
Authors: Karen Simonyan, Andrew Zisserman (Visual Geometry Group, Oxford)

VGG demonstrated that network depth is critical for good performance, achieving
this through a simple, elegant design: using only 3x3 convolutional filters
stacked deeply. This showed that multiple small filters can replace larger
filters while being more expressive and parameter-efficient.

Key Insights:
    - Two 3x3 convolutions have the same receptive field as one 5x5
    - Three 3x3 convolutions have the same receptive field as one 7x7
    - But 3x3 stacks have more non-linearities and fewer parameters
    - Depth matters: VGG-16 and VGG-19 significantly outperformed shallower nets

Architecture Variants:
    - VGG-11: 8 conv + 3 FC layers
    - VGG-13: 10 conv + 3 FC layers
    - VGG-16: 13 conv + 3 FC layers (most popular)
    - VGG-19: 16 conv + 3 FC layers (deepest)

Design Philosophy:
    - All convolutions: 3x3 with stride 1, padding 1
    - All pooling: 2x2 max pooling with stride 2
    - Double filters after each pooling: 64 -> 128 -> 256 -> 512 -> 512
"""


from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from typing import Dict, List


VGG = MLMethod(
    method_id="vgg_2014",
    name="VGG",
    year=2014,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Karen Simonyan", "Andrew Zisserman"],
    paper_title="Very Deep Convolutional Networks for Large-Scale Image Recognition",
    paper_url="https://arxiv.org/abs/1409.1556",
    key_innovation="Demonstrated that very deep networks with small 3x3 filters "
                   "outperform shallower networks with larger filters",
    mathematical_formulation="""
    Receptive Field Equivalence:
        Two 3x3 convolutions: effective receptive field = 5x5
        Three 3x3 convolutions: effective receptive field = 7x7

    Parameter Comparison (for C channels):
        One 7x7 conv: 7 * 7 * C * C = 49C^2 parameters
        Three 3x3 convs: 3 * (3 * 3 * C * C) = 27C^2 parameters
        Savings: 45% fewer parameters with more non-linearities

    Feature Map Sizes (VGG-16):
        Input: 224 x 224 x 3
        After block 1: 112 x 112 x 64
        After block 2: 56 x 56 x 128
        After block 3: 28 x 28 x 256
        After block 4: 14 x 14 x 512
        After block 5: 7 x 7 x 512
    """,
    predecessors=["alexnet_2012"],
    successors=["resnet_2015", "googlenet_2014"],
    tags=["imagenet", "depth", "small_filters", "transfer_learning"],
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for VGG."""
    return VGG


def pseudocode() -> str:
    """Return pseudocode describing VGG architecture."""
    return """
    VGG-16 ARCHITECTURE:

    function VGG16(input_image):
        # Input: 224 x 224 x 3 RGB image

        # BLOCK 1: 64 filters
        x = Conv2D(input_image, filters=64, kernel=3x3, padding='same')  # -> 224x224x64
        x = ReLU(x)
        x = Conv2D(x, filters=64, kernel=3x3, padding='same')  # -> 224x224x64
        x = ReLU(x)
        x = MaxPool(x, size=2x2, stride=2)  # -> 112x112x64

        # BLOCK 2: 128 filters
        x = Conv2D(x, filters=128, kernel=3x3, padding='same')  # -> 112x112x128
        x = ReLU(x)
        x = Conv2D(x, filters=128, kernel=3x3, padding='same')  # -> 112x112x128
        x = ReLU(x)
        x = MaxPool(x, size=2x2, stride=2)  # -> 56x56x128

        # BLOCK 3: 256 filters
        x = Conv2D(x, filters=256, kernel=3x3, padding='same')  # -> 56x56x256
        x = ReLU(x)
        x = Conv2D(x, filters=256, kernel=3x3, padding='same')  # -> 56x56x256
        x = ReLU(x)
        x = Conv2D(x, filters=256, kernel=3x3, padding='same')  # -> 56x56x256
        x = ReLU(x)
        x = MaxPool(x, size=2x2, stride=2)  # -> 28x28x256

        # BLOCK 4: 512 filters
        x = Conv2D(x, filters=512, kernel=3x3, padding='same')  # -> 28x28x512
        x = ReLU(x)
        x = Conv2D(x, filters=512, kernel=3x3, padding='same')  # -> 28x28x512
        x = ReLU(x)
        x = Conv2D(x, filters=512, kernel=3x3, padding='same')  # -> 28x28x512
        x = ReLU(x)
        x = MaxPool(x, size=2x2, stride=2)  # -> 14x14x512

        # BLOCK 5: 512 filters
        x = Conv2D(x, filters=512, kernel=3x3, padding='same')  # -> 14x14x512
        x = ReLU(x)
        x = Conv2D(x, filters=512, kernel=3x3, padding='same')  # -> 14x14x512
        x = ReLU(x)
        x = Conv2D(x, filters=512, kernel=3x3, padding='same')  # -> 14x14x512
        x = ReLU(x)
        x = MaxPool(x, size=2x2, stride=2)  # -> 7x7x512

        # FULLY CONNECTED LAYERS
        x = Flatten(x)  # -> 25088
        x = Dense(x, units=4096)
        x = ReLU(x)
        x = Dropout(x, p=0.5)

        x = Dense(x, units=4096)
        x = ReLU(x)
        x = Dropout(x, p=0.5)

        x = Dense(x, units=1000)
        output = Softmax(x)

        return output


    VGG CONFIGURATIONS:

    VGG-11 (A):  [64] - [128] - [256, 256] - [512, 512] - [512, 512]
    VGG-13 (B):  [64, 64] - [128, 128] - [256, 256] - [512, 512] - [512, 512]
    VGG-16 (D):  [64, 64] - [128, 128] - [256, 256, 256] - [512, 512, 512] - [512, 512, 512]
    VGG-19 (E):  [64, 64] - [128, 128] - [256, 256, 256, 256] - [512, 512, 512, 512] - [512, 512, 512, 512]

    (Each block followed by MaxPool 2x2, numbers indicate filters per conv layer)
    """


def key_equations() -> Dict[str, str]:
    """Return key equations for VGG in LaTeX-style notation."""
    return {
        "3x3_conv":
            "y_{i,j,k} = \\text{ReLU}\\left(\\sum_{m,n,c} w_{m,n,c,k} \\cdot x_{i+m-1, j+n-1, c} + b_k\\right)",

        "receptive_field_2_layers":
            "RF_{2 \\times 3} = 1 + 2(3-1) = 5 \\quad (\\text{same as } 5 \\times 5)",

        "receptive_field_3_layers":
            "RF_{3 \\times 3} = 1 + 3(3-1) = 7 \\quad (\\text{same as } 7 \\times 7)",

        "parameter_savings":
            "\\frac{27C^2}{49C^2} \\approx 0.55 \\quad (\\text{45\\% fewer parameters})",

        "maxpool_output":
            "W_{out} = \\lfloor W_{in} / 2 \\rfloor",

        "total_params_vgg16":
            "P = 138M \\quad (\\text{~123M in FC layers})",

        "feature_map_formula":
            "W_{out} = W_{in} \\quad (\\text{with padding='same' for 3x3 conv})",
    }


def architecture_configs() -> Dict[str, List]:
    """Return VGG architecture configurations."""
    return {
        "VGG-11": {
            "conv_layers": 8,
            "fc_layers": 3,
            "config": "A",
            "blocks": [[64], [128], [256, 256], [512, 512], [512, 512]],
            "params": "133M",
        },
        "VGG-13": {
            "conv_layers": 10,
            "fc_layers": 3,
            "config": "B",
            "blocks": [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
            "params": "133M",
        },
        "VGG-16": {
            "conv_layers": 13,
            "fc_layers": 3,
            "config": "D",
            "blocks": [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
            "params": "138M",
        },
        "VGG-19": {
            "conv_layers": 16,
            "fc_layers": 3,
            "config": "E",
            "blocks": [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
            "params": "144M",
        },
        "training": {
            "optimizer": "SGD with momentum (0.9)",
            "initial_lr": 0.01,
            "weight_decay": 0.0005,
            "batch_size": 256,
            "lr_schedule": "Reduce by 10x when validation accuracy plateaus",
        }
    }


def get_historical_context() -> str:
    """Return historical context and significance of VGG."""
    return """
    VGG Networks (2014) from the Visual Geometry Group at Oxford University
    demonstrated that architectural simplicity combined with depth yields
    excellent results.

    Key Contributions:
    1. Established 3x3 convolutions as the standard building block
    2. Showed clear correlation between depth and performance
    3. Provided simple, modular design that's easy to understand and implement
    4. Created excellent pre-trained models for transfer learning

    Competition Results:
    - 2nd place in ILSVRC-2014 classification (behind GoogLeNet)
    - 1st place in ILSVRC-2014 localization
    - Top-5 error: 7.3% (single model), 6.8% (ensemble)

    Legacy:
    - VGG-16 and VGG-19 became the most popular pre-trained models
    - "VGG features" widely used as generic image descriptors
    - Architecture influenced many subsequent designs
    - Still used today for style transfer and perceptual losses
    """


def get_limitations() -> List[str]:
    """Return known limitations of VGG."""
    return [
        "Very large model size (500+ MB for VGG-16)",
        "~138M parameters, most in FC layers",
        "Slow inference compared to more efficient architectures",
        "High memory consumption during training",
        "Prone to vanishing gradients in deeper variants",
        "No skip connections limits maximum effective depth",
    ]


def get_applications() -> List[str]:
    """Return applications of VGG."""
    return [
        "Transfer learning backbone (widely used)",
        "Neural style transfer (feature matching)",
        "Perceptual loss functions (VGG loss)",
        "Image classification benchmarking",
        "Object detection (Fast R-CNN backbone)",
        "Image similarity and retrieval",
        "Feature extraction for downstream tasks",
    ]
