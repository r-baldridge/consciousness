"""
GoogLeNet / Inception (2014)

Going Deeper with Convolutions
Authors: Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
         Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich

GoogLeNet (named after Yann LeCun's LeNet) introduced the Inception module,
a key architectural innovation that processes input at multiple scales
simultaneously. This allowed building much deeper networks (22 layers) with
fewer parameters than AlexNet (5M vs 60M), winning ILSVRC 2014.

Key Innovations:
    - Inception module: parallel convolutions at multiple scales (1x1, 3x3, 5x5)
    - 1x1 convolutions: dimensionality reduction before expensive operations
    - Auxiliary classifiers: intermediate supervision to combat vanishing gradients
    - Global average pooling: replaces FC layers, reducing parameters
    - Network-in-Network influence: 1x1 convs as mini-networks

Design Philosophy:
    - "Wider" rather than just "deeper"
    - Multi-scale feature extraction within each module
    - Computational efficiency through bottleneck layers
"""


from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from typing import Dict, List


INCEPTION = MLMethod(
    method_id="googlenet_2014",
    name="GoogLeNet/Inception",
    year=2014,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Christian Szegedy", "Wei Liu", "Yangqing Jia", "Pierre Sermanet",
             "Scott Reed", "Dragomir Anguelov", "Dumitru Erhan",
             "Vincent Vanhoucke", "Andrew Rabinovich"],
    paper_title="Going Deeper with Convolutions",
    paper_url="https://arxiv.org/abs/1409.4842",
    key_innovation="Inception modules that process features at multiple scales "
                   "simultaneously using parallel 1x1, 3x3, and 5x5 convolutions, "
                   "with 1x1 convolutions as bottlenecks for efficiency",
    mathematical_formulation="""
    Inception Module (naive):
        Output = Concat([Conv1x1(x), Conv3x3(x), Conv5x5(x), MaxPool3x3(x)])

    Inception Module (with dimensionality reduction):
        branch1 = Conv1x1(x)
        branch2 = Conv3x3(Conv1x1(x))      # 1x1 reduces channels before 3x3
        branch3 = Conv5x5(Conv1x1(x))      # 1x1 reduces channels before 5x5
        branch4 = Conv1x1(MaxPool3x3(x))   # 1x1 after pooling
        Output = Concat([branch1, branch2, branch3, branch4])

    1x1 Convolution (bottleneck):
        y_{i,j,k} = sum_c (w_{c,k} * x_{i,j,c}) + b_k
        Reduces C_in channels to C_out < C_in before expensive convolutions

    Global Average Pooling:
        z_k = (1/HW) * sum_{i,j} x_{i,j,k}
        Outputs one value per channel, replacing FC layers
    """,
    predecessors=["alexnet_2012", "network_in_network_2013"],
    successors=["inception_v2_2015", "inception_v3_2015", "inception_resnet_2016"],
    tags=["imagenet", "inception_module", "1x1_convolution", "multi_scale", "auxiliary_classifiers"],
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for GoogLeNet/Inception."""
    return INCEPTION


def pseudocode() -> str:
    """Return pseudocode describing the Inception architecture."""
    return """
    INCEPTION MODULE (with dimensionality reduction):

    function InceptionModule(x, f1, f3_reduce, f3, f5_reduce, f5, pool_proj):
        # f1: filters for 1x1 branch
        # f3_reduce, f3: filters for 3x3 branch (reduce then conv)
        # f5_reduce, f5: filters for 5x5 branch (reduce then conv)
        # pool_proj: filters for pool branch projection

        # Branch 1: 1x1 convolution
        branch1 = Conv2D(x, filters=f1, kernel=1x1)
        branch1 = ReLU(branch1)

        # Branch 2: 1x1 reduction -> 3x3 convolution
        branch2 = Conv2D(x, filters=f3_reduce, kernel=1x1)
        branch2 = ReLU(branch2)
        branch2 = Conv2D(branch2, filters=f3, kernel=3x3, padding='same')
        branch2 = ReLU(branch2)

        # Branch 3: 1x1 reduction -> 5x5 convolution
        branch3 = Conv2D(x, filters=f5_reduce, kernel=1x1)
        branch3 = ReLU(branch3)
        branch3 = Conv2D(branch3, filters=f5, kernel=5x5, padding='same')
        branch3 = ReLU(branch3)

        # Branch 4: 3x3 max pool -> 1x1 projection
        branch4 = MaxPool(x, size=3x3, stride=1, padding='same')
        branch4 = Conv2D(branch4, filters=pool_proj, kernel=1x1)
        branch4 = ReLU(branch4)

        # Concatenate along channel dimension
        output = Concat([branch1, branch2, branch3, branch4], axis=channels)

        return output


    GOOGLENET ARCHITECTURE:

    function GoogLeNet(input_image):
        # Input: 224 x 224 x 3

        # STEM
        x = Conv2D(input_image, filters=64, kernel=7x7, stride=2)  # -> 112x112x64
        x = ReLU(x)
        x = MaxPool(x, size=3x3, stride=2)  # -> 56x56x64
        x = LocalResponseNorm(x)

        x = Conv2D(x, filters=64, kernel=1x1)  # -> 56x56x64
        x = ReLU(x)
        x = Conv2D(x, filters=192, kernel=3x3, padding='same')  # -> 56x56x192
        x = ReLU(x)
        x = LocalResponseNorm(x)
        x = MaxPool(x, size=3x3, stride=2)  # -> 28x28x192

        # INCEPTION MODULES (3a, 3b)
        x = InceptionModule(x, 64, 96, 128, 16, 32, 32)   # 3a: -> 28x28x256
        x = InceptionModule(x, 128, 128, 192, 32, 96, 64)  # 3b: -> 28x28x480
        x = MaxPool(x, size=3x3, stride=2)  # -> 14x14x480

        # INCEPTION MODULES (4a, 4b, 4c, 4d, 4e)
        x = InceptionModule(x, 192, 96, 208, 16, 48, 64)   # 4a: -> 14x14x512

        # AUXILIARY CLASSIFIER 1 (at 4a output)
        aux1 = AuxiliaryClassifier(x)  # Used during training only

        x = InceptionModule(x, 160, 112, 224, 24, 64, 64)  # 4b: -> 14x14x512
        x = InceptionModule(x, 128, 128, 256, 24, 64, 64)  # 4c: -> 14x14x512
        x = InceptionModule(x, 112, 144, 288, 32, 64, 64)  # 4d: -> 14x14x528

        # AUXILIARY CLASSIFIER 2 (at 4d output)
        aux2 = AuxiliaryClassifier(x)  # Used during training only

        x = InceptionModule(x, 256, 160, 320, 32, 128, 128)  # 4e: -> 14x14x832
        x = MaxPool(x, size=3x3, stride=2)  # -> 7x7x832

        # INCEPTION MODULES (5a, 5b)
        x = InceptionModule(x, 256, 160, 320, 32, 128, 128)  # 5a: -> 7x7x832
        x = InceptionModule(x, 384, 192, 384, 48, 128, 128)  # 5b: -> 7x7x1024

        # CLASSIFIER HEAD
        x = GlobalAveragePool(x)  # -> 1024
        x = Dropout(x, p=0.4)
        x = Dense(x, units=1000)
        output = Softmax(x)

        return output, aux1, aux2


    AUXILIARY CLASSIFIER:

    function AuxiliaryClassifier(x):
        x = AvgPool(x, size=5x5, stride=3)  # -> 4x4xC
        x = Conv2D(x, filters=128, kernel=1x1)
        x = ReLU(x)
        x = Flatten(x)
        x = Dense(x, units=1024)
        x = ReLU(x)
        x = Dropout(x, p=0.7)
        x = Dense(x, units=1000)
        output = Softmax(x)
        return output  # Loss weighted by 0.3 during training
    """


def key_equations() -> Dict[str, str]:
    """Return key equations for Inception in LaTeX-style notation."""
    return {
        "1x1_convolution":
            "y_{i,j,k} = \\sum_{c=1}^{C} w_{c,k} \\cdot x_{i,j,c} + b_k",

        "inception_output":
            "y = \\text{Concat}[\\text{Conv}_{1\\times1}, \\text{Conv}_{3\\times3}, "
            "\\text{Conv}_{5\\times5}, \\text{Pool}_{3\\times3}]",

        "channel_reduction":
            "\\text{FLOPs}_{reduced} = H \\cdot W \\cdot C_{in} \\cdot C_{bottleneck} + "
            "H \\cdot W \\cdot C_{bottleneck} \\cdot k^2 \\cdot C_{out}",

        "global_avg_pool":
            "z_k = \\frac{1}{H \\cdot W} \\sum_{i=1}^{H} \\sum_{j=1}^{W} x_{i,j,k}",

        "total_loss":
            "L = L_{main} + 0.3 \\cdot L_{aux1} + 0.3 \\cdot L_{aux2}",

        "receptive_field_multi_scale":
            "RF = \\{1 \\times 1, 3 \\times 3, 5 \\times 5\\} \\rightarrow "
            "\\text{captures features at multiple scales}",
    }


def inception_module_configs() -> List[Dict]:
    """Return configurations for each Inception module in GoogLeNet."""
    return [
        # (1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, pool_proj) -> output_channels
        {"name": "inception_3a", "config": (64, 96, 128, 16, 32, 32), "output": 256},
        {"name": "inception_3b", "config": (128, 128, 192, 32, 96, 64), "output": 480},
        {"name": "inception_4a", "config": (192, 96, 208, 16, 48, 64), "output": 512},
        {"name": "inception_4b", "config": (160, 112, 224, 24, 64, 64), "output": 512},
        {"name": "inception_4c", "config": (128, 128, 256, 24, 64, 64), "output": 512},
        {"name": "inception_4d", "config": (112, 144, 288, 32, 64, 64), "output": 528},
        {"name": "inception_4e", "config": (256, 160, 320, 32, 128, 128), "output": 832},
        {"name": "inception_5a", "config": (256, 160, 320, 32, 128, 128), "output": 832},
        {"name": "inception_5b", "config": (384, 192, 384, 48, 128, 128), "output": 1024},
    ]


def get_historical_context() -> str:
    """Return historical context and significance of GoogLeNet/Inception."""
    return """
    GoogLeNet (2014) introduced the Inception module, fundamentally changing
    how we think about CNN architecture design.

    Key Insights:
    1. Network-in-Network: 1x1 convolutions act as mini fully-connected networks
    2. Multi-scale processing: Different filter sizes capture different features
    3. Computational efficiency: Bottlenecks reduce computation dramatically
    4. Global Average Pooling: Eliminates majority of parameters in FC layers

    Competition Results:
    - Won ILSVRC 2014 classification with 6.67% top-5 error
    - Beat VGG (2nd place at 7.3%) with 12x fewer parameters

    Evolution of Inception:
    - Inception v1 (GoogLeNet): Original 2014 paper
    - Inception v2: Added Batch Normalization (2015)
    - Inception v3: Factorized convolutions, label smoothing (2015)
    - Inception v4: Cleaner, more uniform design (2016)
    - Inception-ResNet: Combined with residual connections (2016)

    The Inception module concept influenced many subsequent architectures,
    including ResNeXt and Xception (depthwise separable convolutions).
    """


def get_limitations() -> List[str]:
    """Return known limitations of GoogLeNet/Inception."""
    return [
        "Complex architecture with many hyperparameters per module",
        "Auxiliary classifiers add training complexity",
        "Manual tuning of filter counts for each branch",
        "5x5 convolutions still computationally expensive",
        "No residual connections limits further depth increase",
        "Architecture not easily scalable/modular",
    ]


def get_applications() -> List[str]:
    """Return applications of GoogLeNet/Inception."""
    return [
        "Image classification (original application)",
        "Transfer learning (Inception v3 widely used)",
        "Object detection (as backbone)",
        "Medical image analysis",
        "Video classification (frame-level features)",
        "Mobile and embedded vision (efficient variants)",
    ]
