"""
DenseNet - Densely Connected Convolutional Networks (2016)

Densely Connected Convolutional Networks
Authors: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger

DenseNet extends the idea of skip connections by connecting EVERY layer to
EVERY subsequent layer in a feed-forward fashion. Within a dense block,
layer l receives the feature maps of all preceding layers as input:

    x_l = H_l([x_0, x_1, ..., x_{l-1}])

This creates L(L+1)/2 connections in an L-layer network (vs L in traditional).

Key Advantages:
    - Maximum feature reuse (all features accessible to all layers)
    - Alleviates vanishing gradient (direct access to loss gradient)
    - Substantially fewer parameters (narrow layers, k growth rate)
    - Implicit deep supervision (short paths to all features)
    - Feature propagation is efficient (no need to relearn)

Architecture Components:
    - Dense blocks: Densely connected layers
    - Transition layers: 1x1 conv + 2x2 avg pool (between dense blocks)
    - Growth rate (k): Number of feature maps added by each layer
"""


from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from typing import Dict, List


DENSENET = MLMethod(
    method_id="densenet_2016",
    name="DenseNet",
    year=2016,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Gao Huang", "Zhuang Liu", "Laurens van der Maaten", "Kilian Q. Weinberger"],
    paper_title="Densely Connected Convolutional Networks",
    paper_url="https://arxiv.org/abs/1608.06993",
    key_innovation="Dense connections where each layer receives feature maps from ALL "
                   "preceding layers as input, maximizing feature reuse and enabling "
                   "very deep networks with fewer parameters",
    mathematical_formulation="""
    Dense Connectivity:
        x_l = H_l([x_0, x_1, x_2, ..., x_{l-1}])

        where [x_0, x_1, ...] denotes concatenation of feature maps
        and H_l is a composite function: BN -> ReLU -> Conv

    Feature Map Growth:
        At layer l, number of input feature maps = k_0 + k * (l - 1)
        where k_0 is initial features and k is growth rate

    Composite Function H_l:
        H_l(x) = Conv3x3(ReLU(BN(x)))
        or with bottleneck:
        H_l(x) = Conv3x3(ReLU(BN(Conv1x1(ReLU(BN(x))))))

    Transition Layer (between dense blocks):
        x = AvgPool2x2(Conv1x1(ReLU(BN(x))))
        Reduces feature maps by compression factor theta (typically 0.5)

    Total Parameters (compared to ResNet):
        DenseNet-121 (8M) vs ResNet-50 (25M) with similar accuracy
    """,
    predecessors=["resnet_2015", "highway_networks_2015"],
    successors=["efficient_net_2019", "dual_path_networks_2017"],
    tags=["imagenet", "dense_connections", "feature_reuse", "parameter_efficient"],
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for DenseNet."""
    return DENSENET


def pseudocode() -> str:
    """Return pseudocode describing DenseNet architecture."""
    return """
    DENSE LAYER:

    function DenseLayer(x, growth_rate, bottleneck=True):
        # x is a list of feature maps from all previous layers
        combined = Concatenate(x)  # Stack all previous features

        if bottleneck:
            # Bottleneck: reduce channels before 3x3 conv
            out = BatchNorm(combined)
            out = ReLU(out)
            out = Conv2D(out, filters=4*growth_rate, kernel=1x1)  # Bottleneck
            out = BatchNorm(out)
            out = ReLU(out)
            out = Conv2D(out, filters=growth_rate, kernel=3x3, padding=1)
        else:
            out = BatchNorm(combined)
            out = ReLU(out)
            out = Conv2D(out, filters=growth_rate, kernel=3x3, padding=1)

        return out  # Returns k new feature maps


    DENSE BLOCK:

    function DenseBlock(x, num_layers, growth_rate, bottleneck=True):
        features = [x]  # List to collect all feature maps

        for i in range(num_layers):
            new_features = DenseLayer(features, growth_rate, bottleneck)
            features.append(new_features)  # Add new features to the list

        return Concatenate(features)  # Return all features concatenated


    TRANSITION LAYER:

    function TransitionLayer(x, compression=0.5):
        # Reduces spatial dimensions and number of channels
        num_features = int(x.channels * compression)

        x = BatchNorm(x)
        x = ReLU(x)
        x = Conv2D(x, filters=num_features, kernel=1x1)  # Compress channels
        x = AvgPool(x, size=2x2, stride=2)  # Reduce spatial size

        return x


    DENSENET ARCHITECTURE:

    function DenseNet(input_image, num_init_features=64, growth_rate=32,
                      block_config=(6, 12, 24, 16), compression=0.5):
        # block_config specifies layers per dense block
        # DenseNet-121: (6, 12, 24, 16) = 6+12+24+16 = 58 dense layers
        # DenseNet-169: (6, 12, 32, 32) = 82 dense layers
        # DenseNet-201: (6, 12, 48, 32) = 98 dense layers
        # DenseNet-264: (6, 12, 64, 48) = 130 dense layers

        # STEM
        x = Conv2D(input_image, filters=num_init_features, kernel=7x7, stride=2)
        x = BatchNorm(x)
        x = ReLU(x)
        x = MaxPool(x, size=3x3, stride=2, padding=1)  # -> 56x56x64

        # DENSE BLOCKS + TRANSITIONS
        for i, num_layers in enumerate(block_config):
            # Dense block
            x = DenseBlock(x, num_layers, growth_rate)

            # Transition layer (except after last block)
            if i != len(block_config) - 1:
                x = TransitionLayer(x, compression)

        # CLASSIFIER HEAD
        x = BatchNorm(x)
        x = ReLU(x)
        x = GlobalAveragePool(x)
        x = Dense(x, units=1000)
        output = Softmax(x)

        return output


    FEATURE MAP COUNTING EXAMPLE (DenseNet-121, k=32, Dense Block 1):

    Initial features: 64
    After layer 1: 64 + 32 = 96
    After layer 2: 96 + 32 = 128
    After layer 3: 128 + 32 = 160
    After layer 4: 160 + 32 = 192
    After layer 5: 192 + 32 = 224
    After layer 6: 224 + 32 = 256
    After transition (compression=0.5): 128
    """


def key_equations() -> Dict[str, str]:
    """Return key equations for DenseNet in LaTeX-style notation."""
    return {
        "dense_connectivity":
            "x_l = H_l([x_0, x_1, \\ldots, x_{l-1}])",

        "composite_function":
            "H_l = \\text{Conv}_{3\\times3}(\\text{ReLU}(\\text{BN}(\\cdot)))",

        "bottleneck_composite":
            "H_l = \\text{Conv}_{3\\times3}(\\text{ReLU}(\\text{BN}("
            "\\text{Conv}_{1\\times1}(\\text{ReLU}(\\text{BN}(\\cdot))))))",

        "feature_map_count":
            "k_l = k_0 + k \\times (l - 1)",

        "transition_compression":
            "k_{out} = \\lfloor \\theta \\times k_{in} \\rfloor, \\quad \\theta = 0.5",

        "total_connections":
            "\\text{connections} = \\frac{L(L+1)}{2}",

        "gradient_flow":
            "\\frac{\\partial L}{\\partial x_0} = \\sum_{l=1}^{L} \\frac{\\partial L}{\\partial x_l} "
            "\\frac{\\partial x_l}{\\partial x_0}",
    }


def architecture_configs() -> Dict:
    """Return DenseNet architecture configurations."""
    return {
        "DenseNet-121": {
            "block_config": (6, 12, 24, 16),
            "growth_rate": 32,
            "params": "8.0M",
            "top1_error": "25.0%",
            "top5_error": "7.7%",
        },
        "DenseNet-169": {
            "block_config": (6, 12, 32, 32),
            "growth_rate": 32,
            "params": "14.1M",
            "top1_error": "24.0%",
            "top5_error": "7.0%",
        },
        "DenseNet-201": {
            "block_config": (6, 12, 48, 32),
            "growth_rate": 32,
            "params": "20.0M",
            "top1_error": "22.6%",
            "top5_error": "6.3%",
        },
        "DenseNet-264": {
            "block_config": (6, 12, 64, 48),
            "growth_rate": 32,
            "params": "27.2M",
            "top1_error": "22.2%",
            "top5_error": "6.1%",
        },
        "common_hyperparams": {
            "initial_features": 64,
            "compression": 0.5,
            "bottleneck_factor": 4,  # 4*k filters in 1x1 bottleneck
            "growth_rate_typical": [12, 24, 32, 40, 48],
        }
    }


def get_historical_context() -> str:
    """Return historical context and significance of DenseNet."""
    return """
    DenseNet (2016) pushed the idea of feature reuse to its logical extreme:
    connect every layer to every other layer.

    Evolution of Skip Connections:
    - Highway Networks (2015): Gated skip connections
    - ResNet (2015): Identity skip connections (addition)
    - DenseNet (2016): All-to-all connections (concatenation)

    Key Differences from ResNet:
    - ResNet: y = F(x) + x (addition)
    - DenseNet: y = H([x_0, x_1, ..., x_{l-1}]) (concatenation)

    Advantages of concatenation:
    - No information loss from addition
    - Each layer has access to all previous features
    - Gradient flows directly to all layers
    - Implicit deep supervision

    Parameter Efficiency:
    - DenseNet-121 achieves similar accuracy to ResNet-50 with ~3x fewer parameters
    - Narrow layers (small k) are sufficient due to collective knowledge
    - Transition layers compress accumulated features

    Awards and Recognition:
    - Best Paper Award at CVPR 2017
    - >12,000 citations
    - Widely adopted in medical imaging and other domains
    """


def get_limitations() -> List[str]:
    """Return known limitations of DenseNet."""
    return [
        "High memory consumption during training (storing all features)",
        "Concatenation creates large intermediate tensors",
        "Memory-efficient implementation requires special care",
        "Inference can be slower than ResNet despite fewer parameters",
        "Feature explosion requires transition layer compression",
        "GPU memory often the bottleneck, not FLOPs",
    ]


def get_applications() -> List[str]:
    """Return applications of DenseNet."""
    return [
        "Medical image analysis (very popular due to efficiency)",
        "Image classification",
        "Object detection (as backbone)",
        "Semantic segmentation (FC-DenseNet)",
        "Low-resource scenarios (fewer parameters)",
        "Transfer learning (compact pre-trained models)",
    ]
