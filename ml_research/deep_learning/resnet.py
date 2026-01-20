"""
ResNet - Residual Networks (2015)

Deep Residual Learning for Image Recognition
Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)

ResNet introduced skip connections (residual connections) that allow training
of extremely deep networks (100+ layers). This solved the degradation problem
where deeper networks paradoxically performed worse than shallower ones due
to optimization difficulties, not overfitting.

The Core Insight:
    Instead of learning H(x) directly, learn F(x) = H(x) - x
    Then H(x) = F(x) + x (residual + identity)

    If the optimal function is close to identity, F(x) is easier to learn
    (just push weights toward zero) than learning H(x) = x directly.

Key Innovations:
    - Residual connections: y = F(x) + x
    - Batch normalization after every convolution
    - Bottleneck blocks for deeper networks (1x1-3x3-1x1)
    - Global average pooling (no FC layers except final classifier)
    - Enabled training of 152-layer networks (and 1000+ in experiments)

Architecture Variants:
    - ResNet-18: 18 layers (basic blocks)
    - ResNet-34: 34 layers (basic blocks)
    - ResNet-50: 50 layers (bottleneck blocks)
    - ResNet-101: 101 layers (bottleneck blocks)
    - ResNet-152: 152 layers (bottleneck blocks)
"""


from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from typing import Dict, List


RESNET = MLMethod(
    method_id="resnet_2015",
    name="ResNet",
    year=2015,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
    paper_title="Deep Residual Learning for Image Recognition",
    paper_url="https://arxiv.org/abs/1512.03385",
    key_innovation="Residual connections (skip connections) that enable direct gradient "
                   "flow through identity mappings, solving the degradation problem and "
                   "allowing training of networks with 100+ layers",
    mathematical_formulation="""
    Residual Block:
        y = F(x, {W_i}) + x

        where F(x, {W_i}) is the residual function to be learned
        and x is the identity shortcut connection

    Basic Block (ResNet-18/34):
        F(x) = W_2 * ReLU(BN(W_1 * x))
        y = ReLU(BN(F(x) + x))

    Bottleneck Block (ResNet-50/101/152):
        F(x) = W_3 * ReLU(BN(W_2 * ReLU(BN(W_1 * x))))
        where W_1 is 1x1 (reduce), W_2 is 3x3, W_3 is 1x1 (expand)
        y = ReLU(BN(F(x) + x))

    Dimension Matching (when dimensions change):
        y = F(x, {W_i}) + W_s * x
        where W_s is a 1x1 conv that matches dimensions

    Gradient Flow:
        dy/dx = dF/dx + 1
        The +1 ensures gradient always flows through identity path
    """,
    predecessors=["vgg_2014", "googlenet_2014", "highway_networks_2015"],
    successors=["resnext_2016", "densenet_2016", "preact_resnet_2016"],
    tags=["imagenet", "residual_connections", "skip_connections", "very_deep", "batch_norm"],
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for ResNet."""
    return RESNET


def pseudocode() -> str:
    """Return pseudocode describing ResNet architecture."""
    return """
    BASIC RESIDUAL BLOCK (for ResNet-18/34):

    function BasicBlock(x, filters, stride=1, downsample=None):
        identity = x

        # First conv layer
        out = Conv2D(x, filters=filters, kernel=3x3, stride=stride, padding=1)
        out = BatchNorm(out)
        out = ReLU(out)

        # Second conv layer
        out = Conv2D(out, filters=filters, kernel=3x3, stride=1, padding=1)
        out = BatchNorm(out)

        # Shortcut connection
        if downsample is not None:  # When dimensions change
            identity = downsample(x)  # 1x1 conv to match dimensions

        out = out + identity  # RESIDUAL CONNECTION
        out = ReLU(out)

        return out


    BOTTLENECK BLOCK (for ResNet-50/101/152):

    function BottleneckBlock(x, filters, stride=1, downsample=None):
        identity = x
        width = filters  # bottleneck width

        # 1x1 conv (reduce dimensions)
        out = Conv2D(x, filters=width, kernel=1x1)
        out = BatchNorm(out)
        out = ReLU(out)

        # 3x3 conv (main computation)
        out = Conv2D(out, filters=width, kernel=3x3, stride=stride, padding=1)
        out = BatchNorm(out)
        out = ReLU(out)

        # 1x1 conv (expand dimensions, 4x expansion)
        out = Conv2D(out, filters=filters*4, kernel=1x1)
        out = BatchNorm(out)

        # Shortcut connection
        if downsample is not None:
            identity = downsample(x)

        out = out + identity  # RESIDUAL CONNECTION
        out = ReLU(out)

        return out


    RESNET ARCHITECTURE:

    function ResNet(input_image, block_type, layers):
        # layers = [n1, n2, n3, n4] blocks per stage
        # ResNet-50:  [3, 4, 6, 3]
        # ResNet-101: [3, 4, 23, 3]
        # ResNet-152: [3, 8, 36, 3]

        # STEM
        x = Conv2D(input_image, filters=64, kernel=7x7, stride=2, padding=3)
        x = BatchNorm(x)
        x = ReLU(x)
        x = MaxPool(x, size=3x3, stride=2, padding=1)  # -> 56x56x64

        # STAGE 1: 64 filters (56x56)
        x = make_layer(x, block_type, filters=64, blocks=layers[0], stride=1)

        # STAGE 2: 128 filters (28x28)
        x = make_layer(x, block_type, filters=128, blocks=layers[1], stride=2)

        # STAGE 3: 256 filters (14x14)
        x = make_layer(x, block_type, filters=256, blocks=layers[2], stride=2)

        # STAGE 4: 512 filters (7x7)
        x = make_layer(x, block_type, filters=512, blocks=layers[3], stride=2)

        # CLASSIFIER HEAD
        x = GlobalAveragePool(x)  # -> 512 or 2048 (bottleneck)
        x = Dense(x, units=1000)
        output = Softmax(x)

        return output


    function make_layer(x, block_type, filters, blocks, stride):
        # First block may downsample
        downsample = None
        if stride != 1 or x.channels != filters * expansion:
            downsample = Sequential(
                Conv2D(filters=filters*expansion, kernel=1x1, stride=stride),
                BatchNorm()
            )

        x = block_type(x, filters, stride, downsample)

        # Remaining blocks
        for i in range(1, blocks):
            x = block_type(x, filters, stride=1)

        return x


    RESNET CONFIGURATIONS:

    ResNet-18:  BasicBlock,     [2, 2, 2, 2]    ~11M params
    ResNet-34:  BasicBlock,     [3, 4, 6, 3]    ~22M params
    ResNet-50:  BottleneckBlock, [3, 4, 6, 3]    ~25M params
    ResNet-101: BottleneckBlock, [3, 4, 23, 3]   ~44M params
    ResNet-152: BottleneckBlock, [3, 8, 36, 3]   ~60M params
    """


def key_equations() -> Dict[str, str]:
    """Return key equations for ResNet in LaTeX-style notation."""
    return {
        "residual_function":
            "y = F(x, \\{W_i\\}) + x",

        "gradient_flow":
            "\\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial y} "
            "\\left( \\frac{\\partial F}{\\partial x} + 1 \\right)",

        "identity_shortcut":
            "y = F(x) + x \\quad \\text{(when dimensions match)}",

        "projection_shortcut":
            "y = F(x) + W_s x \\quad \\text{(when dimensions differ)}",

        "basic_block":
            "F(x) = W_2 \\cdot \\text{ReLU}(\\text{BN}(W_1 \\cdot x))",

        "bottleneck_block":
            "F(x) = W_3 \\cdot \\text{ReLU}(\\text{BN}(W_2 \\cdot \\text{ReLU}(\\text{BN}(W_1 \\cdot x))))",

        "bottleneck_expansion":
            "C_{out} = 4 \\times C_{bottleneck}",

        "global_avg_pool":
            "z_k = \\frac{1}{H \\times W} \\sum_{i,j} x_{i,j,k}",
    }


def architecture_configs() -> Dict:
    """Return ResNet architecture configurations."""
    return {
        "ResNet-18": {
            "block": "BasicBlock",
            "layers": [2, 2, 2, 2],
            "params": "11.7M",
            "flops": "1.8G",
            "top1_error": "30.2%",
            "top5_error": "10.9%",
        },
        "ResNet-34": {
            "block": "BasicBlock",
            "layers": [3, 4, 6, 3],
            "params": "21.8M",
            "flops": "3.6G",
            "top1_error": "26.7%",
            "top5_error": "8.6%",
        },
        "ResNet-50": {
            "block": "BottleneckBlock",
            "layers": [3, 4, 6, 3],
            "params": "25.6M",
            "flops": "3.8G",
            "top1_error": "24.0%",
            "top5_error": "7.0%",
        },
        "ResNet-101": {
            "block": "BottleneckBlock",
            "layers": [3, 4, 23, 3],
            "params": "44.5M",
            "flops": "7.6G",
            "top1_error": "22.4%",
            "top5_error": "6.2%",
        },
        "ResNet-152": {
            "block": "BottleneckBlock",
            "layers": [3, 8, 36, 3],
            "params": "60.2M",
            "flops": "11.3G",
            "top1_error": "21.4%",
            "top5_error": "5.7%",
        },
        "stage_channels": {
            "basic": [64, 128, 256, 512],
            "bottleneck": [256, 512, 1024, 2048],  # 4x expansion
        }
    }


def get_historical_context() -> str:
    """Return historical context and significance of ResNet."""
    return """
    ResNet (2015) is arguably the most influential architecture in deep learning,
    fundamentally changing how we design neural networks.

    The Degradation Problem:
    - Before ResNet, deeper networks (>20 layers) performed WORSE than shallower ones
    - This wasn't due to overfitting (training error also increased)
    - Theoretically, a deeper network should be at least as good (can learn identity)
    - In practice, optimization failed to find solutions close to identity

    The Residual Solution:
    - Instead of learning y = H(x), learn y = F(x) + x
    - If identity is optimal, F(x) = 0 is easy to learn (push weights to zero)
    - Identity shortcut ensures gradient flows even through many layers
    - Enabled training of networks with 152 layers (and 1000+ in experiments)

    Competition Dominance (2015):
    - 1st place ILSVRC classification: 3.57% top-5 error (first superhuman result)
    - 1st place ILSVRC detection
    - 1st place ILSVRC localization
    - 1st place COCO detection
    - 1st place COCO segmentation

    Lasting Impact:
    - Skip connections became standard in nearly all deep architectures
    - Influenced Transformers (residual connections in attention layers)
    - Foundation for DenseNet, ResNeXt, EfficientNet, etc.
    - Pre-trained ResNets are still the most common backbones in computer vision
    """


def get_limitations() -> List[str]:
    """Return known limitations of ResNet."""
    return [
        "Feature reuse is not maximized (only adds, doesn't concatenate)",
        "Bottleneck blocks can create information bottleneck",
        "Standard ResNet uses BN which has issues with small batches",
        "Addition may not be optimal for all feature combinations",
        "Pre-activation ResNet (identity mappings paper) improves further",
    ]


def get_applications() -> List[str]:
    """Return applications of ResNet."""
    return [
        "Image classification (backbone for most modern systems)",
        "Object detection (Faster R-CNN, RetinaNet backbones)",
        "Semantic segmentation (DeepLab, PSPNet backbones)",
        "Instance segmentation (Mask R-CNN backbone)",
        "Pose estimation",
        "Video analysis",
        "Medical image analysis",
        "Transfer learning (most commonly used pre-trained models)",
        "Feature extraction for similarity/retrieval",
    ]
