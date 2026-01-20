"""
Convolutional Neural Networks / LeNet - Yann LeCun (1989)

Research index entry for LeNet and the convolutional neural network
architecture, which introduced convolutional layers, pooling, and
weight sharing for efficient visual pattern recognition.

Paper: "Backpropagation Applied to Handwritten Zip Code Recognition"
Neural Computation, 1(4), 541-551

Key contributions:
- Convolutional layers with local receptive fields
- Weight sharing for translation invariance
- Pooling (subsampling) for spatial hierarchy
- Foundation for modern computer vision
"""

from typing import Dict, List

from ..core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for LeNet/CNN."""
    return MLMethod(
        method_id="cnn_lecun_1989",
        name="Convolutional Neural Network (LeNet)",
        year=1989,
        era=MethodEra.CLASSICAL,
        category=MethodCategory.ARCHITECTURE,
        lineages=[MethodLineage.CNN_LINE],
        authors=[
            "Yann LeCun",
            "Bernhard Boser",
            "John S. Denker",
            "Donnie Henderson",
            "Richard E. Howard",
            "Wayne Hubbard",
            "Lawrence D. Jackel",
        ],
        paper_title="Backpropagation Applied to Handwritten Zip Code Recognition",
        paper_url="https://ieeexplore.ieee.org/document/6795724",
        key_innovation="""
        CNNs introduced three key architectural principles:

        1. LOCAL RECEPTIVE FIELDS: Each neuron connects only to a small
           local region of the input, exploiting spatial locality in images.

        2. WEIGHT SHARING: The same filter (kernel) is applied across the
           entire input, dramatically reducing parameters and enforcing
           translation equivariance.

        3. POOLING (SUBSAMPLING): Reduces spatial dimensions progressively,
           building a hierarchy from local features to global patterns and
           providing some translation invariance.

        These innovations made it practical to train networks on images
        without requiring full connectivity (which would be prohibitively
        expensive for high-resolution inputs).
        """,
        mathematical_formulation="""
        CONVOLUTION OPERATION:
        For input X and filter K of size k x k:
            (X * K)_ij = sum_{m,n} X_{i+m, j+n} * K_{m,n}

        Or with multiple channels:
            Y_c = sum_{c'} X_{c'} * K_{c,c'} + b_c

        POOLING (MAX):
            Y_ij = max_{(m,n) in pool_region} X_{i*s+m, j*s+n}

        POOLING (AVERAGE):
            Y_ij = (1/|pool|) * sum_{(m,n) in pool_region} X_{i*s+m, j*s+n}

        LENET-5 ARCHITECTURE (1998):
            Input: 32x32 grayscale image
            C1: Convolution (6 filters, 5x5) -> 28x28x6
            S2: Subsampling (2x2 average pool) -> 14x14x6
            C3: Convolution (16 filters, 5x5) -> 10x10x16
            S4: Subsampling (2x2 average pool) -> 5x5x16
            C5: Convolution (120 filters, 5x5) -> 1x1x120
            F6: Fully connected -> 84
            Output: Fully connected -> 10
        """,
        predecessors=["neocognitron_fukushima_1980", "mlp_1986"],
        successors=[
            "alexnet_2012",
            "vgg_2014",
            "resnet_2015",
            "inception_2014",
        ],
        tags=[
            "convolutional",
            "weight_sharing",
            "pooling",
            "computer_vision",
            "image_recognition",
            "translation_equivariance",
        ],
        notes="""
        LeCun's work built on Fukushima's Neocognitron (1980), which
        introduced the idea of hierarchical feature extraction but
        lacked an effective learning algorithm. CNNs combined this
        architecture with backpropagation.

        LeNet was successfully deployed at AT&T/NCR for reading
        handwritten checks, processing millions of checks daily.
        This was one of the first large-scale deployments of neural
        networks in production.

        The field remained relatively dormant until AlexNet (2012)
        demonstrated the power of CNNs on large-scale image recognition
        using GPUs and larger datasets.
        """,
    )


def pseudocode() -> str:
    """Return pseudocode for CNN forward pass and convolution operations."""
    return """
    CONVOLUTIONAL NEURAL NETWORK ALGORITHM
    ======================================

    CONVOLUTION LAYER:
        Input: X of shape (H, W, C_in)
        Filter: K of shape (k, k, C_in, C_out)
        Stride: s
        Padding: p

        # Output dimensions
        H_out = (H + 2*p - k) / s + 1
        W_out = (W + 2*p - k) / s + 1

        # Apply padding
        X_padded = pad(X, p)

        # Convolve
        for c_out in 0 to C_out-1:
            for i in 0 to H_out-1:
                for j in 0 to W_out-1:
                    # Extract receptive field
                    receptive_field = X_padded[i*s : i*s+k, j*s : j*s+k, :]

                    # Dot product with filter
                    Y[i, j, c_out] = sum(receptive_field * K[:, :, :, c_out]) + b[c_out]

        return activation(Y)

    MAX POOLING LAYER:
        Input: X of shape (H, W, C)
        Pool size: p
        Stride: s (typically s = p for non-overlapping)

        H_out = (H - p) / s + 1
        W_out = (W - p) / s + 1

        for c in 0 to C-1:
            for i in 0 to H_out-1:
                for j in 0 to W_out-1:
                    pool_region = X[i*s : i*s+p, j*s : j*s+p, c]
                    Y[i, j, c] = max(pool_region)

        return Y

    BACKPROP THROUGH CONVOLUTION:
        # Given dL/dY (upstream gradient)

        # Gradient w.r.t. input (for backprop to previous layer)
        dL/dX = convolve_full(dL/dY, K_rotated_180)

        # Gradient w.r.t. filter (for weight update)
        dL/dK = convolve(X, dL/dY)

    LENET-5 FORWARD PASS:
        def forward(x):
            # x: 32x32x1

            # Convolutional layers
            x = relu(conv2d(x, C1_weights))    # -> 28x28x6
            x = avg_pool2d(x, 2)                # -> 14x14x6

            x = relu(conv2d(x, C3_weights))    # -> 10x10x16
            x = avg_pool2d(x, 2)                # -> 5x5x16

            x = relu(conv2d(x, C5_weights))    # -> 1x1x120

            # Flatten and fully connected
            x = flatten(x)                      # -> 120
            x = relu(fc(x, F6_weights))        # -> 84
            x = fc(x, output_weights)          # -> 10

            return softmax(x)
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for CNNs."""
    return {
        # Convolution
        "2d_convolution": "(X * K)_ij = sum_{m,n} X_{i+m, j+n} * K_{m,n}",
        "multi_channel_conv": "Y_c = sum_{c'} X_{c'} * K_{c,c'} + b_c",
        # Output dimensions
        "output_height": "H_out = floor((H + 2p - k) / s) + 1",
        "output_width": "W_out = floor((W + 2p - k) / s) + 1",
        # Pooling
        "max_pooling": "Y_ij = max_{(m,n) in R_ij} X_{m,n}",
        "avg_pooling": "Y_ij = (1/|R|) sum_{(m,n) in R_ij} X_{m,n}",
        # Parameter count
        "conv_params": "params = k * k * C_in * C_out + C_out (bias)",
        "fc_params": "params = N_in * N_out + N_out (bias)",
        # Receptive field
        "receptive_field": "RF_l = RF_{l-1} + (k_l - 1) * prod_{i<l} s_i",
        # Translation equivariance
        "equivariance": "T_a[f(X)] = f(T_a[X]) where T_a is translation by a",
        # Backprop
        "conv_grad_input": "dL/dX = full_conv(dL/dY, rot180(K))",
        "conv_grad_weight": "dL/dK = conv(X, dL/dY)",
    }


def get_lenet5_architecture() -> Dict[str, any]:
    """Return detailed LeNet-5 architecture specification."""
    return {
        "input": {
            "shape": (32, 32, 1),
            "description": "Grayscale image, normalized to [-0.1, 1.175]",
        },
        "C1": {
            "type": "Convolution",
            "filters": 6,
            "kernel_size": 5,
            "stride": 1,
            "padding": 0,
            "activation": "tanh",
            "output_shape": (28, 28, 6),
            "parameters": 6 * (5 * 5 * 1 + 1),  # 156
        },
        "S2": {
            "type": "Subsampling (Average Pool)",
            "pool_size": 2,
            "stride": 2,
            "trainable_coefficient": True,
            "output_shape": (14, 14, 6),
            "parameters": 6 * 2,  # 12 (coefficient + bias per feature map)
        },
        "C3": {
            "type": "Convolution",
            "filters": 16,
            "kernel_size": 5,
            "stride": 1,
            "padding": 0,
            "activation": "tanh",
            "output_shape": (10, 10, 16),
            "note": "Sparse connectivity pattern (not all inputs to all outputs)",
            "parameters": 1516,  # Complex sparse pattern
        },
        "S4": {
            "type": "Subsampling (Average Pool)",
            "pool_size": 2,
            "stride": 2,
            "trainable_coefficient": True,
            "output_shape": (5, 5, 16),
            "parameters": 16 * 2,  # 32
        },
        "C5": {
            "type": "Convolution",
            "filters": 120,
            "kernel_size": 5,
            "stride": 1,
            "padding": 0,
            "activation": "tanh",
            "output_shape": (1, 1, 120),
            "parameters": 120 * (5 * 5 * 16 + 1),  # 48120
        },
        "F6": {
            "type": "Fully Connected",
            "units": 84,
            "activation": "tanh",
            "output_shape": (84,),
            "parameters": 84 * (120 + 1),  # 10164
        },
        "Output": {
            "type": "RBF Units (Euclidean)",
            "units": 10,
            "description": "Euclidean distance to prototype patterns",
            "output_shape": (10,),
            "parameters": 10 * 84,  # 840 (fixed prototypes)
        },
        "total_parameters": "~60,000",
    }


def get_cnn_design_principles() -> List[Dict[str, str]]:
    """Return CNN design principles that emerged from LeNet."""
    return [
        {
            "principle": "Local Connectivity",
            "description": """
                Neurons in convolutional layers connect only to a local region
                (receptive field) of the input, exploiting the spatial locality
                of features in images.
            """,
            "benefit": "Dramatically reduces parameters vs fully connected.",
        },
        {
            "principle": "Weight Sharing",
            "description": """
                The same filter is applied across all spatial positions,
                so the same features are detected everywhere in the image.
            """,
            "benefit": "Further parameter reduction and translation equivariance.",
        },
        {
            "principle": "Hierarchical Features",
            "description": """
                Early layers detect simple features (edges, colors).
                Deeper layers combine these into increasingly complex patterns
                (textures, parts, objects).
            """,
            "benefit": "Compositional feature learning.",
        },
        {
            "principle": "Spatial Pooling",
            "description": """
                Pooling reduces spatial dimensions while retaining the most
                important features, providing some translation invariance.
            """,
            "benefit": "Reduced computation and mild invariance.",
        },
        {
            "principle": "Increasing Depth, Decreasing Spatial Size",
            "description": """
                As spatial dimensions decrease through pooling, the number
                of feature channels typically increases.
            """,
            "benefit": "Captures increasingly abstract representations.",
        },
    ]


def get_evolution_to_modern_cnns() -> List[Dict[str, str]]:
    """Return evolution from LeNet to modern CNN architectures."""
    return [
        {
            "year": 1989,
            "architecture": "LeNet-1",
            "innovation": "First successful CNN with backprop",
            "depth": 5,
        },
        {
            "year": 1998,
            "architecture": "LeNet-5",
            "innovation": "Refined architecture for MNIST",
            "depth": 7,
        },
        {
            "year": 2012,
            "architecture": "AlexNet",
            "innovation": "GPU training, ReLU, Dropout, deeper",
            "depth": 8,
        },
        {
            "year": 2014,
            "architecture": "VGGNet",
            "innovation": "Very deep (16-19), small 3x3 filters",
            "depth": 19,
        },
        {
            "year": 2014,
            "architecture": "GoogLeNet/Inception",
            "innovation": "Inception modules, multi-scale processing",
            "depth": 22,
        },
        {
            "year": 2015,
            "architecture": "ResNet",
            "innovation": "Skip connections, very deep (152+)",
            "depth": 152,
        },
        {
            "year": 2017,
            "architecture": "DenseNet",
            "innovation": "Dense connections between all layers",
            "depth": 201,
        },
        {
            "year": 2019,
            "architecture": "EfficientNet",
            "innovation": "Neural architecture search, compound scaling",
            "depth": "variable",
        },
    ]
