"""
AlexNet (2012)

ImageNet Classification with Deep Convolutional Neural Networks
Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton

The watershed moment for deep learning - AlexNet won the 2012 ImageNet Large
Scale Visual Recognition Challenge (ILSVRC) by a massive margin, reducing
the top-5 error rate from 26.2% to 15.3%. This demonstrated that deep
convolutional networks, trained on GPUs with modern techniques, could
dramatically outperform traditional computer vision methods.

Key Innovations:
    - ReLU activation: f(x) = max(0, x) - faster training than sigmoid/tanh
    - GPU training: Split network across 2 GTX 580 GPUs (3GB each)
    - Local Response Normalization (LRN): Cross-channel normalization
    - Overlapping pooling: 3x3 pooling with stride 2
    - Dropout: p=0.5 in fully connected layers
    - Data augmentation: Random crops, horizontal flips, PCA color augmentation

Architecture:
    - 5 convolutional layers + 3 fully connected layers
    - ~60 million parameters
    - Input: 224x224x3 RGB images
    - Conv1: 96 kernels, 11x11, stride 4
    - Conv2: 256 kernels, 5x5
    - Conv3-5: 384, 384, 256 kernels, 3x3
    - FC6-7: 4096 neurons each
    - FC8: 1000 outputs (ImageNet classes)
"""


from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from typing import Dict, List


# Module-level constant for the method
ALEXNET = MLMethod(
    method_id="alexnet_2012",
    name="AlexNet",
    year=2012,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Alex Krizhevsky", "Ilya Sutskever", "Geoffrey E. Hinton"],
    paper_title="ImageNet Classification with Deep Convolutional Neural Networks",
    paper_url="https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html",
    key_innovation="Combined ReLU activations, GPU training, dropout, and data augmentation "
                   "to achieve breakthrough results on ImageNet, initiating the deep learning era",
    mathematical_formulation="""
    ReLU Activation:
        f(x) = max(0, x)

    Local Response Normalization:
        b^i_{x,y} = a^i_{x,y} / (k + alpha * sum_{j=max(0,i-n/2)}^{min(N-1,i+n/2)} (a^j_{x,y})^2)^beta

        where a^i_{x,y} is the activity of neuron at position (x,y) in channel i,
        k=2, n=5, alpha=10^-4, beta=0.75

    Overlapping Max Pooling:
        Pool size: 3x3, stride: 2
        Output size: (input - 3) / 2 + 1

    Dropout (during training):
        y = r * x,  where r ~ Bernoulli(p=0.5)

    Softmax Output:
        P(class_i | x) = exp(z_i) / sum_j exp(z_j)
    """,
    predecessors=["lenet_1998", "neocognitron_1980"],
    successors=["vgg_2014", "googlenet_2014", "resnet_2015"],
    tags=["imagenet", "gpu", "relu", "dropout", "data_augmentation", "breakthrough"],
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for AlexNet."""
    return ALEXNET


def pseudocode() -> str:
    """Return pseudocode describing AlexNet's architecture and training."""
    return """
    ALEXNET ARCHITECTURE:

    function AlexNet(input_image):
        # Input: 224 x 224 x 3 RGB image

        # CONV BLOCK 1
        x = Conv2D(input_image, filters=96, kernel=11x11, stride=4)  # -> 55x55x96
        x = ReLU(x)
        x = LocalResponseNorm(x, n=5, k=2, alpha=1e-4, beta=0.75)
        x = MaxPool(x, size=3x3, stride=2)  # -> 27x27x96

        # CONV BLOCK 2
        x = Conv2D(x, filters=256, kernel=5x5, padding='same')  # -> 27x27x256
        x = ReLU(x)
        x = LocalResponseNorm(x)
        x = MaxPool(x, size=3x3, stride=2)  # -> 13x13x256

        # CONV BLOCKS 3-5 (no pooling between)
        x = Conv2D(x, filters=384, kernel=3x3, padding='same')  # -> 13x13x384
        x = ReLU(x)

        x = Conv2D(x, filters=384, kernel=3x3, padding='same')  # -> 13x13x384
        x = ReLU(x)

        x = Conv2D(x, filters=256, kernel=3x3, padding='same')  # -> 13x13x256
        x = ReLU(x)
        x = MaxPool(x, size=3x3, stride=2)  # -> 6x6x256

        # FULLY CONNECTED LAYERS
        x = Flatten(x)  # -> 9216
        x = Dense(x, units=4096)
        x = ReLU(x)
        x = Dropout(x, p=0.5)

        x = Dense(x, units=4096)
        x = ReLU(x)
        x = Dropout(x, p=0.5)

        x = Dense(x, units=1000)  # 1000 ImageNet classes
        output = Softmax(x)

        return output

    TRAINING PROCEDURE:

    function train_alexnet(dataset, epochs=90):
        model = AlexNet()
        optimizer = SGD(lr=0.01, momentum=0.9, weight_decay=0.0005)

        for epoch in range(epochs):
            for batch in dataset:
                # Data augmentation
                images = random_crop(batch.images, size=224)
                images = random_horizontal_flip(images)
                images = pca_color_augmentation(images)

                # Forward pass
                predictions = model(images)
                loss = cross_entropy(predictions, batch.labels)

                # Backward pass
                gradients = backprop(loss)
                optimizer.step(gradients)

            # Learning rate schedule: reduce by 10x when validation error plateaus
            if validation_error_plateaued():
                optimizer.lr *= 0.1

        return model
    """


def key_equations() -> Dict[str, str]:
    """Return key equations for AlexNet in LaTeX-style notation."""
    return {
        "relu": "f(x) = max(0, x)",

        "local_response_norm":
            "b^i_{x,y} = a^i_{x,y} / (k + \\alpha \\sum_{j=max(0,i-n/2)}^{min(N-1,i+n/2)} (a^j_{x,y})^2)^\\beta",

        "conv_output_size":
            "W_{out} = \\lfloor (W_{in} - K + 2P) / S \\rfloor + 1",

        "pool_output_size":
            "W_{out} = \\lfloor (W_{in} - K) / S \\rfloor + 1",

        "dropout":
            "\\tilde{y} = r \\odot y, \\quad r_i \\sim \\text{Bernoulli}(p)",

        "cross_entropy_loss":
            "L = -\\sum_{i=1}^{C} y_i \\log(\\hat{y}_i)",

        "softmax":
            "P(class_i | x) = \\frac{e^{z_i}}{\\sum_{j=1}^{C} e^{z_j}}",

        "sgd_momentum":
            "v_{t+1} = \\mu v_t - \\eta \\nabla L(\\theta_t), \\quad \\theta_{t+1} = \\theta_t + v_{t+1}",

        "weight_decay":
            "L_{total} = L_{data} + \\lambda ||\\theta||_2^2",
    }


def architecture_details() -> Dict[str, List]:
    """Return detailed architecture specifications."""
    return {
        "layers": [
            {"name": "conv1", "type": "Conv2D", "filters": 96, "kernel": (11, 11), "stride": 4, "output": (55, 55, 96)},
            {"name": "pool1", "type": "MaxPool", "size": (3, 3), "stride": 2, "output": (27, 27, 96)},
            {"name": "conv2", "type": "Conv2D", "filters": 256, "kernel": (5, 5), "padding": "same", "output": (27, 27, 256)},
            {"name": "pool2", "type": "MaxPool", "size": (3, 3), "stride": 2, "output": (13, 13, 256)},
            {"name": "conv3", "type": "Conv2D", "filters": 384, "kernel": (3, 3), "padding": "same", "output": (13, 13, 384)},
            {"name": "conv4", "type": "Conv2D", "filters": 384, "kernel": (3, 3), "padding": "same", "output": (13, 13, 384)},
            {"name": "conv5", "type": "Conv2D", "filters": 256, "kernel": (3, 3), "padding": "same", "output": (13, 13, 256)},
            {"name": "pool5", "type": "MaxPool", "size": (3, 3), "stride": 2, "output": (6, 6, 256)},
            {"name": "fc6", "type": "Dense", "units": 4096},
            {"name": "fc7", "type": "Dense", "units": 4096},
            {"name": "fc8", "type": "Dense", "units": 1000},
        ],
        "parameters": {
            "total": "~60 million",
            "conv_layers": "~2.3 million",
            "fc_layers": "~58 million (mostly in fc6-fc7 connection)",
        },
        "training": {
            "optimizer": "SGD with momentum (0.9)",
            "initial_lr": 0.01,
            "weight_decay": 0.0005,
            "batch_size": 128,
            "epochs": 90,
            "gpus": "2x NVIDIA GTX 580 (3GB each)",
            "training_time": "5-6 days",
        }
    }


def get_historical_context() -> str:
    """Return historical context and significance of AlexNet."""
    return """
    AlexNet (2012) marked the beginning of the deep learning revolution in computer vision.

    Before AlexNet:
    - Computer vision relied heavily on hand-crafted features (SIFT, HOG, etc.)
    - The best ImageNet methods achieved ~26% top-5 error using ensemble methods
    - Deep learning was considered a niche research area
    - GPUs were rarely used for neural network training

    The 2012 ImageNet Competition:
    - AlexNet achieved 15.3% top-5 error, crushing the second place at 26.2%
    - This ~10% absolute improvement was unprecedented
    - Demonstrated that depth + data + compute = better representations

    Impact:
    - Sparked massive investment in deep learning research
    - Led to GPU manufacturers (NVIDIA) focusing on ML workloads
    - Catalyzed development of deep learning frameworks
    - Every subsequent ImageNet winner used deep CNNs
    - Many researchers pivoted their careers to deep learning

    The "ImageNet moment" is often cited as the beginning of the current AI boom.
    """


def get_limitations() -> List[str]:
    """Return known limitations of AlexNet."""
    return [
        "Large number of parameters in FC layers (memory inefficient)",
        "Local Response Normalization later shown to provide minimal benefit",
        "Large first-layer filters (11x11) capture coarse features",
        "Manual GPU parallelization (model split across 2 GPUs)",
        "Requires significant data augmentation to prevent overfitting",
        "Relatively shallow compared to modern architectures",
    ]


def get_applications() -> List[str]:
    """Return applications and uses of AlexNet."""
    return [
        "Image classification (original application)",
        "Transfer learning (pre-trained features)",
        "Object detection (as backbone network)",
        "Feature extraction for visual similarity",
        "Educational benchmark for deep learning",
    ]
