"""
Deep Learning Era Methods (2006-2017)

This module contains research index entries for the deep learning revolution,
covering breakthrough architectures and techniques that enabled training
of very deep neural networks.

Key Methods:
    - AlexNet (2012): ImageNet breakthrough with GPU training
    - VGG (2014): Very deep networks with 3x3 convolutions
    - GoogLeNet/Inception (2014): Inception modules with multi-scale features
    - ResNet (2015): Residual connections enabling 100+ layer networks
    - DenseNet (2016): Dense connections between all layers
    - U-Net (2015): Encoder-decoder for biomedical segmentation
    - Batch Normalization (2015): Normalizing activations for faster training
    - Dropout (2014): Regularization via random neuron deactivation
    - Seq2Seq (2014): Encoder-decoder for sequence transduction

Submodules:
    - generative: Generative models (VAE, GAN, DCGAN, WGAN, StyleGAN)
    - object_detection: Object detection architectures
"""

from .alexnet import (
    ALEXNET,
    get_method_info as get_alexnet_info,
    pseudocode as alexnet_pseudocode,
    key_equations as alexnet_equations,
)
from .vgg import (
    VGG,
    get_method_info as get_vgg_info,
    pseudocode as vgg_pseudocode,
    key_equations as vgg_equations,
)
from .inception import (
    INCEPTION,
    get_method_info as get_inception_info,
    pseudocode as inception_pseudocode,
    key_equations as inception_equations,
)
from .resnet import (
    RESNET,
    get_method_info as get_resnet_info,
    pseudocode as resnet_pseudocode,
    key_equations as resnet_equations,
)
from .densenet import (
    DENSENET,
    get_method_info as get_densenet_info,
    pseudocode as densenet_pseudocode,
    key_equations as densenet_equations,
)
from .unet import (
    UNET,
    get_method_info as get_unet_info,
    pseudocode as unet_pseudocode,
    key_equations as unet_equations,
)
from .batch_norm import (
    BATCH_NORMALIZATION,
    get_method_info as get_batch_norm_info,
    pseudocode as batch_norm_pseudocode,
    key_equations as batch_norm_equations,
)
from .dropout import (
    DROPOUT,
    get_method_info as get_dropout_info,
    pseudocode as dropout_pseudocode,
    key_equations as dropout_equations,
)
from .seq2seq import (
    SEQ2SEQ,
    get_method_info as get_seq2seq_info,
    pseudocode as seq2seq_pseudocode,
    key_equations as seq2seq_equations,
)

# Import submodules if they exist
try:
    from . import generative
except ImportError:
    generative = None

__all__ = [
    # AlexNet
    "ALEXNET",
    "get_alexnet_info",
    "alexnet_pseudocode",
    "alexnet_equations",
    # VGG
    "VGG",
    "get_vgg_info",
    "vgg_pseudocode",
    "vgg_equations",
    # Inception/GoogLeNet
    "INCEPTION",
    "get_inception_info",
    "inception_pseudocode",
    "inception_equations",
    # ResNet
    "RESNET",
    "get_resnet_info",
    "resnet_pseudocode",
    "resnet_equations",
    # DenseNet
    "DENSENET",
    "get_densenet_info",
    "densenet_pseudocode",
    "densenet_equations",
    # U-Net
    "UNET",
    "get_unet_info",
    "unet_pseudocode",
    "unet_equations",
    # Batch Normalization
    "BATCH_NORMALIZATION",
    "get_batch_norm_info",
    "batch_norm_pseudocode",
    "batch_norm_equations",
    # Dropout
    "DROPOUT",
    "get_dropout_info",
    "dropout_pseudocode",
    "dropout_equations",
    # Seq2Seq
    "SEQ2SEQ",
    "get_seq2seq_info",
    "seq2seq_pseudocode",
    "seq2seq_equations",
    # Submodules
    "generative",
]
