"""
Classical Era ML Methods (1980-2006)

This module contains research index entries for classical machine learning
methods spanning from the connectionist revival through the kernel methods era.
"""

from .mlp import (
    get_method_info as get_mlp_info,
    pseudocode as mlp_pseudocode,
    key_equations as mlp_equations,
)
from .backprop_rumelhart import (
    get_method_info as get_backprop_info,
    pseudocode as backprop_pseudocode,
    key_equations as backprop_equations,
)
from .cnn_lecun import (
    get_method_info as get_cnn_info,
    pseudocode as cnn_pseudocode,
    key_equations as cnn_equations,
)
from .rbf_networks import (
    get_method_info as get_rbf_info,
    pseudocode as rbf_pseudocode,
    key_equations as rbf_equations,
)
from .svm import (
    get_method_info as get_svm_info,
    pseudocode as svm_pseudocode,
    key_equations as svm_equations,
)
from .autoencoders import (
    get_method_info as get_autoencoder_info,
    pseudocode as autoencoder_pseudocode,
    key_equations as autoencoder_equations,
)

__all__ = [
    # MLP
    "get_mlp_info",
    "mlp_pseudocode",
    "mlp_equations",
    # Backpropagation
    "get_backprop_info",
    "backprop_pseudocode",
    "backprop_equations",
    # CNN/LeNet
    "get_cnn_info",
    "cnn_pseudocode",
    "cnn_equations",
    # RBF Networks
    "get_rbf_info",
    "rbf_pseudocode",
    "rbf_equations",
    # SVM
    "get_svm_info",
    "svm_pseudocode",
    "svm_equations",
    # Autoencoders
    "get_autoencoder_info",
    "autoencoder_pseudocode",
    "autoencoder_equations",
]
