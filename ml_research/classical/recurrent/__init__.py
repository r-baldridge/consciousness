"""
Classical Recurrent Methods Module

This module contains research indices for classical recurrent neural network
architectures that revolutionized sequence modeling.

Key Methods:
    - Simple RNN / Elman Networks (1990): First practical recurrent networks
    - LSTM (1997): Long Short-Term Memory with gating mechanisms
    - GRU (2014): Gated Recurrent Unit with simplified gating
"""

from .simple_rnn import (
    get_method_info as get_simple_rnn_info,
    pseudocode as simple_rnn_pseudocode,
    key_equations as simple_rnn_equations,
    SIMPLE_RNN,
)
from .lstm import (
    get_method_info as get_lstm_info,
    pseudocode as lstm_pseudocode,
    key_equations as lstm_equations,
    LSTM,
)
from .gru import (
    get_method_info as get_gru_info,
    pseudocode as gru_pseudocode,
    key_equations as gru_equations,
    GRU,
)

__all__ = [
    # Simple RNN / Elman Networks
    "SIMPLE_RNN",
    "get_simple_rnn_info",
    "simple_rnn_pseudocode",
    "simple_rnn_equations",
    # LSTM
    "LSTM",
    "get_lstm_info",
    "lstm_pseudocode",
    "lstm_equations",
    # GRU
    "GRU",
    "get_gru_info",
    "gru_pseudocode",
    "gru_equations",
]
