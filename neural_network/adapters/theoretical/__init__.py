"""
Theoretical Adapters - Forms 13-17
Implementing consciousness theory computations (IIT, GWT, HOT, etc.).
"""

from .iit_adapter import IITAdapter
from .global_workspace_adapter import GlobalWorkspaceAdapter
from .hot_adapter import HOTAdapter
from .predictive_adapter import PredictiveAdapter
from .recurrent_adapter import RecurrentAdapter

__all__ = [
    'IITAdapter',
    'GlobalWorkspaceAdapter',
    'HOTAdapter',
    'PredictiveAdapter',
    'RecurrentAdapter',
]
