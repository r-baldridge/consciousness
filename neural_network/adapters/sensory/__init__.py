"""
Sensory Adapters - Forms 01-06
Processing visual, auditory, tactile, olfactory, gustatory, and proprioceptive inputs.
"""

from .visual_adapter import VisualAdapter
from .auditory_adapter import AuditoryAdapter
from .tactile_adapter import TactileAdapter
from .olfactory_adapter import OlfactoryAdapter
from .gustatory_adapter import GustatoryAdapter
from .proprioceptive_adapter import ProprioceptiveAdapter

__all__ = [
    'VisualAdapter',
    'AuditoryAdapter',
    'TactileAdapter',
    'OlfactoryAdapter',
    'GustatoryAdapter',
    'ProprioceptiveAdapter',
]
