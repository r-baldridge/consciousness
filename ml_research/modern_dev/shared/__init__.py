"""
Shared infrastructure for modern_dev model architectures.

This module provides unified data collection, processing, and loading
for training code repair models (TRM, CTM, and future architectures).
"""

from .data.schema import CanonicalCodeSample, QualityTier
from .data.loaders.trm import TRMDataLoader, TRMDataset, TRMSample
from .data.loaders.ctm import CTMDataLoader, CTMDataset, CTMSample
from .taxonomy.bug_types import BugType, BugCategory, BugInfo, BUG_TAXONOMY

__all__ = [
    # Schema
    'CanonicalCodeSample',
    'QualityTier',
    # Loaders
    'TRMDataLoader',
    'TRMDataset',
    'TRMSample',
    'CTMDataLoader',
    'CTMDataset',
    'CTMSample',
    # Taxonomy
    'BugType',
    'BugCategory',
    'BugInfo',
    'BUG_TAXONOMY',
]

__version__ = '0.1.0'
